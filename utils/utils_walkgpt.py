import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torch.distributed as dist

def infonce_loss(
    pred_embeddings: torch.Tensor,      # [M, 256] segmentation text embeddings
    sam_tokens_256: torch.Tensor,       # [rows, N, 256] row-aligned SAM grid tokens
    seg_row_ids: torch.Tensor,          # [M] row index for each segmentation embedding
    tiny_xattn,                         # Tiny cross-attention module (Q=text, K/V=SAM row tokens)
    *,
    temperature: float = 0.07,
    top_k: int | None = None,           # Optional top-k positive refinement
    exclude_same_row: bool = True,      # Exclude negatives from the same image row
    normalize: bool = True,             # L2-normalize features before similarity
    return_aux: bool = False            # Return auxiliary tensors for analysis
):
    """
    Returns:
        loss: scalar tensor
        (optional) dict with v_pos, attn_w, logits, labels
    """
    device = pred_embeddings.device
    M = pred_embeddings.size(0)
    rows, N, D = sam_tokens_256.shape
    assert D == pred_embeddings.size(-1), "Vision/text feature dims must match for InfoNCE."

    # 1) Gather each segmentation row and compute positive features via tiny cross-attention.
    KV = sam_tokens_256[seg_row_ids]                     # [M, N, 256]
    v_pos, attn_w = tiny_xattn(pred_embeddings, KV)      # v_pos: [M,256], attn_w: [M,N]

    # Optional top-k refinement for positive features.
    if top_k is not None and top_k > 0 and top_k < N:
        vals, idx = torch.topk(attn_w, k=top_k, dim=1)   # [M,K]
        alpha = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
        V_top = torch.gather(KV, 1, idx.unsqueeze(-1).expand(-1, -1, D))  # [M,K,256]
        v_pos = torch.einsum('mk,mkd->md', alpha, V_top)                   # [M,256]

    # 2) Build logits using positive features and in-batch negatives.
    Z = pred_embeddings
    Vpos = v_pos

    if normalize:
        Z    = F.normalize(Z,    dim=-1)
        Vpos = F.normalize(Vpos, dim=-1)

    pos = (Z * Vpos).sum(-1, keepdim=True)               # [M,1]

    V_all = sam_tokens_256.reshape(-1, D)                # [rows*N,256]
    if normalize:
        V_all = F.normalize(V_all, dim=-1)

    all_sim = Z @ V_all.T                                # [M, rows*N]

    # Optionally exclude same-row tokens from the negative set.
    if exclude_same_row:
        # `row_mask` is False for the current row and True otherwise.
        row_mask = torch.ones((M, rows), dtype=torch.bool, device=device)
        row_mask[torch.arange(M, device=device), seg_row_ids] = False      # mask out own row
        token_mask = row_mask.unsqueeze(-1).expand(M, rows, N).reshape(M, rows * N)
        all_sim = all_sim.masked_fill(~token_mask, float('-inf'))

    # Concatenate positive similarity at index 0, followed by all negatives.
    logits = torch.cat([pos, all_sim], dim=1) / temperature
    labels = torch.zeros(M, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels)

    if return_aux:
        return loss, {"v_pos": v_pos, "attn_w": attn_w, "logits": logits, "labels": labels}
    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # Numerical scaling for stable mask losses.
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss



def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def overlap_loss(inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    batch_seg_token_count: int):
    if num_masks == 0:
        return inputs.sum() * 0
    batch_seg_token_count = batch_seg_token_count.cumsum(-1)  
    batch_seg_token_count = torch.cat(
            [torch.zeros(1).long().cuda(), batch_seg_token_count], dim=0
        )
    loss = 0

    for i in range(len(batch_seg_token_count) -1):
        start_i = batch_seg_token_count[i]
        end_i = batch_seg_token_count[i+1]
        assert end_i <= len(targets), (targets.shape, batch_seg_token_count)
        question_inputs = inputs[start_i:end_i]
        question_targets = targets[start_i:end_i]
        if len(question_targets) == 0:
            continue
        n, h, w = question_inputs.shape
        all_targets = torch.zeros_like(question_targets[0]).bool()
        for target in question_targets:
            all_targets = (all_targets | target.bool())
        bg_area = all_targets < 0
        bg_area = bg_area[None].repeat(n, 1, 1)

        overlap_area = (question_inputs > 0).sum(dim=0)
        overlap_area = overlap_area >= 2

        overlap_area = overlap_area[None].repeat(n, 1, 1)
        weight = torch.ones_like(question_inputs)
        weight[~overlap_area] = 0

        q_loss = F.binary_cross_entropy_with_logits(question_inputs, question_targets, weight=weight, reduction="none")
        q_loss = q_loss.flatten(1, 2).mean(1).sum() 
        loss = loss + q_loss
    loss = loss / (num_masks + 1e-8)
    return loss


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.proj_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, kv):
        q = self.q_norm(queries)
        kv = self.kv_norm(kv)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = queries + self.proj_drop(out)
        out = out + self.ffn(out)
        return out


def _infer_hw_from_len(L):
    H = int(math.sqrt(L))
    if H * H != L:
        raise ValueError(f"Token length {L} is not a perfect square.")
    return H, H


def _pool_grid_tokens(tokens, H, W, scale):
    B, L, C = tokens.shape
    x = tokens.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
    x = F.avg_pool2d(x, kernel_size=scale, stride=scale)
    Hp, Wp = x.shape[-2], x.shape[-1]
    x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
    return x, Hp, Wp


class SegAwareGate(nn.Module):
    def __init__(self, d_in, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, kv_tokens):
        logits = self.net(kv_tokens).squeeze(-1)   # [B, K]
        gates = torch.sigmoid(logits).unsqueeze(-1)
        return kv_tokens * gates


class MultiScaleQFormerProjector(nn.Module):
    def __init__(self, sam_dim, llama_dim, grid_size=None, num_heads=8,
                 pad_to_square: bool = True, target_square_side: int | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.d_proj = 1024          # Fixed hidden dimension.
        self.num_layers = 2         # Fixed number of cross-attention layers.
        self.pad_to_square = pad_to_square
        self.target_square_side = target_square_side  # Example: 6 enforces a 6x6 token grid.
        self.pad_token = nn.Parameter(torch.zeros(1, 1, self.d_proj))
        nn.init.trunc_normal_(self.pad_token, std=0.02)
        # Input projection.
        self.sam_to_proj = nn.Linear(sam_dim, self.d_proj)

        # Query groups for different receptive-field scales.
        self.q_x1 = nn.Parameter(torch.randn(1, 12, self.d_proj))
        self.q_x2 = nn.Parameter(torch.randn(1, 8, self.d_proj))
        self.q_x4 = nn.Parameter(torch.randn(1, 8, self.d_proj))
        self.q_global = nn.Parameter(torch.randn(1, 4, self.d_proj))

        # Cross-attention stacks per scale.
        self.cross_x1 = nn.ModuleList([CrossAttnBlock(self.d_proj, num_heads) for _ in range(self.num_layers)])
        self.cross_x2 = nn.ModuleList([CrossAttnBlock(self.d_proj, num_heads) for _ in range(self.num_layers)])
        self.cross_x4 = nn.ModuleList([CrossAttnBlock(self.d_proj, num_heads) for _ in range(self.num_layers)])
        self.cross_glb = nn.ModuleList([CrossAttnBlock(self.d_proj, num_heads) for _ in range(self.num_layers)])

        # Segmentation-aware gating.
        self.gate = SegAwareGate(self.d_proj)

        # Final projection to LLaMA hidden size.
        self.to_llama = nn.Linear(self.d_proj, llama_dim)

        # Query parameter initialization.
        for p in [self.q_x1, self.q_x2, self.q_x4, self.q_global]:
            nn.init.trunc_normal_(p, std=0.02)

    def _global_token(self, tokens):
        return tokens.mean(dim=1, keepdim=True)

    def forward(self, sam_feats, grid_size=None):
        """
        sam_feats: [B, L, sam_dim]
        Returns: [B, 32, llama_dim]
        """
        B, L, _ = sam_feats.shape
        H, W = (grid_size or self.grid_size or _infer_hw_from_len(L))

        feats = self.sam_to_proj(sam_feats)   # [B, L, 1024]

        # Multi-scale token sets.
        x1 = feats
        x2, _, _ = _pool_grid_tokens(feats, H, W, scale=2)
        x4, _, _ = _pool_grid_tokens(feats, H, W, scale=4)
        xg = self._global_token(feats)

        # Apply segmentation-aware gating.
        x1g, x2g, x4g, xgg = self.gate(x1), self.gate(x2), self.gate(x4), self.gate(xg)

        # Cross-attend each query group to its corresponding token scale.
        def process(q_param, cross_layers, kv):
            q = q_param.expand(B, -1, -1)
            for blk in cross_layers:
                q = blk(q, kv)
            return q

        out_x1 = process(self.q_x1, self.cross_x1, x1g)
        out_x2 = process(self.q_x2, self.cross_x2, x2g)
        out_x4 = process(self.q_x4, self.cross_x4, x4g)
        out_glb = process(self.q_global, self.cross_glb, xgg)

        vis_tokens = torch.cat([out_x1, out_x2, out_x4, out_glb], dim=1)  # [B, 32, 1024]
        if self.pad_to_square:
            B, Q, D = vis_tokens.shape
            s = int(math.ceil(math.sqrt(Q))) if self.target_square_side is None else self.target_square_side
            assert s * s >= Q, "target_square_side too small"
            pad = s * s - Q
            if pad > 0:
                vis_tokens = torch.cat([vis_tokens, self.pad_token.expand(B, pad, D)], dim=1)  # [B, s*s, 1024]
        # Project to `llama_dim`.
        vis_tokens = self.to_llama(vis_tokens)  # [B, 32, llama_dim]
        return vis_tokens

class CalibratedTextProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, widen: int = 2, use_residual: bool = False):
        super().__init__()
        mid = max(out_dim * widen, out_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid),
            nn.GELU(),
            nn.Linear(mid, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.use_residual = use_residual and (in_dim == out_dim)
        self.text_type = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.log_temp = nn.Parameter(torch.zeros(1))  # Scalar temperature parameter.

        nn.init.orthogonal_(self.net[3].weight, gain=0.5)
        if self.net[3].bias is not None:
            nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        y = self.net(x)
        if self.use_residual:
            y = y + x
        scale = self.log_temp.exp()  # [1]
        y = F.normalize(y + self.text_type, dim=-1) * scale
        return y


class TinyCrossAttn(nn.Module):
    def __init__(self, d=256, bias=False):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=bias)
        self.wk = nn.Linear(d, d, bias=bias)
        self.wv = nn.Linear(d, d, bias=bias)
        self.out = nn.Linear(d, d, bias=bias)  # Output projection layer.
        self.dropout = nn.Dropout(p=0.0)       # Keep at 0 unless regularization is needed.

    def forward(self, q_vec, kv):
        """
        q_vec: [M, d]      One query per segmentation token.
        kv:    [M, N, d]   Row-aligned SAM tokens for each query.
        returns:
          v_pos: [M, d]    Attention-pooled positive feature.
          attn:  [M, N]    Attention weights over grid tokens.
        """
        M, N, d = kv.shape
        q = self.wq(q_vec).unsqueeze(1)        # [M, 1, d]
        k = self.wk(kv)                        # [M, N, d]
        v = self.wv(kv)                        # [M, N, d]
        # Scaled dot-product attention.
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / (d ** 0.5)  # [M, 1, N]
        attn = attn_logits.softmax(dim=-1)                              # [M, 1, N]
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v).squeeze(1)                          # [M, d]
        out = self.out(ctx)                                             # [M, d]
        return out, attn.squeeze(1)                                     # (v_pos, attn_w)
    
    
def save_out_mm_projector(model_engine, out_dir, fname="out_mm_projector.pt"):
    # Unwrap DeepSpeed/DataParallel containers.
    module = getattr(model_engine, "module", model_engine)
    projector = module.get_model().out_mm_projector

    # Only rank 0 writes the projector checkpoint.
    is_main = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        torch.save(projector.state_dict(), path)
        print(f"[save] wrote projector weights to {path}")
