import torch
import torch.nn as nn

class SAMVisionTower(nn.Module):
    """
    Minimal wrapper so LLaVA can treat SAM like a CLIP tower:
    - input: Bx3xH xW (SAM-preprocessed pixels)
    - returns:
        tokens: BxN xC_raw (flattened patch tokens for the LLM path)
        feats:  BxC_rawxHs xWs (spatial maps for decoders or aux use)
        meta:   {Hs, Ws, N}
    """
    def __init__(self, sam_model: nn.Module):
        super().__init__()
        self.sam = sam_model
        for p in self.sam.parameters():
            p.requires_grad = False
        self._grid = None  # cache (Hs, Ws)

    @torch.no_grad()
    def encode(self, images: torch.Tensor):
        # SAM ViT-H encoder → BxC_raw xHs xWs
        feats = self.sam.image_encoder(images)
        B, C, Hs, Ws = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # BxN xC_raw
        self._grid = (Hs, Ws)
        meta = {"Hs": Hs, "Ws": Ws, "N": Hs * Ws, "C_raw": C}
        return tokens, feats, meta
