# Copyright (c) 2026 Rafi Ibn Sultan
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_walkgpt import MultiScaleQFormerProjector, CalibratedTextProjector, TinyCrossAttn

from .llava_walkgpt.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .segment_anything.modeling import PromptEncoder, TwoWayTransformer, LayerNorm2d, MaskDecoderMultiScale
from utils.utils_walkgpt import dice_loss, sigmoid_ce_loss, infonce_loss



class walkgptMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):  
        super(walkgptMetaModel, self).__init__(config)
        self.logger = kwargs.get("logger", None)
        self.local_rank = kwargs.get("local_rank", 1)
        self.config = config
        
        
        # Resolve runtime flags without overriding explicit config values.
        self.vision_pretrained = kwargs.get("vision_pretrained", getattr(self, "vision_pretrained", None))

        # Populate missing configuration defaults.
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs.get("train_mask_decoder", False)
        if not hasattr(self.config, "out_dim"):
            self.config.out_dim = kwargs.get("out_dim", getattr(self.config, "hidden_size", None))
        if not hasattr(self.config, "vision_tower_for_mask"):
            # False: use SAM visual model path. True: use custom mask adapter path.
            self.config.vision_tower_for_mask = kwargs.get("vision_tower_for_mask", False)

        # Build model modules once during construction.
        self.initialize_walkgpt_modules(self.config)

        # Configure decoder trainability after module initialization.
        if getattr(self.config, "train_mask_decoder", False):
            # SAM decoder path.
            if hasattr(self, "visual_model") and hasattr(self.visual_model, "mask_decoder"):
                self.visual_model.mask_decoder.train()
                for p in self.visual_model.mask_decoder.parameters():
                    p.requires_grad = True
            # Adapter decoder path.
            if hasattr(self, "mask_decoder"):
                self.mask_decoder.train()
                for p in self.mask_decoder.parameters():
                    p.requires_grad = True



    def initialize_walkgpt_modules(self, config):

        if self.config.vision_tower_for_mask:
            
            prompt_embed_dim = 256 
            H = self.config.hidden_size

            image_size = config.resize_vision_tower_size
            mask_decoder_transformer_depth = 2
            if self.local_rank == 0 and self.logger is not None:
                self.logger.info('--------build_sam_decoder--------')
                self.logger.info('--------sam decoder image size {}--------'.format(image_size))
            vit_patch_size = 14
            image_embedding_size = image_size // vit_patch_size
            self.prompt_encoder=PromptEncoder(
                        embed_dim=prompt_embed_dim,
                        image_embedding_size=(image_embedding_size, image_embedding_size),
                        input_image_size=(image_size, image_size),
                        mask_in_chans=16,
                    )
             
            self.mask_decoder=MaskDecoderMultiScale(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=mask_decoder_transformer_depth,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                image_feature_scale_num=config.image_feature_scale_num
            ) 
            
            
            embed_dim = self.config.hidden_size 
            out_chans = prompt_embed_dim
            self.image_feature_neck = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            )

        else:
            prompt_embed_dim = 256 
            H = self.config.hidden_size


            self.tiny_xattn = TinyCrossAttn(d=256)
            self.out_mm_projector = MultiScaleQFormerProjector(
                sam_dim=256,
                llama_dim=H,
                pad_to_square=True,
                target_square_side=6,  # 32 tokens are padded to 36 (6x6).
            )
            print('--------Loading SAM weights now--------')
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
            for param in self.visual_model.parameters():
                param.requires_grad = False
            if config.train_mask_decoder:
                self.visual_model.mask_decoder.train()
                for param in self.visual_model.mask_decoder.parameters():
                    param.requires_grad = True

        # Text projector used to map hidden states to segmentation embedding space.
        in_dim = config.hidden_size
        out_dim = config.out_dim

        self.text_hidden_fcs = nn.ModuleList([
            CalibratedTextProjector(in_dim=in_dim, out_dim=out_dim, widen=2, use_residual=False)
        ])
        
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class walkgptModel(walkgptMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(walkgptModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class walkgptForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        kwargs.update({
            "image_feature_scale_num": 1, 
            "pad_train_clip_images": True,
            "resize_vision_tower": True,
            "resize_vision_tower_size": 448,
            "vision_tower_for_mask": False,
            "separate_mm_projector": True,
        })
        self.logger = kwargs.get("logger", None)
        config.resize_vision_tower = kwargs.get("resize_vision_tower", False)
        config.resize_vision_tower_size = kwargs.get("resize_vision_tower_size", 224)
        config.pad_train_clip_images = kwargs.get("pad_train_clip_images", False)
        config.vision_tower_for_mask = kwargs.get("vision_tower_for_mask", False)
        config.separate_mm_projector = kwargs.get("separate_mm_projector", False)
        config.mm_projector_hidden_dim = 2
        config.mm_projector_out_dim = 1
        self.image_feature_scale_num = kwargs.get("image_feature_scale_num", 1)
        config.image_feature_scale_num = kwargs.get("image_feature_scale_num", 1)
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.nce_loss_weight = kwargs.pop("nce_loss_weight", None)
        
        self.vision_tower_for_mask = kwargs.get("vision_tower_for_mask", False)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.seg_token_num = kwargs.get("seg_token_num", 1)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.local_rank = kwargs.get("local_rank", 1)
        self.pad_train_clip_images = kwargs.get("pad_train_clip_images", False)
        self.masks_process_with_clip = kwargs.get("masks_process_with_clip", False)
        
        logger = kwargs.get("logger", None)
        if isinstance(self.seg_token_idx, list):
            if self.local_rank == 0 and logger is not None:
                print('--------initialize multiseg scalar--------')
            seg_token_num = len(self.seg_token_idx)
            scalar = 1 / seg_token_num
            self.multiseg_scalar = [torch.nn.Parameter(torch.ones([]) * scalar) for _ in range(seg_token_num)]
        if self.image_feature_scale_num > 1:
            scalar = 1 / self.image_feature_scale_num
            self.multiscale_scalar = [torch.nn.Parameter(torch.ones([]) * scalar) for _ in range(self.image_feature_scale_num)]
        super().__init__(config)
        self.model = walkgptModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Final Hugging Face model initialization.
        self.post_init()
        self.iter = 0
        self.iter1 = 0
         
        if config.resize_vision_tower_size != 224:
            if self.local_rank == 0 and self.logger is not None:
                self.logger.info('--------mm_projector requires grad--------')
            for n, p in self.model.named_parameters():
                if any([x in n for x in ["mm_projector"]]):
                    p.requires_grad = True
                    


    def get_visual_embs(self, pixel_values):
        # Support both direct and wrapped model layouts.
        vm = getattr(self, "visual_model", None)

        # Common path: visual model is attached to the inner model.
        if vm is None and hasattr(self, "model") and hasattr(self.model, "visual_model"):
            vm = self.model.visual_model

        if vm is None or not hasattr(vm, "image_encoder"):
            raise RuntimeError(
                "visual_model not initialized. Make sure initialize_walkgpt_modules() "
                "was called during __init__ (before DeepSpeed initialize)."
            )

        out = vm.image_encoder(pixel_values)
        if isinstance(out, dict) and "image_embeddings" in out:
            out = out["image_embeddings"]
        return out



    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        clip_resize_list = None,
        **kwargs,
    ):
        # `offset` maps each image to its corresponding text rows.
        batch_size = images.shape[0]
        assert batch_size == len(offset) - 1
        

        if isinstance(self.seg_token_idx, list):
            seg_token_num = self.seg_token_num
            seg_token_mask = torch.zeros_like(input_ids[:, 1:]).bool()
            for seg_token_idx in self.seg_token_idx:
                seg_token_mask = seg_token_mask | (input_ids[:, 1:] == seg_token_idx)  
        else:
            seg_token_num = self.seg_token_num
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )

        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            # Inference expects a single image paired with multiple prompts.
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1, "inference branch assumes one image"

            # Encode once, then reuse projected tokens across prompt rows.
            with torch.no_grad():
                feats = self.model.visual_model.image_encoder(images[0].unsqueeze(0))
                tokens_raw = feats.flatten(2).transpose(1, 2)

            tokens_proj = self.model.out_mm_projector(tokens_raw)

            # Expand image tokens to match the number of prompt rows.
            tokens_proj_extend = tokens_proj.expand(length, -1, -1).contiguous()
            sam_tokens_256 = tokens_raw.expand(length, -1, -1).contiguous()

            # Keep resize metadata aligned with each prompt row.
            extend_clip_resize_list = [clip_resize_list[0]] * length

            # Feed precomputed vision tokens directly into the language model.
            output_i = super().forward(
                images=tokens_proj_extend,
                attention_mask=attention_masks[:length],
                input_ids=input_ids[:length],
                output_hidden_states=True,
                clip_resize_list=extend_clip_resize_list,
            )

            output = output_i
            output_hidden_states = output.hidden_states

            Lev = self.image_feature_scale_num
            # Match training layout: [num_scales, rows, tokens, hidden_dim].
            output_image_features = torch.stack(
                [tokens_proj_extend for _ in range(Lev)],
                dim=0,
            )


        else:
            sam_tokens_list = []
            extend_clip_resize_list = []

            # Run the SAM encoder once for the full image batch.
            sam_feats = self.get_visual_embs(images)

            # Normalize encoder outputs to shape [B, C, Hs, Ws].
            if isinstance(sam_feats, dict) and "image_embeddings" in sam_feats:
                sam_feats = sam_feats["image_embeddings"]
            if isinstance(sam_feats, (list, tuple)):
                sam_feats = torch.stack(sam_feats, dim=0)

            proj = getattr(self.model, "out_mm_projector", self.model.mm_projector)

            # Expand each image embedding over its associated text rows.
            sam_tokens_256_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]

                feats = sam_feats[i].unsqueeze(0)
                tokens_raw = feats.flatten(2).transpose(1, 2)
                sam_tokens_256_list.append(tokens_raw.expand(end_i - start_i, -1, -1).contiguous())
                proj = getattr(self.model, "out_mm_projector", None)
                if proj is None:
                    # Fallback path for CLIP-token projector.
                    proj = self.model.mm_projector
                tokens_proj = proj(tokens_raw)


                tokens_proj = tokens_proj.expand(end_i - start_i, -1, -1).contiguous()
                sam_tokens_list.append(tokens_proj)

                # Keep resize metadata aligned for each expanded text row.
                extend_clip_resize_list.extend([clip_resize_list[i]] * (end_i - start_i))


            sam_tokens = torch.cat(sam_tokens_list, dim=0)
            sam_tokens_256 = torch.cat(sam_tokens_256_list, dim=0)

            # Forward with precomputed image tokens.
            output = super().forward(
                images=sam_tokens,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                clip_resize_list=extend_clip_resize_list
            )

            output_hidden_states = output.hidden_states
            output_image_features = output.image_features
            

        hidden_states = []

         
        
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        pred_embeddings_nce = pred_embeddings
        
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )
        
        seg_token_offset = seg_token_offset[offset]
        feat_scale_num = self.image_feature_scale_num
        
        # Keep one query per segmentation token (no query fusion).
        pred_embeddings_ = []
        batch_seg_token_counts = []

        for img_idx in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[img_idx], seg_token_offset[img_idx + 1]
            batch_pred_embeddings = pred_embeddings[start_i:end_i]
            total = batch_pred_embeddings.shape[0]

            assert total % (seg_token_num * feat_scale_num) == 0, \
                f"Bad pack: total={total}, seg_token_num={seg_token_num}, feat_scale_num={feat_scale_num}"

            Q = total // (seg_token_num * feat_scale_num)
            D = batch_pred_embeddings.shape[-1]

            batch_pred_embeddings = batch_pred_embeddings.view(Q, feat_scale_num, seg_token_num, D)

            # Use the highest-resolution scale.
            batch_pred_embeddings = batch_pred_embeddings[:, -1, :, :]

            batch_pred_embeddings = batch_pred_embeddings.reshape(Q * seg_token_num, D)

            pred_embeddings_.append(batch_pred_embeddings)
            batch_seg_token_counts.append(batch_pred_embeddings.shape[0])

        pred_embeddings = pred_embeddings_

        multi_scale_num = len(output_image_features)

        # Build row indices for SEG-to-SAM alignment loss.
        rows = sam_tokens_256.size(0)
        row_ids = torch.arange(rows, device=sam_tokens_256.device)
        seg_row_ids = torch.repeat_interleave(row_ids, seg_token_mask.sum(dim=1))
        
        M = seg_row_ids.numel()
        loss_nce = pred_embeddings_nce.new_zeros(())
        tiny_xattn = getattr(self.model, "tiny_xattn", self.model.tiny_xattn)
        if M > 0:
            # For a single-row batch, same-row negatives are not defined.
            exclude_same_row = (sam_tokens_256.size(0) > 1)

            loss_nce, aux = infonce_loss(
                pred_embeddings=pred_embeddings_nce,
                sam_tokens_256=sam_tokens_256,
                seg_row_ids=seg_row_ids,
                tiny_xattn=tiny_xattn,
                temperature=getattr(self.config, "nce_tau", 0.07),
                top_k=getattr(self.config, "nce_topk", 8),
                exclude_same_row=exclude_same_row,
                normalize=True,
                return_aux=True
            )
        
        if not inference:
            output_image_features = torch.stack(output_image_features, dim=0)
        img_embeddings = output_image_features.flatten(1, 2)
        img_token_mask = torch.ones(output_image_features.shape[1], output_image_features.shape[2]).to(seg_token_mask)
        img_token_counts = img_token_mask.int().sum(-1) 
        patch_count = int(img_token_counts[0])
        
        patch_size = int(patch_count**0.5)
        img_token_offset = img_token_counts.cumsum(-1)
        img_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), img_token_offset], dim=0
        )
        img_token_offset = img_token_offset[offset]
        img_embeddings_ = []
        single_img_embeddings = []
        for i in range(len(img_token_offset) - 1):
            start_i, end_i = img_token_offset[i], img_token_offset[i + 1]
            question_num = pred_embeddings_[i].shape[0] 
            img_num = img_embeddings[:, start_i:end_i].shape[1] // patch_count
            single_img_embeddings.append(img_embeddings[:, start_i:end_i].view(multi_scale_num, img_num, patch_count, img_embeddings.shape[-1]).permute(0, 1, 3, 2).view(multi_scale_num, img_num, img_embeddings.shape[-1], patch_size, patch_size)[:, 0])
            if question_num == 0:
                batch_img_embeddings = torch.zeros(multi_scale_num, 0, 4096, patch_size, patch_size).to(img_embeddings)
            else:
                batch_img_embeddings = img_embeddings[:, start_i:end_i].view(multi_scale_num, img_num, patch_count, img_embeddings.shape[-1])
                batch_img_embeddings = batch_img_embeddings.permute(0, 1, 3, 2).view(multi_scale_num, img_num, img_embeddings.shape[-1], patch_size, patch_size)
            img_embeddings_.append(batch_img_embeddings)

        img_embeddings = img_embeddings_
        image_embeddings = img_embeddings

        
        multimask_output = False
        pred_masks = []
        mask_scores = []
        pred_depths = []
         
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_depths.append([])
            

            
            pred_masks.append(pred_mask[:, 0])
            mask_score = (pred_mask[:, 0].sigmoid().flatten(1) * (pred_mask[:, 0] > 0).flatten(1)).sum(1) / ((pred_mask[:, 0] > 0).flatten(1).sum(1) + 1e-6)
            mask_scores.append(mask_score)


        model_output = output
        gt_masks = masks_list


        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "batch_seg_token_counts": batch_seg_token_counts,
                "mask_scores": mask_scores,
            }

        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = ce_loss
        mask_bce_loss = pred_masks[0].sum() * 0
        mask_dice_loss = pred_masks[0].sum() * 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
             
      
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
             
            
        
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)

        mask_loss = mask_bce_loss + mask_dice_loss

        
        nce_loss = 0.2 * loss_nce
        loss = loss + mask_loss + nce_loss

    
        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "nce_loss": nce_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        clip_resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
         
        all_pred_embeddings = []
        all_output_ids = []
        batch_seg_token_counts = []
        with torch.no_grad():
            for idx, input_id in enumerate(input_ids):
                if 0 in input_id:
                    unk_start = torch.where(input_id==0)[0].min()
                    _input_id = input_id[:unk_start]
                else:
                    _input_id = input_id
                outputs = self.generate(
                    images=images_clip,
                    input_ids=_input_id[None],
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    clip_resize_list=clip_resize_list
                )
                output_hidden_states = outputs.hidden_states[-1]
                output_ids = outputs.sequences
                all_output_ids.append(output_ids)
       
                if isinstance(self.seg_token_idx, list):
                    seg_token_num = self.seg_token_num
                    seg_token_mask = torch.zeros_like(output_ids[:, 1:]).bool()
                      
                    for seg_token_idx in self.seg_token_idx:
                        seg_token_mask = seg_token_mask | (output_ids[:, 1:] == seg_token_idx)  
                
                else:
                    seg_token_num = self.seg_token_num
                    seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
                # Account for prepended image tokens in the generated sequence.
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )

                hidden_states = []
            
                assert len(self.model.text_hidden_fcs) == 1
                hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
                feat_scale_num = self.image_feature_scale_num
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                pred_embeddings = last_hidden_state[seg_token_mask]

                if len(pred_embeddings) % (seg_token_num*feat_scale_num) != 0:
                    seg_token_mask = (seg_token_mask*0).bool()
                    pred_embeddings = last_hidden_state[seg_token_mask]

                seg_token_counts = seg_token_mask.int().sum(-1)
                seg_token_offset = seg_token_counts.cumsum(-1)
                seg_token_offset = torch.cat(
                    [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
                )
                seg_token_offset = seg_token_offset[[0, len(seg_token_offset)-1]]
                pred_embeddings_ = []
                
                
                for img_idx in range(len(seg_token_offset) - 1):
                    start_i, end_i = seg_token_offset[img_idx], seg_token_offset[img_idx + 1]
                    batch_pred_embeddings = pred_embeddings[start_i:end_i]
                    total = batch_pred_embeddings.shape[0]

                    assert total % (seg_token_num * feat_scale_num) == 0, \
                        f"Bad pack: total={total}, seg_token_num={seg_token_num}, feat_scale_num={feat_scale_num}"

                    Q = total // (seg_token_num * feat_scale_num)
                    D = batch_pred_embeddings.shape[-1]

                    batch_pred_embeddings = batch_pred_embeddings.view(Q, feat_scale_num, seg_token_num, D)

                    # Use the highest-resolution scale.
                    batch_pred_embeddings = batch_pred_embeddings[:, -1, :, :]

                    batch_pred_embeddings = batch_pred_embeddings.reshape(Q * seg_token_num, D)

                    pred_embeddings_.append(batch_pred_embeddings)
                    batch_seg_token_counts.append(batch_pred_embeddings.shape[0])

                pred_embeddings = pred_embeddings_
                all_pred_embeddings.extend(pred_embeddings)
            
            batch_seg_token_counts = [torch.tensor(batch_seg_token_counts).to(seg_token_counts)]
            pred_embeddings = [torch.cat(all_pred_embeddings)]
            
            multimask_output = False
            pred_masks = []
            mask_scores = []

            image_embeddings = self.get_visual_embs(images)

            
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                
                pred_masks.append(pred_mask[:, 0])
                mask_score = (pred_mask[:, 0].sigmoid().flatten(1) * (pred_mask[:, 0] > 0).flatten(1)).sum(1) / ((pred_mask[:, 0] > 0).flatten(1).sum(1) + 1e-6)
                mask_scores.append(mask_score)
             
        
        return all_output_ids, pred_masks, batch_seg_token_counts, mask_scores


    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
        ) -> torch.Tensor:
            """
            Remove padding and upscale masks to the original image size.

            Arguments:
            masks (torch.Tensor): Batched masks from the mask_decoder,
                in BxCxHxW format.
            input_size (tuple(int, int)): The size of the image input to the
                model, in (H, W) format. Used to remove padding.
            original_size (tuple(int, int)): The original size of the image
                before resizing for input to the model, in (H, W) format.

            Returns:
            (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
                is given by original_size.
            """
           
            target_size = max(input_size)
            dtype = masks.dtype
            if self.vision_tower_for_mask:
                masks = F.interpolate(
                    masks.float(),
                    (target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            
            if not self.masks_process_with_clip:
                assert input_size[0] <= target_size
                assert input_size[1] <= target_size
                masks = masks[..., : input_size[0], : input_size[1]]
                masks = F.interpolate(
                    masks, original_size, mode="bilinear", align_corners=False
                )
            
            masks = masks.to(dtype)
            return masks    

    
