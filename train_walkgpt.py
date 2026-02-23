import argparse
import logging
import math
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
import copy
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from model.walkgpt import walkgptForCausalLM
from model.llava_walkgpt import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, AverageMeter,
                         ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.matcher import match_pred
from utils.multi_reason_seg_val_dataset import MultiReasonSegValDataset
from utils.PAVE_dataset import PAVEDataset, PAVEValDataset

import json

def unwrap_model_with_attr(model, attribute):
    current = model
    for _ in range(10):
        if hasattr(current, attribute):
            return current
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "base_model"):
            current = current.base_model
            continue
        break
    return current

def parse_args(args):
    parser = argparse.ArgumentParser(description="walkgpt Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="/data/Rafi/dataset/PixelLM-13B/hf_model/"         # initializing the weights from pixellm
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="/data/Rafi/dataset/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="PAVE", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="PAVE|val", type=str)
    parser.add_argument("--dataset_dir", default="data/Rafi/dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="walkgpt", type=str)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--steps_per_epoch", default=54, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--ce_loss_weight", default=0.1, type=float)
    parser.add_argument("--dice_loss_weight", default=0.05, type=float)
    parser.add_argument("--bce_loss_weight", default=0.35, type=float)
    parser.add_argument("--nce_loss_weight", default=0.3, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=20, type=int)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/data/Rafi/dataset/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--num_classes_per_question", default=1, type=int)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--masks_process_with_clip", default=False, action="store_true")
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=1024, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--use_expand_question_list", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--projector_ckpt", type=str, default="", help="path to projector-only ckpt")

    
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)

        log_filename = os.path.join(args.log_dir, 'meta.log')
        i = 1
        while os.path.exists(log_filename):
            log_filename = os.path.join(args.log_dir, 'meta_{}.log'.format(str(i)))
            i += 1
        logger = logging.getLogger('walkgpt_logger')
        logger.setLevel(logging.INFO)


        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(args)
    
    else:
        writer = None
        logger = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.seg_token_num*args.image_feature_scale_num == 1:
        tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num*args.image_feature_scale_num)]
        tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]

    extra_text_tokens = [
        "[p]",
        "[/p]",
        "[distance]",
        "[/distance]",
        "[assessment]",
        "[/assessment]",
    ]
    tokenizer.add_tokens(extra_text_tokens)
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_token_num": args.seg_token_num,
        "logger": logger,
        "tokenizer": tokenizer,
        "local_rank": args.local_rank,
        "pad_train_clip_images": args.pad_train_clip_images,
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,
        "separate_mm_projector": args.separate_mm_projector,
        "masks_process_with_clip": args.masks_process_with_clip,
        "image_feature_scale_num": args.image_feature_scale_num,

    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    
    model = walkgptForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_walkgpt_modules(model.get_model().config)
    
    
    for p in vision_tower.parameters():
        p.requires_grad = False
    if args.resize_vision_tower_size == 224:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "mask_decoder",
                                "image_feature_neck",
                                "prompt_encoder",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    if args.weight:
        raw_state = torch.load(args.weight, map_location="cpu")
        state_dict = raw_state
        if isinstance(state_dict, dict) and "module" in state_dict:
            module_block = state_dict["module"]
            if isinstance(module_block, dict):
                state_dict = module_block
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        clean_state = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith("module.") else key
            clean_state[new_key] = value
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        if missing or unexpected:
            warn_msg = "[weight] load_state_dict missing keys: {} unexpected keys: {}".format(missing, unexpected)
            if logger is not None:
                logger.warning(warn_msg)
            else:
                print(warn_msg)
    if getattr(args, "projector_ckpt", ""):
        base = unwrap_model_with_attr(model, "get_model")
        proj = base.get_model().out_mm_projector

        sd = torch.load(args.projector_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]

        norm = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("out_mm_projector."):
                k = k[len("out_mm_projector.") :]
            norm[k] = v

        proj.load_state_dict(norm, strict=True)
    # Make task heads trainable while keeping the vision tower frozen.
    trainable_list = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
    if args.resize_vision_tower_size != 224:
        trainable_list.append('mm_projector')
        trainable_list.append('out_mm_projector')
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in trainable_list
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    
    world_size = max(1, torch.cuda.device_count())
    args.distributed = world_size > 1
    dataset_tokens = [d.strip().lower() for d in args.dataset.split("||") if d.strip()]
    effective_micro_batch = max(1, args.batch_size) * max(
        1, args.grad_accumulation_steps
    ) * world_size
    samples_per_epoch = effective_micro_batch * max(1, args.steps_per_epoch)

    if len(dataset_tokens) == 1 and dataset_tokens[0] == "PAVE":
        PAVE_preview = PAVEDataset(
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            samples_per_epoch=None,
            precision=args.precision,
            image_size=args.image_size,
            seg_token_num=args.seg_token_num * args.image_feature_scale_num,
            pad_train_clip_images=args.pad_train_clip_images,
            masks_process_with_clip=args.masks_process_with_clip,
            preprocessor_config=args.preprocessor_config,
        )
        total_samples = len(PAVE_preview)
        if total_samples == 0:
            raise RuntimeError("PAVE Dataset returned zero training samples.")
        args.steps_per_epoch = max(
            1, math.ceil(total_samples / effective_micro_batch)
        )
        samples_per_epoch = total_samples
        msg = (
            f"[steps_per_epoch] Using {total_samples} PAVE samples, "
            f"global_micro_batch={effective_micro_batch}, "
            f"steps_per_epoch={args.steps_per_epoch}."
        )
        if logger is not None:
            logger.info(msg)
        elif args.local_rank == 0:
            print(msg)

    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=samples_per_epoch,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
        seg_token_num=args.seg_token_num*args.image_feature_scale_num,
        num_classes_per_question=args.num_classes_per_question,
        pad_train_clip_images=args.pad_train_clip_images,
        masks_process_with_clip=args.masks_process_with_clip,
        preprocessor_config=args.preprocessor_config,
        use_expand_question_list=args.use_expand_question_list,

    )
    print("____seg_token_num in data:________: ", args.seg_token_num*args.image_feature_scale_num)
    multi_val = False
    if args.no_eval == False:
        token_num = args.seg_token_num*args.image_feature_scale_num 
        if len(args.val_dataset.split('||')) == 1:
            dataset_key = args.val_dataset.split('|')[0]
            if dataset_key == 'MultiReasonSeg':
                ValDataset_type = MultiReasonSegValDataset
            elif dataset_key.lower() == 'PAVE':
                ValDataset_type = PAVEValDataset
            else:
                ValDataset_type = ValDataset

            val_dataset_names = [args.val_dataset]
            if ValDataset_type is PAVEValDataset:
                PAVE_kwargs = {
                    "tokenizer": tokenizer,
                    "vision_tower": args.vision_tower,
                    "samples_per_epoch": None,
                    "precision": args.precision,
                    "image_size": args.image_size,
                    "seg_token_num": token_num,
                    "pad_val_clip_images": args.pad_train_clip_images,
                    "masks_process_with_clip": args.masks_process_with_clip,
                    "preprocessor_config": args.preprocessor_config,
                }
                val_dataset = PAVEValDataset(**PAVE_kwargs)
            else:
                val_dataset = ValDataset_type(
                    args.dataset_dir,
                    tokenizer,
                    args.vision_tower,
                    args.val_dataset,
                    args.image_size,
                    seg_token_num=token_num,
                    pad_val_clip_images=args.pad_train_clip_images,
                    masks_process_with_clip=args.masks_process_with_clip,
                    preprocessor_config=args.preprocessor_config,
                )
            print(
                f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
            )
        else:
            multi_val = True
            val_dataset_names = args.val_dataset.split('||')
            val_dataset = []
            for val_dataset_name in val_dataset_names:
                dataset_key = val_dataset_name.split('|')[0]
                if dataset_key == 'MultiReasonSeg':
                    ValDataset_type = MultiReasonSegValDataset
                elif dataset_key.lower() == 'PAVE':
                    ValDataset_type = PAVEValDataset
                else:
                    ValDataset_type = ValDataset
                if ValDataset_type is PAVEValDataset:
                    PAVE_kwargs = {
                        "tokenizer": tokenizer,
                        "vision_tower": args.vision_tower,
                        "samples_per_epoch": None,
                        "precision": args.precision,
                        "image_size": args.image_size,
                        "seg_token_num": token_num,
                        "pad_val_clip_images": args.pad_train_clip_images,
                        "masks_process_with_clip": args.masks_process_with_clip,
                        "preprocessor_config": args.preprocessor_config,
                    }
                    val_dataset.append(PAVEValDataset(**PAVE_kwargs))
                else:
                    val_dataset.append(
                        ValDataset_type(
                            args.dataset_dir,
                            tokenizer,
                            args.vision_tower,
                            val_dataset_name,
                            args.image_size,
                            seg_token_num=token_num,
                            pad_val_clip_images=args.pad_train_clip_images,
                            masks_process_with_clip=args.masks_process_with_clip,
                            preprocessor_config=args.preprocessor_config,
                        )                   
                    )

    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": max(0, args.warmup_steps),
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # Resume from the latest checkpoint when auto-resume is enabled.
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # Build validation data loader(s).
    if val_dataset is not None:
        assert args.val_batch_size == 1
        if multi_val:
            val_sampler = [torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False, drop_last=False
            ) for dataset in val_dataset]
            val_loader = [torch.utils.data.DataLoader(
                dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,
                ),
            ) for dataset, sampler in zip(val_dataset, val_sampler)]
        else:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,
                ),
            )

    train_iter = iter(train_loader)
    best_miou, cur_ciou = 0.0, 0.0

    if args.eval_only:
        if args.val_dataset.split('|')[0] == 'MultiReasonSeg':
            ar_validate(val_loader, model_engine, 0, writer, args, logger, val_dataset_names, tokenizer, args.seg_token_num, args.image_feature_scale_num)
        else:
            giou, ciou, miou = validate(val_loader, model_engine, 0, writer, args, logger, val_dataset_names)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch.
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou, miou = validate(val_loader, model_engine, epoch, writer, args, logger, val_dataset_names)
            is_best = miou > best_miou
            best_miou = max(miou, best_miou)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "best_ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_miou{:.3f}_ciou{:.3f}.pth".format(
                            best_miou, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)

        save_dir = os.path.join(args.log_dir, "ckpt_model")
        if args.local_rank == 0:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    epoch_start = time.time()
    epoch_bar = None
    if args.local_rank == 0:
        epoch_bar = tqdm.tqdm(
            total=args.steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
        )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    nce_losses = AverageMeter("NceLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            nce_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # Switch to train mode.
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            nce_loss = output_dict.get("nce_loss")

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            if nce_loss is not None:
                nce_losses.update(nce_loss.item(), input_dict["images"].size(0))

            model.backward(loss)
            model.step()

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                nce_losses.all_reduce()


            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                if nce_losses.count > 0:
                    writer.add_scalar("train/nce_loss", nce_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            if epoch_bar is not None:
                epoch_bar.set_postfix(
                    batch_time=f"{batch_time.avg:.2f}",
                    data_time=f"{data_time.avg:.2f}",
                    epoch_time=f"{time.time() - epoch_start:.1f}s",
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            nce_losses.reset()

        if epoch_bar is not None:
            epoch_bar.update(1)

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    epoch_duration = time.time() - epoch_start
    if epoch_bar is not None:
        epoch_bar.set_postfix(
            epoch_time=f"{epoch_duration:.1f}s",
        )
        epoch_bar.close()
        tqdm.tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} finished in {epoch_duration:.1f}s"
        )
        if writer is not None:
            writer.add_scalar("metrics/epoch_time_sec", epoch_duration, epoch)

    return train_iter



def ar_validate(val_loader, model_engine, epoch, writer, args, logger, val_dataset_names, tokenizer, seg_token_num=1, image_feature_scale_num=1):

    pred_file = []
    acc_iou_list = []
    log_dir = args.log_dir
    out_file = os.path.join(log_dir, 'out_file_{}.json'.format(args.local_rank))
    acc_iou_out_file = os.path.join(log_dir, 'acc_list_{}.json'.format(args.local_rank))
    model_engine.eval()
    if not isinstance(val_loader, list):
        val_loader = [val_loader]
    assert len(val_dataset_names) == len(val_loader)
    k = 0
    for loader, dataset_name in zip(val_loader, val_dataset_names):
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        metric_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tp_counter = torch.zeros(1, device=metric_device, dtype=torch.float64)
        fp_counter = torch.zeros(1, device=metric_device, dtype=torch.float64)
        fn_counter = torch.zeros(1, device=metric_device, dtype=torch.float64)
        num_classes = 2
        target_meter = torch.zeros(num_classes, device=metric_device, dtype=torch.float64)
        pred_meter = torch.zeros(num_classes, device=metric_device, dtype=torch.float64)
        for input_dict in tqdm.tqdm(loader):
            image_pred = {}
            image_pred['answers'] = []
            image_pred['question_gt_category_name'] = []
            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            images = input_dict['images']
            images_clip = input_dict['images_clip']
            resize_list = input_dict['resize_list']
            clip_resize_list = input_dict['clip_resize_list']
            label_list = input_dict['label_list']
            input_ids = input_dict['input_ids']
            gt_masks = input_dict['masks_list']
            questions_list = input_dict['questions_list']
            original_size_list = [label.shape for label in label_list]

          
            if k == 0:
                model_engine(**input_dict)
            
            output_ids, pred_masks, batch_seg_token_counts, _ = model_engine.base_model.evaluate(images_clip, images, input_ids, resize_list, clip_resize_list, original_size_list, max_new_tokens=512, tokenizer=tokenizer)
            text_outputs = []
            for output_id in output_ids:
                _output_id = copy.deepcopy(output_id[0])
                _output_id[_output_id==-200] = 31999
                text_output = tokenizer.decode(_output_id, skip_special_tokens=False)
                text_output = (
                    text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
                    .replace("\n", "")
                    .replace("  ", "")
                )
                text_outputs.append(text_output)
     
            print("idx:", k, "image_path:", input_dict['image_paths'][0], "text_output: ", text_outputs)
            k += 1

            batch_seg_token_count = batch_seg_token_counts[0]
            batch_seg_token_count = batch_seg_token_count.cumsum(-1)  
            batch_seg_token_count = torch.cat(
                    [torch.zeros(1).long().cuda(), batch_seg_token_count], dim=0
                )
            pred_mask = pred_masks[0]
            gt_mask = gt_masks[0]
            max_num = max(len(pred_masks[0]), len(gt_masks[0]))
            assigned_gt_masks = []
            assigned_pred_masks = []

            questions_list = input_dict['questions_list']
            gt_target_count = questions_list[0][1]
            gt_category_name = questions_list[0][2]
            prompt_ins = questions_list[0][3]
            gt_target_count = torch.tensor(gt_target_count).to(batch_seg_token_count).cumsum(-1)  
            gt_target_count = torch.cat(
                    [torch.zeros(1).long().cuda(), gt_target_count], dim=0
                )

            assign_length = []
            assign_indice = []
            assign_acc = []
            total_pred_count = []
            pred_count = []
            assert len(batch_seg_token_count) == len(gt_target_count)
            for j in range(len(batch_seg_token_count) -1):
                start_i = batch_seg_token_count[j]
                end_i = batch_seg_token_count[j+1]
                q_start_i = gt_target_count[j]
                q_end_i = gt_target_count[j+1]
                question_inputs = pred_mask[start_i:end_i]
                question_targets = gt_mask[q_start_i:q_end_i]

                indice = match_pred(question_inputs.detach(), question_targets.detach())
                assigned_pred_mask = pred_mask[start_i:end_i][indice[0]]
                assigned_pred_mask = (assigned_pred_mask > 0).int()
                assigned_gt_mask = gt_mask[q_start_i:q_end_i][indice[1]]
                unassugned_indice = []
                unassugned_indice_pred = []
                for i in range(len(gt_mask[q_start_i:q_end_i])):
                    if i not in indice[1]:
                        unassugned_indice.append(i)
                for i in range(len(pred_mask[start_i:end_i])):
                    if i not in indice[0]:
                        unassugned_indice_pred.append(i)

                unassugned_indice = np.array(unassugned_indice)
                unassugned_indice_pred = np.array(unassugned_indice_pred)
                unassigned_gt_mask = gt_mask[q_start_i:q_end_i][unassugned_indice]
                unassigned_pred = pred_mask[start_i:end_i][unassugned_indice_pred]

                empty_gt = torch.zeros_like(unassigned_pred)
                empty_pred = torch.zeros_like(unassigned_gt_mask)

                assigned_gt_mask = torch.cat((assigned_gt_mask, unassigned_gt_mask))
                assigned_pred_mask = torch.cat((assigned_pred_mask, empty_pred))

                assigned_gt_mask = torch.cat((assigned_gt_mask, empty_gt))
                assigned_pred_mask = torch.cat((assigned_pred_mask, unassigned_pred))

                assigned_gt_masks.append(assigned_gt_mask)
                assigned_pred_masks.append(assigned_pred_mask)

                question_gt_category_name = gt_category_name[j]
                text_output = text_outputs[j]
                sorted_id = sorted(range(len(indice[0])), key=lambda k: indice[0][k], reverse=False)
                sorted_gt_indice = indice[1][sorted_id]
                sorted_pred_indice = indice[0][sorted_id]
                seg_token = ' '.join(['[SEG{}]'.format(str(s)) for s in range(seg_token_num*image_feature_scale_num)]) if seg_token_num*image_feature_scale_num > 1 else '[SEG]'
                _text_output = text_output
                in_count = 0
                question_gt_category_name_list = []
                for count in range(text_output.count(seg_token)):
                    if count in sorted_pred_indice:
                        _text_output = _text_output.replace(seg_token, question_gt_category_name[sorted_gt_indice[in_count]], 1)  
                        question_gt_category_name_list.append(question_gt_category_name[sorted_gt_indice[in_count]][1:-1])
                        in_count += 1
                    else:
                        question_gt_category_name_list.append('None []')
                        _text_output = _text_output.replace(seg_token, '(None [])', 1) 
  
                image_pred['image_path'] = input_dict['image_paths'][0]
                image_pred['questions'] = questions_list[0][0]
                answer = _text_output.split('ASSISTANT:')[-1]
                answer = answer.replace('<unk>', '')
                image_pred['answers'].append(answer)
                image_pred['question_gt_category_name'].append(question_gt_category_name_list)
                assign_length.extend([True]*len(indice[0]))
                assign_length.extend([False]*(len(assigned_gt_mask)-len(indice[0])))
                assign_indice.append(indice[0].tolist())
                total_pred_count.append(len(assigned_gt_mask))
                pred_count.append(len(pred_mask[start_i:end_i]))

            assigned_gt_masks = torch.cat(assigned_gt_masks)
            output_list = torch.cat(assigned_pred_masks)
            num_classes = 2
            metric_mask_device = pred_masks[0].device
            intersection = torch.zeros(num_classes, device=metric_mask_device, dtype=torch.float64)
            union = torch.zeros(num_classes, device=metric_mask_device, dtype=torch.float64)
            acc_iou = torch.zeros(num_classes, device=metric_mask_device, dtype=torch.float64)
            for mask_i, output_i, is_assign in zip(assigned_gt_masks, output_list, assign_length):
                intersection_i, union_i, target_i = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection_i = intersection_i.to(dtype=torch.float64)
                union_i = union_i.to(dtype=torch.float64)
                target_i = target_i.to(dtype=torch.float64)
                pred_i = union_i - target_i + intersection_i

                target_meter += target_i
                pred_meter += pred_i

                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target

                fg_intersection = intersection_i[1]
                fg_union = union_i[1]
                fg_target = target_i[1]
                fg_pred = pred_i[1]

                if fg_target <= 0 and fg_pred <= 0:
                    pass
                elif fg_target <= 0:
                    fp_counter += 1
                elif fg_pred <= 0:
                    fn_counter += 1
                else:
                    fg_iou = fg_intersection / (fg_union + 1e-10)
                    if fg_iou >= 0.5:
                        tp_counter += 1
                    else:
                        fp_counter += 1
                        fn_counter += 1
                assign_acc.append((intersection_i.tolist(), union_i.tolist()))
            image_pred['assign_length'] = assign_length
            image_pred['assign_indice'] = assign_indice
            image_pred['assign_acc'] = assign_acc
            image_pred['total_pred_count'] = total_pred_count
            image_pred['pred_count'] = pred_count
            image_pred['prompt_ins'] = prompt_ins
            pred_file.append(image_pred)
            
            intersection_np = intersection.cpu().numpy()
            union_np = union.cpu().numpy()
            acc_iou_np = (acc_iou / max_num).cpu().numpy()
            intersection_meter.update(intersection_np), union_meter.update(union_np)
            acc_iou_meter.update(acc_iou_np, n=max_num)
            print(acc_iou_np)

            _acc_iou = acc_iou_np.tolist()
            _acc_iou.append(max_num)
            _acc_iou.append(input_dict['image_paths'][0])
            acc_iou_list.append(_acc_iou)
        

        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

        with open(acc_iou_out_file, 'w') as f:
            json.dump(acc_iou_list, f)
        with open(out_file, 'w') as f:
            json.dump(pred_file, f)

        target_sum = target_meter.clone()
        pred_sum = pred_meter.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(target_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pred_sum, op=dist.ReduceOp.SUM)

        sum_device = metric_device
        intersection_sum = torch.as_tensor(intersection_meter.sum, device=sum_device, dtype=torch.float64)
        union_sum = torch.as_tensor(union_meter.sum, device=sum_device, dtype=torch.float64)

        detection_stats = torch.stack([tp_counter, fp_counter, fn_counter]).view(-1)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(detection_stats, op=dist.ReduceOp.SUM)
        tp_total = detection_stats[0].item()
        fp_total = detection_stats[1].item()
        fn_total = detection_stats[2].item()

        iou_class = intersection_sum / (union_sum + 1e-10)
        num_classes = intersection_sum.shape[0]
        class_ids = torch.arange(num_classes, device=sum_device)
        valid_mask_all = union_sum > 0
        valid_mask_fg = valid_mask_all & (class_ids != 0)

        miou_inclusive = iou_class[valid_mask_all].mean().item() if valid_mask_all.any() else float("nan")
        miou_exclusive = iou_class[valid_mask_fg].mean().item() if valid_mask_fg.any() else float("nan")
        ap50_inclusive = tp_total / (tp_total + fp_total + 1e-10) if (tp_total + fp_total) > 0 else float("nan")
        recall_inclusive = tp_total / (tp_total + fn_total + 1e-10) if (tp_total + fn_total) > 0 else float("nan")

        gt_mask_all = target_sum > 0
        pred_mask_all = pred_sum > 0
        iou_hits = (iou_class >= 0.5) & gt_mask_all

        fg_mask = class_ids != 0
        misses_fg = (gt_mask_all & fg_mask) & (~iou_hits)
        false_pos_fg = (pred_mask_all & fg_mask) & (~gt_mask_all)
        tp_fg = (iou_hits & fg_mask).sum().item()
        fn_fg = misses_fg.sum().item()
        fp_fg = false_pos_fg.sum().item()
        ap50_no_bg = float("nan")
        recall_no_bg = float("nan")
        if tp_fg + fp_fg > 0:
            ap50_no_bg = tp_fg / (tp_fg + fp_fg + 1e-10)
        if tp_fg + fn_fg > 0:
            recall_no_bg = tp_fg / (tp_fg + fn_fg + 1e-10)

        ciou = iou_class[1].item()
        giou = acc_iou_meter.avg[1]

        if args.local_rank == 0:
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/ciou", ciou, epoch)
            writer.add_scalar("val/miou_inclusive", miou_inclusive, epoch)
            writer.add_scalar("val/miou_exclusive", miou_exclusive, epoch)
            writer.add_scalar("val/miou", miou_exclusive, epoch)
            writer.add_scalar("val/ap50_inclusive", ap50_inclusive, epoch)
            writer.add_scalar("val/recall_inclusive", recall_inclusive, epoch)
            print(
                (
                    "{}, epoch: {}, giou: {:.4f}, ciou: {:.4f}, mIoU(all): {:.4f}, "
                    "mIoU(no-bg): {:.4f}, AP50(all): {:.4f}, Recall(all): {:.4f}, "
                    "AP50(no-bg): {:.4f}, Recall(no-bg): {:.4f}"
                ).format(
                    dataset_name,
                    epoch,
                    giou,
                    ciou,
                    miou_inclusive,
                    miou_exclusive,
                    ap50_inclusive,
                    recall_inclusive,
                    ap50_no_bg,
                    recall_no_bg,
                )
            )
            logger.info(
                (
                    "{}, epoch: {}, giou: {:.4f}, ciou: {:.4f}, mIoU(all): {:.4f}, "
                    "mIoU(no-bg): {:.4f}, AP50(all): {:.4f}, Recall(all): {:.4f}, "
                    "AP50(no-bg): {:.4f}, Recall(no-bg): {:.4f}"
                ).format(
                    dataset_name,
                    epoch,
                    giou,
                    ciou,
                    miou_inclusive,
                    miou_exclusive,
                    ap50_inclusive,
                    recall_inclusive,
                    ap50_no_bg,
                    recall_no_bg,
                )
            )



def validate(val_loader, model_engine, epoch, writer, args, logger, val_dataset_names):
    model_engine.eval()

    if not isinstance(val_loader, list):
        val_loader = [val_loader]

    # Values returned from the last evaluated dataset.
    last_giou, last_ciou, last_miou = 0.0, 0.0, 0.0

    for loader, dataset_name in zip(val_loader, val_dataset_names):
        if 'NYU' in dataset_name:
            continue

        # Keep per-class vectors (2 classes) throughout aggregation.
        num_classes = 2
        device = torch.device('cuda', torch.cuda.current_device())

        inter_total = torch.zeros(num_classes, device=device, dtype=torch.float64)
        union_total = torch.zeros(num_classes, device=device, dtype=torch.float64)
        giou_sum = torch.zeros(num_classes, device=device, dtype=torch.float64)
        giou_count = torch.zeros(num_classes, device=device, dtype=torch.float64)
        target_total = torch.zeros(num_classes, device=device, dtype=torch.float64)
        pred_total = torch.zeros(num_classes, device=device, dtype=torch.float64)
        tp_counter = torch.zeros(1, device=device, dtype=torch.float64)
        fp_counter = torch.zeros(1, device=device, dtype=torch.float64)
        fn_counter = torch.zeros(1, device=device, dtype=torch.float64)

        for input_dict in tqdm.tqdm(loader):
            torch.cuda.empty_cache()

            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            with torch.no_grad():
                output_dict = model_engine(**input_dict)

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            for mask_i, output_i in zip(masks_list, output_list):
                inter_i, union_i, target_i = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), num_classes, ignore_index=255
                )

                inter_i = inter_i.to(dtype=torch.float64)
                union_i = union_i.to(dtype=torch.float64)
                target_i = target_i.to(dtype=torch.float64)
                pred_i = union_i - target_i + inter_i

                # Accumulate per-class intersections and unions.
                inter_total += inter_i
                union_total += union_i
                target_total += target_i
                pred_total += pred_i

                # Per-class IoU for gIoU averaging; treat no-object as 1.0.
                giou_sample = inter_i / (union_i + 1e-5)
                giou_sample[union_i == 0] += 1.0
                giou_sum   += giou_sample
                giou_count += 1.0

                fg_intersection = inter_i[1]
                fg_union = union_i[1]
                fg_target = target_i[1]
                fg_pred = pred_i[1]

                if fg_target <= 0 and fg_pred <= 0:
                    pass
                elif fg_target <= 0:
                    fp_counter += 1
                elif fg_pred <= 0:
                    fn_counter += 1
                else:
                    fg_iou = fg_intersection / (fg_union + 1e-10)
                    if fg_iou >= 0.5:
                        tp_counter += 1
                    else:
                        fp_counter += 1
                        fn_counter += 1

        target_sum = target_total.clone()
        pred_sum = pred_total.clone()

        # Distributed reduction preserving the class axis.
        if dist.is_available() and dist.is_initialized():
            for t in (inter_total, union_total, giou_sum, giou_count, target_sum, pred_sum):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            detection_stats = torch.stack([tp_counter, fp_counter, fn_counter]).view(-1)
            dist.all_reduce(detection_stats, op=dist.ReduceOp.SUM)
        else:
            detection_stats = torch.stack([tp_counter, fp_counter, fn_counter]).view(-1)

        # Final per-class vectors.
        iou_class = inter_total / (union_total + 1e-10)   # [background, foreground]
        giou_class = giou_sum / (giou_count + 1e-10)      # [background, foreground]

        # Foreground (class 1) scalars for logging and return values.
        ciou = iou_class[1].item()
        giou = giou_class[1].item()
        num_classes = inter_total.shape[0]
        class_ids = torch.arange(num_classes, device=inter_total.device)
        valid_mask_all = union_total > 0
        valid_mask_fg = valid_mask_all & (class_ids != 0)
        miou_inclusive = iou_class[valid_mask_all].mean().item() if valid_mask_all.any() else float("nan")
        miou_exclusive = iou_class[valid_mask_fg].mean().item() if valid_mask_fg.any() else float("nan")
        tp_total = detection_stats[0].item()
        fp_total = detection_stats[1].item()
        fn_total = detection_stats[2].item()
        ap50_inclusive = tp_total / (tp_total + fp_total + 1e-10) if (tp_total + fp_total) > 0 else float("nan")
        recall_inclusive = tp_total / (tp_total + fn_total + 1e-10) if (tp_total + fn_total) > 0 else float("nan")
        gt_mask_all = target_sum > 0
        pred_mask_all = pred_sum > 0
        iou_hits = (iou_class >= 0.5) & gt_mask_all
        fg_mask = class_ids != 0
        misses_fg = (gt_mask_all & fg_mask) & (~iou_hits)
        false_pos_fg = (pred_mask_all & fg_mask) & (~gt_mask_all)
        tp_fg = (iou_hits & fg_mask).sum().item()
        fn_fg = misses_fg.sum().item()
        fp_fg = false_pos_fg.sum().item()
        ap50_no_bg = float("nan")
        recall_no_bg = float("nan")
        if tp_fg + fp_fg > 0:
            ap50_no_bg = tp_fg / (tp_fg + fp_fg + 1e-10)
        if tp_fg + fn_fg > 0:
            recall_no_bg = tp_fg / (tp_fg + fn_fg + 1e-10)
        last_ciou, last_giou, last_miou = ciou, giou, miou_exclusive

        if args.local_rank == 0:
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/ciou", ciou, epoch)
            writer.add_scalar("val/miou_inclusive", miou_inclusive, epoch)
            writer.add_scalar("val/miou_exclusive", miou_exclusive, epoch)
            writer.add_scalar("val/miou", miou_exclusive, epoch)
            writer.add_scalar("val/ap50_inclusive", ap50_inclusive, epoch)
            writer.add_scalar("val/recall_inclusive", recall_inclusive, epoch)
            print(
                f"{dataset_name}, epoch: {epoch}, giou: {giou:.4f}, ciou: {ciou:.4f}, "
                f"mIoU(all): {miou_inclusive:.4f}, mIoU(no-bg): {miou_exclusive:.4f}, "
                f"AP50(all): {ap50_inclusive:.4f}, Recall(all): {recall_inclusive:.4f}, "
                f"AP50(no-bg): {ap50_no_bg:.4f}, Recall(no-bg): {recall_no_bg:.4f}"
            )
            logger.info(
                f"{dataset_name}, epoch: {epoch}, giou: {giou:.4f}, ciou: {ciou:.4f}, "
                f"mIoU(all): {miou_inclusive:.4f}, mIoU(no-bg): {miou_exclusive:.4f}, "
                f"AP50(all): {ap50_inclusive:.4f}, Recall(all): {recall_inclusive:.4f}, "
                f"AP50(no-bg): {ap50_no_bg:.4f}, Recall(no-bg): {recall_no_bg:.4f}"
            )
    return last_giou, last_ciou, last_miou




if __name__ == "__main__":
    main(sys.argv[1:])
