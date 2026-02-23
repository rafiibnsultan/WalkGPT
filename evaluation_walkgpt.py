import argparse
import copy
import json
import logging
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model.walkgpt import walkgptForCausalLM
from model.llava_walkgpt import conversation as conversation_lib
from model.llava_walkgpt.mm_utils import tokenizer_image_token
from utils.dataset import ValDataset, collate_fn
from utils.matcher import match_pred
from utils.multi_reason_seg_val_dataset import MultiReasonSegValDataset
from utils.PAVE_dataset import PAVEValDataset
from utils.utils import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="walkgpt Evaluation")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
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
        "--vision-tower", default="/data/Rafi/dataset/sam_vit_h_4b8939.pth", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--val_dataset", default="PAVE|val", type=str)
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument("--sem_seg_data", default="", type=str)
    parser.add_argument("--refer_seg_data", default="", type=str)
    parser.add_argument("--vqa_data", default="", type=str)
    parser.add_argument("--reason_seg_data", default="", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="walkgpt_eval", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--nce_loss_weight", default=0.2, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/data/Rafi/dataset/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--num_classes_per_question", default=1, type=int)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--masks_process_with_clip", action="store_true", default=False)
    parser.add_argument("--preprocessor_config", default="", type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=1024, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--weight", default="", type=str, help="path to model state dict")
    parser.add_argument("--use_expand_question_list", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)
    parser.add_argument("--projector_ckpt", type=str, default="", help="path to projector-only ckpt")
    parser.add_argument(
        "--save_responses_path",
        default="",
        type=str,
        help="Optional JSON file to store per-image text responses.",
    )
    parser.add_argument(
        "--no_console_text",
        action="store_true",
        help="Disable printing per-sample text responses to stdout.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
        help="Limit evaluation to the first N samples. Use -1 to evaluate all samples.",
    )
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2", "chatml"],
    )
    return parser.parse_args(args)


def setup_distributed(args):
    # Initialize distributed execution when launched with torchrun/deepspeed.
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("CUDA reported available but device_count is 0.")
        if args.local_rank >= device_count:
            args.local_rank = 0
        torch.cuda.set_device(args.local_rank)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return distributed, world_size


def setup_logging(log_dir, local_rank):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("walkgpt_eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if local_rank == 0 and not any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ):
        file_handler = logging.FileHandler(os.path.join(log_dir, "eval.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    return logger


def build_tokenizer(args):
    # Build tokenizer and register segmentation/special multimodal tokens.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.seg_token_num * args.image_feature_scale_num == 1:
        tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = [
            "[SEG{}]".format(i) for i in range(args.seg_token_num * args.image_feature_scale_num)
        ]
        tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [
            tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens
        ]

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
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    return tokenizer


def build_model(args, tokenizer, logger):
    # Construct WalkGPT model and load optional fine-tuned/projector checkpoints.
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

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    vision_tower.to(dtype=torch_dtype, device=device)
    model.get_model().initialize_walkgpt_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False

    if args.resize_vision_tower_size == 224:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    if args.lora_r > 0:

        def find_linear_layers(module, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, sub_module in module.named_modules():
                if (
                    isinstance(sub_module, cls)
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

        lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))

    if args.weight:
        state_dict = torch.load(args.weight, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if logger is not None and hasattr(logger, "info"):
            if missing:
                logger.info("Missing keys when loading weights: %s", missing)
            if unexpected:
                logger.info("Unexpected keys when loading weights: %s", unexpected)

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

    model.to(device=device, dtype=torch_dtype)
    visual_model = getattr(model.get_model(), "visual_model", None)
    if visual_model is not None:
        visual_model.to(device=device, dtype=torch_dtype)
    model.eval()
    return model


def create_val_dataloaders(args, tokenizer):
    # Build one or more validation dataloaders based on `val_dataset`.
    if args.val_batch_size != 1:
        raise ValueError("Evaluation currently assumes --val_batch_size 1")

    token_num = args.seg_token_num * args.image_feature_scale_num
    multi_val = "||" in args.val_dataset

    def build_single_dataset(name):
        prefix = name.split("|")[0]
        if prefix == "PAVE":
            return PAVEValDataset(
                tokenizer=tokenizer,
                vision_tower=args.vision_tower,
                samples_per_epoch=None,
                seg_token_num=token_num,
                pad_val_clip_images=args.pad_train_clip_images,
                masks_process_with_clip=args.masks_process_with_clip,
                preprocessor_config=args.preprocessor_config,
            )
        dataset_cls = MultiReasonSegValDataset if prefix == "MultiReasonSeg" else ValDataset
        return dataset_cls(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            name,
            args.image_size,
            seg_token_num=token_num,
            pad_val_clip_images=args.pad_train_clip_images,
            masks_process_with_clip=args.masks_process_with_clip,
            preprocessor_config=args.preprocessor_config,
        )

    dataset_names = args.val_dataset.split("||") if multi_val else [args.val_dataset]
    datasets = [build_single_dataset(name) for name in dataset_names]

    remaining = args.max_eval_samples if args.max_eval_samples > 0 else None
    if remaining is not None:
        limited = []
        limited_names = []
        for name, dataset in zip(dataset_names, datasets):
            if remaining <= 0:
                break
            length = len(dataset)
            keep = min(length, remaining)
            if keep < length:
                dataset = Subset(dataset, list(range(keep)))
            limited.append(dataset)
            limited_names.append(name)
            remaining -= keep
        datasets = limited
        dataset_names = limited_names
        if not datasets:
            raise ValueError("max_eval_samples limited all datasets to zero length.")

    distributed = dist.is_available() and dist.is_initialized()
    loaders = []
    for dataset in datasets:
        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False, drop_last=False
            )
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
        loaders.append(loader)

    return loaders if multi_val else loaders[0], dataset_names


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


def get_core_model(model):
    module = unwrap_model_with_attr(model, "evaluate")
    return module if hasattr(module, "evaluate") else model


def _prepare_sam_tokens(model, batch):
    # Convert SAM features to projected tokens aligned with text rows.
    images_sam = batch["images"]
    clip_resize_list = batch["clip_resize_list"]
    offset = batch["offset"].tolist()

    sam_feats = model.get_visual_embs(images_sam)
    if isinstance(sam_feats, dict) and "image_embeddings" in sam_feats:
        sam_feats = sam_feats["image_embeddings"]
    if isinstance(sam_feats, (list, tuple)):
        sam_feats = torch.stack(sam_feats, dim=0)

    base = model.get_model()
    projector = getattr(base, "out_mm_projector", None)
    if projector is None:
        projector = base.mm_projector

    sam_tokens_list = []
    extended_clip_resize = []
    for idx in range(len(offset) - 1):
        start_i, end_i = offset[idx], offset[idx + 1]
        feats = sam_feats[idx].unsqueeze(0)
        tokens_raw = feats.flatten(2).transpose(1, 2)
        tokens_proj = projector(tokens_raw)
        tokens_proj = tokens_proj.expand(end_i - start_i, -1, -1).contiguous()
        sam_tokens_list.append(tokens_proj)
        extended_clip_resize.extend([clip_resize_list[idx]] * (end_i - start_i))

    if not sam_tokens_list:
        return None, []

    sam_tokens = torch.cat(sam_tokens_list, dim=0)
    return sam_tokens, extended_clip_resize


def _build_question_prompts(batch, conv_type, use_mm_start_end):
    # Build text prompts for generation from question payloads/conversation fallbacks.
    conv_template = conversation_lib.conv_templates[conv_type]
    prompts = []
    questions_clean = []
    offset = batch["offset"].tolist()
    questions_list = batch["questions_list"]
    conversation_list = batch.get("conversation_list", [])

    for idx in range(len(offset) - 1):
        start_i, end_i = offset[idx], offset[idx + 1]
        q_entry = questions_list[idx]
        if isinstance(q_entry, tuple) and q_entry:
            first_elem = q_entry[0]
            if isinstance(first_elem, (list, tuple)):
                q_entry = first_elem
        if not isinstance(q_entry, (list, tuple)):
            q_entry = [q_entry]
        for j in range(end_i - start_i):
            question = q_entry[j] if j < len(q_entry) else ""
            if question is None:
                question = ""
            if not isinstance(question, str):
                question = str(question)

            if not question.strip():
                conv_idx = start_i + j
                if conv_idx < len(conversation_list):
                    conv_str = conversation_list[conv_idx]
                    if isinstance(conv_str, str):
                        if "USER:" in conv_str:
                            user_part = conv_str.split("USER:", 1)[1]
                            if "ASSISTANT:" in user_part:
                                user_part = user_part.split("ASSISTANT:", 1)[0]
                            question = user_part.strip()
                        else:
                            question = conv_str.strip()

            question_text = question
            question_clean = question_text.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            questions_clean.append(question_clean)

            conv = conv_template.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], question_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            if use_mm_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            else:
                replace_token = DEFAULT_IMAGE_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            prompts.append(prompt)

    return prompts, questions_clean


def generate_predictions_from_questions(model, tokenizer, batch, args, max_new_tokens=512):
    # Generate text responses using projected SAM tokens as visual input.
    with torch.no_grad():
        sam_tokens, clip_resize = _prepare_sam_tokens(model, batch)
        if sam_tokens is None:
            return []

        prompts, questions = _build_question_prompts(
            batch,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
        )
        if not prompts:
            return []

        device = sam_tokens.device
        prompt_tokens = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in prompts
        ]
        if not prompt_tokens:
            return []

        prompt_lengths = [tokens.shape[0] for tokens in prompt_tokens]
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_tokens, batch_first=True, padding_value=pad_token_id
        ).to(device)
        attention_mask = input_ids.ne(pad_token_id)

        outputs = model.generate(
            images=sam_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            return_dict_in_generate=True,
            clip_resize_list=clip_resize,
        )

        sequences = outputs.sequences
        answers = []
        for seq, prompt_len in zip(sequences, prompt_lengths):
            seq = seq.to(device)
            seq[seq == -200] = pad_token_id
            prompt_len = min(prompt_len, seq.shape[0])
            gen_tokens = seq[prompt_len:].tolist()
            if tokenizer.bos_token_id is not None and gen_tokens:
                if gen_tokens[0] == tokenizer.bos_token_id:
                    gen_tokens = gen_tokens[1:]
            while gen_tokens and gen_tokens[-1] in {tokenizer.eos_token_id, pad_token_id}:
                gen_tokens.pop()
            if not gen_tokens:
                text = ""
            else:
                text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
            text = (
                text.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
                .replace("\n", " ")
                .replace("  ", " ")
                .strip()
            )
            answers.append(text)

    if questions and len(questions) != len(answers):
        min_len = min(len(questions), len(answers))
        questions = questions[:min_len]
        answers = answers[:min_len]

    return list(zip(questions, answers))


def ar_validate(val_loader, model, args, logger, dataset_names, tokenizer, text_log=None):
    # Validation path for datasets with assignment-based matching (e.g., MultiReasonSeg).
    writer = SummaryWriter(args.log_dir) if args.local_rank == 0 else None
    pred_file = []
    acc_iou_list = []
    out_file = os.path.join(args.log_dir, f"out_file_{args.local_rank}.json")
    acc_iou_out_file = os.path.join(args.log_dir, f"acc_list_{args.local_rank}.json")

    model.eval()
    if not isinstance(val_loader, list):
        val_loader = [val_loader]
    assert len(dataset_names) == len(val_loader)

    core_model = get_core_model(model)

    k = 0
    for loader, dataset_name in zip(val_loader, dataset_names):
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        iterator = tqdm.tqdm(loader, disable=(args.local_rank != 0))
        for input_dict in iterator:
            image_pred = {"answers": [], "question_gt_category_name": []}
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
            image_paths = input_dict["image_paths"]
            images = input_dict["images"]
            images_clip = input_dict["images_clip"]
            resize_list = input_dict["resize_list"]
            clip_resize_list = input_dict["clip_resize_list"]
            label_list = input_dict["label_list"]
            input_ids = input_dict["input_ids"]
            gt_masks = input_dict["masks_list"]
            questions_list = input_dict["questions_list"]
            question_payload = questions_list[0]
            if isinstance(question_payload, tuple) and question_payload:
                question_texts = question_payload[0]
            elif isinstance(question_payload, (list, tuple)):
                question_texts = question_payload
            else:
                question_texts = [question_payload]
            question_texts = [
                str(q).replace(DEFAULT_IMAGE_TOKEN, "").strip() for q in question_texts
            ]
            original_size_list = [label.shape for label in label_list]

            if k == 0:
                model(**input_dict)

            (
                output_ids,
                pred_masks,
                batch_seg_token_counts,
                mask_scores,
            ) = core_model.evaluate(
                images_clip,
                images,
                input_ids,
                resize_list,
                clip_resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )
            text_outputs = []
            for output_id in output_ids:
                _output_id = copy.deepcopy(output_id[0])
                _output_id[_output_id == -200] = 31999
                text_output = tokenizer.decode(_output_id, skip_special_tokens=False)
                text_output = (
                    text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
                    .replace("\n", "")
                    .replace("  ", "")
                )
                text_outputs.append(text_output)

            if not args.no_console_text:
                print(
                    "idx:",
                    k,
                    "image_path:",
                    image_paths[0],
                    "text_output: ",
                    text_outputs,
                )
            k += 1
            if args.local_rank == 0 and text_log is not None:
                for idx, answer_text in enumerate(text_outputs):
                    question_text = question_texts[idx] if idx < len(question_texts) else ""
                    answer_clean = answer_text.split("ASSISTANT:")[-1].replace("<unk>", "").strip()
                    text_log.append(
                        {
                            "image_path": image_paths[0],
                            "response": f"Question: {question_text} | Answer: {answer_clean}",
                        }
                    )

            batch_seg_token_count = batch_seg_token_counts[0]
            batch_seg_token_count = batch_seg_token_count.cumsum(-1)
            batch_seg_token_count = torch.cat(
                [torch.zeros(1).long().cuda(), batch_seg_token_count], dim=0
            )
            pred_mask = pred_masks[0]
            gt_mask = gt_masks[0]
            mask_score = mask_scores[0]
            max_num = max(len(pred_masks[0]), len(gt_masks[0]))
            assigned_gt_masks = []
            assigned_pred_masks = []

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
            for j in range(len(batch_seg_token_count) - 1):
                start_i = batch_seg_token_count[j]
                end_i = batch_seg_token_count[j + 1]
                q_start_i = gt_target_count[j]
                q_end_i = gt_target_count[j + 1]
                question_inputs = pred_mask[start_i:end_i]
                question_mask_scores = mask_score[start_i:end_i]
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
                sorted_id = sorted(
                    range(len(indice[0])), key=lambda idx: indice[0][idx], reverse=False
                )
                sorted_gt_indice = indice[1][sorted_id]
                sorted_pred_indice = indice[0][sorted_id]
                seg_token = (
                    " ".join(
                        ["[SEG{}]".format(str(s)) for s in range(args.seg_token_num * args.image_feature_scale_num)]
                    )
                    if args.seg_token_num * args.image_feature_scale_num > 1
                    else "[SEG]"
                )
                _text_output = text_output
                question_gt_category_name_list = []
                for count in range(text_output.count(seg_token)):
                    if count in sorted_pred_indice:
                        _text_output = _text_output.replace(seg_token, question_gt_category_name[count], 1)
                        question_gt_category_name_list.append(question_gt_category_name[count])
                    else:
                        _text_output = _text_output.replace(seg_token, "", 1)
                assign_length.append(len(question_targets))
                assign_indice.append(sorted_gt_indice.tolist())
                assign_acc.append(question_mask_scores[sorted_pred_indice].tolist())
                total_pred_count.append(len(question_inputs))
                pred_count.append(len(question_inputs))

                image_pred["answers"].append(_text_output)
                image_pred["question_gt_category_name"].append(question_gt_category_name_list)

            assigned_gt_masks = torch.cat(assigned_gt_masks)
            assigned_pred_masks = torch.cat(assigned_pred_masks)
            assigned_pred_masks = (assigned_pred_masks > 0).int()

            intersection, union, _ = intersectionAndUnionGPU(
                assigned_pred_masks.contiguous().clone(),
                assigned_gt_masks.contiguous(),
                2,
                ignore_index=255,
            )
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] += 1.0

            image_pred["pred_mask_score"] = [score.item() for score in mask_score]
            image_pred["assign_length"] = assign_length
            image_pred["assign_indice"] = assign_indice
            image_pred["assign_acc"] = assign_acc
            image_pred["total_pred_count"] = total_pred_count
            image_pred["pred_count"] = pred_count
            image_pred["prompt_ins"] = prompt_ins
            pred_file.append(image_pred)

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / max_num
            intersection_meter.update(intersection), union_meter.update(union), acc_iou_meter.update(
                acc_iou, n=max_num
            )
            print(acc_iou)

            _acc_iou = acc_iou.tolist()
            _acc_iou.append(max_num)
            _acc_iou.append(image_paths[0])
            acc_iou_list.append(_acc_iou)

        if dist.is_available() and dist.is_initialized():
            intersection_meter.all_reduce()
            union_meter.all_reduce()
            acc_iou_meter.all_reduce()

        if args.local_rank == 0:
            with open(acc_iou_out_file, "w") as f:
                json.dump(acc_iou_list, f)
            with open(out_file, "w") as f:
                json.dump(pred_file, f)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        if args.local_rank == 0:
            if writer is not None:
                writer.add_scalar("val/giou", giou, 0)
                writer.add_scalar("val/ciou", ciou, 0)
            if logger is not None:
                logger.info(
                    "%s, giou: %.4f, ciou: %.4f",
                    dataset_name,
                    giou,
                    ciou,
                )

    if writer is not None:
        writer.close()


def validate(val_loader, model, args, logger, dataset_names, tokenizer, text_log=None):
    # Standard segmentation validation path.
    writer = SummaryWriter(args.log_dir) if args.local_rank == 0 else None
    model.eval()

    if not isinstance(val_loader, list):
        val_loader = [val_loader]

    core_model = get_core_model(model)
    last_giou, last_ciou = 0.0, 0.0

    for loader, dataset_name in zip(val_loader, dataset_names):
        if "NYU" in dataset_name:
            continue

        num_classes = 2
        device = torch.device("cuda", torch.cuda.current_device())

        inter_total = torch.zeros(num_classes, device=device, dtype=torch.float64)
        union_total = torch.zeros(num_classes, device=device, dtype=torch.float64)
        giou_sum = torch.zeros(num_classes, device=device, dtype=torch.float64)
        giou_count = torch.zeros(num_classes, device=device, dtype=torch.float64)

        iterator = tqdm.tqdm(loader, disable=(args.local_rank != 0))
        for input_dict in iterator:
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
                output_dict = model(**input_dict)

            if args.local_rank == 0:
                qa_pairs = generate_predictions_from_questions(core_model, tokenizer, input_dict, args)
                for idx, (question, answer) in enumerate(qa_pairs):
                    line = (
                        f"[val text] dataset={dataset_name}, image={input_dict['image_paths'][0]}, sample={idx} "
                        f"Question: {question} | Answer: {answer}"
                    )
                    if not args.no_console_text:
                        print(line)
                    if text_log is not None:
                        text_log.append(
                            {
                                "image_path": input_dict["image_paths"][0],
                                "response": f"Question: {question} | Answer: {answer}",
                            }
                        )

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            for mask_i, output_i in zip(masks_list, output_list):
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), num_classes, ignore_index=255
                )

                inter_i = inter_i.to(dtype=torch.float64)
                union_i = union_i.to(dtype=torch.float64)

                inter_total += inter_i
                union_total += union_i

                giou_sample = inter_i / (union_i + 1e-5)
                giou_sample[union_i == 0] += 1.0
                giou_sum += giou_sample
                giou_count += 1.0

        if dist.is_available() and dist.is_initialized():
            for tensor in (inter_total, union_total, giou_sum, giou_count):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        iou_class = inter_total / (union_total + 1e-10)
        giou_class = giou_sum / (giou_count + 1e-10)

        ciou = iou_class[1].item()
        giou = giou_class[1].item()
        last_ciou, last_giou = ciou, giou

        if args.local_rank == 0:
            if writer is not None:
                writer.add_scalar("val/giou", giou, 0)
                writer.add_scalar("val/ciou", ciou, 0)
            if logger is not None:
                logger.info(
                    "%s, giou: %.4f, ciou: %.4f",
                    dataset_name,
                    giou,
                    ciou,
                )

    if writer is not None:
        writer.close()

    return last_giou, last_ciou


def main(cli_args):
    args = parse_args(cli_args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    distributed, _ = setup_distributed(args)
    args.distributed = distributed

    logger = setup_logging(args.log_dir, args.local_rank) if args.local_rank == 0 else None
    if logger is not None:
        logger.info("Starting evaluation with args: %s", args)

    tokenizer = build_tokenizer(args)
    model = build_model(args, tokenizer, logger)

    if args.local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        param_msg = f"Model parameters: {total_params:,} (~{total_params / 1e9:.2f}B)"
        seq_len = getattr(args, "model_max_length", None)
        if seq_len and seq_len > 0:
            approx_flops = 6 * total_params * seq_len
            flops_msg = (
                f"Estimated compute per batch (FLOPs, seq_len={seq_len}): {approx_flops:,}"
            )
        else:
            flops_msg = "Estimated compute per batch (FLOPs): unavailable (missing seq_len)"

        if logger is not None:
            logger.info(param_msg)
            logger.info(flops_msg)
        else:
            print(param_msg)
            print(flops_msg)

    val_loader, val_dataset_names = create_val_dataloaders(args, tokenizer)

    text_log = [] if (args.local_rank == 0 and args.save_responses_path) else None

    is_multi_reason = any(name.split("|")[0] == "MultiReasonSeg" for name in val_dataset_names)
    if is_multi_reason:
        ar_validate(val_loader, model, args, logger, val_dataset_names, tokenizer, text_log=text_log)
    else:
        validate(val_loader, model, args, logger, val_dataset_names, tokenizer, text_log=text_log)

    if text_log is not None and args.local_rank == 0:
        save_path = args.save_responses_path
        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(text_log, f, indent=2)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
