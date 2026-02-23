# WalkGPT Training and Evaluation Instructions

This document provides the reference commands used for training and evaluation in the released WalkGPT codebase.

## Environment and Dependencies

Use the following runtime assumptions:

- Python environment: `/home/hm4013/.conda/envs/walkgpt`
- Multi-GPU launcher: `deepspeed` (training) or `torchrun` (standalone evaluation)
- Dataset root: `/data/Rafi/dataset`
- Base language model: `/data/Rafi/dataset/PixelLM-13B/hf_model/`
- CLIP vision tower: `/data/Rafi/dataset/clip-vit-large-patch14-336`
- Preprocessor config: `./configs/preprocessor_448.json`

The commands below explicitly set:

- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `CUDA_VISIBLE_DEVICES=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Adjust `CUDA_VISIBLE_DEVICES` and `--master_port` as needed for your machine.

## 1. Training Command

Use this command for standard WalkGPT training on `PAVE`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hm4013/.conda/envs/walkgpt/bin/deepspeed --master_port=24999 train_walkgpt.py --version "/data/Rafi/dataset/PixelLM-13B/hf_model/" --dataset_dir="/data/Rafi/dataset" --dataset="PAVE" --sample_rates="1" --sem_seg_data="PAVE" --exp_name="walkgpt-13b_reb" --seg_token_num=1 --num_classes_per_question=20 --batch_size=16 --pad_train_clip_images --preprocessor_config="./configs/preprocessor_448.json" --resize_vision_tower --resize_vision_tower_size=448 --use_expand_question_list --image_feature_scale_num=1 --separate_mm_projector --log_base_dir="/data/Rafi/llava_weights/" --projector_ckpt="/home/hm4013/walkgpt/runs/walkllava-7b-projector/ckpt_model/out_mm_projector.pt" --vision-tower="/data/Rafi/dataset/clip-vit-large-patch14-336" --epochs=5 --steps_per_epoch=54
```

## 2. Evaluation via `train_walkgpt.py` (`--eval_only`)

Use this command to run evaluation through the training script pipeline:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hm4013/.conda/envs/walkgpt/bin/deepspeed --master_port=24999 train_walkgpt.py --version "/data/Rafi/dataset/PixelLM-13B/hf_model/" --dataset_dir="/data/Rafi/dataset" --dataset="PAVE" --sample_rates="1" --sem_seg_data="PAVE" --exp_name="walkgpt-13b" --seg_token_num=1 --num_classes_per_question=20 --batch_size=16 --pad_train_clip_images --preprocessor_config="./configs/preprocessor_448.json" --resize_vision_tower --resize_vision_tower_size=448 --use_expand_question_list --image_feature_scale_num=1 --separate_mm_projector --log_base_dir="/data/Rafi/llava_weights/" --projector_ckpt="/home/hm4013/walkgpt/runs/walkllava-7b-projector/ckpt_model/out_mm_projector.pt" --vision-tower="/data/Rafi/dataset/clip-vit-large-patch14-336" --epochs=10 --eval_only --weight="/data/Rafi/llava_weights/walkgpt-13b/ckpt_model/global_step270/mp_rank_00_model_states.pt"
```

## 3. Standalone Evaluation Command

Use this command to evaluate with `walkgpt_evaluation.py` and save text responses:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/hm4013/.conda/envs/walkgpt/bin/torchrun --master_port=24999 walkgpt_evaluation.py --version "/data/Rafi/dataset/PixelLM-13B/hf_model/" --dataset_dir="/data/Rafi/dataset" --val_dataset="PAVE|val" --exp_name="walkgpt-13b-eval" --seg_token_num=1 --num_classes_per_question=20 --pad_train_clip_images --preprocessor_config="./configs/preprocessor_448.json" --resize_vision_tower --resize_vision_tower_size=448 --use_expand_question_list --image_feature_scale_num=1 --separate_mm_projector --log_base_dir="/data/Rafi/llava_weights/" --vision-tower="/data/Rafi/dataset/clip-vit-large-patch14-336" --weight="/data/Rafi/llava_weights/walkgpt-13b_reb/ckpt_model/global_step270/mp_rank_00_model_states.pt" --save_responses_path="/home/hm4013/walkgpt/responses/model_reb.json" --workers=1
```

## Notes

- `--projector_ckpt` must be compatible with the configured projector architecture.
- `--weight` should point to the target checkpoint file (`mp_rank_00_model_states.pt`).
- `--resize_vision_tower_size=448` requires the matching preprocessor configuration (`preprocessor_448.json`).
- For multi-GPU execution, update `CUDA_VISIBLE_DEVICES` and launch configuration accordingly.
