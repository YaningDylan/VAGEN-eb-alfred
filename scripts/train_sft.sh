#!/bin/bash
# Full-parameter SFT on 4x H100, aligned with ERA original hyperparameters.
#
# ERA original: 2 GPU × batch 8 = effective 16
# This setup: 4 GPU × batch 4 = effective 16 (matched)
#
# Key differences from ERA:
#   - ZeRO-2 instead of ZeRO-3 (4xH100 has plenty of VRAM, no offloading needed)
#   - eval_strategy="steps" with 5% hold-out eval set (ERA had eval_strategy="no")
#   - model_max_length=8192 (aligned with ERA)
#
# Prerequisites:
#   1. Convert ERA data to VAGEN format:
#      python scripts/convert_era_to_vagen.py \
#          --trajectory_path /root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset/eb_alfred_trajectory_augmented_prior_dataset.json \
#          --env_anchored_path /root/workspace/era_sft_data/EB-ALFRED_environment_anchored_prior_dataset/eb_alfred_environment_anchored_prior_dataset.json \
#          --external_path /root/workspace/era_sft_data/EB-ALFRED_external_knowledge_prior_dataset/eb_alfred_external_knowledge_prior_dataset.json \
#          --output_dir /root/workspace/era_sft_data/vagen_format
#
#   2. Ensure embench-vagen venv is active
#
# Usage:
#   bash scripts/train_sft.sh

set -euo pipefail

# ── Configuration ──
LLM_VERSION=Qwen2.5-VL-3B-Instruct
LLM_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_DIR="/root/workspace/VAGEN-eb-alfred/exps/sft"
IMAGE_FOLDER="/root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset"
DATA_YAML="/root/workspace/era_sft_data/vagen_format/stage.yaml"

RUN_NAME="${LLM_VERSION}-sft-stage"
echo "SFT_RUN_NAME: ${RUN_NAME}"

# ── GPU setup (4x H100) ──
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4

cd /root/workspace/VAGEN-eb-alfred

torchrun \
    --nproc_per_node ${NPROC} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 29500 \
    scripts/train_sft_vagen.py \
    --deepspeed scripts/ds_zero2_h100.json \
    --data_path "${DATA_YAML}" \
    --image_folder "${IMAGE_FOLDER}" \
    --model_name_or_path "${LLM_PATH}" \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${SAVE_DIR}/checkpoints/${RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --eval_split_ratio 0.05 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --freeze_visual_encoder True \
    --report_to none \
    --run_name "${RUN_NAME}"

echo ""
echo "====================================="
echo "Training complete!"
echo "Model saved to: ${SAVE_DIR}/checkpoints/${RUN_NAME}"
echo "====================================="
