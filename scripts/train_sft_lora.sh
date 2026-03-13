#!/bin/bash
# LoRA SFT training for VAGEN-format EB-ALFRED data.
# Single GPU, no DeepSpeed needed.
set -e

LLM_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_DIR="/root/workspace/VAGEN-eb-alfred/exps/sft/checkpoints/qwen25vl3b-lora"
IMAGE_FOLDER="/root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset"
DATA_YAML="/root/workspace/era_sft_data/vagen_format/stage.yaml"

export CUDA_VISIBLE_DEVICES=0

cd /root/workspace/VAGEN-eb-alfred

/venv/embench-vagen/bin/python scripts/train_sft_lora.py \
    --data_path ${DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --model_name_or_path ${LLM_PATH} \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to none \
    --run_name qwen25vl3b-lora-alfred

echo ""
echo "====================================="
echo "LoRA training complete!"
echo "Adapter saved to: ${SAVE_DIR}"
echo "====================================="
