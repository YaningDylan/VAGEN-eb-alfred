#!/bin/bash
# SFT training for VAGEN-format EB-ALFRED data.
#
# Self-contained trainer (no ERA special tokens).
# Base model: Qwen2.5-VL-3B-Instruct
# Output format: <think>...</think><answer>action name</answer>
#
# Prerequisites:
#   1. Convert ERA data to VAGEN format:
#      python scripts/convert_era_to_vagen.py \
#          --trajectory_path /root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset/eb_alfred_trajectory_augmented_prior_dataset.json \
#          --env_anchored_path /root/workspace/era_sft_data/EB-ALFRED_environment_anchored_prior_dataset/eb_alfred_environment_anchored_prior_dataset.json \
#          --external_path /root/workspace/era_sft_data/EB-ALFRED_external_knowledge_prior_dataset/eb_alfred_external_knowledge_prior_dataset.json \
#          --output_dir /root/workspace/era_sft_data/vagen_format
#
#   2. Ensure embench-vagen venv is active:
#      source /venv/embench-vagen/bin/activate
#
# Usage:
#   bash scripts/train_sft.sh

set -e

# ── Configuration ──
LLM_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_DIR="/root/workspace/VAGEN-eb-alfred/exps/sft"
IMAGE_FOLDER="/root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset"
DATA_YAML="/root/workspace/era_sft_data/vagen_format/stage.yaml"

RUN_NAME="qwen25vl3b-sft-vagen-alfred"
echo "SFT_RUN_NAME: ${RUN_NAME}"

# ── GPU setup ──
export CUDA_VISIBLE_DEVICES=0,1

# ── DeepSpeed config ──
DS_CONFIG="/root/workspace/VAGEN-eb-alfred/scripts/ds_zero3.json"
if [ ! -f "$DS_CONFIG" ]; then
    echo "Creating DeepSpeed Zero-3 config..."
    cat > "$DS_CONFIG" << 'DSEOF'
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
DSEOF
fi

# ── Launch training ──
NPROC=2

cd /root/workspace/VAGEN-eb-alfred

/venv/embench-vagen/bin/torchrun \
    --nproc_per_node ${NPROC} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 29500 \
    scripts/train_sft_vagen.py \
    --deepspeed ${DS_CONFIG} \
    --data_path ${DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --model_name_or_path ${LLM_PATH} \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_DIR}/checkpoints/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --freeze_visual_encoder True \
    --report_to none \
    --run_name ${RUN_NAME}

echo ""
echo "====================================="
echo "Training complete!"
echo "Model saved to: ${SAVE_DIR}/checkpoints/${RUN_NAME}"
echo ""
echo "To evaluate:"
echo "  DISPLAY=:0 /venv/embench-vagen/bin/python scripts/eval_epl_local.py \\"
echo "      --model_path ${SAVE_DIR}/checkpoints/${RUN_NAME} \\"
echo "      --n_episodes 10 --max_turns 30"
echo "====================================="
