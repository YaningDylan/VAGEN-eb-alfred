#!/bin/bash
# Diagnostic script to reproduce and identify the NCCL hang.
#
# Hypothesis: group_by_modality_length=True causes different ranks to receive
# image-only vs text-only batches. Since the visual encoder's merger has
# requires_grad=True, ranks that skip the merger (text-only batches) produce
# no gradient for it, breaking DDP's all-reduce synchronization.
#
# This script runs just 10 training steps with NCCL debug logging to confirm.
# It tests two variants:
#   MODE=reproduce  → original settings (should hang)
#   MODE=fix        → with ddp_find_unused_parameters=True (should work)
#
# Usage:
#   MODE=reproduce bash scripts/debug_nccl.sh
#   MODE=fix       bash scripts/debug_nccl.sh

set -euo pipefail

MODE=${MODE:-reproduce}

LLM_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_FOLDER="/root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset"
DATA_YAML="/root/workspace/era_sft_data/vagen_format/stage.yaml"
EVAL_TRAJ="/root/workspace/era_sft_data/vagen_format/trajectory_vagen.json"
DEBUG_OUT="/root/workspace/VAGEN-eb-alfred/exps/debug"

mkdir -p "${DEBUG_OUT}/checkpoints"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable NCCL debug output so we can see which collective ops each rank issues
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=COLL
# Shorten timeout from 10 minutes to 60 seconds so we fail fast
export NCCL_TIMEOUT=60000
export TORCH_NCCL_BLOCKING_WAIT=1

echo "========================================================"
echo "MODE: ${MODE}"
echo "NCCL_TIMEOUT: ${NCCL_TIMEOUT}ms (60s for fast failure)"
echo "========================================================"

# Extra flag for fix mode
# Fix: disable group_by_modality_length so all ranks always get mixed
# image+text batches, ensuring the visual merger is called on every rank
# at every step → symmetric NCCL all-reduce for merger gradients.
EXTRA_FLAGS=""
GROUP_BY_MODALITY="True"
if [ "${MODE}" = "fix" ]; then
    GROUP_BY_MODALITY="False"
    echo "Fix mode: group_by_modality_length=False (all ranks get same batch type)"
fi

cd /root/workspace/VAGEN-eb-alfred

torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 29501 \
    scripts/train_sft_vagen.py \
    --deepspeed scripts/ds_zero2_h100.json \
    --data_path "${DATA_YAML}" \
    --image_folder "${IMAGE_FOLDER}" \
    --model_name_or_path "${LLM_PATH}" \
    --group_by_modality_length ${GROUP_BY_MODALITY} \
    --bf16 True \
    --output_dir "${DEBUG_OUT}/checkpoints" \
    --max_steps 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_data_path "${EVAL_TRAJ}" \
    --n_eval_samples 20 \
    --eval_strategy "steps" \
    --eval_steps 5 \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --bf16 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --freeze_visual_encoder True \
    --report_to none \
    --run_name "debug-nccl-${MODE}" \
    2>&1 | tee "${DEBUG_OUT}/debug_${MODE}.log"

echo ""
echo "Done. Log: ${DEBUG_OUT}/debug_${MODE}.log"
