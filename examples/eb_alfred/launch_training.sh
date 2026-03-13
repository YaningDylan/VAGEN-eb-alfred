#!/bin/bash
set -x

cd /root/workspace/VAGEN-eb-alfred

PROJECT_NAME="vagen_experiments"
EXPERIMENT_NAME="grpo_eb_alfred_qwen25vl3b"
EXPERIMENT_DIR="exps/${PROJECT_NAME}/${EXPERIMENT_NAME}"
SAVE_CHECKPOINT_DIR="${EXPERIMENT_DIR}/verl_checkpoints"
DATASET_TRAIN="examples/eb_alfred/train_eb_alfred_grpo_vision.yaml"
DATASET_VAL="examples/eb_alfred/val_eb_alfred_grpo_vision.yaml"
agent_loop_config_path="vagen/configs/agent.yaml"
REF_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_LOG="${EXPERIMENT_DIR}/train.log"

mkdir -p "${EXPERIMENT_DIR}"

PYTHONUNBUFFERED=1 /venv/embench-vagen/bin/python3 -m vagen.main_ppo \
    --config-path=/root/workspace/VAGEN-eb-alfred/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=/root/workspace/VAGEN-eb-alfred/${DATASET_TRAIN} \
    data.val_files=/root/workspace/VAGEN-eb-alfred/${DATASET_VAL} \
    data.train_batch_size=4 \
    data.max_prompt_length=9000 \
    data.max_response_length=8000 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,hf_model,optimizer,extra]' \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=/root/workspace/VAGEN-eb-alfred/${agent_loop_config_path} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=/root/workspace/VAGEN-eb-alfred/${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=/root/workspace/VAGEN-eb-alfred/${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=/root/workspace/VAGEN-eb-alfred/${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=16 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    filter.enable=False \
    trainer.total_training_steps=50 2>&1 | \
    tee ${TRAIN_LOG}
