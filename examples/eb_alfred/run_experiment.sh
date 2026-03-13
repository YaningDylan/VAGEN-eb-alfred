#!/bin/bash
# =============================================================================
# EB-ALFRED GRPO Training - Robust Launch Script
#
# This script handles:
#   1. Xorg setup per GPU (with cleanup of stale X servers)
#   2. Environment server launch with proper resource limits
#   3. Training launch
#   4. Background monitoring for hangs, OOM, zombie processes
#   5. Auto-recovery on failure
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR="/root/workspace/VAGEN-eb-alfred"
VENV="/venv/embench-vagen"
PYTHON="${VENV}/bin/python3"
export PATH="${VENV}/bin:${PATH}"
NUM_GPUS=2
ENV_SERVER_PORT=8000
# Reduce thread workers to avoid thread exhaustion (was 128, too many)
ENV_SERVER_THREAD_WORKERS=32
# Limit max concurrent env sessions to prevent resource exhaustion
ENV_SERVER_MAX_SESSIONS=32
ENV_SERVER_SESSION_TIMEOUT=600

# Monitor thresholds
MONITOR_INTERVAL=30        # seconds between health checks
MAX_MEMORY_PERCENT=85      # kill Unity procs if memory exceeds this
MAX_UNITY_PROCS=40         # max Unity processes before cleanup
HANG_TIMEOUT=600           # seconds with no log output = hung

LOG_DIR="${PROJECT_DIR}/exps/vagen_experiments/grpo_eb_alfred_qwen25vl3b"
mkdir -p "${LOG_DIR}"

XORG_LOG="${LOG_DIR}/xorg.log"
ENV_SERVER_LOG="${LOG_DIR}/env_server.log"
TRAIN_LOG="${LOG_DIR}/train.log"
MONITOR_LOG="${LOG_DIR}/monitor.log"

# PID tracking
PIDS_FILE="${LOG_DIR}/.running_pids"

# ─── Helper functions ────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    log "Cleanup triggered, shutting down all processes..."

    # Kill monitor
    [ -n "${MONITOR_PID:-}" ] && kill "$MONITOR_PID" 2>/dev/null || true

    # Kill training
    [ -n "${TRAIN_PID:-}" ] && kill "$TRAIN_PID" 2>/dev/null || true

    # Kill env server
    [ -n "${ENV_SERVER_PID:-}" ] && kill "$ENV_SERVER_PID" 2>/dev/null || true

    # Kill zombie Unity processes
    pkill -f "thor-Linux64" 2>/dev/null || true
    pkill -f "unity.*LinuxPlayer" 2>/dev/null || true

    # Stop Xorg servers we started
    for d in $(seq 0 $((NUM_GPUS - 1))); do
        local lockfile="/tmp/.X${d}-lock"
        if [ -f "$lockfile" ]; then
            local xpid
            xpid=$(cat "$lockfile" 2>/dev/null | tr -d ' ')
            kill "$xpid" 2>/dev/null || true
        fi
    done

    rm -f "$PIDS_FILE"
    log "Cleanup complete."
}

trap cleanup EXIT INT TERM

# ─── Step 1: Setup Xorg per GPU ──────────────────────────────────────────────

setup_xorg() {
    log "Setting up Xorg displays for ${NUM_GPUS} GPUs..."

    # nvidia-smi gives hex bus IDs like "00000000:81:00.0"
    # Xorg needs decimal format like "PCI:129:0:0"
    local gpu_bus_ids_raw
    gpu_bus_ids_raw=($(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader))

    for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
        local display_num=$gpu_idx
        local raw_id="${gpu_bus_ids_raw[$gpu_idx]}"
        # Convert hex bus ID (e.g. "00000000:81:00.0") to Xorg decimal (e.g. "129:0:0")
        local hex_bus hex_dev hex_func
        hex_bus=$(echo "$raw_id" | sed 's/^[0-9A-Fa-f]*://' | cut -d: -f1)
        hex_dev=$(echo "$raw_id" | cut -d: -f3 | cut -d. -f1)
        hex_func=$(echo "$raw_id" | cut -d. -f2)
        local dec_bus=$((16#$hex_bus))
        local dec_dev=$((16#$hex_dev))
        local dec_func=$((16#$hex_func))
        local bus_id="${dec_bus}:${dec_dev}:${dec_func}"
        local lockfile="/tmp/.X${display_num}-lock"
        local socketfile="/tmp/.X11-unix/X${display_num}"

        # Check if a working Xorg is already running on this display
        if [ -f "$lockfile" ]; then
            local xpid
            xpid=$(cat "$lockfile" 2>/dev/null | tr -d ' ')
            if kill -0 "$xpid" 2>/dev/null; then
                log "  Display :${display_num} already running (pid=${xpid}), keeping it."
                continue
            else
                log "  Stale lock for display :${display_num} (pid=${xpid} dead), cleaning up..."
                rm -f "$lockfile" "$socketfile"
            fi
        fi

        # Generate per-GPU xorg config
        local xorg_conf="/tmp/xorg-gpu${gpu_idx}.conf"
        cat > "$xorg_conf" <<XEOF
Section "ServerLayout"
    Identifier     "Layout${gpu_idx}"
    Screen      0  "Screen${gpu_idx}"
EndSection

Section "Device"
    Identifier     "Device${gpu_idx}"
    Driver         "nvidia"
    BusID          "PCI:${bus_id}"
    Option         "ProbeAllGpus"  "false"
    Option         "AllowEmptyInitialConfiguration" "True"
EndSection

Section "Screen"
    Identifier     "Screen${gpu_idx}"
    Device         "Device${gpu_idx}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual     1024 768
    EndSubSection
EndSection

Section "ServerFlags"
    Option "AutoAddGPU" "false"
    Option "AutoAddDevices" "false"
    Option "AutoEnableDevices" "false"
EndSection

Section "Files"
    ModulePath     "/usr/lib/x86_64-linux-gnu/nvidia/xorg"
    ModulePath     "/usr/lib/xorg/modules"
EndSection
XEOF

        log "  Starting Xorg on display :${display_num} (GPU ${gpu_idx}, BusID PCI:${bus_id})..."
        Xorg ":${display_num}" -config "$xorg_conf" -noreset -logfile "/tmp/Xorg.${display_num}.log" &
        local xorg_pid=$!

        # Wait for X to be ready
        local wait_count=0
        while [ ! -S "/tmp/.X11-unix/X${display_num}" ] && [ $wait_count -lt 10 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        if [ -S "/tmp/.X11-unix/X${display_num}" ]; then
            log "  Display :${display_num} ready (pid=${xorg_pid})"
        else
            log "  WARNING: Display :${display_num} may not be ready yet"
        fi
    done
}

# ─── Step 2: Start Environment Server ────────────────────────────────────────

start_env_server() {
    log "Starting EB-ALFRED environment server on port ${ENV_SERVER_PORT}..."

    local x_displays
    x_displays=$(seq -s, 0 $((NUM_GPUS - 1)))

    cd "${PROJECT_DIR}"

    PYTHONUNBUFFERED=1 "${PYTHON}" -m vagen.envs.eb_alfred.serve \
        --port "${ENV_SERVER_PORT}" \
        --host "0.0.0.0" \
        --x-displays "${x_displays}" \
        --session-timeout "${ENV_SERVER_SESSION_TIMEOUT}" \
        --max-sessions "${ENV_SERVER_MAX_SESSIONS}" \
        --thread-workers "${ENV_SERVER_THREAD_WORKERS}" \
        > "${ENV_SERVER_LOG}" 2>&1 &
    ENV_SERVER_PID=$!

    log "Env server started (pid=${ENV_SERVER_PID}), waiting for health check..."

    # Wait for server to be ready
    local wait_count=0
    while [ $wait_count -lt 30 ]; do
        if curl -s "http://localhost:${ENV_SERVER_PORT}/health" > /dev/null 2>&1; then
            log "Env server is healthy!"
            return 0
        fi
        sleep 2
        wait_count=$((wait_count + 1))

        # Check if process died
        if ! kill -0 "$ENV_SERVER_PID" 2>/dev/null; then
            log "ERROR: Env server died during startup. Check ${ENV_SERVER_LOG}"
            tail -20 "${ENV_SERVER_LOG}"
            return 1
        fi
    done

    log "WARNING: Env server didn't respond to health check in 60s, continuing anyway..."
}

# ─── Step 3: Background Monitor ──────────────────────────────────────────────

start_monitor() {
    log "Starting background monitor..."

    (
        last_log_size=0
        stall_count=0

        while true; do
            sleep "${MONITOR_INTERVAL}"

            # 1. Check for zombie Unity processes
            local unity_count
            unity_count=$(pgrep -f "thor-Linux64|unity.*LinuxPlayer" 2>/dev/null | wc -l || echo 0)
            if [ "$unity_count" -gt "$MAX_UNITY_PROCS" ]; then
                echo "[$(date)] WARN: ${unity_count} Unity processes (max=${MAX_UNITY_PROCS}), killing oldest..."
                # Kill Unity processes older than 10 minutes
                pgrep -f "thor-Linux64" 2>/dev/null | while read pid; do
                    local age
                    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
                    if [ -n "$age" ] && [ "$age" -gt 600 ]; then
                        echo "[$(date)] Killing stale Unity process ${pid} (age=${age}s)"
                        kill -9 "$pid" 2>/dev/null || true
                    fi
                done
            fi

            # 2. Check memory usage
            local mem_percent
            mem_percent=$(free | awk '/Mem:/ {printf("%.0f", $3/$2 * 100)}')
            if [ "$mem_percent" -gt "$MAX_MEMORY_PERCENT" ]; then
                echo "[$(date)] CRITICAL: Memory at ${mem_percent}%, killing all Unity processes..."
                pkill -9 -f "thor-Linux64" 2>/dev/null || true
                pkill -9 -f "unity.*LinuxPlayer" 2>/dev/null || true
                sleep 5
            fi

            # 3. Check Xorg health - restart if dead
            for d in $(seq 0 $((NUM_GPUS - 1))); do
                local lockfile="/tmp/.X${d}-lock"
                if [ -f "$lockfile" ]; then
                    local xpid
                    xpid=$(cat "$lockfile" 2>/dev/null | tr -d ' ')
                    if ! kill -0 "$xpid" 2>/dev/null; then
                        echo "[$(date)] WARN: Xorg display :${d} died, restarting..."
                        rm -f "$lockfile" "/tmp/.X11-unix/X${d}"
                        local xorg_conf="/tmp/xorg-gpu${d}.conf"
                        if [ -f "$xorg_conf" ]; then
                            Xorg ":${d}" -config "$xorg_conf" -noreset -logfile "/tmp/Xorg.${d}.log" &
                            sleep 3
                            echo "[$(date)] Xorg display :${d} restarted"
                        fi
                    fi
                fi
            done

            # 4. Check env server health
            if [ -n "${ENV_SERVER_PID:-}" ] && ! kill -0 "$ENV_SERVER_PID" 2>/dev/null; then
                echo "[$(date)] CRITICAL: Env server died! Restarting..."
                start_env_server
            else
                # Check if server is responsive
                if ! curl -s --max-time 10 "http://localhost:${ENV_SERVER_PORT}/health" > /dev/null 2>&1; then
                    echo "[$(date)] WARN: Env server not responding to health check"
                fi
            fi

            # 5. Check for training hang (no log output for HANG_TIMEOUT)
            if [ -f "${TRAIN_LOG}" ]; then
                local cur_size
                cur_size=$(stat -c%s "${TRAIN_LOG}" 2>/dev/null || echo 0)
                if [ "$cur_size" -eq "$last_log_size" ]; then
                    stall_count=$((stall_count + 1))
                    local stall_seconds=$((stall_count * MONITOR_INTERVAL))
                    if [ "$stall_seconds" -ge "$HANG_TIMEOUT" ]; then
                        echo "[$(date)] CRITICAL: Training appears hung (no output for ${stall_seconds}s)"
                        echo "[$(date)] Checking for stuck processes..."
                        ps aux | grep -E "python.*vagen|thor-Linux" | grep -v grep | head -10
                        echo "[$(date)] Active env sessions:"
                        curl -s "http://localhost:${ENV_SERVER_PORT}/sessions" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -20
                    fi
                else
                    stall_count=0
                    last_log_size=$cur_size
                fi
            fi

            # 6. Report status
            local gpu_mem
            gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")
            local sessions
            sessions=$(curl -s "http://localhost:${ENV_SERVER_PORT}/sessions" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('num_sessions','?'))" 2>/dev/null || echo "?")
            echo "[$(date)] Status: mem=${mem_percent}% unity=${unity_count} sessions=${sessions} gpu=[${gpu_mem}]"

        done
    ) >> "${MONITOR_LOG}" 2>&1 &
    MONITOR_PID=$!
    log "Monitor started (pid=${MONITOR_PID}), logging to ${MONITOR_LOG}"
}

# ─── Step 4: Start Training ──────────────────────────────────────────────────

start_training() {
    log "Starting GRPO training..."

    cd "${PROJECT_DIR}"

    local EXPERIMENT_NAME="grpo_eb_alfred_qwen25vl3b"
    local PROJECT_NAME="vagen_experiments"
    local EXPERIMENT_DIR="${PROJECT_DIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}"
    local SAVE_CHECKPOINT_DIR="${EXPERIMENT_DIR}/verl_checkpoints"
    local DATASET_TRAIN="${PROJECT_DIR}/examples/eb_alfred/train_eb_alfred_grpo_vision.yaml"
    local DATASET_VAL="${PROJECT_DIR}/examples/eb_alfred/val_eb_alfred_grpo_vision.yaml"
    local agent_loop_config_path="${PROJECT_DIR}/vagen/configs/agent.yaml"
    local REF_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

    mkdir -p "${EXPERIMENT_DIR}"

    # Set resource limits to prevent thread exhaustion
    # Increase max user processes limit
    ulimit -u 65535 2>/dev/null || true

    PYTHONUNBUFFERED=1 "${PYTHON}" -m vagen.main_ppo \
        --config-path="${PROJECT_DIR}/vagen/configs" \
        --config-name='vagen_multiturn' \
        data.train_files="${DATASET_TRAIN}" \
        data.val_files="${DATASET_VAL}" \
        data.train_batch_size=16 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.agent.agent_loop_config_path="$agent_loop_config_path" \
        actor_rollout_ref.rollout.disable_log_stats=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.critic_warmup=0 \
        trainer.logger='[console,wandb]' \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=5 \
        trainer.test_freq=5 \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}" \
        trainer.validation_data_dir="${EXPERIMENT_DIR}/validation" \
        trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollout_data" \
        trainer.log_val_generations=16 \
        trainer.max_actor_ckpt_to_keep=2 \
        trainer.max_critic_ckpt_to_keep=1 \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path="${REF_MODEL_PATH}" \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size_per_gpu=1 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        filter.enable=False \
        trainer.total_training_steps=50 \
        > >(tee "${TRAIN_LOG}") 2>&1 &
    TRAIN_PID=$!

    log "Training started (pid=${TRAIN_PID}), logging to ${TRAIN_LOG}"
}

# ─── Main ────────────────────────────────────────────────────────────────────

main() {
    log "============================================"
    log "  EB-ALFRED GRPO Training - Robust Launch"
    log "============================================"

    # Kill any leftover processes from previous runs
    log "Cleaning up previous run artifacts..."
    pkill -f "vagen.envs.eb_alfred.serve" 2>/dev/null || true
    pkill -f "thor-Linux64" 2>/dev/null || true
    pkill -f "unity.*LinuxPlayer" 2>/dev/null || true
    # Give processes time to die
    sleep 2

    # Step 1: Xorg
    setup_xorg

    # Step 2: Env server
    start_env_server

    # Step 3: Monitor
    start_monitor

    # Step 4: Training
    start_training

    # Save PIDs
    echo "ENV_SERVER_PID=${ENV_SERVER_PID}" > "$PIDS_FILE"
    echo "TRAIN_PID=${TRAIN_PID}" >> "$PIDS_FILE"
    echo "MONITOR_PID=${MONITOR_PID}" >> "$PIDS_FILE"

    log ""
    log "All components launched successfully!"
    log "  Env Server: pid=${ENV_SERVER_PID}, log=${ENV_SERVER_LOG}"
    log "  Training:   pid=${TRAIN_PID}, log=${TRAIN_LOG}"
    log "  Monitor:    pid=${MONITOR_PID}, log=${MONITOR_LOG}"
    log ""
    log "To follow training:  tail -f ${TRAIN_LOG}"
    log "To follow monitor:   tail -f ${MONITOR_LOG}"
    log "To check sessions:   curl localhost:${ENV_SERVER_PORT}/sessions"
    log ""

    # Wait for training to finish
    wait "$TRAIN_PID"
    TRAIN_EXIT=$?

    log "Training finished with exit code ${TRAIN_EXIT}"

    # Cleanup will be called by trap
    exit $TRAIN_EXIT
}

main "$@"
