# PPO Training Setup Troubleshooting Guide

Environment: VAGEN-eb-alfred, Qwen2.5-VL-3B-Instruct SFT → PPO on EB-ALFRED
Framework: verl + SGLang + Ray + FSDP
Hardware: 4× NVIDIA H200 (143GB each)

---

## 1. `extra_special_tokens` AttributeError (tokenizer_config.json)

**Error:**
```
AttributeError: 'list' object has no attribute 'keys'
```

**Cause:** SFT training used transformers 5.3.0 which saves `extra_special_tokens` as a list. The vagen env uses transformers 4.56.1 which expects a dict.

**Fix:** Edit `tokenizer_config.json` in the SFT model directory:
```python
import json
path = "<model_dir>/tokenizer_config.json"
with open(path) as f: cfg = json.load(f)
cfg['extra_special_tokens'] = {}
with open(path, 'w') as f: json.dump(cfg, f, indent=2)
```

**Note:** If the model is on HuggingFace Hub, download it locally first, fix the file, then point the training script to the local path. Do NOT use the HF repo ID directly — the fix will be lost on fresh downloads.

**After downloading from HF Hub, always apply BOTH fixes (this one + `rope_scaling` below) before running training.**

---

## 2. `rope_scaling` AttributeError in SGLang

**Error:**
```
AttributeError: 'Qwen2_5_VLConfig' object has no attribute 'rope_scaling'
File ".../sglang/srt/models/qwen2_5_vl.py", line 516, in __init__
    self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling
```

**Cause:** transformers 5.3.0 drops the `rope_scaling` field when re-saving Qwen2.5-VL configs. SGLang's qwen2_5_vl.py requires it.

**Fix:** Copy `rope_scaling` from the base model config into the SFT model config:
```python
import json
base_cfg = json.load(open("<base_model_dir>/config.json"))
sft_cfg  = json.load(open("<sft_model_dir>/config.json"))
sft_cfg['rope_scaling'] = base_cfg['rope_scaling']  # {"type": "mrope", "mrope_section": [16, 24, 24]}
# base model path: Qwen/Qwen2.5-VL-3B-Instruct (any local snapshot works)
json.dump(sft_cfg, open("<sft_model_dir>/config.json", 'w'), indent=2)
```

---

## 3. `max_new_tokens` negative value in multi-turn rollout

**Error:**
```
ValueError: max_new_tokens must be at least 0, got -2052
```

**Cause:** In `async_sglang_server.py`, `max_model_len = prompt_length + response_length`. With `max_prompt_length=2048` and `max_response_length=512`, `max_model_len=2560`. In multi-turn rollout, the accumulated context (system prompt + images + history) can exceed this limit, causing a negative value.

**Fix:** Increase `data.max_prompt_length` to cover the maximum multi-turn context length. For EB-ALFRED with `max_turns=6` and `obs_image_size=500`:
```bash
data.max_prompt_length=4096
```
This sets `max_model_len = 4096 + 512 = 4608`, which covers most multi-turn trajectories.

---

## 4. CUDA VMM SIGSEGV (`free_cache_engine=True`)

**Error:**
```
[torch_memory_saver.cpp] CUresult error result=2 func=cu_mem_create line=104
Worker exit type: SYSTEM_ERROR (SIGSEGV)
```

**Cause:** SGLang's `free_cache_engine=True` uses CUDA Virtual Memory Management (VMM) APIs to sleep/wake the model. With larger `max_model_len`, the VMM allocation request fails with `CUDA_ERROR_INVALID_VALUE`.

**Fix:**
```bash
actor_rollout_ref.rollout.free_cache_engine=False
```
This keeps SGLang always loaded in GPU memory. On H200 143GB with a 3B model, this is fine — the model only takes ~6GB.

**Trade-off:** Without sleep/wake, SGLang occupies GPU memory during training updates. With `param_offload=True` for FSDP, this is not a problem.

---

## 5. NCCL/CUDA state pollution from crashed runs

**Errors:**
```
CUDA error: the launch timed out and was terminated (in _move_states_to_device)
ncclUnhandledCudaError: Failed to CUDA calloc async 2432 bytes
Exception: The current node timed out during startup (Ray GCS overloaded)
```

**Cause:** Multiple SIGKILL'd training runs leave residual NCCL shared memory segments, CUDA contexts, and Ray GCS state. Subsequent runs inherit corrupted state.

**Fix — between runs:**
```bash
PATH=/venv/vagen/bin:$PATH ray stop --force
pkill -9 -f "raylet|gcs_server|ray::"
rm -rf /tmp/ray /tmp/ray_* /dev/shm/nccl-* /tmp/nccl-*
ipcs -m | grep -v "^--\|key\|0x00000000" | awk '{print $2}' | xargs -r ipcrm -m
sleep 10
# Verify GPU is clean before restarting:
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

**Fix — if still failing:** Reboot the machine. This fully clears all CUDA contexts and NCCL state. Always verify `memory.used ≈ 0 MiB` on all GPUs before starting a new run.

**Note:** The AI2-THOR (Unity) rendering processes from the ALFRED server appear as GPU processes (~2GB, type "G"), but they do not block CUDA compute. This is expected and harmless.

---

## 6. `/dev/shm` model path lost after reboot

**Cause:** `/dev/shm` is a RAM disk — contents are wiped on reboot. If the SFT model was saved there (e.g., `SAVE_DIR=/dev/shm/sft`), it will be gone after restart.

**Fix:** Either:
1. Upload to HuggingFace Hub before shutting down (done — model at `err00rr/Qwen2.5-VL-3B-Instruct-EB-ALFRED-vagen-sft`)
2. Download and store in `/workspace` (persists across reboots):
```bash
huggingface-cli download err00rr/Qwen2.5-VL-3B-Instruct-EB-ALFRED-vagen-sft \
  --local-dir /workspace/models/Qwen2.5-VL-3B-Instruct-EB-ALFRED-vagen-sft
```
3. Remember to re-apply the `extra_special_tokens` and `rope_scaling` fixes after downloading.

---

## 7. Disk space — use `/dev/shm` for PPO experiment outputs

**Context:** The root disk (`/`) may only have ~27GB free, insufficient for PPO checkpoints (~38GB each). `/dev/shm` is a 503GB RAM disk.

**Fix:** Set `EXPERIMENT_DIR` to `/dev/shm`:
```bash
EXPERIMENT_DIR=/dev/shm/ppo_exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
```
**Warning:** Contents are lost on reboot — save important checkpoints to `/workspace` or HuggingFace Hub promptly.

---

## Recommended launch script settings

See `examples/eb_alfred/train_ppo_no_concat_qwen25vl3b_sft.sh` for the full working script. Key settings that differ from the original `train_ppo_no_concat_qwen25vl3b.sh`:

| Parameter | Original | Fixed |
|-----------|----------|-------|
| `data.max_prompt_length` | 2048 | **4096** |
| `actor_rollout_ref.rollout.free_cache_engine` | True | **False** |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.6 | **0.5** |
| `trainer.val_before_train` | True | **False** |
| `trainer.test_freq` | 20 | **5** |
| `EXPERIMENT_DIR` | `/root/workspace/.../exps/` | **`/dev/shm/ppo_exps/...`** |
| `REF_MODEL_PATH` | `Qwen/Qwen2.5-VL-3B-Instruct` | **local fixed SFT model path** |
