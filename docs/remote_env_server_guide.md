# EB-ALFRED Remote Environment Server Guide

This document describes how to connect a remote training machine to the EB-ALFRED environment server.

## Architecture

```
Training Machine (Remote)                    Env Server (This Machine)
+---------------------------+                +---------------------------+
|  verl / VAGEN training    |                |  FastAPI + AI2-THOR       |
|  +---------------------+ |   HTTP POST    |  +---------------------+ |
|  | GymImageEnvClient   |---/connect-------->| EbAlfredHandler      | |
|  | (per rollout env)   |---/call----------->| (session management) | |
|  +---------------------+ |                |  +---------------------+ |
|                           |                |    GPU 0: X display :0   |
|  rollout workers create   |                |    Unity env instances   |
|  N concurrent clients     |                +---------------------------+
+---------------------------+

Server IP:   localhost (via SSH tunnel) or 100.79.245.124 (Tailscale)
Server Port: 8000
```

## Network

The env server is on a local machine (LAN IP `192.168.0.107`). Cloud training machines
cannot reach this directly. Two options:

### Option A: SSH Reverse Tunnel (recommended for vast.ai)

Run on the **local env server machine** (keep terminal open):
```bash
# One-shot
ssh -p 28663 root@ssh8.vast.ai -R 8000:localhost:8000

# Persistent (auto-reconnect)
autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
    -p 28663 root@ssh8.vast.ai -R 8000:localhost:8000 -N
```

Cloud machine accesses env server via `http://localhost:8000`. No Tailscale needed.

### Option B: Tailscale VPN

Install Tailscale on both machines with the same account:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```
- Env server Tailscale IP: `100.79.245.124`
- Cloud machine accesses env server via `http://100.79.245.124:8000`

### Which `base_urls` to use?

| Method | `base_urls` |
|--------|-------------|
| SSH tunnel | `http://localhost:8000` |
| Tailscale | `http://100.79.245.124:8000` |

All config examples below use `http://localhost:8000` (SSH tunnel).

## Server Status

Health check:
```bash
curl http://localhost:8000/health
# {"ok":true,"service":"gym-env-service","max_inflight":"unlimited"}
```

Active sessions:
```bash
curl http://localhost:8000/sessions
```

### Server Specs
- GPU: 1x NVIDIA RTX 5070 Ti (16GB)
- X Display: :0 (GPU Xorg)
- Max sessions: unlimited (default), adjust with `--max-sessions N` if needed
- Session timeout: 3600s (1 hour idle)

---

## Training Config Setup

### 1. Train Environment Config (train yaml)

Create or modify your train env yaml (e.g. `train_eb_alfred_vision.yaml`):

```yaml
envs:
  - name: RemoteEnv                    # MUST be RemoteEnv (registered client)
    n_envs: 5000                       # total training episodes in dataset
    data_source: eb_alfred
    seed: [1, 5000, 1]
    max_turns: 5
    response_length_per_turn: 512
    config:
      # === Connection config (client-side) ===
      base_urls:
        - "http://localhost:8000"  # <-- via SSH reverse tunnel
      timeout: 600                     # request timeout in seconds (increase for slow tasks)
      retries: 8                       # retry count on failure/503
      backoff: 2.0                     # exponential backoff multiplier

      # === Environment config (passed to server) ===
      eval_set: base
      x_display: "0"                   # ignored by server (server auto-assigns GPU)
      obs_image_size: 500
      max_turns: 5
      max_actions_per_step: 20
      max_env_steps: 30
      action_sep: ","
      prompt_format: free_think
      use_example_in_sys_prompt: true
      format_reward: 0.1
      success_reward: 1.0
```

### 2. Validation Environment Config (val yaml)

Same structure, typically smaller `n_envs` and different seed range:

```yaml
envs:
  - name: RemoteEnv
    n_envs: 32
    data_source: eb_alfred
    seed: [5001, 5050, 1]
    max_turns: 5
    response_length_per_turn: 512
    config:
      base_urls:
        - "http://localhost:8000"
      timeout: 600
      eval_set: base
      x_display: "0"
      obs_image_size: 500
      max_turns: 5
      max_actions_per_step: 20
      max_env_steps: 30
      action_sep: ","
      prompt_format: free_think
      use_example_in_sys_prompt: true
      format_reward: 0.1
      success_reward: 1.0
```

### 3. Training Launch Script

Reference script (adapt paths as needed):

```bash
#!/bin/bash
set -x

PROJECT_NAME="vagen_experiments"
EXPERIMENT_NAME="ppo_eb_alfred_no_concat"

BASEDIR=$(pwd)
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=path/to/train_eb_alfred_vision.yaml
DATASET_VAL=path/to/val_eb_alfred_vision.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent_no_concat.yaml
REF_MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
mkdir -p ${EXPERIMENT_DIR}

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    +data.max_trajectory_length=7000 \
    algorithm.adv_estimator=no_concat_gae_first \
    algorithm.gamma=0.99 \
    algorithm.lam=0.99 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    # ... (other training hyperparams, see train_ppo_no_concat_qwen25vl3b.sh)
```

### 4. Evaluation-Only Config

For standalone evaluation (no training), use the eval config format:

```yaml
envs:
  - name: RemoteEnv
    n_envs: 20
    tag_id: eb_alfred_eval
    seed: [0, 100, 1]
    split: test
    max_turns: 30
    config:
      base_urls:
        - "http://localhost:8000"
      timeout: 300.0
      eval_set: base
      x_display: "0"
      max_turns: 30
      max_actions_per_step: 1
      action_sep: ","
      prompt_format: free_think
      use_example_in_sys_prompt: true
      format_reward: 0.1
      success_reward: 1.0
    chat_config:
      temperature: 0
      max_tokens: 1024
      top_p: 1.0

run:
  backend: "openai"
  max_concurrent_jobs: 3         # limit concurrent env sessions
  resume: skip_completed
```

---

## Batch Size vs Server Capacity

### What happens when batch size > server capacity?

The system has layered backpressure -- it will NOT crash, but may slow down or eventually fail:

```
Layer 1: Server max_sessions limit
   batch request arrives → sessions >= max_sessions?
   ├── No  → create session, proceed
   └── Yes → return HTTP 503

Layer 2: Client retry with exponential backoff
   client receives 503 → retry up to 8 times
   delay = backoff * (2^attempt) * jitter
   attempt 0: ~2s,  attempt 1: ~4s,  attempt 2: ~8s, ...
   total max wait: ~510 seconds

Layer 3: If all retries exhausted → RuntimeError, episode fails
```

### Practical guidance

| Scenario | What happens |
|----------|-------------|
| `train_batch_size=128`, server can handle 128 concurrent | All proceed in parallel, fastest |
| `train_batch_size=128`, server can handle 30 concurrent | First 30 connect, rest get 503 and retry. As episodes finish and close sessions, waiting ones connect. Works but slower. |
| `train_batch_size=128`, server can handle 5 concurrent | Heavy retry pressure. Many episodes may exhaust retries and fail. |

### Recommendations

1. **Match concurrency to server capacity**: The server has 1x RTX 5070 Ti. A reasonable limit is ~20-40 concurrent sessions (depends on scene complexity and VRAM usage).

2. **Set `max_sessions` on server**: If you know the limit, set it explicitly to get clean 503s instead of OOM crashes:
   ```bash
   MAX_SESSIONS=30 bash examples/evaluate/eb_alfred/start_server.sh
   ```

3. **Increase client timeout and retries** if batch size is much larger than capacity:
   ```yaml
   config:
     timeout: 600       # 10 min per request
     retries: 12        # more retries
     backoff: 2.0
   ```

4. **Use multiple env servers**: If one machine isn't enough, start servers on multiple machines and list all URLs:
   ```yaml
   config:
     base_urls:
       - "http://localhost:8000"
       - "http://other-machine:8000"
     failover_after_failures: 4    # switch URL after 4 consecutive failures
   ```

5. **For eval, use `max_concurrent_jobs`** to explicitly limit parallelism:
   ```yaml
   run:
     max_concurrent_jobs: 20    # never more than 20 concurrent episodes
   ```

---

## Monitoring During Training

From the env server machine:

```bash
# Watch active session count
watch -n 5 'curl -s http://localhost:8000/sessions | python3 -m json.tool'

# Check server health
curl http://localhost:8000/health

# Server logs are printed to stdout (the terminal where start_server.sh runs)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Connection refused` | Server not running or firewall | Check server is up, check `ufw`/`iptables` |
| `503 server busy` | Too many concurrent sessions | Reduce `max_concurrent_jobs` or increase `max_sessions` |
| `Timeout` after long wait | AI2-THOR scene load is slow | Increase `timeout` in client config |
| Session unexpectedly closed | Idle too long (>1h default) | Increase `--session-timeout` on server |
| GPU OOM on server | Too many Unity instances | Set `MAX_SESSIONS` to limit concurrent envs |

## Quick Checklist

- [ ] SSH tunnel active: `ssh -p 28663 root@ssh8.vast.ai -R 8000:localhost:8000`
- [ ] Server is running: `curl http://localhost:8000/health` returns `{"ok":true,...}`
- [ ] Network is reachable from cloud: `curl http://localhost:8000/health`
- [ ] Config uses `name: RemoteEnv` (not `EbAlfred`)
- [ ] `base_urls` points to `http://localhost:8000`
- [ ] `timeout` is large enough (>=300 recommended)
- [ ] Concurrency is reasonable for server capacity
