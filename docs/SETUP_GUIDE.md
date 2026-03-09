# EB-ALFRED Environment Server Setup

This guide covers how to set up the EB-ALFRED environment on a dedicated GPU machine and connect it to a separate VAGEN training/evaluation machine.

## Architecture

EB-ALFRED requires a two-machine setup:

```
┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│  Training Machine (Cloud)       │         │  Env Server Machine (GPU)        │
│                                 │  HTTP   │                                  │
│  VAGEN / verl training          │◄───────►│  FastAPI + AI2-THOR (Unity 3D)   │
│  or standalone evaluation       │         │                                  │
│                                 │         │  Requirements:                   │
│  Requirements:                  │         │  - NVIDIA GPU (40-series or A100)│
│  - Python 3.10+                 │         │  - X server (Xvfb or real)       │
│  - vagen package                │         │  - conda env: embench            │
│  - No GPU needed for env        │         │  - ai2thor 2.1.0                 │
│                                 │         │                                  │
│  Config points to env server:   │         │  Runs:                           │
│  base_urls:                     │         │  python -m vagen.envs.eb_alfred  │
│    - "http://localhost:8000"    │         │    .serve --port 8000            │
└─────────────────────────────────┘         └──────────────────────────────────┘
```

Why two machines? AI2-THOR uses a 2019 Unity build that requires a specific GPU + X server setup. The training stack (verl/vllm/sglang) has conflicting dependencies. Separating them avoids all of this.

---

## Part 1: Env Server Machine Setup

### Prerequisites

- Ubuntu Linux
- NVIDIA GPU: **40-series** (4060/4070/4080/4090) or **A100/H100**
  - 50-series (5070 Ti etc.) is **NOT compatible** with ai2thor 2.1.0
- NVIDIA driver installed (`nvidia-smi` works)
- Conda (Miniconda or Anaconda)

### 1.1 Clone the repository

```bash
git clone git@github.com:YaningDylan/VAGEN-eb-alfred.git
cd VAGEN-eb-alfred
```

### 1.2 Create the `embench` conda environment

```bash
conda create -n embench python=3.9 -y
conda activate embench

# Install EmbodiedBench (includes ai2thor)
pip install embodiedbench

# Pin ai2thor to 2.1.0 (required by EmbodiedBench)
pip install ai2thor==2.1.0

# Install VAGEN (for server code)
pip install -e .

# Verify
python -c "import ai2thor; print(ai2thor.__version__)"
# Should print: 2.1.0
```

### 1.3 Download EB-ALFRED dataset

The task data (expert demonstrations + language annotations) is hosted on HuggingFace:

```bash
# Install git-lfs (needed for large files)
sudo apt-get install git-lfs
git lfs install

# Download dataset into the EmbodiedBench data directory
EMBENCH_DATA=$(python -c "import embodiedbench, pathlib; print(pathlib.Path(embodiedbench.__file__).parent / 'envs/eb_alfred/data')")
echo "EmbodiedBench data dir: $EMBENCH_DATA"

cd "$EMBENCH_DATA"
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED json_2.1.0

# Verify
ls json_2.1.0/
# Should contain: valid_seen/  valid_unseen/  tests_seen/  tests_unseen/  ...
```

If `embodiedbench` is not pip-installed, you can also clone the [Embodied-Reasoning-Agent](https://github.com/Embodied-Reasoning-Agent/Embodied-Reasoning-Agent) repo and find the data dir at `eval/EmbodiedBench/embodiedbench/envs/eb_alfred/data/`.

### 1.4 Patch ai2thor (CRITICAL)

ai2thor 2.1.0 has a bug where werkzeug sends `Connection: close` headers, causing Unity SocketException crashes after a few steps.

```bash
python scripts/patch_ai2thor.py

# Verify
python scripts/patch_ai2thor.py --check
# Expected:
#   OK: run_wsgi override with keep-alive
#   OK: threaded server mode
#   OK: shutdown_request bypass
```

### 1.5 Set up X server

AI2-THOR needs an X display for GPU rendering.

```bash
# Option A: Real display (desktop machine) — usually :0 is already available

# Option B: Xvfb (headless server)
sudo apt-get install -y xvfb
Xvfb :0 -screen 0 1024x768x24 &
export DISPLAY=:0
```

Verify: `DISPLAY=:0 glxinfo | head -5`

### 1.6 Quick test: verify Unity starts

```bash
DISPLAY=:0 conda run -n embench python -c "
from ai2thor.controller import Controller
c = Controller(scene='FloorPlan1', gridSize=0.25)
for i in range(3):
    e = c.step('RotateRight')
    print(f'Step {i+1}: success={e.metadata[\"lastActionSuccess\"]}')
c.stop()
print('Unity OK')
"
```

### 1.7 Start the environment server

```bash
DISPLAY=:0 conda run -n embench python -m vagen.envs.eb_alfred.serve \
    --port 8000 \
    --capacity 16 \
    --x-displays 0
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | 8000 | Server port |
| `--capacity` | 16 | Max concurrent Unity environments. Extra requests are queued (not rejected). |
| `--x-displays` | auto | Comma-separated GPU display IDs (e.g. `0,1`). Auto-detects GPUs if omitted. |
| `--max-sessions` | 0 | Max total sessions (0 = unlimited). Sessions beyond this get HTTP 503. |
| `--thread-workers` | 128 | Thread pool size for Unity instance creation. |

Verify:
```bash
curl http://localhost:8000/health
# {"ok":true,"service":"gym-env-service","max_inflight":"unlimited"}

curl http://localhost:8000/sessions
# {"num_sessions":0,"max_sessions":0,...}
```

### Capacity guide (per GPU)

| GPU | VRAM | Recommended `--capacity` |
|-----|------|--------------------------|
| RTX 4060 (8 GB) | 8 GB | 8-16 |
| RTX 4070 (12 GB) | 12 GB | 16-32 |
| RTX 4090 (24 GB) | 24 GB | 32-64 |
| A100 (80 GB) | 80 GB | 64-128 |

Each Unity instance uses ~148 MB VRAM + ~750 MB RAM. The `--capacity` flag controls how many run concurrently; any number of sessions can be queued beyond this limit.

---

## Part 2: Network — Connecting the Two Machines

The training machine (cloud VM) needs to reach the env server (local GPU machine) via HTTP. Use an SSH reverse tunnel:

Run on the **env server machine** (keeps the tunnel open):

```bash
# One-shot
ssh -p <cloud-ssh-port> root@<cloud-host> -R 8000:localhost:8000

# Persistent (auto-reconnect)
autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
    -p <cloud-ssh-port> root@<cloud-host> -R 8000:localhost:8000 -N
```

This forwards the cloud machine's `localhost:8000` back to the env server's `localhost:8000`. The training machine then uses:

```yaml
base_urls:
  - "http://localhost:8000"
```

Verify connectivity (from the **training machine**):

```bash
curl http://localhost:8000/health
# {"ok":true,"service":"gym-env-service","max_inflight":"unlimited"}
```

---

## Part 3: Training Machine Configuration

The training machine only needs the `vagen` package (no ai2thor, no embench). In your YAML config, set `name: RemoteEnv` and point `base_urls` to the env server:

```yaml
config:
  base_urls:
    - "http://localhost:8000"   # via SSH tunnel
```

For a complete config example, see [`examples/evaluate/eb_alfred/config.yaml`](../examples/evaluate/eb_alfred/config.yaml).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` from training machine | Check server is running (`curl health`), check SSH tunnel is active |
| `503 server busy` | Reduce `max_concurrent_jobs` or increase `--max-sessions` on server |
| Timeout on reset | Increase `timeout` in client config (queued sessions need longer) |
| Unity `SocketException` | Run `python scripts/patch_ai2thor.py` on the env server |
| `Cannot open display` | Set `DISPLAY=:0` before starting server; verify X: `ls /tmp/.X11-unix/` |
| GPU OOM | Reduce `--capacity` on server |
| Orphaned Unity processes | `pkill -9 -f 'thor-201909061227'` |
| 50-series GPU hangs on reset | Use a 40-series GPU instead; ai2thor 2.1.0 is incompatible with Blackwell |

## Monitoring

```bash
# Watch active sessions
watch -n 5 'curl -s http://localhost:8000/sessions | python3 -m json.tool'

# GPU usage
watch -n 2 nvidia-smi

# Server logs are printed to stdout
```
