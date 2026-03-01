# EB-ALFRED Setup Guide

Complete guide to set up and run EB-ALFRED evaluation on a new machine.
Tested on Ubuntu with NVIDIA GPUs (RTX 4090, RTX 5070 Ti, A100).

---

## Prerequisites

- Ubuntu Linux with NVIDIA GPU(s)
- NVIDIA driver installed (`nvidia-smi` works)
- Conda (Miniconda or Anaconda)
- X server running (real display or Xvfb)
- Git

---

## Step 1: Clone the Repository

```bash
git clone git@github.com:YaningDylan/VAGEN-eb-alfred.git
cd VAGEN-eb-alfred
```

---

## Step 2: Create Conda Environments

Two separate environments are needed:
- **embench**: Runs the AI2-THOR environment server (Python 3.9, ai2thor 2.1.0)
- **vagen**: Runs the evaluation client (Python 3.10+)

### 2a. Create the `embench` environment (server)

```bash
conda create -n embench python=3.9 -y
conda activate embench

# Install EmbodiedBench (which installs ai2thor 2.1.0)
pip install embodiedbench

# CRITICAL: Pin ai2thor to 2.1.0 (EmbodiedBench requires this version)
pip install ai2thor==2.1.0

# Verify
python -c "import ai2thor; print(ai2thor.__version__)"
# Should print: 2.1.0
```

### 2b. Create the `vagen` environment (eval client)

```bash
conda create -n vagen python=3.10 -y
conda activate vagen

# Install from project root
pip install -e .
```

---

## Step 3: Patch ai2thor (CRITICAL)

ai2thor 2.1.0 has a bug where werkzeug sends `Connection: close` headers,
causing Unity's Mono runtime to throw `SocketException` after a few steps.
This patch MUST be applied before running any experiments.

```bash
conda run -n embench python scripts/patch_ai2thor.py
```

To verify the patch was applied:
```bash
conda run -n embench python scripts/patch_ai2thor.py --check
```

Expected output:
```
  OK: run_wsgi override with keep-alive
  OK: threaded server mode
  OK: shutdown_request bypass
```

### What the patch does

The patch modifies `ai2thor/server.py` in the embench conda environment:

1. **run_wsgi override**: Replaces werkzeug's default `run_wsgi()` method in
   `ThorRequestHandler` to send `Connection: keep-alive` instead of
   `Connection: close`, and skips the drain loop that would consume the next
   request's bytes.

2. **threaded=True**: Forces `werkzeug.serving.make_server()` to use threaded
   mode, so Reset (which creates a new HTTP connection) doesn't deadlock.

3. **shutdown_request bypass**: Patches `shutdown_request` to skip
   `socket.shutdown(SHUT_WR)` which sends TCP FIN and triggers Unity's
   SocketException.

### Quick test after patching

```bash
DISPLAY=:0 conda run -n embench python -c "
from ai2thor.controller import Controller
c = Controller(scene='FloorPlan1', gridSize=0.25)
for i in range(5):
    e = c.step('RotateRight')
    print(f'Step {i+1}: success={e.metadata[\"lastActionSuccess\"]}')
c.step('Reset')
e = c.step('RotateRight')
print(f'After reset: success={e.metadata[\"lastActionSuccess\"]}')
c.stop()
print('All OK')
"
```

---

## Step 4: Set Up X Server

AI2-THOR requires an X display for GPU rendering. If running on a headless
server, use Xvfb:

```bash
# Option A: Real display (desktop machine)
# Usually DISPLAY=:0 is already set

# Option B: Xvfb (headless server)
sudo apt-get install xvfb
Xvfb :0 -screen 0 1024x768x24 &
export DISPLAY=:0
```

Verify X is working:
```bash
DISPLAY=:0 glxinfo | head -5
```

---

## Step 5: Start the Environment Server

```bash
cd VAGEN-eb-alfred

# Basic start (auto-detect GPUs):
DISPLAY=:0 conda run -n embench python -m vagen.envs.eb_alfred.serve \
    --port 8000 --max-sessions 64

# Or use the provided script:
DISPLAY=:0 MAX_SESSIONS=200 bash examples/evaluate/eb_alfred/start_server.sh
```

### Max sessions by GPU

| GPU              | VRAM  | Max Sessions (per GPU) |
|:-----------------|:-----:|:----------------------:|
| RTX 4060 (8 GB)  | 8 GB  | ~31                    |
| RTX 4070 (12 GB) | 12 GB | ~51                    |
| RTX 4080 (16 GB) | 16 GB | ~70                    |
| RTX 4090 (24 GB) | 24 GB | ~110                   |
| A100 (80 GB)     | 80 GB | ~384                   |

For multi-GPU, multiply by N. The server auto-balances across GPUs.

Each Unity instance also uses ~750 MiB RAM. Check that total RAM is sufficient:
`floor((total_ram_GB - 4) / 0.75)` = max instances by RAM.

Verify server is running:
```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

---

## Step 6: Run Evaluation

### Quick test (3 episodes)

```bash
conda run -n vagen python -m vagen.evaluate.run_eval \
    --config examples/evaluate/eb_alfred/config.yaml \
    envs[0].n_envs=3
```

### Full benchmark (128 episodes, dual GPU)

```bash
export OPENAI_API_KEY="sk-..."
bash examples/evaluate/eb_alfred/run_benchmark.sh
```

Or use individual configs:
```bash
conda run -n vagen python -m vagen.evaluate.run_eval \
    --config benchmarking/configs/parallel_500_128ep.yaml
```

### Generate markdown report

After experiments complete:
```bash
conda run -n vagen python benchmarking/aggregate_results.py
# Report at: benchmarking/BENCHMARK_REPORT.md
```

---

## Key Configuration Options

### concat_history (in eval YAML)

Controls whether to send full conversation history to the LLM on each turn:

```yaml
envs:
  - name: RemoteEnv
    concat_history: false    # non-concat: only system + current obs (cheaper)
    # concat_history: true   # concat: full history (more expensive, ~8.5x more tokens)
```

- **non-concat** (`false`): Each API call sends only system prompt + current
  observation. ~25K tokens per 30-turn episode at 500x500. Use this for
  training and benchmarks.
- **concat** (`true`, default): Full conversation history accumulates. ~210K
  tokens per 30-turn episode. Use for evaluation where history context matters.

### Concurrency settings

```yaml
run:
  max_concurrent_jobs: 64      # How many episodes run in parallel
                                # Limited by GPU/RAM capacity

backends:
  openai:
    max_concurrency: 64         # How many API calls at the same time
                                # Limited by API rate limits
```

Rule of thumb: set both to the number of Unity instances your hardware supports.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Eval Client (vagen env)                                │
│                                                         │
│  run_eval.py → runner.py → vision_workflow.py           │
│       │              │              │                    │
│       │         episode_gate    adapter.acompletion()    │
│       │        (semaphore)      (GPT-4.1 API calls)     │
│       │              │              │                    │
│       ▼              ▼              │                    │
│  GymImageEnvClient ────── HTTP ─────┼──┐                │
└──────────────────────────────────────┼──┘                │
                                       │                   │
┌──────────────────────────────────────┼──┐                │
│  Env Server (embench env)            │  │                │
│                                      │  │                │
│  serve.py → handler.py               │  │                │
│       │          │                   │  │                │
│       │    least-loaded GPU          │  │                │
│       │    assignment                │  │                │
│       ▼          ▼                   │  │                │
│  Unity/AI2-THOR instances            │  │                │
│  (one per session, ~148MB VRAM,     │  │                │
│   ~750MB RAM each)                   │  │                │
└──────────────────────────────────────┘  │
                                          │
┌─────────────────────────────────────────┘
│  OpenAI API (GPT-4.1)
│  (rate limited by max_concurrency)
└─────────────────────────────────────────
```

---

## Troubleshooting

### SocketException in Unity
**Symptom**: `System.Net.Sockets.SocketException: Connection reset by peer`
**Fix**: Run `conda run -n embench python scripts/patch_ai2thor.py`

### X display not found
**Symptom**: `Cannot open display ":1"` or similar
**Fix**: Set `DISPLAY=:0` before starting server. Check X is running: `ls /tmp/.X11-unix/`

### ai2thor version mismatch
**Symptom**: Missing `server.py` or different API
**Fix**: `conda run -n embench pip install ai2thor==2.1.0`
Note: ai2thor 5.0.0 is incompatible with EmbodiedBench. Must use 2.1.0.

### Server port already in use
**Fix**: `lsof -i :8000` to find the process, then `kill <PID>`

### Out of GPU memory
**Symptom**: Unity instances crash or fail to start
**Fix**: Reduce `--max-sessions`. Check current usage: `nvidia-smi`

### Orphaned Unity processes
**Fix**: `pkill -f thor-CloudRendering` or `pkill -f ai2thor`

---

## File Reference

| File | Purpose |
|:-----|:--------|
| `scripts/patch_ai2thor.py` | Patches ai2thor 2.1.0 SocketException bug |
| `vagen/envs/eb_alfred/serve.py` | Environment server entry point |
| `vagen/envs/eb_alfred/handler.py` | Multi-GPU session management |
| `vagen/evaluate/run_eval.py` | Evaluation entry point |
| `vagen/evaluate/vision_workflow.py` | Episode runner (concat/non-concat) |
| `vagen/evaluate/runner.py` | Parallel job orchestration |
| `examples/evaluate/eb_alfred/` | Example configs and run scripts |
| `benchmarking/configs/` | Benchmark experiment configs |
| `benchmarking/aggregate_results.py` | Result aggregation + markdown report |
| `docs/eb_alfred_resource_guide.md` | GPU/RAM capacity tables |
