#!/bin/bash
# Run EB-ALFRED benchmark on a dual-GPU machine (e.g., 2× RTX 4090).
#
# This script starts the env server and runs evaluation in one go.
#
# Prerequisites:
#   - Both conda envs set up: 'embench' (server) and 'vagen' (eval client)
#   - ai2thor patched: conda run -n embench python scripts/patch_ai2thor.py
#   - X server running (Xvfb or real display) on DISPLAY :0
#   - OPENAI_API_KEY set in environment
#
# Usage:
#   cd VAGEN-eb-alfred
#   export OPENAI_API_KEY="sk-..."
#   bash examples/evaluate/eb_alfred/run_benchmark.sh
#
# The script auto-detects GPUs and sets max_sessions accordingly.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PROJ_ROOT="$(pwd)"

# ── Configuration ───────────────────────────────────────────────
DISPLAY="${DISPLAY:-:0}"
PORT="${PORT:-8000}"
CONDA_SERVER="${CONDA_SERVER:-embench}"
CONDA_EVAL="${CONDA_EVAL:-vagen}"
CONFIG="${CONFIG:-examples/evaluate/eb_alfred/dual_gpu_benchmark.yaml}"

# Auto-detect GPU count and compute max_sessions
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
# ~110 sessions per GPU at 500x500 for 24GB, ~70 for 16GB
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MEM}" -ge 20000 ]; then
    SESSIONS_PER_GPU=100
else
    SESSIONS_PER_GPU=60
fi
MAX_SESSIONS=$((NUM_GPUS * SESSIONS_PER_GPU))

echo "============================================================"
echo "  EB-ALFRED Benchmark"
echo "  GPUs: ${NUM_GPUS} (${GPU_MEM} MiB each)"
echo "  Max sessions: ${MAX_SESSIONS}"
echo "  Config: ${CONFIG}"
echo "============================================================"

# ── Step 1: Verify ai2thor patch ────────────────────────────────
echo ""
echo "[1/3] Verifying ai2thor patch..."
conda run -n "${CONDA_SERVER}" python scripts/patch_ai2thor.py --check || {
    echo "  Applying patch..."
    conda run -n "${CONDA_SERVER}" python scripts/patch_ai2thor.py
}

# ── Step 2: Start server (background) ──────────────────────────
echo ""
echo "[2/3] Starting EB-ALFRED server on port ${PORT}..."
DISPLAY="${DISPLAY}" MAX_SESSIONS="${MAX_SESSIONS}" PORT="${PORT}" \
    conda run -n "${CONDA_SERVER}" --no-capture-output \
    python -m vagen.envs.eb_alfred.serve \
        --port "${PORT}" --max-sessions "${MAX_SESSIONS}" \
    > /tmp/eb_alfred_server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 30); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" 2>/dev/null; then
        echo "  Server ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ERROR: Server failed to start. Check /tmp/eb_alfred_server.log"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

# ── Step 3: Run evaluation ──────────────────────────────────────
echo ""
echo "[3/3] Running evaluation..."
conda run -n "${CONDA_EVAL}" python -m vagen.evaluate.run_eval --config "${CONFIG}"
EXIT_CODE=$?

# ── Cleanup ─────────────────────────────────────────────────────
echo ""
echo "Stopping server (PID ${SERVER_PID})..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# Generate report if aggregate script exists
if [ -f "benchmarking/aggregate_results.py" ]; then
    echo "Generating report..."
    conda run -n "${CONDA_EVAL}" python benchmarking/aggregate_results.py
fi

echo "Done. Exit code: ${EXIT_CODE}"
exit $EXIT_CODE
