#!/bin/bash
# Start the EB-ALFRED environment server.
#
# EB-ALFRED uses AI2-THOR which requires:
#   - A running X server (e.g., Xvfb or real display) per GPU
#   - GPU(s) for rendering
#   - The 'embench' conda environment with EmbodiedBench installed
#
# Multi-GPU is the default: GPUs are auto-detected via nvidia-smi
# and sessions are distributed to the least-loaded GPU.
#
# Usage:
#   ./start_server.sh                           # auto-detect GPUs
#   X_DISPLAYS=0,1 ./start_server.sh            # explicit GPU list
#   PORT=8001 ./start_server.sh                 # custom port
#
# The server exposes:
#   Health check: http://localhost:8000/health
#   Sessions:     http://localhost:8000/sessions

set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CONDA_ENV="${CONDA_ENV:-embench}"
SESSION_TIMEOUT="${SESSION_TIMEOUT:-3600}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"

# X_DISPLAYS: comma-separated display IDs (e.g. "0,1,2")
# Empty = auto-detect GPUs
X_DISPLAYS="${X_DISPLAYS:-}"

echo "[EB-ALFRED Server] Starting on ${HOST}:${PORT}"
echo "[EB-ALFRED Server] conda env=${CONDA_ENV}"
echo "[EB-ALFRED Server] max_sessions=${MAX_SESSIONS}, session_timeout=${SESSION_TIMEOUT}s"

EXTRA_ARGS=()
if [ -n "${X_DISPLAYS}" ]; then
  echo "[EB-ALFRED Server] X_DISPLAYS=${X_DISPLAYS} (manual override)"
  EXTRA_ARGS+=(--x-displays "${X_DISPLAYS}")
else
  echo "[EB-ALFRED Server] GPU auto-detection enabled"
fi

conda run -n "${CONDA_ENV}" --no-capture-output \
  python -m vagen.envs.eb_alfred.serve \
    --port "${PORT}" \
    --host "${HOST}" \
    --session-timeout "${SESSION_TIMEOUT}" \
    --max-sessions "${MAX_SESSIONS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
