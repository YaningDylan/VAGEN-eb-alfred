#!/bin/bash
# Start the EB-Manipulation environment server.
#
# EB-Manipulation uses CoppeliaSim/PyRep which requires:
#   - COPPELIASIM_ROOT pointing to the CoppeliaSim installation
#   - LD_LIBRARY_PATH and QT_QPA_PLATFORM_PLUGIN_PATH set accordingly
#   - The 'embench' conda environment with EmbodiedBench + PyRep installed
#   - opencv-python-headless (NOT opencv-python) to avoid Qt conflicts
#
# Usage:
#   ./start_server.sh                          # defaults: port=8001
#   PORT=8002 ./start_server.sh                # custom port
#   COPPELIASIM_ROOT=/path/to/CoppeliaSim ./start_server.sh
#
# The server exposes:
#   Health check: http://localhost:8001/health
#   Sessions:     http://localhost:8001/sessions

set -euo pipefail

PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
CONDA_ENV="${CONDA_ENV:-embench}"
SESSION_TIMEOUT="${SESSION_TIMEOUT:-3600}"
MAX_SESSIONS="${MAX_SESSIONS:-1}"           # CoppeliaSim supports only 1 concurrent session

# CoppeliaSim setup - adjust path as needed
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$HOME/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04}"

if [ ! -d "$COPPELIASIM_ROOT" ]; then
  echo "[ERROR] COPPELIASIM_ROOT not found: $COPPELIASIM_ROOT"
  echo "[ERROR] Set COPPELIASIM_ROOT to your CoppeliaSim installation directory."
  exit 1
fi

echo "[EB-Manipulation Server] Starting on ${HOST}:${PORT}"
echo "[EB-Manipulation Server] COPPELIASIM_ROOT=${COPPELIASIM_ROOT}"
echo "[EB-Manipulation Server] conda env=${CONDA_ENV}, max_sessions=${MAX_SESSIONS}"

export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

conda run -n "${CONDA_ENV}" --no-capture-output \
  python -m vagen.envs.eb_manipulation.serve \
    --port "${PORT}" \
    --host "${HOST}" \
    --session-timeout "${SESSION_TIMEOUT}" \
    --max-sessions "${MAX_SESSIONS}" \
    "$@"
