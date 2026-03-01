#!/bin/bash
# Run EB-ALFRED evaluation with VAGEN.
#
# Prerequisites:
#   1. Start the EB-ALFRED env server first: ./start_server.sh
#   2. Set up API key env var for your chosen backend, e.g.:
#        export OPENAI_API_KEY="sk-..."
#        export ANTHROPIC_API_KEY="sk-ant-..."
#
# Usage:
#   ./run_eval.sh                                    # default config
#   ./run_eval.sh custom_config.yaml                 # custom config
#   ./run_eval.sh config.yaml run.backend=claude      # override backend
#   ./run_eval.sh config.yaml envs[0].n_envs=5       # override n_envs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null || true

cd "$SCRIPT_DIR/../../.."
python -m vagen.evaluate.run_eval --config "$CONFIG" "$@"
