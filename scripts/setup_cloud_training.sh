#!/bin/bash
# ============================================================================
# Cloud Training Machine Setup Script
#
# Sets up a fresh cloud machine for VAGEN EB-ALFRED remote training.
# The environment (AI2-THOR) runs on a separate env server -- this machine
# only needs VAGEN + verl.
#
# Network: env server is accessed via SSH reverse tunnel (localhost:8000).
# Start the tunnel from the env server machine BEFORE running training:
#   ssh -p 28663 root@ssh8.vast.ai -R 8000:localhost:8000
#
# Usage:
#   bash setup_cloud_training.sh
#
# Prerequisites:
#   - Ubuntu with NVIDIA GPU + CUDA drivers installed
#   - conda available
# ============================================================================

set -euo pipefail

echo "============================================"
echo "  VAGEN Cloud Training Setup"
echo "============================================"

# -----------------------------------------------------------
# 1. GitHub login (for private repo access)
# -----------------------------------------------------------
echo ""
echo "[1/4] GitHub CLI login"
if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
    echo "  Already logged in."
else
    if ! command -v gh &>/dev/null; then
        echo "  Installing GitHub CLI..."
        (type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
        && sudo mkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) && wget -nv -O"$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && cat "$out" | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
        && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
        && sudo apt update \
        && sudo apt install gh -y
    fi
    echo "  Please log in to GitHub:"
    gh auth login
fi

# -----------------------------------------------------------
# 2. W&B login
# -----------------------------------------------------------
echo ""
echo "[2/4] Weights & Biases login"
if python3 -c "import wandb; wandb.api.api_key" &>/dev/null 2>&1; then
    echo "  Already logged in."
else
    echo "  Please log in to W&B:"
    pip install wandb -q 2>/dev/null || true
    wandb login
fi

# -----------------------------------------------------------
# 3. Install VAGEN (from README)
# -----------------------------------------------------------
echo ""
echo "[3/4] Installing VAGEN"

VAGEN_DIR="${VAGEN_DIR:-$(pwd)/VAGEN}"
VAGEN_REPO="${VAGEN_REPO:-https://github.com/mll-lab-nu/VAGEN.git}"
VAGEN_BRANCH="${VAGEN_BRANCH:-main}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vagen}"

# Create conda env if not exists
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Creating conda environment '${CONDA_ENV_NAME}'..."
    conda create -n "${CONDA_ENV_NAME}" python=3.12 -y
else
    echo "  Conda environment '${CONDA_ENV_NAME}' already exists."
fi

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# Clone repo if not exists
if [ ! -d "${VAGEN_DIR}" ]; then
    echo "  Cloning VAGEN..."
    git clone -b "${VAGEN_BRANCH}" "${VAGEN_REPO}" "${VAGEN_DIR}"
    cd "${VAGEN_DIR}"
    git submodule update --init --recursive
else
    echo "  VAGEN directory already exists at ${VAGEN_DIR}"
    cd "${VAGEN_DIR}"
fi

# Install verl
echo "  Installing verl..."
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..

# Install vagen
echo "  Installing vagen..."
pip install -e .
pip install "trl==0.26.2"

echo "  VAGEN installed successfully."

# -----------------------------------------------------------
# 4. Verify connectivity to env server
# -----------------------------------------------------------
echo ""
echo "[4/4] Verifying connection to env server"

ENV_SERVER_URL="${ENV_SERVER_URL:-http://localhost:8000}"

echo "  Testing ${ENV_SERVER_URL}/health ..."
if curl -sf "${ENV_SERVER_URL}/health" --connect-timeout 10 >/dev/null 2>&1; then
    echo "  Env server is reachable!"
    curl -s "${ENV_SERVER_URL}/health" 2>/dev/null || true
else
    echo "  WARNING: Cannot reach env server at ${ENV_SERVER_URL}"
    echo "  Make sure the SSH reverse tunnel is active from the env server machine:"
    echo "    ssh -p 28663 root@ssh8.vast.ai -R 8000:localhost:8000"
fi

# -----------------------------------------------------------
# Done
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  VAGEN directory:    ${VAGEN_DIR}"
echo "  Conda environment:  ${CONDA_ENV_NAME}"
echo "  Env server:         ${ENV_SERVER_URL}"
echo ""
echo "  Next steps:"
echo "    1. Start SSH tunnel from env server machine:"
echo "       ssh -p 28663 root@ssh8.vast.ai -R 8000:localhost:8000"
echo "    2. conda activate ${CONDA_ENV_NAME}"
echo "    3. cd ${VAGEN_DIR}"
echo "    4. Run training:"
echo "       bash examples/eb_alfred/train_ppo_no_concat_qwen25vl3b.sh"
echo ""
echo "  See docs/remote_env_server_guide.md for full details."
