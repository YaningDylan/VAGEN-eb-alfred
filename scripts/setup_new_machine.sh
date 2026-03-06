#!/usr/bin/env bash
#
# One-stop setup script for VAGEN-eb-alfred on a new machine.
#
# What this does:
#   1. Clone VAGEN-eb-alfred and EmbodiedBench repos
#   2. Create the embench-vagen conda env (Python 3.12)
#   3. Install VAGEN (following README) + EmbodiedBench extras
#   4. Install cuda-nvcc 12.8 (for Blackwell/50xx GPU JIT compilation)
#   5. Add OPENAI_API_KEY to ~/.bashrc
#   6. Login to wandb
#
# Usage:
#   bash setup_new_machine.sh [--openai-key <key>] [--wandb-key <key>] [--workspace <dir>]
#
# Requirements:
#   - conda (miniconda/miniforge)
#   - NVIDIA GPU + driver installed
#   - git
#
set -euo pipefail

# ──────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────
OPENAI_KEY="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"
WANDB_KEY="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"
WORKSPACE="$HOME/workspace"
ENV_NAME="embench-vagen"
SKIP_CLONE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openai-key)   OPENAI_KEY="$2"; shift 2 ;;
        --wandb-key)    WANDB_KEY="$2"; shift 2 ;;
        --workspace)    WORKSPACE="$2"; shift 2 ;;
        --env-name)     ENV_NAME="$2"; shift 2 ;;
        --skip-clone)   SKIP_CLONE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--openai-key KEY] [--wandb-key KEY] [--workspace DIR] [--env-name NAME] [--skip-clone]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ──────────────────────────────────────────────────
# Step 0: Check prerequisites
# ──────────────────────────────────────────────────
info "Checking prerequisites..."
command -v conda >/dev/null 2>&1 || error "conda not found. Install miniconda first: https://docs.conda.io/en/latest/miniconda.html"
command -v git   >/dev/null 2>&1 || error "git not found."
command -v nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi not found. GPU driver may not be installed."

mkdir -p "$WORKSPACE"

# ──────────────────────────────────────────────────
# Step 1: Clone repos
# ──────────────────────────────────────────────────
VAGEN_DIR="$WORKSPACE/VAGEN-eb-alfred"
ERA_DIR="$WORKSPACE/Embodied-Reasoning-Agent"

if [ "$SKIP_CLONE" = false ]; then
    info "=== Step 1/6: Cloning repositories ==="

    if [ ! -d "$VAGEN_DIR" ]; then
        info "Cloning VAGEN-eb-alfred..."
        git clone git@github.com:YaningDylan/VAGEN-eb-alfred.git "$VAGEN_DIR"
    else
        warn "VAGEN-eb-alfred already exists at $VAGEN_DIR, skipping clone."
    fi

    cd "$VAGEN_DIR"
    info "Initializing git submodules (verl)..."
    git submodule update --init --recursive

    if [ ! -d "$ERA_DIR" ]; then
        info "Cloning Embodied-Reasoning-Agent (for EmbodiedBench)..."
        git clone https://github.com/Embodied-Reasoning-Agent/Embodied-Reasoning-Agent.git "$ERA_DIR"
    else
        warn "Embodied-Reasoning-Agent already exists at $ERA_DIR, skipping clone."
    fi
else
    info "=== Step 1/6: Skipping clone (--skip-clone) ==="
    [ -d "$VAGEN_DIR" ] || error "VAGEN-eb-alfred not found at $VAGEN_DIR"
    cd "$VAGEN_DIR"
fi

EMBENCH_DIR="$ERA_DIR/eval/EmbodiedBench"
[ -d "$EMBENCH_DIR" ] || error "EmbodiedBench not found at $EMBENCH_DIR"

# ──────────────────────────────────────────────────
# Step 2: Create conda environment
# ──────────────────────────────────────────────────
info "=== Step 2/6: Creating conda environment '$ENV_NAME' ==="

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Conda env '$ENV_NAME' already exists. Removing and recreating..."
    conda deactivate 2>/dev/null || true
    conda env remove -n "$ENV_NAME" -y
fi

conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

info "Python: $(python --version) at $(which python)"

# ──────────────────────────────────────────────────
# Step 3: Install VAGEN (following README)
# ──────────────────────────────────────────────────
info "=== Step 3/6: Installing VAGEN ==="

cd "$VAGEN_DIR/verl"
info "Running verl install script (this may take a while)..."
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

info "Installing verl (editable, no-deps)..."
pip install --no-deps -e .

cd "$VAGEN_DIR"
info "Installing vagen (editable)..."
pip install -e .

info "Installing trl..."
pip install "trl==0.26.2"

# ──────────────────────────────────────────────────
# Step 4: Install EmbodiedBench + extras
# ──────────────────────────────────────────────────
info "=== Step 4/6: Installing EmbodiedBench & extras ==="

# AI2-THOR 2.1.0 for EB-ALFRED
info "Installing ai2thor 2.1.0..."
pip install ai2thor==2.1.0

# EmbodiedBench dependencies
info "Installing EmbodiedBench dependencies..."
pip install h5py vocab revtok matplotlib seaborn scikit-learn fire

# EmbodiedBench itself (editable)
info "Installing embodiedbench (editable)..."
pip install -e "$EMBENCH_DIR"

# ──────────────────────────────────────────────────
# Step 4b: Install CUDA nvcc 12.8 for Blackwell GPU support
# ──────────────────────────────────────────────────
# Blackwell GPUs (RTX 50xx) have compute_120a, which requires
# CUDA 12.8+ nvcc for JIT compilation (e.g., flashinfer kernels).
# The system nvcc might be too old, so we install via conda.
info "Installing cuda-nvcc 12.8 via conda (for Blackwell GPU JIT support)..."
conda install -y -c nvidia cuda-nvcc=12.8

# Verify
NVCC_PATH="$(which nvcc 2>/dev/null || true)"
if [ -n "$NVCC_PATH" ]; then
    NVCC_VER="$(nvcc --version 2>&1 | grep "release" | head -1)"
    info "nvcc installed: $NVCC_VER ($NVCC_PATH)"
else
    warn "nvcc not found in PATH after install."
fi

# ──────────────────────────────────────────────────
# Step 5: Set up API keys
# ──────────────────────────────────────────────────
info "=== Step 5/6: Setting up API keys ==="

# --- OpenAI API Key ---
if [ -z "$OPENAI_KEY" ]; then
    echo ""
    read -rp "Enter your OpenAI API key (or press Enter to skip): " OPENAI_KEY
fi

if [ -n "$OPENAI_KEY" ]; then
    # Remove any existing OPENAI_API_KEY line, then add new one
    sed -i '/^export OPENAI_API_KEY=/d' ~/.bashrc
    echo "export OPENAI_API_KEY=\"$OPENAI_KEY\"" >> ~/.bashrc
    export OPENAI_API_KEY="$OPENAI_KEY"
    info "OPENAI_API_KEY added to ~/.bashrc"
else
    warn "Skipping OpenAI API key setup."
fi

# ──────────────────────────────────────────────────
# Step 6: Login to wandb
# ──────────────────────────────────────────────────
info "=== Step 6/6: Logging in to Weights & Biases ==="

if [ -n "$WANDB_KEY" ]; then
    wandb login "$WANDB_KEY"
    info "wandb login completed with provided key."
else
    echo ""
    info "Running 'wandb login' interactively..."
    wandb login
fi

# ──────────────────────────────────────────────────
# Post-setup: Helpful notes
# ──────────────────────────────────────────────────
CONDA_PREFIX_VAL="$(conda info --base)/envs/$ENV_NAME"
echo ""
echo "=============================================="
info "Setup complete!"
echo "=============================================="
echo ""
echo "Activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "For Blackwell GPUs (RTX 50xx), set these before training:"
echo "  export CUDA_HOME=$CONDA_PREFIX_VAL"
echo "  export PATH=$CONDA_PREFIX_VAL/bin:\$PATH"
echo ""
echo "If you get 'pidfd_getfd: Operation not permitted' errors:"
echo "  echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope"
echo ""
echo "Start EB-ALFRED server:"
echo "  cd $VAGEN_DIR"
echo "  DISPLAY=:0 python -m vagen.envs.eb_alfred.serve --port 8000 --x-displays 0"
echo ""
echo "Run FrozenLake training test:"
echo "  cd $VAGEN_DIR"
echo "  bash examples/frozenlake/train_grpo_qwen25vl3b_nofilter_text.sh"
echo ""
echo "Source bashrc to load API keys in current shell:"
echo "  source ~/.bashrc"
echo ""
