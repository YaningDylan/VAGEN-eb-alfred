#!/usr/bin/env bash
#
# One-stop setup script for VAGEN-eb-alfred on a new machine.
#
# What this does:
#   1. Clone VAGEN-eb-alfred and EmbodiedBench repos
#   2. Create the embench-vagen conda env (Python 3.12)
#   3. Install VAGEN (following README) + EmbodiedBench extras
#   4. Install cuda-nvcc 12.8 (for Blackwell/50xx GPU JIT compilation)
#   5. Install Xorg + Xvfb for AI2-THOR (Unity) rendering
#   6. Add OPENAI_API_KEY to ~/.bashrc
#   7. Login to wandb
#
# Usage:
#   bash setup.sh [--openai-key KEY] [--wandb-key KEY] [--github-pat PAT]
#                 [--workspace DIR] [--env-name NAME] [--skip-clone] [--start-from N]
#
#   --github-pat PAT  GitHub Personal Access Token for cloning private repos.
#                     Configures HTTPS redirect so SSH keys are not needed.
#   --start-from N    Resume from step N (1-7), skipping earlier steps.
#                     Useful when the script was interrupted mid-way.
#                     Example: interrupted during step 3 → re-run with --start-from 3
#
# Requirements:
#   - conda (miniconda/miniforge) — auto-detected from common install paths
#   - NVIDIA GPU + driver installed
#   - git
#
set -euo pipefail

# ──────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────
OPENAI_KEY="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"
WANDB_KEY="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"
GITHUB_PAT="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"
WORKSPACE="$HOME/workspace"
ENV_NAME="embench-vagen"
SKIP_CLONE=false
START_FROM=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openai-key)   OPENAI_KEY="$2";  shift 2 ;;
        --wandb-key)    WANDB_KEY="$2";   shift 2 ;;
        --github-pat)   GITHUB_PAT="$2";  shift 2 ;;
        --workspace)    WORKSPACE="$2";   shift 2 ;;
        --env-name)     ENV_NAME="$2";    shift 2 ;;
        --skip-clone)   SKIP_CLONE=true;  shift ;;
        --start-from)   START_FROM="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--openai-key KEY] [--wandb-key KEY] [--github-pat PAT]"
            echo "          [--workspace DIR] [--env-name NAME] [--skip-clone] [--start-from N]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────────
# Colors / logging helpers
# ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ──────────────────────────────────────────────────
# Step 0: Auto-detect conda + check prerequisites
# ──────────────────────────────────────────────────
info "Checking prerequisites..."

# Auto-detect conda from common install paths when it is not already in PATH.
if ! command -v conda &>/dev/null; then
    for CONDA_CANDIDATE in \
            "${HOME}/miniconda3/bin/conda"     \
            "${HOME}/anaconda3/bin/conda"      \
            "/opt/miniforge3/bin/conda"        \
            "/opt/miniconda3/bin/conda"        \
            "/usr/local/miniconda3/bin/conda"; do
        if [ -x "$CONDA_CANDIDATE" ]; then
            export PATH="$(dirname "$CONDA_CANDIDATE"):$PATH"
            info "Found conda at $CONDA_CANDIDATE"
            break
        fi
    done
fi

command -v conda >/dev/null 2>&1 || \
    error "conda not found. Install miniconda first: https://docs.conda.io/en/latest/miniconda.html"
command -v git   >/dev/null 2>&1 || error "git not found."
command -v nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi not found. GPU driver may not be installed."

# ──────────────────────────────────────────────────
# Always-needed path variables (set before any step so later steps can
# reference them even when early steps are skipped via --start-from).
# ──────────────────────────────────────────────────
mkdir -p "$WORKSPACE"
VAGEN_DIR="$WORKSPACE/VAGEN-eb-alfred"
ERA_DIR="$WORKSPACE/Embodied-Reasoning-Agent"
EMBENCH_DIR="$ERA_DIR/eval/EmbodiedBench"

# Initialize conda shell integration (required for `conda activate` below).
eval "$(conda shell.bash hook)"

# ──────────────────────────────────────────────────
# GitHub authentication (HTTPS + PAT)
# Redirects git@github.com: SSH URLs to HTTPS so no SSH key is needed.
# Runs unconditionally so cloning always works regardless of --start-from.
# ──────────────────────────────────────────────────
if [ -n "${GITHUB_PAT:-}" ]; then
    info "Configuring GitHub HTTPS authentication..."
    git config --global url."https://github.com/".insteadOf "git@github.com:"
    git config --global credential.helper store
    # Write credentials file (overwrite any existing entry for github.com).
    grep -v "github.com" ~/.git-credentials 2>/dev/null > /tmp/git-creds-tmp || true
    echo "https://x-access-token:${GITHUB_PAT}@github.com" >> /tmp/git-creds-tmp
    mv /tmp/git-creds-tmp ~/.git-credentials
    chmod 600 ~/.git-credentials
    info "GitHub PAT configured."
else
    warn "No GitHub PAT provided; SSH clone will be attempted (may fail on private repos)."
fi

# ──────────────────────────────────────────────────
# Step 1: Clone repos
# ──────────────────────────────────────────────────
if (( START_FROM <= 1 )); then
    info "=== Step 1/7: Cloning repositories ==="

    if [ "$SKIP_CLONE" = false ]; then
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
        info "Skipping clone (--skip-clone)."
        [ -d "$VAGEN_DIR" ] || error "VAGEN-eb-alfred not found at $VAGEN_DIR"
    fi
else
    info "=== Step 1/7: Skipped (--start-from $START_FROM) ==="
fi

[ -d "$VAGEN_DIR" ]   || error "VAGEN-eb-alfred not found at $VAGEN_DIR"
[ -d "$EMBENCH_DIR" ] || error "EmbodiedBench not found at $EMBENCH_DIR"

# ──────────────────────────────────────────────────
# Step 2: Create conda environment
# ──────────────────────────────────────────────────
if (( START_FROM <= 2 )); then
    info "=== Step 2/7: Creating conda environment '$ENV_NAME' ==="

    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Conda env '$ENV_NAME' already exists. Removing and recreating..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
    fi

    conda create -n "$ENV_NAME" python=3.12 -y
else
    info "=== Step 2/7: Skipped (--start-from $START_FROM) ==="
fi

# Activate the environment (needed for all subsequent steps).
conda activate "$ENV_NAME"
info "Python: $(python --version) at $(which python)"

# ──────────────────────────────────────────────────
# Step 3: Install VAGEN (following README)
# ──────────────────────────────────────────────────
if (( START_FROM <= 3 )); then
    info "=== Step 3/7: Installing VAGEN ==="

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
else
    info "=== Step 3/7: Skipped (--start-from $START_FROM) ==="
fi

# ──────────────────────────────────────────────────
# Step 4: Install EmbodiedBench + extras
# ──────────────────────────────────────────────────
if (( START_FROM <= 4 )); then
    info "=== Step 4/7: Installing EmbodiedBench & extras ==="

    # AI2-THOR 2.1.0 for EB-ALFRED
    info "Installing ai2thor 2.1.0..."
    pip install ai2thor==2.1.0

    # EmbodiedBench dependencies
    info "Installing EmbodiedBench dependencies..."
    pip install h5py vocab revtok matplotlib seaborn scikit-learn fire

    # EmbodiedBench itself (editable)
    info "Installing embodiedbench (editable)..."
    pip install -e "$EMBENCH_DIR"

    # cuda-nvcc 12.8 — required for Blackwell GPU (RTX 50xx) JIT compilation
    info "Installing cuda-nvcc 12.8 via conda (for Blackwell GPU JIT support)..."
    conda install -y -c nvidia cuda-nvcc=12.8

    NVCC_PATH="$(which nvcc 2>/dev/null || true)"
    if [ -n "$NVCC_PATH" ]; then
        NVCC_VER="$(nvcc --version 2>&1 | grep "release" | head -1)"
        info "nvcc installed: $NVCC_VER ($NVCC_PATH)"
    else
        warn "nvcc not found in PATH after install."
    fi
else
    info "=== Step 4/7: Skipped (--start-from $START_FROM) ==="
fi

# ──────────────────────────────────────────────────
# Step 5: Install Xorg + Xvfb for AI2-THOR (Unity)
# ──────────────────────────────────────────────────
if (( START_FROM <= 5 )); then
    info "=== Step 5/7: Installing X server (Xorg + Xvfb) for AI2-THOR ==="

    apt-get install -y xserver-xorg-core xvfb

    # Install the NVIDIA Xorg driver that matches the running kernel module.
    # The kernel module version must match nvidia_drv.so exactly.
    # We pin to the exact version reported by /proc/driver/nvidia/version.
    NVRM_VER=$(awk '/NVRM version:/{print $8}' /proc/driver/nvidia/version 2>/dev/null || true)
    if [ -n "$NVRM_VER" ]; then
        info "Detected NVIDIA kernel module version: $NVRM_VER"
        # Derive the major release branch (e.g. 560.35.03 → 560)
        NVRM_BRANCH="${NVRM_VER%%.*}"
        DRV_PKG="xserver-xorg-video-nvidia-${NVRM_BRANCH}"
        DRV_VER="${NVRM_VER}-0ubuntu1"
        info "Installing ${DRV_PKG}=${DRV_VER} ..."
        # Remove any already-installed version to avoid cross-version conflict.
        apt-get remove -y "${DRV_PKG}" 2>/dev/null || true
        apt-get install -y --allow-downgrades "${DRV_PKG}=${DRV_VER}" || {
            warn "Could not install exact version ${DRV_VER}; trying latest ${DRV_PKG}..."
            apt-get install -y --allow-downgrades "${DRV_PKG}" || true
        }

        # If nvidia_drv.so version still doesn't match the kernel module,
        # extract the correct binary from the versioned .deb directly.
        INSTALLED_DRV_VER=$(strings /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so \
            2>/dev/null | grep "^${NVRM_BRANCH}\." | head -1 || true)
        if [ "${INSTALLED_DRV_VER}" != "${NVRM_VER}" ]; then
            warn "nvidia_drv.so version (${INSTALLED_DRV_VER}) != kernel module (${NVRM_VER}). Extracting from deb..."
            TMP_DRV_DIR=$(mktemp -d)
            apt-get download "${DRV_PKG}=${DRV_VER}" -o Dir::Cache::archives="$TMP_DRV_DIR" 2>/dev/null || \
                apt-get download "${DRV_PKG}" -o Dir::Cache::archives="$TMP_DRV_DIR" 2>/dev/null || true
            DEB_FILE=$(ls "${TMP_DRV_DIR}"/*.deb 2>/dev/null | head -1)
            if [ -n "$DEB_FILE" ]; then
                dpkg-deb -x "$DEB_FILE" "${TMP_DRV_DIR}/extracted/"
                EXTRACTED_DRV="${TMP_DRV_DIR}/extracted/usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so"
                if [ -f "$EXTRACTED_DRV" ]; then
                    cp "$EXTRACTED_DRV" /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so
                    info "Replaced nvidia_drv.so with version $(strings "$EXTRACTED_DRV" | grep "^${NVRM_BRANCH}\." | head -1)"
                fi
            fi
            rm -rf "$TMP_DRV_DIR"
        fi

        # Generate xorg.conf for each GPU (one conf per GPU, Display = GPU index + 2).
        # Display offset of 2 avoids conflicts with :0 and :1 commonly used by system.
        DISPLAY_OFFSET=2
        GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
        info "Generating xorg.conf for ${GPU_COUNT} GPU(s)..."
        GPU_IDX=0
        while IFS=',' read -r idx pci_id; do
            idx=$(echo "$idx" | tr -d ' ')
            pci_id=$(echo "$pci_id" | tr -d ' ')
            # Convert PCI domain:bus:slot.fn (e.g. 00000000:81:00.0) to Xorg BusID (PCI:129:0:0)
            BUS_HEX=$(echo "$pci_id" | awk -F: '{print $2}')
            BUS_DEC=$(( 16#${BUS_HEX} ))
            SLOT_FN=$(echo "$pci_id" | awk -F: '{print $3}')
            SLOT=$(echo "$SLOT_FN" | cut -d. -f1)
            FN=$(echo "$SLOT_FN" | cut -d. -f2)
            DISP=$(( DISPLAY_OFFSET + GPU_IDX ))
            CONF="/tmp/xorg_gpu${idx}.conf"
            cat > "$CONF" << XORGEOF
Section "Device"
    Identifier "Device${idx}"
    Driver "nvidia"
    BusID "PCI:${BUS_DEC}:$(( 16#${SLOT} )):${FN}"
EndSection
Section "Screen"
    Identifier "Screen${idx}"
    Device "Device${idx}"
    DefaultDepth 24
    Option "AllowEmptyInitialConfiguration" "True"
    SubSection "Display"
        Depth 24
        Virtual 1024 768
    EndSubSection
EndSection
Section "ServerLayout"
    Identifier "Layout${idx}"
    Screen 0 "Screen${idx}" 0 0
EndSection
XORGEOF
            info "Created ${CONF}  →  Display :${DISP}  (GPU ${idx}, BusID PCI:${BUS_DEC}:$(( 16#${SLOT} )):${FN})"
            (( GPU_IDX++ )) || true
        done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null)
    else
        warn "No NVIDIA kernel module detected; skipping nvidia Xorg driver install."
    fi

    info "X server packages installed."
    info "NOTE: GPU-accelerated Xorg (Plan A) requires CAP_SYS_ADMIN."
    info "      If this container lacks it, use Xvfb (Plan B) instead."
    info "      See README / setup_issues.md Issue 9 for details."
else
    info "=== Step 5/7: Skipped (--start-from $START_FROM) ==="
fi

# ──────────────────────────────────────────────────
# Step 6: Set up API keys
# ──────────────────────────────────────────────────
if (( START_FROM <= 6 )); then
    info "=== Step 6/7: Setting up API keys ==="

    if [ -z "$OPENAI_KEY" ]; then
        echo ""
        read -rp "Enter your OpenAI API key (or press Enter to skip): " OPENAI_KEY
    fi

    if [ -n "$OPENAI_KEY" ]; then
        sed -i '/^export OPENAI_API_KEY=/d' ~/.bashrc
        echo "export OPENAI_API_KEY=\"$OPENAI_KEY\"" >> ~/.bashrc
        export OPENAI_API_KEY="$OPENAI_KEY"
        info "OPENAI_API_KEY added to ~/.bashrc"
    else
        warn "Skipping OpenAI API key setup."
    fi
else
    info "=== Step 6/7: Skipped (--start-from $START_FROM) ==="
fi

# ──────────────────────────────────────────────────
# Step 7: Login to wandb
# ──────────────────────────────────────────────────
if (( START_FROM <= 7 )); then
    info "=== Step 7/7: Logging in to Weights & Biases ==="

    if [ -n "$WANDB_KEY" ]; then
        wandb login "$WANDB_KEY"
        info "wandb login completed with provided key."
    else
        echo ""
        info "Running 'wandb login' interactively..."
        wandb login
    fi
else
    info "=== Step 7/7: Skipped (--start-from $START_FROM) ==="
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
echo "Before starting the EB-ALFRED server, start an X display (once per boot):"
echo ""
echo "  Option A — GPU-accelerated Xorg (fast, requires CAP_SYS_ADMIN capability):"
echo "    GPU 0 → Display :2:"
echo "      Xorg -noreset +extension GLX +extension RANDR +extension RENDER \\"
echo "           -config /tmp/xorg_gpu0.conf :2 &"
echo "    GPU 1 → Display :3:"
echo "      Xorg -noreset +extension GLX +extension RANDR +extension RENDER \\"
echo "           -config /tmp/xorg_gpu1.conf :3 &"
echo "    (xorg.conf files are at /tmp/xorg_gpu*.conf, recreate with --start-from 5)"
echo ""
echo "  Option B — Xvfb software rendering (slower ~7×, no special capabilities needed):"
echo "    Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +extension RANDR &"
echo ""
echo "Start EB-ALFRED server:"
echo "  cd $VAGEN_DIR"
echo "  DISPLAY=:2 python -m vagen.envs.eb_alfred.serve --port 8000 --x-displays 2"
echo "  # (use :1 for Xvfb, adjust --x-displays to match)"
echo ""
echo "Run FrozenLake training test:"
echo "  cd $VAGEN_DIR"
echo "  bash examples/frozenlake/train_grpo_qwen25vl3b_nofilter_text.sh"
echo ""
echo "Source bashrc to load API keys in current shell:"
echo "  source ~/.bashrc"
echo ""
