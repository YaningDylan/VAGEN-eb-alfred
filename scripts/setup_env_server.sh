#!/usr/bin/env bash
#
# Lightweight setup script for deploying the EB-ALFRED environment server
# on a new machine with a 40-series (or compatible) GPU.
#
# This sets up ONLY the env server — no VAGEN training stack, no verl/vllm.
# The server accepts remote gym API calls from a cloud training machine.
#
# What this does:
#   1. Clone VAGEN-eb-alfred and Embodied-Reasoning-Agent repos
#   2. Create a conda env (Python 3.12) with minimal dependencies
#   3. Install ai2thor 2.1.0, EmbodiedBench, and server dependencies
#   4. Set up Xvfb virtual display for headless GPU rendering
#   5. Symlink ALFRED task data
#   6. Print instructions for starting the server + network setup
#
# Usage:
#   bash setup_env_server.sh [OPTIONS]
#
# Options:
#   --workspace DIR       Base directory for repos (default: ~/workspace)
#   --env-name NAME       Conda env name (default: alfred-server)
#   --skip-clone          Skip git clone if repos already exist
#   --x-display ID        X display number to use (default: 1)
#   --port PORT           Server port (default: 8000)
#   --training-host HOST  Cloud training machine IP (for SSH tunnel instructions)
#
# Requirements:
#   - conda (miniconda/miniforge)
#   - NVIDIA GPU (40-series recommended) + driver
#   - git, sudo (for Xvfb/X11 packages)
#
set -euo pipefail

# ──────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────
WORKSPACE="$HOME/workspace"
ENV_NAME="alfred-server"
SKIP_CLONE=false
X_DISPLAY="1"
PORT="8000"
TRAINING_HOST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)      WORKSPACE="$2"; shift 2 ;;
        --env-name)       ENV_NAME="$2"; shift 2 ;;
        --skip-clone)     SKIP_CLONE=true; shift ;;
        --x-display)      X_DISPLAY="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --training-host)  TRAINING_HOST="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
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

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    info "GPU: $GPU_NAME (driver $DRIVER_VER)"

    # Warn about 50-series incompatibility
    if echo "$GPU_NAME" | grep -qiE "5060|5070|5080|5090"; then
        warn "WARNING: RTX 50-series (Blackwell) GPUs are INCOMPATIBLE with ai2thor 2.1.0!"
        warn "The Unity build (2019) hangs on Controller.reset() with these GPUs."
        warn "Use a 40-series or older GPU instead."
        read -rp "Continue anyway? [y/N] " answer
        [[ "$answer" =~ ^[Yy]$ ]] || exit 1
    fi
else
    warn "nvidia-smi not found. GPU driver may not be installed."
fi

mkdir -p "$WORKSPACE"

# ──────────────────────────────────────────────────
# Step 1: Clone repos
# ──────────────────────────────────────────────────
VAGEN_DIR="$WORKSPACE/VAGEN-eb-alfred"
ERA_DIR="$WORKSPACE/Embodied-Reasoning-Agent"

info "=== Step 1/5: Cloning repositories ==="

if [ "$SKIP_CLONE" = false ]; then
    if [ ! -d "$VAGEN_DIR" ]; then
        info "Cloning VAGEN-eb-alfred..."
        git clone git@github.com:YaningDylan/VAGEN-eb-alfred.git "$VAGEN_DIR"
    else
        warn "VAGEN-eb-alfred already exists at $VAGEN_DIR, pulling latest..."
        git -C "$VAGEN_DIR" pull --ff-only || warn "Pull failed, using existing version."
    fi

    if [ ! -d "$ERA_DIR" ]; then
        info "Cloning Embodied-Reasoning-Agent (for EmbodiedBench + ALFRED data)..."
        git clone https://github.com/Embodied-Reasoning-Agent/Embodied-Reasoning-Agent.git "$ERA_DIR"
    else
        warn "Embodied-Reasoning-Agent already exists at $ERA_DIR, skipping."
    fi
else
    info "Skipping clone (--skip-clone)"
    [ -d "$VAGEN_DIR" ] || error "VAGEN-eb-alfred not found at $VAGEN_DIR"
    [ -d "$ERA_DIR" ]   || error "Embodied-Reasoning-Agent not found at $ERA_DIR"
fi

EMBENCH_DIR="$ERA_DIR/eval/EmbodiedBench"
[ -d "$EMBENCH_DIR" ] || error "EmbodiedBench not found at $EMBENCH_DIR"

# ──────────────────────────────────────────────────
# Step 2: Create conda environment (minimal)
# ──────────────────────────────────────────────────
info "=== Step 2/5: Creating conda environment '$ENV_NAME' ==="

eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Conda env '$ENV_NAME' already exists."
    read -rp "Recreate it? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "$ENV_NAME" python=3.12 -y
fi

conda activate "$ENV_NAME"
info "Python: $(python --version) at $(which python)"

# ──────────────────────────────────────────────────
# Step 3: Install dependencies (server-only, no training stack)
# ──────────────────────────────────────────────────
info "=== Step 3/5: Installing dependencies ==="

# Core server dependencies
info "Installing FastAPI + uvicorn (server framework)..."
pip install fastapi uvicorn python-multipart

# AI2-THOR 2.1.0 for EB-ALFRED
info "Installing ai2thor 2.1.0..."
pip install ai2thor==2.1.0

# EmbodiedBench dependencies (only what EBAlfEnv needs)
info "Installing EmbodiedBench dependencies..."
pip install numpy pandas opencv-python networkx h5py tqdm vocab revtok Pillow

# PyTorch (needed by EmbodiedBench model loading, use modern version)
info "Installing PyTorch..."
pip install torch torchvision

# EmbodiedBench itself (editable)
info "Installing EmbodiedBench (editable)..."
pip install -e "$EMBENCH_DIR"

# VAGEN (editable, for server code only)
info "Installing VAGEN (editable)..."
cd "$VAGEN_DIR"
pip install -e .

info "Dependencies installed."

# ──────────────────────────────────────────────────
# Step 4: Set up Xvfb + symlink ALFRED data
# ──────────────────────────────────────────────────
info "=== Step 4/5: Setting up X server and ALFRED data ==="

# Install Xvfb if not present
if ! command -v Xvfb >/dev/null 2>&1; then
    info "Installing Xvfb and X11 dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq xvfb x11-xserver-utils libx11-6 libxrender1 libxi6 \
        libxrandr2 libxcursor1 libxcomposite1 libxdamage1 libxfixes3 libxext6 \
        libgl1-mesa-glx libgl1-mesa-dri
else
    info "Xvfb already installed."
fi

# Symlink ALFRED data (json_2.1.0) if needed
ALFRED_DATA_SRC="$ERA_DIR/eval/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0"
EMBENCH_ALFRED_DATA="$EMBENCH_DIR/embodiedbench/envs/eb_alfred/data"

if [ ! -d "$ALFRED_DATA_SRC" ]; then
    warn "ALFRED data not found at $ALFRED_DATA_SRC"
    warn "You may need to download it manually. Check the EmbodiedBench README."
else
    info "ALFRED task data found at $ALFRED_DATA_SRC"
fi

# Also create a convenience symlink from the EmbodiedBench directory
# (EBAlfEnv looks for data relative to its own module path)
mkdir -p "$EMBENCH_ALFRED_DATA"
if [ ! -e "$EMBENCH_ALFRED_DATA/json_2.1.0" ] && [ -d "$ALFRED_DATA_SRC" ]; then
    ln -sf "$ALFRED_DATA_SRC" "$EMBENCH_ALFRED_DATA/json_2.1.0"
    info "Symlinked ALFRED data."
fi

# ──────────────────────────────────────────────────
# Step 5: Create helper scripts
# ──────────────────────────────────────────────────
info "=== Step 5/5: Creating helper scripts ==="

# Create start script
START_SCRIPT="$VAGEN_DIR/start_alfred_server.sh"
cat > "$START_SCRIPT" << 'SCRIPT_EOF'
#!/usr/bin/env bash
# Start the EB-ALFRED environment server with Xvfb.
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
X_DISPLAY="${X_DISPLAY:-1}"
MAX_SESSIONS="${MAX_SESSIONS:-16}"
SESSION_TIMEOUT="${SESSION_TIMEOUT:-3600}"
CONDA_ENV="${CONDA_ENV:-alfred-server}"

echo "=== EB-ALFRED Environment Server ==="
echo "Port: $PORT | Display: :$X_DISPLAY | Max sessions: $MAX_SESSIONS"

# Start Xvfb if not already running on this display
if ! pgrep -f "Xvfb :${X_DISPLAY}" > /dev/null 2>&1; then
    echo "Starting Xvfb on display :${X_DISPLAY}..."
    Xvfb ":${X_DISPLAY}" -screen 0 1024x768x24 +extension GLX +render -noreset &
    sleep 1
    echo "Xvfb started (PID $!)"
else
    echo "Xvfb already running on :${X_DISPLAY}"
fi

export DISPLAY=":${X_DISPLAY}"

# Run the server
exec conda run -n "${CONDA_ENV}" --no-capture-output \
    python -m vagen.envs.eb_alfred.serve \
        --port "${PORT}" \
        --host "${HOST}" \
        --x-displays "${X_DISPLAY}" \
        --session-timeout "${SESSION_TIMEOUT}" \
        --max-sessions "${MAX_SESSIONS}" \
        "$@"
SCRIPT_EOF
chmod +x "$START_SCRIPT"
info "Created $START_SCRIPT"

# ──────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────
echo ""
echo "=============================================="
info "Setup complete!"
echo "=============================================="
echo ""
echo "1. Start the server:"
echo "   cd $VAGEN_DIR"
echo "   ./start_alfred_server.sh"
echo ""
echo "   Or with custom settings:"
echo "   PORT=8000 MAX_SESSIONS=16 X_DISPLAY=1 ./start_alfred_server.sh"
echo ""
echo "2. Test the server:"
echo "   curl http://localhost:${PORT}/health"
echo ""
echo "3. Connect from cloud training machine:"
echo ""
echo "   Option A: SSH reverse tunnel (simple, no extra software)"
echo "     On this machine, run:"
if [ -n "$TRAINING_HOST" ]; then
echo "     ssh -R ${PORT}:localhost:${PORT} ${TRAINING_HOST}"
else
echo "     ssh -R ${PORT}:localhost:${PORT} <training-machine-ip>"
fi
echo "     Then on training machine, use: base_urls: [\"http://localhost:${PORT}\"]"
echo ""
echo "   Option B: Tailscale VPN (persistent, no port forwarding)"
echo "     Install Tailscale on both machines: curl -fsSL https://tailscale.com/install.sh | sh"
echo "     sudo tailscale up"
echo "     Then on training machine, use: base_urls: [\"http://<tailscale-ip>:${PORT}\"]"
echo ""
echo "4. Kill all Unity processes (cleanup):"
echo "   pkill -9 -f 'thor-201909061227' 2>/dev/null; pkill -9 -f Xvfb 2>/dev/null"
echo ""
