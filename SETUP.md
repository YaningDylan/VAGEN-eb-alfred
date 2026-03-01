# EB-ALFRED 新机器完整部署指南

在一台全新的 GPU 机器上从零搭建 EB-ALFRED 环境并跑通 eval 和 training 的完整流程。

---

## 0. 前置要求

- Ubuntu (18.04/20.04/22.04/24.04)
- NVIDIA GPU（至少 1 张，推荐 A100/H100）
- NVIDIA 驱动已安装（`nvidia-smi` 可用）
- Conda 已安装
- sudo 权限（仅 Xorg 配置步骤需要）

---

## 1. 克隆代码

```bash
# 主 RL 框架（eval + training）
git clone git@github.com:mll-lab-nu/VAGEN.git
cd VAGEN
git submodule update --init --recursive  # verl 等子模块

# EmbodiedBench（AI2-THOR 环境底层）
cd ..
git clone https://github.com/EmbodiedBench/EmbodiedBench.git
# 注意：需要 git-lfs 拉取 EB-ALFRED data
cd EmbodiedBench && git lfs pull
```

---

## 2. 创建 Conda 环境

需要两个独立的 conda 环境，因为 AI2-THOR 2.1.0 依赖 Python 3.9 + 固定版本的 Flask/Werkzeug，和 VAGEN 训练侧不兼容。

### 2a. `embench` 环境（运行 EB-ALFRED 环境服务器）

```bash
conda create -n embench python=3.9 -y
conda activate embench

# 核心依赖（版本严格锁定）
pip install ai2thor==2.1.0
pip install gym==0.23.0 "numpy<2.0" scipy Pillow
pip install networkx revtok vocab h5py tqdm

# PyTorch（GPU 版本，CUDA 12.4）
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 通信 + 工具
pip install opencv-python
pip install fastapi uvicorn httpx

# Flask/Werkzeug 版本锁定（ai2thor 2.1.0 不兼容 werkzeug 2.0+）
pip install "werkzeug==1.0.1" "flask==1.1.4" "markupsafe<2.1" "jinja2<3.0" "itsdangerous<2.0"

# 安装 EmbodiedBench
pip install -e /path/to/EmbodiedBench

# 安装 VAGEN（server 侧需要 envs_remote 的 handler 代码）
pip install -e /path/to/VAGEN
```

验证：
```bash
conda activate embench
python -c "import ai2thor; print(ai2thor.__version__)"          # 2.1.0
python -c "import werkzeug; print(werkzeug.__version__)"        # 1.0.1
python -c "from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv; print('OK')"
```

### 2b. `vagen` 环境（运行 eval/training 客户端）

```bash
conda create -n vagen python=3.10 -y
conda activate vagen

cd /path/to/VAGEN
pip install -e .
# training 额外依赖
pip install wandb sglang
```

---

## 3. Patch ai2thor（必须）

ai2thor 2.1.0 有一个 werkzeug 导致的 `SocketException` bug，不 patch 的话环境会在几步后崩溃。

```bash
conda run -n embench python scripts/patch_ai2thor.py
```

验证 patch 成功：
```bash
conda run -n embench python scripts/patch_ai2thor.py --check
```

预期输出：
```
  OK: run_wsgi override with keep-alive
  OK: threaded server mode
  OK: shutdown_request bypass
```

> 如果 `scripts/patch_ai2thor.py` 不在你的 repo 里，从 VAGEN-eb-alfred 复制过来。

---

## 4. 配置 Xorg（最容易出问题的环节）

AI2-THOR 的 Unity 渲染器需要 GPU 加速的 X server。**Xvfb 不行**（只有软件 OpenGL，Unity 渲染不出来）。

### 4.1 检查现有 X server

```bash
# 看有没有 Xorg 在跑
ps aux | grep -E 'Xorg|Xwayland'

# 看有哪些 display
ls /tmp/.X11-unix/

# 测试连接
DISPLAY=:0 xdpyinfo 2>&1 | head -5
```

如果输出正常（看到 display info），跳到 4.4。

### 4.2 无 X server 的情况：启动 GPU X server

**方案 A：用 EmbodiedBench 自带的 startx.py（推荐）**

```bash
# 需要 sudo 或已配置 Xwrapper
# 先确保 Xwrapper 允许任意用户启动 X
sudo sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config
# 如果文件不存在
echo "allowed_users=anybody" | sudo tee /etc/X11/Xwrapper.config

# 在 tmux 里启动（会阻塞前台）
tmux new -s xserver
conda activate embench
python -m embodiedbench.envs.eb_alfred.scripts.startx 0
# Ctrl+B, D 退出 tmux
```

这会：
1. 用 `lspci` 检测所有 NVIDIA GPU
2. 生成临时 xorg.conf（配置所有 GPU 的 Screen）
3. 启动 `Xorg -noreset +extension GLX +extension RANDR +extension RENDER :0`

**方案 B：手动启动 Xorg**

```bash
# 生成 xorg 配置
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# 启动 X server
sudo Xorg :0 -config /etc/X11/xorg.conf &

# 验证
DISPLAY=:0 glxinfo | head -5
```

**方案 C：多 GPU 多 Display**

如果有 N 张 GPU，每张 GPU 可以对应一个 display：

```bash
# GPU 0 → :0, GPU 1 → :1, ...
python -m embodiedbench.envs.eb_alfred.scripts.startx 0  # 在 tmux 1
python -m embodiedbench.envs.eb_alfred.scripts.startx 1  # 在 tmux 2
```

EB-ALFRED server 会自动检测所有 display 并做 load balancing。

### 4.3 常见 Xorg 问题

**问题：`(EE) no screens found`**

原因：Xorg 无法识别 GPU。
修复：
```bash
# 查看 GPU 信息
nvidia-smi
lspci | grep -i nvidia

# 确保 nvidia 驱动模块加载
lsmod | grep nvidia

# 如果是 headless GPU（如 A100/H100），可能需要
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
# 然后用生成的 /etc/X11/xorg.conf 启动
```

**问题：`Cannot open display ":0"` / `Authorization required`**

修复：
```bash
# 方案 1：允许本地连接
DISPLAY=:0 xhost +local:

# 方案 2：设置 XAUTHORITY
export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
# 或
export XAUTHORITY=$HOME/.Xauthority
```

**问题：`(EE) Fatal server error: Server is already active for display 0`**

修复：
```bash
# 用另一个 display 号
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
# 然后 DISPLAY=:1 启动 server
```

**问题：Xorg 权限不够**

```bash
# 加权限
sudo usermod -aG video,render $(whoami)
# 重新登录生效

# 或直接改 Xwrapper
sudo sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config
```

### 4.4 验证 GPU 渲染

```bash
DISPLAY=:0 conda run -n embench python -c "
from ai2thor.controller import Controller
c = Controller(scene='FloorPlan1', gridSize=0.25)
for i in range(5):
    e = c.step('RotateRight')
    print(f'Step {i+1}: success={e.metadata[\"lastActionSuccess\"]}')
c.stop()
print('GPU rendering OK')
"
```

---

## 5. 启动 EB-ALFRED 环境服务器

```bash
# 基本启动（自动检测 GPU）
DISPLAY=:0 conda run -n embench python -m vagen.envs.eb_alfred.serve \
    --port 8000 --max-sessions 64

# 多 GPU（手动指定 display）
DISPLAY=:0 conda run -n embench python -m vagen.envs.eb_alfred.serve \
    --port 8000 --x-displays 0,1,2,3 --max-sessions 200

# 或用封装脚本
DISPLAY=:0 MAX_SESSIONS=200 bash examples/evaluate/eb_alfred/start_server.sh
```

验证：
```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

### GPU 容量参考

| GPU | VRAM | 单 GPU 最大 Session 数 |
|-----|------|----------------------|
| RTX 4060 (8GB) | 8 GB | ~31 |
| RTX 4090 (24GB) | 24 GB | ~110 |
| A100 (80GB) | 80 GB | ~384 |
| H100 (80GB) | 80 GB | ~384 |

每个 Unity 实例额外占约 750MB 系统内存。

> 建议 `--max-sessions` 设为 GPU 容量的 80%，留余量给训练用的 sglang rollout。

---

## 6. 跑 Evaluation

### 6a. 快速测试（3 episode）

```bash
export OPENAI_API_KEY="sk-..."

conda run -n vagen python -m vagen.evaluate.run_eval \
    --config examples/evaluate/eb_alfred/config.yaml \
    envs[0].n_envs=3
```

### 6b. 完整 eval（50 episode，GPT-4.1）

```bash
conda run -n vagen python -m vagen.evaluate.run_eval \
    --config examples/evaluate/eb_alfred/config.yaml
```

### 6c. 切换模型

```bash
# 用 Claude
conda run -n vagen python -m vagen.evaluate.run_eval \
    --config examples/evaluate/eb_alfred/config.yaml \
    run.backend=claude

# 用本地 sglang 模型
conda run -n vagen python -m vagen.evaluate.run_eval \
    --config examples/evaluate/eb_alfred/config.yaml \
    run.backend=sglang
```

---

## 7. 跑 PPO Training（8x H100 示例）

Training 分两部分：EB-ALFRED server（环境交互）和 VAGEN trainer（PPO 训练）。它们可以在同一台机器上运行，也可以分开。

### 7a. 启动环境服务器

在 tmux 里：
```bash
tmux new -s eb_server

# 8x H100：启动 8 个 display 或只用一个 display
# 方案 1：单 display（简单，所有 Unity 用同一个 X server）
DISPLAY=:0 conda run -n embench python -m vagen.envs.eb_alfred.serve \
    --port 8000 --max-sessions 200 --thread-workers 128

# 方案 2：多 display（每 GPU 一个 X server，负载均衡）
# 先启动多个 X server（需要多个 tmux）
# 然后：
DISPLAY=:0 conda run -n embench python -m vagen.envs.eb_alfred.serve \
    --port 8000 --x-displays 0,1,2,3,4,5,6,7 --max-sessions 400
```

### 7b. 启动 PPO Training

```bash
tmux new -s training

conda activate vagen
cd /path/to/VAGEN

# PPO no-concat, Qwen 3B, 8x H100, ERA-aligned
bash examples/eb_alfred/train_ppo_no_concat_qwen25vl3b.sh
```

关键参数（与 ERA 对齐）：
- `train_batch_size=50`：每步采样 50 个 episode
- `max_prompt_length=2048, max_response_length=512`：token 长度限制
- `max_trajectory_length=7000`：整条轨迹 token 上限
- `gamma=0.99, lam=0.99`：GAE 参数
- `actor.lr=1e-6, critic.lr=1e-5`：学习率
- `freeze_vision_tower=True`：冻结视觉编码器
- `total_training_steps=15`：总训练步数
- `n_gpus_per_node=8`：使用 8 张 GPU

### 7c. 监控训练

```bash
# WandB（推荐）
# 训练会自动上传到 wandb，查看 vagen_experiments/ppo_eb_alfred_no_concat

# 日志
tail -f exps/vagen_experiments/ppo_eb_alfred_no_concat/vagen_experiments_ppo_eb_alfred_no_concat.log

# GPU 利用率
watch -n 2 nvidia-smi
```

---

## 8. Troubleshooting 速查

| 现象 | 原因 | 解决 |
|------|------|------|
| `SocketException: Connection reset` | ai2thor 未 patch | `conda run -n embench python scripts/patch_ai2thor.py` |
| `Cannot open display` | X server 未启动或权限不够 | 见第 4 节 |
| `ImportError: werkzeug` | werkzeug 版本不对 | `pip install "werkzeug==1.0.1"` |
| Unity 进程残留 | server 非正常退出 | `pkill -f thor-CloudRendering` |
| 端口占用 | 上次 server 未关闭 | `lsof -i :8000` 然后 `kill PID` |
| `CUDA out of memory` | Unity 实例太多 | 减少 `--max-sessions` |
| sglang rollout OOM | gpu_memory_utilization 太高 | 训练脚本中降低 `gpu_memory_utilization` |
| eval 全部 401 | API key 未设置 | `export OPENAI_API_KEY="sk-..."` |
| numpy 不兼容 | numpy >= 2.0 和 gym 0.23 冲突 | `pip install "numpy<2.0"` |

---

## 9. 完整启动 Checklist

```
[ ] nvidia-smi 正常
[ ] conda 已安装
[ ] 代码已克隆（VAGEN + EmbodiedBench）
[ ] embench conda 环境创建完成
[ ] vagen conda 环境创建完成
[ ] ai2thor patch 已应用（--check 三个 OK）
[ ] Xorg 已启动并可连接（DISPLAY=:0 xdpyinfo）
[ ] EB-ALFRED server 已启动（/health 返回 OK）
[ ] 快速 eval 测试通过（3 episode）
[ ] API key 已设置（eval）或 sglang 服务已启动（training）
[ ] wandb 已登录（training）
```
