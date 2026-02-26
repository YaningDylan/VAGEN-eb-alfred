# EB-ALFRED 环境搭建与运行指南

## 1. 概述

EB-ALFRED 是 EmbodiedBench 的一个子环境，基于 AI2-THOR 2.1.0（Unity 3D 物理模拟器）。
机器人在家庭场景中执行 household tasks（如 "清洗抹布并放好"）。

- 301 个评估 episode，6 个 eval set
- 162 个离散动作（find/pick/put/open/close/turn on/off/slice）
- 每个 episode 最多 30 步

## 2. Conda 环境搭建

### 2.1 创建 embench 环境

```bash
# 方案 A：最小化安装（推荐，快速）
conda create -n embench python=3.9 -y
conda activate embench
pip install ai2thor==2.1.0 gym==0.23.0 "numpy<2.0" scipy Pillow
pip install networkx revtok vocab h5py tqdm
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python anthropic openai hydra-core omegaconf
pip install google-generativeai  # evaluator planner 依赖

# 方案 B：用 ERA 的完整 environment.yaml
cd /home/yaning/workspace/Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/env/Embench_new
conda env create -f conda_envs/environment.yaml
conda activate embench
```

> **注意**：方案 B 的 environment.yaml 有 346 个包，pip 阶段可能部分失败。
> 方案 A 经过验证可以跑通所有核心功能。

### 2.2 安装 embodiedbench 包

```bash
# embodiedbench 目录下没有 __init__.py，需要先创建
touch /home/yaning/workspace/Embodied-Reasoning-Agent/eval/EmbodiedBench/embodiedbench/__init__.py

# 安装为 editable 模式
pip install -e /home/yaning/workspace/Embodied-Reasoning-Agent/eval/EmbodiedBench
```

### 2.3 验证安装

```bash
conda activate embench
python -c "from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv; print('OK')"
```

## 3. 数据

EB-ALFRED 数据已存在于：
```
/home/yaning/workspace/Embodied-Reasoning-Agent/eval/EmbodiedBench/embodiedbench/envs/eb_alfred/data/
├── json_2.1.0/          # 132 个任务目录（ALFRED task trajectories）
├── splits/splits.json   # 6 个 eval set 的 episode 列表
└── alfred_prompt_examples.json
```

如果数据缺失，从 HuggingFace 下载：
```bash
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```

### Eval Sets 统计

| Eval Set | Episodes | 说明 |
|----------|----------|------|
| base | 51 | 基础任务 |
| common_sense | 50 | 常识推理 |
| complex_instruction | 50 | 复杂指令 |
| spatial | 50 | 空间推理 |
| visual_appearance | 50 | 视觉外观 |
| long_horizon | 50 | 长期规划 |
| **Total** | **301** | |

## 4. GPU X Server 配置（渲染必需）

AI2-THOR 2.1.0 依赖 GPU 加速的 X server 做 OpenGL 渲染。
**Xvfb 不行**（只有软件 OpenGL，Unity 会 CPU 空转）。

### 4.1 检查现有 X server

```bash
ps aux | grep Xorg  # 查看是否有 Xorg 在运行
DISPLAY=:0 xdpyinfo  # 测试能否连接
```

### 4.2 如果无法连接现有 X server

需要管理员帮忙执行以下**任一**操作：

```bash
# 方案 1：允许非 console 用户启动 X（推荐）
sudo sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config
sudo usermod -aG video,render yaning

# 方案 2：已登录用户开放本地访问
# （物理终端上执行）
DISPLAY=:0 xhost +local:

# 方案 3：直接给 docker 权限
sudo usermod -aG docker yaning
```

### 4.3 启动新 X server（获得权限后）

```bash
conda activate embench
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
# 启动 GPU X server 在 :1
```

或手动：
```bash
# 生成 xorg 配置
nvidia-xconfig --query-gpu-info  # 查看 GPU BusID

# 启动 X
Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config /tmp/xorg.conf :1
```

### 4.4 设置 X_DISPLAY

EBAlfEnv 中 `X_DISPLAY` 硬编码为 `'1'`（[EBAlfEnv.py:30](embodiedbench/envs/eb_alfred/EBAlfEnv.py#L30)）。
如果 X server 在其他 display，需要修改该值或运行时 override：

```python
import embodiedbench.envs.eb_alfred.EBAlfEnv as emod
emod.X_DISPLAY = '0'  # 改为你的 display 号
```

## 5. 运行测试

### 5.1 无渲染验证（不需要 X server）

验证 imports、数据、action space、task loading：

```bash
conda activate embench
python -c "
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv, get_global_action_space, ValidEvalSets, ALFRED_SPLIT_PATH
from embodiedbench.envs.eb_alfred.gen import constants
import json

# 数据
with open(ALFRED_SPLIT_PATH) as f:
    splits = json.load(f)
for s in ValidEvalSets:
    print(f'{s}: {len(splits[s])} episodes')

# Action space
actions = get_global_action_space()
print(f'Action space: {len(actions)} actions')

# Task data
from embodiedbench.envs.eb_alfred import utils
task = splits['base'][0]
traj = utils.load_task_json(task)
print(f'Task: {traj[\"task_type\"]} @ FloorPlan{traj[\"scene\"][\"scene_num\"]}')
print(f'Instruction: {task[\"instruction\"]}')
print('ALL CHECKS PASSED')
"
```

### 5.2 带渲染的 Env 测试（需要 GPU X server）

```bash
conda activate embench
DISPLAY=:1 python -c "
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
env = EBAlfEnv(eval_set='base', down_sample_ratio=0.1, selected_indexes=[0])
print(f'Episodes: {env.number_of_episodes}')
obs = env.reset()
print(f'Image: {obs[\"head_rgb\"].shape}, Instruction: {env.episode_language_instruction}')
obs, reward, done, info = env.step(0)  # find a Cabinet
print(f'Reward: {reward}, Feedback: {info[\"env_feedback\"]}')
env.close()
print('TEST PASSED')
"
```

### 5.3 完整评估（需要 GPU X server + API key）

```bash
conda activate embench
cd /home/yaning/workspace/Embodied-Reasoning-Agent/eval/EmbodiedBench

# Claude API
DISPLAY=:1 python -m embodiedbench.main \
    env=eb-alf \
    model_name=claude-sonnet-4-20250514 \
    model_type=remote \
    exp_name=claude_test \
    eval_sets='[base]' \
    down_sample_ratio=0.1 \
    n_shots=0 \
    chat_history=False \
    resolution=500
```

## 6. 关键代码路径

| 文件 | 说明 |
|------|------|
| `embodiedbench/envs/eb_alfred/EBAlfEnv.py` | 核心 Gym 环境（reset/step/close） |
| `embodiedbench/envs/eb_alfred/thor_connector.py` | AI2-THOR 技能交互层（find/pick/put/open...） |
| `embodiedbench/envs/eb_alfred/env/thor_env.py` | AI2-THOR Controller 扩展 |
| `embodiedbench/evaluator/eb_alfred_evaluator.py` | 评估主循环 |
| `embodiedbench/planner/vlm_planner.py` | VLM Agent（支持 Claude/GPT/Gemini） |
| `embodiedbench/planner/remote_model.py` | API 模型封装 |
| `embodiedbench/evaluator/config/system_prompts.py` | System prompt 模板 |
| `embodiedbench/envs/eb_alfred/scripts/startx.py` | 启动 GPU X server |

## 7. 已知问题

### lmdeploy 与 Python 3.9 不兼容

`remote_model.py` 顶部 import 了 `lmdeploy`，最新版使用了 Python 3.10+ 语法（`bool | None`）。
解决方案：安装兼容版本 `pip install lmdeploy==0.6.0` 或者仅在 `model_type=local` 时才需要该模块。

### Gym 版本警告

`gym==0.23.0` 已停止维护，与 numpy>=2.0 不兼容。当前用 `numpy<2.0` 规避。
这不影响功能，只是有 warning。

### AI2-THOR 首次运行下载

首次运行时 AI2-THOR 会自动下载 Unity binary 到 `~/.ai2thor/releases/`。
当前已下载：`thor-201909061227-Linux64`（约 500MB）。

## 8. 环境信息

```
Machine:    mll-4090-3 (RTX 4090)
OS:         Ubuntu 22.04 (kernel 6.8.0-94)
Python:     3.9.x (embench conda env)
ai2thor:    2.1.0
torch:      2.4.0+cu124 (方案 A) / 2.8.0 (当前)
CUDA:       12.4
```
