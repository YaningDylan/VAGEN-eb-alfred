# EB-ALFRED 移植到 upstream VAGEN — 架构评估 v3（已验证）

## 0. 环境验证结果（2026-02-25 实际验证）

| 检查项 | 结果 | 说明 |
|--------|------|------|
| `embench` conda env | **不存在** | 当前只有 base, bagel, vagen 三个 env |
| ai2thor in vagen | **4.3.0**（不兼容） | EBAlfEnv 需要 2.1.0，API 有 breaking changes |
| Python in vagen | **3.10.16** | EBAlfEnv 依赖链需要 3.9 |
| Xvfb | **可用** | `/usr/bin/Xvfb` 已安装 |
| DISPLAY | **未设置** | 需要启动 Xvfb 或 X server |
| EB-ALFRED 数据 | **已下载** | `eval/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0/`（132个任务目录） |
| ERA 的 embench env YAML | **存在** | `ERA-rl/VAGEN/vagen/env/Embench_new/conda_envs/environment.yaml` |

**关键结论**：
- **必须创建 `embench` conda env**（Python 3.9 + ai2thor 2.1.0），不能复用 vagen env
- **RemoteEnv 架构是必须的**，不仅仅是"推荐"——因为 Python 版本和 ai2thor 版本都不兼容
- EB-ALFRED 数据已存在，不需要额外下载
- ERA 提供了完整的 conda environment.yaml，可以直接用

## 1. 核心策略：用 VAGEN 已有的 RemoteEnv 架构

upstream VAGEN 已有完整的 Handler → Service(FastAPI) → Client 三层远程环境架构：

```
训练/评估进程 (vagen conda, Python 3.10)         AI2-THOR 服务进程 (embench conda, Python 3.9)
┌──────────────────────────────┐               ┌────────────────────────────────┐
│ GymImageEnvClient("EbAlfred")│  HTTP/multipart│ FastAPI Service (通用,不改)     │
│   (已有, 不需要写)            │ ◄──────────► │   └── EbAlfredHandler (需写)    │
│   透明代理 GymImageEnv 接口   │               │         └── EbAlfredEnv (需写)  │
└──────────────────────────────┘               │              └── EBAlfEnv (原有)│
                                               └────────────────────────────────┘
```

**关键洞察**：
- **Client 层**：直接用 `GymImageEnvClient`，零代码
- **Service 层**：直接用 `build_gym_service(handler)`，零代码
- **Handler 层**：只需实现 `create_env(env_config)` 一个方法
- **Env 层**：写一个 `EbAlfredEnv(GymImageEnv)` 适配器，包装原有 `EBAlfEnv`

## 2. 需要写的文件

### 2.1 服务端（在独立 conda env 中运行）

```
vagen/envs/eb_alfred/
├── __init__.py
├── eb_alfred_env.py        # [新写] GymImageEnv 适配器，包装 EBAlfEnv
├── handler.py              # [新写] BaseGymHandler 子类，只需 create_env()
├── serve.py                # [新写] 启动脚本，~10行
├── utils/
│   ├── __init__.py
│   ├── prompt.py           # [新写] system_prompt, format_prompt, observation templates
│   └── utils.py            # [新写] parse_response（遵循 VAGEN free_think 格式）
└── embodiedbench/          # [拷贝] 从 ERA repo 拷贝，不改或极少改
    └── envs/eb_alfred/     # EBAlfEnv + thor_connector + data/
```

### 2.2 客户端（在 vagen 主环境中）

**不需要写任何代码**，只需在 `env_registry.yaml` 注册：

```yaml
env_registry:
  Sokoban: vagen.envs.sokoban.sokoban_env.Sokoban
  FrozenLake: vagen.envs.frozenlake.frozenlake_env.FrozenLake
  EbAlfred: vagen.envs_remote.GymImageEnvClient    # 新增这一行
  RemoteEnv: vagen.envs_remote.GymImageEnvClient
```

评估/训练 YAML 中通过 `base_urls` 指向服务：
```yaml
envs:
  - name: EbAlfred
    config:
      base_urls: "http://localhost:22220"
      # ... 其他 env_config 传递给远端
```

## 3. 各文件具体内容评估

### 3.1 `eb_alfred_env.py` — GymImageEnv 适配器（~150行）

**作用**：将 EBAlfEnv 的同步接口适配为 GymImageEnv 的 async 接口。

**关键适配点**：

| ERA 原有 | VAGEN 目标 | 改动 |
|---------|-----------|------|
| `AlfredEnv(BaseEnv)` 同步 | `EbAlfredEnv(GymImageEnv)` async | 加 `async` + `to_thread` |
| `system_prompt() -> str` | `async system_prompt() -> Dict` | `{"obs_str": str}` |
| `obs["multi_modal_data"]` | `obs["multi_modal_input"]` | key 重命名 |
| `reset(seed, global_step)` | `async reset(seed: int)` | 去掉 global_step |
| `compute_reward()` | 不需要 | 删除 |
| `<\|think_start\|>...<\|action_start\|>[id, 'name']<\|action_end\|>` | `<think>...<answer>...</answer>` | **需重新设计 prompt + parse** |

**最大改动：Prompt 和 Action 格式**

ERA 格式（不遵循 VAGEN 约定）：
```
<|think_start|>reasoning<|think_end|>
<|action_start|>[42, 'find a cabinet']<|action_end|>
```

VAGEN free_think 格式（目标）：
```
<think>reasoning</think>
<answer>find a cabinet</answer>
```

需要决定的问题：
- **动作是文本名称还是 ID？** 建议用文本名称（`find a cabinet`），在 env 内部做 name → id 映射
- 这样 parse_response 可以完全复用 VAGEN 的 free_think 格式
- system_prompt 中列出可用动作列表即可

### 3.2 `handler.py` — Handler 实现（~20行）

```python
from vagen.envs_remote.handler import BaseGymHandler
from .eb_alfred_env import EbAlfredEnv

class EbAlfredHandler(BaseGymHandler):
    async def create_env(self, env_config):
        return EbAlfredEnv(env_config)
```

就这么多。VAGEN 的 Handler 已经处理了 session 管理、method dispatch、image 提取。

### 3.3 `serve.py` — 启动脚本（~15行）

```python
import uvicorn
from vagen.envs_remote.service import build_gym_service
from .handler import EbAlfredHandler

handler = EbAlfredHandler(session_timeout=3600, max_sessions=32)
app = build_gym_service(handler)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=22220)
```

### 3.4 `utils/prompt.py` — Prompt 模板（~80行）

遵循 VAGEN 的 Sokoban 模式：
- `system_prompt(skill_set)` → 生成包含动作列表的系统提示
- `format_prompt(prompt_format, ...)` → 支持 free_think 格式
- `init_observation_template(img_str)` → 初始观测模板
- `action_template(valid_actions, img_str, env_feedback, history)` → 步骤观测模板

### 3.5 `utils/utils.py` — 解析函数（~50行）

直接复用 VAGEN 的 free_think 解析逻辑：
- `parse_response(response, action_sep, max_actions, prompt_format)`
- 返回 `{"actions": [...], "format_correct": bool, "think_content": str, ...}`
- 动作是文本字符串（如 `"find a cabinet"`），在 env 的 step() 中做 name→id 映射

### 3.6 `embodiedbench/` — 原有代码（拷贝，少量改动）

需要拷贝的文件：
- `embodiedbench/envs/eb_alfred/EBAlfEnv.py`
- `embodiedbench/envs/eb_alfred/thor_connector.py`
- `embodiedbench/envs/eb_alfred/utils.py`
- `embodiedbench/envs/eb_alfred/env/` 目录
- `embodiedbench/envs/eb_alfred/gen/` 目录
- `embodiedbench/envs/eb_alfred/data/` (splits, rewards config)
- `embodiedbench/main.py` (logger)

**可能的改动**：
- 数据路径从硬编码改为可配置（通过 env_config 传入）
- X_DISPLAY 从硬编码改为可配置

## 4. 依赖隔离策略

### AI2-THOR 环境（服务端）— 需要新建 `embench` conda env

**方案 A：用 ERA 的 environment.yaml（推荐，完整验证过的环境）**
```bash
cd /home/yaning/workspace/Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/env/Embench_new
conda env create -f conda_envs/environment.yaml   # 创建 embench env, Python 3.9.20
conda activate embench
pip install -e .
# 额外安装 VAGEN 远程服务依赖
pip install fastapi uvicorn httpx
```

**方案 B：最小化安装（更快，但可能缺依赖）**
```bash
conda create -n embench python=3.9
conda activate embench
pip install ai2thor==2.1.0 gym==0.23.0 numpy==1.23.5 Pillow
pip install fastapi uvicorn httpx
```

**当前状态**：`embench` env 不存在，需要创建。ERA 的 environment.yaml 有 200+ 依赖（含 torch 2.4.0），安装耗时较长但更可靠。

### VAGEN 主环境（客户端）

无额外依赖。`GymImageEnvClient` 已内置在 VAGEN 中。

## 5. 运行流程

### 启动服务（终端 1）
```bash
conda activate embench
cd /home/yaning/workspace/VAGEN
python -m vagen.envs.eb_alfred.serve
# 监听 http://0.0.0.0:22220
```

### 运行评估（终端 2）
```bash
conda activate vagen
cd /home/yaning/workspace/VAGEN
python -m vagen.evaluate.run_eval --config tests/eval_eb_alfred_claude.yaml
```

评估 YAML 示例：
```yaml
envs:
  - name: EbAlfred
    n_envs: 3
    tag_id: eb_alfred_test
    seed: [0, 3, 1]
    split: test
    max_turns: 30
    config:
      base_urls: "http://localhost:22220"
      render_mode: vision
      image_mode: always
      prompt_format: free_think
      data_path: /path/to/eb_alfred/data
    chat_config:
      temperature: 0
      max_tokens: 1024
```

## 6. 难度评估（修订）

### 确定可行（~4-6小时工作量）

| 任务 | 时间 | 说明 |
|------|------|------|
| `eb_alfred_env.py` 适配器 | 2h | 核心工作，async 包装 + obs 格式转换 |
| `utils/prompt.py` | 1h | 重写 system prompt 和 observation templates |
| `utils/utils.py` | 0.5h | 复用 VAGEN free_think parse，加 action name 映射 |
| `handler.py` + `serve.py` | 0.5h | 极少代码 |
| 注册 + 评估 YAML | 0.5h | 配置文件 |
| 拷贝 + 适配 embodiedbench | 1h | 路径配置化、import 修复 |

### 前置条件（已部分验证，2026-02-25）

| 条件 | 状态 | 说明 |
|------|------|------|
| EB-ALFRED 数据已下载 | **已验证** | `eval/EmbodiedBench/.../data/json_2.1.0/` 有 132 个任务 |
| Xvfb 可用 | **已验证** | `/usr/bin/Xvfb` 已安装 |
| embench conda env | **需创建** | ERA 提供 environment.yaml，当前不存在 |
| AI2-THOR 2.1.0 可运行 | **需验证** | 创建 embench env 后验证 |
| EBAlfEnv reset+step | **需验证** | 需要 X server + AI2-THOR |

**阻塞项**：必须先创建 `embench` conda env 才能继续。vagen env 的 ai2thor 4.3.0 与 EBAlfEnv 不兼容。

## 7. Prompt 格式设计方案

### 推荐：完全遵循 VAGEN 的 free_think 格式

**System Prompt**（参考 Sokoban 的写法）：
```
You are a robot operating in a home. Given a task, accomplish it using the available actions.

Available actions:
0: find a Cabinet
1: find a Fridge
2: pick up the Apple
...

Rules:
1. Use 'find' to navigate to objects before interacting.
2. You can only hold one object at a time.
...

You can take 1 action at a time.
Your response should be in the format of:
<think>...</think><answer>...</answer>

Example:
<think>I need to find the apple first.</think>
<answer>find a Apple</answer>
```

**Step Observation**：
```
<image>
instruction: Put the apple in the fridge.
Last action: find a Cabinet — Last action executed successfully.
Decide your next action.
```

**Action Parse**：
- `<answer>find a Cabinet</answer>` → 在 env 中查表 → action_id=0
- 完全复用 VAGEN 的 `parse_response(free_think)` 逻辑

## 8. 与 ERA 方案的对比

| 方面 | ERA 方案 | 本方案（upstream VAGEN） |
|------|---------|------------------------|
| 服务框架 | Flask batch server + multiprocessing | FastAPI session server (已有) |
| 通信协议 | JSON + base64 images | multipart (已有 codec) |
| 环境管理 | 手动 pipe + worker | Handler session 自动管理 |
| 客户端 | 自己写的 BatchEnvClient | GymImageEnvClient (已有) |
| Prompt 格式 | `<\|think_start\|>...<\|action_start\|>[id,'name']` | `<think>...<answer>name</answer>` |
| 适配代码量 | ~500行 (env+service+client) | ~300行 (env+handler+prompt+parse) |
| 可复用性 | 仅 ERA | 可用于 VAGEN 评估/训练全套 |

## 9. 风险与缓解

| 风险 | 级别 | 缓解 |
|------|------|------|
| AI2-THOR 安装/运行 | 高 | 先在 ERA 原有 env 中验证 |
| X server 不可用 | 中 | 用 Xvfb 虚拟显示 |
| 跨进程图片传输性能 | 低 | multipart codec 已优化，500x500 PNG 几十 KB |
| action name 到 id 映射错误 | 低 | 在 env 内部做 fuzzy match 或 strict match |
| VAGEN 远程服务协议不匹配 | 极低 | 协议是通用的，已有 Sokoban 验证 |

## 10. 建议的执行顺序（已更新）

### Phase 0: 环境搭建（阻塞项）
1. **创建 embench conda env**（15-30min）
   ```bash
   cd /home/yaning/workspace/Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/env/Embench_new
   conda env create -f conda_envs/environment.yaml
   conda activate embench
   pip install -e .
   pip install fastapi uvicorn httpx  # VAGEN 远程服务依赖
   ```
2. **启动 Xvfb**：`Xvfb :1 -screen 0 1024x768x24 &`
3. **验证 AI2-THOR**：`DISPLAY=:1 python -c "import ai2thor; print(ai2thor.__version__)"`
4. **验证 EBAlfEnv**：简单 reset + step 测试

### Phase 1: 适配代码
5. **拷贝 embodiedbench** 到 `vagen/envs/eb_alfred/embodiedbench/`，修复 import
6. **写 eb_alfred_env.py**（核心适配器）
7. **写 prompt.py + utils.py**（遵循 VAGEN free_think 格式）
8. **写 handler.py + serve.py**（极简）

### Phase 2: 测试
9. **本地测试**：启动服务 → GymImageEnvClient 连接 → reset/step
10. **API 评估**：写 YAML → run_eval → Claude 端到端

### 关键路径
```
已验证的数据和 Xvfb
     ↓
[阻塞] 创建 embench env → 验证 AI2-THOR → 验证 EBAlfEnv
     ↓
拷贝 embodiedbench → 写适配代码 → 测试
```

## 11. 已验证的路径和资源

| 资源 | 路径 |
|------|------|
| EB-ALFRED 数据 | `/home/yaning/workspace/Embodied-Reasoning-Agent/eval/EmbodiedBench/embodiedbench/envs/eb_alfred/data/` |
| embodiedbench 源码（eval） | `/home/yaning/workspace/Embodied-Reasoning-Agent/eval/EmbodiedBench/embodiedbench/` |
| embodiedbench 源码（ERA-rl） | `/home/yaning/workspace/Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/env/Embench_new/embodiedbench/` |
| ERA 的 AlfredEnv wrapper | `/home/yaning/workspace/Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/env/Embench_new/alfred_env_for_vagen.py` |
| ERA 的 conda env YAML | `/home/yaning/workspace/Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/env/Embench_new/conda_envs/environment.yaml` |
| ERA 的 Flask server | `/home/yaning/workspace/Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/server/server.py` |
| VAGEN RemoteEnv handler | `/home/yaning/workspace/VAGEN/vagen/envs_remote/handler.py` |
| VAGEN RemoteEnv service | `/home/yaning/workspace/VAGEN/vagen/envs_remote/service.py` |
| VAGEN RemoteEnv client | `/home/yaning/workspace/VAGEN/vagen/envs_remote/gym_image_env_client.py` |
| VAGEN env registry | `/home/yaning/workspace/VAGEN/vagen/configs/env_registry.yaml` |
| Sokoban 参考实现 | `/home/yaning/workspace/VAGEN/vagen/envs/sokoban/sokoban_env.py` |
