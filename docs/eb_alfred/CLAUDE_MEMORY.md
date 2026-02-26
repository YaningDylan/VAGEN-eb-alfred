# EB-ALFRED Migration — Claude Memory & Action List

> 这是跨 conversation 的详细参考文档。包含完成状态、未完成任务、架构决策、
> 已知问题和调试经验。供 Claude 在新 conversation 中快速恢复上下文。

---

## 1. 项目总览

**目标**：将 EmbodiedBench 的 EB-ALFRED 环境（AI2-THOR 3D household tasks）
移植到 VAGEN（VLM Agent + Multi-Turn RL 框架），使其能用于 RL 训练和 API 评估。

**仓库**：
- VAGEN（主仓库）：`/home/yaning/workspace/VAGEN`
- ERA（参考源码）：`/home/yaning/workspace/Embodied-Reasoning-Agent`

**Conda 环境**：
- `vagen`：Python 3.10，VAGEN 训练/评估
- `embench`：Python 3.9，ai2thor 2.1.0，EB-ALFRED 服务端（已在当前机器创建）

**用户偏好**：
- 中文交流
- 希望有文档以防上下文丢失
- 优先使用 VAGEN 已有框架，不写自定义脚本
- 不需要每步确认，可以直接改文件

---

## 2. 已完成任务 (Completed)

### Phase 0: 环境验证 ✅
- [x] 创建 `embench` conda env（Python 3.9, 最小化安装）
- [x] 安装 ai2thor==2.1.0, gym==0.23.0, torch==2.4.0+cu124
- [x] 安装 embodiedbench 包（`pip install -e`，需先创建 `__init__.py`）
- [x] 无渲染验证通过：imports, data (301 episodes, 6 eval sets), action space (162 actions), task loading
- [x] 编写 embench 环境搭建文档：`Embodied-Reasoning-Agent/docs/eb_alfred_setup.md`

### Phase 1: VAGEN 适配器代码 ✅
- [x] 创建目录结构 `vagen/envs/eb_alfred/`（7 个文件）
- [x] 编写 `eb_alfred_env.py` — `EbAlfred(GymImageEnv)` 适配器
  - EbAlfredEnvConfig dataclass（eval_set, resolution, x_display, max_turns, prompt_format, etc.）
  - 4 个 async 方法：close, system_prompt, reset, step
  - seed → episode 映射：`seed % number_of_episodes`
  - 动态 action space：每 episode reset 后重建 `_action_map`
  - Action 匹配：支持 action name（case-insensitive）和 action ID（整数）
  - 奖励：format_reward + success_reward
  - Metrics：turn_metrics (action_is_valid, action_is_effective) + traj_metrics (success)
- [x] 编写 `utils/prompt.py` — system_prompt, format_prompt (free_think), init/action templates
- [x] 编写 `utils/utils.py` — parse_response (free_think), match_action, numpy_to_pil
- [x] 编写 `handler.py` — EbAlfredHandler(BaseGymHandler)，用 asyncio.to_thread 创建 env
- [x] 编写 `serve.py` — FastAPI 服务启动脚本
- [x] 注册 `env_registry.yaml`：`EbAlfred: vagen.envs.eb_alfred.eb_alfred_env.EbAlfred`
- [x] 验证 utils import + parse + match + registry 路径解析

### 文档 ✅
- [x] `docs/eb_alfred/migration_assessment.md` — 架构评估 v3
- [x] `docs/eb_alfred/embench_setup.md` — embench 环境搭建指南
- [x] `docs/eb_alfred/vagen_env_guide.md` — VAGEN 新环境添加通用指南
- [x] `docs/eb_alfred/CLAUDE_MEMORY.md` — 本文件

---

## 3. 未完成任务 (Pending)

### Phase 2: 端到端测试（阻塞于 GPU X server）
- [ ] **在有 GPU X server 的机器上运行带渲染的 EBAlfEnv 测试**
  ```bash
  conda activate embench
  DISPLAY=:1 python -c "
  from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
  env = EBAlfEnv(eval_set='base', selected_indexes=[0])
  obs = env.reset()
  print(f'Image: {obs[\"head_rgb\"].shape}')
  obs, reward, done, info = env.step(0)
  print(f'Reward: {reward}, Feedback: {info[\"env_feedback\"]}')
  env.close()
  "
  ```

- [ ] **在有 GPU X server 的机器上运行 VAGEN EbAlfred 适配器测试**
  ```bash
  conda activate embench  # 注意：直接运行需要 embench env，因为 ai2thor==2.1.0
  cd /path/to/VAGEN
  DISPLAY=:1 python -c "
  import asyncio
  from vagen.envs.eb_alfred.eb_alfred_env import EbAlfred

  async def test():
      env = EbAlfred({
          'eval_set': 'base',
          'x_display': '1',
          'selected_indexes': [0],
          'prompt_format': 'free_think',
      })

      # system_prompt
      sys = await env.system_prompt()
      print('System prompt OK:', len(sys['obs_str']), 'chars')

      # reset
      obs, info = await env.reset(seed=0)
      print('Task:', info['task_instruction'])
      print('Actions:', info['num_actions'])
      assert '<image>' in obs['obs_str']
      assert len(obs['multi_modal_input']['<image>']) == 1
      obs['multi_modal_input']['<image>'][0].save('/tmp/eb_alfred_reset.png')
      print('Reset OK, image saved')

      # step (valid action)
      resp = '<think>I need to find a Cabinet first.</think><answer>find a Cabinet</answer>'
      obs, reward, done, info = await env.step(resp)
      print(f'Step: reward={reward}, done={done}, success={info[\"success\"]}')

      # step (invalid format)
      obs, reward, done, info = await env.step('garbage')
      print(f'Bad format: reward={reward}')

      await env.close()
      print('ALL TESTS PASSED')

  asyncio.run(test())
  "
  ```

- [ ] **启动远程服务 + 客户端测试**
  ```bash
  # Terminal 1: 启动服务（embench env，需要 GPU X server）
  conda activate embench
  cd /path/to/VAGEN
  DISPLAY=:1 python -m vagen.envs.eb_alfred.serve --port 8000

  # Terminal 2: 客户端测试（vagen env）
  conda activate vagen
  cd /path/to/VAGEN
  python -c "
  import asyncio
  from vagen.envs_remote import GymImageEnvClient

  async def test():
      client = GymImageEnvClient({
          'base_urls': ['http://localhost:8000'],
          'eval_set': 'base',
          'x_display': '1',
          'selected_indexes': [0],
      })
      obs, info = await client.reset(seed=0)
      print('Remote reset OK:', obs['obs_str'][:100])
      sys = await client.system_prompt()
      print('Remote system_prompt OK:', len(sys['obs_str']), 'chars')
      resp = '<think>test</think><answer>find a Cabinet</answer>'
      obs, reward, done, info = await client.step(resp)
      print(f'Remote step OK: reward={reward}, done={done}')
      await client.close()
      print('ALL REMOTE TESTS PASSED')
  asyncio.run(test())
  "
  ```

- [ ] **编写评估 YAML 配置**
  需要创建 `tests/eval_eb_alfred_claude.yaml`（参考 `tests/eval_sokoban_claude.yaml`）

- [ ] **Claude API 端到端评估**
  ```bash
  conda activate vagen
  python -m vagen.evaluate.run_eval --config tests/eval_eb_alfred_claude.yaml
  ```

- [ ] **（可选）RL 训练集成测试**

---

## 4. 架构决策记录

### 4.1 为什么用 RemoteEnv 而不是直接本地运行？
- **Python 版本不兼容**：vagen env 是 3.10，embodiedbench 需要 3.9
- **ai2thor 版本不兼容**：vagen 有 4.3.0，EBAlfEnv 需要 2.1.0（breaking changes）
- **跨机器部署**：AI2-THOR 需要 GPU X server，RL 训练可以在另一台机器
- VAGEN 已有完整 RemoteEnv 架构（Handler→Service→Client），零额外开发

### 4.2 Action 格式选择
- **选择 action name (字符串)**，而非 action ID (整数)
- 原因：LLM 更擅长输出自然语言（"find a Cabinet"），不容易记住 162 个数字
- 同时也支持 action ID 作为 fallback（`match_action` 先尝试 int 解析）

### 4.3 Prompt 格式
- **选择 free_think**：`<think>...</think><answer>...</answer>`
- 与 VAGEN 的 Sokoban/FrozenLake 一致
- action 放在 `<answer>` 标签内
- 暂不支持 wm 和 free_wm（EB-ALFRED 观测是图像，不适合 world model 文本描述）

### 4.4 System Prompt vs Observation 中的动态内容
- **system_prompt()**: 静态内容（role, action descriptions, guidelines, format instructions）
- **reset() observation**: 动态内容（task instruction, available actions for this episode, image）
- **step() observation**: 每步内容（last action, feedback, image）
- 原因：system_prompt() 在 VAGEN 中每 session 调用一次，而 EB-ALFRED action space 每 episode 变化

### 4.5 Seed → Episode 映射
- `seed % env.number_of_episodes` 选择 episode index
- EBAlfEnv 内部用 `_current_episode_num` 追踪，每次 reset 前 override
- 支持 `selected_indexes` 过滤后再做 modulo

---

## 5. 关键文件清单

### VAGEN 适配器（新写的 7 个文件）
| 文件 | 行数 | 说明 |
|------|------|------|
| `vagen/envs/eb_alfred/__init__.py` | 0 | 空文件 |
| `vagen/envs/eb_alfred/eb_alfred_env.py` | ~250 | **核心适配器**：EbAlfred(GymImageEnv) |
| `vagen/envs/eb_alfred/utils/__init__.py` | 0 | 空文件 |
| `vagen/envs/eb_alfred/utils/prompt.py` | ~90 | system_prompt, format_prompt, observation templates |
| `vagen/envs/eb_alfred/utils/utils.py` | ~90 | parse_response, match_action, numpy_to_pil |
| `vagen/envs/eb_alfred/handler.py` | ~20 | EbAlfredHandler(BaseGymHandler) |
| `vagen/envs/eb_alfred/serve.py` | ~50 | FastAPI server 启动脚本 |

### 修改的文件
| 文件 | 改动 |
|------|------|
| `vagen/configs/env_registry.yaml` | 添加 `EbAlfred` 行 |

### 文档
| 文件 | 说明 |
|------|------|
| `docs/eb_alfred/CLAUDE_MEMORY.md` | 本文件 - 跨 conversation 参考 |
| `docs/eb_alfred/migration_assessment.md` | 架构评估 v3 |
| `docs/eb_alfred/embench_setup.md` | embench conda env 搭建指南 |
| `docs/eb_alfred/vagen_env_guide.md` | VAGEN 新环境添加通用指南 |

### 参考文件（不修改）
| 文件 | 用途 |
|------|------|
| `vagen/envs/sokoban/sokoban_env.py` | **参考实现**（EB-ALFRED 遵循相同模式） |
| `vagen/envs/sokoban/utils/prompt.py` | prompt 模板参考 |
| `vagen/envs/sokoban/utils/utils.py` | parse_response 参考 |
| `vagen/envs_remote/handler.py` | BaseGymHandler（session 管理） |
| `vagen/envs_remote/service.py` | build_gym_service（FastAPI 工厂） |
| `vagen/envs_remote/examples/simple_example.py` | handler/serve/client 完整示例 |

### EmbodiedBench 关键文件（在 ERA repo 中）
| 文件 | 说明 |
|------|------|
| `eval/EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py` | 核心 Gym 环境 (448行) |
| `eval/EmbodiedBench/embodiedbench/envs/eb_alfred/thor_connector.py` | AI2-THOR 技能交互层 |
| `eval/EmbodiedBench/embodiedbench/envs/eb_alfred/env/thor_env.py` | AI2-THOR Controller 扩展 |
| `eval/EmbodiedBench/embodiedbench/evaluator/config/system_prompts.py` | 原始 system prompt 模板 |
| `eval/EmbodiedBench/embodiedbench/envs/eb_alfred/data/splits/splits.json` | 6 eval set, 301 episodes |

---

## 6. 已知问题 & 调试经验

### GPU X Server（最大阻塞项）
- AI2-THOR 2.1.0 (Unity 2019) **必须**有 GPU 加速的 X server
- **Xvfb 不行**：只有软件 OpenGL，Unity CPU 空转 (2000% CPU)
- **EGL/headless 不行**：ai2thor 2.1.0 的 Unity binary 太旧，不支持
- 需要管理员修改 `/etc/X11/Xwrapper.config`（`allowed_users=anybody`）或执行 `xhost +local:`
- 当前机器 (`mll-4090-3`) 的 X server (:0) 属于用户 `pingyue`，认证受限

### embench Conda Env 注意事项
- ERA 的完整 `environment.yaml`（346 包）pip 阶段容易失败
- **推荐最小化安装**：Python 3.9 + ai2thor==2.1.0 + 必要依赖
- `embodiedbench` 包缺少 `__init__.py`，需手动创建
- `lmdeploy` 最新版与 Python 3.9 不兼容（`bool | None` 语法），但只在 `model_type=local` 时需要
- `numpy>=2.0` 与 `gym==0.23.0` 不兼容，需用 `numpy<2.0`

### EBAlfEnv 内部细节
- `X_DISPLAY` 在 `EBAlfEnv.py:30` 硬编码为 `'1'`，通过 `ebalfenv_mod.X_DISPLAY = config.x_display` override
- `EBAlfEnv.reset()` 内部 `_current_episode_num` 自增，适配器每次 reset 前重置
- Action space 每 episode 动态变化（`generate_additional_action_space()` 添加多实例物体）
- `EBAlfEnv.step()` 接受 int 或 str 类型的 action
- `step()` 返回的 `info['env_feedback']` 是环境反馈文本，对 LLM 有用

### VAGEN 架构要点
- `GymImageEnv` 是 async ABC，所有阻塞操作用 `asyncio.to_thread()`
- `obs["obs_str"]` 中的 `<image>` 占位符数量必须等于图片列表长度
- `info["success"]` 是必需字段，wandb 和评估框架都用
- Registry 用 `importlib.import_module()` 动态加载，路径格式：`package.module.ClassName`
- RemoteEnv: Handler 只需实现 `create_env()`，Service 和 Client 完全复用

---

## 7. 在新机器上的快速启动指南

### 前提条件
1. 机器有 NVIDIA GPU + GPU 加速的 X server（`DISPLAY=:N` 可连接）
2. 安装了 conda
3. Clone 了 VAGEN repo

### 步骤

```bash
# 1. 创建 embench conda env
conda create -n embench python=3.9 -y
conda activate embench
pip install ai2thor==2.1.0 gym==0.23.0 "numpy<2.0" scipy Pillow
pip install networkx revtok vocab h5py tqdm
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python hydra-core omegaconf
pip install google-generativeai
pip install fastapi uvicorn httpx  # VAGEN 远程服务依赖

# 2. 安装 embodiedbench
# 注意：embodiedbench 需要从 ERA repo 安装，或者将 embodiedbench 包拷贝到服务机器
# 如果 ERA repo 可用：
touch /path/to/ERA/eval/EmbodiedBench/embodiedbench/__init__.py
pip install -e /path/to/ERA/eval/EmbodiedBench

# 3. 验证 X server
DISPLAY=:1 xdpyinfo | head -5  # 应该能看到 display 信息

# 4. 测试 EBAlfEnv（原始环境）
DISPLAY=:1 python -c "
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
env = EBAlfEnv(eval_set='base', selected_indexes=[0])
obs = env.reset()
print('Image shape:', obs['head_rgb'].shape)
env.close()
print('EBAlfEnv OK')
"

# 5. 测试 VAGEN 适配器
cd /path/to/VAGEN
DISPLAY=:1 python -c "
import asyncio
from vagen.envs.eb_alfred.eb_alfred_env import EbAlfred
async def test():
    env = EbAlfred({'eval_set': 'base', 'x_display': '1', 'selected_indexes': [0]})
    sys = await env.system_prompt()
    obs, info = await env.reset(seed=0)
    print('Task:', info['task_instruction'])
    resp = '<think>test</think><answer>find a Cabinet</answer>'
    obs, r, done, info = await env.step(resp)
    print(f'reward={r}, done={done}')
    await env.close()
    print('EbAlfred adapter OK')
asyncio.run(test())
"

# 6. 启动远程服务
DISPLAY=:1 python -m vagen.envs.eb_alfred.serve --port 8000

# 7. （另一个终端）客户端测试
conda activate vagen
python -c "
import asyncio
from vagen.envs_remote import GymImageEnvClient
async def test():
    client = GymImageEnvClient({
        'base_urls': ['http://localhost:8000'],
        'eval_set': 'base', 'x_display': '1', 'selected_indexes': [0],
    })
    obs, info = await client.reset(seed=0)
    print('Remote OK:', obs['obs_str'][:80])
    await client.close()
asyncio.run(test())
"
```

---

## 8. EbAlfredEnvConfig 完整参数

```python
@dataclass
class EbAlfredEnvConfig:
    # 环境设置
    eval_set: str = "base"               # base/common_sense/complex_instruction/spatial/visual_appearance/long_horizon
    exp_name: str = "vagen_eval"          # 日志目录名
    down_sample_ratio: float = 1.0        # episode 采样比例 (0~1)
    resolution: int = 500                 # 图像分辨率 (500x500)
    x_display: str = "1"                  # X server display 号
    selected_indexes: List[int] = []      # 选择特定 episode（空=全部）
    detection_box: bool = False           # 是否在图像上画检测框

    # 交互设置
    max_turns: int = 30                   # 最大交互轮数
    max_actions_per_step: int = 1         # 每步最多 1 个 action
    action_sep: str = ","                 # action 分隔符
    image_placeholder: str = "<image>"    # 图像占位符
    prompt_format: str = "free_think"     # 仅支持 free_think
    use_example_in_sys_prompt: bool = True # 是否在 system prompt 中包含示例

    # 奖励设置
    format_reward: float = 0.1            # 正确格式奖励
    success_reward: float = 1.0           # 任务成功奖励
```

---

*Last updated: 2026-02-25*
*Author: Claude (across multiple conversations)*
