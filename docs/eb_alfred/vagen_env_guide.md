# VAGEN 新环境添加完整指南

## 1. 架构概览

```
GymBaseEnv (ABC)            # vagen/envs/gym_base_env.py
  └── GymImageEnv (ABC)     # vagen/envs/gym_image_env.py  (支持多模态图像)
        ├── Sokoban          # vagen/envs/sokoban/sokoban_env.py
        ├── FrozenLake       # vagen/envs/frozenlake/frozenlake_env.py
        └── YourNewEnv       # vagen/envs/yourenv/yourenv_env.py

Registry: vagen/configs/env_registry.yaml  →  动态加载环境类
```

## 2. 必须实现的接口

继承 `GymImageEnv`，实现以下 4 个 async 抽象方法：

### 2.1 `async close() -> None`
清理资源（关闭底层环境、释放 GPU 等）。

### 2.2 `async system_prompt() -> Dict[str, Any]`
返回系统提示词。调用一次，用于 LLM 的 system message。

**返回格式**（纯文本）：
```python
{"obs_str": "You are a XXX solver. Rules: ..."}
```

**返回格式**（含图片，少见）：
```python
{
    "obs_str": "Instructions: <image> ...",
    "multi_modal_input": {"<image>": [PIL.Image.Image]}
}
```

### 2.3 `async reset(seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]`
重置环境，返回 `(obs, info)`。

**obs 格式**（文本模式）：
```python
{"obs_str": "[Initial Observation]:\n# _ X P ...\nDecide your next action(s)."}
```

**obs 格式**（视觉模式）：
```python
{
    "obs_str": "[Initial Observation]:\n<image>\nDecide your next action(s).",
    "multi_modal_input": {"<image>": [PIL.Image.Image]}
}
```

**info**: 空 dict 即可 `{}`

### 2.4 `async step(action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]`
执行动作，返回 `(obs, reward, done, info)`。

- `action_str`: LLM 的原始输出文本（如 `<think>...</think><answer>Up</answer>`）
- `reward`: float 标量奖励
- `done`: bool 是否结束
- `info`: 必须包含 `"success": bool`，可选包含 `"metrics"` dict

**info 标准格式**：
```python
{
    "success": False,
    "metrics": {
        "turn_metrics": {
            "action_is_valid": True,
            "action_is_effective": True,
        },
        "traj_metrics": {
            "success": False,
        }
    },
    # parse_response 返回的字段也会被 update 进来
    "llm_raw_response": "...",
    "format_correct": True,
    "actions": ["up", "left"],
    ...
}
```

## 3. 观测协议（Observation Protocol）

**关键规则**：
- `obs_str` 中的 `<image>` 占位符数量 **必须等于** `multi_modal_input["<image>"]` 列表长度
- 如果没有图片，`obs_str` 中**不要**包含 `<image>`
- 图片必须是 `PIL.Image.Image` 对象（RGB 模式）

## 4. 目录结构

```
vagen/envs/yourenv/
├── __init__.py              # 空文件即可
├── yourenv_env.py           # 主环境类 + @dataclass Config
└── utils/
    ├── __init__.py
    ├── prompt.py            # system_prompt(), format_prompt(), init_observation_template(), action_template()
    └── utils.py             # parse_response(), numpy_to_pil() 等
```

## 5. Config Dataclass 模板

```python
@dataclass
class YourEnvConfig:
    # 环境特定参数
    grid_size: int = 8
    difficulty: str = "easy"

    # 通用参数（推荐保留）
    max_steps: int = 100
    render_mode: str = "text"           # "text" 或 "vision"
    max_actions_per_step: int = 3
    action_sep: str = ","
    image_placeholder: str = "<image>"
    use_example_in_sys_prompt: bool = True
    prompt_format: str = "free_think"   # "free_think", "wm", "free_wm"
    format_reward: float = 0.1
    success_reward: float = 1.0
```

## 6. Prompt 格式说明

### free_think 格式
```
<think>推理过程</think><answer>动作</answer>
```

### wm (World Model) 格式
```
<observation>观察描述</observation><think>推理过程</think><answer>动作</answer><prediction>预测下一步状态</prediction>
```

### free_wm 格式
与 wm 相同但允许标签之间有自由文本，没有 `<think>` 标签。

## 7. 注册环境

编辑 `vagen/configs/env_registry.yaml`：
```yaml
env_registry:
  Sokoban: vagen.envs.sokoban.sokoban_env.Sokoban
  FrozenLake: vagen.envs.frozenlake.frozenlake_env.FrozenLake
  YourEnv: vagen.envs.yourenv.yourenv_env.YourEnv   # 新增
```

## 8. 验证 Checklist

### 8.1 本地环境验证
```bash
# 测试环境创建和基本交互
conda run -n vagen python -c "
import asyncio
from vagen.envs.yourenv.yourenv_env import YourEnv

async def test():
    env = YourEnv({'render_mode': 'text'})

    # 1. system_prompt
    sys = await env.system_prompt()
    assert 'obs_str' in sys
    print('system_prompt OK')

    # 2. reset
    obs, info = await env.reset(seed=42)
    assert 'obs_str' in obs
    print('reset OK:', obs['obs_str'][:100])

    # 3. step (正确格式)
    obs, r, done, info = await env.step('<think>test</think><answer>YourAction</answer>')
    assert 'obs_str' in obs
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert 'success' in info
    print(f'step OK: reward={r}, done={done}')

    # 4. step (错误格式)
    obs, r, done, info = await env.step('garbage input')
    print(f'bad format OK: reward={r}')

    # 5. close
    await env.close()
    print('close OK')

asyncio.run(test())
"
```

### 8.2 注册表验证
```bash
conda run -n vagen python -c "
from vagen.envs.registry import get_env_cls, list_envs
print('Registered envs:', list_envs())
cls = get_env_cls('YourEnv')
print('Class loaded:', cls)
"
```

### 8.3 Vision 模式验证（如果支持）
```bash
conda run -n vagen python -c "
import asyncio
from vagen.envs.yourenv.yourenv_env import YourEnv

async def test():
    env = YourEnv({'render_mode': 'vision'})
    obs, info = await env.reset(seed=42)
    assert 'multi_modal_input' in obs
    imgs = obs['multi_modal_input']['<image>']
    assert len(imgs) > 0
    assert obs['obs_str'].count('<image>') == len(imgs)
    imgs[0].save('/tmp/test_yourenv.png')
    print('Vision mode OK, saved to /tmp/test_yourenv.png')
    await env.close()

asyncio.run(test())
"
```

## 9. API 端到端评估

使用 VAGEN 自带的评估框架 `vagen.evaluate.run_eval`，不需要自己写 workflow。

### 9.1 评估配置 YAML（参考 `tests/eval_sokoban_claude.yaml`）
```yaml
envs:
  - name: YourEnv
    n_envs: 3
    tag_id: yourenv_test
    seed: [0, 100]
    split: test
    max_turns: 5
    config:
      render_mode: vision
      prompt_format: free_think
    chat_config:
      temperature: 0
      max_tokens: 1024

experiment:
  dump_dir: ./rollouts/yourenv_eval
  default_max_turns: 5

run:
  backend: claude
  base_seed: 0
  max_concurrent_jobs: 2
  resume: off
  live_summary: true

backends:
  claude:
    api_key: ""            # 自动读 ANTHROPIC_API_KEY 环境变量
    base_url: null
    model: "claude-sonnet-4-20250514"
    max_concurrency: 2
    max_retries: 6
    min_backoff: 0.5
    max_backoff: 8.0
```

### 9.2 运行评估
```bash
cd /home/yaning/workspace/VAGEN
conda run -n vagen python -m vagen.evaluate.run_eval --config path/to/config.yaml
```

也可以通过命令行 override（注意 list 类型不适合 dotlist override，建议直接写 YAML）：
```bash
conda run -n vagen python -m vagen.evaluate.run_eval \
  --config tests/eval_sokoban_claude.yaml \
  run.backend=claude \
  run.max_concurrent_jobs=4
```

### 9.3 本地环境验证脚本
`tests/test_sokoban_api_e2e.py` — 不调用 API，仅验证 env 接口契约（system_prompt/reset/step/close/确定性）。

## 10. 评估框架核心流程

```
run_eval.py
  ├── 解析 YAML config → EnvSpec 列表
  ├── _expand_jobs() → 展开为具体 job (每个 seed 一个)
  └── run_eval_parallel()
        ├── REGISTRY.build_client(backend, cfg) → AsyncOpenAI / AsyncAnthropic
        ├── REGISTRY.build_adapter(backend, client, model) → OpenAIAdapter / ClaudeAdapter
        └── 并行执行 episodes:
              GenericVisionInferenceWorkflow.arun_episode()
                ├── env = env_cls(env_config)
                ├── env.reset(seed) → 初始观测
                ├── env.system_prompt() → 系统提示
                └── 循环 max_turns 次:
                      ├── adapter.acompletion(messages) → LLM 回复
                      ├── env.step(reply) → (obs, reward, done, info)
                      └── if done: break
```

## 11. 重要注意事项

1. **所有阻塞操作必须用 `asyncio.to_thread()` 包装**，否则会阻塞事件循环
2. **seed 必须保证确定性**：相同 seed 产生相同的初始状态
3. **`info["success"]`** 是必须的，评估框架和 wandb 日志都会用到
4. **图片占位符数量必须匹配**：`obs_str` 中的 `<image>` 数量 = 图片列表长度
5. **奖励设计**：`format_reward` 鼓励正确输出格式，`success_reward` 奖励任务完成
6. **parse_response 返回必须包含 `format_correct` 和 `actions` 字段**
7. Sokoban 参考实现在 `vagen/envs/sokoban/sokoban_env.py`（约 340 行，非常完整）
