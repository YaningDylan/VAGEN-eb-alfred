# PPO Training Issues Log

训练脚本：`examples/eb_alfred/train_ppo_no_concat_qwen25vl3b.sh`

---

## Issue 1: `data.max_trajectory_length` not in Hydra struct

**错误信息：**
```
Could not override 'data.max_trajectory_length'.
To append to your config use +data.max_trajectory_length=7000
Key 'max_trajectory_length' is not in struct
```

**原因：** `max_trajectory_length` 不在 `legacy_data.yaml` 的 struct 中，不能直接覆盖，需要用 `+` 前缀来新增 key。

**解决：** 训练脚本中将 `data.max_trajectory_length=7000` 改为 `+data.max_trajectory_length=7000`。

---

## Issue 2: `freeze_vision_tower` already in actor struct

**错误信息：**
```
An item is already at 'actor_rollout_ref.actor.freeze_vision_tower'.
Either remove + prefix: 'actor_rollout_ref.actor.freeze_vision_tower=True'
```

**原因：** `freeze_vision_tower` 已经定义在 `verl/trainer/config/actor/actor.yaml` 中（值为 false），因此不需要 `+` 前缀。

**解决：** 训练脚本中将 `+actor_rollout_ref.actor.freeze_vision_tower=True` 改为 `actor_rollout_ref.actor.freeze_vision_tower=True`（去掉 `+`）。

---

## Issue 3: `train_batch_size=50` 不能整除 GPU 数 8

**错误信息：**
```
AssertionError: real_train_batch_size (50) must be divisible by minimal possible batch size (8)
```

**原因：** verl 要求 `train_batch_size × rollout.n` 必须能被 `n_gpus_per_node (8)` 整除。50 mod 8 = 2，不满足条件。

**解决：** 将训练脚本中的 `data.train_batch_size=50` 改为 `data.train_batch_size=48`（48 ÷ 8 = 6，满足整除要求）。

---

## Issue 4: `FSDPCriticModelCfg` 缺少 `freeze_vision_tower` 字段

**错误信息：**
```
TypeError: FSDPCriticModelCfg.__init__() got an unexpected keyword argument 'freeze_vision_tower'
```

**原因：** critic 的模型配置 dataclass `FSDPCriticModelCfg`（位于 `verl/workers/config/critic.py`）没有定义 `freeze_vision_tower` 字段，但 fsdp_workers.py 中的 critic 初始化代码支持它（通过 `.get()` 访问）。Hydra 在实例化 dataclass 时传入了该 key，导致报错。

**解决：** 在 `verl/workers/config/critic.py` 的 `FSDPCriticModelCfg` dataclass 中添加 `freeze_vision_tower: bool = False` 字段。

---

## Issue 5: `transformers 4.57.1` 中 `additional_chat_templates` 404 未处理

**错误信息：**
```
huggingface_hub.errors.RemoteEntryNotFoundError: 404 Client Error.
Entry Not Found for url: .../Qwen2.5-VL-3B-Instruct/tree/main/additional_chat_templates
```

**原因：** transformers 4.57.1 在加载 tokenizer 时会尝试从 HuggingFace 获取 `additional_chat_templates` 文件夹。对于不存在该文件夹的模型（如 Qwen2.5-VL-3B），HF Hub 返回 404，但 `list_repo_templates()` 函数的 except 中只捕获了 `GatedRepoError`、`RepositoryNotFoundError`、`RevisionNotFoundError`，没有捕获 `EntryNotFoundError`（`RemoteEntryNotFoundError` 的父类）。

**解决：** 在 `/venv/vagen/lib/python3.10/site-packages/transformers/utils/hub.py` 的 `list_repo_templates()` 函数的 except 中添加：
```python
except EntryNotFoundError:
    return []  # additional_chat_templates dir doesn't exist => no additional templates
```

---

## Issue 6: `trl 0.29.0` 缺少 `AutoModelForCausalLMWithValueHead`

**错误信息：**
```
ImportError: cannot import name 'AutoModelForCausalLMWithValueHead' from 'trl'
```

**原因：** trl 0.29.0（当前最新版）已移除 `AutoModelForCausalLMWithValueHead`，该类在旧版本（≤0.11.x）中可用。verl 的 `load_valuehead_model` 使用此类来为 critic 添加 value head。

**解决：** 降级 trl：`pip install "trl<0.12"`（安装了 trl 0.11.4）。

---

## Issue 7: `flash_attn` 未安装，无法使用 `flash_attention_2`

**错误信息：**
```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error:
the package flash_attn seems to be not installed.
```

**原因：** verl/trl 代码中多处硬编码 `attn_implementation="flash_attention_2"`，但 flash-attn 包未安装。当前环境 torch 2.9.1 + CUDA 12.8，flash-attn 没有预编译 wheel，从源码编译失败。

**解决（两步）：**
1. 在训练脚本中添加 `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` 和 `+critic.model.override_config.attn_implementation=sdpa`（需要 `+` 前缀因为 `override_config` 是空 struct），通过 override_config 覆盖 attention 实现为 PyTorch 内置 SDPA。
2. 修改 `verl/verl/utils/model.py` 的 `load_valuehead_model` 函数，使用 `model_config._attn_implementation` 替换硬编码的 `"flash_attention_2"`：
   ```python
   attn_implementation = getattr(model_config, "_attn_implementation", "flash_attention_2")
   ```

---

## Issue 8: `cachetools` 模块未安装

**错误信息：**
```
ModuleNotFoundError: No module named 'cachetools'
  File ".../vagen/agent_loop/agent_loop_no_concat.py", line 26, in <module>
    from cachetools import LRUCache
```

**原因：** vagen 环境缺少 `cachetools` 包，`agent_loop_no_concat.py` 依赖它。

**解决：** 安装：`pip install cachetools`

---

## Issue 9: sglang API 版本不兼容 (`_launch_subprocesses`)

**错误信息：**
```
TypeError: _launch_subprocesses() missing 3 required positional arguments:
'init_tokenizer_manager_func', 'run_scheduler_process_func', and 'run_detokenizer_process_func'
```

**原因：** verl 的 `async_sglang_server.py` 使用了 `sglang.srt.entrypoints.engine._launch_subprocesses(server_args=server_args)` 的调用方式，但 sglang 0.5.9 改变了该函数签名，增加了三个必填参数。verl `setup.py` 中指定 `sglang[srt,openai]==0.5.5`，但安装时安装了最新版 0.5.9。

**解决：** 降级 sglang 到指定版本：`pip install "sglang[srt,openai]==0.5.5"`

（注意：此操作会同时将 torch 从 2.9.1 降到 2.8.0）

---

## Issue 10: `setuptools 82` 移除了 `pkg_resources`，导致 `gym_sokoban` 无法导入

**错误信息：**
```
ModuleNotFoundError: No module named 'pkg_resources'
  File ".../vagen/envs/sokoban/sokoban_env.py", line 5, in <module>
    from .patch_sokoban_env import PatchedSokobanEnv as SokobanEnv
  File ".../vagen/envs/sokoban/patch_sokoban_env.py", line 1, in <module>
    from gym_sokoban.envs.sokoban_env import SokobanEnv
  File ".../gym_sokoban/__init__.py", line 2, in <module>
    import pkg_resources
```

**原因：** `setuptools >= 70` 将 `pkg_resources` 作为独立包移出，不再自动包含在 `setuptools` 包中。verl 安装脚本安装了 setuptools 82.0.0，不含 `pkg_resources`，但 `gym_sokoban` 的 `__init__.py` 依赖 `import pkg_resources`。

**解决：** 降级 setuptools 到 < 70：
```
/venv/vagen/bin/pip install "setuptools<70"
```
安装了 setuptools 69.5.1，其中包含 `pkg_resources`。

---

## Issue 11: `python3` 使用系统 Python 而非 vagen 虚拟环境

**错误信息：**
```
ModuleNotFoundError: No module named 'hydra'
```

**原因：** 训练脚本直接调用 `python3`，解析到系统的 `/usr/bin/python3`（Python 3.12），而所有依赖都安装在 `/venv/vagen/`（Python 3.10）中。该虚拟环境没有 `activate` 脚本，无法用 `source activate` 激活。

**解决：** 在训练脚本中通过 `export PATH` 将 vagen 的 bin 目录优先放入 PATH：
```bash
export PATH=/venv/vagen/bin:${PATH}
```
加在 `PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo` 之前。

---

## Issue 12: `vagen.envs.eb_navigation` 模块不存在

**错误信息：**
```
ModuleNotFoundError: No module named 'vagen.envs.eb_navigation'
hydra.errors.InstantiationException: Error in call to target 'vagen.agent_loop.gym_agent_loop_no_concat.GymAgentLoop'
```

**原因：** `vagen/configs/env_registry.yaml` 中注册了 `EbNavigation: vagen.envs.eb_navigation.eb_navigation_env.EbNavigation`，但 `vagen/envs/` 目录下并不存在 `eb_navigation` 子目录/模块。`GymAgentLoop` 初始化时会遍历注册表并 `importlib.import_module` 所有环境，因此会报错。

**解决：** 从 `vagen/configs/env_registry.yaml` 中删除 `EbNavigation` 这一行。

---

## Issue 13: sglang 子进程加载 tokenizer 时触发 HuggingFace API 429 限流

**错误信息：**
```
huggingface_hub.errors.HfHubHTTPError: 429 Too Many Requests: you have reached your 'api' rate limit.
  File ".../transformers/utils/hub.py", line 167, in list_repo_templates
httpx.HTTPStatusError: Client error '429 Too Many Requests' for url
  'https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct/tree/main/additional_chat_templates?...'
```

**原因：** transformers `list_repo_templates()` 调用 HuggingFace API 查询 `additional_chat_templates` 目录，多次训练重启后触发 IP 级别的 API 限流（429）。原有的 except 子句捕获了 `requests.exceptions.HTTPError`，但 `HfHubHTTPError` 是 `huggingface_hub` 自己的异常类，并非 `requests.HTTPError` 的子类，因此未被捕获。

**解决：** 在 `/venv/vagen/lib/python3.10/site-packages/transformers/utils/hub.py` 的 `list_repo_templates()` 的 except 中添加对 `HfHubHTTPError` 的处理：
```python
except HfHubHTTPError:
    pass  # rate limit (429) or other HF API errors => try local files
```
（`HfHubHTTPError` 已在该文件顶部从 `huggingface_hub.utils` 导入，无需额外 import。）

---
