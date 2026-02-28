# EB-ALFRED → GraphRL 移植可行性分析

## 概述

本文分析如何将 EB-ALFRED 环境从 VAGEN 移植到 GraphRL。GraphRL 在 VAGEN RL 训练之上增加了：
- 轨迹 → 知识图谱（NetworkX）转换
- 图谱 → SFT 数据集生成
- 迭代式 RL→SFT→RL 流水线
- LLaMA-Factory SFT 训练集成

Sokoban 是目前已完成移植的参考实现。本文先分析 Sokoban 的移植模式，再分析 EB-ALFRED 移植的适配工作。

---

## 第一部分：Sokoban 移植分析

### 1.1 文件结构

```
GraphRL/
├── graphrl/envs/sokoban_text/
│   ├── __init__.py            # 导入时触发注册
│   ├── vagen_graph_builder.py # 轨迹 → 图谱节点/边
│   ├── traj_to_sft.py         # 图谱 → SFT 数据集
│   └── utils/
│       ├── sft_generators.py  # 四种数据集生成逻辑
│       └── parse_rollout.py   # 解析辅助函数
└── examples/sokoban_text/
    ├── pipeline.yaml          # 完整流水线配置
    ├── run.sh                 # 启动脚本
    ├── train.yaml             # VAGEN 训练数据集配置
    └── val.yaml               # VAGEN 验证数据集配置
```

env_registry.yaml 无需修改（直接复用 VAGEN 内置的 `Sokoban` 注册名）。

### 1.2 核心抽象层

GraphRL 提供两个抽象基类，只需实现一个方法：

| 基类 | 需要实现的方法 | 注册表 |
|------|--------------|--------|
| `VagenGraphBuilderNetworkx` | `traj_to_transitions()` | `graph_builder_registry` |
| `BaseGraphSFTGenerator` | `generate_datasets()` | `sft_generator_registry` |

注册机制：在 `__init__.py` 中导入类即触发 `@graph_builder_registry.register("sokoban_text")` 装饰器。

### 1.3 Graph Builder（`vagen_graph_builder.py`）

**输入**：VAGEN RL 训练产生的 ChatML JSONL 文件，每行一个 episode：
```json
{"input": "<|im_start|>user\n...<|im_end|>", "output": "<|im_start|>assistant\n...<|im_end|>"}
```

**父类 `_parse_vagen_line()` 的解析**：
```python
full = (data["input"] + data["output"]).replace("<|endoftext|>", "")
pattern = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
messages = [{"role": role, "content": content} for role, content in pattern.findall(full)]
```

**`traj_to_transitions()` 实现**：遍历 messages，用正则提取：
- `user` 消息中的网格状态（`[Initial Observation]` 或 `After that, the observation is:`）
- `assistant` 消息中的动作（`<answer>...</answer>`）

**输出**：`List[Tuple[VagenNodeData, VagenEdgeData, VagenNodeData]]`
- 每个 tuple = `(src_state, action, dst_state)`
- `VagenNodeData(state=grid_text, obs_str=grid_text)`：state 和 obs_str 相同（纯文本）
- `VagenEdgeData(obs_str=action_text)`

**节点去重**（父类自动处理）：
- `unique_key()` = `sha256(state)[:16]`：相同网格字符串 → 相同节点
- 跨 episode 的相同状态自动合并

**容错处理**：若某步 assistant 未输出合法 `<answer>`，跳过该 transition 但继续追踪后续状态（不错位）。

### 1.4 SFT Generator（`traj_to_sft.py`）

从 `graph.json` 生成四种 LLaMA-Factory ShareGPT 数据集：

| 数据集 | 任务类型 | 示例 |
|--------|---------|------|
| `forward_dynamics_direct` | `(state, actions)` → `<prediction>final_state</prediction>` | 预测 N 步后的状态 |
| `forward_dynamics_mcq` | `(state, actions, 4个选项)` → 正确字母 A-D | 多选题式预测 |
| `inverse_dynamics` | `(state, next_state)` → action sequence | 给定前后状态推动作 |
| `state_reachable` | 多轮对话导航 s0→sN | 多步路径规划 |

数据来源：通过 `graph.sample_paths()` 从图中随机采样路径。

### 1.5 Pipeline 配置（`examples/sokoban_text/pipeline.yaml`）

关键配置项：

```yaml
env_module: graphrl.envs.sokoban_text      # 触发 __init__.py 注册

general_overrides:
  rl:
    graph_builder: sokoban_text            # 使用注册的 builder
    vagen_dir: VAGEN                       # VAGEN 根目录

  traj_to_sft:
    generator: sokoban_text               # 使用注册的 generator

  sft:
    hydra_overrides:
      dataset: forward_dynamics_direct,forward_dynamics_mcq,inverse_dynamics,state_reachable
      template: qwen                      # 纯文本模板
```

### 1.6 Sokoban 移植总结

Sokoban 移植工作量极小，因为：
1. 纯文本状态，`state == obs_str`，节点去重简单
2. 无需图像处理
3. 直接 VAGEN 环境类（无 HTTP/进程隔离）
4. 4 种 SFT 数据集均为文本 ShareGPT 格式

---

## 第二部分：EB-ALFRED 移植分析

### 2.1 与 Sokoban 的关键差异

| 特性 | Sokoban | EB-ALFRED |
|------|---------|-----------|
| 观察类型 | 纯文本（网格字符串） | 多模态（RGB 图像 + 文本） |
| 状态去重 | 文本精确匹配（sha256） | 需要图像感知哈希或位置标识 |
| 环境架构 | 直接 Python 类 | HTTP RemoteEnv（Python 3.9 服务器） |
| Python 版本 | vagen env（3.10）可直接运行 | 需要 embench env（3.9）单独运行服务器 |
| 动作空间 | 固定（上下左右） | 每 episode 动态（162+ 动作，含多实例） |
| 单 episode 长度 | 短（~10 步） | 最多 30 步 |
| SFT 数据集 | 纯文本 | 多模态（含图像路径） |
| LLaMA-Factory 模板 | `qwen`（文本） | `qwen2_vl`（多模态） |

### 2.2 架构：RemoteEnv 对 GraphRL 的影响

**Sokoban 流程**：
```
GraphRL（Python 3.10）
  └─ VagenRLModule 启动 VAGEN 子进程
       └─ VAGEN 直接实例化 Sokoban 环境类
```

**EB-ALFRED 流程**：
```
GraphRL（Python 3.10）
  └─ VagenRLModule 启动 VAGEN 子进程
       └─ VAGEN 使用 RemoteEnv，通过 HTTP 连接
            ↓
       embench server（Python 3.9，独立进程）
         DISPLAY=:0 python -m vagen.envs.eb_alfred.serve --port 8000
```

**影响**：
- GraphRL 本身无需修改——VAGEN 的 RemoteEnv 已处理跨进程通信
- 运行 RL 训练前必须手动启动 embench server
- `run.sh` 脚本需要额外说明（或自动启动 server）
- VAGEN 的 `env_registry.yaml` 已有 `RemoteEnv` 注册项，只需在 `train.yaml`/`val.yaml` 中配置 `base_urls`

### 2.3 RL 训练期间的图像存储机制

VAGEN RL 训练产生的 rollout JSONL 格式（ChatML）：
- 文本内容内联在 `<|im_start|>...<|im_end\|>` 块中
- 图像作为特殊 token 或路径引用，具体格式取决于 VAGEN 的多模态 rollout 实现

根据 VAGEN 评估 rollout（`messages.json`）的格式：
- 每轮用户消息包含 `image_url: {url: "<data_url>"}` 占位符
- 实际图像存储在 `images/turn_XX_YY.png`
- obs_str 中使用 `<image>` 占位符

**Graph Builder 适配**：图像路径从 `rollout_dir/images/turn_{step:02d}_01.png` 定位，通过 `VagenNodeData.source_images` 字段传入，框架自动复制到 `graph/images/` 目录。

### 2.4 节点去重策略

Sokoban 的默认 `sha256(state_text)[:16]` 对 EB-ALFRED 不够直接，因为：
- 图像数据不能嵌入文本 state 字段
- 两个不同 episode 可能恰好到达相同场景状态（有价值的去重）
- 同一 episode 的重复状态（循环）应被识别

**推荐策略**：自定义 `AlfredNodeData`，以 `(obs_text, image_phash)` 为 state key：

```python
class AlfredNodeData(VagenNodeData):
    """
    state = {
        "obs_text": "[Last Action]: ...\n[Feedback]: ...",
        "image_phash": "abc123def456..."    # 感知哈希，8x8 DCT
    }
    source_images = ["/path/to/rollout/images/turn_05_01.png"]
    obs_str = "[Last Action]: ...\n[Feedback]: ...\n<image>"
    """

    def unique_key(self) -> str:
        # 由 obs_text + image_phash 联合确定唯一性
        return VagenNodeData.state_to_id(
            f"{self.state['obs_text']}|{self.state['image_phash']}"
        )

    def bucket_key(self) -> str:
        # 按图像哈希聚桶，快速找到视觉相似节点
        return self.state["image_phash"][:8]

    def is_similar_to(self, other: "NodeData") -> bool:
        if not isinstance(other, AlfredNodeData):
            return False
        # 同一视觉场景（哈希前缀相同）且相同动作反馈
        return (
            self.state["image_phash"][:8] == other.state["image_phash"][:8]
            and self.state["obs_text"] == other.state["obs_text"]
        )
```

**感知哈希计算**（使用 `imagehash` 库）：
```python
import imagehash
phash = str(imagehash.phash(Image.open(image_path), hash_size=8))
```

### 2.5 Graph Builder 实现要点

```python
@graph_builder_registry.register("eb_alfred")
class EbAlfredGraphBuilder(VagenGraphBuilderNetworkx):

    def traj_to_transitions(self, messages, rollout_dir, step_idx, line_idx):
        transitions = []
        current_node = None
        pending_action = None
        turn_idx = 0

        for msg in messages:
            role, content = msg["role"], msg["content"]

            if role == "user":
                # 从 user 消息提取图像路径（文件名模式：turn_XX_01.png）
                turn_idx += 1
                img_path = rollout_dir / "images" / f"turn_{turn_idx:02d}_01.png"
                obs_text = _extract_obs_text(content)  # 去除 <image> 占位符后的文本

                if img_path.exists():
                    phash = _compute_phash(img_path)
                    node = AlfredNodeData(
                        state={"obs_text": obs_text, "image_phash": phash},
                        obs_str=obs_text + "\n<image>",
                        source_images=[str(img_path)],
                    )

                    if current_node is not None and pending_action is not None:
                        transitions.append((current_node, VagenEdgeData(obs_str=pending_action), node))

                    current_node = node
                    pending_action = None

            elif role == "assistant":
                action = _extract_answer(content)   # <answer>...</answer>
                if action:
                    pending_action = action

        return transitions
```

**图像路径解析备注**：
- VAGEN 评估 rollout 中图像按 `turn_{turn_idx:02d}_{img_idx:02d}.png` 存储
- RL 训练 rollout 的图像存储位置需验证（可能在 `rollout_dir` 的相对路径下，也可能嵌入 JSONL）
- 若 RL 训练不存储图像文件，需修改 VAGEN RL rollout worker 以保存图像

### 2.6 SFT 数据集设计

EB-ALFRED 适合以下多模态 SFT 数据集类型：

| 数据集 | 输入 | 输出 | 说明 |
|--------|------|------|------|
| `inverse_dynamics` | 图像 A + 图像 B | 动作名称 | 最直接：从视觉变化推断动作 |
| `forward_dynamics_direct` | 图像 + 动作 | 文字反馈预测 | 预测动作执行后的环境反馈 |
| `forward_dynamics_visual` | 图像 + 动作 | 下一帧图像 | 视觉预测（生成任务，难度高） |
| `goal_conditioned` | 图像 + 任务指令 | 下一步动作 | 类似 RL 策略，但用离线数据蒸馏 |

**推荐优先实现**：`inverse_dynamics` 和 `forward_dynamics_direct`，因为：
- 前者天然适合图形结构（每条边即一个训练样本）
- 后者可用文字输出，避免图像生成的复杂性

**ShareGPT 格式示例**（`inverse_dynamics`）：
```json
{
  "messages": [
    {"role": "system", "content": "You are a robot in a home environment..."},
    {"role": "user", "content": "Given these two observations, what action was taken?\n\nBefore: <image>\nAfter: <image>"},
    {"role": "assistant", "content": "<answer>find a Plate</answer>"}
  ],
  "images": ["graph/images/abc123_0.png", "graph/images/def456_0.png"]
}
```

### 2.7 需要新增的文件

```
graphrl/envs/eb_alfred/
├── __init__.py                # 注册 eb_alfred builder 和 generator
├── vagen_graph_builder.py     # AlfredNodeData + EbAlfredGraphBuilder
├── traj_to_sft.py             # EbAlfredSFTGenerator
└── utils/
    ├── sft_generators.py      # inverse_dynamics 等生成逻辑
    └── parse_rollout.py       # obs_text/action 提取函数

examples/eb_alfred/
├── pipeline.yaml              # 流水线配置（含 RemoteEnv 设置）
├── run.sh                     # 需额外注明：先启动 embench server
├── train.yaml                 # VAGEN 训练数据集配置
└── val.yaml                   # VAGEN 验证数据集配置
```

### 2.8 `pipeline.yaml` 关键配置差异

与 Sokoban 相比，EB-ALFRED 的 pipeline.yaml 需要以下差异：

```yaml
env_module: graphrl.envs.eb_alfred

general_overrides:
  rl:
    graph_builder: eb_alfred
    vagen_dir: VAGEN
    hydra_overrides:
      data:
        train_files: null   # 注入 eb_alfred train.yaml

      actor_rollout_ref:
        rollout:
          # 多模态模型需要更多显存
          gpu_memory_utilization: 0.7
          max_num_batched_tokens: 8192  # 图像占用大量 token

  traj_to_sft:
    generator: eb_alfred
    eb_alfred:
      generators:
        - inverse_dynamics
        - forward_dynamics_direct
      inverse_dynamics:
        num_samples: 1000
      seed: 42

  sft:
    hydra_overrides:
      dataset: inverse_dynamics,forward_dynamics_direct
      template: qwen2_vl          # 必须改为多模态模板！（Sokoban 用 qwen）
      cutoff_len: 16384            # 图像 token 数量大，需更长上下文
```

**`train.yaml`** 需配置 `RemoteEnv`：
```yaml
envs:
  - name: RemoteEnv
    config:
      base_urls:
        - "http://localhost:8000"   # embench server 地址
      eval_set: base
      x_display: "0"
      max_turns: 30
      max_actions_per_step: 1
      prompt_format: free_think
```

### 2.9 env_registry.yaml

GraphRL 的 `env_registry.yaml` 已包含 `RemoteEnv`（通过继承 VAGEN 的 env_registry）。若需直接使用 `EbAlfred` 类（在同一 Python 环境中有 embench），可添加：

```yaml
# graphrl/configs/rl/vagen/env_registry.yaml 新增
EbAlfred: vagen.envs.eb_alfred.eb_alfred_env.EbAlfred
```

但推荐在 RL 训练中使用 `RemoteEnv` 保持进程隔离。

---

## 第三部分：移植步骤与可行性评估

### 3.1 移植步骤（按优先级）

**Phase A：基础移植（约 2-3 天）**

1. 验证 VAGEN RL 训练 rollout 的图像存储格式
   - 运行一次小规模 RL 训练（sokoban 或 eb_alfred），检查 rollout JSONL 文件格式
   - 确认图像是否存储为文件或嵌入 JSONL

2. 创建 `graphrl/envs/eb_alfred/__init__.py`
   - 复制 sokoban_text 结构

3. 实现 `vagen_graph_builder.py`
   - 实现 `AlfredNodeData`（phash 去重）
   - 实现 `EbAlfredGraphBuilder.traj_to_transitions()`
   - 单元测试：用 evaluation rollout（messages.json）验证 transition 提取

4. 实现 `traj_to_sft.py`（先实现 `inverse_dynamics`）
   - 每条边 = 一个训练样本
   - 单元测试：从少量 transitions 生成数据集并验证格式

**Phase B：集成调试（约 1-2 天）**

5. 创建 `examples/eb_alfred/pipeline.yaml` 和配套文件

6. 运行端到端测试（小规模）：
   ```bash
   # 终端 1（embench env）：
   DISPLAY=:0 python -m vagen.envs.eb_alfred.serve --port 8000

   # 终端 2（vagen env + GraphRL）：
   cd /home/march/workspace/Yaning/GraphRL
   python -m graphrl.main --config-path examples/eb_alfred \
     --config-name pipeline \
     experiment_dir=exps/eb_alfred_test \
     iterations=1
   ```

**Phase C：SFT 扩展（可选）**

7. 添加 `forward_dynamics_direct` 数据集
8. 调优 cutoff_len 和 batch size

### 3.2 潜在风险

| 风险 | 严重性 | 缓解方案 |
|------|--------|---------|
| VAGEN RL 训练 rollout 不存储图像文件 | 高 | 修改 VAGEN rollout worker 保存图像；或放弃图像，仅用文本 obs_str |
| `imagehash` 库在 vagen env 未安装 | 低 | `pip install imagehash` |
| 多模态 rollout ChatML 格式与文本不同 | 中 | 重写 `_parse_vagen_line()` 处理多模态 token |
| embench server 在 RL 训练中途崩溃 | 中 | 添加自动重启逻辑；VAGEN RemoteEnv 有重连机制 |
| 图像 token 数量大导致 OOM | 中 | 降低 `gpu_memory_utilization`，减小 batch size |
| SFT 模板选错（用 qwen 而非 qwen2_vl） | 高 | 检查 LLaMA-Factory 支持的多模态模板列表 |

### 3.3 可行性结论

**结论：移植可行，但有一个关键不确定性。**

核心不确定性：**VAGEN RL 训练的 rollout JSONL 文件是否保存图像**。

- 若保存图像（如 `images/turn_XX.png`）：移植工作直接，7 天内可完成 Phase A+B
- 若不保存图像：需先修改 VAGEN 的 rollout 保存逻辑，或接受"纯文本图谱"（仅用 obs_str 文本，放弃图像）

**验证方法**（优先执行）：
```bash
# 在 vagen env 中运行一次 sokoban RL 训练（1分钟），检查 rollout 格式
cd /home/march/workspace/Yaning/GraphRL
python -m graphrl.main ... iterations=1  # 极小规模
# 然后检查 exps/*/iter_0/rl/rollouts/*.jsonl 的格式
```

GraphRL 框架本身对 EB-ALFRED 完全透明——`VagenGraphBuilderNetworkx` 的图像支持（`source_images` 字段、自动复制）已经内置。EB-ALFRED 移植只需实现约 200-300 行 Python 代码（graph builder + SFT generator）。

---

## 附录：关键文件路径速查

| 文件 | 说明 |
|------|------|
| [graphrl/envs/sokoban_text/vagen_graph_builder.py](../../GraphRL/graphrl/envs/sokoban_text/vagen_graph_builder.py) | Sokoban graph builder 参考实现 |
| [graphrl/envs/sokoban_text/traj_to_sft.py](../../GraphRL/graphrl/envs/sokoban_text/traj_to_sft.py) | Sokoban SFT generator 参考实现 |
| [graphrl/modules/rl/vagen/base/vagen_graph_builder_networkx.py](../../GraphRL/graphrl/modules/rl/vagen/base/vagen_graph_builder_networkx.py) | 父类：VagenNodeData/VagenEdgeData/VagenGraphBuilderNetworkx |
| [graphrl/configs/rl/vagen/env_registry.yaml](../../GraphRL/graphrl/configs/rl/vagen/env_registry.yaml) | GraphRL 环境注册表 |
| [examples/sokoban_text/pipeline.yaml](../../GraphRL/examples/sokoban_text/pipeline.yaml) | Sokoban 完整流水线配置（参考模板） |
| [vagen/envs/eb_alfred/eb_alfred_env.py](../vagen/envs/eb_alfred/eb_alfred_env.py) | VAGEN EB-ALFRED 适配器（已完成） |
| [tests/eval_eb_alfred_claude.yaml](../tests/eval_eb_alfred_claude.yaml) | Claude API 评估配置（已验证） |
