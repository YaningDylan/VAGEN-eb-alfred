# ERA Repo 对比问题与调查

对比 VAGEN-eb-alfred 和原始 ERA (Embodied-Reasoning-Agent) repo 的关键差异。

---

## Q1: 原 ERA repo 的图片像素是多少？

### 调查结果

ERA repo 中涉及图片像素的地方有两层：**环境采集层** 和 **模型处理层**。

#### 环境采集层（AI2-THOR 渲染分辨率）

| 环境 | 默认分辨率 | 文件 |
|------|-----------|------|
| EB-Alfred | **500 x 500** | `eval/EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py` (L97) |
| EB-Navigation | **500 x 500** | `eval/EmbodiedBench/embodiedbench/envs/eb_navigation/EBNavEnv.py` (L43) |
| EB-Manipulation | **500 x 500** | `eval/EmbodiedBench/embodiedbench/envs/eb_manipulation/EBManEnv.py` (L61) |
| EB-Habitat | **500 x 500** | `eval/EmbodiedBench/embodiedbench/envs/eb_habitat/EBHabEnv.py` (L112) |

所有环境的默认渲染分辨率均为 **500 x 500**。

#### 模型处理层（Qwen2.5-VL processor 的 min/max pixels）

| 用途 | min_pixels | max_pixels | 文件 |
|------|-----------|-----------|------|
| **SFT 训练** | 500 * 500 = 250,000 | 500 * 500 = 250,000 | `ERA-sft/epl/train.py` (L229-233) |
| **RL 数据集** | 512 * 512 = 262,144 | 2048 * 2048 = 4,194,304 | `ERA-rl/verl/verl/utils/dataset/rl_dataset.py` (L51-72) |

**关键发现**：
- SFT 训练时，图片被固定为 250,000 像素（等效 500x500），即 min = max = 500*500。
- RL 训练的 dataset loader 中，min/max pixels 范围更大（512x512 ~ 2048x2048），但实际环境输出就是 500x500。
- SFT 代码中有注释掉的 480p / 720p / 1080p 配置，但未使用。

### VAGEN-eb-alfred 的情况

- 环境渲染分辨率：**500 x 500**（`vagen/envs/eb_alfred/eb_alfred_env.py` L37）
- 可选 resize：`obs_image_size` 参数可将 obs 缩放到指定尺寸（如 96x96、328x328），默认为 `None`（不缩放）。

### 结论

> **环境分辨率一致**（均为 500x500）。但 VAGEN-eb-alfred 额外提供了 `obs_image_size` 选项，可将观测图缩放；ERA 不做额外缩放。
>
> **需确认：** VAGEN 在训练时的 Qwen2.5-VL processor 的 min_pixels / max_pixels 设了多少？是否和 ERA SFT 的 500*500 固定设置一致？如果不一致，可能导致模型 vision encoder 的输入差异。

---

## Q2: 原 ERA repo 的 multi-turn 格式是否是 non-concat？每一轮 model 会不会接收上几个 turn 的 image 或 text？

### 调查结果

ERA repo 使用的是 **concat（拼接）格式**，不是 non-concat。

#### ERA 的 multi-turn 实现

核心逻辑在 `ERA-rl/VAGEN/vagen/mllm_agent/rollout.py`（L300-350）：

```
Chat 格式: System -> |InitUser| -> |Assistant, User| -> |Assistant, User| -> ... -> |Assistant, User Final|
```

具体行为：
1. **每一轮都会包含之前 turn 的 text 和 image**（在 window 范围内）
2. 使用 **sliding window** 控制历史长度，由 `window_size` 参数控制
3. 对于每一步，会把 `recording[start_step : end_step + 1]` 的历史拼成完整对话

#### Window Size 设置

| 阶段 | window_size | 说明 |
|------|-------------|------|
| **Rollout（推理生成）** | 配置值（如 `window_size: 1`）| 只保留最近几轮作为 context |
| **Update（训练更新）** | `None`（全部历史） | 把完整轨迹用于 loss 计算 |

ERA alfred 训练脚本 (`ERA-rl/VAGEN/vagen/examples/alfred/run.sh`) 中：
- `rollout_manager.window_size=1` — **rollout 时只看当前轮（最近 1 步）**
- `rollout_manager.max_turns=30`
- `rollout_manager.use_multi_turn_reward=True`

#### 每一轮是否接收上几轮的 image / text？

**取决于 window_size**：
- `window_size=1`：model 在 rollout 时 **只看当前轮的 image 和 text**（不包含历史）
- `window_size=5`：model 看最近 5 轮的 image + text
- `window_size=None`：看所有历史

**在 update（梯度更新）阶段**，无论 rollout 时的 window_size 是什么，ERA 都会用 `window_size=None`（完整历史）来做 loss 计算。

### VAGEN-eb-alfred 的情况

VAGEN-eb-alfred 提供了两种模式：
1. **Concat 模式**（默认）：`vagen/agent_loop/gym_agent_loop.py`，对话历史会累积拼接
2. **Non-concat 模式**：`vagen/agent_loop/gym_agent_loop_no_concat.py`，使用 `algorithm.adv_estimator=no_concat_gae_first`

### 结论

> **ERA 使用 concat 格式**，且在 Alfred 训练中 `window_size=1`（rollout 只看当前 step）。
>
> **需确认：** VAGEN-eb-alfred 实际训练 EB-Alfred 时用的是 concat 还是 non-concat？如果是 concat，window_size 设了多少？这会显著影响 model 在每一步能看到的信息量。

---

## Q3: 目前无论是 inference 还是 training，我们提供的参数是否和原来的 ERA repo 保持一致？

### 参数对比

#### Training 参数

| 参数 | ERA (alfred/run.sh) | VAGEN (vagen_multiturn.yaml + PPO 脚本) | 一致？ |
|------|---------------------|------------------------------------------|--------|
| **algorithm** | `masked_gae` (PPO) | `gae` (PPO) / `grpo` (GRPO) | 不同 |
| **gamma** | 0.99 | 1.0 | 不同 |
| **lambda (GAE)** | 0.99 | 1.0 | 不同 |
| **actor LR** | 1e-6 | 1e-6 | 一致 |
| **critic LR** | 1e-5 | 1e-5 | 一致 |
| **train_batch_size** | 50 | 128 / 256 | 不同 |
| **ppo_mini_batch_size** | 16 | 32 | 不同 |
| **max_prompt_length** | 2048 | 1000 / 9000 | 不同 |
| **max_response_length** | 512 | 4000 / 8000 | 不同 |
| **max_trajectory_length** | 7000 | 未明确设置 | 不同 |
| **max_turns** | 30 | 取决于 env config | 需确认 |
| **window_size** | 1 | 未在脚本中显式设置 | 需确认 |
| **use_multi_turn_reward** | True | 未在脚本中显式设置 | 需确认 |
| **use_loss_mask** | True | 未在脚本中显式设置 | 需确认 |
| **use_gae_mask** | True | 未在脚本中显式设置 | 需确认 |
| **reward_scaling** | 未设 | 10.0 | 不同 |
| **reward_bias** | 未设 | -0.5 | 不同 |
| **reward_clip** | 未设 | 20.0 | 不同 |
| **freeze_vision_tower** | True | 未设 | 需确认 |
| **entropy_coeff** | 未设（default） | 0.0 | 需确认 |
| **KL loss** | False (coef=0.001) | False (coef=0.0) | 基本一致 |
| **rollout engine** | vllm | sglang | 不同 |
| **rollout n** | 1 | 1 (PPO) / 8 (GRPO) | 取决于算法 |
| **critic_warmup** | 3 | 0 | 不同 |

#### Inference / Evaluation 参数

| 参数 | ERA (alfred/run.sh) | VAGEN (eval config) | 一致？ |
|------|---------------------|---------------------|--------|
| **eval temperature** | 0.01 | 0 | 接近 |
| **training temperature** | 0.5 | 未在 alfred 脚本中设 | 需确认 |
| **top_p (training)** | 0.9 | 未在 alfred 脚本中设 | 需确认 |
| **top_k** | -1 | 未设 | 需确认 |
| **max_tokens (eval)** | 未明确 | 1024 | 需确认 |

#### 环境 / Reward 参数

| 参数 | ERA | VAGEN-eb-alfred | 一致？ |
|------|-----|-----------------|--------|
| **resolution** | 500 | 500 | 一致 |
| **max_turns** | 30 | 30 | 一致 |
| **max_actions_per_step** | 未确认 | 1 | 需确认 |
| **prompt_format** | 未确认 | free_think | 需确认 |
| **format_reward** | 未确认 | 0.1 | 需确认 |
| **success_reward** | 未确认 | 1.0 | 需确认 |

### 关键差异总结

**明确不一致的参数：**
1. **gamma: 0.99 vs 1.0** — 折扣因子差异会显著影响长 horizon 任务的 reward signal
2. **lambda: 0.99 vs 1.0** — GAE lambda 差异影响 advantage 估计的 bias-variance tradeoff
3. **adv_estimator: masked_gae vs gae** — 可能在 loss mask 处理上不同
4. **max_prompt/response_length** — 差异很大，可能影响截断行为
5. **critic_warmup: 3 vs 0** — ERA 会先 warmup critic 3 步
6. **rollout engine: vllm vs sglang** — 推理引擎不同，可能有细微生成差异
7. **reward_scaling/bias/clip** — VAGEN 有额外的 reward 后处理，ERA 未设

**需要进一步确认的：**
1. VAGEN 训练 EB-Alfred 时实际用的 training script（目前只有 Sokoban 的训练脚本，没看到 Alfred 专用脚本）
2. window_size / use_multi_turn_reward / use_loss_mask / use_gae_mask 的默认值
3. processor 的 min_pixels / max_pixels 在 VAGEN 训练中的设置

### 结论

> **大量参数与 ERA 不一致**，尤其是 gamma、lambda、adv_estimator、sequence lengths、critic_warmup 和 reward 后处理。
>
> 在正式对比实验前，建议：
> 1. 先创建一个与 ERA 完全对齐的 baseline 配置
> 2. 确认 VAGEN 中 masked_gae 等算法是否已实现
> 3. 确认 window_size 等 multi-turn 关键参数的默认值
> 4. 验证 reward 计算逻辑是否等价

---

## 附录：关键文件路径

### ERA repo
- SFT 训练: `Embodied-Reasoning-Agent/ERA-sft/epl/train.py`
- RL 训练脚本: `Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/examples/alfred/run.sh`
- Rollout 逻辑: `Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/mllm_agent/rollout.py`
- PPO 配置: `Embodied-Reasoning-Agent/ERA-rl/VAGEN/vagen/trainer/config/ppo_trainer.yaml`

### VAGEN-eb-alfred repo
- 环境定义: `VAGEN-eb-alfred/vagen/envs/eb_alfred/eb_alfred_env.py`
- 多轮配置: `VAGEN-eb-alfred/vagen/configs/vagen_multiturn.yaml`
- Agent loop (concat): `VAGEN-eb-alfred/vagen/agent_loop/gym_agent_loop.py`
- Agent loop (no-concat): `VAGEN-eb-alfred/vagen/agent_loop/gym_agent_loop_no_concat.py`
- 训练脚本 (PPO 示例): `VAGEN-eb-alfred/examples/sokoban/train_ppo_qwen25vl3b.sh`
- 评估配置: `VAGEN-eb-alfred/examples/evaluate/eb_alfred/config.yaml`
- EB-Alfred 训练配置: `VAGEN-eb-alfred/examples/eb_alfred/train_eb_alfred_vision.yaml`
