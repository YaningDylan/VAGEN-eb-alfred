# NCCL AllReduce Hang Analysis — EB-ALFRED SFT Training

## 问题描述

在使用 4×H200 GPU、DeepSpeed ZeRO-2、`torchrun` 训练 Qwen2.5-VL-3B-Instruct 时，训练会在特定 step 处确定性地挂起，600 秒后触发 NCCL watchdog timeout（SIGABRT）。

## 错误特征

```
[Rank X] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=812747, OpType=ALLREDUCE, NumelIn=2048, Timeout(ms)=600000)
ran for 600033 milliseconds before timing out.
```

关键异常：
- **所有 rank 均在同一 SeqNum 超时**，说明某个 AllReduce 集合操作永远无法完成
- **NumelIn 不一致**：rank 0/2/3 报告 `NumelIn=2048`，rank 1 报告 `NumelIn=4096`
  → 不同 rank 在相同 SeqNum 处尝试 reduce 不同大小的 tensor（协议违规）
- 超时发生在 loss + grad_norm 已成功 log 之后，说明 backward 完成，hang 在 `engine.step()` 内的 AllReduce 阶段

## 崩溃点（seed=123）

| Run | Seed | 崩溃 Step | 说明 |
|-----|------|-----------|------|
| run1-11 | 42 | 355 | 每次必崩 |
| run12 | 123 | 2742 | seed 修复了 step 355，但新增崩溃点 |
| run13 | 123 | 2742 | 同上 |
| run14 | 123 | 2742 | 加 `ddp_find_unused_parameters=True` 无效 |
| run15 | 123 | 2742 | 错误的 dataset index skip 无效 |
| run16 | 123 | 2952 | step 2742 修复后，出现新崩溃点 |
| run17 | 123 | 3330 | step 2742+2952 修复后，step 3329 挂约10分钟后SIGABRT |
| run18 | 123 | — | skip {2742,2952,3330}，从checkpoint-3000恢复，进行中 |

## 根本原因（待深入分析）

### 已确认
- 崩溃是**数据顺序相关**（seed 决定）：seed=42 崩在 355，seed=123 崩在 2742 和 2952
- NumelIn 不一致（2048 vs 4096）指向 **merger 模块参数**（`visual.merger.ln_q.weight: 2048 params`）
- 崩溃点的 batch 分析：各 rank 均有混合（图文/纯文）样本，collator 要求所有 4 个样本都有图才包含 `pixel_values`，因此 merger **理论上不应被调用**

### 未解之谜
1. 如果 merger 未被调用，为何 AllReduce 的 NumelIn 在不同 rank 间不一致？
2. 可能与 DeepSpeed ZeRO-2 的 gradient bucket 分配逻辑有关：
   - 当某参数的 gradient 为 `None` 时，bucket 可能在不同 rank 上被合并方式不同
   - 或与 dataloader worker 在预加载图像时产生的 IO 阻塞有关（某 worker 挂起导致 AllReduce 等待）
3. `overlap_comm=false` 减缓了问题（从必崩到特定 step 才崩），但未根治

### 排除的方案
- `overlap_comm: true → false`：缓解但未根治
- `group_by_modality_length=False`：无效
- `ddp_find_unused_parameters=True`：对 DeepSpeed 无效（DeepSpeed 不走 DDP AllReduce）
- Dataset level index skip：因未能精确模拟训练时的数据顺序（Python `random.shuffle` 在 dataset 构建时使用了不可预测的随机状态），计算的 index 有误

## 临时修复方案（已生效）

在 `train_sft_vagen.py` 中添加 `SafeTrainer`，在 Trainer 层面拦截特定 step：

```python
class SafeTrainer(transformers.Trainer):
    SKIP_GLOBAL_STEPS = {2742, 2743, 2744, 2952, 2953, 2954, 3330, 3331, 3332}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_safe_batch = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step + 1
        if step in self.SKIP_GLOBAL_STEPS and self._last_safe_batch is not None:
            inputs = self._last_safe_batch
        else:
            self._last_safe_batch = inputs
        return super().training_step(model, inputs, num_items_in_batch)
```

效果：用上一步的 batch 替换崩溃 step 的 batch，确保所有 rank 的参数计算图对称，避免 AllReduce 不对称。

## 待分析方向

1. **精确定位崩溃 batch**：需在训练开始时加 per-rank batch index logging，精确记录每个 step 每个 rank 的 dataset indices
2. **验证 merger 是否被调用**：在 `visual.merger.forward` 加 hook，确认在崩溃 step 各 rank 是否对称调用
3. **DeepSpeed bucket 分析**：打印各 rank 的 gradient bucket 组成，确认是否存在非对称情况
4. **IO 假设验证**：在 dataloader worker 的图像加载处加 timeout 和日志，观察是否有 worker 在崩溃 step 附近阻塞

## 环境信息

- Model: Qwen2.5-VL-3B-Instruct
- GPUs: 4× H200 (143GB each)
- DeepSpeed: ZeRO-2, `overlap_comm=false`
- Trainer: HuggingFace Transformers `Trainer`
- liger_kernel: enabled
- Total steps: 4217 (67463 samples, bs=4/GPU, 4 GPUs, 1 epoch)
- Data: ERA EB-ALFRED SFT (trajectory + env_anchored + external_knowledge)
- Script: `VAGEN-eb-alfred/scripts/train_sft_vagen.py`
- DeepSpeed config: `VAGEN-eb-alfred/scripts/ds_zero2_h100.json`
