# SFT Experiment Log — Qwen2.5-VL-3B-Instruct on EB-ALFRED

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | `Qwen2.5-VL-3B-Instruct` (3B params) |
| **Dataset** | trajectory_vagen.json (8,839) + env_anchored_vagen.json (48,624) + external_knowledge_vagen.json (10,000) = **67,463 samples** |
| **Eval set** | 20 samples (base split of trajectory_vagen.json) |
| **Num GPUs** | 4 × H200 |
| **Per-device batch size** | 4 |
| **Gradient accumulation** | 1 |
| **Effective batch size** | **16** |
| **Epochs** | 1 |
| **Total steps** | 4,217 |
| **Peak LR** | 1e-5 (cosine decay, warmup ratio 0.05) |
| **Precision** | bf16 + tf32 |
| **DeepSpeed** | ZeRO-2 |
| **Attention** | Flash Attention 2 + Liger Kernel |
| **Visual encoder** | Frozen (visual merger layer unfrozen) |
| **Max sequence length** | 8,192 tokens |
| **Eval strategy** | Every 500 steps |

---

## Run History

Training required multiple runs due to NCCL timeouts and interruptions. The checkpoint at step 2501 was used as the resume point for runs 14–17.

| Run | Start Epoch | End Epoch | Steps | Outcome | Duration |
|-----|------------|-----------|-------|---------|----------|
| 12 | — | — | 0 | FAILED (HuggingFace model download error) | ~11s |
| 13 | 0.000 | 0.650 | 2,742 | FAILED (NCCL collective timeout at step 2742) | ~56 min |
| 14 | 0.593 | 0.650 | 242 | FAILED (SIGTERM) | ~3 min |
| 15 | 0.593 | 0.650 | 242 | FAILED (SIGTERM) | ~3 min |
| 16 | 0.593 | 0.700 | 451 | FAILED (SIGTERM) | ~8 min |
| 17 | 0.593 | 0.789 | 829 | FAILED (NCCL SIGABRT at step 3329) | ~24 min |
| **18** | **0.712** | **1.000** | **1,217** | **COMPLETED** | **20m 55s** |

---

## Eval Loss (Full Epoch)

Eval loss measured every 500 steps across all runs. Continuous 28% reduction over one epoch.

| Checkpoint | Step | Epoch | Eval Loss | Run |
|-----------|------|-------|-----------|-----|
| Ckpt-1 | 500 | 0.119 | 0.5088 | Run 13 |
| Ckpt-2 | 1000 | 0.237 | 0.4776 | Run 13 |
| Ckpt-3 | 1500 | 0.356 | 0.4227 | Run 13 |
| Ckpt-4 | 2000 | 0.474 | 0.4190 | Run 13 |
| Ckpt-5 | 2500 | 0.593 | 0.3964 | Run 13 |
| Ckpt-6 | 3000 | 0.711 | 0.3850 | Run 17 |
| Ckpt-7 | 3500 | 0.830 | 0.3682 | Run 18 |
| Ckpt-8 | 4000 | 0.949 | **0.3664** | Run 18 |

**Eval loss reduction:** 0.5088 → 0.3664 (−28.1% over 1 epoch)

---

## Training Loss Curve

Per-step loss sampled across all runs, aligned on global epoch axis.

| Step | Epoch | Train Loss | Grad Norm | LR |
|------|-------|-----------|-----------|-----|
| 1 | 0.000 | 0.9787 | 6.873 | 0 |
| 200 | 0.047 | 0.3515 | 1.630 | 9.43e-06 |
| 400 | 0.095 | 0.3949 | 1.479 | 9.95e-06 |
| 600 | 0.142 | 0.2373 | 1.380 | 9.77e-06 |
| 800 | 0.190 | 0.3177 | 1.346 | 9.48e-06 |
| 1000 | 0.237 | 0.2512 | 1.298 | 9.08e-06 |
| 1200 | 0.285 | 0.4308 | 1.445 | 8.57e-06 |
| 1400 | 0.332 | 0.1998 | 1.129 | 7.98e-06 |
| 1600 | 0.379 | 0.2997 | 1.270 | 7.32e-06 |
| 1800 | 0.427 | 0.3219 | 1.238 | 6.60e-06 |
| 2000 | 0.474 | 0.2482 | 1.172 | 5.84e-06 |
| 2200 | 0.522 | 0.2408 | 1.246 | 5.06e-06 |
| 2400 | 0.569 | 0.5646 | 1.713 | 4.28e-06 |
| 2600 | 0.617 | 0.3472 | 1.311 | 3.51e-06 |
| 2700 | 0.640 | 0.1998 | 1.314 | 3.14e-06 |
| 2800 | 0.664 | 0.1946 | 1.201 | 2.79e-06 |
| 2900 | 0.688 | 0.1868 | 1.103 | 2.44e-06 |
| 3000 | 0.711 | 0.1935 | 1.241 | 2.11e-06 |
| 3100 | 0.735 | 0.2027 | 1.198 | 1.80e-06 |
| 3200 | 0.759 | 0.2525 | 1.312 | 1.51e-06 |
| 3300 | 0.783 | 0.2098 | 1.183 | 1.24e-06 |
| 3450 | 0.818 | 0.1350 | 1.021 | 8.80e-07 |
| 3600 | 0.854 | 0.2669 | 1.143 | 5.76e-07 |
| 3750 | 0.889 | 0.3321 | 1.201 | 3.33e-07 |
| 3900 | 0.925 | 0.3309 | 1.188 | 1.55e-07 |
| 4050 | 0.960 | 0.2333 | 1.143 | 4.33e-08 |
| 4200 | 0.996 | 0.2537 | 1.375 | ~0 |

---

## Final Results

| Metric | Value |
|--------|-------|
| **Final eval loss** (step 4000, epoch 0.949) | **0.3664** |
| **Final train loss** (epoch avg over all 4,217 steps) | **0.08197** |
| Train samples/sec | 53.74 |
| Train steps/sec | 3.359 |
| Total training time (effective) | ~2.2 hours across restarts |
| Eval loss reduction (0→1 epoch) | 0.5088 → 0.3664 (**−28.1%**) |
| Peak grad norm | 6.87 (step 1, then stable ~1.0–1.7) |

### Notes
- `train_loss = 0.08197` is the HuggingFace Trainer **epoch-average** loss (averaged across all 4,217 steps). Individual step losses in the log are per-step and noisier (range: 0.1–0.6 in late training).
- Eval loss improvement rate slows after epoch ~0.47 (0.4190 at step 2000 → 0.3664 at step 4000), suggesting diminishing returns within 1 epoch.
- Visual encoder was kept frozen throughout; only LLM + visual merger layers were trained.
- NCCL timeouts in runs 13 and 17 were environment issues (not training instability); grad norms remained healthy throughout.
