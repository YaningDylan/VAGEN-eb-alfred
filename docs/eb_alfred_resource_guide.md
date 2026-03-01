# EB-ALFRED Resource & Capacity Guide

This document provides measured resource usage per AI2-THOR Unity instance
at different rendering resolutions, along with capacity recommendations for
common GPU configurations.

> **Test environment**: ai2thor 2.1.0, Unity 2018.3.6 (Mono), Ubuntu Linux,
> NVIDIA GPU. Measurements averaged over 4 concurrent instances on FloorPlan1
> after 3 warm-up steps.

---

## Per-Instance Resource Usage

| Resolution | VRAM (GPU) | RAM (CPU) | Notes |
|:----------:|:----------:|:---------:|:------|
| 500 × 500  | ~148 MiB   | ~746 MiB  | Default EmbodiedBench resolution |
| 328 × 328  | ~142 MiB   | ~808 MiB  | Mid-range ablation |
| 96 × 96    | ~128 MiB   | ~743 MiB  | Low-res ablation |

**Key observations:**

- VRAM scales modestly with resolution (~128–148 MiB range). The Unity
  engine base footprint dominates; framebuffer differences are small.
- RAM is roughly constant (~750 MiB) regardless of resolution.
- Each Unity process also uses ~1 CPU thread.

---

## GPU Capacity Table

The table below shows **recommended `max_sessions`** for a single GPU.
Numbers assume ~1.5 GiB reserved for the OS/driver and include a 15%
safety margin to avoid OOM under load spikes.

| GPU Model (example)        | Total VRAM | Usable VRAM¹ | 500×500 | 328×328 | 96×96 |
|:---------------------------|:----------:|:------------:|:-------:|:-------:|:-----:|
| RTX 4060 / RTX 3060        | 8 GiB      | ~5.5 GiB     | 31      | 33      | 36    |
| RTX 4070 / RTX 3080        | 12 GiB     | ~9 GiB       | 51      | 54      | 60    |
| RTX 4080 / Tesla T4        | 16 GiB     | ~12.3 GiB    | 70      | 74      | 82    |
| RTX 4090 / A5000 / L40     | 24 GiB     | ~19.1 GiB    | 110     | 115     | 127   |
| A100 (40 GB) / A6000       | 32 GiB     | ~25.9 GiB    | 149     | 155     | 172   |
| A100 (80 GB) / H100        | 80 GiB     | ~66.7 GiB    | 384     | 400     | 444   |

> ¹ Usable = Total × 0.85 − 1.5 GiB (driver/OS overhead)

### Multi-GPU Scaling

With N GPUs, multiply the single-GPU capacity by N. The EB-ALFRED server
automatically load-balances across all detected GPUs.

**Example**: 4 × A100 (80 GB) at 500×500 → `max_sessions` ≈ 4 × 384 = **1536**

---

## RAM Capacity Table

Each Unity instance uses ~750 MiB of system RAM. This may become the
bottleneck before GPU memory on machines with many GPUs.

| System RAM | Max Instances (RAM-limited) |
|:----------:|:--------------------------:|
| 32 GiB     | ~36                        |
| 64 GiB     | ~78                        |
| 128 GiB    | ~163                       |
| 256 GiB    | ~333                       |
| 512 GiB    | ~672                       |

> Formula: `floor((total_ram - 4 GiB OS overhead) / 0.75 GiB)`

---

## Recommended Configuration

### Server Side

```bash
# Start with auto GPU detection + session limit
python -m vagen.envs.eb_alfred.serve \
    --port 8000 \
    --max-sessions <N>    # Use GPU capacity table above
```

### Client Side (YAML)

```yaml
envs:
  - config:
      retries: 30          # Increase for large queues
      backoff: 1.5         # Moderate backoff growth
      timeout: 600         # 10-min timeout per request

run:
  max_concurrent_jobs: 128  # Can exceed max_sessions; server auto-queues
```

### Quick Reference

| Scenario                   | max_sessions | max_concurrent_jobs |
|:---------------------------|:------------:|:-------------------:|
| 1 GPU (16 GiB), inference  | 64           | 128                 |
| 1 GPU (24 GiB), inference  | 100          | 128                 |
| 4 GPU (24 GiB), training   | 400          | 512                 |
| 8 GPU (80 GiB), training   | 3000         | 4096                |

---

## Methodology

Resource measurements were collected with the following procedure:

1. Record baseline GPU memory via `nvidia-smi`
2. Launch N = 4 Unity instances sequentially (Controller → reset → ChangeResolution → 3 warm-up steps)
3. After each instance: wait 1s for memory to settle, record delta
4. Measure RAM via `psutil.Process.memory_info().rss` for all `thor-*` processes
5. Average across 4 instances
6. Verify cleanup: stop all controllers, confirm memory released

Full test script: `tests/test_unity_vram_all.py`
