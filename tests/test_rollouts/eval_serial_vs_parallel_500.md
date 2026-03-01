# EB-ALFRED: Serial vs Parallel Throughput Comparison

**Model:** GPT-4.1 | **Resolution:** 500×500 | **Date:** 2026-03-01
**Display:** GPU Xorg :2 (NVIDIA RTX 5090, single GPU)
**Max turns:** 30 | **eval_set:** base

---

## Table 1 — Performance & Throughput

| Metric | Serial (5 ep) | Parallel (127/128 ep) |
|---|:---:|:---:|
| **Episodes** | 5 | 127 *(128 requested, 1 killed)* |
| **Success rate** | **60.0%** (3/5) | **44.9%** (57/127) |
| **Avg cumulative reward** | 2.020 | 2.341 |
| **Avg turns / episode** | 16.2 | 19.4 |
| **Error episodes** | 0 | 0 |
| **Total wall time** | 647 s (10.8 min) | 2684 s (44.7 min) |
| **Avg wall time / episode** | 129.4 s | **21.1 s** |
| **max_concurrent_jobs** | 1 | 100 |
| **max_concurrency (LLM)** | 1 | 100 |
| **Throughput speedup** | 1× | **6.1×** |

> **Speedup note:** Parallel avg 21.1 s/ep vs serial 129.4 s/ep = 6.1× faster per episode.
> LLM API latency (~1.5–5 s/call) dominates; dummy-action baseline finishes 128 ep in ~2 min.

---

## Table 2 — Token Usage

| Metric | Serial (5 ep) | Parallel (127/128 ep) |
|---|:---:|:---:|
| **Total tokens** | 533,951 | 16,775,261 |
| **Avg prompt tokens / episode** | 106,187 | 131,288 |
| **Avg completion tokens / episode** | 604 | 801 |
| **Avg prompt tokens / turn** | 6,555 | 6,753 |
| **Avg completion tokens / turn** | 37 | 41 |

> Token counts are high because each turn embeds a 500×500 observation image
> (encoded as vision tokens). Prompt grows with conversation history each turn.

---

## Per-Episode Detail — Serial Run

| Seed | Turns | Success | Reward | Prompt tok | Compl tok | Episode wall time |
|:----:|:-----:|:-------:|:------:|:----------:|:---------:|:-----------------:|
| 0 | 30 | ✗ | 3.0 | 217,157 | 1,076 | 215 s |
| 30 | 8 | ✓ | 1.8 | 26,928 | 270 | 44 s |
| 60 | 30 | ✗ | 2.0 | 244,459 | 1,176 | 235 s |
| 90 | 4 | ✓ | 1.4 | 10,626 | 136 | 23 s |
| 120 | 9 | ✓ | 1.9 | 31,763 | 360 | 50 s |

---

## Configs Used

| | Serial | Parallel |
|---|---|---|
| Config file | `tests/eval_eb_alfred_gpt41_5ep_serial_500.yaml` | `tests/eval_eb_alfred_gpt41_128ep_parallel_500.yaml` |
| Seeds | [0, 30, 60, 90, 120] | 0–127 |
| max_concurrent_jobs | 1 | 100 |
| max_concurrency (GPT) | 1 | 100 |
| obs_image_size | 500 | 500 |

---

## Notes

- **Success rate difference (60% vs 45%):** The 5 serial seeds were evenly spaced
  (0, 30, 60, 90, 120); parallel covers all 128 task types including harder ones.
- **Reward paradox (serial lower):** 2.020 vs 2.341 — serial has 2/5 hitting the
  30-turn limit with partial reward; parallel's larger sample stabilises the mean.
- **Parallel bottleneck:** with 100 concurrent Unity instances on one GPU + Xorg,
  CPU is ~147% per instance (~147 cores total at steady state). 100 instances are
  within the feasible range on this 160-core machine.
- **Seed 126 (1/128):** timed out and was killed after the other 127 completed
  (server returned intermittent 500 errors during cleanup; excluded from stats).

---

## Table 3 — Concat=true vs Concat=false (64-way Parallel, seeds 0–63)

> **Hypothesis:** sending only `[system, current_obs]` per turn instead of full history
> reduces prompt tokens ~8× but removes the model's memory of past actions.

| Metric | Concat=true *(ref: 20ep seeds 0–19)* | Concat=false *(64ep seeds 0–63)* |
|---|:---:|:---:|
| **Episodes** | 20 | 64 |
| **Success rate** | **50.0%** (10/20) | **0.0%** (0/64) |
| **Avg cumulative reward** | 2.330 | 2.813 *(format-only)* |
| **Avg turns / episode** | 18.6 | 28.1 |
| **Total eval wall time** | — | ~912 s (15.2 min) |
| **Effective throughput** | ~21 s/ep *(from 128ep run)* | **14.2 s/ep** |
| **max_concurrent_jobs** | 20 | 64 |
| **max_concurrency (LLM)** | 100 | 256 |
| **concat_history** | true | false |

> **Throughput note:** No-concat 64 ep in ~912 s = 14.2 s effective/ep vs concat=true
> 128 ep in 2684 s = 21.1 s/ep — **1.5× faster effective throughput**.
> Fastest episode: 303 s; slowest: 828 s (max turns = 30 with no memory, model keeps trying).

---

## Table 4 — Token Usage: Concat=true vs Concat=false

| Metric | Concat=true (parallel ref) | Concat=false (64ep parallel) | Reduction |
|---|:---:|:---:|:---:|
| **Avg prompt tokens / episode** | 124,467 | 22,834 | **5.5×** |
| **Avg prompt tokens / turn** | 5,347 | 815 | **6.6×** |
| **Avg completion tokens / episode** | 758 | 1,113 | — *(more turns)* |
| **Avg completion tokens / turn** | 41 | 39 | ~1× |
| **Total prompt tokens (run)** | ~2,489,340 *(20ep)* | 1,461,385 *(64ep)* | — |

> Concat=true prompt grows each turn (previous images accumulate). Concat=false sends
> one fresh image per turn → near-constant prompt size (~815 tokens/turn = 1 image + text).

---

## Table 5 — Episode Timing Breakdown (Concat=false 64-ep parallel only)

> Per-episode timing instrumented in `vision_workflow.py`; concat=true runs pre-dated
> the instrumentation and do not have this data.

| Component | Avg per episode | Avg per turn | % of total |
|---|:---:|:---:|:---:|
| **Reset / env-init** | 113.5 s | — *(one-time)* | 25.0% |
| **LLM API wait** | 166.8 s | **5.97 s** | 36.8% |
| **Env step (Unity)** | 157.2 s | **5.68 s** | 34.7% |
| **Other overhead** | 15.8 s | — | 3.5% |
| **Total episode** | **453.3 s** | **11.65 s** | 100% |

> **Reset inflation:** 113.5 s includes 64 Unity processes starting concurrently on one GPU.
> In steady-state (pre-warmed sessions), reset would be ~1–2 s.
> **LLM vs Env balance:** with no-concat, LLM (37%) and Env step (35%) are nearly equal
> bottlenecks — the network round-trip to localhost:8000 + Unity render per step takes ~5.7 s
> at 64-way parallelism.

---

## Configs Used (extended)

| | Serial | Parallel (concat=true) | Parallel (no-concat) |
|---|---|---|---|
| Config file | `tests/eval_eb_alfred_gpt41_5ep_serial_500.yaml` | `tests/eval_eb_alfred_gpt41_128ep_parallel_500.yaml` | `tests/eval_eb_alfred_gpt41_64ep_noconcat.yaml` |
| Seeds | [0, 30, 60, 90, 120] | 0–127 | 0–63 |
| max_concurrent_jobs | 1 | 100 | 64 |
| max_concurrency (GPT) | 1 | 100 | 256 |
| concat_history | true | true | **false** |
| obs_image_size | 500 | 500 | 500 |
