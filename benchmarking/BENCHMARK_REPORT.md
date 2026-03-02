# EB-ALFRED Benchmark Report

**Generated**: 2026-02-28 22:35:09
**Model**: GPT-4.1 (temperature=0)
**Max turns per episode**: 30
**Server**: EB-ALFRED on localhost:8000 (ai2thor 2.1.0)

---

## Overview

| Experiment | Resolution | Mode | Episodes | Concurrency | Success Rate | Wall Time | Throughput |
|:-----------|:----------:|:----:|:--------:|:-----------:|:------------:|:---------:|:----------:|
| 500x500 serial | 500x500 | serial | 10/10 | 1 | **40.0%** | 19.3 min | 0.5 ep/min |
| 500x500 parallel | 500x500 | parallel | 128/128 | 20 | **44.5%** | 35.6 min | 3.6 ep/min |
| serial_328_10ep | — | — | — | — | — | — | — |
| parallel_328_128ep | — | — | — | — | — | — | — |
| serial_96_10ep | — | — | — | — | — | — | — |
| parallel_96_128ep | — | — | — | — | — | — | — |

## Timing

| Experiment | Wall Time (s) | Wall Time (min) | Effective per-Episode (s) | Throughput (ep/min) |
|:-----------|:-------------:|:---------------:|:-------------------------:|:-------------------:|
| 500x500 serial | 1157.7 | 19.3 | 115.77 | 0.5 |
| 500x500 parallel | 2134.2 | 35.6 | 16.67 | 3.6 |

> **Effective per-episode** = total wall time / number of episodes (amortized time).

## Success & Rewards

| Experiment | Completed | Success | Success Rate | Avg Reward | Avg Turns | Median Turns |
|:-----------|:---------:|:-------:|:------------:|:----------:|:---------:|:------------:|
| 500x500 serial | 10 | 4 | 40.0% | 2.47 | 21.4 | 30.0 |
| 500x500 parallel | 128 | 57 | 44.5% | 2.404 | 20.3 | 28.5 |

## Token Consumption

| Experiment | Total Tokens | Prompt Tokens | Completion Tokens | API Calls | Avg Tokens/Ep | Avg Calls/Ep |
|:-----------|:------------:|:-------------:|:-----------------:|:---------:|:-------------:|:------------:|
| 500x500 serial | 1,417,129 | 1,408,364 | 8,765 | 214 | 141,713 | 21.4 |
| 500x500 parallel | 17,654,258 | 17,547,704 | 106,554 | 2,601 | 137,924 | 20.3 |

### Per-Episode Token Averages

| Experiment | Avg Prompt/Ep | Avg Completion/Ep | Avg Total/Ep |
|:-----------|:-------------:|:-----------------:|:------------:|
| 500x500 serial | 140,836 | 876 | 141,713 |
| 500x500 parallel | 137,091 | 832 | 137,924 |

## Resolution Comparison

### Serial (10 episodes each)

| Resolution | Success Rate | Avg Reward | Avg Tokens/Ep | Wall Time (s) |
|:----------:|:------------:|:----------:|:-------------:|:-------------:|
| 500x500 | 40.0% | 2.47 | 141,713 | 1157.7 |

### Parallel (128 episodes each, concurrency=20)

| Resolution | Success Rate | Avg Reward | Avg Tokens/Ep | Wall Time (min) | Throughput (ep/min) |
|:----------:|:------------:|:----------:|:-------------:|:---------------:|:-------------------:|
| 500x500 | 44.5% | 2.404 | 137,924 | 35.6 | 3.6 |

## Finish Reasons

- **500x500 serial**: done: 10
- **500x500 parallel**: done: 128

## Configuration

```yaml
model: gpt-4.1
temperature: 0
max_tokens: 1024
max_turns: 30
max_concurrency (API): 20
max_concurrent_jobs (serial): 1
max_concurrent_jobs (parallel): 20
server: EB-ALFRED on localhost:8000
GPU: NVIDIA RTX 5070 Ti (16 GB)
```
