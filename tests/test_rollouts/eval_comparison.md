# VAGEN Inference Evaluation Report

> Generated: 2026-03-06 14:00

## Overview

| Environment | Model | Mode | Acts/Turn | Episodes | Success Rate | Avg Reward | Avg Turns | Errors | Prompt/Turn | Compl/Turn | Tokens/Turn | Tokens/Task | Avg Time/Ep |
|-------------|-------|------|:---------:|:--------:|:------------:|-----------:|----------:|-------:|------------:|-----------:|------------:|------------:|------------:|
| EB-ALFRED | Claude Sonnet 4 | concat | 1 | 20 | **35.0%** | 1.73 | 13.8/30 | 5 | N/A | N/A | N/A | N/A | N/A |
| EB-ALFRED | GPT-4.1 | concat | 1 | 20 | **60.0%** | 2.24 | 17.2/30 | 0 | 6,219 | 41 | 6,260 | 107,678 | N/A |
| EB-ALFRED | GPT-5 Mini | no-concat | 1 | 20 | **5.0%** | 2.45 | 28.1/30 | 1 | 3,496 | 631 | 4,126 | 115,942 | 355.8s |
| EB-ALFRED | GPT-5 Nano | no-concat | 1 | 20 | **0.0%** | 0.46 | 30.0/30 | 0 | 3,592 | 985 | 4,577 | 137,331 | 255.9s |
| EB-ALFRED | GPT-4.1 | no-concat | 20 | 10 | **80.0%** | 1.50 | 7.0/15 | 0 | 3,432 | 112 | 3,545 | 24,819 | 30.5s |
| EB-ALFRED | GPT-5.4 | no-concat | 20 | 10 | **60.0%** | 1.03 | 5.0/15 | 0 | 3,466 | 114 | 3,581 | 17,905 | 21.6s |
| EB-ALFRED | GPT-5 Mini | no-concat | 20 | 10 | **50.0%** | 0.89 | 6.4/15 | 1 | 3,507 | 825 | 4,333 | 27,734 | 100.2s |
| EB-ALFRED | GPT-5 Nano | no-concat | 20 | 10 | **10.0%** | 0.13 | 13.6/15 | 0 | 3,578 | 1,020 | 4,599 | 62,556 | 130.0s |
| Sokoban | GPT-4.1 | concat | 1 | 20 | **40.0%** | 0.86 | 7.0/10 | 0 | 1,981 | 52 | 2,033 | 14,234 | N/A |

### EB-ALFRED Multi-Action Comparison (no-concat, 20 acts/turn, 15 max turns, 10 episodes)

| Model | Success Rate | Avg Reward | Avg Turns | Errors | Prompt/Turn | Compl/Turn | Tokens/Task | LLM Latency/Turn | Avg Time/Ep |
|-------|:------------:|-----------:|----------:|-------:|------------:|-----------:|------------:|------------------:|------------:|
| GPT-4.1 | **80.0%** | 1.50 | 7.0/15 | 0 | 3,432 | 112 | 24,819 | 3.0s | 30.5s |
| GPT-5.4 | **60.0%** | 1.03 | 5.0/15 | 0 | 3,466 | 114 | 17,905 | 2.6s | 21.6s |
| GPT-5 Mini | **50.0%** | 0.89 | 6.4/15 | 1 | 3,507 | 825 | 27,734 | 14.0s | 100.2s |
| GPT-5 Nano | **10.0%** | 0.13 | 13.6/15 | 0 | 3,578 | 1,020 | 62,556 | 8.9s | 130.0s |

**Per-Turn Failure Analysis:**

| Model | Total Turns | Format Error | Action Invalid | Valid but Ineffective | Effective |
|-------|:----------:|:------------:|:--------------:|:---------------------:|:---------:|
| GPT-4.1 | 70 | 0 (0.0%) | 1 (1.4%) | 15 (21.4%) | **54 (77.1%)** |
| GPT-5.4 | 50 | 0 (0.0%) | 7 (14.0%) | 12 (24.0%) | **31 (62.0%)** |
| GPT-5 Mini | 64 | 25 (39.1%) | 4 (6.2%) | 8 (12.5%) | **27 (42.2%)** |
| GPT-5 Nano | 136 | 133 (97.8%) | 0 (0.0%) | 0 (0.0%) | **3 (2.2%)** |

---

## Detailed Results

### EB-ALFRED — Claude Sonnet 4

| Metric | Value |
|--------|-------|
| Success Rate | **7/20 = 35.0%** |
| Clean Success Rate | 7/15 = 46.7% (excl. model errors) |
| Model Errors | 5/20 |
| Avg Reward | 1.73 |
| Avg Turns | 13.8 / 30 |

<details>
<summary>Per-Episode Results (20 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|--------|------|
| 1 | 207 | 30 | 3.00 | N | - | done | Put two rolls of toilet paper on the back of ... |
| 2 | 100 | 30 | 3.00 | N | - | done | Put two sets of keys on the couch. |
| 3 | 315 | 4 | 1.40 | Y | - | done | hold a pillow while turning on a lamp |
| 4 | 423 | 4 | 1.40 | Y | - | done | Place a newspaper on a couch. |
| 5 | 531 | 8 | 1.80 | Y | - | done | Put two sets of keys on the shelf. |
| 6 | 639 | 30 | 3.00 | N | - | done | Place a rinsed slice of tomato in the microwa... |
| 7 | 747 | 30 | 3.00 | N | - | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 10 | 2.00 | Y | - | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 24 | 2.40 | N | - | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 30 | 3.00 | N | - | done | Clean a rag, put it away. |
| 11 | 1179 | 30 | 3.00 | N | - | done | Put a glass with a butter knife in it and put... |
| 12 | 1287 | 30 | 3.00 | N | - | done | Put a bowl with the watch in it on the shelf. |
| 13 | 1395 | 8 | 1.80 | Y | - | done | Relocate two books to a bedroom desk. |
| 14 | 1503 | 4 | 1.40 | Y | - | done | Turn on the lamp while holding the remote. |
| 15 | 1719 | 4 | 1.40 | Y | - | done | Put a phone on a bed. |
| 16 | 1827 | 0 | 0.00 | N | - | model_error | Place a cleaned sponge in a bathtub. |
| 17 | 1611 | 0 | 0.00 | N | - | model_error | Examine a tv remote next to the light of a ta... |
| 18 | 1935 | 0 | 0.00 | N | - | model_error | Put two newspapers away in a drawer. |
| 19 | 2043 | 0 | 0.00 | N | - | model_error | Put two rolls of toilet paper on the back of ... |
| 20 | 2151 | 0 | 0.00 | N | - | model_error | hold a pillow while turning on a lamp |

</details>

### EB-ALFRED — GPT-4.1

| Metric | Value |
|--------|-------|
| Success Rate | **12/20 = 60.0%** |
| Model Errors | 0/20 |
| Avg Reward | 2.24 |
| Avg Turns | 17.2 / 30 |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 2,139,507 | 14,060 | 2,153,567 |
| Avg per task | 107,678 | 703 | 107,678 |
| Avg per turn | 6,219 | 41 | 6,260 |

<details>
<summary>Per-Episode Results (20 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|--------|------|
| 1 | 315 | 4 | 1.40 | Y | 10,625 | done | hold a pillow while turning on a lamp |
| 2 | 100 | 30 | 3.00 | N | 211,670 | done | Put two sets of keys on the couch. |
| 3 | 207 | 30 | 3.00 | N | 213,411 | done | Put two rolls of toilet paper on the back of ... |
| 4 | 423 | 4 | 1.40 | Y | 12,935 | done | Place a newspaper on a couch. |
| 5 | 531 | 9 | 1.90 | Y | 32,250 | done | Put two sets of keys on the shelf. |
| 6 | 639 | 30 | 3.00 | N | 225,142 | done | Place a rinsed slice of tomato in the microwa... |
| 7 | 747 | 9 | 1.90 | Y | 33,441 | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 17 | 2.70 | Y | 95,585 | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 19 | 1.90 | N | 103,707 | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 30 | 2.20 | N | 213,534 | done | Clean a rag, put it away. |
| 11 | 1179 | 30 | 4.00 | Y | 233,527 | done | Put a glass with a butter knife in it and put... |
| 12 | 1287 | 14 | 2.40 | Y | 63,574 | done | Put a bowl with the watch in it on the shelf. |
| 13 | 1395 | 9 | 1.90 | Y | 32,239 | done | Relocate two books to a bedroom desk. |
| 14 | 1503 | 4 | 1.40 | Y | 11,926 | done | Turn on the lamp while holding the remote. |
| 15 | 1611 | 30 | 2.20 | N | 220,622 | done | Examine a tv remote next to the light of a ta... |
| 16 | 1719 | 4 | 1.40 | Y | 11,823 | done | Put a phone on a bed. |
| 17 | 1827 | 21 | 2.10 | N | 119,624 | done | Place a cleaned sponge in a bathtub. |
| 18 | 1935 | 16 | 2.60 | Y | 83,981 | done | Put two newspapers away in a drawer. |
| 19 | 2043 | 30 | 3.00 | N | 213,318 | done | Put two rolls of toilet paper on the back of ... |
| 20 | 2151 | 4 | 1.40 | Y | 10,633 | done | hold a pillow while turning on a lamp |

</details>

### EB-ALFRED — GPT-5 Mini (no-concat)

| Metric | Value |
|--------|-------|
| Success Rate | **1/20 = 5.0%** |
| Model Errors | 1/20 |
| Avg Reward | 2.45 |
| Avg Turns | 28.1 / 30 |
| Total Wall Time | 7,116s (118.6 min) |
| Avg Time/Episode | 355.8s |
| Avg LLM Latency/Turn | 12.2s |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 1,964,263 | 354,577 | 2,318,840 |
| Avg per task | 98,213 | 17,729 | 115,942 |
| Avg per turn | 3,496 | 631 | 4,126 |

<details>
<summary>Per-Episode Results (20 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Time (s) | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|---------:|--------|------|
| 1 | 100 | 30 | 2.80 | N | 115,699 | 392.8 | done | Put two sets of keys on the couch. |
| 2 | 207 | 30 | 2.90 | N | 119,316 | 404.8 | done | Put two rolls of toilet paper on the back of ... |
| 3 | 315 | 30 | 3.00 | N | 108,705 | 288.7 | done | hold a pillow while turning on a lamp |
| 4 | 423 | 28 | 3.80 | Y | 121,699 | 282.2 | done | Place a newspaper on a couch. |
| 5 | 531 | 30 | 2.80 | N | 116,394 | 399.3 | done | Put two sets of keys on the shelf. |
| 6 | 639 | 30 | 0.90 | N | 141,036 | 506.6 | done | Place a rinsed slice of tomato in the microwa... |
| 7 | 747 | 30 | 3.00 | N | 121,767 | 361.4 | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 30 | 1.20 | N | 151,732 | 496.4 | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 12 | 1.10 | N | 47,329 | 146.1 | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 30 | 2.90 | N | 119,011 | 385.5 | done | Clean a rag, put it away. |
| 11 | 1179 | 12 | 0.50 | N | 58,369 | 193.1 | model_error | Put a glass with a butter knife in it and put... |
| 12 | 1287 | 30 | 0.40 | N | 132,327 | 529.5 | done | Put a bowl with the watch in it on the shelf. |
| 13 | 1395 | 30 | 3.00 | N | 116,460 | 339.3 | done | Relocate two books to a bedroom desk. |
| 14 | 1503 | 30 | 3.00 | N | 121,323 | 291.3 | done | Turn on the lamp while holding the remote. |
| 15 | 1611 | 30 | 2.90 | N | 123,205 | 325.0 | done | Examine a tv remote next to the light of a ta... |
| 16 | 1719 | 30 | 3.00 | N | 120,093 | 289.5 | done | Put a phone on a bed. |
| 17 | 1827 | 30 | 2.90 | N | 120,927 | 376.3 | done | Place a cleaned sponge in a bathtub. |
| 18 | 1935 | 30 | 3.00 | N | 135,551 | 377.6 | done | Put two newspapers away in a drawer. |
| 19 | 2043 | 30 | 2.80 | N | 118,507 | 428.9 | done | Put two rolls of toilet paper on the back of ... |
| 20 | 2151 | 30 | 3.00 | N | 109,390 | 301.9 | done | hold a pillow while turning on a lamp |

</details>

### EB-ALFRED — GPT-5 Nano (no-concat)

| Metric | Value |
|--------|-------|
| Success Rate | **0/20 = 0.0%** |
| Model Errors | 0/20 |
| Avg Reward | 0.46 |
| Avg Turns | 30.0 / 30 |
| Total Wall Time | 5,119s (85.3 min) |
| Avg Time/Episode | 255.9s |
| Avg LLM Latency/Turn | 8.2s |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 2,155,432 | 591,196 | 2,746,628 |
| Avg per task | 107,772 | 29,560 | 137,331 |
| Avg per turn | 3,592 | 985 | 4,577 |

<details>
<summary>Per-Episode Results (20 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Time (s) | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|---------:|--------|------|
| 1 | 100 | 30 | 0.20 | N | 128,388 | 237.1 | done | Put two sets of keys on the couch. |
| 2 | 207 | 30 | 0.00 | N | 132,524 | 253.9 | done | Put two rolls of toilet paper on the back of ... |
| 3 | 315 | 30 | 1.10 | N | 127,120 | 244.9 | done | hold a pillow while turning on a lamp |
| 4 | 423 | 30 | 0.90 | N | 147,788 | 232.5 | done | Place a newspaper on a couch. |
| 5 | 531 | 30 | 0.30 | N | 130,745 | 248.3 | done | Put two sets of keys on the shelf. |
| 6 | 639 | 30 | 0.10 | N | 145,742 | 263.5 | done | Place a rinsed slice of tomato in the microwa... |
| 7 | 747 | 30 | 0.80 | N | 134,591 | 242.6 | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 30 | 0.10 | N | 156,928 | 253.9 | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 30 | 0.30 | N | 132,078 | 260.7 | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 30 | 0.10 | N | 132,825 | 254.5 | done | Clean a rag, put it away. |
| 11 | 1179 | 30 | 0.00 | N | 152,924 | 261.1 | done | Put a glass with a butter knife in it and put... |
| 12 | 1287 | 30 | 0.10 | N | 135,842 | 274.6 | done | Put a bowl with the watch in it on the shelf. |
| 13 | 1395 | 30 | 0.50 | N | 131,410 | 270.7 | done | Relocate two books to a bedroom desk. |
| 14 | 1503 | 30 | 0.70 | N | 139,247 | 270.0 | done | Turn on the lamp while holding the remote. |
| 15 | 1611 | 30 | 2.20 | N | 134,009 | 235.2 | done | Examine a tv remote next to the light of a ta... |
| 16 | 1719 | 30 | 0.60 | N | 138,617 | 259.9 | done | Put a phone on a bed. |
| 17 | 1827 | 30 | 0.10 | N | 135,773 | 264.4 | done | Place a cleaned sponge in a bathtub. |
| 18 | 1935 | 30 | 0.10 | N | 150,007 | 269.7 | done | Put two newspapers away in a drawer. |
| 19 | 2043 | 30 | 0.00 | N | 132,524 | 265.1 | done | Put two rolls of toilet paper on the back of ... |
| 20 | 2151 | 30 | 1.10 | N | 127,546 | 256.1 | done | hold a pillow while turning on a lamp |

</details>

### EB-ALFRED — GPT-4.1 (no-concat, multi-action)

| Metric | Value |
|--------|-------|
| Success Rate | **8/10 = 80.0%** |
| Model Errors | 0/10 |
| Avg Reward | 1.50 |
| Avg Turns | 7.0 / 15 |
| Max Actions/Turn | 20 |
| Total Wall Time | 305s (5.1 min) |
| Avg Time/Episode | 30.5s |
| Avg LLM Latency/Turn | 3.0s |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 240,282 | 7,912 | 248,194 |
| Avg per task | 24,028 | 791 | 24,819 |
| Avg per turn | 3,432 | 112 | 3,545 |

<details>
<summary>Per-Episode Results (10 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Time (s) | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|---------:|--------|------|
| 1 | 100 | 15 | 1.50 | N | 49,661 | 57.9 | done | Put two sets of keys on the couch. |
| 2 | 207 | 4 | 1.40 | Y | 13,659 | 21.7 | done | Put two rolls of toilet paper on the back of ... |
| 3 | 315 | 4 | 1.40 | Y | 13,052 | 17.8 | done | hold a pillow while turning on a lamp |
| 4 | 423 | 2 | 1.20 | Y | 7,891 | 12.6 | done | Place a newspaper on a couch. |
| 5 | 531 | 4 | 1.40 | Y | 13,543 | 20.7 | done | Put two sets of keys on the shelf. |
| 6 | 639 | 5 | 1.50 | Y | 19,343 | 22.4 | done | Place a rinsed slice of tomato in the microwave |
| 7 | 747 | 9 | 1.90 | Y | 31,997 | 39.2 | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 8 | 1.80 | Y | 33,843 | 30.3 | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 10 | 1.00 | N | 34,424 | 43.1 | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 9 | 1.90 | Y | 30,781 | 39.7 | done | Clean a rag, put it away. |

</details>

### EB-ALFRED — GPT-5.4 (no-concat, multi-action)

| Metric | Value |
|--------|-------|
| Success Rate | **6/10 = 60.0%** |
| Model Errors | 0/10 |
| Avg Reward | 1.03 |
| Avg Turns | 5.0 / 15 |
| Max Actions/Turn | 20 |
| Total Wall Time | 216s (3.6 min) |
| Avg Time/Episode | 21.6s |
| Avg LLM Latency/Turn | 2.6s |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 173,356 | 5,700 | 179,056 |
| Avg per task | 17,336 | 570 | 17,905 |
| Avg per turn | 3,466 | 114 | 3,581 |

<details>
<summary>Per-Episode Results (10 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Time (s) | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|---------:|--------|------|
| 1 | 100 | 8 | 0.80 | N | 26,797 | 33.8 | done | Put two sets of keys on the couch. |
| 2 | 207 | 4 | 1.40 | Y | 13,834 | 19.8 | done | Put two rolls of toilet paper on the back of ... |
| 3 | 315 | 1 | 1.10 | Y | 3,299 | 10.7 | done | hold a pillow while turning on a lamp |
| 4 | 423 | 2 | 1.20 | Y | 8,017 | 11.9 | done | Place a newspaper on a couch. |
| 5 | 531 | 3 | 1.30 | Y | 10,214 | 14.5 | done | Put two sets of keys on the shelf. |
| 6 | 639 | 4 | 0.40 | N | 16,122 | 25.6 | done | Place a rinsed slice of tomato in the microwa... |
| 7 | 747 | 15 | 0.80 | N | 53,912 | 44.9 | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 2 | 1.20 | Y | 8,516 | 11.1 | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 10 | 1.00 | N | 34,884 | 33.6 | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 1 | 1.10 | Y | 3,461 | 9.8 | done | Clean a rag, put it away. |

</details>

### EB-ALFRED — GPT-5 Mini (no-concat, multi-action)

| Metric | Value |
|--------|-------|
| Success Rate | **5/10 = 50.0%** |
| Model Errors | 1/10 |
| Avg Reward | 0.89 |
| Avg Turns | 6.4 / 15 |
| Max Actions/Turn | 20 |
| Total Wall Time | 1,002s (16.7 min) |
| Avg Time/Episode | 100.2s |
| Avg LLM Latency/Turn | 14.0s |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 224,540 | 52,800 | 277,340 |
| Avg per task | 22,454 | 5,280 | 27,734 |
| Avg per turn | 3,507 | 825 | 4,333 |

<details>
<summary>Per-Episode Results (10 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Time (s) | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|---------:|--------|------|
| 1 | 100 | 2 | 0.10 | N | 8,178 | 52.8 | model_error | Put two sets of keys on the couch. |
| 2 | 207 | 15 | 0.40 | N | 65,450 | 249.2 | done | Put two rolls of toilet paper on the back of ... |
| 3 | 315 | 1 | 1.10 | Y | 3,671 | 16.8 | done | hold a pillow while turning on a lamp |
| 4 | 423 | 1 | 1.10 | Y | 4,116 | 13.5 | done | Place a newspaper on a couch. |
| 5 | 531 | 1 | 1.10 | Y | 4,244 | 25.5 | done | Put two sets of keys on the shelf. |
| 6 | 639 | 15 | 0.70 | N | 70,027 | 225.3 | done | Place a rinsed slice of tomato in the microwa... |
| 7 | 747 | 12 | 1.20 | N | 50,432 | 157.8 | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 1 | 1.10 | Y | 5,083 | 24.9 | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 15 | 1.00 | N | 62,156 | 217.8 | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 1 | 1.10 | Y | 3,987 | 18.4 | done | Clean a rag, put it away. |

</details>

### EB-ALFRED — GPT-5 Nano (no-concat, multi-action)

| Metric | Value |
|--------|-------|
| Success Rate | **1/10 = 10.0%** |
| Model Errors | 0/10 |
| Avg Reward | 0.13 |
| Avg Turns | 13.6 / 15 |
| Max Actions/Turn | 20 |
| Total Wall Time | 1,300s (21.7 min) |
| Avg Time/Episode | 130.0s |
| Avg LLM Latency/Turn | 8.9s |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 486,905 | 138,660 | 625,565 |
| Avg per task | 48,691 | 13,866 | 62,556 |
| Avg per turn | 3,578 | 1,020 | 4,599 |

<details>
<summary>Per-Episode Results (10 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Time (s) | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|---------:|--------|------|
| 1 | 100 | 15 | 0.00 | N | 65,294 | 144.1 | done | Put two sets of keys on the couch. |
| 2 | 207 | 15 | 0.00 | N | 67,049 | 139.3 | done | Put two rolls of toilet paper on the back of ... |
| 3 | 315 | 15 | 0.00 | N | 65,459 | 139.9 | done | hold a pillow while turning on a lamp |
| 4 | 423 | 1 | 1.10 | Y | 4,958 | 18.0 | done | Place a newspaper on a couch. |
| 5 | 531 | 15 | 0.00 | N | 66,584 | 144.3 | done | Put two sets of keys on the shelf. |
| 6 | 639 | 15 | 0.00 | N | 73,754 | 133.6 | done | Place a rinsed slice of tomato in the microwa... |
| 7 | 747 | 15 | 0.20 | N | 68,735 | 144.5 | done | Move a bowl with a watch inside from the desk... |
| 8 | 855 | 15 | 0.00 | N | 79,454 | 150.4 | done | place a sauce pan with a spatula in it on the... |
| 9 | 963 | 15 | 0.00 | N | 66,974 | 147.6 | done | Turn on the lamp while holding the teapot. |
| 10 | 1071 | 15 | 0.00 | N | 67,304 | 138.1 | done | Clean a rag, put it away. |

</details>

### Sokoban — GPT-4.1

| Metric | Value |
|--------|-------|
| Success Rate | **8/20 = 40.0%** |
| Model Errors | 0/20 |
| Avg Reward | 0.86 |
| Avg Turns | 7.0 / 10 |

**Token Usage:**

| | Prompt | Completion | Total |
|---|------:|----------:|------:|
| Total (all episodes) | 277,330 | 7,345 | 284,675 |
| Avg per task | 14,234 | 367 | 14,234 |
| Avg per turn | 1,981 | 52 | 2,033 |

<details>
<summary>Per-Episode Results (20 episodes)</summary>

| # | Seed | Turns | Reward | OK? | Tokens | Reason | Task |
|--:|-----:|------:|-------:|:---:|-------:|--------|------|
| 1 | 315 | 10 | 0.50 | N | 21,337 | max_turns |  |
| 2 | 423 | 10 | 0.60 | N | 21,444 | max_turns |  |
| 3 | 207 | 1 | 1.10 | Y | 631 | done |  |
| 4 | 531 | 1 | 1.10 | Y | 601 | done |  |
| 5 | 100 | 10 | 0.70 | N | 22,124 | max_turns |  |
| 6 | 639 | 10 | 0.20 | N | 20,747 | max_turns |  |
| 7 | 747 | 10 | 0.60 | N | 21,754 | max_turns |  |
| 8 | 855 | 10 | 1.00 | N | 21,505 | max_turns |  |
| 9 | 963 | 10 | 0.10 | N | 20,961 | max_turns |  |
| 10 | 1179 | 10 | 1.00 | N | 22,385 | max_turns |  |
| 11 | 1071 | 10 | 0.10 | N | 20,815 | max_turns |  |
| 12 | 1287 | 10 | 0.60 | N | 21,945 | max_turns |  |
| 13 | 1395 | 2 | 1.20 | Y | 1,522 | done |  |
| 14 | 1503 | 10 | 0.80 | N | 21,780 | max_turns |  |
| 15 | 1611 | 2 | 1.20 | Y | 1,575 | done |  |
| 16 | 1719 | 9 | 1.90 | Y | 18,045 | done |  |
| 17 | 1935 | 2 | 1.20 | Y | 1,525 | done |  |
| 18 | 1827 | 10 | 1.00 | N | 21,844 | max_turns |  |
| 19 | 2043 | 1 | 1.10 | Y | 595 | done |  |
| 20 | 2151 | 2 | 1.20 | Y | 1,540 | done |  |

</details>

---

## Config Reference

| Config | Path |
|--------|------|
| EB-ALFRED Claude | `tests/eval_eb_alfred_claude_20ep.yaml` |
| EB-ALFRED GPT-4.1 | `tests/eval_eb_alfred_gpt41_20ep.yaml` |
| EB-ALFRED GPT-5 Mini (no-concat) | `tests/eval_eb_alfred_gpt5mini_20ep_noconcat.yaml` |
| EB-ALFRED GPT-5 Nano (no-concat) | `tests/eval_eb_alfred_gpt5nano_20ep_noconcat.yaml` |
| EB-ALFRED GPT-4.1 (no-concat, multi) | `tests/eval_eb_alfred_gpt41_10ep_noconcat_multi.yaml` |
| EB-ALFRED GPT-5.4 (no-concat, multi) | `tests/eval_eb_alfred_gpt54_10ep_noconcat_multi.yaml` |
| EB-ALFRED GPT-5 Mini (no-concat, multi) | `tests/eval_eb_alfred_gpt5mini_10ep_noconcat_multi.yaml` |
| EB-ALFRED GPT-5 Nano (no-concat, multi) | `tests/eval_eb_alfred_gpt5nano_10ep_noconcat_multi.yaml` |
| Sokoban GPT-4.1 | `tests/eval_sokoban_gpt41_20ep.yaml` |
