# VAGEN Inference Evaluation Report

> Generated: 2026-02-28 13:32

## Overview

| Environment | Model | Episodes | Success Rate | Avg Reward | Avg Turns | Errors | Tokens/Task | Tokens/Turn |
|-------------|-------|:--------:|:------------:|-----------:|----------:|-------:|------------:|------------:|
| EB-ALFRED | Claude Sonnet 4 | 20 | **35.0%** | 1.73 | 13.8/30 | 5 | N/A | N/A |
| EB-ALFRED | GPT-4.1 | 20 | **60.0%** | 2.24 | 17.2/30 | 0 | 107,678 | 6,260 |
| Sokoban | GPT-4.1 | 20 | **40.0%** | 0.86 | 7.0/10 | 0 | 14,234 | 2,033 |

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
| Sokoban GPT-4.1 | `tests/eval_sokoban_gpt41_20ep.yaml` |
