#!/usr/bin/env python3
"""
Aggregate EB-ALFRED benchmark results into a comprehensive markdown report.

Reads metrics.json files from each experiment's result directory, computes
summary statistics (success rate, token consumption, timing), and writes
a markdown report to benchmarking/BENCHMARK_REPORT.md.

Usage:
    cd VAGEN-eb-alfred
    python benchmarking/aggregate_results.py
"""

import json
import os
import sys
import statistics
from datetime import datetime
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJ_ROOT / "benchmarking" / "results"
REPORT_PATH = PROJ_ROOT / "benchmarking" / "BENCHMARK_REPORT.md"

# Experiment definitions: (dir_name, mode, resolution, n_episodes, concurrency)
EXPERIMENTS = [
    ("serial_500_10ep",      "serial",   500, 10,  1),
    ("parallel_500_128ep",   "parallel", 500, 128, 20),
    ("serial_328_10ep",      "serial",   328, 10,  1),
    ("parallel_328_128ep",   "parallel", 328, 128, 20),
    ("serial_96_10ep",       "serial",   96,  10,  1),
    ("parallel_96_128ep",    "parallel", 96,  128, 20),
]


def load_experiment(exp_dir: Path) -> dict:
    """Load all metrics.json files from an experiment directory."""
    episodes = []
    tag_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]

    # Results are nested: results/<exp>/tag_<tag_id>/<rollout_id>/metrics.json
    for tag_dir in tag_dirs:
        for rollout_dir in sorted(tag_dir.iterdir()):
            metrics_path = rollout_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                    m["_file_mtime"] = os.path.getmtime(metrics_path)
                    episodes.append(m)

    # Also check for flat layout: results/<exp>/<rollout_id>/metrics.json
    if not episodes:
        for rollout_dir in sorted(exp_dir.iterdir()):
            if rollout_dir.is_dir():
                metrics_path = rollout_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        m = json.load(f)
                        m["_file_mtime"] = os.path.getmtime(metrics_path)
                        episodes.append(m)

    return episodes


def compute_stats(episodes: list, mode: str, resolution: int, n_expected: int, concurrency: int) -> dict:
    """Compute aggregate statistics for an experiment."""
    if not episodes:
        return {"status": "no_data"}

    n = len(episodes)
    successes = sum(1 for e in episodes if e.get("success", False))
    success_rate = successes / n if n > 0 else 0

    rewards = [e.get("cumulative_reward", 0) for e in episodes]
    turns = [e.get("num_turns", 0) for e in episodes]

    # Token usage
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    num_calls = 0
    episodes_with_tokens = 0
    per_episode_tokens = []

    for e in episodes:
        tu = e.get("token_usage", {})
        if tu:
            episodes_with_tokens += 1
            pt = tu.get("prompt_tokens", 0)
            ct = tu.get("completion_tokens", 0)
            tt = tu.get("total_tokens", 0)
            nc = tu.get("num_calls", 0)
            prompt_tokens += pt
            completion_tokens += ct
            total_tokens += tt
            num_calls += nc
            per_episode_tokens.append(tt)

    # Wall time estimation from file modification times
    mtimes = [e["_file_mtime"] for e in episodes if "_file_mtime" in e]
    wall_time_s = max(mtimes) - min(mtimes) if len(mtimes) > 1 else 0

    # For parallel: effective per-episode time = wall_time / n_episodes
    # For serial: effective per-episode time = wall_time / n_episodes (same formula)
    effective_per_ep = wall_time_s / n if n > 0 and wall_time_s > 0 else 0

    # Finish reasons
    finish_reasons = {}
    for e in episodes:
        fr = e.get("finish_reason", "unknown")
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    stats = {
        "mode": mode,
        "resolution": f"{resolution}x{resolution}",
        "concurrency": concurrency,
        "n_expected": n_expected,
        "n_completed": n,
        "success_rate": round(success_rate * 100, 1),
        "successes": successes,
        "avg_reward": round(statistics.mean(rewards), 3) if rewards else 0,
        "avg_turns": round(statistics.mean(turns), 1) if turns else 0,
        "median_turns": round(statistics.median(turns), 1) if turns else 0,
        "min_turns": min(turns) if turns else 0,
        "max_turns": max(turns) if turns else 0,
        "wall_time_s": round(wall_time_s, 1),
        "wall_time_min": round(wall_time_s / 60, 1),
        "effective_per_ep_s": round(effective_per_ep, 2),
        "throughput_ep_per_min": round(n / (wall_time_s / 60), 1) if wall_time_s > 0 else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "num_api_calls": num_calls,
        "avg_tokens_per_episode": round(total_tokens / n, 0) if n > 0 and total_tokens > 0 else 0,
        "avg_prompt_per_episode": round(prompt_tokens / n, 0) if n > 0 and prompt_tokens > 0 else 0,
        "avg_completion_per_episode": round(completion_tokens / n, 0) if n > 0 and completion_tokens > 0 else 0,
        "avg_calls_per_episode": round(num_calls / n, 1) if n > 0 and num_calls > 0 else 0,
        "finish_reasons": finish_reasons,
    }
    return stats


def generate_report(all_stats: list) -> str:
    """Generate a markdown report from all experiment statistics."""
    lines = []
    lines.append("# EB-ALFRED Benchmark Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model**: GPT-4.1 (temperature=0)")
    lines.append(f"**Max turns per episode**: 30")
    lines.append(f"**Server**: EB-ALFRED on localhost:8000 (ai2thor 2.1.0)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Overview Table ──
    lines.append("## Overview")
    lines.append("")
    lines.append("| Experiment | Resolution | Mode | Episodes | Concurrency | Success Rate | Wall Time | Throughput |")
    lines.append("|:-----------|:----------:|:----:|:--------:|:-----------:|:------------:|:---------:|:----------:|")
    for s in all_stats:
        if s.get("status") == "no_data":
            lines.append(f"| {s['label']} | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {s['resolution']} {s['mode']} | {s['resolution']} | {s['mode']} | "
            f"{s['n_completed']}/{s['n_expected']} | {s['concurrency']} | "
            f"**{s['success_rate']}%** | {s['wall_time_min']} min | "
            f"{s['throughput_ep_per_min']} ep/min |"
        )
    lines.append("")

    # ── Timing Table ──
    lines.append("## Timing")
    lines.append("")
    lines.append("| Experiment | Wall Time (s) | Wall Time (min) | Effective per-Episode (s) | Throughput (ep/min) |")
    lines.append("|:-----------|:-------------:|:---------------:|:-------------------------:|:-------------------:|")
    for s in all_stats:
        if s.get("status") == "no_data":
            continue
        lines.append(
            f"| {s['resolution']} {s['mode']} | {s['wall_time_s']} | "
            f"{s['wall_time_min']} | {s['effective_per_ep_s']} | "
            f"{s['throughput_ep_per_min']} |"
        )
    lines.append("")
    lines.append("> **Effective per-episode** = total wall time / number of episodes (amortized time).")
    lines.append("")

    # ── Success & Reward Table ──
    lines.append("## Success & Rewards")
    lines.append("")
    lines.append("| Experiment | Completed | Success | Success Rate | Avg Reward | Avg Turns | Median Turns |")
    lines.append("|:-----------|:---------:|:-------:|:------------:|:----------:|:---------:|:------------:|")
    for s in all_stats:
        if s.get("status") == "no_data":
            continue
        lines.append(
            f"| {s['resolution']} {s['mode']} | {s['n_completed']} | "
            f"{s['successes']} | {s['success_rate']}% | {s['avg_reward']} | "
            f"{s['avg_turns']} | {s['median_turns']} |"
        )
    lines.append("")

    # ── Token Consumption Table ──
    lines.append("## Token Consumption")
    lines.append("")
    lines.append("| Experiment | Total Tokens | Prompt Tokens | Completion Tokens | API Calls | Avg Tokens/Ep | Avg Calls/Ep |")
    lines.append("|:-----------|:------------:|:-------------:|:-----------------:|:---------:|:-------------:|:------------:|")
    for s in all_stats:
        if s.get("status") == "no_data":
            continue
        lines.append(
            f"| {s['resolution']} {s['mode']} | {s['total_tokens']:,} | "
            f"{s['prompt_tokens']:,} | {s['completion_tokens']:,} | "
            f"{s['num_api_calls']:,} | {int(s['avg_tokens_per_episode']):,} | "
            f"{s['avg_calls_per_episode']} |"
        )
    lines.append("")

    # ── Per-Episode Token Breakdown ──
    lines.append("### Per-Episode Token Averages")
    lines.append("")
    lines.append("| Experiment | Avg Prompt/Ep | Avg Completion/Ep | Avg Total/Ep |")
    lines.append("|:-----------|:-------------:|:-----------------:|:------------:|")
    for s in all_stats:
        if s.get("status") == "no_data":
            continue
        lines.append(
            f"| {s['resolution']} {s['mode']} | {int(s['avg_prompt_per_episode']):,} | "
            f"{int(s['avg_completion_per_episode']):,} | {int(s['avg_tokens_per_episode']):,} |"
        )
    lines.append("")

    # ── Resolution Comparison ──
    lines.append("## Resolution Comparison")
    lines.append("")
    lines.append("### Serial (10 episodes each)")
    lines.append("")
    serial = [s for s in all_stats if s.get("mode") == "serial" and s.get("status") != "no_data"]
    if serial:
        lines.append("| Resolution | Success Rate | Avg Reward | Avg Tokens/Ep | Wall Time (s) |")
        lines.append("|:----------:|:------------:|:----------:|:-------------:|:-------------:|")
        for s in serial:
            lines.append(
                f"| {s['resolution']} | {s['success_rate']}% | {s['avg_reward']} | "
                f"{int(s['avg_tokens_per_episode']):,} | {s['wall_time_s']} |"
            )
    else:
        lines.append("*No serial results available yet.*")
    lines.append("")

    lines.append("### Parallel (128 episodes each, concurrency=20)")
    lines.append("")
    parallel = [s for s in all_stats if s.get("mode") == "parallel" and s.get("status") != "no_data"]
    if parallel:
        lines.append("| Resolution | Success Rate | Avg Reward | Avg Tokens/Ep | Wall Time (min) | Throughput (ep/min) |")
        lines.append("|:----------:|:------------:|:----------:|:-------------:|:---------------:|:-------------------:|")
        for s in parallel:
            lines.append(
                f"| {s['resolution']} | {s['success_rate']}% | {s['avg_reward']} | "
                f"{int(s['avg_tokens_per_episode']):,} | {s['wall_time_min']} | "
                f"{s['throughput_ep_per_min']} |"
            )
    else:
        lines.append("*No parallel results available yet.*")
    lines.append("")

    # ── Finish Reasons ──
    lines.append("## Finish Reasons")
    lines.append("")
    for s in all_stats:
        if s.get("status") == "no_data":
            continue
        fr = s.get("finish_reasons", {})
        reasons_str = ", ".join(f"{k}: {v}" for k, v in sorted(fr.items()))
        lines.append(f"- **{s['resolution']} {s['mode']}**: {reasons_str}")
    lines.append("")

    # ── Configuration ──
    lines.append("## Configuration")
    lines.append("")
    lines.append("```yaml")
    lines.append("model: gpt-4.1")
    lines.append("temperature: 0")
    lines.append("max_tokens: 1024")
    lines.append("max_turns: 30")
    lines.append("max_concurrency (API): 20")
    lines.append("max_concurrent_jobs (serial): 1")
    lines.append("max_concurrent_jobs (parallel): 20")
    lines.append("server: EB-ALFRED on localhost:8000")
    lines.append("GPU: NVIDIA RTX 5070 Ti (16 GB)")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    all_stats = []

    for dir_name, mode, resolution, n_episodes, concurrency in EXPERIMENTS:
        exp_dir = RESULTS_DIR / dir_name
        if not exp_dir.exists():
            print(f"  [SKIP] {dir_name}: directory not found")
            all_stats.append({"status": "no_data", "label": dir_name})
            continue

        episodes = load_experiment(exp_dir)
        if not episodes:
            print(f"  [SKIP] {dir_name}: no metrics.json files found")
            all_stats.append({"status": "no_data", "label": dir_name})
            continue

        stats = compute_stats(episodes, mode, resolution, n_episodes, concurrency)
        print(
            f"  [OK]   {dir_name}: {stats['n_completed']}/{n_episodes} episodes, "
            f"success={stats['success_rate']}%, tokens={stats['total_tokens']:,}"
        )
        all_stats.append(stats)

    # Generate report
    report = generate_report(all_stats)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to: {REPORT_PATH}")

    # Also save raw stats as JSON
    stats_path = RESULTS_DIR / "aggregate_stats.json"
    # Remove non-serializable fields
    clean_stats = []
    for s in all_stats:
        cs = {k: v for k, v in s.items() if not k.startswith("_")}
        clean_stats.append(cs)
    with open(stats_path, "w") as f:
        json.dump(clean_stats, f, indent=2)
    print(f"Stats JSON:  {stats_path}")


if __name__ == "__main__":
    main()
