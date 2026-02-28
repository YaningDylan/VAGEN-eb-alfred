"""Generate a unified markdown report for all evaluations."""
import json
import os
import glob
import datetime


def load_rollouts(rollout_dir):
    """Load all metrics from a rollout directory."""
    metrics_files = sorted(glob.glob(os.path.join(rollout_dir, "*/metrics.json")))
    rows = []
    for mf in metrics_files:
        with open(mf) as f:
            m = json.load(f)

        infos = m.get("infos", [])
        task = ""
        episode_idx = None
        for info_item in infos:
            if isinstance(info_item, dict):
                if "task_instruction" in info_item:
                    task = info_item["task_instruction"]
                if "episode_idx" in info_item:
                    episode_idx = info_item["episode_idx"]
                if task:
                    break

        error_msg = ""
        ed = m.get("error_details")
        if isinstance(ed, dict):
            error_msg = ed.get("message", "")[:100]
        elif isinstance(ed, str):
            error_msg = ed[:100]

        tu = m.get("token_usage", {})
        rows.append({
            "rollout_id": m.get("rollout_id", "?"),
            "seed": m.get("seed", "?"),
            "episode_idx": episode_idx,
            "task": task,
            "num_turns": m.get("num_turns", 0),
            "max_turns": m.get("max_turns", 30),
            "cumulative_reward": m.get("cumulative_reward", 0),
            "success": m.get("success", False),
            "finish_reason": m.get("finish_reason", "?"),
            "error": error_msg,
            "prompt_tokens": tu.get("prompt_tokens", 0),
            "completion_tokens": tu.get("completion_tokens", 0),
            "total_tokens": tu.get("total_tokens", 0),
        })
    return rows


def compute_stats(rows):
    """Compute summary statistics."""
    total = len(rows)
    successes = sum(1 for r in rows if r["success"])
    errors = [r for r in rows if r["finish_reason"] not in ("done", "max_turns")]
    clean = [r for r in rows if r["finish_reason"] in ("done", "max_turns")]
    clean_successes = sum(1 for r in clean if r["success"])

    token_rows = [r for r in rows if r["total_tokens"] > 0]
    total_prompt = sum(r["prompt_tokens"] for r in token_rows)
    total_completion = sum(r["completion_tokens"] for r in token_rows)
    total_all = sum(r["total_tokens"] for r in token_rows)
    total_turns = sum(r["num_turns"] for r in token_rows)

    return {
        "total": total,
        "successes": successes,
        "success_rate": successes / total * 100 if total else 0,
        "clean_total": len(clean),
        "clean_successes": clean_successes,
        "clean_rate": clean_successes / len(clean) * 100 if clean else 0,
        "num_errors": len(errors),
        "avg_reward": sum(r["cumulative_reward"] for r in rows) / total if total else 0,
        "avg_turns": sum(r["num_turns"] for r in rows) / total if total else 0,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "total_tokens": total_all,
        "avg_tokens_per_task": total_all / len(token_rows) if token_rows else 0,
        "avg_prompt_per_task": total_prompt / len(token_rows) if token_rows else 0,
        "avg_completion_per_task": total_completion / len(token_rows) if token_rows else 0,
        "avg_tokens_per_turn": total_all / total_turns if total_turns else 0,
        "avg_prompt_per_turn": total_prompt / total_turns if total_turns else 0,
        "avg_completion_per_turn": total_completion / total_turns if total_turns else 0,
    }


def env_section(env_name, model_name, rows, stats, max_turns):
    """Generate markdown section for one env+model combination."""
    lines = []
    lines.append(f"### {env_name} — {model_name}")
    lines.append("")

    # Summary table
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Success Rate | **{stats['successes']}/{stats['total']} = {stats['success_rate']:.1f}%** |")
    if stats["num_errors"] > 0:
        lines.append(f"| Clean Success Rate | {stats['clean_successes']}/{stats['clean_total']} = {stats['clean_rate']:.1f}% (excl. model errors) |")
    lines.append(f"| Model Errors | {stats['num_errors']}/{stats['total']} |")
    lines.append(f"| Avg Reward | {stats['avg_reward']:.2f} |")
    lines.append(f"| Avg Turns | {stats['avg_turns']:.1f} / {max_turns} |")
    lines.append("")

    # Token table (only if data exists)
    if stats["total_tokens"] > 0:
        lines.append("**Token Usage:**")
        lines.append("")
        lines.append("| | Prompt | Completion | Total |")
        lines.append("|---|------:|----------:|------:|")
        lines.append(f"| Total (all episodes) | {stats['total_prompt']:,} | {stats['total_completion']:,} | {stats['total_tokens']:,} |")
        lines.append(f"| Avg per task | {stats['avg_tokens_per_task']:,.0f} | {stats['avg_completion_per_task']:,.0f} | {stats['avg_tokens_per_task']:,.0f} |")
        lines.append(f"| Avg per turn | {stats['avg_prompt_per_turn']:,.0f} | {stats['avg_completion_per_turn']:,.0f} | {stats['avg_tokens_per_turn']:,.0f} |")
        lines.append("")

    # Per-episode table
    lines.append("<details>")
    lines.append(f"<summary>Per-Episode Results ({stats['total']} episodes)</summary>")
    lines.append("")
    lines.append("| # | Seed | Turns | Reward | OK? | Tokens | Reason | Task |")
    lines.append("|--:|-----:|------:|-------:|:---:|-------:|--------|------|")
    for i, r in enumerate(rows, 1):
        ok = "Y" if r["success"] else "N"
        t = r["task"][:45] + "..." if len(r["task"]) > 45 else r["task"]
        tok = f"{r['total_tokens']:,}" if r["total_tokens"] > 0 else "-"
        lines.append(f"| {i} | {r['seed']} | {r['num_turns']} | {r['cumulative_reward']:.2f} | {ok} | {tok} | {r['finish_reason']} | {t} |")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


# ── Load all data ──
base = "/home/march/workspace/Yaning/VAGEN-eb-alfred"

datasets = []

# 1. EB-ALFRED Claude
alfred_claude_dir = os.path.join(base, "tests/test_rollouts/claude_20ep/tag_eb_alfred_claude_20ep")
if os.path.isdir(alfred_claude_dir):
    rows = load_rollouts(alfred_claude_dir)
    stats = compute_stats(rows)
    datasets.append(("EB-ALFRED", "Claude Sonnet 4", rows, stats, 30))

# 2. EB-ALFRED GPT-4.1
alfred_gpt_dir = os.path.join(base, "tests/test_rollouts/gpt41_20ep/tag_eb_alfred_gpt41_20ep")
if os.path.isdir(alfred_gpt_dir):
    rows = load_rollouts(alfred_gpt_dir)
    stats = compute_stats(rows)
    datasets.append(("EB-ALFRED", "GPT-4.1", rows, stats, 30))

# 3. Sokoban GPT-4.1
sokoban_gpt_dir = os.path.join(base, "tests/test_rollouts/sokoban_gpt41_20ep/tag_sokoban_gpt41_20ep")
if os.path.isdir(sokoban_gpt_dir):
    rows = load_rollouts(sokoban_gpt_dir)
    stats = compute_stats(rows)
    datasets.append(("Sokoban", "GPT-4.1", rows, stats, 10))


# ── Build markdown ──
md = []
md.append("# VAGEN Inference Evaluation Report")
md.append("")
md.append(f"> Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
md.append("")

# Cross-env comparison table
md.append("## Overview")
md.append("")
md.append("| Environment | Model | Episodes | Success Rate | Avg Reward | Avg Turns | Errors | Tokens/Task | Tokens/Turn |")
md.append("|-------------|-------|:--------:|:------------:|-----------:|----------:|-------:|------------:|------------:|")
for env_name, model_name, rows, stats, max_turns in datasets:
    sr = f"**{stats['success_rate']:.1f}%**"
    tok_task = f"{stats['avg_tokens_per_task']:,.0f}" if stats["total_tokens"] > 0 else "N/A"
    tok_turn = f"{stats['avg_tokens_per_turn']:,.0f}" if stats["total_tokens"] > 0 else "N/A"
    md.append(f"| {env_name} | {model_name} | {stats['total']} | {sr} | {stats['avg_reward']:.2f} | {stats['avg_turns']:.1f}/{max_turns} | {stats['num_errors']} | {tok_task} | {tok_turn} |")
md.append("")

# Detailed sections
md.append("---")
md.append("")
md.append("## Detailed Results")
md.append("")
for env_name, model_name, rows, stats, max_turns in datasets:
    md.append(env_section(env_name, model_name, rows, stats, max_turns))

# Config reference
md.append("---")
md.append("")
md.append("## Config Reference")
md.append("")
md.append("| Config | Path |")
md.append("|--------|------|")
md.append("| EB-ALFRED Claude | `tests/eval_eb_alfred_claude_20ep.yaml` |")
md.append("| EB-ALFRED GPT-4.1 | `tests/eval_eb_alfred_gpt41_20ep.yaml` |")
md.append("| Sokoban GPT-4.1 | `tests/eval_sokoban_gpt41_20ep.yaml` |")
md.append("")

report_text = "\n".join(md)

out_path = os.path.join(base, "tests/test_rollouts/eval_comparison.md")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(report_text)

print(report_text)
print()
print("Saved to:", out_path)
