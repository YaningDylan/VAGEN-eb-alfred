"""Generate evaluation report from GPT-4.1 rollout metrics, including token usage."""
import json
import os
import glob
import datetime

rollout_dir = "tests/test_rollouts/gpt41_20ep/tag_eb_alfred_gpt41_20ep"
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

# ── Stats ──
total = len(rows)
successes = sum(1 for r in rows if r["success"])
done_error = [r for r in rows if r["finish_reason"] not in ("done", "max_turns")]
success_rate = successes / total * 100 if total else 0
avg_reward = sum(r["cumulative_reward"] for r in rows) / total if total else 0
avg_turns = sum(r["num_turns"] for r in rows) / total if total else 0

clean_rows = [r for r in rows if r["finish_reason"] in ("done", "max_turns")]
clean_total = len(clean_rows)
clean_successes = sum(1 for r in clean_rows if r["success"])
clean_rate = clean_successes / clean_total * 100 if clean_total else 0

# Token stats (only for episodes with data)
token_rows = [r for r in rows if r["total_tokens"] > 0]
total_prompt = sum(r["prompt_tokens"] for r in token_rows)
total_completion = sum(r["completion_tokens"] for r in token_rows)
total_all = sum(r["total_tokens"] for r in token_rows)
total_turns_with_tokens = sum(r["num_turns"] for r in token_rows)

avg_tokens_per_task = total_all / len(token_rows) if token_rows else 0
avg_prompt_per_task = total_prompt / len(token_rows) if token_rows else 0
avg_completion_per_task = total_completion / len(token_rows) if token_rows else 0
avg_tokens_per_turn = total_all / total_turns_with_tokens if total_turns_with_tokens else 0
avg_prompt_per_turn = total_prompt / total_turns_with_tokens if total_turns_with_tokens else 0
avg_completion_per_turn = total_completion / total_turns_with_tokens if total_turns_with_tokens else 0

# ── Build Report ──
L = []
L.append("=" * 95)
L.append("  EB-ALFRED Inference Evaluation Report")
L.append("  Model: GPT-4.1")
L.append("  Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
L.append("  Eval Set: base  |  Max Turns: 30  |  Episodes: " + str(total))
L.append("=" * 95)
L.append("")

L.append("## Summary")
L.append("  Overall Success Rate:  %d/%d = %.1f%%" % (successes, total, success_rate))
L.append("  Clean Success Rate:    %d/%d = %.1f%%  (excluding model errors)" % (clean_successes, clean_total, clean_rate))
L.append("  Average Reward:        %.2f" % avg_reward)
L.append("  Average Turns:         %.1f" % avg_turns)
L.append("  Model Errors:          %d/%d" % (len(done_error), total))
L.append("")

L.append("## Token Usage")
L.append("  ┌─────────────────────────┬──────────────┬──────────────┬──────────────┐")
L.append("  │                         │   Prompt     │  Completion  │    Total     │")
L.append("  ├─────────────────────────┼──────────────┼──────────────┼──────────────┤")
L.append("  │ Total (all episodes)    │ %12s │ %12s │ %12s │" % (f"{total_prompt:,}", f"{total_completion:,}", f"{total_all:,}"))
L.append("  │ Avg per task            │ %12s │ %12s │ %12s │" % (f"{avg_prompt_per_task:,.0f}", f"{avg_completion_per_task:,.0f}", f"{avg_tokens_per_task:,.0f}"))
L.append("  │ Avg per turn            │ %12s │ %12s │ %12s │" % (f"{avg_prompt_per_turn:,.0f}", f"{avg_completion_per_turn:,.0f}", f"{avg_tokens_per_turn:,.0f}"))
L.append("  └─────────────────────────┴──────────────┴──────────────┴──────────────┘")
L.append("")

L.append("## Per-Episode Results")
L.append("")
L.append("%2s  %6s  %3s  %5s  %7s  %4s  %10s  %8s  %s" % ("#", "Seed", "Ep", "Turns", "Reward", "OK?", "Tokens", "Reason", "Task"))
L.append("-" * 105)
for i, r in enumerate(rows, 1):
    ok = "Y" if r["success"] else "N"
    ep = r["episode_idx"] if r["episode_idx"] is not None else "?"
    t = r["task"][:42] + "..." if len(r["task"]) > 42 else r["task"]
    tok = f"{r['total_tokens']:,}" if r["total_tokens"] > 0 else "-"
    L.append(
        "%2d  %6s  %3s  %5d  %7.2f  %4s  %10s  %10s  %s"
        % (i, r["seed"], ep, r["num_turns"], r["cumulative_reward"], ok, tok, r["finish_reason"], t)
    )
L.append("-" * 105)
L.append("")

L.append("## Successful Episodes")
for r in rows:
    if r["success"]:
        L.append(
            "  [seed=%s] %s (turns=%d, reward=%.2f, tokens=%s)"
            % (r["seed"], r["task"], r["num_turns"], r["cumulative_reward"], f"{r['total_tokens']:,}")
        )
L.append("")

L.append("## Failed Episodes (no model error)")
for r in rows:
    if not r["success"] and r["finish_reason"] in ("done", "max_turns"):
        L.append(
            "  [seed=%s] %s (turns=%d, reason=%s, tokens=%s)"
            % (r["seed"], r["task"], r["num_turns"], r["finish_reason"], f"{r['total_tokens']:,}")
        )

if done_error:
    L.append("")
    L.append("## Model Errors")
    for r in rows:
        if r["finish_reason"] not in ("done", "max_turns"):
            L.append(
                "  [seed=%s] reason=%s, error: %s"
                % (r["seed"], r["finish_reason"], r["error"][:80])
            )

L.append("")
L.append("=" * 95)
L.append("  Rollout data: " + os.path.abspath(rollout_dir))
L.append("=" * 95)

report = "\n".join(L)

# Write files
rp = "tests/test_rollouts/gpt41_20ep/eval_report.txt"
os.makedirs(os.path.dirname(rp), exist_ok=True)
with open(rp, "w") as f:
    f.write(report)

sp = "tests/test_rollouts/gpt41_20ep/eval_summary.json"
summary = {
    "model": "gpt-4.1",
    "eval_set": "base",
    "num_episodes": total,
    "success_rate": round(success_rate, 1),
    "clean_success_rate": round(clean_rate, 1),
    "num_successes": successes,
    "num_model_errors": len(done_error),
    "avg_reward": round(avg_reward, 2),
    "avg_turns": round(avg_turns, 1),
    "token_stats": {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_all,
        "avg_tokens_per_task": round(avg_tokens_per_task),
        "avg_prompt_per_task": round(avg_prompt_per_task),
        "avg_completion_per_task": round(avg_completion_per_task),
        "avg_tokens_per_turn": round(avg_tokens_per_turn),
        "avg_prompt_per_turn": round(avg_prompt_per_turn),
        "avg_completion_per_turn": round(avg_completion_per_turn),
    },
    "episodes": rows,
}
with open(sp, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(report)
print()
print("Report saved to:", os.path.abspath(rp))
print("JSON summary saved to:", os.path.abspath(sp))
