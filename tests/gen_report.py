"""Generate evaluation report from rollout metrics."""
import json
import os
import glob
import datetime

rollout_dir = "tests/test_rollouts/claude_20ep/tag_eb_alfred_claude_20ep"
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

    rows.append(
        {
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
        }
    )

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

L = []
L.append("=" * 90)
L.append("  EB-ALFRED Inference Evaluation Report")
L.append("  Model: Claude Sonnet 4 (claude-sonnet-4-20250514)")
L.append("  Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
L.append("  Eval Set: base  |  Max Turns: 30  |  Episodes: " + str(total))
L.append("=" * 90)
L.append("")
L.append("## Summary")
L.append("  Overall Success Rate:  %d/%d = %.1f%%" % (successes, total, success_rate))
L.append(
    "  Clean Success Rate:    %d/%d = %.1f%%  (excluding model errors)"
    % (clean_successes, clean_total, clean_rate)
)
L.append("  Average Reward:        %.2f" % avg_reward)
L.append("  Average Turns:         %.1f" % avg_turns)
L.append("  Model Errors:          %d/%d" % (len(done_error), total))
L.append("")
L.append("## Per-Episode Results")
L.append("")
L.append(
    "%2s  %6s  %3s  %5s  %7s  %4s  %12s  %s"
    % ("#", "Seed", "Ep", "Turns", "Reward", "OK?", "Reason", "Task")
)
L.append("-" * 100)
for i, r in enumerate(rows, 1):
    ok = "Y" if r["success"] else "N"
    ep = r["episode_idx"] if r["episode_idx"] is not None else "?"
    t = r["task"][:50] + "..." if len(r["task"]) > 50 else r["task"]
    L.append(
        "%2d  %6s  %3s  %5d  %7.2f  %4s  %12s  %s"
        % (i, r["seed"], ep, r["num_turns"], r["cumulative_reward"], ok, r["finish_reason"], t)
    )
L.append("-" * 100)
L.append("")

L.append("## Successful Episodes")
for r in rows:
    if r["success"]:
        L.append(
            "  [seed=%s] %s (turns=%d, reward=%.2f)"
            % (r["seed"], r["task"], r["num_turns"], r["cumulative_reward"])
        )
L.append("")

L.append("## Failed Episodes (no model error)")
for r in rows:
    if not r["success"] and r["finish_reason"] in ("done", "max_turns"):
        L.append(
            "  [seed=%s] %s (turns=%d, reason=%s)"
            % (r["seed"], r["task"], r["num_turns"], r["finish_reason"])
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
L.append("=" * 90)
L.append("  Rollout data: " + os.path.abspath(rollout_dir))
L.append("=" * 90)

report = "\n".join(L)

# Write report
rp = "tests/test_rollouts/claude_20ep/eval_report.txt"
os.makedirs(os.path.dirname(rp), exist_ok=True)
with open(rp, "w") as f:
    f.write(report)

# Write JSON summary
sp = "tests/test_rollouts/claude_20ep/eval_summary.json"
summary = {
    "model": "claude-sonnet-4-20250514",
    "eval_set": "base",
    "num_episodes": total,
    "success_rate": round(success_rate, 1),
    "clean_success_rate": round(clean_rate, 1),
    "num_successes": successes,
    "num_model_errors": len(done_error),
    "avg_reward": round(avg_reward, 2),
    "avg_turns": round(avg_turns, 1),
    "episodes": rows,
}
with open(sp, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(report)
print()
print("Report saved to:", os.path.abspath(rp))
print("JSON summary saved to:", os.path.abspath(sp))
