#!/usr/bin/env python3
"""
EB-ALFRED Environment Throughput Benchmark.

Measures per-episode timing and total wall time for serial vs parallel
execution at different AI2-THOR rendering resolutions.

Prerequisites:
    Start the EB-ALFRED server first:
        conda run -n embench python -m vagen.envs.eb_alfred.serve \
            --port 8000 --max-sessions 64

Usage:
    cd VAGEN-eb-alfred

    # Exp 1: serial, 500x500, 10 episodes
    python benchmarking/bench_throughput.py \
        --resolution 500 --episodes 10 --mode serial

    # Exp 2: parallel, 500x500, 128 episodes, 20 concurrent
    python benchmarking/bench_throughput.py \
        --resolution 500 --episodes 128 --mode parallel --concurrency 20
"""

import asyncio
import argparse
import json
import os
import sys
import time
import statistics

# Add project root to path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from vagen.envs_remote.gym_image_env_client import GymImageEnvClient

# ── Constants ──────────────────────────────────────────────────────────
BASE_SEED = 42
STEPS_PER_EPISODE = 10
# A navigation action available in every EB-ALFRED episode.
DUMMY_ACTION = "<think>Benchmark: rotating to observe surroundings.</think><answer>RotateRight</answer>"


def generate_seeds(n):
    """Generate n deterministic seeds from BASE_SEED."""
    return [BASE_SEED + i * 7 for i in range(n)]


# ── Single episode runner ──────────────────────────────────────────────

async def run_episode(env_config, seed, episode_id):
    """Run one episode: connect → reset → N steps → close. Return timing."""
    t0 = time.monotonic()

    env = GymImageEnvClient(env_config)
    obs, info = await env.reset(seed=seed)
    t_reset = time.monotonic()

    actual_steps = 0
    for _ in range(STEPS_PER_EPISODE):
        obs, reward, done, step_info = await env.step(DUMMY_ACTION)
        actual_steps += 1
        if done:
            break
    t_steps = time.monotonic()

    await env.close()
    t_close = time.monotonic()

    return {
        "episode_id": episode_id,
        "seed": seed,
        "total_s": round(t_close - t0, 3),
        "reset_s": round(t_reset - t0, 3),
        "steps_s": round(t_steps - t_reset, 3),
        "close_s": round(t_close - t_steps, 3),
        "actual_steps": actual_steps,
    }


# ── Serial runner ─────────────────────────────────────────────────────

async def bench_serial(env_config, seeds):
    results = []
    for i, seed in enumerate(seeds):
        r = await run_episode(env_config, seed, i)
        results.append(r)
        print(
            f"  [{i+1:3d}/{len(seeds)}] seed={seed:4d}  "
            f"total={r['total_s']:6.2f}s  "
            f"reset={r['reset_s']:.2f}  steps={r['steps_s']:.2f}  close={r['close_s']:.2f}",
            flush=True,
        )
    return results


# ── Parallel runner ───────────────────────────────────────────────────

async def bench_parallel(env_config, seeds, max_concurrent):
    sem = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(seeds)
    results = []
    lock = asyncio.Lock()

    async def gated(seed, eid):
        nonlocal completed
        async with sem:
            r = await run_episode(env_config, seed, eid)
        async with lock:
            completed += 1
            results.append(r)
            if completed % 10 == 0 or completed == total:
                print(f"  [{completed:3d}/{total}] latest ep={eid}, {r['total_s']:.2f}s", flush=True)
        return r

    tasks = [asyncio.create_task(gated(seed, i)) for i, seed in enumerate(seeds)]
    await asyncio.gather(*tasks)
    return sorted(results, key=lambda x: x["episode_id"])


# ── Summary ───────────────────────────────────────────────────────────

def summarize(results, mode, resolution, max_concurrent, wall_time):
    totals = [r["total_s"] for r in results]
    resets = [r["reset_s"] for r in results]
    steps_all = [r["steps_s"] for r in results]

    summary = {
        "mode": mode,
        "resolution": f"{resolution}x{resolution}",
        "num_episodes": len(results),
        "max_concurrent": max_concurrent,
        "steps_per_episode": STEPS_PER_EPISODE,
        "base_seed": BASE_SEED,
        "wall_time_s": round(wall_time, 2),
        "wall_time_min": round(wall_time / 60, 2),
        "avg_episode_s": round(statistics.mean(totals), 3),
        "median_episode_s": round(statistics.median(totals), 3),
        "min_episode_s": round(min(totals), 3),
        "max_episode_s": round(max(totals), 3),
        "std_episode_s": round(statistics.stdev(totals), 3) if len(totals) > 1 else 0,
        "avg_reset_s": round(statistics.mean(resets), 3),
        "avg_steps_s": round(statistics.mean(steps_all), 3),
        "throughput_ep_per_min": round(len(results) / (wall_time / 60), 1),
    }

    print(f"\n{'='*60}")
    print(f"  Mode:           {mode}")
    print(f"  Resolution:     {resolution}x{resolution}")
    print(f"  Episodes:       {len(results)}")
    print(f"  Concurrency:    {max_concurrent}")
    print(f"  Steps/episode:  {STEPS_PER_EPISODE}")
    print(f"  ────────────────────────────────────")
    print(f"  Wall time:      {wall_time:.1f}s  ({wall_time/60:.1f} min)")
    print(f"  Avg episode:    {summary['avg_episode_s']:.2f}s")
    print(f"  Median episode: {summary['median_episode_s']:.2f}s")
    print(f"  Min / Max:      {summary['min_episode_s']:.2f}s / {summary['max_episode_s']:.2f}s")
    print(f"  Std dev:        {summary['std_episode_s']:.2f}s")
    print(f"  Throughput:     {summary['throughput_ep_per_min']:.1f} ep/min")
    print(f"  Avg reset:      {summary['avg_reset_s']:.2f}s")
    print(f"  Avg steps:      {summary['avg_steps_s']:.2f}s  ({STEPS_PER_EPISODE} steps)")
    print(f"{'='*60}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="EB-ALFRED Throughput Benchmark")
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--mode", choices=["serial", "parallel"], required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    seeds = generate_seeds(args.episodes)
    max_conc = args.concurrency if args.mode == "parallel" else 1

    env_config = {
        "base_urls": [args.server],
        "timeout": 600,
        "retries": 30,
        "backoff": 1.5,
        "eval_set": "base",
        "resolution": args.resolution,
        "max_turns": 30,
        "max_actions_per_step": 1,
        "action_sep": ",",
        "prompt_format": "free_think",
        "use_example_in_sys_prompt": True,
        "format_reward": 0.1,
        "success_reward": 1.0,
    }

    print(f"EB-ALFRED Throughput Benchmark")
    print(f"  Mode:        {args.mode}")
    print(f"  Resolution:  {args.resolution}x{args.resolution}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Concurrency: {max_conc}")
    print(f"  Server:      {args.server}")
    print(f"  Base seed:   {BASE_SEED}")
    print()

    t_start = time.monotonic()

    if args.mode == "serial":
        results = await bench_serial(env_config, seeds)
    else:
        results = await bench_parallel(env_config, seeds, max_conc)

    wall_time = time.monotonic() - t_start

    summary = summarize(results, args.mode, args.resolution, max_conc, wall_time)
    summary["episodes"] = results

    # Save
    out_dir = os.path.join(PROJ_ROOT, "benchmarking", args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"bench_{args.mode}_{args.resolution}_{args.episodes}ep.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {fpath}")


if __name__ == "__main__":
    asyncio.run(main())
