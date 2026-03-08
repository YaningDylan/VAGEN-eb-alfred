"""Test how many concurrent AI2-THOR envs can start+reset successfully."""
import asyncio
import os
import sys
import time
import concurrent.futures

# Must be run inside embench-vagen conda env
os.environ["DISPLAY"] = ":0"

from vagen.envs.eb_alfred.eb_alfred_env import EbAlfred

ENV_CONFIG = {
    "eval_set": "base",
    "x_display": "0",
    "obs_image_size": 500,
    "max_turns": 5,
    "max_actions_per_step": 20,
    "max_env_steps": 5,
    "action_sep": ",",
    "prompt_format": "free_think",
    "use_example_in_sys_prompt": True,
    "format_reward": 0.1,
    "success_reward": 1.0,
}


def start_and_reset(idx, seed=0):
    """Start one env and reset it. Returns (idx, elapsed, success, error)."""
    t0 = time.time()
    try:
        env = EbAlfred(ENV_CONFIG)
        create_time = time.time() - t0
        print(f"[{idx}] EbAlfred created in {create_time:.1f}s, resetting with seed={seed}...")
        loop = asyncio.new_event_loop()
        obs, info = loop.run_until_complete(env.reset(seed))
        loop.close()
        elapsed = time.time() - t0
        print(f"[{idx}] Reset done in {elapsed:.1f}s total")
        return idx, elapsed, True, None, env
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{idx}] FAILED after {elapsed:.1f}s: {e}")
        return idx, elapsed, False, str(e), None


def test_concurrent(n):
    """Test n concurrent env startups."""
    print(f"\n{'='*60}")
    print(f"Testing {n} concurrent envs...")
    print(f"{'='*60}")

    t0 = time.time()
    envs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(start_and_reset, i, i % 4) for i in range(n)]

        # Wait with 300s timeout (same as cloud client)
        done, not_done = concurrent.futures.wait(futures, timeout=300)

        total_time = time.time() - t0

        succeeded = 0
        failed = 0
        timed_out = len(not_done)

        for f in done:
            idx, elapsed, success, error, env = f.result()
            if success:
                succeeded += 1
                envs.append(env)
            else:
                failed += 1

        print(f"\n--- Results for n={n} ---")
        print(f"Succeeded: {succeeded}/{n}")
        print(f"Failed: {failed}/{n}")
        print(f"Timed out: {timed_out}/{n}")
        print(f"Total time: {total_time:.1f}s")

        # Cleanup
        print("Cleaning up envs...")
        import asyncio as _aio
        for env in envs:
            try:
                _loop = _aio.new_event_loop()
                _loop.run_until_complete(env.close())
                _loop.close()
            except:
                pass

        # Cancel timed out
        for f in not_done:
            f.cancel()

    return succeeded, failed, timed_out, total_time


if __name__ == "__main__":
    counts = [1, 2, 4, 8]
    if len(sys.argv) > 1:
        counts = [int(x) for x in sys.argv[1:]]

    results = {}
    for n in counts:
        s, f, t, elapsed = test_concurrent(n)
        results[n] = (s, f, t, elapsed)

        if s < n:
            print(f"\n*** n={n} failed ({s}/{n} succeeded). Stopping here. ***")
            break

        # Wait between tests for cleanup
        print("Waiting 10s for cleanup...")
        time.sleep(10)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for n, (s, f, t, elapsed) in results.items():
        status = "OK" if s == n else "PARTIAL" if s > 0 else "FAIL"
        print(f"  n={n}: {s}/{n} succeeded, {f} failed, {t} timed out, {elapsed:.1f}s [{status}]")
