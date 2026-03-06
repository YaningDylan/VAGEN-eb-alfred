"""
Test concurrent environment creation to verify:
1. _CREATE_LOCK prevents X_DISPLAY race condition
2. _least_loaded_display distributes across GPUs with random tie-breaking
3. _pending_counts correctly tracks in-flight creations
"""

import asyncio
import json
import time
import httpx

SERVER = "http://localhost:8765"
N_CONCURRENT = 2  # Number of concurrent connect+reset requests


async def connect_and_reset(client: httpx.AsyncClient, idx: int, seed: int):
    """Send a connect request with seed (connect+reset in one round-trip)."""
    t0 = time.monotonic()
    data = json.dumps({"env_config": {"eval_set": "base", "resolution": 300}, "seed": seed})
    try:
        resp = await client.post(
            f"{SERVER}/connect",
            data={"data": data},
            timeout=300.0,
        )
        elapsed = time.monotonic() - t0
        print(f"  [{idx}] status={resp.status_code} elapsed={elapsed:.1f}s")
        return resp.status_code, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  [{idx}] ERROR after {elapsed:.1f}s: {e}")
        return None, elapsed


async def main():
    print(f"=== Concurrent creation test: {N_CONCURRENT} envs ===\n")

    # Check health
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SERVER}/health")
        print(f"Health: {r.json()}\n")

    # Launch N concurrent connect+reset requests
    print(f"Launching {N_CONCURRENT} concurrent connect+reset requests...")
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [
            connect_and_reset(client, i, seed=i)
            for i in range(N_CONCURRENT)
        ]
        results = await asyncio.gather(*tasks)

    total = time.monotonic() - t0
    print(f"\nTotal wall time: {total:.1f}s")

    successes = sum(1 for code, _ in results if code == 200)
    print(f"Successes: {successes}/{N_CONCURRENT}")

    # Check sessions
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{SERVER}/sessions")
        stats = r.json()
        print(f"\nActive sessions: {stats['num_sessions']}")
        for s in stats.get("sessions", []):
            print(f"  session={s['session_id'][:8]}... idle={s['idle_seconds']:.1f}s")

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
