"""
Staggered close test: verify that when one env closes, the next queued env
starts immediately (not waiting for the whole batch to finish).

Each client has a DIFFERENT action delay (1-10s), so close times are staggered.
We log timestamps to verify that queued envs start creating as soon as a slot frees.

Usage:
    python tests/load_test_staggered.py --url http://localhost:8000 --num-clients 32 --capacity 8
"""

import argparse
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


import httpx


def encode_multipart(data: dict) -> Tuple[str, bytes]:
    boundary = f"loadtest_{uuid.uuid4().hex}"
    crlf = b"\r\n"
    bnd = boundary.encode()
    body = bytearray()
    data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    body += b"--" + bnd + crlf
    body += b'Content-Disposition: form-data; name="data"' + crlf
    body += b"Content-Type: application/json; charset=utf-8" + crlf + crlf
    body += data_bytes + crlf
    body += b"--" + bnd + b"--" + crlf
    return boundary, bytes(body)


def decode_response(content_type: str, body: bytes) -> dict:
    ct = content_type or ""
    boundary = ""
    for p in ct.split(";"):
        p = p.strip()
        if p.lower().startswith("boundary="):
            boundary = p.split("=", 1)[1].strip().strip('"')
    if not boundary:
        return json.loads(body)
    marker = ("--" + boundary).encode()
    for chunk in body.split(marker):
        chunk = chunk.strip()
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()
        header_blob, _, payload = chunk.partition(b"\r\n\r\n")
        if not payload:
            continue
        payload = payload.rstrip(b"\r\n")
        if b"application/json" in header_blob:
            return json.loads(payload)
    return {}


# Global timeline log: (timestamp_relative, event, client_id)
timeline: List[Tuple[float, str, int]] = []
T0 = 0.0


def log_event(event: str, client_id: int):
    timeline.append((time.time() - T0, event, client_id))


async def run_one_episode(
    client: httpx.AsyncClient,
    base_url: str,
    client_id: int,
    action_delay: float,
    env_config: dict,
    timeout: float,
) -> dict:
    result = {"client_id": client_id, "action_delay": action_delay, "error": None}

    try:
        # Connect
        log_event("connect_start", client_id)
        boundary, body = encode_multipart({"env_config": env_config})
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        resp = await client.post(f"{base_url}/connect", content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = decode_response(resp.headers.get("content-type", ""), resp.content)
        session_id = data.get("session_id")
        log_event("connect_done", client_id)

        if not session_id:
            result["error"] = "No session_id"
            return result

        # Reset (blocks until env is ready)
        log_event("reset_start", client_id)
        boundary, body = encode_multipart({
            "session_id": session_id, "method": "reset", "params": {"seed": client_id},
        })
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        resp = await client.post(f"{base_url}/call", content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        log_event("reset_done", client_id)
        result["reset_done_t"] = time.time() - T0

        # Staggered action delay
        log_event(f"action_wait_{action_delay:.1f}s", client_id)
        await asyncio.sleep(action_delay)

        # Step
        boundary, body = encode_multipart({
            "session_id": session_id, "method": "step", "params": {"action_str": "look"},
        })
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        resp = await client.post(f"{base_url}/call", content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()

        # Close (releases slot)
        log_event("close_start", client_id)
        boundary, body = encode_multipart({
            "session_id": session_id, "method": "close", "params": {},
        })
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        resp = await client.post(f"{base_url}/call", content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        log_event("close_done", client_id)
        result["close_done_t"] = time.time() - T0

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log_event(f"error", client_id)

    result["total_time"] = time.time() - T0
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--num-clients", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--resolution", type=int, default=300)
    args = parser.parse_args()

    env_config = {"eval_set": "base", "resolution": args.resolution}

    # Assign staggered delays: first batch (0-15) gets 1-8s, second batch gets 1-8s too
    # This ensures close times within each batch are spread out
    import random
    random.seed(42)
    delays = [random.uniform(1.0, 10.0) for _ in range(args.num_clients)]

    print(f"=== Staggered Close Test ===")
    print(f"Server:   {args.url}")
    print(f"Clients:  {args.num_clients}")
    print(f"Delays:   {[f'{d:.1f}' for d in delays[:8]]}... (1-10s random)")
    print()

    global T0
    T0 = time.time()

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=200),
    ) as client:
        tasks = [
            run_one_episode(client, args.url, i, delays[i], env_config, args.timeout)
            for i in range(args.num_clients)
        ]

        results = []
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            status = "OK" if r["error"] is None else f"ERR: {r['error'][:50]}"
            print(f"  [{len(results):3d}/{args.num_clients}] client={r['client_id']:3d} "
                  f"delay={r['action_delay']:.1f}s total={r['total_time']:.1f}s - {status}")

    wall = time.time() - T0

    # Sort timeline and print
    timeline.sort(key=lambda x: x[0])

    print(f"\n=== Timeline (close_done → next reset_done pairs) ===")
    print(f"Looking for: when a slot is freed (close_done), how quickly does the next queued env become ready (reset_done)?")
    print()

    # Find close_done events and the next reset_done that follows
    close_events = [(t, cid) for t, ev, cid in timeline if ev == "close_done"]
    reset_events = [(t, cid) for t, ev, cid in timeline if ev == "reset_done"]

    # Sort both by time
    close_events.sort()
    reset_events.sort()

    # For each close, find the next reset_done that happens AFTER it (from a different batch)
    # We know first 16 resets happen at ~7s (batch 1). After that, each close should trigger a new env.
    print(f"{'Time':>7s}  {'Event':>12s}  {'Client':>6s}")
    print("-" * 35)

    # Print interleaved close/reset events after the first batch
    first_batch_end = 0
    for t, ev, cid in timeline:
        if ev in ("close_done", "reset_done") and t > 12.0:  # after first batch
            if first_batch_end == 0:
                first_batch_end = t
            print(f"{t:7.1f}s  {ev:>12s}  client={cid}")
            if t - first_batch_end > 40:  # limit output
                print("  ...")
                break

    # Compute gap: time between each close_done and the next reset_done
    print(f"\n=== Slot Reuse Latency ===")
    print(f"Gap = time from close_done to the next queued reset_done")
    print()

    close_times = sorted([t for t, _ in close_events])
    reset_times = sorted([t for t, _ in reset_events])

    # Remove first-batch resets (those happen at ~7-10s, not triggered by closes)
    # First batch resets are the first 16
    late_resets = reset_times[16:]  # resets from batch 2+

    gaps = []
    ci = 0
    for rt in late_resets:
        # Find the close that happened just before this reset
        while ci < len(close_times) - 1 and close_times[ci + 1] < rt:
            ci += 1
        if ci < len(close_times) and close_times[ci] < rt:
            gap = rt - close_times[ci]
            gaps.append(gap)

    if gaps:
        print(f"  Samples:  {len(gaps)}")
        print(f"  Min gap:  {min(gaps):.1f}s")
        print(f"  Max gap:  {max(gaps):.1f}s")
        print(f"  Avg gap:  {sum(gaps)/len(gaps):.1f}s")
        print(f"  Median:   {sorted(gaps)[len(gaps)//2]:.1f}s")
        print()
        print(f"  If avg gap ≈ env_creation_time (~7s), slots are reused IMMEDIATELY.")
        print(f"  If avg gap >> env_creation_time, there's unnecessary batching delay.")
    else:
        print("  No late resets found (all fit in first batch?)")

    # Summary
    ok = sum(1 for r in results if r["error"] is None)
    print(f"\n=== Summary ===")
    print(f"Wall clock:   {wall:.1f}s")
    print(f"Success:      {ok}/{args.num_clients}")
    print(f"Failed:       {args.num_clients - ok}/{args.num_clients}")


if __name__ == "__main__":
    asyncio.run(main())
