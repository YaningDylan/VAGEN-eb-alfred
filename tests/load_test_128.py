"""
Load test: 128 concurrent clients against EB-ALFRED server.

Each client simulates one episode:
  connect → reset → sleep 5s (LLM inference) → step("look") → close

Measures total wall-clock time and per-phase timing.

Usage:
    python tests/load_test_128.py --url http://localhost:8000 --num-clients 128 --action-delay 5
"""

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ---------- Minimal multipart codec (inlined to avoid import issues) ----------

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
    """Extract JSON data from multipart response."""
    # Find boundary
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


# ---------- Client simulation ----------

@dataclass
class EpisodeResult:
    client_id: int
    session_id: Optional[str] = None
    connect_time: float = 0.0
    reset_time: float = 0.0
    step_time: float = 0.0
    close_time: float = 0.0
    total_time: float = 0.0
    error: Optional[str] = None
    queued: bool = False
    estimated_wait_s: float = 0.0


async def run_one_episode(
    client: httpx.AsyncClient,
    base_url: str,
    client_id: int,
    action_delay: float,
    env_config: dict,
    timeout: float,
) -> EpisodeResult:
    """Simulate one episode: connect → reset → delay → step → close."""
    result = EpisodeResult(client_id=client_id)
    t_start = time.time()

    try:
        # 1) Connect
        t0 = time.time()
        boundary, body = encode_multipart({"env_config": env_config})
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        resp = await client.post(f"{base_url}/connect", content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = decode_response(resp.headers.get("content-type", ""), resp.content)
        result.session_id = data.get("session_id")
        result.queued = data.get("status") == "queued"
        result.estimated_wait_s = data.get("estimated_wait_s", 0)
        result.connect_time = time.time() - t0

        if not result.session_id:
            result.error = "No session_id returned"
            return result

        # 2) Reset (may block if queued, waiting for capacity slot)
        t0 = time.time()
        boundary, body = encode_multipart({
            "session_id": result.session_id,
            "method": "reset",
            "params": {"seed": client_id},
        })
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        # Extended timeout for queued sessions
        reset_timeout = timeout + result.estimated_wait_s + 300
        resp = await client.post(f"{base_url}/call", content=body, headers=headers, timeout=reset_timeout)
        resp.raise_for_status()
        result.reset_time = time.time() - t0

        # 3) Simulate LLM inference delay
        await asyncio.sleep(action_delay)

        # 4) Step
        t0 = time.time()
        boundary, body = encode_multipart({
            "session_id": result.session_id,
            "method": "step",
            "params": {"action_str": "look"},
        })
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        resp = await client.post(f"{base_url}/call", content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result.step_time = time.time() - t0

        # 5) Close
        t0 = time.time()
        boundary, body = encode_multipart({
            "session_id": result.session_id,
            "method": "close",
            "params": {},
        })
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        resp = await client.post(f"{base_url}/call", content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result.close_time = time.time() - t0

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    result.total_time = time.time() - t_start
    return result


async def main():
    parser = argparse.ArgumentParser(description="Load test for EB-ALFRED server")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--num-clients", type=int, default=128, help="Number of concurrent clients")
    parser.add_argument("--action-delay", type=float, default=5.0, help="Simulated LLM delay (seconds)")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request")
    parser.add_argument("--eval-set", default="base", help="Eval set")
    parser.add_argument("--resolution", type=int, default=300, help="Image resolution")
    args = parser.parse_args()

    env_config = {
        "eval_set": args.eval_set,
        "resolution": args.resolution,
    }

    print(f"=== Load Test ===")
    print(f"Server:       {args.url}")
    print(f"Clients:      {args.num_clients}")
    print(f"Action delay: {args.action_delay}s")
    print(f"Env config:   {env_config}")
    print()

    # Use a single httpx client with high connection limits
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=200),
    ) as client:
        t_wall_start = time.time()

        # Launch all clients concurrently
        tasks = [
            run_one_episode(client, args.url, i, args.action_delay, env_config, args.timeout)
            for i in range(args.num_clients)
        ]

        # Track progress
        done_count = 0
        results: List[EpisodeResult] = []

        for coro in asyncio.as_completed(tasks):
            r = await coro
            done_count += 1
            results.append(r)
            status = "OK" if r.error is None else f"ERR: {r.error[:60]}"
            queued_str = " [queued]" if r.queued else ""
            print(
                f"  [{done_count:3d}/{args.num_clients}] client={r.client_id:3d}{queued_str} "
                f"total={r.total_time:6.1f}s "
                f"(connect={r.connect_time:.1f} reset={r.reset_time:.1f} "
                f"step={r.step_time:.1f} close={r.close_time:.1f}) "
                f"- {status}"
            )

        t_wall_total = time.time() - t_wall_start

    # Summary
    ok_results = [r for r in results if r.error is None]
    err_results = [r for r in results if r.error is not None]

    print()
    print(f"=== Summary ===")
    print(f"Wall clock:       {t_wall_total:.1f}s")
    print(f"Successful:       {len(ok_results)}/{args.num_clients}")
    print(f"Failed:           {len(err_results)}/{args.num_clients}")

    if ok_results:
        avg_total = sum(r.total_time for r in ok_results) / len(ok_results)
        avg_connect = sum(r.connect_time for r in ok_results) / len(ok_results)
        avg_reset = sum(r.reset_time for r in ok_results) / len(ok_results)
        avg_step = sum(r.step_time for r in ok_results) / len(ok_results)
        avg_close = sum(r.close_time for r in ok_results) / len(ok_results)
        max_total = max(r.total_time for r in ok_results)
        min_total = min(r.total_time for r in ok_results)

        print(f"\nTiming (successful episodes):")
        print(f"  Avg total:    {avg_total:.1f}s")
        print(f"  Min total:    {min_total:.1f}s")
        print(f"  Max total:    {max_total:.1f}s")
        print(f"  Avg connect:  {avg_connect:.1f}s")
        print(f"  Avg reset:    {avg_reset:.1f}s  (includes queue wait)")
        print(f"  Avg step:     {avg_step:.1f}s")
        print(f"  Avg close:    {avg_close:.1f}s")
        print(f"  Action delay: {args.action_delay}s (fixed)")

        queued = [r for r in ok_results if r.queued]
        print(f"\n  Queued:       {len(queued)}/{len(ok_results)}")

    if err_results:
        print(f"\nErrors:")
        for r in err_results[:10]:
            print(f"  client={r.client_id}: {r.error}")
        if len(err_results) > 10:
            print(f"  ... and {len(err_results) - 10} more")


if __name__ == "__main__":
    asyncio.run(main())
