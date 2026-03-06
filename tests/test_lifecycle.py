"""
Test full env lifecycle: create → reset → step → close.
Measures timing at each stage and verifies cleanup.
"""
import asyncio
import json
import os
import subprocess
import time
import httpx

SERVER = "http://localhost:8765"


def count_unity_procs():
    """Count running Unity/Thor processes."""
    r = subprocess.run(["pgrep", "-c", "-f", "thor-201909"], capture_output=True, text=True)
    return int(r.stdout.strip()) if r.returncode == 0 else 0


async def main():
    async with httpx.AsyncClient() as c:
        # Health check
        r = await c.get(f"{SERVER}/health", timeout=5)
        print(f"Health: {r.json()}")
        print(f"Unity procs before: {count_unity_procs()}")

        # --- Connect (no reset, just create env) ---
        t0 = time.monotonic()
        data = json.dumps({"env_config": {"eval_set": "base", "resolution": 300}})
        r = await c.post(f"{SERVER}/connect", data={"data": data}, timeout=120)
        t_create = time.monotonic() - t0
        assert r.status_code == 200, f"Connect failed: {r.status_code}"
        # Parse multipart to get session_id
        from vagen.envs_remote.multipart_codec import decode_multipart
        ct = r.headers["content-type"]
        result_data, _ = decode_multipart(ct, r.content)
        sid = result_data["session_id"]
        print(f"\n1. CREATE: {t_create:.1f}s  session={sid[:8]}...")
        print(f"   Unity procs: {count_unity_procs()}")

        # --- Reset ---
        t0 = time.monotonic()
        data = json.dumps({"session_id": sid, "method": "reset", "params": {"seed": 0}})
        r = await c.post(f"{SERVER}/call", data={"data": data}, timeout=600)
        t_reset = time.monotonic() - t0
        print(f"\n2. RESET:  {t_reset:.1f}s  status={r.status_code}")
        print(f"   Unity procs: {count_unity_procs()}")

        # --- Step (simple action) ---
        t0 = time.monotonic()
        action = "<think>Looking around.</think><answer>find a mug</answer>"
        data = json.dumps({"session_id": sid, "method": "step", "params": {"action_str": action}})
        r = await c.post(f"{SERVER}/call", data={"data": data}, timeout=120)
        t_step = time.monotonic() - t0
        print(f"\n3. STEP:   {t_step:.1f}s  status={r.status_code}")

        # --- Close ---
        procs_before_close = count_unity_procs()
        t0 = time.monotonic()
        data = json.dumps({"session_id": sid, "method": "close", "params": {}})
        r = await c.post(f"{SERVER}/call", data={"data": data}, timeout=60)
        t_close = time.monotonic() - t0
        print(f"\n4. CLOSE:  {t_close:.1f}s  status={r.status_code}")

        # Wait a moment for process cleanup
        await asyncio.sleep(2)
        procs_after_close = count_unity_procs()
        print(f"   Unity procs: {procs_before_close} -> {procs_after_close}")

        # Check sessions
        r = await c.get(f"{SERVER}/sessions", timeout=5)
        stats = r.json()
        print(f"   Active sessions: {stats['num_sessions']}")

        # --- Summary ---
        print(f"\n{'='*40}")
        print(f"CREATE: {t_create:.1f}s")
        print(f"RESET:  {t_reset:.1f}s")
        print(f"STEP:   {t_step:.1f}s")
        print(f"CLOSE:  {t_close:.1f}s")
        print(f"Unity cleanup: {'OK' if procs_after_close < procs_before_close else 'LEAK!'}")
        print(f"Session cleanup: {'OK' if stats['num_sessions'] == 0 else 'LEAK!'}")


if __name__ == "__main__":
    asyncio.run(main())
