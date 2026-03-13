"""
EB-ALFRED handler for the remote gym environment service.

This is the only component that needs customization.
It implements create_env() to instantiate EB-ALFRED environments
with automatic multi-GPU load balancing.
"""

import asyncio
import logging
import os
import signal
import subprocess
from typing import Any, Dict, List, Optional

from vagen.envs_remote.handler import BaseGymHandler
from .eb_alfred_env import EbAlfred

LOGGER = logging.getLogger(__name__)

# Limit concurrent env creations to avoid CPU saturation from many Unity startups
_ENV_CREATION_SEMAPHORE: Optional[asyncio.Semaphore] = None

def _get_creation_semaphore() -> asyncio.Semaphore:
    global _ENV_CREATION_SEMAPHORE
    if _ENV_CREATION_SEMAPHORE is None:
        _ENV_CREATION_SEMAPHORE = asyncio.Semaphore(1)
    return _ENV_CREATION_SEMAPHORE


def detect_gpu_displays() -> List[str]:
    """Auto-detect available GPUs via nvidia-smi, return display list.

    Assumes display :i maps to GPU i (standard convention for
    multi-GPU X server setups with Xvfb or xinit per GPU).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            indices = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            if indices:
                return indices
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ["0"]


class EbAlfredHandler(BaseGymHandler):
    """Handler for EB-ALFRED with automatic multi-GPU load balancing.

    By default, auto-detects available GPUs and distributes new
    sessions to the least-loaded GPU. Single-GPU is just the
    special case where only one GPU is detected.
    """

    def __init__(
        self,
        x_displays: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Args:
            x_displays: List of X display IDs to use (e.g. ["0", "1"]).
                         None = auto-detect GPUs via nvidia-smi.
            **kwargs: Passed to BaseGymHandler (session_timeout, max_sessions).
        """
        super().__init__(**kwargs)
        self._x_displays = x_displays if x_displays is not None else detect_gpu_displays()
        LOGGER.info(f"[Handler] Using X displays: {self._x_displays}")

    _creation_counter: int = 0

    def _least_loaded_display(self) -> str:
        """Pick the display with the fewest active sessions.
        Breaks ties with a round-robin counter to distribute evenly."""
        counts = {d: 0 for d in self._x_displays}
        for ctx in self._sessions.values():
            d = getattr(ctx.env, "_assigned_display", None)
            if d in counts:
                counts[d] += 1
        min_count = min(counts.values())
        candidates = [d for d, c in counts.items() if c == min_count]
        chosen = candidates[self._creation_counter % len(candidates)]
        self._creation_counter += 1
        LOGGER.debug(f"[Handler] GPU load: {counts}, assigning display :{chosen}")
        return chosen

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        """
        Create an EbAlfred environment on the least-loaded GPU.

        AI2-THOR startup is blocking, so we offload to a thread.
        Uses a semaphore to serialize creations (Unity startup is very CPU-heavy).
        On timeout, kills any leaked Thor/Unity child processes.
        """
        display = self._least_loaded_display()
        env_config = {**env_config, "x_display": display}

        # Track Thor PIDs before creation so we can clean up leaks on timeout
        def _get_thor_pids():
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "thor-201909"],
                    capture_output=True, text=True, timeout=5,
                )
                return set(int(p) for p in result.stdout.strip().split('\n') if p.strip())
            except Exception:
                return set()

        async with _get_creation_semaphore():
            LOGGER.warning(f"[Handler] Creating env on display :{display} (semaphore acquired)")
            pids_before = _get_thor_pids()

            try:
                env = await asyncio.wait_for(
                    asyncio.to_thread(EbAlfred, env_config),
                    timeout=180.0,
                )
            except asyncio.TimeoutError:
                LOGGER.error(f"[Handler] Env creation timed out on display :{display}")
                # Kill any new Thor processes that were spawned by the leaked thread
                pids_after = _get_thor_pids()
                leaked = pids_after - pids_before
                for pid in leaked:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        LOGGER.warning(f"[Handler] Killed leaked Thor process {pid}")
                    except OSError:
                        pass
                raise RuntimeError(f"Env creation timed out on display :{display}")

        env._assigned_display = display
        LOGGER.info(
            f"[Handler] Created env on display :{display} "
            f"(GPU load: { {d: sum(1 for c in self._sessions.values() if getattr(c.env, '_assigned_display', None) == d) for d in self._x_displays} })"
        )
        return env
