"""
EB-ALFRED handler for the remote gym environment service.

This is the only component that needs customization.
It implements create_env() to instantiate EB-ALFRED environments.
"""

import asyncio
from typing import Any, Dict

from vagen.envs_remote.handler import BaseGymHandler
from .eb_alfred_env import EbAlfred


class EbAlfredHandler(BaseGymHandler):
    """Handler for EB-ALFRED remote environment service."""

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        """
        Create an EbAlfred environment instance.

        AI2-THOR startup is blocking, so we offload to a thread.
        """
        return await asyncio.to_thread(EbAlfred, env_config)
