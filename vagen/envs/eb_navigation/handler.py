"""
EB-Navigation handler for the remote gym environment service.

This is the only component that needs customization.
It implements create_env() to instantiate EB-Navigation environments.
"""

import asyncio
from typing import Any, Dict

from vagen.envs_remote.handler import BaseGymHandler
from .eb_navigation_env import EbNavigation


class EbNavigationHandler(BaseGymHandler):
    """Handler for EB-Navigation remote environment service."""

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        """
        Create an EbNavigation environment instance.

        AI2-THOR startup is blocking, so we offload to a thread.
        """
        return await asyncio.to_thread(EbNavigation, env_config)
