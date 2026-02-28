"""
EB-Manipulation handler for the remote gym environment service.

This is the only component that needs customization.
It implements create_env() to instantiate EB-Manipulation environments.
"""

import asyncio
from typing import Any, Dict

from vagen.envs_remote.handler import BaseGymHandler
from .eb_manipulation_env import EbManipulation


class EbManipulationHandler(BaseGymHandler):
    """Handler for EB-Manipulation remote environment service."""

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        """
        Create an EbManipulation environment instance.

        CoppeliaSim/PyRep startup is blocking, so we offload to a thread.
        """
        return await asyncio.to_thread(EbManipulation, env_config)
