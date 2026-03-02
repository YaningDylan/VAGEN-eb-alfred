"""
EB-Navigation environment adapter for VAGEN.

Wraps the EBNavigationEnv from EmbodiedBench as a GymImageEnv,
enabling integration with VAGEN's RL training and evaluation pipeline.

The underlying EBNavigationEnv uses AI2-THOR for 3D household navigation.
It requires a GPU with CloudRendering support.
"""

import asyncio
import os
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional

from .utils.prompt import (
    DISCRETE_SKILLSET,
    system_prompt,
    format_prompt,
    init_observation_template,
    action_template,
)
from .utils.utils import parse_response, match_action, numpy_to_pil

from vagen.envs.gym_image_env import GymImageEnv


@dataclass
class EbNavigationEnvConfig:
    """Configuration for EB-Navigation environment."""

    # Environment settings
    eval_set: str = "base"
    exp_name: str = "vagen_eval"
    down_sample_ratio: float = 1.0
    resolution: int = 500
    fov: int = 100
    selected_indexes: List[int] = field(default_factory=list)
    detection_box: bool = False

    # Interaction settings
    max_turns: int = 20
    max_actions_per_step: int = 1
    action_sep: str = ","
    image_placeholder: str = "<image>"
    prompt_format: str = "free_think"
    use_example_in_sys_prompt: bool = True

    # Observation image settings
    obs_image_size: Optional[int] = None  # Resize obs image to this size (square). None = use original.

    # Reward settings
    format_reward: float = 0.1
    success_reward: float = 1.0


class EbNavigation(GymImageEnv):
    """
    EB-Navigation environment implementing the GymImageEnv async interface.

    Wraps EBNavigationEnv from EmbodiedBench, which uses AI2-THOR for
    3D household navigation tasks (e.g., "navigate to the Bread in the room").

    Key features:
    - 8 discrete actions (move, rotate, tilt camera)
    - Fixed action space across all episodes
    - Vision-only observations (RGB images from AI2-THOR)
    - Binary reward: success when within 1m of target object
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        # Filter config keys to only those in the dataclass
        valid_keys = EbNavigationEnvConfig.__dataclass_fields__
        filtered = {k: v for k, v in env_config.items() if k in valid_keys}
        self.config = EbNavigationEnvConfig(**filtered)

        from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv

        self.env = EBNavigationEnv(
            eval_set=self.config.eval_set,
            exp_name=self.config.exp_name,
            down_sample_ratio=self.config.down_sample_ratio,
            resolution=self.config.resolution,
            fov=self.config.fov,
            selected_indexes=self.config.selected_indexes,
            boundingbox=self.config.detection_box,
        )

        # Action lookup (fixed for all episodes)
        self._action_list = list(DISCRETE_SKILLSET)
        self._action_map = {a.lower(): a for a in self._action_list}

        # Adapter state (reset per episode)
        self._total_turns: int = 0
        self._last_action: str = ""
        self._last_feedback: str = ""

    # ------------------------------------------------------------------
    # GymImageEnv abstract methods
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close AI2-THOR process."""
        await asyncio.to_thread(self.env.close)

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Return the static system prompt.

        Includes role description, action descriptions, guidelines,
        and format instructions. Navigation has a fixed action space
        so all action info is in the system prompt.
        """
        sys_str = system_prompt()
        fmt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.use_example_in_sys_prompt,
            prompt_format=self.config.prompt_format,
        )
        return {"obs_str": sys_str + "\n\n" + fmt_str}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment for a new episode.

        The seed selects which episode to load from the dataset
        (seed % number_of_episodes). After reset, the observation
        includes the task instruction and the initial RGB image.
        """
        # Select episode based on seed
        episode_idx = seed % self.env.number_of_episodes
        self.env._current_episode_num = episode_idx

        await asyncio.to_thread(self.env.reset)

        # Reset adapter state
        self._total_turns = 0
        self._last_action = ""
        self._last_feedback = ""

        # Build observation
        obs = self._build_obs(init=True)
        info = {
            "task_instruction": self.env.episode_language_instruction,
            "num_actions": len(self._action_list),
            "eval_set": self.config.eval_set,
            "episode_idx": episode_idx,
        }
        return obs, info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step given the LLM's response.

        Parses <think>...</think><answer>...</answer> from action_str,
        matches the action against the 8 discrete navigation actions,
        and executes it in AI2-THOR.
        """
        self._total_turns += 1

        # Parse LLM response
        parsed = parse_response(
            response=action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
            prompt_format=self.config.prompt_format,
        )

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        actions = parsed.get("actions", [])
        format_correct = parsed.get("format_correct", False)

        metrics = {
            "turn_metrics": {
                "action_is_valid": False,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        if format_correct and actions:
            reward += self.config.format_reward

            for action_name in actions:
                action_idx = match_action(action_name, self._action_list, self._action_map)

                if action_idx is None:
                    # Action name not recognized
                    self._last_action = action_name
                    self._last_feedback = (
                        f"Action '{action_name}' is not a recognized action. "
                        f"Please use one of the 8 available actions (ID 0-7)."
                    )
                    break

                metrics["turn_metrics"]["action_is_valid"] = True

                # Execute in AI2-THOR via the underlying env
                # EBNavigationEnv.step expects (action_int, reasoning, i_flag)
                obs_raw, step_reward, step_done, step_info = (
                    await asyncio.to_thread(
                        self.env.step, action_idx, {}, 0
                    )
                )

                self._last_action = self._action_list[action_idx]
                self._last_feedback = step_info.get("env_feedback", "")

                action_success = step_info.get("last_action_success", False)
                if action_success:
                    metrics["turn_metrics"]["action_is_effective"] = True

                task_success = step_info.get("task_success", 0.0)
                if task_success > 0:
                    done = True
                    reward += self.config.success_reward
                    metrics["traj_metrics"]["success"] = True
                    break

                if step_done:
                    done = True
                    break
        else:
            # Format error: no valid actions parsed
            self._last_action = parsed.get("action_content", "")
            self._last_feedback = (
                "Could not parse a valid action from your response. "
                "Please use the format: <think>...</think><answer>action name or action ID</answer>"
            )

        # Check turn limit
        if self._total_turns >= self.config.max_turns:
            done = True

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]

        obs = self._build_obs(init=False)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, init: bool) -> Dict[str, Any]:
        """Build observation dict with image and text."""
        frame = self.env.env.last_event.frame
        img = numpy_to_pil(frame)
        if self.config.obs_image_size is not None:
            sz = self.config.obs_image_size
            img = img.resize((sz, sz), Image.LANCZOS)
        img_str = self.config.image_placeholder

        if init:
            obs_str = init_observation_template(
                task_instruction=self.env.episode_language_instruction,
                img_str=img_str,
            )
        else:
            obs_str = action_template(
                last_action=self._last_action,
                env_feedback=self._last_feedback,
                img_str=img_str,
            )

        return {
            "obs_str": obs_str + "\n",
            "multi_modal_input": {
                self.config.image_placeholder: [img]
            },
        }


# ------------------------------
# Local async test (optional)
# ------------------------------
if __name__ == "__main__":
    import fire
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    async def main_async(
        eval_set: str = "base",
        resolution: int = 500,
        save_path: str = "./test_eb_navigation",
        prompt_format: str = "free_think",
    ):
        cfg = {
            "eval_set": eval_set,
            "resolution": resolution,
            "prompt_format": prompt_format,
        }
        env = EbNavigation(cfg)

        print("System Prompt:")
        sys_prompt = await env.system_prompt()
        print(sys_prompt["obs_str"])
        print("\n" + "=" * 50 + "\n")

        obs, info = await env.reset(seed=0)
        print(f"Task: {info['task_instruction']}")
        print(f"Available actions: {info['num_actions']}")
        print(f"Observation:\n{obs['obs_str'][:200]}...")

        step = 0
        os.makedirs(save_path, exist_ok=True)
        if "multi_modal_input" in obs:
            img = obs["multi_modal_input"][env.config.image_placeholder][0]
            img.save(os.path.join(save_path, f"step_{step}.png"))

        while True:
            step += 1
            print(f"\nStep {step}:")
            try:
                action_input = input("Enter action (or 'quit'): ")
            except EOFError:
                action_input = "quit"

            if action_input.lower() == "quit":
                break

            if not action_input.startswith("<think>"):
                action_input = (
                    f"<think>Executing the action.</think>"
                    f"<answer>{action_input}</answer>"
                )

            obs, reward, done, info = await env.step(action_input)
            if "multi_modal_input" in obs:
                img = obs["multi_modal_input"][env.config.image_placeholder][0]
                img.save(os.path.join(save_path, f"step_{step}.png"))
            print(f"Reward: {reward}, Done: {done}")
            print(f"Success: {info.get('success', False)}")
            print(f"Observation:\n{obs['obs_str'][:200]}...")

            if done:
                print("Episode finished!")
                break

        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(**kwargs))

    fire.Fire(main)
