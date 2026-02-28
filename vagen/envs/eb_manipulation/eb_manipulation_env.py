"""
EB-Manipulation environment adapter for VAGEN.

Wraps the EBManEnv from EmbodiedBench as a GymImageEnv,
enabling integration with VAGEN's RL training and evaluation pipeline.

The underlying EBManEnv uses PyRep/CoppeliaSim for robot manipulation
simulation (Franka Panda arm). It runs headless and does not require
a GPU-accelerated X server (unlike EB-ALFRED/AI2-THOR).
"""

import asyncio
import copy
import os
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional

from .utils.prompt import (
    system_prompt,
    format_prompt,
    init_observation_template,
    action_template,
)
from .utils.utils import parse_response, parse_action_vector, numpy_to_pil

from vagen.envs.gym_image_env import GymImageEnv


@dataclass
class EbManipulationEnvConfig:
    """Configuration for EB-Manipulation environment."""

    # Environment settings
    eval_set: str = "base"
    exp_name: str = "vagen_eval"
    down_sample_ratio: float = 1.0
    img_size: Tuple[int, int] = (500, 500)
    selected_indexes: List[int] = field(default_factory=list)
    log_path: Optional[str] = None

    # Interaction settings
    max_turns: int = 15
    max_actions_per_step: int = 1
    action_sep: str = ","
    image_placeholder: str = "<image>"
    prompt_format: str = "free_think"
    use_example_in_sys_prompt: bool = True
    camera_view: str = "front_rgb"

    # Reward settings
    format_reward: float = 0.1
    success_reward: float = 1.0


class EbManipulation(GymImageEnv):
    """
    EB-Manipulation environment implementing the GymImageEnv async interface.

    Wraps EBManEnv from EmbodiedBench, which uses PyRep/CoppeliaSim for
    robot manipulation tasks (pick, stack, place, wipe).

    Key features:
    - 7D discrete action space: [X, Y, Z, Roll, Pitch, Yaw, Gripper]
    - Multiple camera views (front, wrist, shoulder, overhead)
    - 4 task types across 5 evaluation sets
    - Object coordinate information for spatial reasoning
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        # Filter config keys to only those in the dataclass
        valid_keys = EbManipulationEnvConfig.__dataclass_fields__
        filtered = {k: v for k, v in env_config.items() if k in valid_keys}
        self.config = EbManipulationEnvConfig(**filtered)

        # Import and create EBManEnv
        import amsolver.task_environment as _amsolver_te
        import embodiedbench.envs.eb_manipulation.EBManEnv as _ebman_mod
        from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv

        # Patch TTMS_FOLDER to be absolute (it's relative by default,
        # which only works if cwd == EmbodiedBench root)
        _eb_man_dir = os.path.dirname(os.path.abspath(_ebman_mod.__file__))
        _abs_ttms = os.path.join(_eb_man_dir, "")
        _ebman_mod.TTMS_FOLDER = _abs_ttms
        _amsolver_te.TTMS_FOLDER = _abs_ttms

        self.env = EBManEnv(
            eval_set=self.config.eval_set,
            render_mode=None,  # No Qt window; images come from obs cameras
            img_size=self.config.img_size,
            down_sample_ratio=self.config.down_sample_ratio,
            selected_indexes=self.config.selected_indexes,
            log_path=self.config.log_path,
        )

        # Adapter state (reset per episode)
        self._total_turns: int = 0
        self._last_action_str: str = ""
        self._last_feedback: str = ""
        self._object_coords: str = ""

    # ------------------------------------------------------------------
    # GymImageEnv abstract methods
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close CoppeliaSim/PyRep process.

        CoppeliaSim often segfaults during Qt shutdown. We isolate
        the shutdown in a forked child process so the parent (server)
        survives.
        """

        def _fork_close():
            pid = os.fork()
            if pid == 0:
                # Child: perform the close (may segfault)
                try:
                    self.env.close()
                except Exception:
                    pass
                os._exit(0)
            else:
                # Parent: wait for child (with timeout)
                import signal as _sig

                def _alarm_handler(signum, frame):
                    try:
                        os.kill(pid, _sig.SIGKILL)
                    except OSError:
                        pass

                old = _sig.signal(_sig.SIGALRM, _alarm_handler)
                _sig.alarm(15)
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
                finally:
                    _sig.alarm(0)
                    _sig.signal(_sig.SIGALRM, old)

        try:
            await asyncio.to_thread(_fork_close)
        except Exception:
            pass

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Return the static system prompt.

        Includes robot description, action space definition, coordinate
        frame description, and format instructions.
        """
        sys_str = system_prompt()
        fmt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.use_example_in_sys_prompt,
            prompt_format=self.config.prompt_format,
        )
        return {"obs_str": sys_str + "\n" + fmt_str}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment for a new episode.

        The seed selects which episode to load from the dataset
        (seed % number_of_episodes). After reset, the observation
        includes the task instruction, object coordinates, and
        the front RGB image from CoppeliaSim.
        """
        # Select episode based on seed
        episode_idx = seed % self.env.number_of_episodes
        self.env._current_episode_num = episode_idx

        description, obs = await asyncio.to_thread(self.env.reset)

        # Reset adapter state
        self._total_turns = 0
        self._last_action_str = ""
        self._last_feedback = ""

        # Extract object coordinates
        self._object_coords = self._get_object_coords(self.env.last_frame_obs)

        # Build observation
        obs_dict = self._build_obs(init=True)
        info = {
            "task_instruction": self.env.episode_language_instruction,
            "eval_set": self.config.eval_set,
            "episode_idx": episode_idx,
            "task_class": getattr(self.env, "task_class", "unknown"),
        }
        return obs_dict, info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step given the LLM's response.

        Parses <think>...</think><answer>[X,Y,Z,Roll,Pitch,Yaw,Gripper]</answer>
        from action_str, converts to discrete action, and executes in CoppeliaSim.
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
            action_vector = parse_action_vector(actions[0])

            if action_vector is None:
                # Could not parse the 7D action vector
                self._last_action_str = parsed.get("action_content", "")
                self._last_feedback = (
                    "Could not parse a valid 7D action vector. "
                    "Please output [X, Y, Z, Roll, Pitch, Yaw, Gripper] with integer values."
                )
            else:
                reward += self.config.format_reward
                metrics["turn_metrics"]["action_is_valid"] = True

                self._last_action_str = str(action_vector)

                # Execute in CoppeliaSim (pass None for recorder)
                obs_raw, step_reward, step_done, step_info = (
                    await asyncio.to_thread(self.env.step, action_vector, None)
                )

                self._last_feedback = step_info.get("env_feedback", "")

                action_success = step_info.get("action_success", 0.0)
                if action_success:
                    metrics["turn_metrics"]["action_is_effective"] = True

                task_success = step_info.get("task_success", 0.0)
                if task_success:
                    done = True
                    reward += self.config.success_reward
                    metrics["traj_metrics"]["success"] = True

                if step_done:
                    done = True

                # Update object coordinates
                self._object_coords = self._get_object_coords(self.env.last_frame_obs)
        else:
            # Format error: no valid action parsed
            self._last_action_str = parsed.get("action_content", "")
            self._last_feedback = (
                "Could not parse a valid action from your response. "
                "Please use the format: "
                "<think>...</think><answer>[X, Y, Z, Roll, Pitch, Yaw, Gripper]</answer>"
            )

        # Check turn limit
        if self._total_turns >= self.config.max_turns:
            done = True

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]

        obs_dict = self._build_obs(init=False)
        return obs_dict, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, init: bool) -> Dict[str, Any]:
        """Build observation dict with image and text."""
        frame = self.env.last_frame_obs[self.config.camera_view]
        img = numpy_to_pil(frame)
        img_str = self.config.image_placeholder

        if init:
            obs_str = init_observation_template(
                task_instruction=self.env.episode_language_instruction,
                object_coords=self._object_coords,
                img_str=img_str,
            )
        else:
            obs_str = action_template(
                last_action=self._last_action_str,
                env_feedback=self._last_feedback,
                object_coords=self._object_coords,
                img_str=img_str,
            )

        return {
            "obs_str": obs_str + "\n",
            "multi_modal_input": {
                self.config.image_placeholder: [img]
            },
        }

    def _get_object_coords(self, obs_dict: Dict) -> str:
        """
        Extract object coordinate information from observation.

        Uses form_object_coord_for_input from EmbodiedBench if available,
        otherwise returns a simple description.
        """
        try:
            from embodiedbench.envs.eb_manipulation.eb_man_utils import (
                form_object_coord_for_input,
            )

            task_class = getattr(self.env, "task_class", None)
            if task_class and obs_dict:
                avg_obj_coord, _, _, _ = form_object_coord_for_input(
                    copy.deepcopy(obs_dict),
                    task_class,
                    [self.config.camera_view],
                )
                return str(avg_obj_coord)
        except Exception:
            pass

        return "Object coordinates not available."


# ------------------------------
# Local async test (optional)
# ------------------------------
if __name__ == "__main__":
    import fire
    import os
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    async def main_async(
        eval_set: str = "base",
        img_size: Tuple[int, int] = (500, 500),
        save_path: str = "./test_eb_manipulation",
        prompt_format: str = "free_think",
    ):
        cfg = {
            "eval_set": eval_set,
            "img_size": img_size,
            "prompt_format": prompt_format,
        }
        env = EbManipulation(cfg)

        print("System Prompt:")
        sys_prompt = await env.system_prompt()
        print(sys_prompt["obs_str"])
        print("\n" + "=" * 50 + "\n")

        obs, info = await env.reset(seed=0)
        print(f"Task: {info['task_instruction']}")
        print(f"Task class: {info['task_class']}")
        print(f"Observation:\n{obs['obs_str'][:300]}...")

        step = 0
        os.makedirs(save_path, exist_ok=True)
        if "multi_modal_input" in obs:
            img = obs["multi_modal_input"][env.config.image_placeholder][0]
            img.save(os.path.join(save_path, f"step_{step}.png"))

        while True:
            step += 1
            print(f"\nStep {step}:")
            try:
                action_input = input("Enter action vector (e.g., [50,30,40,60,60,60,1]) or 'quit': ")
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
            print(f"Observation:\n{obs['obs_str'][:300]}...")

            if done:
                print("Episode finished!")
                break

        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(**kwargs))

    fire.Fire(main)
