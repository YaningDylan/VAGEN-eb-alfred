"""
Local sanity check for Sokoban environment (no API calls).

Validates the env interface contract before running full API evaluation.
For API-based end-to-end testing, use the built-in evaluation framework:

    # Claude backend, 3 envs, 5 turns
    cd VAGEN
    bash examples/evaluate/sokoban/run_eval.sh examples/evaluate/sokoban/config.yaml \
        run.backend=claude \
        envs[0].n_envs=3 \
        run.max_concurrent_jobs=2

Usage:
    python tests/test_sokoban_api_e2e.py
    python tests/test_sokoban_api_e2e.py --render_mode text
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vagen.envs.sokoban.sokoban_env import Sokoban


def make_sokoban_config(render_mode: str = "vision") -> dict:
    return {
        "dim_room": (6, 6),
        "max_steps": 100,
        "num_boxes": 1,
        "render_mode": render_mode,
        "max_actions_per_step": 2,
        "action_sep": ",",
        "image_placeholder": "<image>",
        "use_example_in_sys_prompt": True,
        "min_solution_steps": (1, 5),
        "reset_seed_max_tries": 10000,
        "min_solution_bfs_max_depth": 100,
        "prompt_format": "free_think",
        "format_reward": 0.1,
        "success_reward": 1.0,
    }


async def test_env_local(render_mode: str = "text") -> bool:
    """Verify env interface contract locally without any API calls."""
    print(f"\n{'=' * 60}")
    print(f"Local environment sanity check (render_mode={render_mode})")
    print("=" * 60)

    env_config = make_sokoban_config(render_mode=render_mode)
    env = Sokoban(env_config)

    try:
        # system_prompt
        sys_obs = await env.system_prompt()
        assert "obs_str" in sys_obs, "system_prompt must return dict with 'obs_str'"
        print(f"  [OK] system_prompt: {len(sys_obs['obs_str'])} chars")

        # reset
        obs, info = await env.reset(seed=42)
        assert "obs_str" in obs, "reset obs must have 'obs_str'"
        if render_mode == "vision":
            assert "multi_modal_input" in obs, "vision mode must return multi_modal_input"
            imgs = obs["multi_modal_input"]["<image>"]
            assert len(imgs) == obs["obs_str"].count("<image>"), "image count mismatch"
            print(f"  [OK] reset (vision): {len(imgs)} image(s)")
        else:
            print(f"  [OK] reset (text): {len(obs['obs_str'])} chars")

        # step with valid format
        action_str = "<think>I will move up.</think><answer>Up</answer>"
        obs, reward, done, info = await env.step(action_str)
        assert isinstance(reward, (int, float)), "reward must be numeric"
        assert isinstance(done, bool), "done must be bool"
        assert "success" in info, "info must contain 'success'"
        print(f"  [OK] step: reward={reward}, done={done}, success={info['success']}")

        # step with bad format
        obs, reward, done, info = await env.step("random gibberish")
        print(f"  [OK] bad format step: reward={reward} (should be 0)")

        # determinism
        obs1, _ = await env.reset(seed=123)
        env2 = Sokoban(env_config)
        obs2, _ = await env2.reset(seed=123)
        assert obs1["obs_str"] == obs2["obs_str"], "same seed must produce same observation"
        print("  [OK] determinism: seed=123 produces identical observations")
        await env2.close()

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await env.close()


async def main():
    parser = argparse.ArgumentParser(description="Local Sokoban env sanity check")
    parser.add_argument("--render_mode", type=str, default="all", choices=["text", "vision", "all"])
    args = parser.parse_args()

    modes = ["text", "vision"] if args.render_mode == "all" else [args.render_mode]
    results = {}
    for mode in modes:
        results[mode] = await test_env_local(render_mode=mode)

    print(f"\n{'=' * 60}")
    all_passed = all(results.values())
    for mode, passed in results.items():
        print(f"  {mode}: {'PASSED' if passed else 'FAILED'}")
    print(f"{'=' * 60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
