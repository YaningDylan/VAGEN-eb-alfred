"""
Self-contained end-to-end evaluation for EB-Manipulation with Claude API.

Runs N episodes, calls Claude for each turn, saves images + trajectories.
Compatible with Python 3.9 (embench conda env).

Usage:
    # Set environment variables first:
    export COPPELIASIM_ROOT=<path_to_CoppeliaSim>
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    export DISPLAY=:1

    # Run 20 episodes
    conda run -n embench python tests/eval_eb_manipulation_e2e.py

    # Run 5 episodes
    conda run -n embench python tests/eval_eb_manipulation_e2e.py --n_episodes 5
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from PIL import Image

# Add vagen to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vagen.envs.eb_manipulation.eb_manipulation_env import EbManipulation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eb_man_eval")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_images(obs: Dict[str, Any]) -> List[Image.Image]:
    mm = obs.get("multi_modal_input", {})
    imgs = mm.get("<image>", [])
    if isinstance(imgs, list) and imgs and isinstance(imgs[0], Image.Image):
        return imgs
    return []


# ---------------------------------------------------------------------------
# Claude API wrapper
# ---------------------------------------------------------------------------

class ClaudeAgent:
    """Minimal async Claude wrapper for multi-turn vision conversations."""

    def __init__(self, model: str, api_key: str, temperature: float = 0, max_tokens: int = 1024):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, system_text: str, messages: List[Dict]) -> str:
        """Send messages to Claude and get response."""
        resp = await self.client.messages.create(
            model=self.model,
            system=system_text,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        texts = []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
        return "\n".join(t for t in texts if t.strip())


def build_user_message(text: str, images: List[Image.Image]) -> Dict:
    """Build a Claude-format user message with text and images."""
    content = []
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": pil_to_base64(img),
            },
        })
    if text.strip():
        clean_text = text.replace("<image>", "").strip()
        if clean_text:
            content.append({"type": "text", "text": clean_text})
    return {"role": "user", "content": content}


def build_assistant_message(text: str) -> Dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    env_config: Dict[str, Any],
    agent: ClaudeAgent,
    seed: int,
    max_turns: int,
    dump_dir: str,
) -> Dict[str, Any]:
    """Run a single episode and save results."""
    rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    ep_dir = os.path.join(dump_dir, rid)
    img_dir = os.path.join(ep_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    env = EbManipulation(env_config)

    messages: List[Dict] = []
    assistant_texts: List[str] = []
    rewards: List[float] = []
    infos: List[Dict] = []
    cumulative_reward = 0.0
    terminated = False
    finish_reason = "max_turns"
    success = False
    system_text = ""

    try:
        # Reset
        obs, info = await env.reset(seed=seed)
        infos.append(info)

        # Get system prompt
        sys_obs = await env.system_prompt()
        system_text = sys_obs.get("obs_str", "")

        # Build initial user message
        user_imgs = extract_images(obs)
        user_text = obs.get("obs_str", "")
        messages.append(build_user_message(user_text, user_imgs))

        # Save initial image
        for i, img in enumerate(user_imgs):
            img.save(os.path.join(img_dir, f"turn_00_{i:02d}.png"))

        logger.info("  Task: %s", info.get("task_instruction", "N/A"))

        # Turn loop
        for t in range(max_turns):
            # Call Claude
            try:
                reply = await agent.chat(system_text, messages)
            except Exception as e:
                logger.error("  Claude error at turn %d: %s", t, e)
                finish_reason = "model_error"
                break

            assistant_texts.append(reply)
            messages.append(build_assistant_message(reply))

            # Step environment
            try:
                next_obs, r, done, step_info = await env.step(reply)
            except Exception as e:
                logger.error("  Env error at turn %d: %s", t, e)
                finish_reason = "env_error"
                break

            rewards.append(float(r))
            cumulative_reward += float(r)
            infos.append(step_info)

            # Save images
            step_imgs = extract_images(next_obs)
            for i, img in enumerate(step_imgs):
                img.save(os.path.join(img_dir, f"turn_{t+1:02d}_{i:02d}.png"))

            # Append next obs as user message
            next_text = next_obs.get("obs_str", "")
            messages.append(build_user_message(next_text, step_imgs))

            # Log turn
            turn_metrics = step_info.get("metrics", {}).get("turn_metrics", {})
            logger.info(
                "  Turn %d: reward=%.2f, valid=%s, effective=%s",
                t + 1, r,
                turn_metrics.get("action_is_valid", False),
                turn_metrics.get("action_is_effective", False),
            )

            if done:
                terminated = True
                finish_reason = "done"
                success = step_info.get("success", False)
                break

        # Check success from last info
        if not success and infos:
            success = bool(infos[-1].get("success", False))

    except Exception as e:
        logger.error("  Episode failed: %s", e)
        import traceback
        traceback.print_exc()
        finish_reason = "error"
    finally:
        try:
            await env.close()
        except Exception:
            pass

    # Build metrics
    metrics = {
        "rollout_id": rid,
        "seed": seed,
        "terminated": terminated,
        "finish_reason": finish_reason,
        "success": success,
        "cumulative_reward": cumulative_reward,
        "rewards": rewards,
        "num_turns": len(assistant_texts),
        "max_turns": max_turns,
    }

    # Save metrics
    with open(os.path.join(ep_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save assistant responses
    with open(os.path.join(ep_dir, "assistant_texts.json"), "w") as f:
        json.dump(assistant_texts, f, indent=2, ensure_ascii=False)

    # Save transcript (human-readable)
    lines = [f"SYSTEM:\n{system_text}\n{'='*40}\n"]
    for msg in messages:
        role = msg.get("role", "").upper()
        content = msg.get("content", [])
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        n_imgs = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image")
        text = " ".join(texts).strip()
        img_note = f" [+{n_imgs} image(s)]" if n_imgs else ""
        lines.append(f"{role}{img_note}:\n{text}\n{'-'*40}\n")
    with open(os.path.join(ep_dir, "transcript.txt"), "w") as f:
        f.write("\n".join(lines))

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def run_evaluation(
    n_episodes: int = 20,
    eval_set: str = "base",
    max_turns: int = 15,
    dump_dir: str = "./eval_results/eb_manipulation",
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0,
    max_tokens: int = 1024,
    base_seed: int = 0,
):
    tag_dir = os.path.join(dump_dir, f"tag_eb_manipulation_{eval_set}")
    os.makedirs(tag_dir, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    agent = ClaudeAgent(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens)

    env_config = {
        "eval_set": eval_set,
        "img_size": [500, 500],
        "max_turns": max_turns,
        "max_actions_per_step": 1,
        "action_sep": ",",
        "prompt_format": "free_think",
        "use_example_in_sys_prompt": True,
        "format_reward": 0.1,
        "success_reward": 1.0,
        "camera_view": "front_rgb",
    }

    results = []
    successes = 0
    total_reward = 0.0
    total_turns = 0
    start_time = time.time()

    for i in range(n_episodes):
        seed = base_seed + i
        logger.info("=== Episode %d/%d (seed=%d) ===", i + 1, n_episodes, seed)

        metrics = await run_episode(
            env_config=env_config,
            agent=agent,
            seed=seed,
            max_turns=max_turns,
            dump_dir=tag_dir,
        )
        results.append(metrics)

        s = metrics.get("success", False)
        if s:
            successes += 1
        total_reward += metrics.get("cumulative_reward", 0.0)
        total_turns += metrics.get("num_turns", 0)

        logger.info(
            "  => success=%s, reward=%.2f, turns=%d | Running: %d/%d (%.0f%%)",
            s, metrics["cumulative_reward"], metrics["num_turns"],
            successes, i + 1, 100.0 * successes / (i + 1),
        )

    elapsed = time.time() - start_time
    n = max(n_episodes, 1)

    # Final report
    print("\n" + "=" * 60)
    print("EB-MANIPULATION EVALUATION REPORT")
    print("=" * 60)
    print(f"  Eval set:       {eval_set}")
    print(f"  Model:          {model}")
    print(f"  Episodes:       {n_episodes}")
    print(f"  Max turns:      {max_turns}")
    print(f"  Success rate:   {successes}/{n_episodes} ({100.0 * successes / n:.1f}%)")
    print(f"  Avg reward:     {total_reward / n:.3f}")
    print(f"  Avg turns:      {total_turns / n:.1f}")
    print(f"  Total time:     {elapsed:.1f}s ({elapsed / n:.1f}s/ep)")
    print(f"  Results dir:    {os.path.abspath(tag_dir)}")
    print("=" * 60)

    print(f"\n  {'Ep':>3}  {'Seed':>5}  {'OK?':>4}  {'Reward':>7}  {'Turns':>5}  {'Reason':<12}")
    for i, r in enumerate(results):
        print(
            f"  {i+1:3d}  {r['seed']:5d}  "
            f"{'Y' if r['success'] else 'N':>4}  "
            f"{r['cumulative_reward']:7.2f}  "
            f"{r['num_turns']:5d}  "
            f"{r['finish_reason']:<12}"
        )

    summary = {
        "eval_set": eval_set, "model": model, "n_episodes": n_episodes,
        "max_turns": max_turns,
        "success_rate": successes / n, "avg_reward": total_reward / n,
        "avg_turns": total_turns / n, "total_time_s": elapsed,
        "episodes": results,
    }
    path = os.path.join(dump_dir, "eval_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary: {path}")


def main():
    parser = argparse.ArgumentParser(description="EB-Manipulation Eval with Claude")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--eval_set", type=str, default="base")
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--dump_dir", type=str, default="./eval_results/eb_manipulation")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--base_seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run_evaluation(**vars(args)))


if __name__ == "__main__":
    main()
