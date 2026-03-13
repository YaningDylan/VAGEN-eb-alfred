"""
Local evaluation of ERA EPL-Only model on EB-ALFRED.

Loads the SFT checkpoint with transformers, runs multi-turn evaluation
against a local EbAlfred environment (AI2-THOR).

Usage:
    python scripts/eval_epl_local.py \
        --model_path /root/workspace/models/EPL-Only-Model_EB-Alfred \
        --n_episodes 5 \
        --max_turns 30 \
        --x_display 0
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from vagen.envs.eb_alfred.eb_alfred_env import EbAlfred

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_messages(
    system_text: str,
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build chat messages list for Qwen2.5-VL processor."""
    messages = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    for turn in history:
        role = turn["role"]
        content_parts = []
        text = turn.get("text", "")
        images = turn.get("images", [])
        # Insert images where <image> appears
        if images and "<image>" in text:
            parts = text.split("<image>")
            for i, part in enumerate(parts):
                if part:
                    content_parts.append({"type": "text", "text": part})
                if i < len(images):
                    content_parts.append({"type": "image", "image": images[i]})
        else:
            content_parts.append({"type": "text", "text": text})
        messages.append({"role": role, "content": content_parts})
    return messages


@torch.no_grad()
def generate_response(
    model,
    processor,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    device: str = "cuda:0",
) -> str:
    """Generate a single response from the model."""
    from qwen_vl_utils import process_vision_info

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    # Only decode newly generated tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


async def run_episode(
    model,
    processor,
    env: EbAlfred,
    seed: int,
    max_turns: int,
    device: str,
    save_dir: str = None,
) -> Dict[str, Any]:
    """Run a single evaluation episode."""
    t0 = time.time()

    # Reset environment
    obs, info = await env.reset(seed=seed)
    sys_prompt = await env.system_prompt()
    system_text = sys_prompt["obs_str"]

    task = info["task_instruction"]
    logger.info(f"[Seed {seed}] Task: {task}")
    logger.info(f"[Seed {seed}] Available actions: {info['num_actions']}")

    # Save initial image
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img = obs["multi_modal_input"]["<image>"][0]
        img.save(os.path.join(save_dir, "step_0.png"))

    # Build conversation history
    history = []
    # Initial user turn
    init_images = obs["multi_modal_input"].get("<image>", [])
    history.append({
        "role": "user",
        "text": obs["obs_str"],
        "images": init_images,
    })

    success = False
    total_reward = 0.0
    turn = 0

    for turn in range(1, max_turns + 1):
        # Build messages and generate
        messages = build_messages(system_text, history)
        response = generate_response(model, processor, messages, device=device)

        logger.info(f"[Seed {seed}] Turn {turn}: {response[:200]}...")

        # Add assistant response to history
        history.append({"role": "assistant", "text": response, "images": []})

        # Step environment
        obs, reward, done, step_info = await env.step(response)
        total_reward += reward
        success = step_info.get("success", False)

        # Save step image
        if save_dir:
            step_images = obs["multi_modal_input"].get("<image>", [])
            if step_images:
                step_images[0].save(os.path.join(save_dir, f"step_{turn}.png"))

        logger.info(
            f"[Seed {seed}] Turn {turn} | reward={reward:.2f} | done={done} | success={success}"
        )

        if done:
            break

        # Add environment feedback as next user turn
        step_images = obs["multi_modal_input"].get("<image>", [])
        history.append({
            "role": "user",
            "text": obs["obs_str"],
            "images": step_images,
        })

    elapsed = time.time() - t0
    result = {
        "seed": seed,
        "task": task,
        "success": success,
        "total_reward": total_reward,
        "num_turns": turn,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save trajectory
    if save_dir:
        # Save text-only history for review
        text_history = []
        for h in history:
            text_history.append({
                "role": h["role"],
                "text": h["text"],
                "n_images": len(h.get("images", [])),
            })
        with open(os.path.join(save_dir, "trajectory.json"), "w") as f:
            json.dump({"result": result, "history": text_history}, f, indent=2, ensure_ascii=False)

    return result


async def main_async(args):
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    device = f"cuda:{args.gpu_id}"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        device_map=device,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=500 * 500,
    )
    logger.info("Model loaded.")

    # Create environment
    env_config = {
        "eval_set": args.eval_set,
        "resolution": 500,
        "x_display": str(args.x_display),
        "max_turns": args.max_turns,
        "max_actions_per_step": args.max_actions_per_step,
        "max_env_steps": args.max_env_steps,
        "action_sep": ",",
        "prompt_format": "free_think",
        "use_example_in_sys_prompt": True,
        "format_reward": 0.1,
        "success_reward": 1.0,
    }

    results = []
    for i in range(args.n_episodes):
        seed = args.start_seed + i
        save_dir = os.path.join(args.output_dir, f"episode_{seed}") if args.output_dir else None

        # Create fresh env per episode to avoid AI2-THOR state issues
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {i+1}/{args.n_episodes} (seed={seed})")
        logger.info(f"{'='*60}")

        env = EbAlfred(env_config)
        try:
            result = await run_episode(
                model=model,
                processor=processor,
                env=env,
                seed=seed,
                max_turns=args.max_turns,
                device=device,
                save_dir=save_dir,
            )
            results.append(result)
            status = "SUCCESS" if result["success"] else "FAIL"
            logger.info(
                f"[Seed {seed}] {status} | reward={result['total_reward']:.2f} | "
                f"turns={result['num_turns']} | time={result['elapsed_seconds']}s"
            )
        except Exception as e:
            logger.error(f"[Seed {seed}] Episode failed: {e}", exc_info=True)
            results.append({"seed": seed, "success": False, "error": str(e)})
        finally:
            await env.close()

    # Summary
    n_success = sum(1 for r in results if r.get("success", False))
    n_total = len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Success rate: {n_success}/{n_total} ({100*n_success/max(n_total,1):.1f}%)")
    logger.info(f"Avg reward: {sum(r.get('total_reward',0) for r in results)/max(n_total,1):.3f}")
    logger.info(f"Avg turns: {sum(r.get('num_turns',0) for r in results)/max(n_total,1):.1f}")

    if args.output_dir:
        summary_path = os.path.join(args.output_dir, "summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "model": args.model_path,
                "eval_set": args.eval_set,
                "n_episodes": n_total,
                "n_success": n_success,
                "success_rate": n_success / max(n_total, 1),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EPL model on EB-ALFRED locally")
    parser.add_argument("--model_path", type=str,
                        default="/root/workspace/models/EPL-Only-Model_EB-Alfred")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--max_turns", type=int, default=30)
    parser.add_argument("--max_actions_per_step", type=int, default=1,
                        help="Actions per LLM response (1 for single-action, 20 for multi)")
    parser.add_argument("--max_env_steps", type=int, default=30)
    parser.add_argument("--eval_set", type=str, default="base")
    parser.add_argument("--x_display", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./eval_results/epl_eval")
    parser.add_argument("--flash_attn", action="store_true", default=False)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
