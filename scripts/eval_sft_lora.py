"""
Evaluation for LoRA SFT checkpoints on EB-ALFRED.

Supports: base model (no adapter), LoRA adapter, or full checkpoint.
Tracks per-step metrics: action_valid, action_effective, action_text.

Usage:
    python scripts/eval_sft_lora.py \
        --base_model Qwen/Qwen2.5-VL-3B-Instruct \
        --adapter_path /path/to/checkpoint-1000 \
        --n_episodes 5 --x_display 0 --gpu_id 0
"""

import argparse
import asyncio
import json
import logging
import os
import time

import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from vagen.envs.eb_alfred.eb_alfred_env import EbAlfred

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_messages(system_text, history):
    messages = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    for turn in history:
        role = turn["role"]
        content_parts = []
        text = turn.get("text", "")
        images = turn.get("images", [])
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
def generate_response(model, processor, messages, max_new_tokens=1024, temperature=0.0, device="cuda:0"):
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
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True)


async def run_episode(model, processor, env, seed, max_turns, device, save_dir=None):
    t0 = time.time()
    obs, info = await env.reset(seed=seed)
    sys_prompt = await env.system_prompt()
    system_text = sys_prompt["obs_str"]
    task = info["task_instruction"]
    logger.info(f"[Seed {seed}] Task: {task}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img = obs["multi_modal_input"]["<image>"][0]
        img.save(os.path.join(save_dir, "step_0.png"))

    history = []
    init_images = obs["multi_modal_input"].get("<image>", [])
    history.append({"role": "user", "text": obs["obs_str"], "images": init_images})

    success = False
    total_reward = 0.0
    per_turn_metrics = []

    for turn in range(1, max_turns + 1):
        messages = build_messages(system_text, history)
        response = generate_response(model, processor, messages, device=device)

        logger.info(f"[Seed {seed}] Turn {turn}: {response[:200]}")

        history.append({"role": "assistant", "text": response, "images": []})

        obs, reward, done, step_info = await env.step(response)
        total_reward += reward
        success = step_info.get("success", False)

        # Per-turn metrics
        turn_m = step_info.get("metrics", {}).get("turn_metrics", {})
        turn_metric = {
            "turn": turn,
            "response": response,
            "reward": reward,
            "done": done,
            "success": success,
            "action_is_valid": turn_m.get("action_is_valid", False),
            "action_is_effective": turn_m.get("action_is_effective", False),
        }
        per_turn_metrics.append(turn_metric)

        if save_dir:
            step_images = obs["multi_modal_input"].get("<image>", [])
            if step_images:
                step_images[0].save(os.path.join(save_dir, f"step_{turn}.png"))

        logger.info(
            f"[Seed {seed}] Turn {turn} | reward={reward:.2f} | valid={turn_metric['action_is_valid']} "
            f"| effective={turn_metric['action_is_effective']} | done={done} | success={success}"
        )

        if done:
            break

        step_images = obs["multi_modal_input"].get("<image>", [])
        history.append({"role": "user", "text": obs["obs_str"], "images": step_images})

    elapsed = time.time() - t0

    # Aggregate metrics
    n_valid = sum(1 for m in per_turn_metrics if m.get("action_is_valid"))
    n_effective = sum(1 for m in per_turn_metrics if m.get("action_is_effective"))
    n_turns = len(per_turn_metrics)

    result = {
        "seed": seed,
        "task": task,
        "success": success,
        "total_reward": total_reward,
        "num_turns": n_turns,
        "n_valid_actions": n_valid,
        "n_effective_actions": n_effective,
        "valid_action_rate": n_valid / max(n_turns, 1),
        "effective_action_rate": n_effective / max(n_turns, 1),
        "elapsed_seconds": round(elapsed, 1),
        "per_turn": per_turn_metrics,
    }

    if save_dir:
        with open(os.path.join(save_dir, "trajectory.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


async def evaluate_model(model, processor, args, model_name):
    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATING: {model_name}")
    logger.info(f"{'='*70}")

    results = []
    for i in range(args.n_episodes):
        seed = args.start_seed + i
        save_dir = os.path.join(args.output_dir, model_name, f"episode_{seed}")

        logger.info(f"\nEpisode {i+1}/{args.n_episodes} (seed={seed})")

        env_config = {
            "eval_set": args.eval_set,
            "resolution": 500,
            "x_display": str(args.x_display),
            "max_turns": args.max_turns,
            "max_actions_per_step": 20,
            "max_env_steps": args.max_env_steps,
            "action_sep": ",",
            "prompt_format": "free_think",
            "use_example_in_sys_prompt": True,
            "format_reward": 0.1,
            "success_reward": 1.0,
        }

        env = EbAlfred(env_config)
        try:
            result = await run_episode(
                model=model, processor=processor, env=env,
                seed=seed, max_turns=args.max_turns,
                device=f"cuda:{args.gpu_id}", save_dir=save_dir,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"[Seed {seed}] Episode failed: {e}", exc_info=True)
            results.append({"seed": seed, "success": False, "error": str(e), "num_turns": 0,
                            "total_reward": 0, "n_valid_actions": 0, "n_effective_actions": 0})
        finally:
            await env.close()

    # Summary
    n_total = len(results)
    n_success = sum(1 for r in results if r.get("success", False))
    avg_reward = sum(r.get("total_reward", 0) for r in results) / max(n_total, 1)
    avg_turns = sum(r.get("num_turns", 0) for r in results) / max(n_total, 1)
    avg_valid = sum(r.get("n_valid_actions", 0) for r in results) / max(sum(r.get("num_turns", 0) for r in results), 1)
    avg_effective = sum(r.get("n_effective_actions", 0) for r in results) / max(sum(r.get("num_turns", 0) for r in results), 1)

    summary = {
        "model": model_name,
        "n_episodes": n_total,
        "n_success": n_success,
        "success_rate": n_success / max(n_total, 1),
        "avg_reward": round(avg_reward, 3),
        "avg_turns": round(avg_turns, 1),
        "valid_action_rate": round(avg_valid, 3),
        "effective_action_rate": round(avg_effective, 3),
        "results": results,
    }

    summary_path = os.path.join(args.output_dir, model_name, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def load_model(base_model_path, adapter_path, gpu_id, flash_attn=False):
    device = f"cuda:{gpu_id}"
    logger.info(f"Loading base model: {base_model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if flash_attn else "eager",
        device_map=device,
    )

    if adapter_path:
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged.")

    model.eval()

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        min_pixels=500 * 500,
        max_pixels=500 * 500,
    )
    return model, processor


async def main_async(args):
    all_summaries = []

    # Define models to evaluate
    models = []
    if args.eval_base:
        models.append(("base_qwen25vl3b", None))
    if args.eval_ckpt500:
        models.append(("lora_ckpt500", os.path.join(args.lora_dir, "checkpoint-500")))
    if args.eval_ckpt1000:
        models.append(("lora_ckpt1000", os.path.join(args.lora_dir, "checkpoint-1000")))

    for model_name, adapter_path in models:
        model, processor = load_model(args.base_model, adapter_path, args.gpu_id, args.flash_attn)
        summary = await evaluate_model(model, processor, args, model_name)
        all_summaries.append(summary)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"{'Model':<20} {'Success':>8} {'Rate':>8} {'Avg Reward':>11} {'Avg Turns':>10} {'Valid%':>8} {'Effect%':>8}")
    print("-" * 90)
    for s in all_summaries:
        print(f"{s['model']:<20} {s['n_success']}/{s['n_episodes']:>5} {s['success_rate']*100:>7.1f}% "
              f"{s['avg_reward']:>10.3f} {s['avg_turns']:>10.1f} {s['valid_action_rate']*100:>7.1f}% "
              f"{s['effective_action_rate']*100:>7.1f}%")
    print("=" * 90)

    # Save combined summary
    combined_path = os.path.join(args.output_dir, "comparison.json")
    with open(combined_path, "w") as f:
        # Strip per_turn from combined (too verbose)
        slim = []
        for s in all_summaries:
            ss = {k: v for k, v in s.items() if k != "results"}
            ss["episodes"] = [{k: v for k, v in r.items() if k != "per_turn"} for r in s.get("results", [])]
            slim.append(ss)
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--lora_dir", type=str,
                        default="/root/workspace/VAGEN-eb-alfred/exps/sft/checkpoints/qwen25vl3b-lora")
    parser.add_argument("--eval_base", action="store_true", default=False)
    parser.add_argument("--eval_ckpt500", action="store_true", default=False)
    parser.add_argument("--eval_ckpt1000", action="store_true", default=False)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--max_turns", type=int, default=30)
    parser.add_argument("--max_env_steps", type=int, default=30)
    parser.add_argument("--eval_set", type=str, default="base")
    parser.add_argument("--x_display", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./eval_results/sft_comparison")
    parser.add_argument("--flash_attn", action="store_true", default=False)
    args = parser.parse_args()

    if not (args.eval_base or args.eval_ckpt500 or args.eval_ckpt1000):
        args.eval_base = True
        args.eval_ckpt500 = True
        args.eval_ckpt1000 = True

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
