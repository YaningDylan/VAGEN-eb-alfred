"""
Convert ERA EPL SFT datasets to VAGEN format.

ERA format (per sample):
  - system: ERA-style system prompt with action list
  - human: "<image>\n instruction: {task}\n interaction_history: [...]\n ..."
  - gpt (think_or_action=0): "visual_description: ... reasoning_and_reflection: ..."
  - gpt (think_or_action=1): "[id, 'action name']"

VAGEN format (per sample):
  - conversations: [
      {"role": "system", "content": system_prompt_with_task_and_actions},
      {"role": "user",   "content": "<image>\n[Current Observation]\nDecide your next action."},
      {"role": "assistant", "content": "<think>...</think><answer>action name</answer>"},
    ]

Usage:
    python scripts/convert_era_to_vagen.py \
        --trajectory_path /root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset/eb_alfred_trajectory_augmented_prior_dataset.json \
        --env_anchored_path /root/workspace/era_sft_data/EB-ALFRED_environment_anchored_prior_dataset/eb_alfred_environment_anchored_prior_dataset.json \
        --external_path /root/workspace/era_sft_data/EB-ALFRED_external_knowledge_prior_dataset/eb_alfred_external_knowledge_prior_dataset.json \
        --output_dir /root/workspace/era_sft_data/vagen_format \
        --images_dir /root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset
"""

import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional


# ── VAGEN-style system prompt template (matches eb_alfred_env.py prompt.py) ──

VAGEN_SYSTEM_TEMPLATE = """You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules
- Find: Parameterized by the name of the receptacle to navigate to. Always valid if the object exists in the scene.
- Pick up: Parameterized by the name of the object to pick. Only valid if close to the object, not already holding something, and the object is not in a closed receptacle.
- Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if holding an object.
- Drop: Parameterized by the name of the object to put down. Different from 'put down' as this does not guarantee the held object will be put into a specified receptacle.
- Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and close to the receptacle.
- Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and close to the receptacle.
- Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and close to the object.
- Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and close to the object.
- Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and close to the object.

## Guidelines
1. Output a plan of actions. Each plan should include no more than 20 actions.
2. Always locate an object using 'find' before interacting with it.
3. Make sure to match the action name and its corresponding action id in the output. Use 'put down' rather than 'drop' to place objects in specific receptacles.
4. Do not repeat the same failed action sequence. Try to modify the action sequence because previous actions did not lead to success.
5. Objects may have multiple instances (e.g., Cabinet_2, Cabinet_3). Explore different instances if needed.
6. Use environment feedback to refine your plan. If an action fails, reflect on the reason and adjust accordingly.

## Task Examples

Example 1: Pick up the alarm clock and turn on the lamp
<think>I need to find the alarm clock, pick it up, then find the desk lamp and turn it on.</think>
<answer>[12, find a AlarmClock] | [78, pick up the AlarmClock] | [25, find a DeskLamp] | [99, turn on the DeskLamp]</answer>

Example 2: Set the box on the table
<think>I need to find the box, pick it up, then find the dining table and put it down.</think>
<answer>[5, find a Box] | [70, pick up the Box] | [20, find a DiningTable] | [110, put down the object in hand]</answer>

## Current Task
{task_instruction}

## Available Actions (0~{n_actions})
{action_list_str}

You should output a plan of up to 20 actions at a time, separated by "|".
Output each action using the format [action_id, action_name].
Your response should be in the format of:
<think>...</think><answer>[id1, action1] | [id2, action2] | ...</answer>

Example 1:
<think>I need to find a mug first. Let me navigate to where mugs might be.</think>
<answer>[42, find a Mug]</answer>

Example 2:
<think>The mug is nearby and I'm not holding anything. I should pick it up.</think>
<answer>[78, pick up the Mug]</answer>

Example 3:
<think>I'm holding the mug and I'm near the table. Let me put it down.</think>
<answer>[110, put down the object in hand]</answer>"""


def extract_action_with_id(action_str: str) -> Optional[str]:
    """
    Extract action from ERA format like "[64, 'find a Ladle']"
    and return in VAGEN format "[64, find a Ladle]" (preserving action ID).
    """
    action_str = action_str.strip()

    # Try parsing as Python literal: [id, 'action name']
    try:
        parsed = ast.literal_eval(action_str)
        if isinstance(parsed, list) and len(parsed) == 2:
            action_id = parsed[0]
            action_name = str(parsed[1])
            return f"[{action_id}, {action_name}]"
    except (ValueError, SyntaxError):
        pass

    # Fallback: regex for [id, 'action name']
    match = re.search(r"\[\s*(\d+)\s*,\s*['\"](.+?)['\"]\s*\]", action_str)
    if match:
        return f"[{match.group(1)}, {match.group(2)}]"

    # If it's just a plain action name (no ID available)
    if not action_str.startswith("["):
        return action_str

    return None


def extract_action_list_from_system(system_text: str) -> List[str]:
    """
    Extract action list from ERA system prompt.
    Format: "action id 0: find a Cart, \naction id 1: find a Potato, ..."
    """
    actions = []
    pattern = r"action id \d+: (.+?)(?:,\s*\n|$)"
    matches = re.findall(pattern, system_text)
    for m in matches:
        actions.append(m.strip().rstrip(","))
    return actions


def extract_task_from_user(user_text: str) -> str:
    """Extract task instruction from ERA user message."""
    match = re.search(r"instruction:\s*(.+?)(?:\n|$)", user_text)
    if match:
        return match.group(1).strip()
    return ""


def build_vagen_system_prompt(task: str, action_list: List[str]) -> str:
    """Build VAGEN-style system prompt."""
    action_list_str = "\n".join(
        f"action id {i}: {a}" for i, a in enumerate(action_list)
    )
    return VAGEN_SYSTEM_TEMPLATE.format(
        task_instruction=task,
        n_actions=len(action_list) - 1,
        action_list_str=action_list_str,
    )


def convert_trajectory_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert one ERA trajectory sample to VAGEN format.

    Returns a dict with:
      - image: path to image
      - conversations: list of {role, content} dicts
      - metadata: original metadata
    """
    convs = sample["conversations"]
    image = sample.get("image", "")
    metadata = sample.get("metadata", {})

    # Find system, human, gpt turns
    system_text = ""
    user_text = ""
    think_text = ""
    action_text = ""

    for c in convs:
        role = c.get("from", c.get("role", ""))
        value = c.get("value", c.get("content", ""))

        if role == "system":
            system_text = value
        elif role == "human" or role == "user":
            user_text = value
        elif role == "gpt" or role == "assistant":
            toa = c.get("think_or_action", 2)
            if toa == 0:
                think_text = value
            elif toa == 1:
                action_text = value
            else:
                # Plain text - treat as thinking
                if not think_text:
                    think_text = value

    # Extract task and action list
    task = extract_task_from_user(user_text)
    if not task:
        return None

    action_list = extract_action_list_from_system(system_text)
    if not action_list:
        return None

    # Extract action with ID preserved
    action_token = extract_action_with_id(action_text)
    if not action_token:
        return None

    # Build VAGEN format
    system_prompt = build_vagen_system_prompt(task, action_list)

    # Determine if this is the first step or a later step
    step_id = metadata.get("step_id", 0)
    if step_id == 0:
        user_content = f"<image>\n[Current Observation]:\nDecide your next action."
    else:
        # For later steps, include feedback from interaction history
        # Extract the last action and feedback from interaction_history in user text
        user_content = f"<image>\n[Current Observation]:\nDescribe what you see, reflect on the feedback, and plan your next actions."

    # Build assistant response in VAGEN format (preserving action ID)
    assistant_content = f"<think>{think_text}</think><answer>{action_token}</answer>"

    vagen_sample = {
        "image": image,
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": user_content},
            {"from": "gpt", "value": assistant_content},
        ],
        "metadata": metadata,
    }
    return vagen_sample


def convert_env_anchored_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert environment-anchored prior sample.
    These may have different structure (QA about environment).
    Preserve multi-modal format, wrap in <think>/<answer> where applicable.
    """
    convs = sample.get("conversations", [])
    image = sample.get("image", "")
    metadata = sample.get("metadata", {})

    new_convs = []
    for c in convs:
        role = c.get("from", c.get("role", ""))
        value = c.get("value", c.get("content", ""))

        if role in ("system", "human", "user"):
            new_convs.append({"from": role if role != "user" else "human", "value": value})
        elif role in ("gpt", "assistant"):
            toa = c.get("think_or_action", 2)
            if toa == 0:
                # Pure thinking - wrap in think tags
                new_convs.append({"from": "gpt", "value": f"<think>{value}</think>"})
            elif toa == 1:
                # Action - extract and wrap
                action_name = extract_action_with_id(value)
                if action_name:
                    # Merge with previous think if exists
                    if new_convs and new_convs[-1]["from"] == "gpt" and "<think>" in new_convs[-1]["value"]:
                        prev = new_convs.pop()
                        think_part = prev["value"]
                        # Close think and add answer
                        if think_part.endswith("</think>"):
                            merged = f"{think_part}<answer>{action_name}</answer>"
                        else:
                            merged = f"{think_part}</think><answer>{action_name}</answer>"
                        new_convs.append({"from": "gpt", "value": merged})
                    else:
                        new_convs.append({"from": "gpt", "value": f"<think>Executing action.</think><answer>{action_name}</answer>"})
                else:
                    new_convs.append({"from": "gpt", "value": value})
            else:
                # Plain text (e.g., environment QA) - keep as is
                new_convs.append({"from": "gpt", "value": value})

    if not new_convs:
        return None

    return {
        "image": image,
        "conversations": new_convs,
        "metadata": metadata,
    }


def convert_external_knowledge_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert external knowledge prior sample.
    These are general QA pairs (OpenO1-SFT) - keep format mostly as-is.
    No action wrapping needed since these are pure reasoning data.
    """
    convs = sample.get("conversations", [])
    image = sample.get("image", "")
    metadata = sample.get("metadata", {})

    new_convs = []
    for c in convs:
        role = c.get("from", c.get("role", ""))
        value = c.get("value", c.get("content", ""))
        new_convs.append({"from": role if role != "user" else "human", "value": value})

    if not new_convs:
        return None

    return {
        "image": image,
        "conversations": new_convs,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert ERA SFT data to VAGEN format")
    parser.add_argument("--trajectory_path", type=str, default=None)
    parser.add_argument("--env_anchored_path", type=str, default=None)
    parser.add_argument("--external_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/root/workspace/era_sft_data/vagen_format")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_converted = []
    stats = {}

    # 1. Trajectory augmented prior dataset
    if args.trajectory_path and os.path.exists(args.trajectory_path):
        print(f"Converting trajectory data from {args.trajectory_path}...")
        with open(args.trajectory_path) as f:
            traj_data = json.load(f)
        print(f"  Raw samples: {len(traj_data)}")

        converted = []
        skipped = 0
        for sample in traj_data:
            result = convert_trajectory_sample(sample)
            if result:
                converted.append(result)
            else:
                skipped += 1

        out_path = os.path.join(args.output_dir, "trajectory_vagen.json")
        with open(out_path, "w") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        print(f"  Converted: {len(converted)}, Skipped: {skipped}")
        print(f"  Saved to: {out_path}")
        stats["trajectory"] = len(converted)
        all_converted.extend(converted)

    # 2. Environment anchored prior dataset
    if args.env_anchored_path and os.path.exists(args.env_anchored_path):
        print(f"\nConverting env-anchored data from {args.env_anchored_path}...")
        with open(args.env_anchored_path) as f:
            env_data = json.load(f)
        print(f"  Raw samples: {len(env_data)}")

        converted = []
        skipped = 0
        for sample in env_data:
            result = convert_env_anchored_sample(sample)
            if result:
                converted.append(result)
            else:
                skipped += 1

        out_path = os.path.join(args.output_dir, "env_anchored_vagen.json")
        with open(out_path, "w") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        print(f"  Converted: {len(converted)}, Skipped: {skipped}")
        print(f"  Saved to: {out_path}")
        stats["env_anchored"] = len(converted)
        all_converted.extend(converted)

    # 3. External knowledge prior dataset
    if args.external_path and os.path.exists(args.external_path):
        print(f"\nConverting external knowledge data from {args.external_path}...")
        with open(args.external_path) as f:
            ext_data = json.load(f)
        print(f"  Raw samples: {len(ext_data)}")

        converted = []
        skipped = 0
        for sample in ext_data:
            result = convert_external_knowledge_sample(sample)
            if result:
                converted.append(result)
            else:
                skipped += 1

        out_path = os.path.join(args.output_dir, "external_knowledge_vagen.json")
        with open(out_path, "w") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        print(f"  Converted: {len(converted)}, Skipped: {skipped}")
        print(f"  Saved to: {out_path}")
        stats["external_knowledge"] = len(converted)
        all_converted.extend(converted)

    # 4. Save combined dataset
    if all_converted:
        combined_path = os.path.join(args.output_dir, "combined_vagen.json")
        with open(combined_path, "w") as f:
            json.dump(all_converted, f, indent=2, ensure_ascii=False)
        print(f"\n=== Combined dataset: {len(all_converted)} samples -> {combined_path}")

    # 5. Create stage.yaml for training
    stage_yaml = {
        "datasets": []
    }
    for name in ["trajectory_vagen", "env_anchored_vagen", "external_knowledge_vagen"]:
        path = os.path.join(args.output_dir, f"{name}.json")
        if os.path.exists(path):
            stage_yaml["datasets"].append({
                "json_path": path,
                "images_folder": "",
                "sampling_strategy": "all",
            })

    stage_path = os.path.join(args.output_dir, "stage.yaml")
    import yaml
    with open(stage_path, "w") as f:
        yaml.dump(stage_yaml, f, default_flow_style=False)
    print(f"Stage config: {stage_path}")

    print(f"\nDone! Stats: {stats}")
    print(f"Total: {len(all_converted)} samples")


if __name__ == "__main__":
    main()
