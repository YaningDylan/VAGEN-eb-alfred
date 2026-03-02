"""Utility functions for EB-Navigation environment in VAGEN."""

import re
from typing import Dict, List, Optional
from PIL import Image
import numpy as np


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>

    For EB-Navigation, the <answer> tag contains an action name
    (e.g., "Move forward by 0.25") or an action ID (e.g., "0").
    """
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)

    format_correct = match is not None

    if not match:
        think_content = ""
        action_content = ""
        actions = []
    else:
        think_content = match.group(1).strip()
        action_content = match.group(2).strip()

        if max_actions == 1:
            actions = [action_content.strip()] if action_content.strip() else []
        else:
            actions = [a.strip() for a in action_content.split(action_sep) if a.strip()]
            if len(actions) > max_actions:
                actions = actions[:max_actions]
                action_content = action_sep.join(actions)

    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_wm(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    """
    Parse wm format response:
    <observation>...</observation>
    <think>...</think>
    <answer>...</answer>
    <prediction>...</prediction>
    """
    pattern = (
        r'<observation>(.*?)</observation>\s*'
        r'<think>(.*?)</think>\s*'
        r'<answer>(.*?)</answer>\s*'
        r'<prediction>(.*?)</prediction>'
    )

    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None

    if not match:
        observation_content = ""
        think_content = ""
        prediction_content = ""
        action_content = ""
        actions: List[str] = []
    else:
        observation_content = match.group(1).strip()
        think_content = match.group(2).strip()
        action_content = match.group(3).strip()
        prediction_content = match.group(4).strip()

        if max_actions == 1:
            actions = [action_content.strip()] if action_content.strip() else []
        else:
            actions = [a.strip() for a in action_content.split(action_sep) if a.strip()]
            if len(actions) > max_actions:
                actions = actions[:max_actions]
                action_content = action_sep.join(actions)

    llm_response = (
        f"<observation>{observation_content}</observation>"
        f"<think>{think_content}</think>"
        f"<answer>{action_content}</answer>"
        f"<prediction>{prediction_content}</prediction>"
    )

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": think_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = ",",
    max_actions: int = 1,
) -> Dict:
    """Parse LLM response based on the specified prompt format."""
    if prompt_format == "free_think":
        return parse_free_think(response, action_sep, max_actions)
    elif prompt_format == "wm":
        return parse_wm(response, action_sep, max_actions)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def match_action(
    action_name: str,
    action_list: List[str],
    action_map: Dict[str, str],
) -> Optional[int]:
    """
    Match a parsed action against the valid navigation action set.

    Supports two formats:
      - Action name (case-insensitive): "Move forward by 0.25"
      - Action ID (integer): "0"

    Returns the action index (int) if matched, None otherwise.
    """
    name = action_name.strip()

    # Try as integer action ID
    try:
        idx = int(name)
        if 0 <= idx < len(action_list):
            return idx
    except ValueError:
        pass

    # Try exact match by name (case-insensitive)
    matched_name = action_map.get(name.lower())
    if matched_name is not None:
        return action_list.index(matched_name)

    # Try fuzzy match: check if the input is a substring of any action name
    name_lower = name.lower()
    for i, action in enumerate(action_list):
        if name_lower in action.lower() or action.lower() in name_lower:
            return i

    return None


def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")
