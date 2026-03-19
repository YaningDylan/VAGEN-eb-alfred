import re
from typing import Dict, List, Optional
from PIL import Image
import numpy as np


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>

    For EB-ALFRED, the <answer> tag typically contains a single action name
    (e.g., "find a Cabinet") or an action ID (e.g., "42").
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

    reasoning_content = think_content

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
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


def parse_action_token(token: str) -> tuple:
    """
    Parse a single action token that may contain an action ID.

    Supported formats:
      - "[id, action_name]"  e.g. "[64, find a Ladle]"
      - "id, action_name"    e.g. "64, find a Ladle"
      - plain action name    e.g. "find a Ladle"
      - plain integer ID     e.g. "42"

    Returns (action_id_or_none, action_name_or_original).
    """
    text = token.strip()

    # Try "[id, name]" or "[id, 'name']"
    m = re.match(r"^\[?\s*(\d+)\s*,\s*['\"]?(.+?)['\"]?\s*\]?$", text)
    if m:
        return int(m.group(1)), m.group(2).strip()

    # Plain integer
    try:
        return int(text), None
    except ValueError:
        pass

    # Plain action name
    return None, text


def match_action(
    action_name: str,
    action_list: List[str],
    action_map: Dict[str, str],
) -> Optional[str]:
    """
    Match a parsed action against the valid action set.

    Supports formats:
      - "[id, action_name]": "[64, find a Ladle]" — match by ID
      - Action name (case-insensitive): "find a Cabinet"
      - Action ID (integer): "42"

    Returns the original action string if matched, None otherwise.
    """
    action_id, action_name_parsed = parse_action_token(action_name)

    # If we got an action ID, use it directly
    if action_id is not None:
        if 0 <= action_id < len(action_list):
            return action_list[action_id]

    # Try exact match by name (case-insensitive)
    if action_name_parsed:
        matched = action_map.get(action_name_parsed.lower())
        if matched:
            return matched

    return None


def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")
