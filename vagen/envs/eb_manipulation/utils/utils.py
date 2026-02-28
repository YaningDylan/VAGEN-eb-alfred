import re
from typing import Dict, List, Optional
from PIL import Image
import numpy as np


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>

    For EB-Manipulation, the <answer> tag should contain a 7D discrete action
    vector, e.g., "[50, 30, 40, 60, 60, 60, 1]".
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
        actions = [action_content.strip()] if action_content.strip() else []

    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
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
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def parse_action_vector(action_str: str) -> Optional[List[int]]:
    """
    Parse a 7D action vector string into a list of integers.

    Accepts formats like:
      - "[50, 30, 40, 60, 60, 60, 1]"
      - "50, 30, 40, 60, 60, 60, 1"
      - "[50,30,40,60,60,60,1]"

    Returns list of 7 ints, or None if parsing fails.
    """
    # Remove brackets if present
    cleaned = action_str.strip().strip("[]")

    try:
        values = [int(x.strip()) for x in cleaned.split(",")]
    except (ValueError, AttributeError):
        return None

    if len(values) != 7:
        return None

    return values


def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")
