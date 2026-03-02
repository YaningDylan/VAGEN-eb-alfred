"""Prompt templates for EB-Navigation environment in VAGEN."""

# Fixed action set for navigation (8 discrete actions)
DISCRETE_SKILLSET = [
    "Move forward by 0.25",
    "Move backward by 0.25",
    "Move rightward by 0.25",
    "Move leftward by 0.25",
    "Rotate to the right by 90 degrees.",
    "Rotate to the left by 90 degrees.",
    "Tilt the camera upward by 30 degrees.",
    "Tilt the camera downward by 30 degrees.",
]


def system_prompt():
    """Static system prompt for EB-Navigation tasks."""
    actions_str = "\n".join(f"  {i}: {a}" for i, a in enumerate(DISCRETE_SKILLSET))
    return f"""You are a robot navigating a home environment. Your goal is to navigate to a target object and get as close as possible to it.

## Available Actions
{actions_str}

## Guidelines
1. Observe the scene carefully to identify the target object or likely locations.
2. Use movement actions (forward, backward, left, right) to traverse the space. Each move covers 0.25 meters.
3. Use rotation actions to change your facing direction by 90 degrees.
4. Use camera tilt to look up or down by 30 degrees if the target might be above or below your current view.
5. If an action fails (e.g., collision with a wall), try a different direction.
6. Plan efficient paths — avoid unnecessary back-and-forth movements.
7. You succeed when you are within 1 meter of the target object."""


def init_observation_template(task_instruction, img_str):
    """Template for initial observation after reset."""
    return f"""[Task]: {task_instruction}

[Current Observation]:
{img_str}

Decide your next action."""


def action_template(last_action, env_feedback, img_str):
    """Template for step observation with feedback."""
    return f"""[Last Action]: {last_action}
[Feedback]: {env_feedback}

[Current Observation]:
{img_str}

Decide your next action."""


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think"):
    """Generate format prompt based on the specified format."""
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return wm_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_think format."""
    base = f"""You should output {max_actions_per_step} action(s) at a time.
Output the action name exactly as listed in the available actions, or the action ID (integer 0-7).
Your response should be in the format of:
<think>...</think><answer>action name or action ID</answer>"""

    if add_example:
        examples = """
Example 1:
<think>I can see a table ahead but the target bread is not visible. I should rotate to look around the room.</think>
<answer>Rotate to the right by 90 degrees.</answer>

Example 2:
<think>I can see the bread on the counter ahead. I should move forward to get closer.</think>
<answer>0</answer>

Example 3:
<think>The target might be on a shelf above. Let me tilt the camera up to check.</think>
<answer>Tilt the camera upward by 30 degrees.</answer>"""
        return base + "\n" + examples

    return base


def wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for wm format with observation and prediction tags."""
    base = f"""You should output {max_actions_per_step} action(s) at a time.
Output the action name exactly as listed in the available actions, or the action ID (integer 0-7).
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>action name or action ID</answer><prediction>...</prediction>.

Rules for <observation>:
- Describe the current scene: what objects you see, your relative position, and any obstacles.

Rules for <prediction>:
- Predict what will change after your action: what you expect to see, and whether you'll be closer to the target.

Rules for <answer>:
- Output exactly 1 action name or action ID."""

    if add_example:
        examples = """
Example 1:
<observation>I see a kitchen with counters and a microwave. The target bread is not visible from this angle.</observation>
<think>I should rotate to scan the room and find where the bread might be.</think>
<answer>Rotate to the right by 90 degrees.</answer>
<prediction>I will face a new direction and may see the bread or other clues about its location.</prediction>

Example 2:
<observation>I can see bread on a counter about 2 meters ahead. The path forward is clear.</observation>
<think>The bread is directly ahead. I should move forward to get closer to it.</think>
<answer>Move forward by 0.25</answer>
<prediction>I will be 0.25 meters closer to the bread. It should appear larger in my view.</prediction>

Example 3:
<observation>I am in a narrow hallway and my last forward movement failed due to collision.</observation>
<think>There's an obstacle ahead. I should try moving sideways to get around it.</think>
<answer>Move rightward by 0.25</answer>
<prediction>I will shift right by 0.25 meters, potentially clearing the obstacle.</prediction>"""
        return base + "\n" + examples

    return base
