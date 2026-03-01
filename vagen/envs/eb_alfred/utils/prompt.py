def system_prompt():
    """Static system prompt for EB-ALFRED household robot tasks."""
    return """You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules
- Find: Navigate to a receptacle or object. Always valid if the object exists in the scene.
- Pick up: Pick up a nearby object. Only valid if close to the object, not already holding something, and the object is not in a closed receptacle.
- Put down: Put the held object into a nearby receptacle. Only valid if holding an object.
- Drop: Drop the held object. Does not guarantee placement into a specific receptacle.
- Open: Open a nearby closed receptacle.
- Close: Close a nearby open receptacle.
- Turn on: Turn on a nearby turned-off object.
- Turn off: Turn off a nearby turned-on object.
- Slice: Slice a nearby sliceable object.

## Guidelines
1. Always locate an object using 'find' before interacting with it.
2. Use 'put down' rather than 'drop' to place objects in specific receptacles.
3. Do not repeat the same failed action sequence. Adjust your plan based on feedback.
4. Objects may have multiple instances (e.g., Cabinet_2, Cabinet_3). Try different instances if needed.
5. Use environment feedback to refine your plan. If an action fails, reflect on why and adjust accordingly."""


def init_observation_template(task_instruction, action_list, img_str):
    """Template for initial observation after reset."""
    actions_str = ", ".join(f"{i}: {a}" for i, a in enumerate(action_list))
    return f"""[Task]: {task_instruction}

[Available Actions (0~{len(action_list) - 1})]: {actions_str}

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
Output the action name exactly as listed in the available actions, or the action ID (integer).
Your response should be in the format of:
<think>...</think><answer>action name or action ID</answer>"""

    if add_example:
        examples = """
Example 1:
<think>I need to find a mug first. Let me navigate to where mugs might be.</think>
<answer>find a Mug</answer>

Example 2:
<think>The mug is nearby and I'm not holding anything. I should pick it up.</think>
<answer>pick up the Mug</answer>

Example 3:
<think>I'm holding the mug and I'm near the table. Let me put it down.</think>
<answer>put down the object in hand</answer>"""
        return base + "\n" + examples

    return base


def wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for wm format with observation and prediction tags."""
    base = f"""You should output {max_actions_per_step} action(s) at a time.
Output the action name exactly as listed in the available actions, or the action ID (integer).
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>action name or action ID</answer><prediction>...</prediction>.

Rules for <observation>:
- Describe the current scene: what objects you see, your position, what you are holding, and relevant receptacle states.

Rules for <prediction>:
- Predict what will change after your action: where you will be, what you will see, and the expected result.

Rules for <answer>:
- Output exactly 1 action name or action ID."""

    if add_example:
        examples = """
Example 1:
<observation>I see a kitchen with a counter, a microwave, and a mug on the counter. I am not holding anything.</observation>
<think>I need to pick up the mug. First, I should find it to get close to it.</think>
<answer>find a Mug</answer>
<prediction>I will navigate to the mug and see it up close on the counter.</prediction>

Example 2:
<observation>I am close to a Mug on the counter. I am not holding anything. The mug is within reach.</observation>
<think>The mug is nearby and I'm not holding anything. I should pick it up.</think>
<answer>pick up the Mug</answer>
<prediction>I will be holding the mug. The counter will no longer have the mug on it.</prediction>

Example 3:
<observation>I am holding a Mug. I see a table nearby with an empty spot.</observation>
<think>I'm holding the mug and I'm near the table. Let me put it down.</think>
<answer>put down the object in hand</answer>
<prediction>The mug will be placed on the table. I will no longer be holding anything.</prediction>"""
        return base + "\n" + examples

    return base
