def system_prompt():
    """Static system prompt for EB-Manipulation robot tasks."""
    return """You are a Franka Panda robot with a parallel gripper. You can perform various manipulation tasks by outputting gripper actions to accomplish a given task, guided by images of your current state.

## Input Space
- Each input object is represented as a 3D discrete position: [X, Y, Z].
- There is a red XYZ coordinate frame located in the top-left corner of the table. The X-Y plane is the table surface.
- The allowed range of X, Y, Z is [0, 100].
- Objects are ordered by Y in ascending order.

## Output Action Space
- Each output action is a 7D discrete vector: [X, Y, Z, Roll, Pitch, Yaw, Gripper].
- X, Y, Z: 3D discrete position of the gripper (same coordinate system as input objects). Range: [0, 100].
- Roll, Pitch, Yaw: 3D discrete orientation as Euler angles. Range: [0, 120], each unit = 3 degrees.
- Gripper: 0 = close, 1 = open.

## Color Space
Objects can be described using one of: red, maroon, lime, green, blue, navy, yellow, cyan, magenta, silver, gray, olive, purple, teal, azure, violet, rose, black, white.

## Guidelines
1. Carefully observe the scene and identify all objects, their types, colors, and positions.
2. Plan a sequence of actions to accomplish the task step by step.
3. To pick up an object: move gripper above it (open), lower to it, close gripper, then lift.
4. To place an object: move to target location, open gripper.
5. If an action fails, adjust your plan based on feedback."""


def init_observation_template(task_instruction, object_coords, img_str):
    """Template for initial observation after reset."""
    return f"""[Task]: {task_instruction}

[Object Coordinates]: {object_coords}

[Current Observation]:
{img_str}

Decide your next action."""


def action_template(last_action, env_feedback, object_coords, img_str):
    """Template for step observation with feedback."""
    return f"""[Last Action]: {last_action}
[Feedback]: {env_feedback}

[Object Coordinates]: {object_coords}

[Current Observation]:
{img_str}

Decide your next action."""


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think"):
    """Generate format prompt based on the specified format."""
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_think format."""
    base = f"""You should output {max_actions_per_step} action(s) at a time.
Output the action as a 7D vector [X, Y, Z, Roll, Pitch, Yaw, Gripper] with integer values.
Your response should be in the format of:
<think>...</think><answer>[X, Y, Z, Roll, Pitch, Yaw, Gripper]</answer>"""

    if add_example:
        examples = """
Example 1:
<think>I need to move the gripper above the red cube at position [50, 30, 40]. I'll position the gripper above it with the gripper open.</think>
<answer>[50, 30, 55, 60, 60, 60, 1]</answer>

Example 2:
<think>The gripper is now above the cube. I need to lower it and close the gripper to grasp the cube.</think>
<answer>[50, 30, 40, 60, 60, 60, 0]</answer>

Example 3:
<think>I've grasped the cube. Now I need to lift it up and move it to the target container at [70, 60, 50].</think>
<answer>[70, 60, 60, 60, 60, 60, 0]</answer>"""
        return base + "\n" + examples

    return base
