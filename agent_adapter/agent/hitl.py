from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any
from agent_framework import ChatAgent, ChatClientProtocol, ai_function
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory

# Based on the official example, I want to understand how the data is generated.
# https://docs.copilotkit.ai/microsoft-agent-framework/human-in-the-loop
class StepStatus(str, Enum):
    """Status of a task step."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class TaskStep(BaseModel):
    """A single step in a task execution plan."""

    description: str = Field(..., description="The text of the step in imperative form (e.g., 'Dig hole', 'Open door')")
    status: StepStatus = Field(default=StepStatus.ENABLED, description="Whether the step is enabled or disabled")


@ai_function(
    name="generate_task_steps",
    description="Generate execution steps for a task",
    approval_mode="always_require",
)

def generate_task_steps(steps: list[TaskStep]) -> str:
    """Make up 10 steps (only a couple of words per step) that are required for a task.

    The step should be in imperative form (i.e. Dig hole, Open door, ...).
    Each step will have status='enabled' by default.

    Args:
        steps: An array of 10 step objects, each containing description and status

    Returns:
        Confirmation message
    """
    print(f"Generated {len(steps)} execution steps for the task.")
    return f"Generated {len(steps)} execution steps for the task."

# During the first interaction, the backend generates CUSTOM data called `function_approval_request` to indicate that the operation requires user approval.
# And TOOL_CALL_START data with `"toolCallName":"confirm_changes"`
def get_hitl_agent(chat_client: ChatClientProtocol[Any]) -> ChatAgent[Any]:
    """Create a human-in-the-loop agent using tool-based approach for predictive state.

    Args:
        chat_client: The chat client to use for the agent

    Returns:
        A configured ChatAgent instance with human-in-the-loop capabilities
    """
    return ChatAgent(
        name="human_in_the_loop_agent",
        instructions="""You are a helpful assistant that can perform any task by breaking it down into steps.

    When asked to perform a task, you MUST call the `generate_task_steps` function with the proper
    number of steps per the request.

    Rules for steps:
    - Each step description should be in imperative form (e.g., "Dig hole", "Open door", "Prepare ingredients")
    - Each step should be brief (only a couple of words)
    - All steps must have status='enabled' initially

    Example steps for "Build a robot":
    1. "Design blueprint"
    2. "Gather components"
    3. "Assemble frame"
    4. "Install motors"
    5. "Wire electronics"
    6. "Program controller"
    7. "Test movements"
    8. "Add sensors"
    9. "Calibrate systems"
    10. "Final testing"

    IMPORTANT: When you call generate_task_steps, the user will be shown the steps and asked to approve.
    Do NOT output any text along with the function call - just call the function.
    After the user approves and the function executes, THEN provide a brief acknowledgment like:
    "The plan has been created with X steps selected."
    """,
        chat_client=chat_client,
        tools=[generate_task_steps],
    )