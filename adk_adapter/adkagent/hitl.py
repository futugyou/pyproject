from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))

from ag_ui_adk import ADKAgent
from google.adk.models.google_llm import BaseLlm
from google.adk.agents import Agent, LlmAgent, BaseAgent
from google.genai import types

# Choosing to write code in a way that's not meant for humans in an example intended for human reading is just ridiculous.
DEFINE_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_task_steps",
        "description": "Make up 10 steps (only a couple of words per step) that are required for a task. The step should be in imperative form (i.e. Dig hole, Open door, ...)",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "The text of the step in imperative form"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["enabled"],
                                "description": "The status of the step, always 'enabled'"
                            }
                        },
                        "required": ["description", "status"]
                    },
                    "description": "An array of 10 step objects, each containing text and status"
                }
            },
            "required": ["steps"]
        }
    }
}

def build_hitl_agent(llm: BaseLlm) -> BaseAgent:
    base_agent = LlmAgent(
        name="human_in_loop_agent",
        model=llm,
        instruction=f"""
        You are a human-in-the-loop task planning assistant that helps break down complex tasks into manageable steps with human oversight and approval.

        **Your Primary Role:**
        - Generate clear, actionable task steps for any user request
        - Facilitate human review and modification of generated steps
        - Execute only human-approved steps

        **When a user requests a task:**
        1. ALWAYS call the `generate_task_steps` function to create 10 step breakdown
        2. Each step must be:
        - Written in imperative form (e.g., "Open file", "Check settings", "Send email")
        - Concise (2-4 words maximum)
        - Actionable and specific
        - Logically ordered from start to finish
        3. Initially set all steps to "enabled" status
        4. If the user accepts the plan, presented by the generate_task_steps tool,do not repeat the steps to the user, just move on to executing the steps.
        5. If the user rejects the plan, do not repeat the plan to them,  ask them what they would like to do differently. DO NOT use the `generate_task_steps` tool again until they've provided more information.


        **When executing steps:**
        - Only execute steps with "enabled" status.
        - For each step you are executing, tell the user what you are doing.
        - Pretend you are executing the step in real life and refer to it in the current tense. End each step with an ellipsis.
        - Each step MUST be on a new line. DO NOT combine steps into one line.
        - For example for the following steps:
            - Inhale deeply
            - Exhale forcefully
            - Produce sound
            a good response would be:
            ```
            Inhaling deeply...
            Exhaling forcefully...
            Producing sound...
            ```
            a bad response would be `Inhaling deeply... Exhaling forcefully... Producing sound...` because it is on one line.
        - Skip any steps marked as "disabled"
        - Afterwards, confirm the execution of the steps to the user, e.g. if the user asked for a plan to go to mars, respond like "I have completed the plan and gone to mars"
        - EVERY STEP AND THE CONFIRMATION MUST BE ON A NEW LINE. DO NOT COMBINE THEM INTO ONE LINE. USE A <br> TAG TO SEPARATE THEM.

        **Key Guidelines:**
        - Always generate exactly 10 steps
        - Make steps granular enough to be independently enabled/disabled

        Tool reference: {DEFINE_TASK_TOOL}
        """,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.7,  # Slightly higher temperature for creativity
            top_p=0.9,
            top_k=40
        ),
    )

    return base_agent
