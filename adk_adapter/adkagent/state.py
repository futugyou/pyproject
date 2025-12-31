from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from typing import Dict
from pydantic import BaseModel
from google.adk.models.google_llm import BaseLlm
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import ToolContext
from google.adk import tools as adk_tools
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint

from adk_adapter import client_factory


class AgentState(BaseModel):
    """State for the agent."""

    language: str = "english"


# ADK provides ToolContext which allows setting the state, but MAF currently lacks this functionality.
def set_language(tool_context: ToolContext, new_language: str) -> Dict[str, str]:
    """Sets the language preference for the user.
    Args:
        tool_context (ToolContext): The tool context for accessing state.
        new_language (str): The language to save in state.
    Returns:
        Dict[str, str]: A dictionary indicating success status and message.
    """
    tool_context.state["language"] = new_language
    return {"status": "success", "message": f"Language set to {new_language}"}


def build_state_agent(llm: BaseLlm) -> BaseAgent:
    base_agent = LlmAgent(
        name="adk_state",
        model=llm,
        instruction="""
        You are a helpful assistant that can change language settings.
        """,
        tools=[set_language],
    )

    return base_agent


if __name__ == "__main__":
    from adk_adapter import adkutil

    llm = client_factory.build_llm()
    base_agent = build_state_agent(llm)
    asyncio.run(adkutil.run_agent(base_agent, "hello"))
