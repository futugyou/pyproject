from typing import Dict, List
from pydantic import BaseModel
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import ToolContext
from google.adk.models.google_llm import BaseLlm

from adk_adapter import client_factory


class AgentState(BaseModel):
    """State for the agent."""

    question: str = ""  # Input: received from frontend
    answer: str = ""  # Output: sent to frontend
    resources: List[str] = []  # Internal: not shared with frontend


def answer_question(tool_context: ToolContext, answer: str) -> Dict[str, str]:
    """Stores the answer to the user's question.

    Args:
        tool_context (ToolContext): The tool context for accessing state.
        answer (str): The answer to store in state.

    Returns:
        Dict[str, str]: A dictionary indicating success status.
    """
    tool_context.state["answer"] = answer
    
    question = tool_context.state.get("question", "")
    resources = tool_context.state.get("resources", [])
    print(f"resources is: {resources}, question is: {question}, answer is: {answer}")

    return {"status": "success", "message": "Answer stored."}


def add_resource(tool_context: ToolContext, resource: str) -> Dict[str, str]:
    """Adds a resource to the internal resources list.

    Args:
        tool_context (ToolContext): The tool context for accessing state.
        resource (str): The resource URL or reference to add.

    Returns:
        Dict[str, str]: A dictionary indicating success status.
    """
    resources = tool_context.state.get("resources", [])
    resources.append(resource)
    tool_context.state["resources"] = resources

    question = tool_context.state.get("question", "")
    answer = tool_context.state.get("answer", "")
    print(f"resources is: {resources}, question is: {question}, answer is: {answer}")

    return {"status": "success", "message": "Resource added."}


def build_answer_agent(llm: BaseLlm) -> BaseAgent:
    base_agent = LlmAgent(
        name="adk_answer",
        model=llm,
        instruction="""
    You are a helpful assistant. When answering questions:
    1. Use add_resource to track any sources you reference (internal use)
    2. Use answer_question to provide your final answer to the user
    
    The question from the user is available in state as 'question'.
        """,
        tools=[answer_question, add_resource],
    )

    print("answer_agent was called.")
    return base_agent


if __name__ == "__main__":
    import asyncio
    from adk_adapter import adkutil

    llm = client_factory.build_llm()
    base_agent = build_answer_agent(llm)
    asyncio.run(adkutil.run_agent(base_agent, "hello"))
