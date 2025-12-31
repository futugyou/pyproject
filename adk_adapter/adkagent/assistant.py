from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


from google.adk.models.google_llm import BaseLlm
from google.adk.agents import LlmAgent, BaseAgent
from google.adk import tools as adk_tools
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint

from adk_adapter import client_factory


def build_assistant_agent(llm: BaseLlm) -> BaseAgent:
    base_agent = LlmAgent(
        name="assistant",
        model=llm,
        instruction="""
        You are a helpful assistant. Help users by answering their questions and assisting with their needs.
        - If the user greets you, please greet them back with specifically with "Hello".
        - If the user greets you and does not make any request, greet them and ask "how can I assist you?"
        - If the user makes a statement without making a request, you do not need to tell them you can't do anything about it.
        Try to say something conversational about it in response, making sure to mention the topic directly.
        - If the user asks you a question, if possible you can answer it using previous context without telling them that you cannot look it up.
        Only tell the user that you cannot search if you do not have enough information already to answer.
        """,
        tools=[adk_tools.preload_memory_tool.PreloadMemoryTool()],
    )

    return base_agent


if __name__ == "__main__":
    import asyncio
    from adk_adapter import adkutil

    llm = client_factory.build_llm()
    base_agent = build_assistant_agent(llm)
    asyncio.run(adkutil.run_agent(base_agent, "hello"))
