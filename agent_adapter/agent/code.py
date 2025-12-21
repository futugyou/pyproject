from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from agent_framework import ChatAgent, HostedCodeInterpreterTool
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory


def get_code_agent() -> ChatAgent:
    client = client_factory.build_client("openai")
    agent = client.create_agent(
        instructions="You are a helpful assistant that can write and execute Python/golang/C# code to solve problems.",
        name="code",
        tools=HostedCodeInterpreterTool(),
    )
    return agent


async def run(query: str) -> str:
    agent = get_code_agent()
    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


if __name__ == "__main__":
    asyncio.run(
        run(
            "Generate the fibonacci numbers to 100 using python code, show the code and execute it. use C#"
        )
    )
