from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory
from agent_adapter.tools.weather import get_weather


def get_weather_agent() -> ChatAgent:
    client = client_factory.build_client("openai")

    agent = client.create_agent(
        instructions="You are a helpful weather assistant",
        name="weather",
        tools=get_weather,
    )
    return agent


agent = get_weather_agent()


async def run(query: str) -> str:
    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


if __name__ == "__main__":
    asyncio.run(run("What is the weather like in Amsterdam?"))
