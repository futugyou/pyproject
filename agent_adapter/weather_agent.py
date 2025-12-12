import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, project_root)

import asyncio
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory
from agent_adapter.tools.weather import get_weather


async def WeatherAgent(query: str) -> str:
    client = client_factory.build_client("openai")

    agent = client.create_agent(
        instructions="You are a helpful weather assistant",
        name="weather",
        tools=get_weather,
    )

    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


if __name__ == "__main__":
    asyncio.run(WeatherAgent("What is the weather like in Amsterdam?"))
