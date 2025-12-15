from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory


def get_weather_agent() -> ChatAgent:
    client = client_factory.build_client("openai")

    agent = client.create_agent(
        instructions="You are good at telling jokes.", name="Joker"
    )
    return agent


agent = get_weather_agent()


async def run(query: str) -> str:
    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


async def JokeWithEmojis(query: str) -> list[str]:
    messages: list[str] = []
    thread = agent.get_new_thread()
    result = await agent.run(query, thread=thread)
    text = result.text
    messages.append(text)
    print(f"message: {text}\n")
    result = await agent.run("Add emojis to previous jokes", thread=thread)
    text = result.text
    messages.append(text)
    print(f"message: {text}\n")
    return messages


if __name__ == "__main__":
    # asyncio.run(JokeAgent("Tell me a joke about a pirate."))
    asyncio.run(JokeWithEmojis("Tell me a joke about a pirate."))
