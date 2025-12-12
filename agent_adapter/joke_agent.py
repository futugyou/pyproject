import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, project_root)

import asyncio
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory


async def JokeAgent(query: str) -> str:
    client = client_factory.build_client("openai")

    agent = client.create_agent(
        instructions="You are good at telling jokes.", name="Joker"
    )

    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


async def JokeWithEmojis(query: str) -> list[str]:
    messages: list[str] = []
    client = client_factory.build_client("openai")

    agent = client.create_agent(
        instructions="You are good at telling jokes.", name="Joker"
    )
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
