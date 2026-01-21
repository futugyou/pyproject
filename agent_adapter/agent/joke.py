from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from agent_framework import ChatAgent, ChatClientProtocol
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory


def get_joke_agent(client: ChatClientProtocol) -> ChatAgent:
    agent = client.as_agent(instructions="You are good at telling jokes.", name="Joker")
    return agent


async def run(client: ChatClientProtocol, query: str) -> str:
    agent = get_joke_agent(client)
    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


async def JokeWithEmojis(client: ChatClientProtocol, query: str) -> list[str]:
    agent = get_joke_agent(client)
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
    client = client_factory.build_client("openai")
    # asyncio.run(JokeAgent(client, "Tell me a joke about a pirate."))
    asyncio.run(JokeWithEmojis(client, "Tell me a joke about a pirate."))
