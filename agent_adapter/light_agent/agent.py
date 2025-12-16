from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory
from agent_adapter.tools.light import get_lights, change_state


def get_light_agent() -> ChatAgent:
    client = client_factory.build_client("openai")

    agent = client.create_agent(
        instructions="You are a useful light assistant. can tall user the status of the lights and can help user control the lights on and off",
        name="light",
        tools=[get_lights, change_state],
    )
    return agent


agent = get_light_agent()


async def run(query: str) -> str:
    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


if __name__ == "__main__":
    asyncio.run(run("Can you tell me the status of all the lights?"))
    asyncio.run(run("can you turn off all the lights"))
