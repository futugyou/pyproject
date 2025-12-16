from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from typing import AsyncGenerator
from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory
from agent_adapter.tools.light import get_lights, change_state, LightInfo, LightListInfo


def get_light_agent() -> ChatAgent:
    client = client_factory.build_client("openai")

    agent = client.create_agent(
        instructions="You are a useful light assistant. can tall user the status of the lights and can help user control the lights on and off",
        name="light",
        tools=[get_lights, change_state],
    )
    return agent


agent = get_light_agent()


async def get_lights() -> AsyncGenerator[str, None]:
    response = await agent.run(
        "Can you tell me the status of all the lights?", response_format=LightListInfo
    )
    if response.value:
        lights = response.value
        for light in lights.items:
            yield f"Light {light.id}:{light.name} is {'on' if light.is_on else 'off'}"
    else:
        yield "No structured data found in response"


async def change_light_state() -> str:
    response = await agent.run(
        "can you turn off all the lights?", response_format=LightInfo
    )
    if response.user_input_requests:
        for user_input_needed in response.user_input_requests:
            print(f"Function: {user_input_needed.function_call.name}")
            print(f"Arguments: {user_input_needed.function_call.arguments}")

    approval_message = ChatMessage(
        role=Role.USER, contents=[user_input_needed.create_response(True)]
    )

    response = await agent.run(
        [
            "can you turn off all the lights?",
            ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed]),
            approval_message,
        ],
        response_format=LightInfo,
    )
    if response.value:
        light = response.value
        print(light)
        return f"Light {light.id}:{light.name} is {'on' if light.is_on else 'off'}"
    else:
        return "No structured data found in response"


async def run(query: str) -> str:
    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


async def pack_run():
    async for light_status in get_lights():
        print(light_status)


if __name__ == "__main__":
    # asyncio.run(pack_run())
    asyncio.run(change_light_state())
