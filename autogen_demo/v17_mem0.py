from autogen_core.models import ModelFamily
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.mem0 import Mem0Memory

import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

load_dotenv()


async def get_weather(city: str, units: str = "imperial") -> str:
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."
    else:
        return f"Sorry, I don't know the weather in {city}."


async def run() -> None:
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_CHAT_MODEL_ID"),
        base_url=os.getenv("OPENAI_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.ANY,
            "structured_output": True,
            "multiple_system_messages": True,
        },
    )

    mem0_memory = Mem0Memory(
        is_cloud=True,
        api_key=os.getenv("MEM0_API_KEY"),
        limit=5,  # Maximum number of memories to retrieve
    )

    # Add user preferences to memory
    await mem0_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "units"},
        )
    )

    await mem0_memory.add(
        MemoryContent(
            content="Meal recipe must be vegan",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "dietary"},
        )
    )

    # Create assistant with mem0 memory
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[mem0_memory],
    )

    # Ask about the weather
    stream = assistant_agent.run_stream(task="What are my dietary preferences?")
    await Console(stream)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
