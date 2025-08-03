import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily, UserMessage
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.memory.null_memory import NullMemory
from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()


async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is 75 degrees."


async def run() -> None:
    sk_client = OpenAIChatCompletion(
        ai_model_id=os.getenv("GOOGLE_CHAT_MODEL_ID"),
        service_id="default",
        async_client=AsyncOpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url=os.getenv("GOOGLE_URL"),
        ),
    )
    settings = OpenAIChatPromptExecutionSettings(
        temperature=0.2,
    )

    model_client = SKChatCompletionAdapter(
        sk_client,
        kernel=Kernel(memory=NullMemory()),
        prompt_settings=settings,
        model_info={
            "function_calling": True,
            "json_output": True,
            "vision": True,
            "family": ModelFamily.ANY,
            "structured_output": True,
        },
    )

    # Call the model directly.
    response = await model_client.create(
        [UserMessage(content="What is the capital of France?", source="test")]
    )
    print(response)

    # Create an assistant agent with the model client.
    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
        tools=[get_weather],
    )
    # Call the assistant with a task.
    await Console(assistant.run_stream(task="What is the weather in Paris and London?"))


if __name__ == "__main__":
    asyncio.run(run())
