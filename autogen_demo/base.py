import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import BaseChatAgent
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


async def run() -> None:
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("GOOGLE_CHAT_MODEL_ID"),
        base_url=os.getenv("GOOGLE_URL"),
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.ANY,
            "structured_output": True,
        },
    )

    response = await model_client.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
    print(response)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
