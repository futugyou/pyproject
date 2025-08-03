from io import BytesIO
import PIL
import requests
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.ui import Console
from autogen_core import Image
from autogen_core.models import UserMessage, ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dotenv import load_dotenv

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

    agent = AssistantAgent("assistant", model_client=model_client)

    pil_image = PIL.Image.open(
        BytesIO(requests.get("https://picsum.photos/300/200").content)
    )
    img = Image(pil_image)
    multi_modal_message = MultiModalMessage(
        content=["Can you describe the content of this image?", img], source="user"
    )
    # result = await agent.run(task=multi_modal_message)
    # print(result.messages[-1].content)

    async def assistant_run_stream() -> None:
        await Console(
            agent.run_stream(task=multi_modal_message),
            output_stats=True,
        )

    await assistant_run_stream()

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
