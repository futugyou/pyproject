import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()

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
        },
    )

    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
    )
    team = MagenticOneGroupChat([assistant], model_client=model_client)
    await Console(team.run_stream(task="Write a Python script to fetch data from an API."))
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
