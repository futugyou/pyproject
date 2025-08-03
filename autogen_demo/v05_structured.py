import asyncio
import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_core.models import UserMessage, ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


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

    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="Categorize the input as happy, sad, or neutral following the JSON format.",
        # Define the output content type of the agent.
        output_content_type=AgentResponse,
        model_client_stream=True,
    )

    # async for message in agent.run_stream(task="I am happy."):  # type: ignore
    #     print(message)

    result = await Console(agent.run_stream(task="I am happy."))

    # Check the last message in the result, validate its type, and print the thoughts and response.
    assert isinstance(result.messages[-1], StructuredMessage)
    assert isinstance(result.messages[-1].content, AgentResponse)
    print("Thought: ", result.messages[-1].content.thoughts)
    print("Response: ", result.messages[-1].content.response)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
