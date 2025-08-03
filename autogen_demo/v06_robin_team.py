import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import UserMessage, ModelFamily
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

    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat(
        [primary_agent, critic_agent], termination_condition=text_termination
    )

    result = await team.run(task="Write a short poem about the fall season.")

    for msg in result.messages:
        if isinstance(msg, TextMessage):
            print(f"{msg.source}:{msg.content}\n")

    await team.reset()  # Reset the team for a new task.
    async for message in team.run_stream(
        task="Write a short poem about the fall season."
    ):  # type: ignore
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        else:
            print(message)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
