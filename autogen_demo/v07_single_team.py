import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    ExternalTermination,
    TextMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import UserMessage, ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from function_call_termination import FunctionCallTermination

load_dotenv()


def increment_number(number: int) -> int:
    """Increment a number by 1."""
    return number + 1


def approve() -> None:
    """Approve the message when all feedbacks have been addressed."""
    pass


async def run() -> None:
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("GOOGLE_CHAT_MODEL_ID"),
        base_url=os.getenv("GOOGLE_URL"),
        api_key=os.getenv("GOOGLE_API_KEY"),
        parallel_tool_calls=False,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.ANY,
            "structured_output": True,
        },
    )

    looped_assistant = AssistantAgent(
        "looped_assistant",
        model_client=model_client,
        tools=[increment_number],  # Register the tool.
        system_message="You are a helpful AI assistant, use the tool to increment the number.",
    )
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        tools=[approve],  # Register the approve function as a tool.
        system_message="Provide constructive feedback. Use the approve tool to approve when all feedbacks are addressed.",
    )

    # Termination condition that stops the task if the agent responds with a text message.
    termination_condition = TextMessageTermination("looped_assistant")

    function_call_termination = FunctionCallTermination(function_name="approve")

    # Create a team with the looped assistant agent and the termination condition.
    team = RoundRobinGroupChat(
        [looped_assistant],
        termination_condition=termination_condition,
    )

    # Run the team with a task and print the messages to the console.
    # TextMessage
    # ToolCallRequestEvent
    # ToolCallExecutionEvent
    # ToolCallSummaryMessage
    # TextMessage
    # TaskResult
    async for message in team.run_stream(task="Increment the number 5 to 7."):  # type: ignore
        print(type(message).__name__, message)

    team = RoundRobinGroupChat(
        [primary_agent, critic_agent],
        termination_condition=function_call_termination,
    )

    async for message in team.run_stream(
        task="Write a unique, Haiku about the weather in Paris"
    ):  # type: ignore
        print(type(message).__name__, message)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
