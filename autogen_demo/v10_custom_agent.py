import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination, SourceMatchTermination
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import UserMessage, ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from count_down_agent import CountDownAgent
from arithmetic_agent import ArithmeticAgent
from gemini_assistant_agent import GeminiAssistantAgent

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

    countdown_agent = CountDownAgent("CountDownAgent")

    async for message in countdown_agent.on_messages_stream([], CancellationToken()):
        if isinstance(message, Response):
            print(message.chat_message)
        else:
            print(message)

    print("\n")

    add_agent = ArithmeticAgent("add_agent", "Adds 1 to the number.", lambda x: x + 1)
    multiply_agent = ArithmeticAgent(
        "multiply_agent", "Multiplies the number by 2.", lambda x: x * 2
    )
    subtract_agent = ArithmeticAgent(
        "subtract_agent", "Subtracts 1 from the number.", lambda x: x - 1
    )
    divide_agent = ArithmeticAgent(
        "divide_agent", "Divides the number by 2 and rounds down.", lambda x: x // 2
    )
    identity_agent = ArithmeticAgent(
        "identity_agent", "Returns the number as is.", lambda x: x
    )

    # The termination condition is to stop after 10 messages.
    termination_condition = MaxMessageTermination(10)
    source_termination = SourceMatchTermination("identity_agent")

    # Create a selector group chat.
    selector_group_chat = SelectorGroupChat(
        [add_agent, multiply_agent, subtract_agent, divide_agent, identity_agent],
        model_client=model_client,
        termination_condition=termination_condition | source_termination,
        allow_repeated_speaker=True,  # Allow the same agent to speak multiple times, necessary for this task.
        selector_prompt=(
            "Available roles:\n{roles}\nTheir job descriptions:\n{participants}\n"
            "Current conversation history:\n{history}\n"
            "Please select the most appropriate role for the next message, and only return the role name."
        ),
    )

    # Run the selector group chat with a given task and stream the response.
    task: List[BaseChatMessage] = [
        TextMessage(
            content="Apply the operations to turn the given number into 22.",
            source="user",
        ),
        TextMessage(content="10", source="user"),
    ]
    stream = selector_group_chat.run_stream(task=task)
    await Console(stream)
    await model_client.close()

    print("\n")

    gemini_assistant = GeminiAssistantAgent("gemini_assistant")
    await Console(gemini_assistant.run_stream(task="What is the capital of New York?"))


if __name__ == "__main__":
    asyncio.run(run())
