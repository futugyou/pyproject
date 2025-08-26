import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from autogen import ConversableAgent

from semantic_kernel.agents import (
    AutoGenConversableAgent,
    AutoGenConversableAgentThread,
)


async def main():
    thread: AutoGenConversableAgentThread = None

    cathy = ConversableAgent(
        "cathy",
        system_message="Your name is Cathy and you are a part of a duo of comedians.",
        llm_config={
            "config_list": [
                {
                    "model": os.getenv("OPENAI_CHAT_MODEL_ID"),
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "api_type": "openai",
                    "temperature": 0.9,
                    "base_url": os.getenv("OPENAI_URL"),
                }
            ]
        },
        human_input_mode="NEVER",  # Never ask for human input.
    )

    cathy_autogen_agent = AutoGenConversableAgent(conversable_agent=cathy)

    joe = ConversableAgent(
        "joe",
        system_message="Your name is Joe and you are a part of a duo of comedians.",
        llm_config={
            "config_list": [
                {
                    "model": os.getenv("OPENAI_CHAT_MODEL_ID"),
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "api_type": "openai",
                    "temperature": 0.7,
                    "base_url": os.getenv("OPENAI_URL"),
                }
            ]
        },
        human_input_mode="NEVER",  # Never ask for human input.
    )

    joe_autogen_agent = AutoGenConversableAgent(conversable_agent=joe)

    async for response in cathy_autogen_agent.invoke(
        recipient=joe_autogen_agent,
        message="Tell me a joke about the stock market.",
        thread=thread,
        max_turns=3,
    ):
        print(f"# {response.role} - {response.name or '*'}: '{response}'")
        thread = response.thread

    # Cleanup: Delete the thread and agent
    await thread.delete() if thread else None


if __name__ == "__main__":
    asyncio.run(main())
