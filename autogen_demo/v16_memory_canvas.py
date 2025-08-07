from autogen_core.models import ModelFamily
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.memory.canvas import TextCanvasMemory

import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

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
            "multiple_system_messages": True,
        },
    )

    text_canvas_memory = TextCanvasMemory()

    # Get tools for working with the canvas
    update_file_tool = text_canvas_memory.get_update_file_tool()
    apply_patch_tool = text_canvas_memory.get_apply_patch_tool()

    # Create an agent with the canvas memory and tools
    writer_agent = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="A writer agent that creates and updates stories.",
        system_message="""
        You are a Writer Agent. Your focus is to generate a story based on the user's request.

        Instructions for using the canvas:

        - The story should be stored on the canvas in a file named "story.md".
        - If "story.md" does not exist, create it by calling the 'update_file' tool.
        - If "story.md" already exists, generate a unified diff (patch) from the current
          content to the new version, and call the 'apply_patch' tool to apply the changes.

        IMPORTANT: Do not include the full story text in your chat messages.
        Only write the story content to the canvas using the tools.
        """,
        tools=[update_file_tool, apply_patch_tool],
        memory=[text_canvas_memory],
    )

    # Send a message to the agent
    await writer_agent.on_messages(
        [
            TextMessage(
                content="Write a short story about a bunny and a sunflower.",
                source="user",
            )
        ],
        CancellationToken(),
    )

    # Retrieve the content from the canvas
    story_content = text_canvas_memory.canvas.get_latest_content("story.md")
    print("Story content from canvas:")
    print(story_content)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
