import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
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

    # Create the writer agent
    writer = AssistantAgent(
        "writer",
        model_client=model_client,
        system_message="Draft a short paragraph on climate change.",
    )

    # Create two editor agents
    editor1 = AssistantAgent(
        "editor1",
        model_client=model_client,
        system_message="Edit the paragraph for grammar.",
    )

    editor2 = AssistantAgent(
        "editor2",
        model_client=model_client,
        system_message="Edit the paragraph for style.",
    )

    # Create the final reviewer agent
    final_reviewer = AssistantAgent(
        "final_reviewer",
        model_client=model_client,
        system_message="Consolidate the grammar and style edits into a final version.",
    )

    # Build the workflow graph
    builder = DiGraphBuilder()
    builder.add_node(writer).add_node(editor1).add_node(editor2).add_node(
        final_reviewer
    )

    # Fan-out from writer to editor1 and editor2
    builder.add_edge(writer, editor1)
    builder.add_edge(writer, editor2)

    # Fan-in both editors into final reviewer
    builder.add_edge(editor1, final_reviewer)
    builder.add_edge(editor2, final_reviewer)

    # Build and validate the graph
    graph = builder.build()

    # Create the flow
    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
    )

    # Run the workflow
    await Console(flow.run_stream(task="Write a short paragraph about climate change."))
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
