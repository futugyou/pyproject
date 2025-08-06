import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

from autogen_agentchat.agents import (
    AssistantAgent,
    MessageFilterAgent,
    MessageFilterConfig,
    PerSourceFilter,
)
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

    # Create agents
    researcher = AssistantAgent(
        "researcher",
        model_client=model_client,
        system_message="Summarize key facts about climate change.",
    )
    analyst = AssistantAgent(
        "analyst",
        model_client=model_client,
        system_message="Review the summary and suggest improvements.",
    )
    presenter = AssistantAgent(
        "presenter",
        model_client=model_client,
        system_message="Prepare a presentation slide based on the final summary.",
    )

    # Apply message filtering
    filtered_analyst = MessageFilterAgent(
        name="analyst",
        wrapped_agent=analyst,
        filter=MessageFilterConfig(
            per_source=[PerSourceFilter(source="researcher", position="last", count=1)]
        ),
    )

    filtered_presenter = MessageFilterAgent(
        name="presenter",
        wrapped_agent=presenter,
        filter=MessageFilterConfig(
            per_source=[PerSourceFilter(source="analyst", position="last", count=1)]
        ),
    )

    # Build the flow
    builder = DiGraphBuilder()
    builder.add_node(researcher).add_node(filtered_analyst).add_node(filtered_presenter)
    builder.add_edge(researcher, filtered_analyst).add_edge(
        filtered_analyst, filtered_presenter
    )

    # Build and validate the graph
    graph = builder.build()

    # Create the flow
    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
    )

    # Run the workflow
    await Console(flow.run_stream(task="Summarize key facts about climate change."))
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
