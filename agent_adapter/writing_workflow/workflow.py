from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
import random
from typing_extensions import Never
from agent_framework import (
    AgentRunEvent,
    Workflow,
    WorkflowBuilder,
    WorkflowOutputEvent,
)


from agent_adapter.executor.aggregator import Aggregator
from agent_adapter.executor.dispatcher import Dispatcher
from agent_adapter.executor.sum import Sum
from agent_adapter.executor.average import Average
from agent_adapter import client_factory


def get_workflow() -> Workflow:
    client = client_factory.build_client("openai")

    writer_agent = client.create_agent(
        instructions=(
            "You are an excellent content writer. You create new content and edit contents based on the feedback."
        ),
        name="writer",
    )

    reviewer_agent = client.create_agent(
        instructions=(
            "You are an excellent content reviewer."
            "Provide actionable feedback to the writer about the provided content."
            "Provide the feedback in the most concise manner possible."
        ),
        name="reviewer",
    )

    workflow = (
        WorkflowBuilder(name="writing_workflow")
        .set_start_executor(writer_agent)
        .add_edge(writer_agent, reviewer_agent)
        .build()
    )
    return workflow


workflow = get_workflow()


async def main():
    # Run the workflow and stream events
    events = await workflow.run("Write an advertising slogan for Coca-Cola.")
    # Print agent run events and final outputs
    for event in events:
        if isinstance(event, AgentRunEvent):
            print(f"{event.executor_id}: {event.data}")

    print(f"{'=' * 60}\nWorkflow Outputs: {events.get_outputs()}")
    # Summarize the final run state (e.g., COMPLETED)
    print("Final state:", events.get_final_state())


if __name__ == "__main__":
    asyncio.run(main())
