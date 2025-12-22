from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
import random
from typing_extensions import Never
from agent_framework import (
    ChatClientProtocol,
    ChatMessage,
    AgentRunEvent,
    Workflow,
    WorkflowBuilder,
    WorkflowOutputEvent,
)


from agent_adapter.executor.writer import Writer
from agent_adapter.executor.reviewer import Reviewer
from agent_adapter import client_factory


def get_writing_workflow(client: ChatClientProtocol) -> Workflow:
    writer = Writer(client)
    reviewer = Reviewer(client)

    workflow = (
        WorkflowBuilder(name="writing_workflow")
        .set_start_executor(writer)
        .add_edge(writer, reviewer)
        .build()
    )
    return workflow


async def main():
    client = client_factory.build_client("openai")
    workflow = get_writing_workflow(client)
    events = await workflow.run(
        ChatMessage(role="user", text="Write an advertising slogan for Coca-Cola.")
    )
    # Print agent run events and final outputs
    for event in events:
        if isinstance(event, AgentRunEvent):
            print(f"{event.executor_id}: {event.data}")

    print(f"{'=' * 60}\nWorkflow Outputs: {events.get_outputs()}")
    # Summarize the final run state (e.g., COMPLETED)
    print("Final state:", events.get_final_state())


if __name__ == "__main__":
    asyncio.run(main())
