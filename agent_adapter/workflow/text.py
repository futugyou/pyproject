from typing import cast, Any, List
from agent_adapter.executor.reverse_text import reverse_text
from agent_adapter.executor.upper_text import UpperCase
from agent_framework import Workflow, WorkflowBuilder, WorkflowOutputEvent, ChatMessage, Role
from typing_extensions import Never
import asyncio
from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


def get_text_workflow() -> Workflow:
    upper_case = UpperCase(id="upper_case_executor")
    workflow = (
        WorkflowBuilder(name="text_workflow")
        .add_edge(upper_case, reverse_text)
        .set_start_executor(upper_case)
        .build()
    )
    return workflow


async def main():
    workflow = get_text_workflow()
    msg = [ChatMessage(role="user", text="hello world")]
    async for event in workflow.run_stream(msg):
        print(f"Event: {event}")
        if isinstance(event, WorkflowOutputEvent):
            raw_data = event.data if isinstance(
                event.data, list) else [event.data]
            messages = cast(List[ChatMessage], raw_data)
            for message in messages:
                print(f"Workflow completed: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
