from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from typing_extensions import Never
from agent_framework import Workflow, WorkflowBuilder, WorkflowOutputEvent


from agent_adapter.executor.upper_text import UpperCase
from agent_adapter.executor.reverse_text import reverse_text


def get_workflow() -> Workflow:
    upper_case = UpperCase(id="upper_case_executor")
    workflow = (
        WorkflowBuilder(name="text_workflow")
        .add_edge(upper_case, reverse_text)
        .set_start_executor(upper_case)
        .build()
    )
    return workflow


workflow = get_workflow()


async def main():
    # Run the workflow and stream events
    async for event in workflow.run_stream("hello world"):
        print(f"Event: {event}")
        if isinstance(event, WorkflowOutputEvent):
            print(f"Workflow completed with result: {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
