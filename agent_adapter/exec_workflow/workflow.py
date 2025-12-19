from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
import random
from typing_extensions import Never
from agent_framework import Workflow, WorkflowBuilder, WorkflowOutputEvent


from agent_adapter.executor.aggregator import Aggregator
from agent_adapter.executor.dispatcher import Dispatcher
from agent_adapter.executor.sum import Sum
from agent_adapter.executor.average import Average


def get_workflow() -> Workflow:
    dispatcher = Dispatcher(id="dispatcher")
    average = Average(id="average")
    summation = Sum(id="summation")
    aggregator = Aggregator(id="aggregator")

    workflow = (
        WorkflowBuilder(name="exec_workflow")
        .set_start_executor(dispatcher)
        .add_fan_out_edges(dispatcher, [average, summation])
        .add_fan_in_edges([average, summation], aggregator)
        .build()
    )
    return workflow


workflow = get_workflow()


async def main():
    # Run the workflow and stream events
    output: list[int | float] | None = None
    async for event in workflow.run_stream([random.randint(1, 100) for _ in range(10)]):
        if isinstance(event, WorkflowOutputEvent):
            output = event.data

    if output is not None:
        print(output)


if __name__ == "__main__":
    asyncio.run(main())
