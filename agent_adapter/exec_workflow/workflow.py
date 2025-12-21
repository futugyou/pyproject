from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
from dotenv import load_dotenv

load_dotenv()

import asyncio
import random
from typing_extensions import Never
from agent_framework import (
    Workflow,
    WorkflowBuilder,
    WorkflowOutputEvent,
    CheckpointStorage,
)


from agent_adapter.executor.aggregator import Aggregator
from agent_adapter.executor.dispatcher import Dispatcher
from agent_adapter.executor.sum import Sum
from agent_adapter.executor.average import Average
from agent_adapter.checkpoint.postgres import PostgresCheckpointStorage


def get_workflow(checkpointStorage: CheckpointStorage | None = None) -> Workflow:
    dispatcher = Dispatcher(id="dispatcher")
    average = Average(id="average")
    summation = Sum(id="summation")
    aggregator = Aggregator(id="aggregator")

    builder = (
        WorkflowBuilder(name="exec_workflow")
        .set_start_executor(dispatcher)
        .add_fan_out_edges(dispatcher, [average, summation])
        .add_fan_in_edges([average, summation], aggregator)
    )

    if checkpointStorage is not None:
        builder = builder.with_checkpointing(checkpointStorage)

    workflow = builder.build()
    return workflow


workflow = get_workflow(PostgresCheckpointStorage(os.getenv("POSTGRES_URI")))


async def main():
    # Run the workflow and stream events
    output: list[int | float] | None = None
    async for event in workflow.run_stream([random.randint(1, 100) for _ in range(10)]):
        if isinstance(event, WorkflowOutputEvent):
            output = event.data

    if output is not None:
        print(output)


async def run_checkpoint():
    checkpoint_storage = PostgresCheckpointStorage(os.getenv("POSTGRES_URI"))
    workflow = get_workflow(checkpoint_storage)
    output: list[int | float] | None = None
    async for event in workflow.run_stream([random.randint(1, 100) for _ in range(10)]):
        if isinstance(event, WorkflowOutputEvent):
            output = event.data

    if output is not None:
        print(output)

    checkpoints = await checkpoint_storage.list_checkpoints(workflow.id)
    print(f"checkpoints count: {len(checkpoints)}")

    saved_checkpoint = checkpoints[1]
    async for event in workflow.run_stream(
        checkpoint_id=saved_checkpoint.checkpoint_id,
        checkpoint_storage=checkpoint_storage,
    ):
        if isinstance(event, WorkflowOutputEvent):
            output = event.data

    if output is not None:
        print(output)


if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(run_checkpoint())
