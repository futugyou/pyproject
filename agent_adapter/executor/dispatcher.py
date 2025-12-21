from typing_extensions import Never
from agent_framework import handler, WorkflowContext, Executor
from .checkpoint import CheckpointExecutor


class Dispatcher(CheckpointExecutor):
    """
    The sole purpose of this executor is to dispatch the input of the workflow to
    other executors.
    """

    @handler
    async def handle(self, numbers: list[int], ctx: WorkflowContext[list[int]]):
        if not numbers:
            raise RuntimeError("Input must be a valid list of integers.")
        self._messages = numbers
        await ctx.send_message(numbers)
