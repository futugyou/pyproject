from typing_extensions import Never
from agent_framework import handler, WorkflowContext, Executor
from .checkpoint import CheckpointExecutor


class Sum(CheckpointExecutor):
    """Calculate the sum of a list of integers."""

    @handler
    async def handle(self, numbers: list[int], ctx: WorkflowContext[int]):
        self._messages = numbers
        total: int = sum(numbers)
        await ctx.send_message(total)
