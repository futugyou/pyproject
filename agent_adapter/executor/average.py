from typing_extensions import Never
from agent_framework import handler, WorkflowContext, Executor
from .checkpoint import CheckpointExecutor


class Average(CheckpointExecutor):
    """Calculate the average of a list of integers."""

    @handler
    async def handle(self, numbers: list[int], ctx: WorkflowContext[float]):
        self._messages = numbers
        average: float = sum(numbers) / len(numbers)
        await ctx.send_message(average)
