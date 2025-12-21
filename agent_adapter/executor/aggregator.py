from typing_extensions import Never
from agent_framework import handler, WorkflowContext, Executor
from .checkpoint import CheckpointExecutor


class Aggregator(CheckpointExecutor):
    """Aggregate the results from the different tasks and yield the final output."""

    @handler
    async def handle(
        self, results: list[int | float], ctx: WorkflowContext[Never, list[int | float]]
    ):
        """Receive the results from the source executors.

        The framework will automatically collect messages from the source executors
        and deliver them as a list.

        Args:
            results (list[int | float]): execution results from upstream executors.
                The type annotation must be a list of union types that the upstream
                executors will produce.
            ctx (WorkflowContext[Never, list[int | float]]): A workflow context that can yield the final output.
        """
        await ctx.yield_output(results)
