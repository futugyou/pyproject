from typing_extensions import Never
from agent_framework import handler, WorkflowContext, Executor, ChatMessage, Role
from .checkpoint import CheckpointExecutor


class Dispatcher(CheckpointExecutor):
    """
    The sole purpose of this executor is to dispatch the input of the workflow to
    other executors.
    """

    @handler
    async def dispatcher(
        self, messages: list[ChatMessage], ctx: WorkflowContext[list[int]]
    ):
        for i, message in enumerate(messages):
            if message.role == Role.USER:
                numbers = [int(x) for x in message.text.split(",")]
            break
        self._messages = numbers
        await ctx.send_message(numbers)
