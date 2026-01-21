from typing_extensions import Never
from agent_framework import WorkflowContext, executor, ChatMessage, Role


@executor(id="reverse_text_executor")
async def reverse_text(
    messages: list[ChatMessage], ctx: WorkflowContext[Never, list[ChatMessage]]
) -> None:
    """Reverse the input and yield the workflow output."""
    for i, message in enumerate(messages):
        if message.role == Role.USER:
            messages[i] = ChatMessage(
                role=message.role,
                text=message.text[::-1],
            )

    # Yield the final output for this workflow run
    await ctx.yield_output(messages)
