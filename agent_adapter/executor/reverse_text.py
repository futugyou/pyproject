from typing_extensions import Never
from agent_framework import WorkflowContext, executor


@executor(id="reverse_text_executor")
async def reverse_text(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """Reverse the input and yield the workflow output."""
    result = text[::-1]

    # Yield the final output for this workflow run
    await ctx.yield_output(result)
