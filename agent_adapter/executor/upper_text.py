from agent_framework import handler, WorkflowContext, Executor


class UpperCase(Executor):
    def __init__(self, id: str):
        super().__init__(id=id)

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        result = text.upper()

        # Send the result to the next executor in the workflow.
        await ctx.send_message(result)
