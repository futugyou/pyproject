from agent_framework import handler, WorkflowContext, Executor, ChatMessage, Role


class UpperCase(Executor):
    def __init__(self, id: str):
        super().__init__(id=id)

    @handler
    async def to_upper_case(
        self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        for i, message in enumerate(messages):
            if message.role == Role.USER:
                messages[i] = ChatMessage(
                    role=message.role,
                    text=message.text.upper(),
                )
        await ctx.send_message(messages)
