from typing_extensions import Never
from agent_framework import (
    BaseChatClient,
    ChatAgent,
    handler,
    WorkflowContext,
    Executor,
    ChatMessage,
)


class Writer(Executor):
    agent: ChatAgent

    def __init__(self, chat_client: BaseChatClient, id: str = "writer"):
        # Create a domain specific agent using your configured AzureOpenAIChatClient.
        self.agent = chat_client.create_agent(
            instructions=(
                "You are an excellent content writer. You create new content and edit contents based on the feedback."
            ),
            name="writer",
        )
        super().__init__(id=id)

    @handler
    async def handle(
        self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        response = await self.agent.run(messages)
        messages.extend(response.messages)
        await ctx.send_message(messages)
