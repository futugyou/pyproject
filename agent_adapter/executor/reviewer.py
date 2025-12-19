from typing_extensions import Never
from agent_framework import (
    BaseChatClient,
    ChatAgent,
    handler,
    WorkflowContext,
    Executor,
    ChatMessage,
)


class Reviewer(Executor):
    agent: ChatAgent

    def __init__(self, chat_client: BaseChatClient, id: str = "reviewer"):
        # Create a domain specific agent using your configured AzureOpenAIChatClient.
        self.agent = chat_client.create_agent(
            instructions=(
                "You are an excellent content reviewer."
                "Provide actionable feedback to the writer about the provided content."
                "Provide the feedback in the most concise manner possible."
            ),
            name="reviewer",
        )
        super().__init__(id=id)

    @handler
    async def handle(
        self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]
    ) -> None:
        response = await self.agent.run(messages)
        await ctx.yield_output(response.text)
