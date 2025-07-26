import asyncio

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.completion_usage import CompletionUsage

USER_INPUTS = [
    "Why is the sky blue?",
    "What is the capital of France?",
]


async def main():
    from service import chat_completion_service

    # 1. Create the agent by specifying the service
    agent = ChatCompletionAgent(
        service=chat_completion_service,
        name="Assistant",
        instructions="Answer questions about the world in one sentence.",
    )
    thread: ChatHistoryAgentThread = None
    completion_usage = CompletionUsage()
    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        # 2. Invoke the agent for a response
        response = await agent.get_response(
            messages=user_input,
            thread=thread,
        )
        thread = response.thread
        if response.metadata.get("usage"):
            completion_usage += response.metadata["usage"]
        # 3. Print the response
        print(f"# {response.name}: {response}")

    print(
        f"\nTotal Completion Usage: {completion_usage.model_dump_json(indent=4)}"
    )

    await thread.delete() if thread else None


if __name__ == "__main__":
    asyncio.run(main())
