import asyncio

from semantic_kernel.agents import ChatCompletionAgent

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

    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        # 2. Invoke the agent for a response
        response = await agent.get_response(
            messages=user_input,
        )
        # 3. Print the response
        print(f"# {response.name}: {response}")


if __name__ == "__main__":
    asyncio.run(main())
