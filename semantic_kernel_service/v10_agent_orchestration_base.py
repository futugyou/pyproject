import asyncio

from semantic_kernel.agents import Agent, ChatCompletionAgent, ConcurrentOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime


def get_agents() -> list[Agent]:
    from service import chat_completion_service

    physics_agent = ChatCompletionAgent(
        name="PhysicsExpert",
        instructions="You are an expert in physics. You answer questions from a physics perspective.",
        service=chat_completion_service,
    )
    chemistry_agent = ChatCompletionAgent(
        name="ChemistryExpert",
        instructions="You are an expert in chemistry. You answer questions from a chemistry perspective.",
        service=chat_completion_service,
    )

    return [physics_agent, chemistry_agent]

async def main():
    """Main function to run the agents."""
    # 1. Create a concurrent orchestration with multiple agents
    agents = get_agents()
    concurrent_orchestration = ConcurrentOrchestration(members=agents)

    # 2. Create a runtime and start it
    runtime = InProcessRuntime()
    runtime.start()

    # 3. Invoke the orchestration with a task and the runtime
    orchestration_result = await concurrent_orchestration.invoke(
        task="What is temperature?",
        runtime=runtime,
    )

    # 4. Wait for the results
    # Note: the order of the results is not guaranteed to be the same
    # as the order of the agents in the orchestration.
    value = await orchestration_result.get(timeout=20)
    for item in value:
        print(f"# {item.name}: {item.content}")

    # 5. Stop the runtime after the invocation is complete
    await runtime.stop_when_idle()



if __name__ == "__main__":
    asyncio.run(main())
