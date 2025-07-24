import asyncio
from pydantic import BaseModel

from semantic_kernel.agents import Agent, ChatCompletionAgent, SequentialOrchestration
from semantic_kernel.agents.orchestration.tools import structured_outputs_transform
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent


def get_agents() -> list[Agent]:
    from service import chat_completion_service

    concept_extractor_agent = ChatCompletionAgent(
        name="ConceptExtractorAgent",
        instructions=(
            "You are a marketing analyst. Given a product description, identify:\n"
            "- Key features\n"
            "- Target audience\n"
            "- Unique selling points\n\n"
        ),
        service=chat_completion_service,
    )
    writer_agent = ChatCompletionAgent(
        name="WriterAgent",
        instructions=(
            "You are a marketing copywriter. Given a block of text describing features, audience, and USPs, "
            "compose a compelling marketing copy (like a newsletter section) that highlights these points. "
            "Output should be short (around 150 words), output just the copy as a single text block."
        ),
        service=chat_completion_service,
    )
    format_proof_agent = ChatCompletionAgent(
        name="FormatProofAgent",
        instructions=(
            "You are an editor. Given the draft copy, correct grammar, improve clarity, ensure consistent tone, "
            "give format and make it polished. Output the final improved copy as a single text block."
        ),
        service=chat_completion_service,
    )

    # The order of the agents in the list will be the order in which they are executed
    return [concept_extractor_agent, writer_agent, format_proof_agent]


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    print(f"# {message.name}\n{message.content}")


async def main():
    from service import chat_completion_service

    agents = get_agents()
    sequential_orchestration = SequentialOrchestration(
        members=agents,
        agent_response_callback=agent_response_callback,
    )

    runtime = InProcessRuntime()
    runtime.start()
    orchestration_result = await sequential_orchestration.invoke(
        task="An eco-friendly stainless steel water bottle that keeps drinks cold for 24 hours",
        runtime=runtime,
    )

    value = await orchestration_result.get(timeout=20)
    print(f"***** Final Result *****\n{value}")

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
