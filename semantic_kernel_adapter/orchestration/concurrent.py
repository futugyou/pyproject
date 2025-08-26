import asyncio
from pydantic import BaseModel

from semantic_kernel.agents import Agent, ChatCompletionAgent, ConcurrentOrchestration
from semantic_kernel.agents.orchestration.tools import structured_outputs_transform
from semantic_kernel.agents.runtime import InProcessRuntime


class ArticleAnalysis(BaseModel):
    """A model to hold the analysis of an article."""

    themes: list[str]
    sentiments: list[str]
    entities: list[str]


def get_agents(chat_completion_service) -> list[Agent]:
    theme_agent = ChatCompletionAgent(
        name="ThemeAgent",
        instructions="You are an expert in identifying themes in articles. Given an article, identify the main themes.",
        service=chat_completion_service,
    )
    sentiment_agent = ChatCompletionAgent(
        name="SentimentAgent",
        instructions="You are an expert in sentiment analysis. Given an article, identify the sentiment.",
        service=chat_completion_service,
    )
    entity_agent = ChatCompletionAgent(
        name="EntityAgent",
        instructions="You are an expert in entity recognition. Given an article, extract the entities.",
        service=chat_completion_service,
    )

    return [theme_agent, sentiment_agent, entity_agent]


task = """
On a dark winter night, a ghost walks the ramparts of Elsinore Castle in Denmark. Discovered first by a pair of watchmen, then by the scholar Horatio, the ghost resembles the recently deceased King Hamlet, whose brother Claudius has inherited the throne and married the king’s widow, Queen Gertrude. When Horatio and the watchmen bring Prince Hamlet, the son of Gertrude and the dead king, to see the ghost, it speaks to him, declaring ominously that it is indeed his father’s spirit, and that he was murdered by none other than Claudius. Ordering Hamlet to seek revenge on the man who usurped his throne and married his wife, the ghost disappears with the dawn.
"""


async def main():
    from ..service import build_kernel_pipeline

    kernel = build_kernel_pipeline()
    chat_completion_service = kernel.get_service("default")

    agents = get_agents(chat_completion_service)
    concurrent_orchestration = ConcurrentOrchestration[str, ArticleAnalysis](
        members=agents,
        output_transform=structured_outputs_transform(
            ArticleAnalysis, chat_completion_service
        ),
    )

    runtime = InProcessRuntime()
    runtime.start()

    orchestration_result = await concurrent_orchestration.invoke(
        task=task,
        runtime=runtime,
    )

    value = await orchestration_result.get(timeout=20)
    if isinstance(value, ArticleAnalysis):
        print(value.model_dump_json(indent=2))
    else:
        print("Unexpected result type:", type(value))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
