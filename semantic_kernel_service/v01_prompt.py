import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid4

from semantic_kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.in_memory import InMemoryStore
from semantic_kernel.data.vector import (
    VectorStore,
    VectorStoreCollection,
    VectorStoreField,
    vectorstoremodel,
)
from semantic_kernel.filters import (
    FilterTypes,
    FunctionInvocationContext,
    PromptRenderContext,
)
from semantic_kernel.functions import FunctionResult, KernelFunction

COLLECTION_NAME = "llm_responses"
RECORD_ID_KEY = "cache_record_id"


# Define a simple data model to store, the prompt and the result
# we annotate the prompt field as the vector field, the prompt itself will not be stored.
# and if you use `include_vectors` in the search, it will return the vector, but not the prompt.
@vectorstoremodel(collection_name=COLLECTION_NAME)
@dataclass
class CacheRecord:
    result: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    prompt: Annotated[str | None, VectorStoreField("vector", dimensions=1536)] = None
    id: Annotated[str, VectorStoreField("key")] = field(
        default_factory=lambda: str(uuid4())
    )


class PromptCacheFilter:
    """A filter to cache the results of the prompt rendering and function invocation."""

    def __init__(
        self,
        vector_store: VectorStore,
        score_threshold: float = 0.2,
    ):
        if vector_store.embedding_generator is None:
            raise ValueError("The vector store must have an embedding generator.")
        self.vector_store = vector_store
        self.collection: VectorStoreCollection[str, CacheRecord] = (
            vector_store.get_collection(record_type=CacheRecord)
        )
        self.score_threshold = score_threshold

    async def on_prompt_render(
        self,
        context: PromptRenderContext,
        next: Callable[[PromptRenderContext], Awaitable[None]],
    ):
        """Filter to cache the rendered prompt and the result of the function.

        It uses the score threshold to determine if the result should be cached.
        The direction of the comparison is based on the default distance metric for
        the in memory vector store, which is cosine distance, so the closer to 0 the
        closer the match.
        """
        print("on_prompt_render")
        await next(context)
        await self.collection.ensure_collection_exists()
        results = await self.collection.search(
            context.rendered_prompt, vector_property_name="prompt", top=1
        )
        async for result in results.results:
            if result.score is not None and result.score < self.score_threshold:
                print("Cache hit, score:", result.score)
                context.function_result = FunctionResult(
                    function=context.function.metadata,
                    value=result.record.result,
                    rendered_prompt=context.rendered_prompt,
                    metadata={RECORD_ID_KEY: result.record.id},
                )

    async def on_function_invocation(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ):
        """Filter to store the result in the cache if it is new."""
        print("on_function_invocation")
        await next(context)
        result = context.result
        if result and result.rendered_prompt and RECORD_ID_KEY not in result.metadata:
            print("Cache miss")
            cache_record = CacheRecord(
                prompt=result.rendered_prompt, result=str(result)
            )
            await self.collection.ensure_collection_exists()
            await self.collection.upsert(cache_record)


prompt = """{{$input}}
Summarize the content above.
"""
prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="summarize",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
)

input_text = """
Demo (ancient Greek poet)
From Wikipedia, the free encyclopedia
Demo or Damo (Greek: Δεμώ, Δαμώ; fl. c. AD 200) was a Greek woman of the Roman period, known for a single epigram, engraved upon the Colossus of Memnon, which bears her name. She speaks of herself therein as a lyric poetess dedicated to the Muses, but nothing is known of her life.[1]
Identity
Demo was evidently Greek, as her name, a traditional epithet of Demeter, signifies. The name was relatively common in the Hellenistic world, in Egypt and elsewhere, and she cannot be further identified. The date of her visit to the Colossus of Memnon cannot be established with certainty, but internal evidence on the left leg suggests her poem was inscribed there at some point in or after AD 196.[2]
Epigram
There are a number of graffiti inscriptions on the Colossus of Memnon. Following three epigrams by Julia Balbilla, a fourth epigram, in elegiac couplets, entitled and presumably authored by 'Demo' or 'Damo' (the Greek inscription is difficult to read), is a dedication to the Muses.[2] The poem is traditionally published with the works of Balbilla, though the internal evidence suggests a different author.[1]
In the poem, Demo explains that Memnon has shown her special respect. In return, Demo offers the gift for poetry, as a gift to the hero. At the end of this epigram, she addresses Memnon, highlighting his divine status by recalling his strength and holiness.[2]
Demo, like Julia Balbilla, writes in the artificial and poetic Aeolic dialect. The language indicates she was knowledgeable in Homeric poetry—'bearing a pleasant gift', for example, alludes to the use of that phrase throughout the Iliad and Odyssey.[a][2]
"""


async def generate_summarize(
    kernel: Kernel, summarize: KernelFunction, input_text: str
) -> str:
    summary = await kernel.invoke(summarize, input=input_text)
    return str(summary)


if __name__ == "__main__":

    async def main():
        from service import kernel, text_embedding_service

        # create the in-memory vector store
        vector_store = InMemoryStore(embedding_generator=text_embedding_service)
        # create the cache filter and add the filters to the kernel
        cache = PromptCacheFilter(vector_store=vector_store)
        kernel.add_filter(FilterTypes.PROMPT_RENDERING, cache.on_prompt_render)
        kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, cache.on_function_invocation)

        summarize = kernel.add_function(
            function_name="summarizeFunc",
            plugin_name="summarizePlugin",
            prompt_template_config=prompt_template_config,
        )
        summary = await generate_summarize(kernel, summarize, input_text)
        print(summary)
        summary = await generate_summarize(kernel, summarize, input_text)
        print(summary)

    asyncio.run(main())
