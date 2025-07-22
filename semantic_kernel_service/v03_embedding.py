from semantic_kernel import Kernel
import asyncio
from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid4
from semantic_kernel.connectors.in_memory import InMemoryStore
from semantic_kernel.data.vector import (
    VectorStoreField,
    vectorstoremodel,
    VectorSearchProtocol,
)

@vectorstoremodel(collection_name="simple-model")
@dataclass
class SimpleModel:
    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    id: Annotated[str, VectorStoreField("key")] = field(
        default_factory=lambda: str(uuid4())
    )
    embedding: Annotated[
        list[float] | str | None, VectorStoreField("vector", dimensions=1536)
    ] = None
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = self.text


records = [
    SimpleModel(text="Your budget for 2024 is $100,000"),
    SimpleModel(text="Your savings from 2023 are $50,000"),
    SimpleModel(text="Your investments are $80,000"),
]


async def init_embedding(kernel: Kernel) -> VectorSearchProtocol:
    in_memory_store = InMemoryStore()
    embedding_gen = kernel.get_service(service_id="embedding")
    
    collection = in_memory_store.get_collection(record_type=SimpleModel)
    await collection.ensure_collection_exists()
    collection.embedding_generator = embedding_gen
    await collection.upsert(records)
    return collection



async def search_memory_examples(
    kernel: Kernel, collection: VectorSearchProtocol, questions: list[str]
) -> None:
    for question in questions:
        print(f"Question: {question}")
        results = await collection.search(question, top=1)
        async for result in results.results:
            print(f"Answer: {result.record.text}")
            print(f"Score: {result.score}\n")


if __name__ == "__main__":

    async def main():
        from service import kernel

        collection = await init_embedding(kernel)
        await search_memory_examples(
            kernel,
            collection,
            questions=[
                "What is my budget for 2024?",
                "What are my savings from 2023?",
                "What are my investments?",
            ],
        )

    asyncio.run(main())
