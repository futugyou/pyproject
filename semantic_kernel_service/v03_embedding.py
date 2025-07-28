from semantic_kernel import Kernel
import asyncio
import os
from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid4
from semantic_kernel.connectors.in_memory import InMemoryStore
from semantic_kernel.connectors.mongodb import MongoDBAtlasStore
from semantic_kernel.data.vector import (
    VectorStore,
    VectorStoreField,
    vectorstoremodel,
    VectorSearchProtocol,
    VectorStoreCollection,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelFunction
from semantic_kernel.prompt_template import PromptTemplateConfig


@vectorstoremodel
@dataclass
class ChatHistoryModel:
    session_id: Annotated[str, VectorStoreField("key")]
    user_id: Annotated[str, VectorStoreField("data", is_indexed=True)]
    messages: Annotated[list[dict[str, str]], VectorStoreField("data", is_indexed=True)]


class ChatHistoryInVectorStore(ChatHistory):
    session_id: str
    user_id: str
    store: VectorStore
    collection: VectorStoreCollection[str, ChatHistoryModel] | None = None

    async def create_collection(self, collection_name: str) -> None:
        """Create a collection with the inbuild data model using the vector store.

        First create the collection, then call this method to create the collection itself.
        """
        self.collection = self.store.get_collection(
            collection_name=collection_name,
            record_type=ChatHistoryModel,
        )

        exists: bool = await self.collection.collection_exists()
        if not exists:
            await self.collection.ensure_collection_exists()

    async def store_messages(self) -> None:
        """Store the chat history in the VectorStore.

        Note that we use model_dump to convert the chat message content into a serializable format.
        """
        if self.collection:
            await self.collection.upsert(
                ChatHistoryModel(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    messages=[msg.model_dump() for msg in self.messages],
                )
            )

    async def read_messages(self) -> None:
        """Read the chat history from the VectorStore.

        Note that we use the model_validate method to convert the serializable format back into a ChatMessageContent.
        """
        if self.collection:
            record = await self.collection.get(self.session_id)
            if record:
                for message in record.messages:
                    self.messages.append(ChatMessageContent.model_validate(message))


@vectorstoremodel(collection_name="simple-model")
@dataclass
class SimpleModel:
    id: Annotated[str, VectorStoreField("key")] = field(
        default_factory=lambda: str(uuid4())
    )
    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    embedding: Annotated[
        list[float] | str | None, VectorStoreField("vector", dimensions=3072)
    ] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = self.text


records = [
    SimpleModel(text="Your budget for 2024 is $100,000"),
    SimpleModel(text="Your savings from 2023 are $50,000"),
    SimpleModel(text="Your investments are $80,000"),
]


async def init_embedding(
    kernel: Kernel,
) -> tuple[VectorSearchProtocol, ChatHistoryInVectorStore]:
    embedding_gen = kernel.get_service(service_id="embedding")
    use_mongo = os.environ.get("USE_MONGODB_EMBEDDING", "false")
    vectorStore: VectorStore = None
    if use_mongo == "true":
        vectorStore = MongoDBAtlasStore(
            connection_string=os.environ["MONGODB_CONNECTION_STRING"],
            database_name=os.environ["MONGODB_DATABASE_NAME"],
        )
    else:
        vectorStore = InMemoryStore()

    history = ChatHistoryInVectorStore(
        store=vectorStore, session_id="session_id", user_id="user"
    )
    await history.create_collection(collection_name="chat_history")

    collection = vectorStore.get_collection(
        record_type=SimpleModel,
        collection_name="simple-model",
        embedding_generator=embedding_gen,
    )
    collection.index_name = "simple_model_index"

    exists: bool = await collection.collection_exists()
    if not exists:
        await collection.ensure_collection_exists()

    await collection.upsert(records)
    return collection, history


async def search_memory_examples(
    kernel: Kernel, collection: VectorSearchProtocol, questions: list[str]
) -> None:
    for question in questions:
        print(f"Question: {question}")
        # https://github.com/microsoft/semantic-kernel/issues/12812
        results = await collection.search(question, top=1)
        async for result in results.results:
            print(f"Answer: {result.record.text}")
            print(f"Score: {result.score}\n")


async def setup_chat_with_memory(
    kernel: Kernel,
    service_id: str,
) -> KernelFunction:
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if
    it does not have an answer.

    Information about me, from previous conversations:
    - {{recall 'budget by year'}} What is my budget for 2024?
    - {{recall 'savings from previous year'}} What are my savings from 2023?
    - {{recall 'investments'}} What are my investments?

    {{$request}}
    """.strip()

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        execution_settings={
            service_id: kernel.get_service(
                service_id
            ).get_prompt_execution_settings_class()(service_id=service_id)
        },
    )

    return kernel.add_function(
        function_name="chat_with_memory",
        plugin_name="chat",
        prompt_template_config=prompt_template_config,
    )


async def setup_recall_function(
    kernel: Kernel,
    collection: VectorSearchProtocol,
) -> KernelFunction:
    function = kernel.add_function(
        plugin_name="memory",
        function=collection.create_search_function(
            function_name="recall",
            description="Searches the memory for relevant information based on the input query.",
        ),
    )

    return function


async def chat(
    user_input: str,
    chat_func: KernelFunction,
    kernel: Kernel,
    history: ChatHistoryInVectorStore,
):
    await history.read_messages()
    if len(history.messages) == 0:
        history.add_system_message(
            "You are a ChatBot can have a conversation with you about any topic."
        )

    print(f"User: {user_input}")
    history.add_user_message(user_input)
    answer = await kernel.invoke(chat_func, request=user_input)
    print(f"ChatBot:> {answer}")
    if result:
        history.add_message(answer)
    await history.store_messages()


if __name__ == "__main__":

    async def main():
        from service import kernel

        collection, history = await init_embedding(kernel)
        await search_memory_examples(
            kernel,
            collection,
            questions=[
                "What is my budget for 2024?",
                "What are my savings from 2023?",
                "What are my investments?",
            ],
        )

        await setup_recall_function(kernel, collection)
        chat_func = await setup_chat_with_memory(kernel, "default")
        await chat("What is my budget for 2024?", chat_func, kernel, history)
        await chat("What are my savings from 2023?", chat_func, kernel, history)

    asyncio.run(main())
