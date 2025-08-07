from autogen_core.models import ModelFamily
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType

from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)

import tempfile
import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

load_dotenv()


async def get_weather(city: str, units: str = "imperial") -> str:
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."
    else:
        return f"Sorry, I don't know the weather in {city}."


async def run() -> None:
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_CHAT_MODEL_ID"),
        base_url=os.getenv("OPENAI_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.ANY,
            "structured_output": True,
            "multiple_system_messages": True,
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_user_memory = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="preferences",
                persistence_path=tmpdir,  # Use the temp directory here
                k=2,  # Return top k results
                score_threshold=0.4,  # Minimum similarity score
                # NEED `pip install sentence_transformers`
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                    model_name="all-MiniLM-L6-v2"  # Use default model for testing
                ),
            )
        )
        # Add user preferences to memory
        await chroma_user_memory.add(
            MemoryContent(
                content="The weather should be in metric units",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        )

        await chroma_user_memory.add(
            MemoryContent(
                content="Meal recipe must be vegan",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "dietary"},
            )
        )

        # Create assistant agent with ChromaDB memory
        assistant_agent = AssistantAgent(
            name="assistant_agent",
            model_client=model_client,
            tools=[get_weather],
            memory=[chroma_user_memory],
        )

        stream = assistant_agent.run_stream(task="What is the weather in New York?")
        await Console(stream)

        await model_client.close()
        await chroma_user_memory.close()


if __name__ == "__main__":
    asyncio.run(run())
