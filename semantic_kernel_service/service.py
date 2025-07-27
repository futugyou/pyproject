from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIAudioToText,
)
from semantic_kernel.connectors.ai.google.google_ai import (
    GoogleAIEmbeddingPromptExecutionSettings,
    GoogleAITextEmbedding,
)
from openai import AsyncOpenAI
from mem0 import MemoryClient
import os
from dotenv import load_dotenv


load_dotenv()
kernel = Kernel()
chat_completion_service = OpenAIChatCompletion(
    ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID"),
    service_id="default",
    async_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_URL"),
    ),
)

kernel.add_service(chat_completion_service)

text_embedding_service = GoogleAITextEmbedding(
    embedding_model_id=os.getenv("GOOGLE_TEXT_EMBEDDING_MODEL_ID"),
    service_id="embedding",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
kernel.add_service(text_embedding_service)

memoryClient = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

audio_to_text_service = OpenAIAudioToText(
    ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID"),
    service_id="audio_to_text",
    async_client=AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_URL"),
    ),
)
