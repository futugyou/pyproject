from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import (
    GoogleAIEmbeddingPromptExecutionSettings,
    GoogleAITextEmbedding,
)
from openai import AsyncOpenAI
from app.core.config import (
    OPENAI_CHAT_MODEL_ID,
    OPENAI_API_KEY,
    OPENAI_URL,
    GOOGLE_API_KEY,
    GOOGLE_TEXT_EMBEDDING_MODEL_ID,
)


kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        ai_model_id=OPENAI_CHAT_MODEL_ID,
        service_id="default",
        async_client=AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_URL,
        ),
    )
)

kernel.add_service(
    GoogleAITextEmbedding(
        embedding_model_id=GOOGLE_TEXT_EMBEDDING_MODEL_ID,
        service_id="embedding",
        api_key=GOOGLE_API_KEY,
    )
)
