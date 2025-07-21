from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel import Kernel
from openai import AsyncOpenAI
from app.core.config import OPENAI_CHAT_MODEL_ID, OPENAI_API_KEY, OPENAI_URL

kernel = Kernel()
service_id = "default"
kernel.add_service(
    OpenAIChatCompletion(
        ai_model_id=OPENAI_CHAT_MODEL_ID,
        service_id=service_id,
        async_client=AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_URL,
        ),
    )
)
