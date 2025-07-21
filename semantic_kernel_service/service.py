from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

service_id = "default"
load_dotenv()
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID"),
        service_id=service_id,
        # api_key=os.getenv("OPENAI_API_KEY"),
        async_client=AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_URL"),
        ),
    )
)
