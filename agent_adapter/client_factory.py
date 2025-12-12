import os
from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatClient

load_dotenv()


def build_client(type: str) -> OpenAIChatClient:
    model_id = os.getenv("GOOGLE_CHAT_MODEL_ID")
    base_url = os.getenv("GOOGLE_URL")
    api_key = os.getenv("GOOGLE_API_KEY")

    if type == "openai":
        model_id = os.getenv("OPENAI_CHAT_MODEL_ID")
        base_url = os.getenv("OPENAI_URL")
        api_key = os.getenv("OPENAI_API_KEY")

    return OpenAIChatClient(
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
    )
