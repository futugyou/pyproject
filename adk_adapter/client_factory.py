import os
from dotenv import load_dotenv

from google.adk.models.google_llm import Gemini, BaseLlm

load_dotenv()


def build_llm() -> BaseLlm:
    return Gemini(
        model=os.getenv("GOOGLE_CHAT_MODEL_ID"), api_key=os.getenv("GOOGLE_API_KEY")
    )
