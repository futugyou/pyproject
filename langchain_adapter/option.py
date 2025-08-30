from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class LangChainOption(BaseSettings):
    lang_google_api_key: str
    lang_google_chat_model: str
    lang_google_embedding_model: str

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
