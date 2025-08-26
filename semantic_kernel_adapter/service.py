from pydantic_settings import BaseSettings
from pydantic import ConfigDict

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIAudioToText,
)
from semantic_kernel.connectors.ai.google.google_ai import (
    GoogleAITextEmbedding,
)
from openai import AsyncOpenAI
from mem0 import MemoryClient


class SemanticKernelOption(BaseSettings):
    chat_model_id: str
    chat_api_key: str
    chat_base_url: str

    embedding_model_id: str
    embedding_api_key: str
    embedding_base_url: str

    audio_model_id: str
    audio_api_key: str
    audio_base_url: str

    mem0_api_key: str

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_kernel() -> Kernel:
    return Kernel()


def get_chat_completion_service(
    option: SemanticKernelOption, service_id: str = "default"
) -> OpenAIChatCompletion:
    return OpenAIChatCompletion(
        ai_model_id=option.chat_model_id,
        service_id=service_id,
        async_client=AsyncOpenAI(
            api_key=option.chat_api_key,
            base_url=option.chat_base_url,
        ),
    )


def get_text_embedding_service(
    option: SemanticKernelOption, service_id: str = "embedding"
) -> GoogleAITextEmbedding:
    return GoogleAITextEmbedding(
        embedding_model_id=option.embedding_model_id,
        service_id=service_id,
        api_key=option.embedding_api_key,
    )


def get_audio_to_text_service(
    option: SemanticKernelOption, service_id: str = "audio_to_text"
) -> OpenAIAudioToText:
    return OpenAIChatCompletion(
        ai_model_id=option.chat_model_id,
        service_id=service_id,
        async_client=AsyncOpenAI(
            api_key=option.chat_api_key,
            base_url=option.chat_base_url,
        ),
    )


def build_kernel_pipeline(
    option: SemanticKernelOption | None = None,
    kernel: Kernel | None = None,
    chat_service_id: str = "default",
    embedding_service_id: str = "embedding",
    audio_service_id: str = "audio_to_text",
) -> Kernel:
    if option is None:
        option = SemanticKernelOption()
    if kernel is None:
        kernel = get_kernel()
    kernel.add_service(get_chat_completion_service(option, chat_service_id))
    kernel.add_service(get_text_embedding_service(option, embedding_service_id))
    kernel.add_service(get_audio_to_text_service(option, audio_service_id))
    return kernel


def get_memory_client(option: SemanticKernelOption | None = None) -> MemoryClient:
    if option is None:
        option = SemanticKernelOption()
    return MemoryClient(api_key=option.mem0_api_key)
