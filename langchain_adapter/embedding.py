from langchain_google_genai import GoogleGenerativeAIEmbeddings

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


def generate_embedding(input_texts: List[str], config: LangChainOption):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )
    return embeddings_model.embed_documents(input_texts)


if __name__ == "__main__":
    result = generate_embedding(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!",
        ],
        LangChainOption(),
    )
    print(result)
