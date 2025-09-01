import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


async def async_stream(config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    chain = llm | JsonOutputParser()
    async for text in chain.astream(
        'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'
    ):
        print(text, end="|", flush=True)


if __name__ == "__main__":
    asyncio.run(async_stream(LangChainOption()))
