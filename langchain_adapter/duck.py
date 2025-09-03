from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


prompt = ChatPromptTemplate.from_template("""Converts the following user input into a search engine query:

{input}""")


def use_duckducksreach(input_text: str, config: LangChainOption):
    search = DuckDuckGoSearchRun()
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    chain = prompt | model | StrOutputParser() | search

    return chain.invoke({"input": input_text})


if __name__ == "__main__":
    # duckduckgo_search.exceptions.DuckDuckGoSearchException: 
    # duckduck return 302....
    result = use_duckducksreach(
        "I'm trying to figure out what games are playing tonight.", LangChainOption()
    )
    print(result)
