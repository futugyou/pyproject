from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


def get_full_chain(config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    chain = (
        PromptTemplate.from_template(
            """Given the user question below, classify it as either being about `LangChain`, `Gemini`, or `Other`.

            Do not respond with more than one word.

            <question>
            {question}
            </question>

            Classification:"""
        )
        | model
        | StrOutputParser()
    )

    # sub chain
    langchain_chain = (
        PromptTemplate.from_template(
            """You are an expert in langchain. \
            Always answer questions starting with "As Harrison Chase told me". \
            Respond to the following question:

            Question: {question}
            Answer:"""
        )
        | model
    )
    gemini_chain = (
        PromptTemplate.from_template(
            """You are an expert in gemini. \
            Always answer questions starting with "As Dario Amodei told me". \
            Respond to the following question:

            Question: {question}
            Answer:"""
        )
        | model
    )
    general_chain = (
        PromptTemplate.from_template(
            """Respond to the following question:

            Question: {question}
            Answer:"""
        )
        | model
    )

    branch = RunnableBranch(
        (lambda x: "gemini" in x["topic"].lower(), gemini_chain),
        (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
        general_chain,
    )

    full_chain = {"topic": chain, "question": lambda x: x["question"]} | branch

    return full_chain


if __name__ == "__main__":
    chain = get_full_chain(LangChainOption())
    result = chain.invoke({"question": "how do I use Gemini?"})
    print(result.content)
    print()

    result = chain.invoke({"question": "how do I use LangChain?"})
    print(result.content)
    print()

    result = chain.invoke({"question": "whats 2 + 2"})
    print(result.content)
