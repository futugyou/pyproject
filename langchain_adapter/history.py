from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


store = {}


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant who is good at {ability}. Please answer in 20 words or less.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_runnable(config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    runnable = prompt | model

    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


if __name__ == "__main__":
    runnable = get_runnable(LangChainOption())
    result = runnable.invoke(
        {"ability": "Mathematics", "input": "What does the cosine function mean?"},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(1, result.content)

    result = runnable.invoke(
        {"ability": "Mathematics", "input": "What"},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(2, result.content)

    result = runnable.invoke(
        {"ability": "Mathematics", "input": "What"},
        config={"configurable": {"session_id": "abc9999"}},
    )
    print(3, result.content)
