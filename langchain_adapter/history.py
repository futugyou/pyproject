from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
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


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


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
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="user id",
                description="user id",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="conversation id",
                description="conversation id",
                default="",
                is_shared=True,
            ),
        ],
    )


if __name__ == "__main__":
    runnable = get_runnable(LangChainOption())
    result = runnable.invoke(
        {"ability": "Mathematics", "input": "What does the cosine function mean?"},
        config={"configurable": {"user_id": "abc123", "conversation_id": "1"}},
    )
    print(1, result.content)

    result = runnable.invoke(
        {"ability": "Mathematics", "input": "What"},
        config={"configurable": {"user_id": "abc123", "conversation_id": "1"}},
    )
    print(2, result.content)

    result = runnable.invoke(
        {"ability": "Mathematics", "input": "What"},
        config={"configurable": {"user_id": "abc9999", "conversation_id": "2"}},
    )
    print(3, result.content)
