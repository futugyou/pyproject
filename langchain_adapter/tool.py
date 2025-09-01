import asyncio

from operator import itemgetter
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]


def calculate(query: str, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    llm_with_tools = llm.bind_tools(tools)
    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    return llm_with_tools.invoke(messages)


function = {
    "name": "solver",
    "description": "Formulates and solves an equation",
    "parameters": {
        "type": "object",
        "properties": {
            "equation": {
                "type": "string",
                "description": "The algebraic expression of the equation",
            },
            "solution": {
                "type": "string",
                "description": "The solution to the equation",
            },
        },
        "required": ["equation", "solution"],
    },
}

tools2 = [function]


async def bind_tools(config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    model = model.bind(tools=tools2)
    # TypeError: GenerativeServiceAsyncClient.generate_content() got an unexpected keyword argument 'function_call'
    # model = model.bind(function_call={"name": "solver"}, functions=[function])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write out the following equation using algebraic symbols then solve it.",
            ),
            ("human", "{equation_statement}"),
        ]
    )

    runnable = {"equation_statement": RunnablePassthrough()} | prompt | model
    result = await runnable.ainvoke("x raised to the third plus seven equals 12")

    print(result)


def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


async def calculate_length(config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    prompt = ChatPromptTemplate.from_template("what is {a} + {b}")

    # itemgetter("foo") === lambda d: d["foo"]
    chain = (
        {
            "a": itemgetter("foo") | RunnableLambda(length_function),
            "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
            | RunnableLambda(multiple_length_function),
        }
        | prompt
        | model
    )

    result = await chain.ainvoke({"foo": "bar", "bar": "gah"})

    print(result)


if __name__ == "__main__":
    # result = calculate("What is 3 * 12? Also, what is 11 + 49?", LangChainOption())
    # print(result)
    asyncio.run(calculate_length(LangChainOption()))
