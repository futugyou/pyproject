from langchain.chat_models import init_chat_model
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


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

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke({"input": query})
    return response["output"]


if __name__ == "__main__":
    result = calculate("What is 3 * 12? Also, what is 11 + 49?", LangChainOption())
    print(result)
