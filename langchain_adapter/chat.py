from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:

example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


def generate_joke(input_text: str, config: LangChainOption) -> Joke:
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    structured_llm = llm.with_structured_output(Joke)
    few_shot_structured_llm = prompt | structured_llm
    return few_shot_structured_llm.invoke(input_text)


def generate_joke_stream(input_text: str, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    structured_llm = llm.with_structured_output(Joke)
    few_shot_structured_llm = prompt | structured_llm
    for chunk in few_shot_structured_llm.stream(input_text):
        print(chunk.punchline, end="|", flush=True)


if __name__ == "__main__":
    # result = generate_joke("Tell me a joke about cats", LangChainOption())
    # print(result)
    generate_joke_stream("Tell me a joke about cats", LangChainOption())
