from langchain.chat_models import init_chat_model
from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


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

    return structured_llm.invoke(input_text)


if __name__ == "__main__":
    result = generate_joke("Tell me a joke about cats", LangChainOption())
    print(result)
