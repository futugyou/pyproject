from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import dumpd, dumps, load, loads

from typing import Optional
from pydantic import BaseModel, Field
import json

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


def get_chain(config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    structured_llm = llm.with_structured_output(Joke)
    return prompt | structured_llm


def generate_joke(input_text: str, chain) -> Joke:
    return chain.invoke(input_text)


def generate_joke_stream(input_text: str, chain):
    for chunk in chain.stream(input_text):
        print(chunk.punchline, end="|", flush=True)


def save_chain(chain):
    string_representation = dumps(chain, pretty=True)
    with open("./langchain_adapter/files/save_chat.json", "w") as fp:
        json.dump(string_representation, fp)


def load_chain(path, query, config: LangChainOption):
    # NotImplementedError: Trying to load an object that doesn't implement serialization: 
    # {'lc': 1, 'type': 'not_implemented', 'id': ['langchain_core', 'output_parsers', 'openai_tools', 'PydanticToolsParser'], 
    # 'repr': "PydanticToolsParser(first_tool_only=True, tools=[<class '__main__.Joke'>])", 'name': 'PydanticToolsParser'}
    with open(path, "r") as fp:
        chain = loads(
            json.load(fp), secrets_map={"GOOGLE_API_KEY": config.lang_google_api_key}
        )
        result = generate_joke(query, chain)
        print(result)


if __name__ == "__main__":
    config = LangChainOption()
    chain = get_chain(config)
    # result = generate_joke("Tell me a joke about cats", chain)
    # print(result)
    # generate_joke_stream("Tell me a joke about cats", chain)
    # save_chain(chain)
    load_chain(
        "./langchain_adapter/files/save_chat.json", "Tell me a joke about cats", config
    )
