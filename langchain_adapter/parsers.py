from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import DatetimeOutputParser, PydanticOutputParser
from langchain.output_parsers.enum import EnumOutputParser
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from pydantic import BaseModel, Field, field_validator
from typing import List
from enum import Enum

from .option import LangChainOption


class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


def list_parser(input_text: str, config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    output_parser = CommaSeparatedListOutputParser()

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="List five {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions},
    )
    _input = prompt.format(subject=input_text)
    output = model.invoke(_input)

    print(output_parser.parse(output.content))


def datetime_parser(input_text: str, config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    output_parser = DatetimeOutputParser()
    template = """Answer the users question:

    {question}

    {format_instructions}"""
    prompt = PromptTemplate.from_template(
        template,
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    chain = prompt | model | output_parser
    result = chain.invoke({"question": input_text})
    print(result)


def enum_parser():
    parser = EnumOutputParser(enum=Colors)

    parser.parse("red")
    parser.parse(" green")
    parser.parse("blue\n")
    # langchain_core.exceptions.OutputParserException: Response 'yellow' is not one of the expected values: ['red', 'green', 'blue']
    # parser.parse("yellow")


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # V2 style field validator
    @field_validator("setup")
    def question_ends_with_question_mark(cls, value: str) -> str:
        if not value.endswith("?"):
            raise ValueError("Badly formed question!")
        return value


def pydantic_parser(input_text: str, config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    output_parser = PydanticOutputParser(pydantic_object=Joke)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    chain = prompt | model | output_parser
    result = chain.invoke({"query": input_text})
    print(result)


if __name__ == "__main__":
    config = LangChainOption()
    # list_parser("ice cream flavors", config)
    # datetime_parser("around when was bitcoin founded?", config)
    # enum_parser()
    pydantic_parser("Tell me a joke.", config)
