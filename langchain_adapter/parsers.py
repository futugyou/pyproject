from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import (
    DatetimeOutputParser,
    PydanticOutputParser,
    OutputFixingParser,
    RetryOutputParser,
    RetryWithErrorOutputParser,
    StructuredOutputParser,
    ResponseSchema,
)
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


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


def retry_parser(input_text: str, config: LangChainOption):
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    output_parser = PydanticOutputParser(pydantic_object=Action)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    # chain = prompt | model | output_parser
    # result = chain.invoke({"query": input_text})
    # print(result)

    bad_response = '{"action": "search"}'
    # Field required [type=missing, input_value={'action': 'search'}, input_type=dict]
    # output_parser.parse(bad_response)

    fix_parser = OutputFixingParser.from_llm(parser=output_parser, llm=model)
    print(fix_parser.parse(bad_response))

    retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=model)
    print(
        retry_parser.parse_with_prompt(
            bad_response, prompt.format_prompt(query="who is leo di caprios gf?")
        )
    )


def structured_parser(input_text: str, config: LangChainOption):
    response_schemas = [
        ResponseSchema(name="answer", description="answer to the user's question"),
        ResponseSchema(
            name="source",
            description="source used to answer the user's question, should be a website.",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="answer the users question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )

    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    output_parser = PydanticOutputParser(pydantic_object=Action)

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
    # pydantic_parser("Tell me a joke.", config)
    # retry_parser("who is leo di caprios gf?", config)
    structured_parser("what is the population of france?", config)
