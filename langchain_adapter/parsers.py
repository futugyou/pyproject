from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


from .option import LangChainOption


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


if __name__ == "__main__":
    config = LangChainOption()
    list_parser("ice cream flavors", config)
