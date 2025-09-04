from langchain_community.llms import AI21
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from .option import LangChainOption

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


# https://studio.ai21.com/
def get_chain(config: LangChainOption):
    llm = AI21(ai21_api_key=config.ai21_api_key)

    return prompt | llm


if __name__ == "__main__":
    chain = get_chain(LangChainOption())
    result = chain.invoke(
        "What NFL team won the Super Bowl in the year Justin Beiber was born?"
    )
    # ValueError: AI21 /complete call failed with status code 404. Details: None
    # NEED langchain_community upgrade
    print(result)
