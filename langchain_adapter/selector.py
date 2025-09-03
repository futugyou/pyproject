from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain.prompts import (
    PromptTemplate,
    StringPromptTemplate,
    FewShotPromptTemplate,
)
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
    LengthBasedExampleSelector,
)
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from typing import Dict, List
import numpy as np

from .option import LangChainOption


class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        return np.random.choice(self.examples, size=2, replace=False)


def custom_selector(config: LangChainOption):
    examples = [{"foo": "1"}, {"foo": "2"}, {"foo": "3"}]

    selector = CustomExampleSelector(examples)

    print(selector.select_examples({"foo": "foo"}))

    selector.add_example({"foo": "4"})
    print(selector.examples)

    print(selector.select_examples({"foo": "foo"}))


def length_selector(config: LangChainOption):
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=25,
    )
    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
    )
    print(dynamic_prompt.format(adjective="big"))

    long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
    print(dynamic_prompt.format(adjective=long_string))

    new_example = {"input": "big", "output": "small"}
    dynamic_prompt.example_selector.add_example(new_example)
    print(dynamic_prompt.format(adjective="enthusiastic"))


def mmr_selector(config: LangChainOption):
    embedding = GoogleGenerativeAIEmbeddings(
        model=config.lang_google_embedding_model,
        google_api_key=config.lang_google_api_key,
    )
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]

    # example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embedding,
        FAISS,
        k=3,
    )
    mmr_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
    )
    print(mmr_prompt.format(adjective="worried"))


if __name__ == "__main__":
    # custom_selector(LangChainOption())
    # length_selector(LangChainOption())
    mmr_selector(LangChainOption())
