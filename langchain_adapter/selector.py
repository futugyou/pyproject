from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain.prompts.example_selector.base import BaseExampleSelector

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


if __name__ == "__main__":
    custom_selector(LangChainOption())
