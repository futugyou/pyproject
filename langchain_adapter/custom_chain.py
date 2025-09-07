from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate

from .option import LangChainOption


class CustomChain(Chain, BaseModel):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )
        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"


def call_custom_chain(topic: str, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )
    chain = CustomChain(
        prompt=PromptTemplate.from_template("tell us a joke about {topic}"),
        llm=llm,
    )

    return chain.invoke({"topic": topic}, callbacks=[StdOutCallbackHandler()])


if __name__ == "__main__":
    config = LangChainOption()
    # result = generate_joke("Tell me a joke about cats", config)
    # print(result)
    # generate_joke_stream("Tell me a joke about cats", config)
    result = call_custom_chain("callbacks", config)
    print(result)
