from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function, KernelPlugin, KernelFunction
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatPromptExecutionSettings,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from pathlib import Path
from typing import Annotated
import asyncio
import random

prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$paragraph_count}} paragraphs long
- Be written in this language: {{$language}}
"""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="story",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="paragraph_count", description="The number of paragraphs", is_required=True),
        InputVariable(name="language", description="The language of the story", is_required=True),
    ],
)


class GenerateNumberPlugin:
    """
    Description: Generate a number between a min and a max.
    """

    @kernel_function(
        name="GenerateNumber",
        description="Generate a random number between min and max",
    )
    def generate_number(
        self,
        min: Annotated[int, "the minimum number of paragraphs"],
        max: Annotated[int, "the maximum number of paragraphs"] = 10,
    ) -> Annotated[int, "the output is a number"]:
        """
        Generate a number between min-max
        Example:
            min="4" max="10" => rand(4,8)
        Args:
            min -- The lower limit for the random number generation
            max -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(min, max))
        except ValueError as e:
            print(f"Invalid input {min} and {max}")
            raise e


async def generate_kernel_plugin_function(
    kernel: Kernel,
) -> tuple[KernelPlugin, KernelFunction]:
    func = kernel.add_function(
        function_name="CorgiStory",
        plugin_name="CorgiPlugin",
        prompt_template_config=prompt_template_config,
    )

    plugin = kernel.add_plugin(
        GenerateNumberPlugin(), "GenerateNumberPlugin"
    )
    return plugin, func


if __name__ == "__main__":

    async def main():
        from service import kernel

        plugin, func = await generate_kernel_plugin_function(kernel)
        generateNumber = plugin["GenerateNumber"]
        number_result = await generateNumber(kernel, min=1, max=5)
        print(number_result)
        story = await func.invoke(kernel, paragraph_count=number_result.value, language="Spanish")
        print(
            f"Generating a corgi story exactly {number_result.value} paragraphs long."
        )
        print("=====================================================")
        print(story)

    asyncio.run(main())
