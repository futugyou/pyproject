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
- The two names of the corgis are {{GenerateNames.generate_names}}
"""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="story",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(
            name="paragraph_count",
            description="The number of paragraphs",
            is_required=True,
        ),
        InputVariable(
            name="language", description="The language of the story", is_required=True
        ),
    ],
)


class GenerateNamesPlugin:
    """
    Description: Generate character names.
    """

    # The default function name will be the name of the function itself, however you can override this
    # by setting the name=<name override> in the @kernel_function decorator. In this case, we're using
    # the same name as the function name for simplicity.
    @kernel_function(description="Generate character names", name="generate_names")
    def generate_names(self) -> str:
        """
        Generate two names.
        Returns:
            str
        """
        names = {"Hoagie", "Hamilton", "Bacon", "Pizza", "Boots", "Shorts", "Tuna"}
        first_name = random.choice(list(names))
        names.remove(first_name)
        second_name = random.choice(list(names))
        return f"{first_name}, {second_name}"


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
    kernel.add_plugin(GenerateNamesPlugin(), plugin_name="GenerateNames")
    func = kernel.add_function(
        function_name="CorgiStory",
        plugin_name="CorgiPlugin",
        prompt_template_config=prompt_template_config,
    )

    plugin = kernel.add_plugin(GenerateNumberPlugin(), "GenerateNumberPlugin")
    return plugin, func


if __name__ == "__main__":

    async def main():
        from ..service import build_kernel_pipeline

        kernel = build_kernel_pipeline()

        plugin, func = await generate_kernel_plugin_function(kernel)
        generateNumber = plugin["GenerateNumber"]
        number_result = await generateNumber(kernel, min=1, max=5)
        print(number_result)
        story = await func.invoke(
            kernel, paragraph_count=number_result.value, language="Spanish"
        )
        print(
            f"Generating a corgi story exactly {number_result.value} paragraphs long."
        )
        print("=====================================================")
        print(story)

    asyncio.run(main())
