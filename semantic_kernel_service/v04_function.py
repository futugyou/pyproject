from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function, KernelPlugin, KernelFunction
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatPromptExecutionSettings,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from pathlib import Path
import asyncio
import random

prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$input}} paragraphs long. It must be this length.
"""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="story",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
)


class GenerateNumberPlugin:
    """
    Description: Generate a number between 3-x.
    """

    @kernel_function(
        description="Generate a random number between 3-x",
        name="GenerateNumberThreeOrHigher",
    )
    def generate_number_three_or_higher(self, input: str) -> str:
        """
        Generate a number between 3-<input>
        Example:
            "8" => rand(3,8)
        Args:
            input -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(3, int(input)))
        except ValueError as e:
            print(f"Invalid input {input}")
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
        generate_number_three_or_higher = plugin["GenerateNumberThreeOrHigher"]
        number_result = await generate_number_three_or_higher(kernel, input=6)
        print(number_result)
        story = await func.invoke(kernel, input=number_result.value)
        print(
            f"Generating a corgi story exactly {number_result.value} paragraphs long."
        )
        print("=====================================================")
        print(story)

    asyncio.run(main())
