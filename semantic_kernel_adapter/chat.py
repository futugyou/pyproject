from semantic_kernel import Kernel
from pathlib import Path


async def generate_joke(kernel: Kernel, topic: str, style: str = "silly") -> str:
    script_dir = Path(__file__).parent

    plugins_directory = script_dir.parent / "prompt_template_samples"
    funFunctions = kernel.add_plugin(
        parent_directory=str(plugins_directory), plugin_name="FunPlugin"
    )

    jokeFunction = funFunctions["Joke"]
    result = await kernel.invoke(jokeFunction, input=topic, style=style)
    return str(result)
