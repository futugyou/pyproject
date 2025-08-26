from semantic_kernel import Kernel
from pathlib import Path
import asyncio


async def generate_joke(kernel: Kernel, topic: str, style: str = "silly") -> str:
    script_dir = Path(__file__).parent

    plugins_directory = script_dir.parent / "prompt_template_samples"
    funFunctions = kernel.add_plugin(
        parent_directory=str(plugins_directory), plugin_name="FunPlugin"
    )

    jokeFunction = funFunctions["Joke"]
    result = await kernel.invoke(jokeFunction, input=topic, style=style)
    return str(result)


if __name__ == "__main__":

    async def main():
        from ..service import build_kernel_pipeline

        kernel = build_kernel_pipeline()
        result = await generate_joke(kernel, "travel to dinosaur age", "silly")
        print(result)

    asyncio.run(main())
