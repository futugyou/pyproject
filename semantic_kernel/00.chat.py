from pathlib import Path
import asyncio

from service import kernel

script_dir = Path(__file__).parent

plugins_directory = script_dir.parent / "prompt_template_samples"
funFunctions = kernel.add_plugin(
    parent_directory=str(plugins_directory), plugin_name="FunPlugin"
)

jokeFunction = funFunctions["Joke"]

if __name__ == "__main__":

    async def main():
        result = await kernel.invoke(
            jokeFunction, input="travel to dinosaur age", style="silly"
        )
        print(result)

    asyncio.run(main())
