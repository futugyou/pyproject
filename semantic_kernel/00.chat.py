from pathlib import Path
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel import Kernel
import os
import asyncio

from dotenv import load_dotenv
load_dotenv()

script_dir = Path(__file__).parent
service_id = "default"

kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID"),
        service_id=service_id,
        # api_key=os.getenv("OPENAI_API_KEY"),
        async_client=AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_URL"),
        )
    )
)

plugins_directory = script_dir.parent / "prompt_template_samples"
funFunctions = kernel.add_plugin(parent_directory=str(
    plugins_directory), plugin_name="FunPlugin")

jokeFunction = funFunctions["Joke"]


async def main():
    result = await kernel.invoke(jokeFunction, input="travel to dinosaur age", style="silly")
    print(result)

asyncio.run(main())
