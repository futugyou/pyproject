from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from pathlib import Path
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel import Kernel
import os
import asyncio
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

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
        ),
    )
)

prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$user_input}}
ChatBot: """

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="chat",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(
            name="user_input", description="The user input", is_required=True
        ),
        InputVariable(
            name="history", description="The conversation history", is_required=True
        ),
    ],
)

chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
)
chat_history = ChatHistory()
chat_history.add_system_message(
    "You are a helpful chatbot who is good about giving book recommendations."
)


async def chat(input_text: str) -> None:
    # Save new message in the context variables
    print(f"User: {input_text}")

    # Process the user message and get an answer
    answer = await kernel.invoke(
        chat_function, KernelArguments(user_input=input_text, history=chat_history)
    )

    # Show the response
    print(f"ChatBot: {answer}")

    chat_history.add_user_message(input_text)
    chat_history.add_assistant_message(str(answer))


async def main():
    await chat("Hi, I'm looking for book suggestions")
    await chat(
        "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?"
    )
    print(chat_history)


asyncio.run(main())
