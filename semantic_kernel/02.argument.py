from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from openai import AsyncOpenAI
import asyncio

from service import kernel

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


if __name__ == "__main__":

    async def main():
        await chat("Hi, I'm looking for book suggestions")
        await chat(
            "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?"
        )
        print(chat_history)

    asyncio.run(main())
