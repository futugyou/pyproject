from semantic_kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelFunction, KernelArguments
from openai import AsyncOpenAI
import asyncio

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


async def chat(
    input_text: str,
    kernel: Kernel,
    chat_function: KernelFunction,
    chat_history: ChatHistory,
) -> None:
    answer = await kernel.invoke(
        chat_function, KernelArguments(user_input=input_text, history=chat_history)
    )
    chat_history.add_user_message(input_text)
    chat_history.add_assistant_message(str(answer))


async def generate_arguments(
    kernel: Kernel, input_texts: list[str], chat_history: ChatHistory
) -> ChatHistory:
    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )
    for input_text in input_texts:
        await chat(input_text, kernel, chat_function, chat_history)
    return chat_history


if __name__ == "__main__":

    async def main():
        from service import kernel

        chat_history = ChatHistory()
        chat_history.add_system_message(
            "You are a helpful chatbot who is good about giving book recommendations."
        )

        chat_function = await generate_arguments(
            kernel,
            [
                "Hi, I'm looking for book suggestions",
                "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?",
            ],
            chat_history,
        )
        print(chat_history)

    asyncio.run(main())
