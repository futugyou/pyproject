from semantic_kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory, ChatHistorySummarizationReducer
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
    await chat_history.add_message_async(answer.value[0])
    # chat_history.add_assistant_message(str(answer))


async def chat_history_with_summary(
    kernel: Kernel, input_texts: list[str]
) -> ChatHistory:
    chat_history = ChatHistorySummarizationReducer(
        service=kernel.get_service("default"),
        target_count=1,
        threshold_count=0,
        auto_reduce=True,
    )
    chat_history.add_system_message(
        "You are a helpful chatbot who is good about giving book recommendations."
    )
    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )

    for input_text in input_texts:
        await chat(input_text, kernel, chat_function, chat_history)

    return chat_history


def get_chat_history_summary(chat_history: ChatHistory) -> str:
    summary = ""
    for msg in chat_history.messages:
        if msg.metadata and msg.metadata.get("__summary__"):
            summary += msg.content
    return summary


if __name__ == "__main__":

    async def main():
        from ..service import build_kernel_pipeline

        kernel = build_kernel_pipeline()
        input_texts = [
            "Hi, I'm looking for book suggestions",
            "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?",
        ]
        chat_history = await chat_history_with_summary(kernel, input_texts)
        print(chat_history)
        print()
        summary = get_chat_history_summary(chat_history)
        print("Summary:", summary)

    asyncio.run(main())
