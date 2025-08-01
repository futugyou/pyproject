from collections.abc import Awaitable, Callable

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.connectors.google_search import GoogleSearch
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters import FilterTypes, FunctionInvocationContext
from semantic_kernel.functions import KernelParameterMetadata

from service import kernel, chat_completion_service

kernel.add_function(
    plugin_name="google",
    function=GoogleSearch().create_search_function(
        description="Get details about Semantic Kernel concepts.",
        parameters=[
            KernelParameterMetadata(
                name="query",
                description="The search query.",
                type="str",
                is_required=True,
                type_object=str,
            ),
            KernelParameterMetadata(
                name="top",
                description="The number of results to return.",
                type="int",
                is_required=False,
                default_value=2,
                type_object=int,
            ),
            KernelParameterMetadata(
                name="skip",
                description="The number of results to skip.",
                type="int",
                is_required=False,
                default_value=0,
                type_object=int,
            ),
            KernelParameterMetadata(
                name="siteSearch",
                description="The site to search.",
                default_value="https://github.com/",
                type="str",
                is_required=False,
                type_object=str,
            ),
        ],
    ),
)
chat_function = kernel.add_function(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)
settings = OpenAIChatPromptExecutionSettings(
    service_id="default",
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    function_choice_behavior=FunctionChoiceBehavior.Auto(),
)

system_message = """
You are a chat bot, specialized in Semantic Kernel, Microsoft LLM orchestration SDK.
Assume questions are related to that, and use the Google search plugin to find answers.
"""
history = ChatHistory(system_message=system_message)
history.add_user_message("Hi there, who are you?")
history.add_assistant_message(
    "I am Mosscap, a chat bot. I'm trying to figure out what people need."
)


@kernel.filter(filter_type=FilterTypes.FUNCTION_INVOCATION)
async def log_google_filter(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
):
    if context.function.plugin_name == "google":
        print("Calling Google search with arguments:")
        if "query" in context.arguments:
            print(f'  Query: "{context.arguments["query"]}"')
        if "siteSearch" in context.arguments:
            print(f'  siteSearch: "{context.arguments["siteSearch"]}"')
        await next(context)
        print("Google search completed.")
    else:
        await next(context)


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    history.add_user_message(user_input)
    result = await chat_completion_service.get_chat_message_content(
        history, settings, kernel=kernel
    )
    if result:
        print(f"Mosscap:> {result}")
        history.add_message(result)
    return True


async def main():
    chatting = True
    print(
        "Welcome to the chat bot!\
        \n  Type 'exit' to exit.\
        \n  Try to find out more about the inner workings of Semantic Kernel."
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
