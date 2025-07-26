import asyncio

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from typing import Annotated
from semantic_kernel.functions import kernel_function, KernelArguments
from service import kernel, chat_completion_service


@kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
    """A filter that will be called for each function call in the response."""
    print("\nAuto function invocation filter")
    print(f"Function: {context.function.name}")
    print(f"Request sequence: {context.request_sequence_index}")
    print(f"Function sequence: {context.function_sequence_index}")

    # as an example
    function_calls = context.chat_history.messages[-1].items
    print(f"Number of function calls: {len(function_calls)}")
    # if we don't call next, it will skip this function, and go to the next one
    await next(context)


class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"


# Simulate a conversation with the agent
USER_INPUTS = [
    "Hello",
    "What is the special soup?",
    "What does that cost?",
    "Thank you",
]


async def main():
    kernel.add_plugin(MenuPlugin(), plugin_name="menu")
    settings = kernel.get_prompt_execution_settings_from_service_id(
        service_id="default"
    )
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # 1. Create the agent by specifying the service
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Assistant",
        instructions="Answer questions about the world in one sentence.",
        arguments=KernelArguments(settings=settings),
    )
    thread: ChatHistoryAgentThread = None
    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        # 2. Invoke the agent for a response
        response = await agent.get_response(
            messages=user_input,
            thread=thread,
        )
        thread = response.thread
        # 3. Print the response
        print(f"# {response.name}: {response}")

    await thread.delete() if thread else None


if __name__ == "__main__":
    asyncio.run(main())
