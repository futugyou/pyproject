import asyncio
from pydantic import BaseModel
from semantic_kernel.agents import (
    Agent,
    ChatCompletionAgent,
    HandoffOrchestration,
    OrchestrationHandoffs,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import (
    AuthorRole,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    StreamingChatMessageContent,
)
from semantic_kernel.functions import kernel_function


class OrderStatusPlugin:
    @kernel_function
    def check_order_status(self, order_id: str) -> str:
        """Check the status of an order."""
        # Simulate checking the order status
        return f"Order {order_id} is shipped and will arrive in 2-3 days."


class OrderRefundPlugin:
    @kernel_function
    def process_refund(self, order_id: str, reason: str) -> str:
        """Process a refund for an order."""
        # Simulate processing a refund
        print(f"Processing refund for order {order_id} due to: {reason}")
        return f"Refund for order {order_id} has been processed successfully."


class OrderReturnPlugin:
    @kernel_function
    def process_return(self, order_id: str, reason: str) -> str:
        """Process a return for an order."""
        # Simulate processing a return
        print(f"Processing return for order {order_id} due to: {reason}")
        return f"Return for order {order_id} has been processed successfully."


def get_agents(chat_completion_service) -> tuple[list[Agent], OrchestrationHandoffs]:
    support_agent = ChatCompletionAgent(
        name="TriageAgent",
        description="A customer support agent that triages issues.",
        instructions="Handle customer requests.",
        service=chat_completion_service,
    )

    refund_agent = ChatCompletionAgent(
        name="RefundAgent",
        description="A customer support agent that handles refunds.",
        instructions="Handle refund requests.",
        service=chat_completion_service,
        plugins=[OrderRefundPlugin()],
    )

    order_status_agent = ChatCompletionAgent(
        name="OrderStatusAgent",
        description="A customer support agent that checks order status.",
        instructions="Handle order status requests.",
        service=chat_completion_service,
        plugins=[OrderStatusPlugin()],
    )

    order_return_agent = ChatCompletionAgent(
        name="OrderReturnAgent",
        description="A customer support agent that handles order returns.",
        instructions="Handle order return requests.",
        service=chat_completion_service,
        plugins=[OrderReturnPlugin()],
    )

    # Define the handoff relationships between agents
    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            source_agent=support_agent.name,
            target_agents={
                refund_agent.name: "Transfer to this agent if the issue is refund related",
                order_status_agent.name: "Transfer to this agent if the issue is order status related",
                order_return_agent.name: "Transfer to this agent if the issue is order return related",
            },
        )
        .add(
            source_agent=refund_agent.name,
            target_agent=support_agent.name,
            description="Transfer to this agent if the issue is not refund related",
        )
        .add(
            source_agent=order_status_agent.name,
            target_agent=support_agent.name,
            description="Transfer to this agent if the issue is not order status related",
        )
        .add(
            source_agent=order_return_agent.name,
            target_agent=support_agent.name,
            description="Transfer to this agent if the issue is not order return related",
        )
    )

    return [
        support_agent,
        refund_agent,
        order_status_agent,
        order_return_agent,
    ], handoffs


def streaming_agent_response_callback(
    message: StreamingChatMessageContent, is_final: bool
) -> None:
    """Observer function to print the messages from the agents.

    Please note that this function is called whenever the agent generates a response,
    including the internal processing messages (such as tool calls) that are not visible
    to other agents in the orchestration.

    In streaming mode, the FunctionCallContent and FunctionResultContent are provided as a
    complete message.

    Args:
        message (StreamingChatMessageContent): The streaming message content from the agent.
        is_final (bool): Indicates if this is the final part of the message.
    """
    global is_new_message
    if is_new_message:
        print(f"{message.name}: ", end="", flush=True)
        is_new_message = False
    print(message.content, end="", flush=True)

    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(
                f"Calling '{item.name}' with arguments '{item.arguments}'",
                end="",
                flush=True,
            )
        if isinstance(item, FunctionResultContent):
            print(f"Result from '{item.name}' is '{item.result}'", end="", flush=True)

    if is_final:
        print()
        is_new_message = True


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents.

    Please note that this function is called whenever the agent generates a response,
    including the internal processing messages (such as tool calls) that are not visible
    to other agents in the orchestration.
    """
    print(f"{message.name}: {message.content}")
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"Calling '{item.name}' with arguments '{item.arguments}'")
        if isinstance(item, FunctionResultContent):
            print(f"Result from '{item.name}' is '{item.result}'")


def human_response_function() -> ChatMessageContent:
    """Observer function to print the messages from the agents."""
    user_input = input("User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


async def main():
    from ..service import build_kernel_pipeline

    kernel = build_kernel_pipeline()
    chat_completion_service = kernel.get_service("default")

    agents, handoffs = get_agents(chat_completion_service)
    handoff_orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        # agent_response_callback=agent_response_callback,
        streaming_agent_response_callback=streaming_agent_response_callback,
        human_response_function=human_response_function,
    )

    runtime = InProcessRuntime()
    runtime.start()
    handoff_result = await handoff_orchestration.invoke(
        task="Greet the customer who is reaching out for support.",
        runtime=runtime,
    )

    try:
        # Attempt to get the result will result in an exception due to cancellation
        value = await handoff_result.get()
        print(f"***** Final Result *****\n{value}")
    except Exception as e:
        print(e)
    finally:
        # 5. Stop the runtime
        await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
