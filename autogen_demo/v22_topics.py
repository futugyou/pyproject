import asyncio
from dataclasses import dataclass
from typing import Callable

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
    AgentId,
    SingleThreadedAgentRuntime,
    type_subscription,
    TypeSubscription,
    TopicId,
)


@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class ImageMessage:
    url: str
    source: str


class RoutedBySenderAgent(RoutedAgent):
    @message_handler(match=lambda msg, ctx: msg.source.startswith("user1"))  # type: ignore
    async def on_user1_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(
            f"Hello from user 1 handler, {message.source}, you said {message.content}!"
        )

    @message_handler(match=lambda msg, ctx: msg.source.startswith("user2"))  # type: ignore
    async def on_user2_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(
            f"Hello from user 2 handler, {message.source}, you said {message.content}!"
        )

    @message_handler(match=lambda msg, ctx: msg.source.startswith("user2"))  # type: ignore
    async def on_image_message(
        self, message: ImageMessage, ctx: MessageContext
    ) -> None:
        print(f"Hello, {message.source}, you sent me {message.url}!")


@dataclass
class Message:
    content: str


class InnerAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        return Message(content=f"Hello from inner, {message.content}")


class OuterAgent(RoutedAgent):
    def __init__(self, description: str, inner_agent_type: str):
        super().__init__(description)
        self.inner_agent_id = AgentId(inner_agent_type, self.id.key)

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"Received message: {message.content}")
        # Send a direct message to the inner agent and receives a response.
        response = await self.send_message(
            Message(f"Hello from outer, {message.content}"), self.inner_agent_id
        )
        print(f"Received inner response: {response.content}")


@type_subscription(topic_type="default")
class ReceivingAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"Received a message: {message.content}")


class BroadcastingAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        await self.publish_message(
            Message("Publishing a message from broadcasting agent!"),
            topic_id=TopicId(type="default", source=self.id.key),
        )


async def run() -> None:
    runtime = SingleThreadedAgentRuntime()
    await RoutedBySenderAgent.register(
        runtime, "my_agent", lambda: RoutedBySenderAgent("Routed by sender agent")
    )
    runtime.start()
    agent_id = AgentId("my_agent", "default")
    await runtime.send_message(
        TextMessage(content="Hello, World!", source="user1-test"), agent_id
    )
    await runtime.send_message(
        TextMessage(content="Hello, World!", source="user2-test"), agent_id
    )
    await runtime.send_message(
        ImageMessage(url="https://example.com/image.jpg", source="user1-test"), agent_id
    )
    await runtime.send_message(
        ImageMessage(url="https://example.com/image.jpg", source="user2-test"), agent_id
    )
    await runtime.stop_when_idle()

    print("")

    runtime = SingleThreadedAgentRuntime()
    await InnerAgent.register(runtime, "inner_agent", lambda: InnerAgent("InnerAgent"))
    await OuterAgent.register(
        runtime, "outer_agent", lambda: OuterAgent("OuterAgent", "inner_agent")
    )
    runtime.start()
    outer_agent_id = AgentId("outer_agent", "default")
    await runtime.send_message(Message(content="Hello, World!"), outer_agent_id)
    await runtime.stop_when_idle()

    print()

    runtime = SingleThreadedAgentRuntime()

    # Option 1: with type_subscription decorator
    # The type_subscription class decorator automatically adds a TypeSubscription to
    # the runtime when the agent is registered.
    await ReceivingAgent.register(
        runtime, "receiving_agent", lambda: ReceivingAgent("Receiving Agent")
    )

    # Option 2: with TypeSubscription
    await BroadcastingAgent.register(
        runtime, "broadcasting_agent", lambda: BroadcastingAgent("Broadcasting Agent")
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type="default", agent_type="broadcasting_agent")
    )

    # Start the runtime and publish a message.
    runtime.start()
    await runtime.publish_message(
        Message("Hello, World! From the runtime!"),
        topic_id=TopicId(type="default", source="default"),
    )
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(run())
