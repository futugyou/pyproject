import asyncio
from dataclasses import dataclass
import os
import argparse
from typing import Any, Literal, Annotated, Callable


from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)


@dataclass
class MyMessage:
    content: str


@default_subscription
class MyAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__("My agent")
        self._name = name
        self._counter = 0

    @message_handler
    async def my_message_handler(self, message: MyMessage, ctx: MessageContext) -> None:
        self._counter += 1
        if self._counter > 5:
            return
        content = f"{self._name}: Hello x {self._counter}"
        print(content)
        await self.publish_message(MyMessage(content=content), DefaultTopicId())


async def server() -> None:
    from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost

    service = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
    service.start()

    try:
        # Wait for the service to stop
        if os.name == "nt":
            # On Windows, the signal is not available, so we wait for a new event
            await asyncio.Event().wait()
        else:
            await service.stop_when_signal()
    except KeyboardInterrupt:
        print("Stopping service...")
    finally:
        await service.stop()


async def client() -> None:
    import asyncio

    from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime

    worker1 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker1.start()
    await MyAgent.register(worker1, "worker1", lambda: MyAgent("worker1"))

    worker2 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker2.start()
    await MyAgent.register(worker2, "worker2", lambda: MyAgent("worker2"))

    await worker2.publish_message(MyMessage(content="Hello!"), DefaultTopicId())

    # Let the agents run for a while.
    await asyncio.sleep(5)
    await worker1.stop()
    await worker2.stop()


def parse_arguments():
    parser = argparse.ArgumentParser(description="choose server or client")
    parser.add_argument(
        "--type",
        type=str,
        choices=["server", "client"],
        default="client",
        help="choose server or client",
    )
    return parser.parse_args()


async def main(runtype: Literal["server", "client"] = "client"):
    if runtype == "server":
        await server()
    if runtype == "client":
        await client()


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(runtype=args.type))
