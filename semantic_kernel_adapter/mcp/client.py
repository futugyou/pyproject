import asyncio
import os
import argparse
from typing import Any, Literal, Annotated

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import (
    MCPStdioPlugin,
    MCPStreamableHttpPlugin,
    MCPPluginBase,
    MCPSsePlugin,
)
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from dotenv import load_dotenv

load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the Semantic Kernel MCP client.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio", "http"],
        default="stdio",
        help="Transport method to use (default: stdio).",
    )
    return parser.parse_args()


async def main(transport: Literal["sse", "stdio", "http"] = "stdio"):
    from ..service import build_kernel_pipeline

    kernel = build_kernel_pipeline()
    chat_completion_service = kernel.get_service("default")

    mcp_agent: MCPPluginBase = None
    if transport == "stdio":
        mcp_agent = MCPStdioPlugin(
            name="Menu",
            description="Menu plugin, for details about the menu, call this plugin.",
            command="uv",
            args=[
                "--directory=./semantic_kernel_adapter/mcp",
                "run",
                "server.py",
            ],
            env={
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "OPENAI_CHAT_MODEL_ID": os.getenv("OPENAI_CHAT_MODEL_ID"),
                "OPENAI_URL": os.getenv("OPENAI_URL"),
                "USEAGENT": "True",
            },
        )
    elif transport == "sse":
        mcp_agent = MCPSsePlugin(
            url="http://0.0.0.0:8000/sse",
            name="Menu",
            description="Menu plugin, for details about the menu, call this plugin.",
        )
    else:
        mcp_agent = MCPStreamableHttpPlugin(
            url="http://0.0.0.0:8000/mcp",
            name="Menu",
            description="Menu plugin, for details about the menu, call this plugin.",
        )

    restaurant_agent = await mcp_agent.__aenter__()

    agent = ChatCompletionAgent(
        service=chat_completion_service,
        name="PersonalAssistant",
        instructions="Help the user with restaurant bookings.",
        plugins=[restaurant_agent, TimePlugin()],
    )
    thread: ChatHistoryAgentThread | None = None
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        # 3. Invoke the agent for a response
        response = await agent.get_response(messages=user_input, thread=thread)
        print(f"# {response.name}: {response} ")
        thread = response.thread
    # 4. Cleanup: Clear the thread
    await thread.delete() if thread else None
    await mcp_agent.__aexit__(None, None, None)


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(transport=args.transport))
