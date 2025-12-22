from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))


import asyncio
from agent_framework import ChatAgent, MCPStreamableHTTPTool, HostedMCPTool
from agent_framework.openai import OpenAIChatClient

from agent_adapter import client_factory


def get_docs_mcp_tool() -> MCPStreamableHTTPTool:
    return MCPStreamableHTTPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
    )


# I don't understand the differences between MCPStreamableHTTPTool and HostedMCPTool, so how do I choose between them?
def get_docs_hostmcp_tool() -> HostedMCPTool:
    return HostedMCPTool(
        name="Microsoft Learn MCP",
        description="Tool for learning Microsoft.",
        url="https://learn.microsoft.com/api/mcp",
        approval_mode="never_require",
    )


def get_docs_agent(mcp_tool: MCPStreamableHTTPTool) -> ChatAgent:
    client = client_factory.build_client("openai")
    agent = client.create_agent(
        instructions="You help with Microsoft documentation questions.",
        name="ms_docs",
        tools=mcp_tool,
    )
    return agent


async def run(query: str) -> str:
    async with get_docs_mcp_tool() as tool, get_docs_agent(tool) as agent:
        result = await agent.run(query)
        text = result.text
        print(f"message: {text}")
        return text


if __name__ == "__main__":
    asyncio.run(
        run("How to create an Azure storage account using az cli? Keep it short.")
    )
