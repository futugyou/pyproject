import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()


async def run() -> None:
    fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])

    # Create an MCP workbench which provides a session to the mcp server.
    async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore
        tools = await workbench.list_tools()
        print(tools)

        # Create an agent that can use the fetch tool.
        # I try to use google llm, but it may be miss the id `[FunctionCall(id='', arguments='{"url":"https://en.wikipedia.org/wiki/Seattle"}', name='fetch')]`
        model_client = OpenAIChatCompletionClient(
            model=os.getenv("OPENAI_CHAT_MODEL_ID"),
            base_url=os.getenv("OPENAI_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.ANY,
                "structured_output": True,
            },
        )

        fetch_agent = AssistantAgent(
            name="fetcher",
            model_client=model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
        )

        # Let the agent fetch the content of a URL and summarize it.
        result = await fetch_agent.run(
            task="Summarize the content of https://en.wikipedia.org/wiki/Seattle"
        )

        for msg in result.messages:
            if isinstance(msg, TextMessage):
                print(msg.content)

        # Close the connection to the model client.
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
