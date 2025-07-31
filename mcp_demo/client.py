import asyncio
import logging
import argparse
import datetime
from typing import Any, Generic, Literal, Annotated
from pydantic import BaseModel, Field, AnyUrl

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.metadata_utils import get_display_name
from mcp.types import PromptReference, ResourceTemplateReference, ElicitResult

logger = logging.getLogger(__name__)


class UserInfo(BaseModel):
    name: Annotated[str, Field(max_length=10)]
    age: Annotated[int, Field(ge=0, le=120)]


async def progress_callback(
    progress: float, total: float | None, message: str | None
) -> None:
    print(f"Progress: {progress}, Total: {total}, Message: {message}")


async def call_some_tools(session: ClientSession):
    result = await session.call_tool("add", {"a": "123", "b": "456"})
    for value in result.content:
        print(f"Tool 'add' result: {value.text}")
    print("\n")

    result = await session.call_tool("get_time")
    for value in result.content:
        print(f"Tool 'get_time' result: {value.text}")
    print("\n")

    result = await session.call_tool(
        "get_user_info", {"userFilter": {"name": "John", "age": "30"}}
    )
    for value in result.content:
        user = UserInfo.model_validate_json(value.text)
        print(f"Tool 'get_user_info' result: {user}")
    print("\n")

    result = await session.call_tool(
        "book_table", {"date": "2024-12-25", "time": "12-12-12"}
    )
    for value in result.content:
        print(f"Tool 'book_table' result: {value.text}")
    print("\n")

    result = await session.call_tool(
        "process_data",
        {"data": "hello, everyone"},
        progress_callback=progress_callback,
    )
    for value in result.content:
        print(f"Tool 'process_data' result: {value.text}")


async def display_prompts(session: ClientSession):
    prompts = await session.list_prompts()

    for prompt in prompts.prompts:
        display_name = get_display_name(prompt)
        print(f"Prompt: {display_name}")
        if prompt.description:
            print(f"   {prompt.description}")
        if prompt.arguments:
            for arg in prompt.arguments:
                print(f"      {arg.name}: {arg.description}")

        result = await session.complete(
            ref=PromptReference(type="ref/prompt", name=prompt.name),
            argument={"name": "style", "value": "f"},
        )
        print(f"Completions for 'style' argument: {result.completion.values}\n")


async def display_tools(session: ClientSession):
    tools_response = await session.list_tools()

    for tool in tools_response.tools:
        display_name = get_display_name(tool)
        print(f"Tool: {display_name}")
        if tool.description:
            print(f"   {tool.description}\n")


async def display_resources(session: ClientSession):
    resources_response = await session.list_resources()

    for resource in resources_response.resources:
        display_name = get_display_name(resource)
        print(f"Resource: {display_name} ({resource.uri})\n")

    templates_response = await session.list_resource_templates()
    for template in templates_response.resourceTemplates:
        display_name = get_display_name(template)
        print(f"Resource Template: {display_name}")
        result = await session.complete(
            ref=ResourceTemplateReference(
                type="ref/resource", uri=template.uriTemplate
            ),
            argument={"name": "honorifics", "value": ""},
            context_arguments={"owner": "modelcontextprotocol"},
        )
        print(
            f"Completions for 'honorifics' with owner='modelcontextprotocol': {result.completion.values}\n"
        )


async def elicitation_callback(context, params):
    if params.message.startswith("No tables available on"):
        return ElicitResult(
            action="accept",
            content={"checkAlternative": "true", "alternativeDate": "2024-12-26"},
        )
    else:
        raise ValueError(f"Unexpected elicitation message: {params.message}")


async def run_session(session: ClientSession):
    try:
        await session.initialize()
    except Exception as e:
        logging.error("Session initialization failed: %s", e)
        return

    try:
        await display_tools(session)
        await display_resources(session)
        await display_prompts(session)
        await call_some_tools(session)
    except Exception as e:
        logging.error("Business logic execution failed: %s", e)


async def main(transport: Literal["stdio", "streamable-http"] = "streamable-http"):
    try:
        if transport == "stdio":
            server_params = StdioServerParameters(
                command="uv",
                args=["run", "server.py", "--transport", "stdio"],
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(
                    read, write, elicitation_callback=elicitation_callback
                ) as session:
                    await run_session(session)
        else:
            async with streamablehttp_client("http://127.0.0.1:8080/mcp") as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(
                    read_stream, write_stream, elicitation_callback=elicitation_callback
                ) as session:
                    await run_session(session)
    except FileNotFoundError as e:
        logging.error("Failed to start client due to missing file or command: %s", e)
    except ConnectionError as e:
        logging.error("Connection failed: %s", e)
    # except Exception as e:
    #     logging.error("Unexpected error occurred during client setup: %s", e)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the MCP client.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport method to use (default: streamable-http).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(transport=args.transport))
