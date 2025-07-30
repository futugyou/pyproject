import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.metadata_utils import get_display_name


async def display_prompts(session: ClientSession):
    prompts = await session.list_prompts()

    for prompt in prompts.prompts:
        display_name = get_display_name(prompt)
        print(f"Prompt: {display_name}")
        if prompt.description:
            print(f"   {prompt.description}\n")
        if prompt.arguments:
            for arg in prompt.arguments:
                print(f"      {arg.name}: {arg.description}")


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
        print(f"Resource: {display_name} ({resource.uri})")

    templates_response = await session.list_resource_templates()
    for template in templates_response.resourceTemplates:
        display_name = get_display_name(template)
        print(f"Resource Template: {display_name}\n")


async def main():
    async with streamablehttp_client("http://127.0.0.1:8080/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            await display_tools(session)
            await display_resources(session)
            await display_prompts(session)


if __name__ == "__main__":
    asyncio.run(main())
