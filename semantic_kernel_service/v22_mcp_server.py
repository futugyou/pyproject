import logging
import argparse
from typing import Any, Literal, Annotated
import os

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.prompt_template import (
    InputVariable,
    KernelPromptTemplate,
    PromptTemplateConfig,
)
from mcp.server.lowlevel.server import Server
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

"""
This sample demonstrates how to expose your Semantic Kernel `kernel` instance as a MCP server.

To run this sample, set up your MCP host (like Claude Desktop or VSCode Github Copilot Agents)
with the following configuration:
```json
{
    "mcpServers": {
        "sk": {
            "command": "uv",
            "args": [
                "--directory=./semantic_kernel_service",
                "run",
                "v22_mcp_server.py"
            ],
            "env": {
                "OPENAI_API_KEY": "<your_openai_api_key>",
                "OPENAI_CHAT_MODEL_ID": "gpt-4o-mini",
                "OPENAI_URL": "https://models.github.ai/inference",
                "USEAGENT": "True"
            }
        }
    }
}
```

Note: You might need to set the uv to its full path.

Alternatively, you can run this as a SSE server, by setting the same environment variables as above, 
and running the following command:
```bash
uv --directory=./semantic_kernel_service run sk_mcp_server.py --transport sse --port 8000
```
This will start a server that listens for incoming requests on port 8000.

In both cases, uv will make sure to install semantic-kernel with the mcp extra for you in a temporary venv.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the Semantic Kernel MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio", "http"],
        default="stdio",
        help="Transport method to use (default: stdio).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to use for SSE transport (required if transport is 'sse').",
    )
    return parser.parse_args()


class MenuPlugin:
    """A sample Menu Plugin used for the sample."""

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


template = """{{$messages}}
---
Group the following PRs into one of these buckets for release notes, keeping the same order: 

-New Features 
-Enhancements and Improvements
-Bug Fixes
-Python Package Updates 

Include the output in raw markdown.
"""

prompt = KernelPromptTemplate(
    prompt_template_config=PromptTemplateConfig(
        name="release_notes_prompt",
        description="This creates the prompts for a full set of release notes based on the PR messages given.",
        template=template,
        input_variables=[
            InputVariable(
                name="messages",
                description="These are the PR messages, they are a single string with new lines.",
                is_required=True,
                json_schema='{"type": "string"}',
            )
        ],
    )
)


def run(
    transport: Literal["sse", "stdio", "http"] = "stdio", port: int | None = None
) -> None:
    @kernel_function()
    def echo_function(message: str, extra: str = "") -> str:
        """Echo a message as a function"""
        return f"Function echo: {message} {extra}"

    chat_completion_service = OpenAIChatCompletion(service_id="default")
    base_url = os.environ.get("OPENAI_URL")
    use_agent = os.environ.get("USEAGENT")
    if base_url:
        chat_completion_service = OpenAIChatCompletion(
            ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID"),
            service_id="default",
            async_client=AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_URL"),
            ),
        )

    server: Server = None
    if use_agent == "True":
        agent = ChatCompletionAgent(
            service=chat_completion_service,
            name="Host",
            instructions="Answer questions about the menu.",
            plugins=[MenuPlugin()],
        )
        server = agent.as_mcp_server(server_name="sk", prompts=[prompt])
    else:
        kernel = Kernel()
        kernel.add_service(chat_completion_service)
        kernel.add_function("echo", echo_function, "echo_function")
        kernel.add_function(
            plugin_name="prompt",
            function_name="prompt",
            prompt_template_config=PromptTemplateConfig(
                name="prompt",
                description="This is a prompt",
                template="Please repeat this: {{$message}} and this: {{$extra}}",
                input_variables=[
                    InputVariable(
                        name="message",
                        description="This is the message.",
                        is_required=True,
                        json_schema='{ "type": "string", "description": "This is the message."}',
                    ),
                    InputVariable(
                        name="extra",
                        description="This is extra.",
                        default="default",
                        is_required=False,
                        json_schema='{ "type": "string", "description": "This is the message."}',
                    ),
                ],
            ),
        )
        server = kernel.as_mcp_server(server_name="sk", prompts=[prompt])

    if transport == "sse" and port is not None:
        import nest_asyncio
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )
        nest_asyncio.apply()
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)  # nosec
    elif transport == "stdio":
        import anyio
        from mcp.server.stdio import stdio_server

        async def handle_stdin(
            stdin: Any | None = None, stdout: Any | None = None
        ) -> None:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )

        anyio.run(handle_stdin)
    else:
        import contextlib
        import nest_asyncio
        import uvicorn
        from collections.abc import AsyncIterator
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount
        from starlette.types import Receive, Scope, Send

        session_manager = StreamableHTTPSessionManager(
            app=server,
            json_response=True,
        )

        async def handle_streamable_http(
            scope: Scope, receive: Receive, send: Send
        ) -> None:
            await session_manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            """Context manager for managing session manager lifecycle."""
            async with session_manager.run():
                logger.info("Application started with StreamableHTTP session manager!")
                try:
                    yield
                finally:
                    logger.info("Application shutting down...")

        starlette_app = Starlette(
            debug=True,
            routes=[
                Mount("/mcp", app=handle_streamable_http),
            ],
            lifespan=lifespan,
        )

        nest_asyncio.apply()
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    args = parse_arguments()
    run(transport=args.transport, port=args.port)
