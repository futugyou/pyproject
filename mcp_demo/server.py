import logging
import argparse
import datetime
from typing import Any, Generic, Literal, Annotated
from pydantic import BaseModel, Field, AnyUrl
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import TextResource
from mcp.types import (
    Completion,
    CompletionArgument,
    CompletionContext,
    PromptReference,
    ResourceTemplateReference,
)

logger = logging.getLogger(__name__)


class Shrimp(BaseModel):
    name: Annotated[str, Field(max_length=10)]


class ShrimpTank(BaseModel):
    shrimp: list[Shrimp]


class UserRequest(BaseModel):
    name: Annotated[str, Field(max_length=10)]
    age: Annotated[int, Field(ge=0, le=120)]


class UserInfo(BaseModel):
    name: Annotated[str, Field(max_length=10)]
    age: Annotated[int, Field(ge=0, le=120)]


def create_resource_server() -> FastMCP:
    app = FastMCP(
        name="MCP_DEMO",
        # port=8080,
        debug=True,
    )

    @app.tool()
    async def get_time() -> dict[str, Any]:
        """
        Get the current server time.

        This tool demonstrates that system information can be protected
        by OAuth authentication. User must be authenticated to access it.
        """

        now = datetime.datetime.now()

        return {
            "current_time": now.isoformat(),
            "timezone": "UTC",  # Simplified for demo
            "timestamp": now.timestamp(),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @app.tool()
    async def get_user_info(
        userFilter: Annotated[UserRequest, Field(description="User sreach filter")],
    ) -> list[UserInfo]:
        """
        get user info by name and age
        """
        print(f"name: {userFilter.name}, age: {userFilter.age}")
        users: list[UserInfo] = [
            UserInfo(name="Alice", age=30),
            UserInfo(name="Bob", age=25),
        ]

        return users

    @app.tool()
    def name_shrimp(
        tank: ShrimpTank,
        extra_names: Annotated[list[str], Field(max_length=10)],
    ) -> list[str]:
        """List all shrimp names in the tank"""
        return [shrimp.name for shrimp in tank.shrimp] + extra_names

    @app.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @app.resource("greeting://{honorifics}/{name}")
    def get_greeting(honorifics: str, name: str) -> str:
        """Get a personalized greeting"""
        return f"Hello, {honorifics} {name}!"

    @app.prompt()
    def greet_user(
        name: Annotated[str, Field(description="The user's name")],
        style: Annotated[
            str, Field(default="friendly", description="The style of greeting")
        ],
    ) -> str:
        """Generate a greeting prompt"""
        styles = {
            "friendly": "Please write a warm, friendly greeting",
            "formal": "Please write a formal, professional greeting",
            "casual": "Please write a casual, relaxed greeting",
        }

        return f"{styles.get(style, styles['friendly'])} for someone named {name}."

    @app.completion()
    async def handle_completion(
        ref: PromptReference | ResourceTemplateReference,
        argument: CompletionArgument,
        context: CompletionContext | None,
    ) -> Completion | None:
        if isinstance(ref, ResourceTemplateReference):
            if (
                ref.uri == "greeting://{honorifics}/{name}"
                and argument.name == "honorifics"
            ):
                # if context and context.arguments and context.arguments.get("owner") == "modelcontextprotocol":
                repos = ["Mr.", "Miss", "Mrs.", "Ms."]
                return Completion(values=repos, hasMore=False)

        if isinstance(ref, PromptReference):
            if ref.name == "greet_user" and argument.name == "style":
                styles = ["friendly", "formal", "casual"]
                return Completion(
                    values=[
                        style for style in styles if style.startswith(argument.value)
                    ],
                    hasMore=False,
                )

        return None

    text_resource = TextResource(
        uri=AnyUrl("resource://text"),
        name="text_resource",
        description="Basic resource",
        text="you got a text",
    )
    app.add_resource(text_resource)

    return app


def main(
    transport: Literal["stdio", "streamable-http"] = "streamable-http",
) -> int:
    try:
        logger.info(f"transport: {transport}")
        print(f"transport: {transport}")
        mcp_server = create_resource_server()
        mcp_server.run(transport=transport)
        logger.info("Server stopped")
        return 0
    except Exception:
        logger.exception("Server error")
        return 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport method to use (default: streamable-http).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(f"Running {__file__} at {datetime.datetime.now()}")
    args = parse_arguments()
    main(transport=args.transport)
