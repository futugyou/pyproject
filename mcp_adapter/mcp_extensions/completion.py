from typing import Any, Generic, Literal, Annotated
from pydantic import BaseModel, Field, AnyUrl
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.resources import TextResource
from mcp.types import (
    PromptReference,
    ResourceTemplateReference,
    Completion,
    CompletionArgument,
    CompletionContext,
)


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
                values=[style for style in styles if style.startswith(argument.value)],
                hasMore=False,
            )
    return None
