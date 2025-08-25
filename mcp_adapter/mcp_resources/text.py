from mcp.server.fastmcp.resources import TextResource
from pydantic import BaseModel, Field, AnyUrl

text_resource = TextResource(
    uri=AnyUrl("resource://text"),
    name="text_resource",
    description="Basic resource",
    text="you got a text",
)
