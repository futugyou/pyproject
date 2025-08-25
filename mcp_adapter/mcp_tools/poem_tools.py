from mcp.server.fastmcp import Context
from mcp.types import (
    ClientCapabilities,
    SamplingCapability,
    TextContent,
    SamplingMessage,
)


async def generate_poem(topic: str, ctx: Context) -> str:
    """Generate a poem using LLM sampling."""
    prompt = f"Write a short poem about {topic}"
    if ctx.session.check_client_capability(
        ClientCapabilities(sampling=SamplingCapability())
    ):
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt),
                )
            ],
            max_tokens=100,
        )
        return result.content.text
    return prompt
