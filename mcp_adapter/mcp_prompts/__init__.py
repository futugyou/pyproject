from mcp.server.fastmcp import FastMCP
from . import user_prompts


def register_prompts(app: FastMCP):
    """
    Register all tools with the FastMCP application instance.
    """
    app.prompt()(user_prompts.greet_user)
