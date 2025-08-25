from mcp.server.fastmcp import FastMCP
from . import completion


def register_extensions(app: FastMCP):
    """
    Register all tools with the FastMCP application instance.
    """
    app.completion()(completion.handle_completion)
