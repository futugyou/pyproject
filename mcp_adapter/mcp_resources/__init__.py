from mcp.server.fastmcp import FastMCP
from . import greet, text


def register_resources(app: FastMCP):
    """
    Register all tools with the FastMCP application instance.
    """
    app.resource("greeting://{honorifics}/{name}")(greet.get_greeting)
    app.add_resource(text.text_resource)
