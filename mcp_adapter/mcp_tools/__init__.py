from mcp.server.fastmcp import FastMCP
from . import (
    time_tools,
    user_tools,
    name_tools,
    book_tools,
    process_tools,
    screenshot_tools,
    poem_tools,
    math_tools,
)


def register_tools(app: FastMCP):
    """
    Register all tools with the FastMCP application instance.
    """
    app.tool()(time_tools.get_time)
    app.tool()(user_tools.get_user_info)
    app.tool()(name_tools.name_shrimp)
    app.tool()(book_tools.book_table)
    app.tool()(process_tools.process_data)
    app.tool()(screenshot_tools.take_screenshot)
    app.tool()(poem_tools.generate_poem)
    app.tool()(math_tools.add)
