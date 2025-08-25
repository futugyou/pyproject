import logging
import argparse
import datetime
from typing import Literal
from mcp.server.fastmcp import FastMCP

from .mcp_tools import register_tools
from .mcp_prompts import register_prompts
from .mcp_resources import register_resources
from .mcp_extensions import register_extensions

logger = logging.getLogger(__name__)


def create_mcp_server() -> FastMCP:
    app = FastMCP(
        name="mcp_adapter", port=8080, debug=True, dependencies=["pyautogui", "Pillow"]
    )

    register_tools(app)

    register_prompts(app)

    register_resources(app)

    register_extensions(app)

    return app


def main(
    transport: Literal["stdio", "streamable-http"] = "streamable-http",
) -> int:
    try:
        logger.info(f"transport: {transport}")
        print(f"transport: {transport}")
        mcp_server = create_mcp_server()
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
