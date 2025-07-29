import logging
from typing import Any, Annotated
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class Shrimp(BaseModel):
    name: Annotated[str, Field(max_length=10)]


class ShrimpTank(BaseModel):
    shrimp: list[Shrimp]


def create_resource_server() -> FastMCP:
    app = FastMCP(
        name="MCP_DEMO",
        port=8080,
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

    @app.resource("greeting://{name}")
    def get_greeting(name: str) -> str:
        """Get a personalized greeting"""
        return f"Hello, {name}!"

    return app


def main() -> int:
    try:
        mcp_server = create_resource_server()

        mcp_server.run(transport="streamable-http")
        logger.info("Server stopped")
        return 0
    except Exception:
        logger.exception("Server error")
        return 1


if __name__ == "__main__":
    main()
