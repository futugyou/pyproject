from fastapi import FastAPI
from app.api.v1 import router as v1_router
from mcp_demo.server import create_resource_server
from app.mcp_openapi_merge import build_mcp_openapi_dict, merge_openapi_into_app
from contextlib import asynccontextmanager

MCP_SERVER_NAME = "mcp_demo"
MCP_PREFIX = f"/api/v1/{MCP_SERVER_NAME}"

# This is a FastMCP instance (with tools)
mcp_server = create_resource_server()
# This is Starlette (only a few endpoints like /sse)
mcp_app = mcp_server.streamable_http_app()


# Use custom lifespan: start MCP sub-application and merge OpenAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.router.lifespan_context(mcp_app):
        # Build a "virtual endpoint" OpenAPI (with a custom prefix)
        mcp_oa = build_mcp_openapi_dict(
            mcp_server, title="My MCP Server", version="1.0.0", prefix=MCP_PREFIX
        )

        merge_openapi_into_app(app, mcp_oa)
        yield


app = FastAPI(lifespan=lifespan)
app.mount(MCP_PREFIX, mcp_app, MCP_SERVER_NAME)
app.include_router(v1_router, prefix="/api/v1")
