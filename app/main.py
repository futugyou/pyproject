from fastapi import FastAPI
from app.api.v1 import router as v1_router
from mcp_demo.server import create_resource_server
from app.mcp_openapi_merge import build_mcp_openapi_dict, merge_openapi_into_app
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class McpServer:
    name: str
    prefix: str
    group: str
    version: str
    mcp_server: Any
    mcp_app: Callable


mcp_servers = [
    McpServer(
        name="mcp_demo",
        prefix="/api/v1/mcp_demo",
        group="mcp_demo",
        version="1.0.0",
        mcp_server=(server := create_resource_server()),
        mcp_app=server.streamable_http_app(),
    )
]


# Use custom lifespan: start MCP sub-application and merge OpenAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        for mcp in mcp_servers:
            await stack.enter_async_context(
                mcp.mcp_app.router.lifespan_context(mcp.mcp_app)
            )

            mcp_oa = build_mcp_openapi_dict(
                mcp.mcp_server,
                title=mcp.name,
                version=mcp.version,
                prefix=mcp.prefix,
                group=mcp.group,
            )

        merge_openapi_into_app(app, mcp_oa)

        yield


app = FastAPI(lifespan=lifespan)

for mcp in mcp_servers:
    app.mount(mcp.prefix, mcp.mcp_app, mcp.name)

app.include_router(v1_router, prefix="/api/v1")
