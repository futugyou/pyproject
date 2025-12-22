from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from web.api.v1 import router as v1_router
from mcp_adapter.server import create_mcp_server
from web.mcp_openapi_merge import (
    build_mcp_openapi_dict,
    merge_openapi_into_app,
)
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
        name="mcp_adapter",
        prefix="/api/v1/mcp_adapter",
        group="mcp_adapter",
        version="1.0.0",
        mcp_server=(server := create_mcp_server()),
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

            mcp_oa = await build_mcp_openapi_dict(
                mcp.mcp_server,
                title=mcp.name,
                version=mcp.version,
                prefix=mcp.prefix,
                group=mcp.group,
            )

            merge_openapi_into_app(app, mcp_oa)

        yield


app = FastAPI(lifespan=lifespan)


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title="scalar doc")


for mcp in mcp_servers:
    app.mount(mcp.prefix, mcp.mcp_app, mcp.name)

app.include_router(v1_router, prefix="/api/v1")
