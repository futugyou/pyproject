from fastapi import FastAPI
from app.api.v1 import router as v1_router
from mcp_demo.server import create_resource_server

mcp_app = create_resource_server().streamable_http_app()

# app = FastAPI()
app = FastAPI(lifespan=mcp_app.router.lifespan_context)

app.mount("/api/v1", mcp_app, "mcp")
app.include_router(v1_router, prefix="/api/v1")
