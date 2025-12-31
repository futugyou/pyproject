from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
from typing import Callable, Any
from agent_adapter.agui import register_agents

from adk_adapter.agui import register_adk_agents


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        yield


app = FastAPI(
    title="Vercel + FastAPI + MAF",
    description="Vercel + FastAPI + MAF",
    version="1.0.0",
    lifespan=lifespan,
)

register_agents(app)
register_adk_agents(app)


@app.get("/")
def read_root():
    return {"Python": "on Vercel"}


@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title="scalar doc")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=5001, reload=True)
