# pyproject

Write a project in Python

## Init

```
pip install uv
uv init 
uv add dotenv fastapi motor "semantic-kernel[google]" uvicorn mcp
```

## Structure

`api` use for vercle

`app` use for faskapi server

`google_colab` only for google colab

And other floders are services used by `app`

## Format

```
pip install ruff
ruff format .
```

## Run

```
uv sync --all-packages --upgrade
uv run uvicorn app.main:app --reload
uv run -m mcp_adapter.server or uv run -m mcp_adapter.main
uv run -m mcp_adapter.client

devui ./agent_adapter
```

## Other

```
source .venv/bin/activate
```
