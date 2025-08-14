# pyproject

Write a project in Python

## Init

```
pip install uv
uv init 
pip add dotenv fastapi motor "semantic-kernel[google]" uvicorn mcp

uv run uvicorn app.main:app --reload
```

## Structure

`api` use for vercle

`app` use for faskapi server

`transformers_demo` only for google colab

And other floders are services used by `app`

## Format

```
pip install ruff
ruff format .
```