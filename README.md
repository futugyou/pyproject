# pyproject
python project

## init

```
pip install uv
uv init 
pip add dotenv fastapi motor "semantic-kernel[google]" uvicorn mcp

uv run uvicorn app.main:app --reload
```

## caption

`api` use for vercle
`app` use for faskapi server
`transformers_demo` only for google colab
and other floders are services used by `app`

## format

```
pip install ruff
ruff format .
```