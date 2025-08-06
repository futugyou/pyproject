# pyproject
python project

## local

```
pip install uv
pip install "uvicorn[standard]"
pip install --pre -r requirements.txt

uvicorn app.main:app --reload
```

## format

```
pip install ruff
ruff format .
```