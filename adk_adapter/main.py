from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))


from fastapi import FastAPI
from adk_adapter import agui

app = FastAPI()
agui.register_adk_agents(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
