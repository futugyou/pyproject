from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))

import os
from dotenv import load_dotenv

load_dotenv()

from agent_framework.devui import serve
# from agent_framework.observability import configure_otel_providers

# Enable console output for local development
# configure_otel_providers()

from agent_adapter import otel
from agent_adapter.agent.weather import get_weather_agent
from agent_adapter.agent.joke import get_joke_agent
from agent_adapter.agent.light import get_light_agent
from agent_adapter.agent.mslearn import (
    get_docs_agent,
    get_docs_mcp_tool,
    get_docs_hostmcp_tool,
)
from agent_adapter.agent.code import get_code_agent

from agent_adapter.workflow.text import get_text_workflow
from agent_adapter.workflow.exec import get_exec_workflow
from agent_adapter.workflow.writing import get_writing_workflow


from agent_adapter.checkpoint.postgres import PostgresCheckpointStorage
from agent_adapter import client_factory


def main():
    otel.otel_configure()
    client = client_factory.build_client("openai")
    entities = [
        get_weather_agent(client),
        get_joke_agent(client),
        get_light_agent(client),
        get_text_workflow(),
        get_exec_workflow(PostgresCheckpointStorage(os.getenv("POSTGRES_URI"))),
        get_writing_workflow(client),
        get_docs_agent(client, get_docs_hostmcp_tool()),
        get_code_agent(client),
    ]

    serve(entities=entities, port=8090, auto_open=True, tracing_enabled=True)


if __name__ == "__main__":
    main()
