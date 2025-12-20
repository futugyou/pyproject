from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))


from dotenv import load_dotenv

load_dotenv()
from agent_framework.devui import serve
# from agent_framework.observability import configure_otel_providers

# Enable console output for local development
# configure_otel_providers()

from agent_adapter import otel
from agent_adapter.weather_agent.agent import agent as weather_agent
from agent_adapter.joke_agent.agent import agent as joke_agent
from agent_adapter.light_agent.agent import agent as light_agent
from agent_adapter.text_workflow.workflow import workflow as text_workflow
from agent_adapter.exec_workflow.workflow import workflow as exec_workflow
from agent_adapter.writing_workflow.workflow import workflow as writing_workflow
from agent_adapter.mslearn_agent.agent import agent as mslearn_agent


def main():
    otel.otel_configure()
    entities = [
        weather_agent,
        joke_agent,
        light_agent,
        text_workflow,
        exec_workflow,
        writing_workflow,
        mslearn_agent,
    ]

    serve(entities=entities, port=8090, auto_open=True, tracing_enabled=True)


if __name__ == "__main__":
    main()
