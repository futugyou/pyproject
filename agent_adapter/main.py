from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))


from agent_framework.devui import serve
from agent_adapter.weather_agent.agent import agent as weather_agent
from agent_adapter.joke_agent.agent import agent as joke_agent
from agent_adapter.light_agent.agent import agent as light_agent


def main():
    entities = [weather_agent, joke_agent, light_agent]

    serve(entities=entities, port=8090, auto_open=True)


if __name__ == "__main__":
    main()
