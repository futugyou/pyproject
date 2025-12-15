import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, project_root)

from agent_framework.devui import serve
from agent_adapter.weather_agent import agent as weather_agent
from agent_adapter.joke_agent import agent as joke_agent


def main():
    entities = [weather_agent, joke_agent]

    serve(entities=entities, port=8090, auto_open=True)


if __name__ == "__main__":
    main()
