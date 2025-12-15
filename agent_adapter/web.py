
import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, project_root)

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from agent_framework.devui import serve
from agent_adapter import client_factory
from agent_adapter.tools.weather import get_weather

client = client_factory.build_client("openai")

agent = ChatAgent(
    name="WeatherAgent",
    chat_client=client,
    tools=[get_weather]
)

# Launch DevUI
serve(entities=[agent], auto_open=True)
# Opens browser to http://localhost:8080