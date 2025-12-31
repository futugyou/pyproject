from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from adk_adapter.adkutil import agui_agent_wrapper

from adk_adapter.adkagent.assistant import (
    build_assistant_agent,
)

from adk_adapter.adkagent.weather import (
    build_weather_agent,
)

from adk_adapter.adkagent.state import (
    build_state_agent,
)

from adk_adapter import client_factory


def register_adk_agents(app):
    app_name = "app_name"
    user_id = "user_id"
    llm = client_factory.build_llm()

    add_adk_fastapi_endpoint(
        app=app,
        agent=agui_agent_wrapper(build_assistant_agent(llm), app_name, user_id),
        path="/adk_assistant",
    )

    add_adk_fastapi_endpoint(
        app=app,
        agent=agui_agent_wrapper(build_weather_agent(llm), app_name, user_id),
        path="/adk_weather",
    )

    add_adk_fastapi_endpoint(
        app=app,
        agent=agui_agent_wrapper(build_state_agent(llm), app_name, user_id),
        path="/adk_state",
    )
