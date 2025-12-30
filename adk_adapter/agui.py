from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint

from adk_adapter.adkagent.assistant import (
    build_assistant_agent,
    build_assistant_adk_agent,
)

from adk_adapter import client_factory


def register_adk_agents(app):
    llm = client_factory.build_llm()

    base_agent = build_assistant_agent(llm)

    add_adk_fastapi_endpoint(
        app=app,
        agent=build_assistant_adk_agent(base_agent),
        path="/adk_assistant",
    )
