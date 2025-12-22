from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint

from agent_adapter.agent.weather import get_weather_agent
from agent_adapter.agent.joke import get_joke_agent
from agent_adapter.agent.light import get_light_agent
from agent_adapter.agent.mslearn import (
    get_docs_agent,
    get_docs_mcp_tool,
    get_docs_hostmcp_tool,
)
from agent_adapter.agent.code import get_code_agent
from agent_adapter.agent.state import get_state_agent

from agent_adapter.workflow.text import get_text_workflow
from agent_adapter.workflow.exec import get_exec_workflow
from agent_adapter.workflow.writing import get_writing_workflow

from agent_adapter import client_factory


def register_agents(app):
    client = client_factory.build_client("openai")

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_weather_agent(client),
        path="/weather",
    )

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_joke_agent(client),
        path="/joke",
    )

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_light_agent(client),
        path="/light",
    )

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_code_agent(client),
        path="/code",
    )

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_docs_agent(client, get_docs_mcp_tool()),
        path="/msdocs",
    )

    # Workflow's start executor cannot handle list[ChatMessage]
    
    # add_agent_framework_fastapi_endpoint(
    #     app=app,
    #     agent=get_text_workflow().as_agent(name="text-handler"),
    #     path="/text-handler",
    # )

    # add_agent_framework_fastapi_endpoint(
    #     app=app,
    #     agent=get_exec_workflow().as_agent(name="number-handler"),
    #     path="/number-handler",
    # )

    # add_agent_framework_fastapi_endpoint(
    #     app=app,
    #     agent=get_writing_workflow(client).as_agent(name="writing-handler"),
    #     path="/writing",
    # )

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_state_agent(client),
        path="/state",
    )
