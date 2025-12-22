
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

def register_agents(app):
    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_weather_agent(),
        path="/weather",
    )
    
    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_joke_agent(),
        path="/joke",
    )
    
    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_light_agent(),
        path="/light",
    )
    
    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_code_agent(),
        path="/code",
    )
    
    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=get_docs_agent(get_docs_mcp_tool()),
        path="/msdocs",
    )
    