from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))

 
from typing import Annotated
from pydantic import Field
from agent_framework import ChatAgent, ai_function
from agent_framework.openai import OpenAIChatClient
from agent_framework.ag_ui import AgentFrameworkAgent, RecipeConfirmationStrategy

from agent_adapter import client_factory

STATE_SCHEMA: dict[str, object] = {
    "language": {
        "type": "string",
        "enum": ["english", "spanish"],
        "description": "Preferred language.",
    }
}

PREDICT_STATE_CONFIG: Dict[str, Dict[str, str]] = {
    "language": {"tool": "update_language", "tool_argument": "language"}
}


@ai_function(
    name="update_language",
    description="Update the preferred language (english or spanish).",
)
def update_language(
    language: Annotated[
        str, Field(description="Preferred language: 'english' or 'spanish'")
    ],
) -> str:
    normalized = (language or "").strip().lower()
    if normalized not in ("english", "spanish"):
        return "Language unchanged. Use 'english' or 'spanish'."
    return f"Language updated to {normalized}."


def get_state_agent(client: ChatClientProtocol) -> AgentFrameworkAgent:
    agent = client.create_agent(
        instructions="You are a helpful assistant",
        name="state_agent",
        tools=[update_language],
    )

    return AgentFrameworkAgent(
        agent=agent,
        name="state_agent",
        description="Assistant that tracks a simple language state.",
        state_schema=STATE_SCHEMA,
        predict_state_config=PREDICT_STATE_CONFIG,
        confirmation_strategy=RecipeConfirmationStrategy(),
        require_confirmation=False,
    )
