import asyncio
import os
from dotenv import load_dotenv
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


def refund_flight(flight_id: str) -> str:
    """Refund a flight"""
    print(f"Flight {flight_id} refunded")
    return f"Flight {flight_id} refunded"


async def run() -> None:
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("GOOGLE_CHAT_MODEL_ID"),
        base_url=os.getenv("GOOGLE_URL"),
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.ANY,
            "structured_output": True,
        },
    )

    travel_agent = AssistantAgent(
        "travel_agent",
        model_client=model_client,
        handoffs=["flights_refunder", "user"],
        system_message="""You are a travel agent.
        The flights_refunder is in charge of refunding flights.
        If you need information from the user, you must first send your message, then you can handoff to the user.
        Use TERMINATE when the travel planning is complete.""",
    )

    flights_refunder = AssistantAgent(
        "flights_refunder",
        model_client=model_client,
        handoffs=["travel_agent", "user"],
        tools=[refund_flight],
        system_message="""You are an agent specialized in refunding flights.
        You only need flight reference numbers to refund a flight.
        You have the ability to refund a flight using the refund_flight tool.
        If you need information from the user, you must first send your message, then you can handoff to the user.
        When the transaction is complete, handoff to the travel agent to finalize.""",
    )

    termination = HandoffTermination(target="user") | TextMentionTermination(
        "TERMINATE"
    )
    team = Swarm([travel_agent, flights_refunder], termination_condition=termination)

    task = "I need to refund my flight."

    async def run_team_stream() -> None:
        task_result = await Console(team.run_stream(task=task))
        last_message = task_result.messages[-1]

        while (
            isinstance(last_message, HandoffMessage) and last_message.target == "user"
        ):
            user_message = input("User: ")

            task_result = await Console(
                team.run_stream(
                    task=HandoffMessage(
                        source="user", target=last_message.source, content=user_message
                    )
                )
            )
            last_message = task_result.messages[-1]

    # The code randomly reports errors after calling `refund_flight`
    # such as `TypeError: 'NoneType' object is not subscriptable`
    # and `Error: tool 'TERMINATE' not found in any workbench`
    await run_team_stream()
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run())
