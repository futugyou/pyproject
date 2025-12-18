from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

from typing import AsyncGenerator
from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.openai import OpenAIChatClient
from agent_framework.observability import (
    configure_otel_providers,
    get_tracer,
    get_meter,
)
from opentelemetry import trace
from opentelemetry.trace.span import format_trace_id

from agent_adapter import client_factory
from agent_adapter import otel
from agent_adapter.storage import history
from agent_adapter.tools.light import get_lights, change_state, LightInfo, LightListInfo
from agent_adapter.middleware.agent import logging_agent_middleware
from agent_adapter.middleware.function import logging_function_middleware
from agent_adapter.middleware.chat import LoggingChatMiddleware


def get_light_agent() -> ChatAgent:
    client = client_factory.build_client("openai")
    chat = LoggingChatMiddleware()

    agent = client.create_agent(
        instructions="You are a useful light assistant. can tall user the status of the lights and can help user control the lights on and off",
        name="light",
        middleware=[logging_agent_middleware, logging_function_middleware, chat],
        tools=[get_lights, change_state],
        chat_message_store_factory=lambda: history.PostgresChatMessageStore(
            postgres_url=os.getenv("POSTGRES_URI")
        ),
    )
    return agent


agent = get_light_agent()


async def get_lights() -> AsyncGenerator[str, None]:
    response = await agent.run(
        "Can you tell me the status of all the lights?", response_format=LightListInfo
    )
    if response.value:
        lights = response.value
        for light in lights.items:
            yield f"Light {light.id}:{light.name} is {'on' if light.is_on else 'off'}"
    else:
        yield "No structured data found in response"


async def change_light_state() -> str:
    with get_tracer().start_as_current_span(
        name="change_light_state", kind=trace.SpanKind.CLIENT
    ):
        response = await agent.run(
            "can you turn off all the lights?", response_format=LightInfo
        )
        counter = get_meter().create_counter("llm_call_counter")
        counter.add(1, {"func": "list"})
        if response.user_input_requests:
            for user_input_needed in response.user_input_requests:
                logging.info(
                    f"user_approval_needed function: {user_input_needed.function_call.name}, arguments: {user_input_needed.function_call.arguments}"
                )
                print(f"Function: {user_input_needed.function_call.name}")
                print(f"Arguments: {user_input_needed.function_call.arguments}")

        approval_message = ChatMessage(
            role=Role.USER, contents=[user_input_needed.create_response(True)]
        )

        response = await agent.run(
            [
                "can you turn off all the lights?",
                ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed]),
                approval_message,
            ],
            response_format=LightInfo,
        )
        counter.add(1, {"func": "change"})
        if response.value:
            light = response.value
            logging.info(
                f"change light state result: light {light.id}:{light.name} is {'on' if light.is_on else 'off'}"
            )
            print(light)
            return f"Light {light.id}:{light.name} is {'on' if light.is_on else 'off'}"
        else:
            logging.info("No structured data found in response")
            return "No structured data found in response"


async def run(query: str) -> str:
    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


async def pack_run():
    async for light_status in get_lights():
        print(light_status)


if __name__ == "__main__":
    otel.otel_configure()
    with get_tracer().start_as_current_span(
        "light_agent_span", kind=trace.SpanKind.CLIENT
    ) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")
        # asyncio.run(pack_run())
        asyncio.run(change_light_state())
