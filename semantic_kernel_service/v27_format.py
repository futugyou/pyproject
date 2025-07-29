# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json

from pydantic import BaseModel, ConfigDict

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory, StreamingChatMessageContent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

from service import kernel, chat_completion_service

system_message = """
You are a helpful math tutor. Guide the user through the solution step by step.
"""


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str
    output: str


class Reasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Step]
    final_answer: str


request_settings = OpenAIChatPromptExecutionSettings(
    service_id="default",
    max_tokens=2000,
    temperature=0.7,
    top_p=0.8,
    function_choice_behavior=FunctionChoiceBehavior.Auto(
        filters={"excluded_plugins": ["chat"]}
    ),
    response_format=Reasoning,
)

chat_function = kernel.add_function(
    prompt=system_message + """{{$chat_history}}""",
    function_name="chat",
    plugin_name="chat",
    prompt_execution_settings=request_settings,
)

history = ChatHistory()
history.add_user_message("how can I solve 8x + 7y = -23, and 4x=12?")


async def main():
    stream = True
    if stream:
        answer = kernel.invoke_stream(
            chat_function,
            chat_history=history,
        )
        print("Mosscap:> ", end="")
        result_content: list[StreamingChatMessageContent] = []
        async for message in answer:
            result_content.append(message[0])
            print(str(message[0]), end="", flush=True)
        if result_content:
            result = "".join([str(content) for content in result_content])
    else:
        result = await kernel.invoke(
            chat_function,
            chat_history=history,
        )
        reasoned_result = Reasoning.model_validate(json.loads(result.value[0].content))
        print(f"{reasoned_result.model_dump_json(indent=4)}")
        history.add_assistant_message(str(result))


if __name__ == "__main__":
    asyncio.run(main())
