# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import os


from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIAudioToText,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import AudioContent, ChatHistory

logging.basicConfig(level=logging.WARNING)
AUDIO_FILEPATH = os.path.join(os.path.dirname(__file__), "output.wav")

system_message = """
You are a chat bot. Your name is Mosscap and
you have one goal: figure out what people need.
Your full name, should you need to know it, is
Splendid Speckled Mosscap. You communicate
effectively, but you tend to answer with long
flowery prose.
"""

history = ChatHistory()
history.add_user_message("Hi there, who are you?")
history.add_assistant_message(
    "I am Mosscap, a chat bot. I'm trying to figure out what people need."
)


# GitHub models do not have this feature, and Gemini needs use complete
async def chat(
    chat_service: OpenAIChatCompletion, audio_to_text_service: OpenAIAudioToText
) -> bool:
    try:
        user_input = await audio_to_text_service.get_text_content(
            AudioContent.from_audio_file(AUDIO_FILEPATH)
        )
        print(user_input.text)
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if "exit" in user_input.text.lower():
        print("\n\nExiting chat...")
        return False

    history.add_user_message(user_input.text)

    chunks = chat_service.get_streaming_chat_message_content(
        chat_history=history,
        settings=OpenAIChatPromptExecutionSettings(
            max_tokens=2000,
            temperature=0.7,
            top_p=0.8,
        ),
    )

    print("Mosscap:> ", end="")
    answer = ""
    async for message in chunks:
        print(str(message), end="")
        answer += str(message)
    print("\n")

    history.add_assistant_message(str(answer))

    return True


async def main() -> None:
    from ..service import build_kernel_pipeline

    kernel = build_kernel_pipeline()
    chat_completion_service = kernel.get_service("default")
    audio_to_text_service = kernel.get_service("audio_to_text")

    print(
        "Instruction: when it's your turn to speak, press the spacebar to start recording."
        " Release the spacebar to stop recording."
    )
    chatting = True
    while chatting:
        chatting = await chat(chat_completion_service, audio_to_text_service)


if __name__ == "__main__":
    asyncio.run(main())
