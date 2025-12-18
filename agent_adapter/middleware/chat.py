import logging
from agent_framework import (
    ChatContext,
    ChatMessage,
    ChatMiddleware,
    ChatResponse,
    Role,
    chat_middleware,
)


class LoggingChatMiddleware(ChatMiddleware):
    async def process(
        self,
        context: ChatContext,
        next: Callable[[ChatContext], Awaitable[None]],
    ) -> None:
        logging.info("[LoggingChatMiddleware] Processing start...")

        for i, message in enumerate(context.messages):
            content = message.text if message.text else str(message.contents)
            logging.info(f"[LoggingChatMiddleware] Message {i + 1} ({message.role.value}): {content}")

        logging.info(f"[LoggingChatMiddleware] Total messages: {len(context.messages)}")

        await next(context)

        logging.info("[LoggingChatMiddleware] Processing completed")


@chat_middleware
async def logging_chat_middleware(
    context: ChatContext,
    next: Callable[[ChatContext], Awaitable[None]],
) -> None:
    logging.info("[logging_chat_middleware] Processing input...")

    for i, message in enumerate(context.messages):
        content = message.text if message.text else str(message.contents)
        logging.info(f"[LoggingChatMiddleware] Message {i + 1} ({message.role.value}): {content}")

    await next(context)

    logging.info("[logging_chat_middleware] Processing completed")
