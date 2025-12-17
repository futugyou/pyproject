import logging
from agent_framework import function_middleware, FunctionInvocationContext


@function_middleware
async def logging_function_middleware(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    logging.info(f"Function:{context.function.name} starting...")
    await next(context)
    logging.info(f"Function:{context.function.name} finished!")
