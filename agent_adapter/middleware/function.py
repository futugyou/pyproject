import logging
import time
from agent_framework import FunctionMiddleware,function_middleware, FunctionInvocationContext


@function_middleware
async def logging_function_middleware(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    logging.info(f"Function:{context.function.name} starting...")
    await next(context)
    logging.info(f"Function:{context.function.name} finished!")


class LoggingFunctionMiddleware(FunctionMiddleware):
    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        function_name = context.function.name
        logging.info(f"Function:{function_name} starting...")

        start_time = time.time()

        await next(context)

        end_time = time.time()
        duration = end_time - start_time

        logging.info(f"Function {function_name} completed in {duration:.5f}s.")