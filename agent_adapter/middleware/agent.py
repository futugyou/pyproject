import logging
import time
from agent_framework import AgentMiddleware, agent_middleware, AgentRunContext


@agent_middleware
async def logging_agent_middleware(
    context: AgentRunContext,
    next: Callable[[AgentRunContext], Awaitable[None]],
) -> None:
    logging.info(f"Agent:{context.agent.name} starting...")
    await next(context)
    logging.info(f"Agent:{context.agent.name} finished!")


class LogginAgentMiddleware(AgentMiddleware):
    async def process(
        self,
        context: AgentRunContext,
        next: Callable[[AgentRunContext], Awaitable[None]],
    ) -> None:
        logging.info(f"Agent:{context.agent.name} starting...")

        context.metadata["some_tag"] = get_tracer().get_current_span().context.trace_id

        start_time = time.time()

        try:
            await next(context)
        except Exception as e:
            logging.info(f"Agent:{context.agent.name} Caught Exception: {e}")
            # Override function result to provide custom message in response.
            context.result = ("Sorry for the inconvenience, please try again later.",)

        end_time = time.time()
        duration = end_time - start_time

        logging.info(f"Agent:{context.agent.name} completed in {duration:.5f}s.")
