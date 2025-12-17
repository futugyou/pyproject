import logging
from agent_framework import agent_middleware,AgentRunContext
 
@agent_middleware
async def logging_agent_middleware(
    context: AgentRunContext,
    next: Callable[[AgentRunContext], Awaitable[None]],
) -> None:
    logging.info(f"Agent:{context.agent.name} starting...")
    await next(context)
    logging.info(f"Agent:{context.agent.name} finished!")
