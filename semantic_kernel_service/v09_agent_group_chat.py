import asyncio
from semantic_kernel import Kernel
from semantic_kernel.agents import (
    AgentGroupChat,
    ChatCompletionAgent,
    ChatHistoryAgentThread,
)
from semantic_kernel.agents.strategies import (
    TerminationStrategy,
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import (
    kernel_function,
    KernelArguments,
    KernelFunctionFromPrompt,
)
from openai import AsyncOpenAI
from typing import Annotated
import os
from dotenv import load_dotenv

load_dotenv()


def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
    kernel = Kernel()
    chat_completion_service = OpenAIChatCompletion(
        ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID"),
        service_id=service_id,
        async_client=AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_URL"),
        ),
    )

    kernel.add_service(chat_completion_service)
    return kernel


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        last_message = history[-1].content.lower()
        return "approved" in last_message and "not approved" not in last_message


REVIEWER_NAME = "ArtDirector"
REVIEWER_INSTRUCTIONS = """
You are an art director who has opinions about copywriting born of a love for David Ogilvy.
The goal is to determine if the given copy is acceptable to print.
If so, state that it is approved.
If not, provide insight on how to refine suggested copy without example.
"""

COPYWRITER_NAME = "CopyWriter"
COPYWRITER_INSTRUCTIONS = """
You are a copywriter with ten years of experience and are known for brevity and a dry humor.
The goal is to refine and decide on the single best copy as an expert in the field.
Only provide a single proposal per response.
You're laser focused on the goal at hand.
Don't waste time with chit chat.
Consider suggestions when refining an idea.
"""

TASK = "a slogan for a new line of electric cars."

termination_function = KernelFunctionFromPrompt(
    function_name="termination",
    prompt="""
    Determine if the copy has been approved.  If so, respond with a single word: yes
    History:
    {{$history}}
    """,
)

selection_function = KernelFunctionFromPrompt(
    function_name="selection",
    prompt=f"""
    Determine which participant takes the next turn in a conversation based on the the most recent participant.
    State only the name of the participant to take the next turn.
    No participant should take more than one turn in a row.
    
    Choose only from these participants:
    - {REVIEWER_NAME}
    - {COPYWRITER_NAME}
    
    Always follow these rules when selecting the next participant:
    - After user input, it is {COPYWRITER_NAME}'s turn.
    - After {COPYWRITER_NAME} replies, it is {REVIEWER_NAME}'s turn.
    - After {REVIEWER_NAME} provides feedback, it is {COPYWRITER_NAME}'s turn.
    History:
    {{{{$history}}}}
    """,
)


async def main():
    # 1. Create the reviewer agent based on the chat completion service
    agent_reviewer = ChatCompletionAgent(
        kernel=_create_kernel_with_chat_completion("artdirector"),
        name=REVIEWER_NAME,
        instructions=REVIEWER_INSTRUCTIONS,
    )

    # 2. Create the copywriter agent based on the chat completion service
    agent_writer = ChatCompletionAgent(
        kernel=_create_kernel_with_chat_completion("copywriter"),
        name=COPYWRITER_NAME,
        instructions=COPYWRITER_INSTRUCTIONS,
    )

    # 3. Place the agents in a group chat with a custom termination strategy
    group_chat = AgentGroupChat(
        agents=[agent_writer, agent_reviewer],
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_reviewer],
            function=termination_function,
            kernel=_create_kernel_with_chat_completion("termination"),
            result_parser=lambda result: str(result.value[0]).lower() == "yes",
            history_variable_name="history",
            maximum_iterations=6,
        ),
        # termination_strategy=ApprovalTerminationStrategy(
        #     agents=[agent_reviewer],
        #     maximum_iterations=6,
        # ),
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel=_create_kernel_with_chat_completion("selection"),
            result_parser=lambda result: str(result.value[0])
            if result.value is not None
            else COPYWRITER_NAME,
            agent_variable_name="agents",
            history_variable_name="history",
        ),
    )

    # 4. Add the task as a message to the group chat
    await group_chat.add_chat_message(message=TASK)
    print(f"# User: {TASK}")

    # 5. Invoke the chat
    async for content in group_chat.invoke():
        print(f"# {content.name}: {content.content}")


if __name__ == "__main__":
    asyncio.run(main())
