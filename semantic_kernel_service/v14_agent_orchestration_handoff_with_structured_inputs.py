import asyncio
from enum import Enum
from pydantic import BaseModel
from semantic_kernel.agents import (
    Agent,
    ChatCompletionAgent,
    HandoffOrchestration,
    OrchestrationHandoffs,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import (
    AuthorRole,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)
from semantic_kernel.functions import kernel_function


class GitHubLabels(Enum):
    """Enum representing GitHub labels."""

    PYTHON = "python"
    DOTNET = ".NET"
    BUG = "bug"
    ENHANCEMENT = "enhancement"
    QUESTION = "question"
    VECTORSTORE = "vectorstore"
    AGENT = "agent"


class GithubIssue(BaseModel):
    """Model representing a GitHub issue."""

    id: str
    title: str
    body: str
    labels: list[str] = []


class Plan(BaseModel):
    """Model representing a plan for resolving a GitHub issue."""

    tasks: list[str]


class GithubPlugin:
    """Plugin for GitHub related operations."""

    @kernel_function
    async def add_labels(self, issue_id: str, labels: list[GitHubLabels]) -> None:
        """Add labels to a GitHub issue."""
        await asyncio.sleep(1)  # Simulate network delay
        print(f"Adding labels {labels} to issue {issue_id}")

    @kernel_function(description="Create a plan to resolve the issue.")
    async def create_plan(self, issue_id: str, plan: Plan) -> None:
        """Create tasks for a GitHub issue."""
        await asyncio.sleep(1)  # Simulate network delay
        print(
            f"Creating plan for issue {issue_id} with tasks:\n{plan.model_dump_json(indent=2)}"
        )


def get_agents() -> tuple[list[Agent], OrchestrationHandoffs]:
    from service import chat_completion_service

    triage_agent = ChatCompletionAgent(
        name="TriageAgent",
        description="An agent that triages GitHub issues",
        instructions="Given a GitHub issue, triage it.",
        service=chat_completion_service,
    )
    python_agent = ChatCompletionAgent(
        name="PythonAgent",
        description="An agent that handles Python related issues",
        instructions="You are an agent that handles Python related GitHub issues.",
        service=chat_completion_service,
        plugins=[GithubPlugin()],
    )
    dotnet_agent = ChatCompletionAgent(
        name="DotNetAgent",
        description="An agent that handles .NET related issues",
        instructions="You are an agent that handles .NET related GitHub issues.",
        service=chat_completion_service,
        plugins=[GithubPlugin()],
    )

    # Define the handoff relationships between agents
    handoffs = {
        triage_agent.name: {
            python_agent.name: "Transfer to this agent if the issue is Python related",
            dotnet_agent.name: "Transfer to this agent if the issue is .NET related",
        },
    }

    return [triage_agent, python_agent, dotnet_agent], handoffs


GithubIssueSample = GithubIssue(
    id="12345",
    title=(
        "Bug: SQLite Error 1: 'ambiguous column name:' when including VectorStoreRecordKey in "
        "VectorSearchOptions.Filter"
    ),
    body=(
        "Describe the bug"
        "When using column names marked as [VectorStoreRecordData(IsFilterable = true)] in "
        "VectorSearchOptions.Filter, the query runs correctly."
        "However, using the column name marked as [VectorStoreRecordKey] in VectorSearchOptions.Filter, the query "
        "throws exception 'SQLite Error 1: ambiguous column name: StartUTC"
        ""
        "To Reproduce"
        "Add a filter for the column marked [VectorStoreRecordKey]. Since that same column exists in both the "
        "vec_TestTable and TestTable, the data for both columns cannot be returned."
        ""
        "Expected behavior"
        "The query should explicitly list the vec_TestTable column names to retrieve and should omit the "
        "[VectorStoreRecordKey] column since it will be included in the primary TestTable columns."
        ""
        "Platform"
        ""
        "Microsoft.SemanticKernel.Connectors.Sqlite v1.46.0-preview"
        "Additional context"
        "Normal DBContext logging shows only normal context queries. Queries run by VectorizedSearchAsync() don't "
        "appear in those logs and I could not find a way to enable logging in semantic search so that I could "
        "actually see the exact query that is failing. It would have been very useful to see the failing semantic "
        "query."
    ),
    labels=[],
)


# The default input transform will attempt to serialize an object into a string by using
# `json.dump()`. However, an object of a Pydantic model type cannot be directly serialize
# by `json.dump()`. Thus, we will need a custom transform.
def custom_input_transform(input_message: GithubIssue) -> ChatMessageContent:
    return ChatMessageContent(
        role=AuthorRole.USER, content=input_message.model_dump_json()
    )


async def main():
    from service import chat_completion_service

    agents, handoffs = get_agents()
    handoff_orchestration = HandoffOrchestration[GithubIssue, ChatMessageContent](
        members=agents,
        handoffs=handoffs,
        input_transform=custom_input_transform,
    )

    runtime = InProcessRuntime()
    runtime.start()
    handoff_result = await handoff_orchestration.invoke(
        task=GithubIssueSample,
        runtime=runtime,
    )

    try:
        # Attempt to get the result will result in an exception due to cancellation
        value = await handoff_result.get()
        print(f"***** Final Result *****\n{value}")
    except Exception as e:
        print(e)
    finally:
        # 5. Stop the runtime
        await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
