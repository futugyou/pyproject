from typing import Dict, List
from pydantic import BaseModel
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent, SequentialAgent
from google.adk.tools import ToolContext
from google.adk.models.google_llm import BaseLlm

from adk_adapter import client_factory

SESSION_ID_BASE = "loop_exit_tool_session"
STATE_INITIAL_TOPIC = "initial_topic"
STATE_CURRENT_DOC = "current_document"
STATE_CRITICISM = "criticism"

COMPLETION_PHRASE = "No major issues found."


def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    # Return empty dict as tools should typically return JSON-serializable output
    return {}


def initial_writer_agent(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="InitialWriterAgent",
        model=llm,
        include_contents="none",
        instruction=f"""You are a Creative Writing Assistant tasked with starting a story.
        Write the *first draft* of a short story (aim for 2-4 sentences).
        Base the content *only* on the topic provided below. Try to introduce a specific element (like a character, a setting detail, or a starting action) to make it engaging.
        Topic: {{initial_topic}}

        Output *only* the story/document text. Do not add introductions or explanations.
    """,
        description="Writes the initial document draft based on the topic, aiming for some initial substance.",
        output_key=STATE_CURRENT_DOC,
    )


def critic_agent_in_loop(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="CriticAgent",
        model=llm,
        include_contents="none",
        instruction=f"""You are a Constructive Critic AI reviewing a short document draft (typically 2-6 sentences). Your goal is balanced feedback.

        **Document to Review:**
        ```
        {{current_document}}
        ```

        **Task:**
        Review the document for clarity, engagement, and basic coherence according to the initial topic (if known).

        IF you identify 1-2 *clear and actionable* ways the document could be improved to better capture the topic or enhance reader engagement (e.g., "Needs a stronger opening sentence", "Clarify the character's goal"):
        Provide these specific suggestions concisely. Output *only* the critique text.

        ELSE IF the document is coherent, addresses the topic adequately for its length, and has no glaring errors or obvious omissions:
        Respond *exactly* with the phrase "{COMPLETION_PHRASE}" and nothing else. It doesn't need to be perfect, just functionally complete for this stage. Avoid suggesting purely subjective stylistic preferences if the core is sound.

        Do not add explanations. Output only the critique OR the exact completion phrase.
    """,
        description="Reviews the current draft, providing critique if clear improvements are needed, otherwise signals completion.",
        output_key=STATE_CRITICISM,
    )


def code_refactorer_agent(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="CodeRefactorerAgent",
        model=llm,
        # Change 3: Improved instruction, correctly using state key injection
        instruction="""You are a Python Code Refactoring AI.
Your goal is to improve the given Python code based on the provided review comments.

  **Original Code:**
  ```python
  {generated_code}
  ```

  **Review Comments:**
  {review_comments}

**Task:**
Carefully apply the suggestions from the review comments to refactor the original code.
If the review comments state "No major issues found," return the original code unchanged.
Ensure the final code is complete, functional, and includes necessary imports and docstrings.

**Output:**
Output *only* the final, refactored Python code block, enclosed in triple backticks (```python ... ```). 
Do not add any other text before or after the code block.
""",
        description="Refactors code based on review comments.",
        output_key="refactored_code",  # Stores output in state['refactored_code']
    )


def refiner_agent_in_loop(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="RefinerAgent",
        model=llm,
        # Relies solely on state via placeholders
        include_contents="none",
        instruction=f"""You are a Creative Writing Assistant refining a document based on feedback OR exiting the process.
        **Current Document:**
        ```
        {{current_document}}
        ```
        **Critique/Suggestions:**
        {{criticism}}

        **Task:**
        Analyze the 'Critique/Suggestions'.
        IF the critique is *exactly* "{COMPLETION_PHRASE}":
        You MUST call the 'exit_loop' function. Do not output any text.
        ELSE (the critique contains actionable feedback):
        Carefully apply the suggestions to improve the 'Current Document'. Output *only* the refined document text.

        Do not add explanations. Either output the refined document OR call the exit_loop function.
    """,
        description="Refines the document based on critique, or calls exit_loop if critique indicates completion.",
        tools=[exit_loop],  # Provide the exit_loop tool
        output_key=STATE_CURRENT_DOC,  # Overwrites state['current_document'] with the refined version
    )


def refinement_loop(llm: BaseLlm) -> BaseAgent:
    return LoopAgent(
        name="RefinementLoop",
        sub_agents=[
            critic_agent_in_loop(llm),
            refiner_agent_in_loop(llm),
        ],
        max_iterations=5,  # Limit loops
    )


def root_agent(llm: BaseLlm) -> BaseAgent:
    return SequentialAgent(
        name="IterativeWritingPipeline",
        sub_agents=[
            initial_writer_agent(llm),
            refinement_loop(llm),
        ],
        description="Writes an initial document and then iteratively refines it with critique using an exit tool.",
    )


if __name__ == "__main__":
    import asyncio
    from adk_adapter import adkutil

    llm = client_factory.build_llm()
    base_agent = root_agent(llm)
    asyncio.run(
        adkutil.run_agent(
            base_agent,
            "a robot developing unexpected emotions",
            initial_state={
                STATE_INITIAL_TOPIC: "a robot developing unexpected emotions"
            },
        )
    )
