from typing import Dict, List
from pydantic import BaseModel
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import ToolContext
from google.adk.models.google_llm import BaseLlm
from google.adk.agents.sequential_agent import SequentialAgent

from adk_adapter import client_factory


def code_writer_agent(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="CodeWriterAgent",
        model=llm,
        # Change 3: Improved instruction
        instruction="""You are a Python Code Generator.
Based *only* on the user's request, write Python code that fulfills the requirement.
Output *only* the complete Python code block, enclosed in triple backticks (```python ... ```). 
Do not add any other text before or after the code block.
""",
        description="Writes initial Python code based on a specification.",
        output_key="generated_code",  # Stores output in state['generated_code']
    )


def code_reviewer_agent(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="CodeReviewerAgent",
        model=llm,
        # Change 3: Improved instruction, correctly using state key injection
        instruction="""You are an expert Python Code Reviewer. 
    Your task is to provide constructive feedback on the provided code.

    **Code to Review:**
    ```python
    {generated_code}
    ```

**Review Criteria:**
1.  **Correctness:** Does the code work as intended? Are there logic errors?
2.  **Readability:** Is the code clear and easy to understand? Follows PEP 8 style guidelines?
3.  **Efficiency:** Is the code reasonably efficient? Any obvious performance bottlenecks?
4.  **Edge Cases:** Does the code handle potential edge cases or invalid inputs gracefully?
5.  **Best Practices:** Does the code follow common Python best practices?

**Output:**
Provide your feedback as a concise, bulleted list. Focus on the most important points for improvement.
If the code is excellent and requires no changes, simply state: "No major issues found."
Output *only* the review comments or the "No major issues" statement.
""",
        description="Reviews code and provides feedback.",
        output_key="review_comments",  # Stores output in state['review_comments']
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


def code_pipeline_agent(llm: BaseLlm) -> BaseAgent:
    return SequentialAgent(
        name="CodePipelineAgent",
        sub_agents=[
            code_writer_agent(llm),
            code_reviewer_agent(llm),
            code_refactorer_agent(llm),
        ],
        description="Executes a sequence of code writing, reviewing, and refactoring.",
        # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
    )


if __name__ == "__main__":
    import asyncio
    from adk_adapter import adkutil

    llm = client_factory.build_llm()
    base_agent = code_pipeline_agent(llm)
    asyncio.run(
        adkutil.run_agent(
            base_agent, "Write a double-checked locking example using python."
        )
    )
