from typing import Dict, List
from pydantic import BaseModel
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from google.adk.agents import ParallelAgent, LlmAgent, BaseAgent, SequentialAgent
from google.adk.tools import ToolContext, google_search
from google.adk.models.google_llm import BaseLlm

from adk_adapter import client_factory

SESSION_ID_BASE = "loop_exit_tool_session"
STATE_INITIAL_TOPIC = "initial_topic"
STATE_CURRENT_DOC = "current_document"
STATE_CRITICISM = "criticism"

COMPLETION_PHRASE = "No major issues found."


def researcher_agent_1(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="RenewableEnergyResearcher",
        model=llm,
        instruction="""You are an AI Research Assistant specializing in energy.
    Research the latest advancements in 'renewable energy sources'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches renewable energy sources.",
        tools=[google_search],
        # Store result in state for the merger agent
        output_key="renewable_energy_result",
    )


def researcher_agent_2(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="EVResearcher",
        model=llm,
        instruction="""You are an AI Research Assistant specializing in transportation.
    Research the latest developments in 'electric vehicle technology'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches electric vehicle technology.",
        tools=[google_search],
        # Store result in state for the merger agent
        output_key="ev_technology_result",
    )


def researcher_agent_3(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="CarbonCaptureResearcher",
        model=llm,
        instruction="""You are an AI Research Assistant specializing in climate solutions.
        Research the current state of 'carbon capture methods'.
        Use the Google Search tool provided.
        Summarize your key findings concisely (1-2 sentences).
        Output *only* the summary.
        """,
        description="Researches carbon capture methods.",
        tools=[google_search],
        # Store result in state for the merger agent
        output_key="carbon_capture_result",
    )


def parallel_research_agent(llm: BaseLlm) -> BaseAgent:
    return ParallelAgent(
        name="ParallelWebResearchAgent",
        sub_agents=[
            researcher_agent_1(llm),
            researcher_agent_2(llm),
            researcher_agent_3(llm),
        ],
        description="Runs multiple research agents in parallel to gather information.",
    )


def merger_agent(llm: BaseLlm) -> BaseAgent:
    return LlmAgent(
        name="SynthesisAgent",
        model=llm,  # Or potentially a more powerful model if needed for synthesis
        instruction="""You are an AI Assistant responsible for combining research findings into a structured report.

    Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.

    **Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not present in these specific summaries.**

    **Input Summaries:**

    *   **Renewable Energy:**
        {renewable_energy_result}

    *   **Electric Vehicles:**
        {ev_technology_result}

    *   **Carbon Capture:**
        {carbon_capture_result}

    **Output Format:**

    ## Summary of Recent Sustainable Technology Advancements

    ### Renewable Energy Findings
    (Based on RenewableEnergyResearcher's findings)
    [Synthesize and elaborate *only* on the renewable energy input summary provided above.]

    ### Electric Vehicle Findings
    (Based on EVResearcher's findings)
    [Synthesize and elaborate *only* on the EV input summary provided above.]

    ### Carbon Capture Findings
    (Based on CarbonCaptureResearcher's findings)
    [Synthesize and elaborate *only* on the carbon capture input summary provided above.]

    ### Overall Conclusion
    [Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

    Output *only* the structured report following this format. Do not include introductory or concluding phrases outside this structure, and strictly adhere to using only the provided input summary content.
    """,
        description="Combines research findings from parallel agents into a structured, cited report, strictly grounded on provided inputs.",
    )


def sequential_pipeline_agent(llm: BaseLlm) -> BaseAgent:
    return SequentialAgent(
        name="ResearchAndSynthesisPipeline",
        sub_agents=[
            parallel_research_agent(llm),
            merger_agent(llm),
        ],
        description="Coordinates parallel research and synthesizes the results.",
    )


if __name__ == "__main__":
    import asyncio
    from adk_adapter import adkutil

    llm = client_factory.build_llm()
    base_agent = sequential_pipeline_agent(llm)
    asyncio.run(
        adkutil.run_agent(
            base_agent,
            "Summarize recent sustainable tech advancements.",
        )
    )
