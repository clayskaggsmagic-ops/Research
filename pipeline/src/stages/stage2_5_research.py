"""
Stage 2.5 — Background Research Agent (Bosse et al. A.10)

A SEPARATE ReAct agent that gathers comprehensive research for each
proto-question BEFORE refinement. Without thorough research, LLMs
generate plausible-but-invalid questions.

For each proto-question, the agent produces a structured research brief:
  - Background and context (status quo as of simulation_date)
  - Data and information (relevant sources, publication schedules)
  - Recent numbers and events (current values, base rates)
  - Trends (factors for YES vs NO resolution)

It also flags questions that should be dropped:
  - Event already happened before simulation_date
  - Resolution source doesn't exist
  - Question is based on hallucinated facts

Entry point: run_stage2_5(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.schemas import PipelineState, Question
from src.tools import get_stage2_tools  # same web_search + extract tools

logger = logging.getLogger(__name__)


# ── Structured Output Schema ──────────────────────────────────────────────────


class ResearchBrief(BaseModel):
    """Structured research output for a single proto-question."""

    background_and_context: str = Field(
        description=(
            "Status quo as of simulation_date, historical context, "
            "recent developments, key stakeholders, regulatory/legal framework."
        )
    )
    data_and_information: str = Field(
        description=(
            "Relevant data sources and how to access them, "
            "publication schedules, where new information will appear."
        )
    )
    recent_numbers_and_events: str = Field(
        description=(
            "For threshold questions: current number. "
            "What the outcome would have been in prior periods. "
            "Relevant projections, forecasts, or base rates."
        )
    )
    trends: str = Field(
        description=(
            "Factors that make YES resolution more likely. "
            "Factors that make NO resolution more likely."
        )
    )
    sources_cited: list[str] = Field(
        description="URLs and source names referenced in the research"
    )


class ResearchFlags(BaseModel):
    """Flags for problems found during research."""

    already_resolved: bool = Field(
        default=False,
        description="True if the event has already happened before simulation_date",
    )
    resolution_source_missing: bool = Field(
        default=False,
        description="True if the proposed resolution source doesn't actually exist",
    )
    hallucinated_facts: bool = Field(
        default=False,
        description="True if the question is based on facts that don't exist",
    )
    flag_details: str = Field(
        default="",
        description="Explanation of any flags raised",
    )


class ResearchResponse(BaseModel):
    """Full structured output from the research agent."""

    question_id: str = Field(description="The question this research is for")
    research_brief: ResearchBrief
    flags: ResearchFlags
    overall_quality_assessment: str = Field(
        description=(
            "Brief assessment: is this question viable for a forecasting tournament? "
            "Rate as VIABLE, NEEDS_REVISION, or DROP."
        )
    )


# ── Agent Prompt ──────────────────────────────────────────────────────────────


RESEARCH_AGENT_SYSTEM_PROMPT = """\
You are a meticulous research analyst working for a forecasting tournament. \
Your job is to gather comprehensive background research on a proto-question \
so that it can be refined into a high-quality forecasting question.

## CRITICAL: Today's Date
Today is **{simulation_date}**. You do NOT know what will happen after this \
date. All your research should capture the state of the world AS OF \
{simulation_date}. Do NOT include any information from after this date.

## The Question Under Review
Question ID: {question_id}
Title: {title}
Type: {question_type}
Question: {question_text}
Background (draft): {background}
{options_block}
Domain: {domain}

## Your Research Task

You must produce a thorough, well-sourced research brief covering FOUR areas:

### 1. BACKGROUND AND CONTEXT
- What is the status quo as of {simulation_date}?
- What is the historical context and what are the recent developments?
- Who are the key stakeholders and players involved?
- What is the regulatory/legal framework, if applicable?

### 2. DATA AND INFORMATION
- What data sources are relevant? How can they be accessed?
- When has past data been published? Is there a publication schedule?
- Where should we expect new information to become available?

### 3. RECENT NUMBERS AND EVENTS
- For threshold questions ("will X exceed Y"): what is the current number?
- What would the outcome have been in prior periods?
- Are there relevant projections, forecasts, or base rates?

### 4. TRENDS
- List factors that make a YES resolution more likely
- List factors that make a NO resolution more likely

## QUALITY FLAGS — Check for Problems
You MUST also check whether this question has any of these problems:
- **Already resolved**: Has the event already happened before {simulation_date}?
- **Resolution source missing**: Does the proposed way to check the answer \
actually exist? (e.g., does the agency publish the relevant data?)
- **Hallucinated facts**: Is the question based on facts that don't exist?

If you find ANY of these problems, flag them clearly.

## Research Standards
- Include links and source names for EVERYTHING you cite
- All information must be easily digestible
- This is important work, and any mistakes could be hugely embarrassing
- Be THOROUGH — better to over-research than under-research\
"""


# ── Run Research Agent for One Question ──────────────────────────────────────


async def run_research_agent(
    question: Question,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.4,
) -> tuple[str, list[str], str]:
    """
    Run the background research agent for a single question.

    Returns:
        (research_brief_text, flags_list, quality_assessment)
    """
    tools = get_stage2_tools()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        response_format=ResearchResponse,
    )

    # Format options block
    options_block = ""
    if question.options:
        opts = "\n".join(f"  {chr(65+i)}. {o}" for i, o in enumerate(question.options))
        options_block = f"Options:\n{opts}"

    prompt = RESEARCH_AGENT_SYSTEM_PROMPT.format(
        simulation_date=question.simulation_date,
        question_id=question.question_id,
        title=question.title,
        question_type=question.question_type.value,
        question_text=question.question_text,
        background=question.background,
        options_block=options_block,
        domain=question.domain.value,
    )

    logger.info(
        "Stage 2.5: researching question %s (domain=%s)",
        question.question_id, question.domain.value,
    )

    try:
        result = await agent.ainvoke(
            {"messages": [("system", prompt), ("human", "Begin your research now. Be thorough.")]},
        )

        structured: ResearchResponse | None = result.get("structured_response")
        if not structured:
            logger.warning("Question %s: no structured response from research agent", question.question_id)
            return ("", [], "NEEDS_REVISION")

        # Assemble the research brief as formatted text
        brief = structured.research_brief
        research_text = (
            f"## Background and Context\n{brief.background_and_context}\n\n"
            f"## Data and Information\n{brief.data_and_information}\n\n"
            f"## Recent Numbers and Events\n{brief.recent_numbers_and_events}\n\n"
            f"## Trends\n{brief.trends}\n\n"
            f"## Sources\n" + "\n".join(f"- {s}" for s in brief.sources_cited)
        )

        # Collect flags
        flags: list[str] = []
        if structured.flags.already_resolved:
            flags.append(f"ALREADY_RESOLVED: {structured.flags.flag_details}")
        if structured.flags.resolution_source_missing:
            flags.append(f"RESOLUTION_SOURCE_MISSING: {structured.flags.flag_details}")
        if structured.flags.hallucinated_facts:
            flags.append(f"HALLUCINATED_FACTS: {structured.flags.flag_details}")

        logger.info(
            "Question %s: research complete. Quality: %s. Flags: %s",
            question.question_id,
            structured.overall_quality_assessment,
            flags if flags else "none",
        )

        return (research_text, flags, structured.overall_quality_assessment)

    except Exception:
        logger.exception("Question %s: research agent failed", question.question_id)
        return ("", ["AGENT_FAILURE"], "NEEDS_REVISION")


# ── Entry Point ──────────────────────────────────────────────────────────────


async def run_stage2_5(state: PipelineState) -> PipelineState:
    """
    Stage 2.5 entry point — run as a LangGraph node.

    1. For each proto-question, run the research agent
    2. Enrich the question with background_research
    3. Drop questions flagged as DROP
    4. Store results in state.researched_questions
    """
    config = state.config
    model_name = config.drafter_model
    temperature = 0.4  # lower temp for factual research

    logger.info(
        "Stage 2.5 starting — %d proto-questions to research, model=%s",
        len(state.proto_questions), model_name,
    )

    researched: list[Question] = []
    dropped = 0

    for question in state.proto_questions:
        research_text, flags, quality = await run_research_agent(
            question=question,
            model_name=model_name,
            temperature=temperature,
        )

        # Enrich the question
        question.background_research = research_text
        question.research_flags = flags

        if quality == "DROP":
            logger.warning(
                "Question %s DROPPED — flags: %s",
                question.question_id, flags,
            )
            dropped += 1
            continue

        researched.append(question)
        logger.info(
            "Question %s researched (quality=%s, %d flags). Total: %d",
            question.question_id, quality, len(flags), len(researched),
        )

    state.researched_questions = researched

    logger.info(
        "Stage 2.5 complete — %d researched, %d dropped (from %d total)",
        len(researched), dropped, len(state.proto_questions),
    )

    return state
