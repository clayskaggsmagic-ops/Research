"""
Stage 3 — Refinement Agent (Bosse et al. A.9)

A SEPARATE agent from the question writer. It adds precise, objective
resolution criteria WITHOUT changing the question substance.

For each researched question, the agent adds:
  - resolution_criteria: Exact YES/NO conditions (unambiguous)
  - resolution_source: Specific database/site
  - fine_print: Edge cases
  - base_rate_estimate + base_rate_reasoning
  - resolution_date: After simulation_date

Key rules from A.9:
  - ALL relevant terms clearly defined with links
  - Resolution source must unambiguously tell us the outcome
  - "Pars pro toto" — measurable indicators as proxies
  - Include timezones, set start AND end dates
  - Resolvable by a human in ~10 minutes
  - Avoid "will X happen before Y" — use fixed dates

Entry point: run_stage3(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.schemas import PipelineState, Question
from src.tools import get_stage2_tools

logger = logging.getLogger(__name__)


# ── Structured Output Schema ──────────────────────────────────────────────────


class RefinedResolution(BaseModel):
    """Resolution infrastructure produced by the refinement agent."""

    resolution_criteria: str = Field(
        description=(
            "Exact, unambiguous conditions for resolution. For binary: precise YES/NO "
            "conditions. For action_selection: how to match each option to the outcome. "
            "A stranger must be able to resolve it without judgment calls."
        )
    )
    resolution_source: str = Field(
        description=(
            "Specific database, website, or official record that will contain the "
            "answer. Examples: Federal Register, USTR press releases, OFAC SDN List, "
            "Congressional Record, archived Truth Social posts, AP/Reuters reporting."
        )
    )
    fine_print: str = Field(
        description=(
            "Edge case handling: what happens if action is partial, reversed, done "
            "via subordinate, resolution source is unavailable, government shutdown "
            "delays publication, etc."
        )
    )
    resolution_date: str = Field(
        description=(
            "YYYY-MM-DD — the date by which this question resolves. Must be AFTER "
            "the simulation_date. Include timezone (assume ET unless specified). "
            "Should be a fixed calendar date, not relative to another event."
        )
    )
    base_rate_estimate: float = Field(
        description="Historical frequency 0.0-1.0. How often has similar action occurred?"
    )
    base_rate_reasoning: str = Field(
        description=(
            "Explain the base rate: what historical precedents were examined, "
            "how many times similar actions occurred, time period studied."
        )
    )


class RefinementResponse(BaseModel):
    """Full structured output from the refinement agent for one question."""

    question_id: str = Field(description="The question being refined")
    resolution: RefinedResolution
    question_text_revision: str | None = Field(
        default=None,
        description=(
            "ONLY if the question text needs minor clarification for resolvability "
            "(e.g., adding a timezone or specific threshold). Otherwise null — "
            "do NOT change the substance of the question."
        ),
    )
    title_revision: str | None = Field(
        default=None,
        description="Minor title clarification if needed, otherwise null.",
    )


# ── Agent Prompt ──────────────────────────────────────────────────────────────


REFINEMENT_SYSTEM_PROMPT = """\
You are a senior Metaculus question editor. Your job is to take a draft \
forecasting question and add precise, objective resolution criteria. You must \
NOT change the substance of the question — only add the infrastructure needed \
to resolve it unambiguously.

## CRITICAL: Today's Date
Today is **{simulation_date}**. You do NOT know what will happen after this \
date. You are refining a question that will be asked on {simulation_date}.

## The Question to Refine
Question ID: {question_id}
Title: {title}
Type: {question_type}
Question: {question_text}
Background: {background}
{options_block}
Domain: {domain}

## Background Research (from Stage 2.5)
{background_research}

## Your Task: Add Resolution Infrastructure

### 1. RESOLUTION CRITERIA
Write exact, unambiguous conditions:
- **Binary**: "Resolves YES if [precise condition]. Resolves NO otherwise."
- **Action Selection**: "Resolves as option [X] if [precise condition for X]." \
for each option.
- ALL relevant terms MUST be clearly defined (with links to Wikipedia, \
official organizations, etc.)
- Use "pars pro toto" — find specific measurable indicators as proxies for \
broader concepts. Example: instead of "war", use "10+ missiles fired" or \
"formal declaration of war by Congress"
- The question should be resolvable by a human within approximately 10 minutes \
of checking the resolution source

### 2. RESOLUTION SOURCE
Name the SPECIFIC database, website, or official record:
- Federal Register (federalregister.gov)
- USTR press releases (ustr.gov)
- OFAC SDN List (treasury.gov/ofac)
- Congressional Record (congress.gov)
- AP/Reuters reporting
- Other official government publications
Use web_search to VERIFY the source actually exists and publishes this data.

### 3. FINE PRINT (Edge Cases)
Address these scenarios:
- What if the action is partial (e.g., tariffs reduced but not to the \
specified level)?
- What if the action is reversed after being taken?
- What if it's done by a subordinate rather than the President directly?
- What if the resolution source is temporarily unavailable?
- What if a government shutdown delays publication?
- What if the action takes a different legal form than expected?

### 4. BASE RATE
Use web_search to research historical precedents:
- How often has a similar presidential action occurred?
- What is the historical frequency (0.0-1.0)?
- Explain your reasoning with specific examples

### 5. RESOLUTION DATE
Set a specific calendar date (YYYY-MM-DD):
- Must be AFTER {simulation_date}
- Should be when we expect the outcome to be definitively known
- Include timezone assumption (default: 11:59 PM ET)
- Use a FIXED date, never "before event Y happens"
- Avoid "will condition X happen before condition Y" — use fixed dates

## PRESIDENTIAL ATTRIBUTION CHECK
Before refining, verify this question is genuinely about the President's \
personal decision. If Trump's name could be removed and the question would \
still make sense (e.g., "Will the Fed raise rates?" or "Will Congress pass X?"), \
note this concern in the fine_print field: "WARNING: This question may not be \
about a presidential decision." This flags it for Stage 4 rejection.

## RULES
- Do NOT change the question substance — only add resolution infrastructure
- Minor clarifications to question_text are OK (adding timezone, specific \
threshold) but flag them in question_text_revision
- When December 31 comes, there ABSOLUTELY needs to be a resolution source \
that unambiguously tells us the correct outcome\
"""


# ── Run Refinement Agent for One Question ────────────────────────────────────


async def run_refinement_agent(
    question: Question,
    model_name: str = "gemini-3-pro-preview",
    temperature: float = 0.3,
) -> Question:
    """
    Run the refinement agent for a single question.

    Returns the question enriched with resolution fields.
    """
    tools = get_stage2_tools()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        response_format=RefinementResponse,
    )

    # Format options block
    options_block = ""
    if question.options:
        opts = "\n".join(f"  {chr(65 + i)}. {o}" for i, o in enumerate(question.options))
        options_block = f"Options:\n{opts}"

    prompt = REFINEMENT_SYSTEM_PROMPT.format(
        simulation_date=question.simulation_date,
        question_id=question.question_id,
        title=question.title,
        question_type=question.question_type.value,
        question_text=question.question_text,
        background=question.background,
        options_block=options_block,
        domain=question.domain.value,
        background_research=question.background_research or "(no research available)",
    )

    logger.info(
        "Stage 3: refining question %s (domain=%s)",
        question.question_id, question.domain.value,
    )

    try:
        result = await agent.ainvoke(
            {"messages": [("system", prompt), ("human", "Refine this question now. Verify resolution sources exist.")]},
        )

        structured: RefinementResponse | None = result.get("structured_response")
        if not structured:
            logger.warning("Question %s: no structured response from refinement agent", question.question_id)
            return question  # return unmodified

        # Enrich the question with resolution fields
        res = structured.resolution
        question.resolution_criteria = res.resolution_criteria
        question.resolution_source = res.resolution_source
        question.fine_print = res.fine_print
        question.resolution_date = res.resolution_date
        question.base_rate_estimate = res.base_rate_estimate
        question.base_rate_reasoning = res.base_rate_reasoning

        # Apply minor text revisions if flagged
        if structured.question_text_revision:
            logger.info(
                "Question %s: text revised: %s",
                question.question_id, structured.question_text_revision[:100],
            )
            question.question_text = structured.question_text_revision

        if structured.title_revision:
            question.title = structured.title_revision

        logger.info(
            "Question %s: refined. Resolution date: %s, base rate: %.2f",
            question.question_id, question.resolution_date, question.base_rate_estimate,
        )

        return question

    except Exception:
        logger.exception("Question %s: refinement agent failed", question.question_id)
        return question  # return unmodified on failure


# ── Entry Point ──────────────────────────────────────────────────────────────


async def run_stage3(state: PipelineState) -> PipelineState:
    """
    Stage 3 entry point — run as a LangGraph node.

    1. For each researched question, run the refinement agent
    2. Populate resolution fields
    3. Store in state.refined_questions
    """
    config = state.config
    model_name = config.drafter_model  # same model as Stages 2-2.5
    temperature = 0.3  # lower temp for precise criteria

    logger.info(
        "Stage 3 starting — %d researched questions to refine, model=%s",
        len(state.researched_questions), model_name,
    )

    refined: list[Question] = []

    for question in state.researched_questions:
        refined_q = await run_refinement_agent(
            question=question,
            model_name=model_name,
            temperature=temperature,
        )
        refined.append(refined_q)

        has_criteria = refined_q.resolution_criteria is not None
        logger.info(
            "Question %s → refined=%s (total: %d)",
            refined_q.question_id, has_criteria, len(refined),
        )

    state.refined_questions = refined

    # Count how many were fully refined
    fully_refined = sum(1 for q in refined if q.resolution_criteria is not None)

    logger.info(
        "Stage 3 complete — %d/%d fully refined",
        fully_refined, len(refined),
    )

    return state
