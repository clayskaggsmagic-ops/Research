"""
Post-Pipeline Node A — Ground Truth Resolver (Bosse et al. A.21)

Since all events in the question set have ALREADY HAPPENED, this node
determines the actual outcome for each question using adversarial resolution.

The resolver:
  1. Uses web search to find concrete evidence from specified resolution sources
  2. Binary → YES/NO with citations; MC → matching option letter
  3. AMBIGUOUS if contradictory evidence; ANNULLED if question invalid
  4. Produces a "bullet-proof" derivation that would convince someone who
     just LOST money betting on the question

Entry point: run_ground_truth_resolver(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.schemas import PipelineState, Question, QuestionType, ResolutionStatus
from src.tools import get_stage2_tools

logger = logging.getLogger(__name__)


# ── Structured output ────────────────────────────────────────────────────────


class ResolutionResult(BaseModel):
    """Adversarial ground truth resolution (A.21)."""

    question_id: str
    correct_answer: str = Field(
        description="YES or NO for binary; option letter (A/B/C/D/E) for MC"
    )
    resolution_status: str = Field(
        description="resolved_yes | resolved_no | resolved_option | ambiguous | annulled"
    )
    resolution_evidence: str = Field(
        description="Direct quotes and URLs from resolution sources proving the answer"
    )
    resolution_derivation: str = Field(
        description="Bullet-proof argument: ALL links, ALL search queries verbatim, "
        "explicit assumptions, literal interpretation of resolution criteria"
    )
    resolution_weaknesses: str = Field(
        description="Self-critique: 'ONE subtle mistake in your derivation — what would it be?'"
    )
    search_queries_used: list[str] = Field(
        description="Every search query used during resolution, verbatim"
    )


# ── Prompt ───────────────────────────────────────────────────────────────────


RESOLVER_PROMPT = """\
You are an adversarial ground truth resolver for a forecasting tournament. \
Your resolution must be BULLET-PROOF — it should convince even someone who \
just LOST A BUNCH OF MONEY betting on this question and tries to find \
loopholes to overturn its resolution.

## Question Under Resolution
ID: {question_id}
Type: {question_type}
Title: {title}
Question: {question_text}
{options_block}

## Resolution Infrastructure
Criteria: {resolution_criteria}
Source: {resolution_source}
Fine print: {fine_print}
Resolution date: {resolution_date}

## Your Task:

1. **Search** — Use web_search to find the actual outcome. Search the SPECIFIC \
resolution sources listed above. Record EVERY search query you use verbatim.

2. **Gather evidence** — Find concrete, dated evidence. Direct quotes are \
preferred over paraphrasing. Include ALL URLs.

3. **Apply resolution criteria LITERALLY** — The resolution criteria ALWAYS \
supersede "reasonable interpretations of the question." Be extremely literal. \
If criteria say ">= 50%" and the actual number is 49.9%, that's a NO.

4. **Determine answer**:
   - Binary: YES or NO
   - MC: The option letter (A, B, C, D, or E) that best matches
   - AMBIGUOUS: Evidence is genuinely contradictory or criteria don't cover \
     what actually happened
   - ANNULLED: The question itself became invalid (e.g., resolution source \
     ceased to exist)

5. **Write derivation** — Make it bullet-proof:
   - Show your work step by step
   - Make ALL assumptions explicit and visible
   - Include ALL links to evidence
   - Be literal, not "reasonable"

6. **Self-critique** — What's the ONE subtle mistake in your derivation? \
Where could someone find a loophole? If there IS a genuine weakness, say \
so — don't pretend it's airtight if it's not.

Set resolution_status to one of: resolved_yes, resolved_no, resolved_option, \
ambiguous, annulled\
"""


# ── Run Resolver ─────────────────────────────────────────────────────────────


async def resolve_question(
    question: Question,
    model_name: str = "gemini-3.1-pro-preview",
    temperature: float = 0.2,
) -> Question:
    """Resolve a single question using adversarial web research."""

    tools = get_stage2_tools()
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    agent = create_react_agent(model=llm, tools=tools, response_format=ResolutionResult)

    options_block = ""
    if question.options:
        opts = "\n".join(f"  {chr(65 + i)}. {o}" for i, o in enumerate(question.options))
        options_block = f"Options:\n{opts}"

    prompt = RESOLVER_PROMPT.format(
        question_id=question.question_id,
        question_type=question.question_type.value,
        title=question.title,
        question_text=question.question_text,
        options_block=options_block,
        resolution_criteria=question.resolution_criteria or "(not set)",
        resolution_source=question.resolution_source or "(not set)",
        fine_print=question.fine_print or "(not set)",
        resolution_date=question.resolution_date or "(not set)",
    )

    try:
        result = await agent.ainvoke(
            {"messages": [("system", prompt), ("human", f"Resolve question {question.question_id} now.")]},
        )
        resp: ResolutionResult | None = result.get("structured_response")

        if resp:
            question.correct_answer = resp.correct_answer
            question.resolution_evidence = resp.resolution_evidence
            question.resolution_derivation = resp.resolution_derivation
            question.resolution_weaknesses = resp.resolution_weaknesses
            question.search_queries_used = resp.search_queries_used

            # Map status string to enum
            status_map = {
                "resolved_yes": ResolutionStatus.RESOLVED_YES,
                "resolved_no": ResolutionStatus.RESOLVED_NO,
                "resolved_option": ResolutionStatus.RESOLVED_OPTION,
                "ambiguous": ResolutionStatus.AMBIGUOUS,
                "annulled": ResolutionStatus.ANNULLED,
            }
            question.resolution_status = status_map.get(
                resp.resolution_status, ResolutionStatus.AMBIGUOUS
            )

            logger.info(
                "Resolved %s: %s = %s",
                question.question_id, question.resolution_status.value, question.correct_answer,
            )
        else:
            logger.warning("Resolver returned no structured response for %s", question.question_id)

    except Exception:
        logger.exception("Resolution failed for %s", question.question_id)

    return question


# ── Entry Point ──────────────────────────────────────────────────────────────


async def run_ground_truth_resolver(state: PipelineState) -> PipelineState:
    """
    Post-pipeline Node A — resolve all questions in the final manifest.
    Runs on resolver_model (gemini-3.1-pro-preview).
    """
    import asyncio

    config = state.config
    questions = state.final_manifest

    logger.info(
        "Ground truth resolution starting — %d questions, model=%s",
        len(questions), config.resolver_model,
    )

    tasks = [
        resolve_question(q, model_name=config.resolver_model, temperature=0.2)
        for q in questions
    ]
    resolved = await asyncio.gather(*tasks)

    state.final_manifest = list(resolved)

    resolved_count = sum(1 for q in resolved if q.resolution_status != ResolutionStatus.PENDING)
    ambiguous_count = sum(1 for q in resolved if q.resolution_status == ResolutionStatus.AMBIGUOUS)

    logger.info(
        "Ground truth resolution complete — %d/%d resolved, %d ambiguous",
        resolved_count, len(questions), ambiguous_count,
    )

    return state
