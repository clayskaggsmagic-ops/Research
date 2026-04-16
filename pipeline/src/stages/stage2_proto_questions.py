"""
Stage 2 — Proto-Question Generator (ReAct Web Agent)

Architecture (per Bosse et al. 2026, Appendix A.8):
  1. For each DecisionSeed from Stage 1, a ReAct agent:
     a. Researches the political context around the simulation_date
     b. Generates 1-3 forecasting proto-questions
  2. Questions are written AS IF asked on simulation_date — no hindsight
  3. Output: list of Question objects stored in state.proto_questions

Entry point: run_stage2(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import logging
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.config import SEED_AGENT_MODEL
from src.schemas import (
    DecisionSeed,
    DomainType,
    PipelineState,
    Question,
    QuestionType,
)
from src.tools import get_stage2_tools

logger = logging.getLogger(__name__)


# ── Structured Output Schema for Agent ─────────────────────────────────────────


class GeneratedQuestion(BaseModel):
    """A single proto-question proposed by the research agent."""

    title: str = Field(
        description="Short, specific claim — e.g. 'Will Trump impose ≥25% tariffs on EU auto imports?'"
    )
    background: str = Field(
        description="2-4 paragraph context as of simulation_date — must NOT reveal the outcome."
    )
    question_text: str = Field(
        description=(
            "The forecasting question. MUST include Trump's name. "
            "For binary: 'Will Trump X by Y?' "
            "For action_selection: 'Which action will Trump take regarding X?'"
        )
    )
    question_type: str = Field(
        description="'binary' or 'action_selection'"
    )
    options: list[str] | None = Field(
        default=None,
        description=(
            "For action_selection ONLY: 4-5 mutually exclusive options ordered "
            "from most aggressive to most passive. MUST always include 'Take no action' "
            "as the final option."
        ),
    )
    trump_attribution: str = Field(
        description=(
            "1-2 sentences explaining WHY this is Trump's personal decision. "
            "If you can't explain why, DO NOT generate this question."
        )
    )
    rationale: str = Field(
        description="Why this is a good forecasting question (entropy, difficulty, independence)"
    )


class ProtoQuestionResponse(BaseModel):
    """Structured output from the proto-question agent for one seed."""

    seed_id: str = Field(description="The seed this response is for")
    questions: list[GeneratedQuestion] = Field(
        description="1-3 forecasting questions inspired by the seed"
    )
    research_summary: str = Field(
        description="Brief summary of what context was researched"
    )


# ── Agent Prompt ───────────────────────────────────────────────────────────────


PROTO_QUESTION_SYSTEM_PROMPT = """\
You are an admin at Metaculus, a forecasting platform, and you want to design \
a forecasting tournament. You collected a bunch of real-world events and are \
now going through them 1 by 1 in search of inspiration for forecasting questions.

## CRITICAL: Today's Date
Today is **{simulation_date}**. You do NOT know what will happen after this \
date. You must write questions, background, and options AS IF you are living \
on {simulation_date}. NEVER reveal the actual outcome of the event.

## The Event (your inspiration)
Domain: {domain}
Context: {event_description}
Possible actions being discussed: {alternatives}

## Your Task
Based on this event, return 1-3 proto-questions (+ your own rationale for why \
you believe each is a good question). You can take the input as inspiration, \
but please do some research yourself to identify suitable questions. The \
question(s) should have the following properties:

1. **Near-term future**: Questions should resolve within ~30-90 days of \
{simulation_date}. Pay attention to today's date and don't suggest questions \
that would resolve before {simulation_date}.

2. **Actually resolvable**: We need a source of truth that tells us whether \
the answer is yes or no. This source has to actually exist at the resolution \
date. Think: Federal Register, official press releases, Congressional records, \
credible reporting (AP, Reuters, NYT).

3. **High entropy**: Non-trivial questions with answers that aren't almost \
certainly true or false. A good forecast should be between 5% and 95%. \
Two good forecasters should be able to reasonably disagree by ≥20 \
percentage points.

4. **Research rewards quality**: Doing more research should lead to a \
better forecast. Avoid pure base-rate or coin-flip questions. Avoid \
questions where looking up one existing number or forecast settles it.

5. **Independent**: Questions should be diverse. A forecaster can't just \
identify one confounder and effectively "bet on" it across many questions.

## Question Types
Generate a MIX of:
- **BINARY**: "Will Trump [specific action] by [date]?" — clear yes/no
- **ACTION_SELECTION**: "Which action will Trump take regarding X?" + 4-5 \
mutually exclusive options from most aggressive to most passive. ALWAYS \
include "Take no action" as the final option.

## CRITICAL: {leader} PERSONAL ATTRIBUTION REQUIREMENT
Every question MUST be about a decision that **{leader}** personally makes, \
directs, or has direct authority over. This is NON-NEGOTIABLE.

✅ GOOD examples (Trump is the decision-maker):
- "Will Trump impose tariffs on X by Y?"
- "Will Trump sign an executive order on X?"
- "Will Trump fire/nominate X?"
- "Will Trump veto the X bill?"
- "Which action will Trump take regarding X?"

❌ BAD examples (someone else is the decision-maker — DO NOT GENERATE THESE):
- "What will the Department of Energy do regarding the SPR?" (agency, not Trump)
- "Will Congress pass the X bill?" (Congress, not Trump)
- "Will the Fed raise interest rates?" (Fed, not Trump)
- "Will the Supreme Court rule on X?" (SCOTUS, not Trump)
- "Will the stock market reach X?" (market forces, not Trump)
- "What will NATO do about X?" (foreign institution, not Trump)

Trump's NAME must appear in every question_text. The question must be \
about HIS personal action, directive, order, nomination, statement, or \
decision — not about downstream consequences or institutional processes \
that happen independently of the President.

Attribution evidence from the seed event: {attribution_evidence}

## ANTI-HINDSIGHT RULES
- You are on {simulation_date}. Everything after this date is the unknown future.
- The background must describe the status quo as of {simulation_date}.
- Do NOT hint at, suggest, or reveal what actually happened.
- Use phrases like "it remains to be seen", "experts are divided", "multiple \
paths are possible."
- If you can't write the question without revealing the outcome, skip it.

## Research Instructions
Use web_search to:
- Find the political context around {simulation_date}
- Check what experts and analysts were saying around that date
- Look for analogous prediction market questions and their prices
- Understand what the key uncertainties were at the time\
"""


# ── Run Single Seed Agent ────────────────────────────────────────────────────


async def run_seed_agent(
    seed: DecisionSeed,
    model_name: str = "gemini-3-pro-preview",
    temperature: float = 0.7,
) -> list[Question]:
    """
    Run the proto-question agent for a single seed.

    Returns a list of Question objects (proto-questions only — resolution
    fields are left null).
    """
    tools = get_stage2_tools()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        response_format=ProtoQuestionResponse,
    )

    # Format alternatives as bullet points
    alternatives_text = "\n".join(f"  - {a}" for a in seed.plausible_alternatives)

    prompt = PROTO_QUESTION_SYSTEM_PROMPT.format(
        simulation_date=seed.simulation_date,
        domain=seed.domain.value,
        event_description=seed.event_description,
        alternatives=alternatives_text,
        leader=getattr(seed, '_leader', 'Donald J. Trump'),
        attribution_evidence=seed.attribution_evidence,
    )

    logger.info(
        "Stage 2: processing seed %s (domain=%s, sim_date=%s)",
        seed.seed_id, seed.domain.value, seed.simulation_date,
    )

    try:
        result = await agent.ainvoke(
            {"messages": [("system", prompt), ("human", "Begin your research and generate proto-questions now.")]},
        )

        structured: ProtoQuestionResponse | None = result.get("structured_response")
        if not structured:
            logger.warning("Seed %s: no structured response from agent", seed.seed_id)
            return []

        logger.info(
            "Seed %s: agent generated %d proto-questions. Research: %s",
            seed.seed_id,
            len(structured.questions),
            structured.research_summary[:200],
        )

        # Convert GeneratedQuestion → Question
        questions: list[Question] = []
        for i, gq in enumerate(structured.questions):
            # Validate question type
            try:
                q_type = QuestionType(gq.question_type)
            except ValueError:
                logger.warning(
                    "Seed %s, question %d: invalid type '%s', defaulting to binary",
                    seed.seed_id, i, gq.question_type,
                )
                q_type = QuestionType.BINARY

            # Validate action_selection has options
            options = gq.options
            if q_type == QuestionType.ACTION_SELECTION:
                if not options or len(options) < 3:
                    logger.warning(
                        "Seed %s, question %d: action_selection missing options, converting to binary",
                        seed.seed_id, i,
                    )
                    q_type = QuestionType.BINARY
                    options = None
                else:
                    # Ensure "Take no action" is present
                    has_no_action = any("no action" in o.lower() for o in options)
                    if not has_no_action:
                        options.append("Take no action")

            question = Question(
                question_id=f"Q-{seed.seed_id}-{i + 1:02d}",
                seed_id=seed.seed_id,
                question_type=q_type,
                title=gq.title,
                background=gq.background,
                question_text=gq.question_text,
                options=options,
                simulation_date=seed.simulation_date,
                domain=seed.domain,
            )
            questions.append(question)

        return questions

    except Exception:
        logger.exception("Seed %s: proto-question agent failed", seed.seed_id)
        return []


# ── Entry Point ────────────────────────────────────────────────────────────────


async def run_stage2(state: PipelineState) -> PipelineState:
    """
    Stage 2 entry point — run as a LangGraph node.

    1. For each seed, run the proto-question agent (sequential to respect rate limits)
    2. Collect all proto-questions
    3. Return updated state
    """
    config = state.config
    model_name = config.drafter_model
    temperature = config.drafter_temperature

    logger.info(
        "Stage 2 starting — %d seeds to process, model=%s",
        len(state.seeds), model_name,
    )

    all_questions: list[Question] = []

    for seed in state.seeds:
        questions = await run_seed_agent(
            seed=seed,
            model_name=model_name,
            temperature=temperature,
        )
        all_questions.extend(questions)
        logger.info(
            "Seed %s → %d questions (total so far: %d)",
            seed.seed_id, len(questions), len(all_questions),
        )

    state.proto_questions = all_questions

    logger.info(
        "Stage 2 complete — %d proto-questions from %d seeds (avg %.1f per seed)",
        len(all_questions),
        len(state.seeds),
        len(all_questions) / max(len(state.seeds), 1),
    )

    return state
