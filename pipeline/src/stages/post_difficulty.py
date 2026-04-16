"""
Post-Pipeline Node B — Difficulty Scorer

Estimates difficulty for each question using three methods:
  1. Historical base rate — how often Trump has done this type of thing
  2. Prediction market proxy — Kalshi/Polymarket prices near simulation_date
  3. Editorial judgment — how constrained was the decision space

Assigns:
  - difficulty: easy / medium / hard
  - time_horizon: short (1-30d) / medium (31-180d) / long (181-365d)
  - prediction_market_benchmark (if found)

Entry point: run_difficulty_scorer(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import logging
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.schemas import (
    Difficulty,
    PipelineState,
    PredictionMarketBenchmark,
    Question,
    TimeHorizon,
)
from src.tools import get_stage2_tools

logger = logging.getLogger(__name__)


# ── Structured Output ────────────────────────────────────────────────────────


class DifficultyResult(BaseModel):
    """Multi-method difficulty assessment."""

    question_id: str

    # Method 1: Base rate
    base_rate_frequency: float = Field(
        description="How often Trump has done this type of thing historically (0.0-1.0)"
    )
    base_rate_analysis: str = Field(description="Brief reasoning for base rate estimate")

    # Method 2: Prediction market
    market_found: bool = Field(description="Was a relevant prediction market found?")
    market_source: str | None = Field(
        default=None, description="Kalshi, Polymarket, Metaculus, etc."
    )
    market_url: str | None = Field(default=None, description="URL to the market question")
    market_price: float | None = Field(
        default=None, description="Market probability near simulation_date, 0.0-1.0"
    )
    market_date: str | None = Field(
        default=None, description="Date market price was observed"
    )

    # Method 3: Editorial judgment
    decision_space_constraints: str = Field(
        description="How constrained was the decision space? Many options or binary?"
    )
    information_availability: str = Field(
        description="How much public information was available to forecasters?"
    )
    expert_disagreement: str = Field(
        description="Did experts/pundits disagree significantly?"
    )

    # Final assessment
    difficulty: str = Field(description="easy / medium / hard")
    time_horizon: str = Field(description="short / medium / long")
    reasoning: str = Field(description="Overall difficulty reasoning")


# ── Prompt ───────────────────────────────────────────────────────────────────


DIFFICULTY_PROMPT = """\
You are a difficulty scorer for a forecasting tournament about US presidential \
decisions. Assess how hard it would have been to predict the outcome of this \
question as of the simulation_date.

## Question
ID: {question_id} | Type: {question_type} | Domain: {domain}
Title: {title}
Question: {question_text}
Simulation date: {simulation_date}
Resolution date: {resolution_date}
{options_block}

## Score Difficulty Using Three Methods:

### Method 1: Historical Base Rate
- How often has Trump (or any recent president) done something like this?
- Use web_search to find historical precedents

### Method 2: Prediction Market Proxy
- Search Kalshi, Polymarket, and Metaculus for analogous questions near \
the simulation_date ({simulation_date})
- If found, record the market price/probability

### Method 3: Editorial Judgment
- How constrained was the decision space? (binary choice vs. many options)
- How much public information was available to forecasters?
- Did experts and pundits disagree significantly?

## Difficulty Scale:
- **easy**: Base rate very high/low (>85% or <15%), markets strongly agree, \
experts consensus, minimal uncertainty
- **medium**: Base rate moderate (15-85%), some expert disagreement, \
multiple plausible outcomes
- **hard**: No clear base rate, significant expert disagreement, \
unprecedented situation, many confounding factors

## Time Horizon:
Calculate days between simulation_date and resolution_date:
- **short**: 1-30 days
- **medium**: 31-180 days
- **long**: 181-365 days\
"""


# ── Run Scorer ───────────────────────────────────────────────────────────────


def _compute_time_horizon(simulation_date: str, resolution_date: str | None) -> TimeHorizon:
    """Compute time horizon from dates (fallback if LLM gets it wrong)."""
    if not resolution_date:
        return TimeHorizon.MEDIUM
    try:
        sim = datetime.strptime(simulation_date, "%Y-%m-%d")
        res = datetime.strptime(resolution_date, "%Y-%m-%d")
        days = (res - sim).days
        if days <= 30:
            return TimeHorizon.SHORT
        elif days <= 180:
            return TimeHorizon.MEDIUM
        else:
            return TimeHorizon.LONG
    except ValueError:
        return TimeHorizon.MEDIUM


async def score_question_difficulty(
    question: Question,
    model_name: str = "gemini-3.1-pro-preview",
    temperature: float = 0.3,
) -> Question:
    """Score difficulty for a single question."""

    tools = get_stage2_tools()
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    agent = create_react_agent(model=llm, tools=tools, response_format=DifficultyResult)

    options_block = ""
    if question.options:
        opts = "\n".join(f"  {chr(65 + i)}. {o}" for i, o in enumerate(question.options))
        options_block = f"Options:\n{opts}"

    prompt = DIFFICULTY_PROMPT.format(
        question_id=question.question_id,
        question_type=question.question_type.value,
        domain=question.domain.value,
        title=question.title,
        question_text=question.question_text,
        simulation_date=question.simulation_date,
        resolution_date=question.resolution_date or "(not set)",
        options_block=options_block,
    )

    try:
        result = await agent.ainvoke(
            {"messages": [("system", prompt), ("human", f"Score difficulty for {question.question_id} now.")]},
        )
        resp: DifficultyResult | None = result.get("structured_response")

        if resp:
            # Map difficulty
            diff_map = {"easy": Difficulty.EASY, "medium": Difficulty.MEDIUM, "hard": Difficulty.HARD}
            question.difficulty = diff_map.get(resp.difficulty, Difficulty.MEDIUM)

            # Time horizon — use deterministic calculation, not LLM
            question.time_horizon = _compute_time_horizon(
                question.simulation_date, question.resolution_date
            )

            # Prediction market benchmark
            if resp.market_found and resp.market_price is not None:
                question.prediction_market_benchmark = PredictionMarketBenchmark(
                    source=resp.market_source or "unknown",
                    question_url=resp.market_url,
                    price_at_simulation_date=resp.market_price,
                    recorded_date=resp.market_date,
                )

            # Update base rate if not already set
            if question.base_rate_estimate is None:
                question.base_rate_estimate = resp.base_rate_frequency
                question.base_rate_reasoning = resp.base_rate_analysis

            logger.info(
                "Scored %s: difficulty=%s, time_horizon=%s, market=%s",
                question.question_id,
                question.difficulty.value,
                question.time_horizon.value,
                f"{resp.market_price:.0%}" if resp.market_price else "none",
            )
        else:
            logger.warning("Difficulty scorer returned no structured response for %s", question.question_id)

    except Exception:
        logger.exception("Difficulty scoring failed for %s", question.question_id)
        # Fallback: set deterministic time horizon at minimum
        question.time_horizon = _compute_time_horizon(
            question.simulation_date, question.resolution_date
        )

    return question


# ── Entry Point ──────────────────────────────────────────────────────────────


async def run_difficulty_scorer(state: PipelineState) -> PipelineState:
    """
    Post-pipeline Node B — score difficulty for all manifest questions.
    Runs on resolver_model (gemini-3.1-pro-preview).
    Can run IN PARALLEL with ground truth resolution.
    """
    import asyncio

    config = state.config
    questions = state.final_manifest

    logger.info(
        "Difficulty scoring starting — %d questions, model=%s",
        len(questions), config.resolver_model,
    )

    tasks = [
        score_question_difficulty(q, model_name=config.resolver_model, temperature=0.3)
        for q in questions
    ]
    scored = await asyncio.gather(*tasks)

    state.final_manifest = list(scored)

    difficulty_dist = {}
    for q in scored:
        d = q.difficulty.value if q.difficulty else "unscored"
        difficulty_dist[d] = difficulty_dist.get(d, 0) + 1

    market_count = sum(1 for q in scored if q.prediction_market_benchmark is not None)

    logger.info(
        "Difficulty scoring complete — distribution=%s, %d/%d have market benchmarks",
        difficulty_dist, market_count, len(questions),
    )

    return state
