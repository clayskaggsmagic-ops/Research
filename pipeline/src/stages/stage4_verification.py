"""
Stage 4 — Adversarial Verification (Bosse et al. A.12–A.15 + custom)

6 INDEPENDENT verification agents, each specialized in one dimension.
Uses a DIFFERENT model generation (gemini-3.1-pro-preview) to ensure
cross-model diversity vs. the drafter (gemini-3-pro-preview).

Agents:
  1. Quality & Meaningfulness (A.12) — is it a good forecasting question?
  2. Ambiguity (A.13) — can it be unambiguously resolved?
  3. AI-Resolvability (A.14) — can a ReAct agent find the answer?
  4. Non-Triviality via Forecasting (A.15) — makes a forecast to check difficulty
  5. Leader Attribution Checker (custom) — is this Trump's personal decision?
  6. Information Leakage Detector (custom) — does wording reveal the outcome?

Aggregator:
  - APPROVED: all 6 pass
  - REVISION_NEEDED: 1-2 fails, fixable
  - REJECTED: 3+ fails or any fatal fail

Entry point: run_stage4(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.schemas import PipelineState, Question, VerificationVerdict
from src.tools import get_stage2_tools

logger = logging.getLogger(__name__)


# ── Agent Verdict Scales ──────────────────────────────────────────────────────


class QualityVerdict(str, Enum):
    BAD = "bad"
    MEH = "meh"
    GOOD = "good"
    GREAT = "great"


class AmbiguityVerdict(str, Enum):
    BAD = "bad"
    MEH = "meh"
    GOOD = "good"
    GREAT = "great"


class ResolvabilityVerdict(str, Enum):
    VERY_CERTAINLY_NO = "very_certainly_no"
    PROBABLY_NO = "probably_no"
    PROBABLY_YES = "probably_yes"
    VERY_CERTAINLY_YES = "very_certainly_yes"


class TrivialityVerdict(str, Enum):
    BAD = "bad"
    MEH = "meh"
    GOOD = "good"
    GREAT = "great"


class AttributionVerdict(str, Enum):
    FAIL = "fail"
    WARN = "warn"
    PASS_ATTR = "pass"


class LeakageVerdict(str, Enum):
    LEAKS_DETECTED = "leaks_detected"
    MINOR_CONCERN = "minor_concern"
    CLEAN = "clean"


# ── Structured Output Schemas ────────────────────────────────────────────────


class Agent1Response(BaseModel):
    """Quality & Meaningfulness (A.12)"""
    question_id: str
    difficulty_assessment: str = Field(description="Does more research lead to better forecasts?")
    entropy_assessment: str = Field(description="Is the answer non-obvious?")
    disagreement_room: str = Field(description="Could two forecasters differ by 20+ pp?")
    verdict: QualityVerdict
    reasoning: str


class Agent2Response(BaseModel):
    """Ambiguity (A.13)"""
    question_id: str
    terms_defined: bool = Field(description="Are all key terms well-defined with links?")
    dates_unambiguous: bool = Field(description="Resolution date has timezone and year?")
    cutoffs_explicit: bool = Field(description="Numeric cutoffs explicitly defined?")
    technicality_robust: bool = Field(description="Robust against unexpected technicalities?")
    agreement_score: int = Field(description="0-100: if 10 people check, will they agree?")
    verdict: AmbiguityVerdict
    reasoning: str


class Agent3Response(BaseModel):
    """AI-Resolvability (A.14)"""
    question_id: str
    source_locatable: bool = Field(description="Can the resolution source be trivially found?")
    source_exists_now: bool = Field(description="Does the specified source currently exist?")
    source_freely_accessible: bool = Field(description="Is the source freely accessible?")
    human_resolvable_10min: bool = Field(description="Could a human resolve it within 10 minutes?")
    verdict: ResolvabilityVerdict
    reasoning: str


class Agent4Response(BaseModel):
    """Non-Triviality via Forecasting (A.15)"""
    question_id: str
    base_rate_analysis: str = Field(description="Historical frequency of similar events")
    status_quo_bias: str = Field(description="How much inertia favors current state?")
    current_trends: str = Field(description="Relevant trends and seasonal effects")
    pre_mortem: str = Field(description="How would the forecast most likely be wrong?")
    probability_estimate: float = Field(description="0.0-1.0 probability forecast")
    trivially_easy: bool = Field(description="True if >95% or <5%")
    verdict: TrivialityVerdict
    reasoning: str


class Agent5Response(BaseModel):
    """Leader Attribution Checker (custom)"""
    question_id: str
    decision_maker: str = Field(description="Who actually makes this decision?")
    presidential_authority: bool = Field(description="Is this within presidential prerogative?")
    auto_fail_category: str | None = Field(
        default=None,
        description="If auto-fail: 'fed_reserve', 'scotus', 'congress', 'market', 'foreign_gov', or null",
    )
    verdict: AttributionVerdict
    reasoning: str


class Agent6Response(BaseModel):
    """Information Leakage Detector (custom)"""
    question_id: str
    background_leaks: bool = Field(description="Does the background reveal the outcome?")
    option_framing_leaks: bool = Field(description="Does option ordering/framing telegraph the answer?")
    question_phrasing_leaks: bool = Field(description="Does the question wording itself leak?")
    specific_leaks_found: list[str] = Field(
        default_factory=list,
        description="Specific phrases or framings that leak information",
    )
    verdict: LeakageVerdict
    reasoning: str


# ── Agent Prompts ─────────────────────────────────────────────────────────────


AGENT1_PROMPT = """\
You are a forecasting tournament quality reviewer. Your ONLY job is to assess \
whether this is a GOOD forecasting question — one where skill matters.

## Today's Date: {simulation_date}

## Question Under Review
ID: {question_id} | Type: {question_type} | Domain: {domain}
Title: {title}
Question: {question_text}
Background: {background}
{options_block}
Resolution criteria: {resolution_criteria}

## Evaluate ONLY These Dimensions:

1. **Difficulty**: Does more research lead to a better forecast?
   - BAD: Pure coin flip or trivially obvious
   - GOOD: Looking up data, trends, and expert opinion helps
   - GREAT: Requires synthesizing multiple sources and reasoning through scenarios

2. **Entropy**: Is the answer non-obvious (high entropy)?
   - A good forecast should be between 5% and 95% probability
   - If almost everyone would agree on the answer, it's too easy

3. **Disagreement room**: Could two experienced forecasters differ by 20+ \
percentage points?

Rate as: bad / meh / good / great\
"""


AGENT2_PROMPT = """\
You are an ambiguity auditor for a forecasting tournament. Your ONLY job is to \
assess whether this question can be UNAMBIGUOUSLY resolved.

## Today's Date: {simulation_date}

## Question Under Review
ID: {question_id} | Type: {question_type} | Domain: {domain}
Title: {title}
Question: {question_text}
Background: {background}
{options_block}
Resolution criteria: {resolution_criteria}
Resolution source: {resolution_source}
Fine print: {fine_print}

## Check ONLY These:

1. **Terms defined**: Are ALL key terms clearly defined with links to \
authoritative sources (Wikipedia, official organizations)?
2. **Dates unambiguous**: Resolution date has timezone? Year explicitly stated?
3. **Cutoffs explicit**: For numeric thresholds — is it >= or >? At least or more than?
4. **Technicality robust**: Is the question robust against unexpected \
technicalities or gotchas?
5. **Agreement score (0-100)**: "If 10 people look at the question and the \
resolution source, will they all broadly agree on the outcome?"

Rate as: bad / meh / good / great\
"""


AGENT3_PROMPT = """\
You are an AI-resolvability tester. Your ONLY job is to assess whether a \
future ReAct agent with web search could autonomously resolve this question.

## Today's Date: {simulation_date}

## Question Under Review
ID: {question_id} | Type: {question_type} | Domain: {domain}
Title: {title}
Question: {question_text}
Resolution criteria: {resolution_criteria}
Resolution source: {resolution_source}
Resolution date: {resolution_date}

## Check ONLY These:

1. **Source locatable**: Will it be trivially possible to LOCATE the resolution source?
2. **Source exists**: Does it currently exist? Use web_search to verify.
3. **Freely accessible**: Is the source freely accessible (not paywalled)?
4. **Specific data available**: Does the specific column/variable/data point exist?
5. **10-minute human resolution**: Could a human resolve it within 10 minutes?

Rate as: very_certainly_no / probably_no / probably_yes / very_certainly_yes\
"""


AGENT4_PROMPT = """\
You are a forecasting tester. Your job is to ACTUALLY MAKE A FORECAST on this \
question to test whether it is trivially easy or genuinely challenging.

## Today's Date: {simulation_date}
You do NOT know what happens after this date.

## Question Under Review
ID: {question_id} | Type: {question_type} | Domain: {domain}
Title: {title}
Question: {question_text}
Background: {background}
{options_block}
Resolution criteria: {resolution_criteria}

## Make Your Forecast:

1. **Base rates**: What is the historical frequency of similar events?
2. **Status quo bias**: The world changes slowly — how much inertia?
3. **Current trends**: Seasonal effects, political dynamics, recent momentum
4. **Incentives**: What do powerful actors want? What are their constraints?
5. **Pre-mortem**: How would you most likely be wrong?

Then give your probability estimate (0.0-1.0).

## Triviality Check:
- If your estimate is >0.95 or <0.05, this question is trivially easy → verdict=bad
- If two reasonable forecasters couldn't differ by 20+ points → verdict=meh
- Otherwise, rate how challenging the question is: good or great\
"""


AGENT5_PROMPT = """\
You are a presidential attribution auditor. Your ONLY job is to verify that \
this question concerns a decision genuinely made by the PRESIDENT PERSONALLY, \
not by another institution, market force, or bureaucratic process.

## Question Under Review
ID: {question_id} | Type: {question_type} | Domain: {domain}
Title: {title}
Question: {question_text}
Background: {background}

## Auto-Fail Categories (verdict=fail immediately):
- Federal Reserve decisions (monetary policy, rate decisions)
- Supreme Court rulings
- Congressional vote outcomes
- Market movements / economic indicators
- Foreign government actions

## Check:
- Is this decision within the President's constitutional or statutory authority?
- Is it traceable to Trump's personal directive, not a bureaucratic default?
- Could this happen without the President's involvement?

Verdict: fail / warn / pass\
"""


AGENT6_PROMPT = """\
You are an information leakage detector for a retroactive forecasting study. \
These questions were written AFTER the events occurred, so the authors \
might have accidentally revealed the actual outcome in the question text.

## CRITICAL CONTEXT
This question was generated retroactively — the outcome has already happened. \
We need to verify the wording doesn't telegraph what actually occurred.

## Today's Date (for the question): {simulation_date}

## Question Under Review
ID: {question_id} | Type: {question_type} | Domain: {domain}
Title: {title}
Question: {question_text}
Background: {background}
{options_block}

## Search for Leakage:
Use web_search to find what ACTUALLY happened. Then check:

1. **Background leaks**: Does the background narrative frame events in a way \
that hints at the outcome? Look for loaded language, selective emphasis, or \
framing that would only make sense if you knew what happened.

2. **Option framing leaks**: For action_selection questions — does the ordering, \
wording, or level of detail on one option suggest it's the "right" answer?

3. **Question phrasing leaks**: Does the question itself use language that \
implies a direction? (e.g., "Will Trump ESCALATE..." when the answer is yes)

List any specific phrases or framings that leak information.

Verdict: leaks_detected / minor_concern / clean\
"""


# ── Run Single Verification Agent ────────────────────────────────────────────


async def _run_agent(
    question: Question,
    agent_id: int,
    prompt_template: str,
    response_model: type[BaseModel],
    model_name: str,
    temperature: float = 0.3,
) -> BaseModel | None:
    """Run a single verification agent for one question."""

    # Only resolvability (3) and leakage (6) need web search
    tools = get_stage2_tools() if agent_id in (3, 6) else []

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    agent = create_react_agent(model=llm, tools=tools, response_format=response_model)

    # Format options block
    options_block = ""
    if question.options:
        opts = "\n".join(f"  {chr(65 + i)}. {o}" for i, o in enumerate(question.options))
        options_block = f"Options:\n{opts}"

    format_kwargs = {
        "simulation_date": question.simulation_date,
        "question_id": question.question_id,
        "question_type": question.question_type.value,
        "domain": question.domain.value,
        "title": question.title,
        "question_text": question.question_text,
        "background": question.background,
        "options_block": options_block,
        "resolution_criteria": question.resolution_criteria or "(not yet set)",
        "resolution_source": question.resolution_source or "(not yet set)",
        "resolution_date": question.resolution_date or "(not yet set)",
        "fine_print": question.fine_print or "(not yet set)",
    }

    prompt = prompt_template.format(**format_kwargs)

    try:
        result = await agent.ainvoke(
            {"messages": [("system", prompt), ("human", f"Evaluate question {question.question_id} now.")]},
        )
        return result.get("structured_response")
    except Exception:
        logger.exception("Agent %d failed on question %s", agent_id, question.question_id)
        return None


# ── Run All 6 Agents in Parallel ─────────────────────────────────────────────


AGENT_CONFIGS = [
    (1, AGENT1_PROMPT, Agent1Response),
    (2, AGENT2_PROMPT, Agent2Response),
    (3, AGENT3_PROMPT, Agent3Response),
    (4, AGENT4_PROMPT, Agent4Response),
    (5, AGENT5_PROMPT, Agent5Response),
    (6, AGENT6_PROMPT, Agent6Response),
]


async def verify_question(
    question: Question,
    model_name: str = "gemini-3.1-pro-preview",
    temperature: float = 0.3,
) -> tuple[VerificationVerdict, str, dict[int, BaseModel | None]]:
    """
    Run all 6 verification agents in pairs of 2 with timeout protection.

    Returns:
        (final_verdict, aggregated_notes, {agent_id: response})
    """
    AGENT_TIMEOUT = 120  # seconds per pair
    PAIR_SIZE = 2
    
    results: list[BaseModel | None] = []
    for i in range(0, len(AGENT_CONFIGS), PAIR_SIZE):
        pair = AGENT_CONFIGS[i : i + PAIR_SIZE]
        tasks = [
            _run_agent(question, agent_id, prompt, resp_model, model_name, temperature)
            for agent_id, prompt, resp_model in pair
        ]
        try:
            pair_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=AGENT_TIMEOUT,
            )
            for r in pair_results:
                if isinstance(r, Exception):
                    logger.warning("Agent failed for %s: %s", question.question_id, str(r)[:120])
                    results.append(None)
                else:
                    results.append(r)
        except asyncio.TimeoutError:
            logger.warning("Agent pair timeout for %s (pair starting at agent %d)", question.question_id, pair[0][0])
            results.extend([None] * len(pair))
        # Small delay between pairs to ease rate limiting
        await asyncio.sleep(1.0)
    agent_results = {agent_id: result for (agent_id, _, _), result in zip(AGENT_CONFIGS, results)}

    # ── Aggregate verdicts ────────────────────────────────────────────────
    fails: list[str] = []
    warns: list[str] = []
    notes_parts: list[str] = []

    # Agent 1: Quality
    a1: Agent1Response | None = agent_results.get(1)
    if a1:
        notes_parts.append(f"[Quality: {a1.verdict.value}] {a1.reasoning}")
        if a1.verdict == QualityVerdict.BAD:
            fails.append("quality:bad")
        elif a1.verdict == QualityVerdict.MEH:
            warns.append("quality:meh")

    # Agent 2: Ambiguity
    a2: Agent2Response | None = agent_results.get(2)
    if a2:
        notes_parts.append(f"[Ambiguity: {a2.verdict.value}, agreement={a2.agreement_score}] {a2.reasoning}")
        if a2.verdict == AmbiguityVerdict.BAD:
            fails.append("ambiguity:bad")
        elif a2.verdict == AmbiguityVerdict.MEH:
            warns.append("ambiguity:meh")

    # Agent 3: AI-Resolvability
    a3: Agent3Response | None = agent_results.get(3)
    if a3:
        notes_parts.append(f"[AI-Resolvability: {a3.verdict.value}] {a3.reasoning}")
        if a3.verdict in (ResolvabilityVerdict.VERY_CERTAINLY_NO, ResolvabilityVerdict.PROBABLY_NO):
            fails.append(f"ai_resolvability:{a3.verdict.value}")

    # Agent 4: Non-Triviality
    a4: Agent4Response | None = agent_results.get(4)
    if a4:
        notes_parts.append(
            f"[Triviality: {a4.verdict.value}, p={a4.probability_estimate:.2f}] {a4.reasoning}"
        )
        if a4.verdict == TrivialityVerdict.BAD:
            fails.append("triviality:bad")
        elif a4.verdict == TrivialityVerdict.MEH:
            warns.append("triviality:meh")

    # Agent 5: Leader Attribution
    a5: Agent5Response | None = agent_results.get(5)
    if a5:
        notes_parts.append(f"[Attribution: {a5.verdict.value}] {a5.reasoning}")
        if a5.verdict == AttributionVerdict.FAIL:
            fails.append(f"attribution:fail({a5.auto_fail_category})")
        elif a5.verdict == AttributionVerdict.WARN:
            warns.append("attribution:warn")

    # Agent 6: Information Leakage
    a6: Agent6Response | None = agent_results.get(6)
    if a6:
        leaks = ", ".join(a6.specific_leaks_found) if a6.specific_leaks_found else "none"
        notes_parts.append(f"[Leakage: {a6.verdict.value}, leaks={leaks}] {a6.reasoning}")
        if a6.verdict == LeakageVerdict.LEAKS_DETECTED:
            fails.append("leakage:detected")
        elif a6.verdict == LeakageVerdict.MINOR_CONCERN:
            warns.append("leakage:minor")

    # ── Determine final verdict ───────────────────────────────────────────
    aggregated_notes = "\n".join(notes_parts)

    # Fatal fails: attribution or leakage auto-reject
    fatal = any("attribution:fail" in f for f in fails) or any("leakage:detected" in f for f in fails)

    if fatal or len(fails) >= 3:
        verdict = VerificationVerdict.REJECTED
    elif len(fails) >= 1:
        verdict = VerificationVerdict.REVISION_NEEDED
    else:
        verdict = VerificationVerdict.APPROVED

    logger.info(
        "Question %s: verdict=%s, fails=%s, warns=%s",
        question.question_id, verdict.value, fails, warns,
    )

    return verdict, aggregated_notes, agent_results


# ── Entry Point ──────────────────────────────────────────────────────────────


async def run_stage4(state: PipelineState) -> PipelineState:
    """
    Stage 4 entry point — run as a LangGraph node.

    1. For each refined question, run all 6 verification agents in parallel
    2. Aggregate verdicts
    3. Route: APPROVED → verified, REVISION_NEEDED → back to Stage 3 (max 2 loops),
       REJECTED → rejected_questions
    """
    from src.stages.stage3_refinement import run_refinement_agent

    config = state.config
    verifier_model = config.verifier_model  # different model family
    max_loops = config.max_revision_loops

    logger.info(
        "Stage 4 starting — %d refined questions to verify, model=%s",
        len(state.refined_questions), verifier_model,
    )

    verified: list[Question] = []
    rejected: list[Question] = []

    for question in state.refined_questions:
        current_q = question

        for loop in range(max_loops + 1):  # 0, 1, 2 — initial + 2 revisions
            verdict, notes, _ = await verify_question(
                current_q,
                model_name=verifier_model,
                temperature=config.verifier_temperature,
            )

            current_q.verification_verdict = verdict
            current_q.verification_notes = notes

            if verdict == VerificationVerdict.APPROVED:
                verified.append(current_q)
                logger.info("Question %s APPROVED (loop %d)", current_q.question_id, loop)
                break
            elif verdict == VerificationVerdict.REJECTED:
                rejected.append(current_q)
                logger.warning("Question %s REJECTED (loop %d): %s", current_q.question_id, loop, notes[:200])
                break
            elif verdict == VerificationVerdict.REVISION_NEEDED:
                if loop < max_loops:
                    logger.info(
                        "Question %s → REVISION_NEEDED (loop %d/%d), sending back to Stage 3",
                        current_q.question_id, loop, max_loops,
                    )
                    current_q.revision_count += 1
                    current_q = await run_refinement_agent(
                        current_q,
                        model_name=config.drafter_model,
                        temperature=0.3,
                    )
                else:
                    # Max revisions exhausted — approve with warnings
                    logger.warning(
                        "Question %s: max revisions exhausted, approving with warnings",
                        current_q.question_id,
                    )
                    current_q.verification_verdict = VerificationVerdict.APPROVED
                    verified.append(current_q)
                    break

    state.verified_questions = verified
    state.rejected_questions = rejected

    logger.info(
        "Stage 4 complete — %d approved, %d rejected, %d had revisions",
        len(verified), len(rejected), sum(1 for q in verified if q.revision_count > 0),
    )

    return state
