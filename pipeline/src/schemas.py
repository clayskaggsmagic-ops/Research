"""
Shared Pydantic schemas for the question generation pipeline.

These types flow through the LangGraph StateGraph:
  Seeds → Proto-Questions → Refined Questions → Verified Questions → Final Manifest
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────


class DomainType(str, Enum):
    """Decision domains for Trump presidential actions."""

    TRADE_TARIFFS = "trade_tariffs"
    EXECUTIVE_ORDERS = "executive_orders"
    PERSONNEL = "personnel"
    FOREIGN_POLICY = "foreign_policy"
    LEGISLATIVE = "legislative"
    PUBLIC_COMMS = "public_comms"
    LEGAL_JUDICIAL = "legal_judicial"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TimeHorizon(str, Enum):
    SHORT = "short"  # 1-30 days
    MEDIUM = "medium"  # 31-180 days
    LONG = "long"  # 181-365 days


class VerificationVerdict(str, Enum):
    APPROVED = "approved"
    REVISION_NEEDED = "revision_needed"
    REJECTED = "rejected"


class QuestionType(str, Enum):
    BINARY = "binary"
    ACTION_SELECTION = "action_selection"


class ResolutionStatus(str, Enum):
    RESOLVED_YES = "resolved_yes"
    RESOLVED_NO = "resolved_no"
    RESOLVED_OPTION = "resolved_option"  # for MC — correct_answer holds the option
    AMBIGUOUS = "ambiguous"
    ANNULLED = "annulled"
    PENDING = "pending"


# ── Data Models ────────────────────────────────────────────────────────────────


class Source(BaseModel):
    """A citation for a decision seed or question."""

    name: str
    url: str
    date: str  # YYYY-MM-DD


class DecisionSeed(BaseModel):
    """
    A real decision Trump made, extracted from a structured data source.
    This is the raw material that gets turned into prediction questions.
    """

    seed_id: str = Field(description="e.g. FED-REG-001, TRADE-003")
    event_description: str = Field(description="What happened — the context around the decision")
    decision_taken: str = Field(description="The specific action Trump took")
    decision_date: str = Field(description="YYYY-MM-DD when the action was taken")
    simulation_date: str = Field(
        description="YYYY-MM-DD, 1-30 days before decision_date. "
        "This is the 'fake today' the prediction model sees."
    )
    domain: DomainType
    plausible_alternatives: list[str] = Field(
        description="What else Trump could have done, including 'no action'"
    )
    sources: list[Source]
    attribution_evidence: str = Field(
        description="Why we believe this was Trump's personal decision"
    )
    leader_attributable: bool = Field(
        default=True,
        description="False if attribution is uncertain — flagged for review",
    )


class PredictionMarketBenchmark(BaseModel):
    """Price from a prediction market for an analogous question."""

    source: str = Field(description="Kalshi, Polymarket, etc.")
    question_url: str | None = None
    price_at_simulation_date: float | None = Field(
        default=None, description="Market probability near the simulation_date"
    )
    recorded_date: str | None = None


class Question(BaseModel):
    """
    A prediction question generated from a DecisionSeed.
    Progressively enriched as it flows through the pipeline stages.
    """

    # ── Identity ──
    question_id: str = Field(description="e.g. Q-001")
    seed_id: str = Field(description="Links back to the source DecisionSeed")
    question_type: QuestionType

    # ── Question content (Stage 2: Proto-Question Generator) ──
    title: str = Field(description="Short, specific claim")
    background: str = Field(description="Context as of simulation_date — must NOT reveal outcome")
    question_text: str = Field(description="The actual question")
    options: list[str] | None = Field(
        default=None, description="For action_selection only — 4-5 mutually exclusive options"
    )
    simulation_date: str = Field(description="The 'fake today' — inherited from the seed")
    domain: DomainType

    # ── Background research (Stage 2.5: Background Research Agent) ──
    background_research: str | None = Field(
        default=None,
        description="Structured research brief: context, data, numbers, trends — produced by A.10 agent",
    )
    research_flags: list[str] = Field(
        default_factory=list,
        description="Problems found during research: hallucinated facts, already-resolved events, etc.",
    )

    # ── Resolution infrastructure (Stage 3: Refinement Agent) ──
    resolution_criteria: str | None = Field(
        default=None, description="Exact YES/NO conditions or option-matching rules"
    )
    resolution_source: str | None = Field(
        default=None, description="Federal Register, USTR, OFAC SDN List, etc."
    )
    fine_print: str | None = Field(
        default=None, description="Edge case handling"
    )
    resolution_date: str | None = Field(
        default=None, description="YYYY-MM-DD — must be after simulation_date"
    )
    base_rate_estimate: float | None = Field(
        default=None, description="Historical frequency, 0.0-1.0"
    )
    base_rate_reasoning: str | None = None

    # ── Verification (Stage 4: Cross-Model Verification) ──
    verification_verdict: VerificationVerdict | None = None
    verification_notes: str | None = Field(
        default=None, description="Issues found by the Critic, or 'all checks passed'"
    )
    revision_count: int = Field(default=0, description="How many revision loops this went through")

    # ── Difficulty & enrichment (Post-pipeline) ──
    difficulty: Difficulty | None = None
    time_horizon: TimeHorizon | None = None
    prediction_market_benchmark: PredictionMarketBenchmark | None = None

    # ── Ground truth (Post-pipeline, locked before predictions) ──
    correct_answer: str | None = Field(
        default=None, description="YES/NO for binary, option letter for MC"
    )
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    resolution_evidence: str | None = Field(
        default=None, description="Cited evidence for the ground truth answer"
    )
    resolution_derivation: str | None = Field(
        default=None,
        description="Bullet-proof argument with ALL links and search queries verbatim (A.21)",
    )
    resolution_weaknesses: str | None = Field(
        default=None,
        description="Self-critique: 'ONE subtle mistake in your derivation — what would it be?'",
    )
    search_queries_used: list[str] = Field(
        default_factory=list,
        description="Verbatim search queries used during resolution",
    )


class Prediction(BaseModel):
    """A single prediction collected from the model under test."""

    question_id: str
    prompt_type: Literal["persona", "analyst", "base_rate"]
    run_number: int = Field(description="1-10 (we run each prompt 10 times)")
    model_id: str = Field(description="Exact model version string")
    predicted_answer: str = Field(description="YES/NO or option letter")
    confidence: float = Field(description="0.0 to 1.0")
    reasoning: str
    option_ranking: list[str] | None = Field(
        default=None, description="For MC only — all options ranked"
    )


# ── Pipeline State ─────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Configuration for the pipeline run."""

    training_cutoff_date: str = Field(description="YYYY-MM-DD — model's knowledge cutoff")
    today_date: str = Field(description="YYYY-MM-DD — actual date of pipeline execution")
    leader: str = Field(default="Donald J. Trump")

    # Model assignments — cross-model verification requires different families
    drafter_model: str = Field(
        default="gemini-2.5-flash",
        description="Model for Stages 2-3 (proto-question gen + refinement)",
    )
    verifier_model: str = Field(
        default="gemini-2.5-flash",
        description="Model for Stage 4 — MUST be different generation from drafter",
    )
    resolver_model: str = Field(
        default="gemini-2.5-flash",
        description="Model for ground truth resolution + difficulty scoring",
    )
    prediction_model: str = Field(
        default="gemini-2.5-flash",
        description="Model under test — version-pinned for all predictions",
    )

    # Temperatures
    drafter_temperature: float = 0.7
    verifier_temperature: float = 0.3  # lower = more conservative critic
    prediction_temperature: float = 0.7  # >0 to measure variance across 10 runs

    # Limits
    max_revision_loops: int = 2
    predictions_per_question: int = 10


class PipelineState(BaseModel):
    """
    The LangGraph state object. Flows through all nodes.
    Each stage appends to or modifies items in this state.
    """

    config: PipelineConfig

    # Stage 1 output
    seeds: list[DecisionSeed] = Field(default_factory=list)

    # Stage 2 output
    proto_questions: list[Question] = Field(default_factory=list)

    # Stage 2.5 output
    researched_questions: list[Question] = Field(default_factory=list)

    # Stage 3 output
    refined_questions: list[Question] = Field(default_factory=list)

    # Stage 4 output
    verified_questions: list[Question] = Field(default_factory=list)
    revision_queue: list[Question] = Field(
        default_factory=list, description="Questions routed back for revision"
    )
    rejected_questions: list[Question] = Field(default_factory=list)

    # Stage 5 output
    final_manifest: list[Question] = Field(default_factory=list)

    # Post-pipeline
    predictions: list[Prediction] = Field(default_factory=list)
