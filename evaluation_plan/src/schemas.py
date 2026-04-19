"""
Shared schemas for the leader-prediction experiments.

Record flow:
  Question (from final_manifest.json) → Resolution → Prediction → Score

Kept intentionally light — no database, just JSONL/JSON on disk.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────


class ExperimentId(str, Enum):
    E1 = "e1"           # Trump + CHRONOS broad
    E1_PRIME = "e1p"    # Trump + CHRONOS top_k=8 (compression control)
    E2 = "e2"           # Trump + CHRONOS refined (two-stage retrieval)
    E3 = "e3"           # Trump only (no context)
    E4 = "e4"           # Analyst + CHRONOS broad (reuses E1 briefing)
    E5 = "e5"           # Analyst + web search (answerability gate)


class QuestionFormat(str, Enum):
    BINARY = "binary"
    ACTION_SELECTION = "action_selection"


class ResolutionVerdict(str, Enum):
    YES = "YES"
    NO = "NO"
    UNRESOLVED = "UNRESOLVED"


# ── Resolution (ground truth) ──────────────────────────────────────────────────


class ResolutionEvidence(BaseModel):
    url: str
    quote: str | None = None
    source: str | None = None  # e.g., "federalregister.gov", "USTR", "OFAC"


class ResolutionPass(BaseModel):
    """One independent resolution attempt. We run two passes per question."""

    model_config = ConfigDict(extra="forbid")

    pass_id: Literal["A", "B"]
    model_id: str
    verdict: ResolutionVerdict | int    # int = option index for action_selection
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[ResolutionEvidence] = Field(default_factory=list)
    notes: str | None = None


class QuestionResolution(BaseModel):
    """Final ground truth for one question — aggregated from two passes."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_format: QuestionFormat
    # For binary: "YES"/"NO"/"UNRESOLVED". For action_selection: option index (0-based) or "UNRESOLVED".
    outcome: ResolutionVerdict | int
    agreement: bool                     # did pass A and pass B agree?
    manual_review_required: bool
    resolved_at: datetime
    passes: list[ResolutionPass]


# ── Prediction records ─────────────────────────────────────────────────────────


class BinaryPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ActionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Keys are option letters: "A", "B", ..., matching the options_block in the prompt.
    probabilities: dict[str, float]
    reasoning: str

    @field_validator("probabilities")
    @classmethod
    def _sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"probabilities must sum to ~1.0, got {total:.4f}")
        return v


class PredictionRecord(BaseModel):
    """One sample — one (question, experiment, sample_idx) triple."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    question_id: str
    experiment: ExperimentId
    sample_idx: int
    question_format: QuestionFormat

    # Reproducibility
    model_id: str
    temperature: float
    prompt_hash: str                    # sha256 of (system + user) text
    briefing_hash: str | None = None    # sha256 of the briefing if one was used

    # Answer
    binary: BinaryPrediction | None = None
    action: ActionPrediction | None = None
    raw_response: str                   # the model's full text (for audit)

    # Telemetry
    tokens_in: int
    tokens_out: int
    latency_ms: int
    created_at: datetime

    # Failures
    error: str | None = None            # parse failure or API error, if any


# ── Score summaries ────────────────────────────────────────────────────────────


class ExperimentScore(BaseModel):
    """Aggregate metrics for one experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentId
    n_questions: int
    n_samples_per_question: int
    n_resolved: int

    # Brier
    brier_raw: float
    brier_calibrated: float

    # Other metrics
    log_loss_raw: float
    log_loss_calibrated: float
    ece_raw: float
    ece_calibrated: float
    top1_accuracy: float | None = None
    top2_accuracy: float | None = None

    # Murphy decomposition
    reliability: float
    resolution: float
    uncertainty: float

    # Temperature scaling
    temperature_fit: float
