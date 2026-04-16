"""
Stage 5 — Deduplication & Batch Balancing (Bosse et al. A.20)

1. LLM-based pairwise similarity scoring (1-4 scale)
   - Remove questions scoring 3+ against any other (keep better-written)
   - Paper achieved mean intra-cluster similarity of 1.32/4.0

2. Distribution balancing — flag imbalances against targets:
   Domain:     trade_tariffs ~25%, executive_orders ~20%, personnel ~15%,
               foreign_policy ~15%, legislative ~10%, public_comms ~10%,
               legal_judicial ~5%
   Difficulty:  easy ~20%, medium ~60%, hard ~20%
   Format:     binary ~60%, action_selection ~40%

3. Versioned manifest export as JSON with full metadata.

Entry point: run_stage5(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import MANIFEST_DIR
from src.schemas import (
    DomainType,
    PipelineConfig,
    PipelineState,
    Question,
    QuestionType,
)

logger = logging.getLogger(__name__)


# ── Target Distributions ────────────────────────────────────────────────────

DOMAIN_TARGETS = {
    DomainType.TRADE_TARIFFS: 0.25,
    DomainType.EXECUTIVE_ORDERS: 0.20,
    DomainType.PERSONNEL: 0.15,
    DomainType.FOREIGN_POLICY: 0.15,
    DomainType.LEGISLATIVE: 0.10,
    DomainType.PUBLIC_COMMS: 0.10,
    DomainType.LEGAL_JUDICIAL: 0.05,
}

DIFFICULTY_TARGETS = {"easy": 0.20, "medium": 0.60, "hard": 0.20}
FORMAT_TARGETS = {QuestionType.BINARY: 0.60, QuestionType.ACTION_SELECTION: 0.40}


# ── LLM Similarity Scoring ──────────────────────────────────────────────────


class SimilarityScore(BaseModel):
    """Pairwise similarity between two forecasting questions."""
    question_a_id: str
    question_b_id: str
    score: int = Field(
        description="1=completely different topics, 2=same broad domain but different specifics, "
        "3=substantially overlapping in scope, 4=effectively duplicate questions"
    )
    reasoning: str = Field(description="Brief explanation of why this score")
    keep: str = Field(description="ID of the better-written question to keep if score >= 3")


SIMILARITY_PROMPT = """\
You are a deduplication judge for a forecasting tournament. Compare these two \
questions and score their similarity on a 1-4 scale:

1 = Completely different topics (e.g., tariffs vs. Supreme Court nomination)
2 = Same broad domain but different specifics (e.g., both about tariffs but \
different countries/products)
3 = Substantially overlapping — a forecaster's research for one largely \
transfers to the other
4 = Effectively duplicate — same event, same resolution, minor wording differences

## Question A
ID: {id_a}
Title: {title_a}
Question: {text_a}
Domain: {domain_a}

## Question B
ID: {id_b}
Title: {title_b}
Question: {text_b}
Domain: {domain_b}

If score >= 3, indicate which question is better-written (clearer criteria, \
better options, more precise resolution source) in the 'keep' field. If < 3, \
set keep to question_a_id.\
"""


async def score_pair(
    q_a: Question,
    q_b: Question,
    llm: ChatGoogleGenerativeAI,
) -> SimilarityScore | None:
    """Score similarity between two questions."""
    prompt = SIMILARITY_PROMPT.format(
        id_a=q_a.question_id,
        title_a=q_a.title,
        text_a=q_a.question_text,
        domain_a=q_a.domain.value,
        id_b=q_b.question_id,
        title_b=q_b.title,
        text_b=q_b.question_text,
        domain_b=q_b.domain.value,
    )
    try:
        result = await llm.with_structured_output(SimilarityScore).ainvoke(prompt)
        return result
    except Exception:
        logger.exception("Similarity scoring failed for %s vs %s", q_a.question_id, q_b.question_id)
        return None


async def deduplicate(
    questions: list[Question],
    model_name: str = "gemini-3-pro-preview",
    threshold: int = 3,
) -> list[Question]:
    """
    LLM-based pairwise deduplication (Bosse A.20).

    For each pair scoring >= threshold, drop the worse question.
    Returns the deduplicated list.
    """
    if len(questions) <= 1:
        return questions

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)

    # Score all pairs
    pairs = list(combinations(questions, 2))
    logger.info("Scoring %d pairs for deduplication...", len(pairs))

    drop_ids: set[str] = set()

    # Process in batches to avoid rate limits
    import asyncio

    batch_size = 10
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        tasks = [score_pair(q_a, q_b, llm) for q_a, q_b in batch]
        results = await asyncio.gather(*tasks)

        for (q_a, q_b), score in zip(batch, results):
            if score and score.score >= threshold:
                # Drop the worse one
                drop_id = q_b.question_id if score.keep == q_a.question_id else q_a.question_id
                drop_ids.add(drop_id)
                logger.info(
                    "DEDUP: %s vs %s → score=%d, dropping %s (%s)",
                    q_a.question_id, q_b.question_id, score.score, drop_id, score.reasoning[:80],
                )

    deduped = [q for q in questions if q.question_id not in drop_ids]
    logger.info("Deduplication: %d → %d questions (%d dropped)", len(questions), len(deduped), len(drop_ids))

    return deduped


# ── Distribution Analysis ────────────────────────────────────────────────────


def compute_distribution(questions: list[Question]) -> dict:
    """Compute actual distributions and compare against targets."""
    n = len(questions) or 1

    # Domain distribution
    domain_counts = {}
    for q in questions:
        domain_counts[q.domain] = domain_counts.get(q.domain, 0) + 1
    domain_dist = {d.value: domain_counts.get(d, 0) / n for d in DomainType}
    domain_imbalances = {}
    for d in DomainType:
        actual = domain_counts.get(d, 0) / n
        target = DOMAIN_TARGETS.get(d, 0)
        diff = actual - target
        if abs(diff) > 0.10:  # flag if >10pp off target
            domain_imbalances[d.value] = {"actual": round(actual, 3), "target": target, "diff": round(diff, 3)}

    # Format distribution
    format_counts = {}
    for q in questions:
        format_counts[q.question_type] = format_counts.get(q.question_type, 0) + 1
    format_dist = {qt.value: format_counts.get(qt, 0) / n for qt in QuestionType}

    # Difficulty distribution (may be None for some questions)
    diff_counts = {"easy": 0, "medium": 0, "hard": 0, "unscored": 0}
    for q in questions:
        if q.difficulty:
            diff_counts[q.difficulty.value] = diff_counts.get(q.difficulty.value, 0) + 1
        else:
            diff_counts["unscored"] += 1
    diff_dist = {k: v / n for k, v in diff_counts.items()}

    return {
        "total_questions": len(questions),
        "domain_distribution": {k: round(v, 3) for k, v in domain_dist.items()},
        "domain_imbalances": domain_imbalances,
        "format_distribution": {k: round(v, 3) for k, v in format_dist.items()},
        "difficulty_distribution": {k: round(v, 3) for k, v in diff_dist.items()},
    }


# ── Manifest Export ──────────────────────────────────────────────────────────


def export_manifest(
    questions: list[Question],
    config: PipelineConfig,
    distribution: dict,
    output_dir: Path | None = None,
) -> Path:
    """
    Export the final question manifest as versioned JSON.

    Includes: pipeline version, model versions, date, distribution stats,
    and all question objects.
    """
    output_dir = output_dir or MANIFEST_DIR

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"manifest_{timestamp}.json"

    manifest = {
        "metadata": {
            "pipeline_version": "0.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": {
                "drafter": config.drafter_model,
                "verifier": config.verifier_model,
                "resolver": config.resolver_model,
                "prediction": config.prediction_model,
            },
            "leader": config.leader,
            "training_cutoff_date": config.training_cutoff_date,
        },
        "distribution": distribution,
        "questions": [q.model_dump(mode="json") for q in questions],
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        pass

    filepath = output_dir / filename
    filepath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Manifest exported: %s (%d questions)", filepath, len(questions))

    return filepath


# ── Entry Point ──────────────────────────────────────────────────────────────


async def run_stage5(state: PipelineState) -> PipelineState:
    """
    Stage 5 entry point — dedup, balance-check, and export manifest.

    1. LLM-based pairwise deduplication
    2. Compute distribution stats and flag imbalances
    3. Export versioned JSON manifest
    4. Store result in state.final_manifest
    """
    config = state.config
    questions = state.verified_questions

    logger.info("Stage 5 starting — %d verified questions", len(questions))

    # ── Step 1: Deduplicate ───────────────────────────────────────────────
    deduped = await deduplicate(questions, model_name=config.drafter_model)

    # ── Step 2: Distribution analysis ─────────────────────────────────────
    distribution = compute_distribution(deduped)

    if distribution["domain_imbalances"]:
        logger.warning(
            "Domain imbalances detected: %s",
            json.dumps(distribution["domain_imbalances"], indent=2),
        )

    # ── Step 3: Export manifest ───────────────────────────────────────────
    manifest_path = export_manifest(deduped, config, distribution)

    # ── Step 4: Store in state ────────────────────────────────────────────
    state.final_manifest = deduped

    logger.info(
        "Stage 5 complete — %d final questions, manifest at %s",
        len(deduped), manifest_path,
    )

    return state
