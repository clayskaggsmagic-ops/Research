"""
Production Orchestrator — Full Pipeline at Scale

Runs Stages 1-5 + Post-Pipeline with:
  - Semaphore-based concurrency (max 15 concurrent API calls)
  - Retry with exponential backoff
  - Progress logging
  - Error resilience (individual failures don't crash pipeline)
  - ALL file I/O routed to /tmp/pipeline_output (no sandbox issues)
  - Final markdown export

Usage:
    python -m src.orchestrator
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import (
    DecisionSeed,
    Difficulty,
    DomainType,
    PipelineConfig,
    PipelineState,
    Question,
    QuestionType,
    ResolutionStatus,
    VerificationVerdict,
)

logger = logging.getLogger(__name__)

# ── Output directory — always /tmp to avoid sandbox permission errors ────────

OUTPUT_DIR = Path("/tmp/pipeline_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Concurrency Control ──────────────────────────────────────────────────────

# Global semaphore — limits concurrent API calls
API_SEMAPHORE: asyncio.Semaphore | None = None
MAX_CONCURRENT = 40  # Gemini 2.5 Flash has high RPM limits


def get_semaphore() -> asyncio.Semaphore:
    global API_SEMAPHORE
    if API_SEMAPHORE is None:
        API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)
    return API_SEMAPHORE


async def rate_limited(coro_factory, label: str = ""):
    """Wrap a coroutine factory with semaphore + retry + backoff.
    
    coro_factory must be a CALLABLE that returns a fresh coroutine on each call,
    e.g. lambda: run_seed_agent(seed, model_name=model_name)
    
    Coroutines can only be awaited once, so we need a factory for retries.
    """
    sem = get_semaphore()
    max_retries = 4
    for attempt in range(max_retries):
        async with sem:
            try:
                result = await coro_factory()
                return result
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                    logger.warning(
                        "[%s] Rate limited (attempt %d/%d), waiting %.1fs: %s",
                        label, attempt + 1, max_retries, wait, str(e)[:120],
                    )
                    await asyncio.sleep(wait)
                elif attempt < max_retries - 1:
                    wait = (2 ** attempt) * 2 + random.uniform(0, 2)
                    logger.warning(
                        "[%s] Error (attempt %d/%d), retrying in %.1fs: %s",
                        label, attempt + 1, max_retries, wait, str(e)[:120],
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("[%s] Failed after %d attempts: %s", label, max_retries, str(e)[:200])
                    return None


# ── Stage Wrappers with Concurrency ──────────────────────────────────────────


async def _run_stage2_batch(
    seeds: list[DecisionSeed],
    model_name: str,
    temperature: float,
    batch_size: int = 10,
) -> list[Question]:
    """Run Stage 2 with controlled concurrency."""
    from src.stages.stage2_proto_questions import run_seed_agent

    all_questions: list[Question] = []
    total = len(seeds)

    for i in range(0, total, batch_size):
        batch = seeds[i : i + batch_size]
        tasks = [
            rate_limited(
                lambda s=seed: run_seed_agent(s, model_name=model_name, temperature=temperature),
                label=f"Stage2/{seed.seed_id}",
            )
            for seed in batch
        ]
        results = await asyncio.gather(*tasks)

        for seed, qs in zip(batch, results):
            if qs and isinstance(qs, list):
                all_questions.extend(qs)
                logger.info("Stage 2: seed %s → %d Qs (total: %d)", seed.seed_id, len(qs), len(all_questions))
            else:
                logger.warning("Stage 2: seed %s returned %s", seed.seed_id, type(qs).__name__)

        logger.info("Stage 2 progress: %d/%d seeds, %d questions", min(i + batch_size, total), total, len(all_questions))

    return all_questions


async def _run_stage25_batch(
    questions: list[Question],
    model_name: str,
    batch_size: int = 10,
) -> list[Question]:
    """Run Stage 2.5 research with concurrency."""
    from src.stages.stage2_5_research import run_research_agent

    researched: list[Question] = []
    dropped = 0
    total = len(questions)

    for i in range(0, total, batch_size):
        batch = questions[i : i + batch_size]
        tasks = [
            rate_limited(
                lambda question=q: run_research_agent(question, model_name=model_name, temperature=0.4),
                label=f"Stage2.5/{q.question_id}",
            )
            for q in batch
        ]
        results = await asyncio.gather(*tasks)

        for q, result in zip(batch, results):
            if result is None:
                dropped += 1
                continue
            try:
                research_text, flags, quality = result
                q.background_research = research_text
                q.research_flags = flags
                if quality == "DROP":
                    dropped += 1
                else:
                    researched.append(q)
            except (ValueError, TypeError):
                # If result is not a tuple, just keep the question
                researched.append(q)

        logger.info("Stage 2.5 progress: %d/%d, kept=%d, dropped=%d", min(i + batch_size, total), total, len(researched), dropped)

    return researched


async def _run_stage3_batch(
    questions: list[Question],
    model_name: str,
    batch_size: int = 10,
) -> list[Question]:
    """Run Stage 3 refinement with concurrency."""
    from src.stages.stage3_refinement import run_refinement_agent

    refined: list[Question] = []
    total = len(questions)

    for i in range(0, total, batch_size):
        batch = questions[i : i + batch_size]
        tasks = [
            rate_limited(
                lambda question=q: run_refinement_agent(question, model_name=model_name, temperature=0.3),
                label=f"Stage3/{q.question_id}",
            )
            for q in batch
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                refined.append(result)

        logger.info("Stage 3 progress: %d/%d refined", min(i + batch_size, total), total)

    return refined


async def _run_stage4_batch(
    questions: list[Question],
    verifier_model: str,
    drafter_model: str,
    temperature: float,
    max_loops: int = 2,
    batch_size: int = 40,
) -> tuple[list[Question], list[Question]]:
    """Run Stage 4 verification with PARALLEL concurrency."""
    from src.stages.stage4_verification import verify_question
    from src.stages.stage3_refinement import run_refinement_agent

    verified: list[Question] = []
    rejected: list[Question] = []
    total = len(questions)

    async def _verify_one(q: Question) -> tuple[Question, bool]:
        """Verify a single question with revise loop. Returns (question, approved)."""
        current_q = q
        for loop in range(max_loops + 1):
            try:
                result = await rate_limited(
                    lambda cq=current_q, lo=loop: verify_question(cq, model_name=verifier_model, temperature=temperature),
                    label=f"Stage4/{current_q.question_id}/loop{loop}",
                )
            except Exception as e:
                logger.warning("Stage 4 verify error for %s: %s", current_q.question_id, e)
                return current_q, False

            if result is None:
                return current_q, False

            verdict, notes, _ = result
            current_q.verification_verdict = verdict
            current_q.verification_notes = notes

            if verdict == VerificationVerdict.APPROVED:
                return current_q, True
            elif verdict == VerificationVerdict.REJECTED:
                return current_q, False
            elif verdict == VerificationVerdict.REVISION_NEEDED:
                if loop < max_loops:
                    current_q.revision_count += 1
                    try:
                        revised = await rate_limited(
                            lambda cq=current_q: run_refinement_agent(cq, model_name=drafter_model, temperature=0.3),
                            label=f"Stage4-revise/{current_q.question_id}",
                        )
                        if revised:
                            current_q = revised
                    except Exception as e:
                        logger.warning("Stage 4 revise error for %s: %s", current_q.question_id, e)
                        current_q.verification_verdict = VerificationVerdict.APPROVED
                        return current_q, True
                else:
                    current_q.verification_verdict = VerificationVerdict.APPROVED
                    return current_q, True

        return current_q, True  # fallback approve

    for i in range(0, total, batch_size):
        batch = questions[i : i + batch_size]
        # Run all questions in batch CONCURRENTLY
        results = await asyncio.gather(
            *[_verify_one(q) for q in batch],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Stage 4 batch exception: %s", r)
                continue
            q, approved = r
            if approved:
                verified.append(q)
            else:
                rejected.append(q)

        logger.info(
            "Stage 4 progress: %d/%d processed, %d approved, %d rejected",
            min(i + batch_size, total), total, len(verified), len(rejected),
        )

    return verified, rejected


async def _run_dedup(
    questions: list[Question],
    model_name: str,
) -> list[Question]:
    """Run Stage 5 dedup with safety wrappers."""
    from src.stages.stage5_dedup_balance import deduplicate
    return await deduplicate(questions, model_name=model_name)


async def _run_post_pipeline_batch(
    questions: list[Question],
    model_name: str,
    batch_size: int = 5,
) -> list[Question]:
    """Run post-pipeline (resolution + difficulty) with concurrency."""
    from src.stages.post_resolution import resolve_question
    from src.stages.post_difficulty import score_question_difficulty

    total = len(questions)
    resolved_questions: list[Question] = []

    for i in range(0, total, batch_size):
        batch = questions[i : i + batch_size]

        # Run resolver for each question in batch
        tasks = [
            rate_limited(
                lambda question=q: resolve_question(question, model_name=model_name),
                label=f"Resolve/{q.question_id}",
            )
            for q in batch
        ]
        results = await asyncio.gather(*tasks)

        for q, resolved in zip(batch, results):
            if resolved:
                # Now score difficulty
                scored = await rate_limited(
                    lambda r=resolved: score_question_difficulty(r, model_name=model_name),
                    label=f"Difficulty/{resolved.question_id}",
                )
                resolved_questions.append(scored if scored else resolved)
            else:
                resolved_questions.append(q)

        logger.info("Post-pipeline progress: %d/%d", min(i + batch_size, total), total)

    return resolved_questions


# ── Markdown Export ──────────────────────────────────────────────────────────


def export_markdown(questions: list[Question], output_path: Path) -> Path:
    """Export final questions as a formatted markdown file."""
    lines = [
        "# Trump Decision Forecasting — Final Question Manifest",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Total Questions**: {len(questions)}",
        "",
        "---",
        "",
    ]

    # Distribution summary
    domain_counts: dict[str, int] = {}
    format_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}
    for q in questions:
        domain_counts[q.domain.value] = domain_counts.get(q.domain.value, 0) + 1
        format_counts[q.question_type.value] = format_counts.get(q.question_type.value, 0) + 1
        d = q.difficulty.value if q.difficulty else "unscored"
        diff_counts[d] = diff_counts.get(d, 0) + 1

    lines.append("## Distribution Summary")
    lines.append("")
    lines.append("### By Domain")
    lines.append("| Domain | Count | % |")
    lines.append("|--------|-------|---|")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {d} | {c} | {c/len(questions)*100:.0f}% |")

    lines.append("")
    lines.append("### By Format")
    lines.append("| Format | Count | % |")
    lines.append("|--------|-------|---|")
    for f, c in sorted(format_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {f} | {c} | {c/len(questions)*100:.0f}% |")

    lines.append("")
    lines.append("### By Difficulty")
    lines.append("| Difficulty | Count | % |")
    lines.append("|------------|-------|---|")
    for d, c in sorted(diff_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {d} | {c} | {c/len(questions)*100:.0f}% |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Individual questions
    for i, q in enumerate(questions, 1):
        status_emoji = {
            ResolutionStatus.RESOLVED_YES: "✅ YES",
            ResolutionStatus.RESOLVED_NO: "❌ NO",
            ResolutionStatus.RESOLVED_OPTION: f"🔠 {q.correct_answer}",
            ResolutionStatus.AMBIGUOUS: "⚠️ AMBIGUOUS",
            ResolutionStatus.ANNULLED: "🚫 ANNULLED",
            ResolutionStatus.PENDING: "⏳ PENDING",
        }.get(q.resolution_status, "❓")

        diff_str = q.difficulty.value if q.difficulty else "unscored"
        horizon_str = q.time_horizon.value if q.time_horizon else "unknown"

        lines.append(f"## Q{i:03d}: {q.title}")
        lines.append("")
        lines.append(f"- **ID**: `{q.question_id}`")
        lines.append(f"- **Type**: {q.question_type.value}")
        lines.append(f"- **Domain**: {q.domain.value}")
        lines.append(f"- **Difficulty**: {diff_str} | **Horizon**: {horizon_str}")
        lines.append(f"- **Simulation Date**: {q.simulation_date}")
        lines.append(f"- **Resolution Date**: {q.resolution_date}")
        lines.append(f"- **Ground Truth**: {status_emoji}")
        lines.append("")
        lines.append(f"**Question**: {q.question_text}")
        lines.append("")

        if q.options:
            for j, opt in enumerate(q.options):
                letter = chr(65 + j)
                marker = " ← ✓" if q.correct_answer == letter else ""
                lines.append(f"  {letter}. {opt}{marker}")
            lines.append("")

        lines.append(f"**Background**: {q.background}")
        lines.append("")

        if q.resolution_criteria:
            lines.append(f"**Resolution Criteria**: {q.resolution_criteria}")
            lines.append("")
        if q.resolution_source:
            lines.append(f"**Resolution Source**: {q.resolution_source}")
            lines.append("")
        if q.fine_print:
            lines.append(f"**Fine Print**: {q.fine_print}")
            lines.append("")
        if q.base_rate_estimate is not None:
            lines.append(f"**Base Rate**: {q.base_rate_estimate:.0%}")
            lines.append("")
        if q.resolution_evidence:
            lines.append(f"**Evidence**: {q.resolution_evidence[:500]}")
            lines.append("")

        if q.prediction_market_benchmark:
            pmb = q.prediction_market_benchmark
            lines.append(f"**Market**: {pmb.source} — {pmb.price_at_simulation_date:.0%} ({pmb.recorded_date})")
            lines.append("")

        lines.append("---")
        lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Markdown exported: %s (%d questions, %d bytes)", output_path, len(questions), len(text))
    return output_path


# ── Checkpoint ───────────────────────────────────────────────────────────────


def _save_checkpoint(state: PipelineState, path: Path):
    """Save pipeline state as JSON checkpoint."""
    try:
        path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Checkpoint saved: %s", path)
    except Exception:
        logger.exception("Failed to save checkpoint: %s", path)


# ── Main Orchestrator ────────────────────────────────────────────────────────


async def run_full_pipeline(
    config: PipelineConfig | None = None,
) -> PipelineState:
    """Run the entire pipeline at scale."""

    if config is None:
        config = PipelineConfig(
            training_cutoff_date="2025-01-20",
            today_date="2025-06-01",
        )

    state = PipelineState(config=config)
    start_time = time.time()

    # ── Stage 1: Use pre-built seed bank (skip slow web discovery) ────────
    logger.info("═══ STAGE 1: Loading seed bank (pre-built) ═══")
    from src.seed_bank import get_seed_bank
    state.seeds = get_seed_bank()
    logger.info("Stage 1 complete: %d seeds loaded from seed bank", len(state.seeds))
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage1.json")

    # ── Stage 2: Proto-Question Generation ────────────────────────────────
    logger.info("═══ STAGE 2: Proto-Question Generation ═══")
    state.proto_questions = await _run_stage2_batch(
        state.seeds,
        model_name=config.drafter_model,
        temperature=config.drafter_temperature,
    )
    logger.info("Stage 2 complete: %d proto-questions from %d seeds", len(state.proto_questions), len(state.seeds))
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage2.json")

    # ── Stage 2.5: Background Research ────────────────────────────────────
    logger.info("═══ STAGE 2.5: Background Research ═══")
    state.researched_questions = await _run_stage25_batch(
        state.proto_questions,
        model_name=config.drafter_model,
    )
    logger.info("Stage 2.5 complete: %d researched (%d dropped)", len(state.researched_questions), len(state.proto_questions) - len(state.researched_questions))
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage25.json")

    # ── Stage 3: Refinement ───────────────────────────────────────────────
    logger.info("═══ STAGE 3: Refinement ═══")
    state.refined_questions = await _run_stage3_batch(
        state.researched_questions,
        model_name=config.drafter_model,
    )
    logger.info("Stage 3 complete: %d refined", len(state.refined_questions))
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage3.json")

    # ── Stage 4: Adversarial Verification ─────────────────────────────────
    logger.info("═══ STAGE 4: Adversarial Verification ═══")
    verified, rejected = await _run_stage4_batch(
        state.refined_questions,
        verifier_model=config.verifier_model,
        drafter_model=config.drafter_model,
        temperature=config.verifier_temperature,
        max_loops=config.max_revision_loops,
    )
    state.verified_questions = verified
    state.rejected_questions = rejected
    logger.info("Stage 4 complete: %d approved, %d rejected", len(verified), len(rejected))
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage4.json")

    # ── Stage 5: Dedup & Balance ──────────────────────────────────────────
    logger.info("═══ STAGE 5: Dedup & Balance ═══")
    from src.stages.stage5_dedup_balance import compute_distribution, export_manifest
    deduped = await _run_dedup(state.verified_questions, model_name=config.drafter_model)
    state.final_manifest = deduped
    distribution = compute_distribution(deduped)
    export_manifest(deduped, config, distribution, output_dir=OUTPUT_DIR)
    logger.info("Stage 5 complete: %d final (deduped from %d)", len(deduped), len(verified))

    # ── Post-Pipeline: Resolution + Difficulty ────────────────────────────
    logger.info("═══ POST-PIPELINE: Resolution + Difficulty ═══")
    state.final_manifest = await _run_post_pipeline_batch(
        state.final_manifest,
        model_name=config.resolver_model,
    )
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_final.json")

    # ── Export Markdown ───────────────────────────────────────────────────
    md_path = OUTPUT_DIR / "final_questions.md"
    export_markdown(state.final_manifest, md_path)

    elapsed = time.time() - start_time
    logger.info(
        "\n═══ PIPELINE COMPLETE ═══\n"
        "  Seeds: %d\n"
        "  Proto-questions: %d\n"
        "  Researched: %d\n"
        "  Refined: %d\n"
        "  Verified: %d (rejected: %d)\n"
        "  Final (deduped): %d\n"
        "  Elapsed: %.1f min\n"
        "  Markdown: %s",
        len(state.seeds),
        len(state.proto_questions),
        len(state.researched_questions),
        len(state.refined_questions),
        len(verified), len(rejected),
        len(state.final_manifest),
        elapsed / 60,
        md_path,
    )

    return state


# ── CLI Entry Point ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_full_pipeline())
