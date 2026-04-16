"""
Resume script — loads Stage 2 checkpoint and runs Stages 2.5-5 + Post-Pipeline
using gemini-2.5-flash exclusively (to bypass gemini-2.5-pro daily quota).

Usage:
    python -m src.resume_pipeline
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import (
    PipelineConfig,
    PipelineState,
    Question,
)
from src.orchestrator import (
    OUTPUT_DIR,
    _run_stage25_batch,
    _run_stage3_batch,
    _run_stage4_batch,
    _run_dedup,
    _run_post_pipeline_batch,
    _save_checkpoint,
    export_markdown,
    MAX_CONCURRENT,
)

logger = logging.getLogger(__name__)


async def resume_from_stage2():
    """Load checkpoint and resume from Stage 2.5 with flash model."""

    # ── Load checkpoint ──────────────────────────────────────────────────
    ckpt_path = OUTPUT_DIR / "checkpoint_stage2.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt_data = json.loads(ckpt_path.read_text())

    # Reconstruct PipelineState
    state = PipelineState.model_validate(ckpt_data)
    logger.info(
        "Checkpoint loaded: %d seeds, %d proto_questions",
        len(state.seeds),
        len(state.proto_questions),
    )

    # Override config to use flash for EVERYTHING
    config = PipelineConfig(
        training_cutoff_date="2025-01-20",
        today_date="2025-06-01",
        drafter_model="gemini-2.5-flash",
        verifier_model="gemini-2.5-flash",
        resolver_model="gemini-2.5-flash",
        prediction_model="gemini-2.5-flash",
    )
    state.config = config

    start_time = time.time()

    # ── Stage 2.5: Background Research ────────────────────────────────────
    logger.info("═══ STAGE 2.5: Background Research (FLASH) ═══")
    state.researched_questions = await _run_stage25_batch(
        state.proto_questions,
        model_name=config.drafter_model,
    )
    logger.info(
        "Stage 2.5 complete: %d researched (%d dropped)",
        len(state.researched_questions),
        len(state.proto_questions) - len(state.researched_questions),
    )
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage25.json")

    # ── Stage 3: Refinement ───────────────────────────────────────────────
    logger.info("═══ STAGE 3: Refinement (FLASH) ═══")
    state.refined_questions = await _run_stage3_batch(
        state.researched_questions,
        model_name=config.drafter_model,
    )
    logger.info("Stage 3 complete: %d refined", len(state.refined_questions))
    _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage3.json")

    # ── Stage 4: Adversarial Verification ─────────────────────────────────
    logger.info("═══ STAGE 4: Adversarial Verification (FLASH) ═══")
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
    logger.info("═══ STAGE 5: Dedup & Balance (FLASH) ═══")
    from src.stages.stage5_dedup_balance import compute_distribution, export_manifest
    deduped = await _run_dedup(state.verified_questions, model_name=config.drafter_model)
    state.final_manifest = deduped
    distribution = compute_distribution(deduped)
    export_manifest(deduped, config, distribution, output_dir=OUTPUT_DIR)
    logger.info("Stage 5 complete: %d final (deduped from %d)", len(deduped), len(verified))

    # ── Post-Pipeline: Resolution + Difficulty ────────────────────────────
    logger.info("═══ POST-PIPELINE: Resolution + Difficulty (FLASH) ═══")
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
        "\n═══ PIPELINE COMPLETE (RESUME) ═══\n"
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(resume_from_stage2())
