#!/usr/bin/env python3
"""
run_all.py — Single entry point for the Trump Decision Forecasting Pipeline.

Runs the full Bosse et al. (2026) pipeline: Seeds → Proto-Questions → Research
→ Refinement → Verification → Dedup → Resolution + Difficulty → Markdown Export.

Features:
  - Loads .env automatically (GOOGLE_API_KEY, TAVILY_API_KEY required)
  - Checkpoint resume: skips completed stages if checkpoints exist
  - Exponential backoff with jitter on all API calls
  - Semaphore-based concurrency (configurable)
  - Progress saved at every stage boundary
  - Final output copied to evaluation_plan/output/

Usage:
    # Full fresh run (clears old checkpoints):
    uv run python run_all.py --fresh

    # Resume from last checkpoint:
    uv run python run_all.py

    # Skip verification (faster, uses programmatic export):
    uv run python run_all.py --skip-stage4

    # Custom concurrency and model:
    uv run python run_all.py --concurrency 10 --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── .env loading ─────────────────────────────────────────────────────────────

def load_dotenv_manual():
    """Load .env file without requiring python-dotenv to be installed."""
    candidates = [
        Path("/tmp/pipeline_output/.env"),
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",
    ]
    for env_path in candidates:
        try:
            if not env_path.exists():
                continue
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if value and key not in os.environ:
                        os.environ[key] = value
            print(f"✅ Loaded .env from {env_path}")
            return True
        except (PermissionError, OSError):
            continue
    return False


def check_env():
    """Verify required environment variables are set."""
    missing = []
    for key in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print(f"   Create a .env file in {Path(__file__).parent} with:")
        for k in missing:
            print(f"     {k}=your_key_here")
        sys.exit(1)


# ── Output paths ─────────────────────────────────────────────────────────────

PIPELINE_DIR = Path(__file__).parent
OUTPUT_DIR = Path("/tmp/pipeline_output")
FINAL_OUTPUT_DIR = PIPELINE_DIR.parent / "evaluation_plan" / "output"


# ── Checkpoint management ────────────────────────────────────────────────────

STAGE_CHECKPOINTS = {
    "stage1": "checkpoint_stage1.json",
    "stage2": "checkpoint_stage2.json",
    "stage25": "checkpoint_stage25.json",
    "stage3": "checkpoint_stage3.json",
    "stage4": "checkpoint_stage4.json",
    "final": "checkpoint_final.json",
}


def get_last_checkpoint() -> str | None:
    """Find the most advanced checkpoint that exists."""
    last = None
    for stage in ("stage1", "stage2", "stage25", "stage3", "stage4", "final"):
        path = OUTPUT_DIR / STAGE_CHECKPOINTS[stage]
        if path.exists():
            last = stage
    return last


def clear_checkpoints():
    """Remove all checkpoint files for a fresh run."""
    for fname in STAGE_CHECKPOINTS.values():
        path = OUTPUT_DIR / fname
        if path.exists():
            path.unlink()
            print(f"  🗑  Removed {fname}")


def load_checkpoint(stage: str) -> dict | None:
    """Load a specific checkpoint."""
    path = OUTPUT_DIR / STAGE_CHECKPOINTS[stage]
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── Copy results to final output directory ───────────────────────────────────

def copy_to_final_output():
    """Copy the finished markdown and JSON to evaluation_plan/output/."""
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in ("final_questions.md", "final_manifest.json"):
        src = OUTPUT_DIR / fname
        dst = FINAL_OUTPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  📋 Copied {fname} → {dst}")


# ── Main pipeline runner ─────────────────────────────────────────────────────

async def run_pipeline(args: argparse.Namespace):
    """Run the full pipeline with checkpoint resume."""

    from src.orchestrator import (
        OUTPUT_DIR as _,
        MAX_CONCURRENT,
        run_full_pipeline,
        export_markdown,
        _save_checkpoint,
        _run_stage2_batch,
        _run_stage25_batch,
        _run_stage3_batch,
        _run_stage4_batch,
        _run_dedup,
        _run_post_pipeline_batch,
    )
    from src.schemas import PipelineConfig, PipelineState

    # Update concurrency if specified
    if args.concurrency:
        import src.orchestrator as orch
        orch.MAX_CONCURRENT = args.concurrency
        orch.API_SEMAPHORE = None  # Force re-creation

    config = PipelineConfig(
        training_cutoff_date="2025-01-20",
        today_date="2025-06-01",
        drafter_model=args.model,
        verifier_model=args.verifier_model,
        resolver_model=args.model,
    )

    # Determine resume point
    last_ckpt = get_last_checkpoint() if not args.fresh else None
    if last_ckpt:
        print(f"\n🔄 Resuming from checkpoint: {last_ckpt}")
    else:
        print("\n🚀 Starting fresh pipeline run")

    state = PipelineState(config=config)
    start_time = time.time()

    # ── Stage 1: Seeds ────────────────────────────────────────────────────
    if last_ckpt and last_ckpt in ("stage1", "stage2", "stage25", "stage3", "stage4", "final"):
        ckpt = load_checkpoint("stage1")
        from src.schemas import DecisionSeed
        state.seeds = [DecisionSeed(**s) for s in ckpt["seeds"]]
        print(f"  ✅ Stage 1: {len(state.seeds)} seeds (from checkpoint)")
    else:
        print("\n═══ STAGE 1: Loading Seeds ═══")
        from src.seed_bank import get_seed_bank
        state.seeds = get_seed_bank()
        _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage1.json")
        print(f"  ✅ Stage 1: {len(state.seeds)} seeds loaded")

    # ── Stage 2: Proto-Question Generation ────────────────────────────────
    if last_ckpt and last_ckpt in ("stage2", "stage25", "stage3", "stage4", "final"):
        ckpt = load_checkpoint("stage2")
        from src.schemas import Question
        state.proto_questions = [Question(**q) for q in ckpt["proto_questions"]]
        print(f"  ✅ Stage 2: {len(state.proto_questions)} proto-questions (from checkpoint)")
    else:
        print("\n═══ STAGE 2: Proto-Question Generation ═══")
        # Per-batch checkpointing: load any partial progress
        partial_path = OUTPUT_DIR / "checkpoint_stage2_partial.json"
        accumulated = []
        done_seed_ids = set()
        if partial_path.exists():
            try:
                partial = json.loads(partial_path.read_text())
                accumulated = partial.get("questions", [])
                done_seed_ids = set(partial.get("done_seed_ids", []))
                print(f"  🔄 Resuming Stage 2: {len(accumulated)} Qs from {len(done_seed_ids)} seeds already done")
            except Exception:
                pass

        remaining_seeds = [s for s in state.seeds if s.seed_id not in done_seed_ids]
        bs = args.batch_size
        for i in range(0, len(remaining_seeds), bs):
            batch = remaining_seeds[i:i + bs]
            batch_ids = [s.seed_id for s in batch]
            print(f"  📦 Batch {i // bs + 1}: seeds {batch_ids}")
            try:
                batch_qs = await _run_stage2_batch(
                    batch,
                    model_name=config.drafter_model,
                    temperature=config.drafter_temperature,
                    batch_size=bs,
                )
                for q in batch_qs:
                    accumulated.append(q.model_dump(mode="json"))
                done_seed_ids.update(batch_ids)
            except Exception as e:
                logger.warning("Batch failed (%s), saving progress and continuing", e)
                # Mark these seeds as done even on failure to avoid infinite retry
                done_seed_ids.update(batch_ids)

            # Save partial checkpoint after every batch
            partial_path.write_text(json.dumps({
                "done_seed_ids": list(done_seed_ids),
                "questions": accumulated,
            }, indent=2, default=str))
            print(f"  💾 Saved partial: {len(accumulated)} Qs from {len(done_seed_ids)}/{len(state.seeds)} seeds")

        # Reconstruct Question objects
        from src.schemas import Question
        state.proto_questions = [Question(**q) for q in accumulated]
        _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage2.json")
        # Clean up partial
        if partial_path.exists():
            partial_path.unlink()
        print(f"  ✅ Stage 2: {len(state.proto_questions)} proto-questions generated")

    # ── Stage 2.5: Research ───────────────────────────────────────────────
    if last_ckpt and last_ckpt in ("stage25", "stage3", "stage4", "final"):
        ckpt = load_checkpoint("stage25")
        from src.schemas import Question
        state.researched_questions = [Question(**q) for q in ckpt["researched_questions"]]
        print(f"  ✅ Stage 2.5: {len(state.researched_questions)} researched (from checkpoint)")
    else:
        print("\n═══ STAGE 2.5: Background Research ═══")
        state.researched_questions = await _run_stage25_batch(
            state.proto_questions,
            model_name=config.drafter_model,
            batch_size=args.batch_size,
        )
        _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage25.json")
        dropped = len(state.proto_questions) - len(state.researched_questions)
        print(f"  ✅ Stage 2.5: {len(state.researched_questions)} researched ({dropped} dropped)")

    # ── Stage 3: Refinement ───────────────────────────────────────────────
    if last_ckpt and last_ckpt in ("stage3", "stage4", "final"):
        ckpt = load_checkpoint("stage3")
        from src.schemas import Question
        state.refined_questions = [Question(**q) for q in ckpt["refined_questions"]]
        print(f"  ✅ Stage 3: {len(state.refined_questions)} refined (from checkpoint)")
    else:
        print("\n═══ STAGE 3: Refinement ═══")
        state.refined_questions = await _run_stage3_batch(
            state.researched_questions,
            model_name=config.drafter_model,
            batch_size=args.batch_size,
        )
        _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage3.json")
        print(f"  ✅ Stage 3: {len(state.refined_questions)} refined")

    # ── Stage 4: Verification (optional) ──────────────────────────────────
    if args.skip_stage4:
        print("\n⏩ Skipping Stage 4 (--skip-stage4)")
        state.verified_questions = state.refined_questions
        state.rejected_questions = []
    elif last_ckpt and last_ckpt in ("stage4", "final"):
        ckpt = load_checkpoint("stage4")
        from src.schemas import Question
        state.verified_questions = [Question(**q) for q in ckpt.get("verified_questions", [])]
        state.rejected_questions = [Question(**q) for q in ckpt.get("rejected_questions", [])]
        print(f"  ✅ Stage 4: {len(state.verified_questions)} verified, {len(state.rejected_questions)} rejected (from checkpoint)")
    else:
        print("\n═══ STAGE 4: Adversarial Verification ═══")
        verified, rejected = await _run_stage4_batch(
            state.refined_questions,
            verifier_model=config.verifier_model,
            drafter_model=config.drafter_model,
            temperature=config.verifier_temperature,
            max_loops=config.max_revision_loops,
            batch_size=args.batch_size,  # Full parallel batches
        )
        state.verified_questions = verified
        state.rejected_questions = rejected
        _save_checkpoint(state, OUTPUT_DIR / "checkpoint_stage4.json")
        print(f"  ✅ Stage 4: {len(verified)} verified, {len(rejected)} rejected")

    # ── Stage 5: Dedup ────────────────────────────────────────────────────
    if last_ckpt == "final":
        ckpt = load_checkpoint("final")
        from src.schemas import Question
        state.final_manifest = [Question(**q) for q in ckpt.get("final_manifest", [])]
        print(f"  ✅ Stage 5+Post: {len(state.final_manifest)} final (from checkpoint)")
    else:
        print("\n═══ STAGE 5: Dedup & Balance ═══")
        try:
            state.final_manifest = await _run_dedup(
                state.verified_questions,
                model_name=config.drafter_model,
            )
        except Exception as e:
            logger.warning("Stage 5 dedup failed (%s), using verified set directly", e)
            state.final_manifest = state.verified_questions
        print(f"  ✅ Stage 5: {len(state.final_manifest)} after dedup")

        # ── Post-Pipeline: Resolution + Difficulty ────────────────────────
        print("\n═══ POST-PIPELINE: Resolution + Difficulty ═══")
        try:
            state.final_manifest = await _run_post_pipeline_batch(
                state.final_manifest,
                model_name=config.resolver_model,
                batch_size=args.batch_size,
            )
        except Exception as e:
            logger.warning("Post-pipeline scoring failed (%s), exporting as-is", e)

        _save_checkpoint(state, OUTPUT_DIR / "checkpoint_final.json")

    # ── Export ─────────────────────────────────────────────────────────────
    print("\n═══ EXPORTING ═══")
    md_path = OUTPUT_DIR / "final_questions.md"
    export_markdown(state.final_manifest, md_path)

    # Also export JSON
    json_path = OUTPUT_DIR / "final_manifest.json"
    json_path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "generated": datetime.now(timezone.utc).isoformat(),
                "pipeline": "bosse2026-trump-specific",
                "model": config.drafter_model,
                "total_questions": len(state.final_manifest),
                "questions": [q.model_dump(mode="json") for q in state.final_manifest],
            },
            indent=2,
            default=str,
        )
    )

    # Copy to final output directory
    copy_to_final_output()

    elapsed = time.time() - start_time
    print(f"""
═══════════════════════════════════════════════
  PIPELINE COMPLETE
═══════════════════════════════════════════════
  Seeds:           {len(state.seeds)}
  Proto-questions:  {len(state.proto_questions)}
  Researched:       {len(state.researched_questions)}
  Refined:          {len(state.refined_questions)}
  Verified:         {len(state.verified_questions)} (rejected: {len(state.rejected_questions)})
  Final:            {len(state.final_manifest)}
  Elapsed:          {elapsed / 60:.1f} min
  Output:           {FINAL_OUTPUT_DIR}
═══════════════════════════════════════════════
""")

    return state


# ── Programmatic export (no API needed) ──────────────────────────────────────

def run_export_only():
    """Run the programmatic export from the latest checkpoint, no API needed."""
    from src.export_final import main as export_main
    export_main()
    copy_to_final_output()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trump Decision Forecasting Pipeline — Full Run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear all checkpoints and start from scratch",
    )
    parser.add_argument(
        "--skip-stage4", action="store_true",
        help="Skip adversarial verification (much faster, uses all refined Qs)",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Just re-export from the latest checkpoint (no API calls)",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Drafter/resolver model (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--verifier-model", default="gemini-2.5-flash",
        help="Verifier model for Stage 4 (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=15,
        help="Max concurrent API calls (default: 15)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size for parallel processing (default: 10)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="a"),
        ],
    )

    # Ensure output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load environment
    if load_dotenv_manual():
        print("✅ Loaded .env")
    else:
        print("⚠️  No .env file found, using existing environment variables")

    if args.export_only:
        run_export_only()
        return

    check_env()

    if args.fresh:
        print("\n🧹 Clearing old checkpoints...")
        clear_checkpoints()

    # Show resume status
    last = get_last_checkpoint()
    if last:
        print(f"📍 Last checkpoint: {last}")

    print(f"🤖 Drafter model:  {args.model}")
    print(f"🔍 Verifier model: {args.verifier_model}")
    print(f"⚡ Concurrency:    {args.concurrency}")
    print(f"📦 Batch size:     {args.batch_size}")

    asyncio.run(run_pipeline(args))


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
