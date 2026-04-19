"""CHRONOS Research Pipeline — LangGraph StateGraph wiring all agents.

This is the orchestration layer. It connects all 6 agents into a directed
graph with conditional routing:

    START → coordinator → discovery → extraction → cleaning
         → temporal_validator → indexing → coverage_auditor → CONDITIONAL
                                                                 ↓
                                                     complete? → END
                                                     gaps?     → coordinator (loop)

The coverage loop runs up to MAX_LOOPS iterations (default 5).
Checkpoint/resume: every node auto-saves state after success.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date
from typing import Callable

from langgraph.graph import END, StateGraph

from .checkpoint import (
    clean_checkpoints,
    get_resume_node,
    list_checkpoints,
    load_latest_checkpoint,
    save_checkpoint,
    NODE_ORDER,
)
from .config import settings
from .models import SwarmState

# Agent node imports
from .agents.coordinator import coordinator_node
from .agents.discovery import discovery_node
from .agents.extraction import extraction_node
from .agents.cleaning import cleaning_node
from .agents.temporal_validator import temporal_validator_node
from .agents.indexing import indexing_node
from .agents.coverage_auditor import coverage_auditor_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_LOOPS = 5


# ---------------------------------------------------------------------------
# Checkpoint-aware node wrapper
# ---------------------------------------------------------------------------

def checkpoint_wrapper(
    node_fn: Callable,
    node_name: str,
) -> Callable:
    """Wrap a LangGraph node to auto-save a checkpoint after success."""

    async def wrapped(state: SwarmState) -> SwarmState:
        logger.info(f"▶ Entering node: {node_name}")
        result = await node_fn(state)

        # Save checkpoint
        try:
            result.last_completed_node = node_name
            save_checkpoint(result, node_name=node_name, run_id=result.run_id)
        except Exception as e:
            logger.warning(f"[checkpoint] Failed to save after {node_name}: {e}")
            # Non-fatal — don't break the pipeline

        return result

    wrapped.__name__ = node_fn.__name__
    return wrapped


# ---------------------------------------------------------------------------
# Safety wrapper — loop counter + max-loop guard
# ---------------------------------------------------------------------------

async def loop_guard_node(state: SwarmState) -> SwarmState:
    """Increment loop counter and check max iterations.

    This runs BEFORE coordinator on loop-back to prevent infinite loops.
    """
    state.loop_count += 1
    logger.info(f"=== Research Loop {state.loop_count}/{MAX_LOOPS} ===")

    if state.loop_count > MAX_LOOPS:
        state.errors.append(
            f"Maximum loop count ({MAX_LOOPS}) reached — stopping collection. "
            f"Coverage gaps may remain: {state.coverage_gaps}"
        )
        state.collection_complete = True  # Force stop

    return state


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def should_continue(state: SwarmState) -> str:
    """Route after coverage_auditor: continue looping or end."""
    if state.collection_complete:
        return "end"
    if state.loop_count >= MAX_LOOPS:
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build the CHRONOS research pipeline StateGraph.

    Every node is wrapped with checkpoint_wrapper for auto-save.
    """
    graph = StateGraph(SwarmState)

    # Wrap all nodes with checkpoint saving
    wrapped_nodes = {
        "loop_guard": checkpoint_wrapper(loop_guard_node, "loop_guard"),
        "coordinator": checkpoint_wrapper(coordinator_node, "coordinator"),
        "discovery": checkpoint_wrapper(discovery_node, "discovery"),
        "extraction": checkpoint_wrapper(extraction_node, "extraction"),
        "cleaning": checkpoint_wrapper(cleaning_node, "cleaning"),
        "temporal_validator": checkpoint_wrapper(temporal_validator_node, "temporal_validator"),
        "indexing": checkpoint_wrapper(indexing_node, "indexing"),
        "coverage_auditor": checkpoint_wrapper(coverage_auditor_node, "coverage_auditor"),
    }

    # --- Add nodes ---
    for name, fn in wrapped_nodes.items():
        graph.add_node(name, fn)

    # --- Linear edges: the main pipeline ---
    graph.set_entry_point("loop_guard")
    graph.add_edge("loop_guard", "coordinator")
    graph.add_edge("coordinator", "discovery")
    graph.add_edge("discovery", "extraction")
    graph.add_edge("extraction", "cleaning")
    graph.add_edge("cleaning", "temporal_validator")
    graph.add_edge("temporal_validator", "indexing")
    graph.add_edge("indexing", "coverage_auditor")

    # --- Conditional edge: coverage loop ---
    graph.add_conditional_edges(
        "coverage_auditor",
        should_continue,
        {
            "continue": "loop_guard",
            "end": END,
        },
    )

    return graph


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    subject_name: str,
    start_date: date,
    end_date: date,
    resume_run_id: str | None = None,
) -> SwarmState:
    """Run the full CHRONOS research pipeline.

    Args:
        subject_name: Leader name (e.g., "Donald J. Trump")
        start_date: Collection start date
        end_date: Collection end date
        resume_run_id: If set, resume from last checkpoint of this run

    Returns:
        Final SwarmState with all results
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    # --- Resume from checkpoint if requested ---
    initial_state = None
    if resume_run_id:
        checkpoint = load_latest_checkpoint(run_id=resume_run_id)
        if checkpoint:
            initial_state, last_node = checkpoint
            resume_node = get_resume_node(last_node)
            if use_rich:
                console.print(Panel(
                    f"[bold yellow]Resuming from checkpoint[/bold yellow]\n\n"
                    f"Run ID: {resume_run_id}\n"
                    f"Last completed: {last_node}\n"
                    f"Resuming at: {resume_node}\n"
                    f"Loop: {initial_state.loop_count}",
                    title="♻️ Resume Mode",
                    border_style="yellow",
                ))
            else:
                print(f"RESUME: run={resume_run_id} last={last_node} next={resume_node}")
        else:
            if use_rich:
                console.print(f"[bold red]No checkpoint found for run_id={resume_run_id}[/bold red]")
            else:
                print(f"ERROR: No checkpoint for run_id={resume_run_id}")
            return SwarmState()

    if initial_state is None:
        initial_state = SwarmState(
            subject_name=subject_name,
            collection_start=start_date,
            collection_end=end_date,
        )

    if use_rich:
        console.print(Panel(
            f"[bold cyan]CHRONOS Research Swarm[/bold cyan]\n\n"
            f"Subject: [bold]{initial_state.subject_name}[/bold]\n"
            f"Window: {initial_state.collection_start} → {initial_state.collection_end}\n"
            f"Run ID: {initial_state.run_id}\n"
            f"Max loops: {MAX_LOOPS}",
            title="🔬 Starting Pipeline",
            border_style="cyan",
        ))
    else:
        print(f"CHRONOS: {initial_state.subject_name} | "
              f"{initial_state.collection_start} → {initial_state.collection_end} | "
              f"run={initial_state.run_id}")

    # Build and compile graph
    graph = build_graph()
    compiled = graph.compile()

    # Run the pipeline
    logger.info(
        f"Starting pipeline for {initial_state.subject_name} "
        f"({initial_state.collection_start} → {initial_state.collection_end}) "
        f"run_id={initial_state.run_id}"
    )

    try:
        final_state = await compiled.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if use_rich:
            console.print(f"[bold red]Pipeline error:[/bold red] {e}")
            console.print(
                f"[yellow]State saved in checkpoints. "
                f"Resume with: --resume {initial_state.run_id}[/yellow]"
            )
        raise

    # Clean old checkpoints (keep last 3 per run)
    clean_checkpoints(run_id=initial_state.run_id, keep_latest=3)

    # Print summary
    _print_summary(final_state, console if use_rich else None)

    return final_state


def _print_summary(state, console=None) -> None:
    """Print a summary report of the pipeline run."""
    # LangGraph ainvoke returns a dict, not a SwarmState
    if isinstance(state, dict):
        get = state.get
    else:
        get = lambda k, d=None: getattr(state, k, d)

    quarantine_count = len(get("quarantined_records", []))
    indexed_count = get("indexed_count", 0)
    total = indexed_count + quarantine_count
    quarantine_rate = quarantine_count / max(total, 1)
    loop_count = get("loop_count", 0)
    collection_complete = get("collection_complete", False)
    coverage_gaps = get("coverage_gaps", [])
    errors = get("errors", [])
    events_per_month = get("events_per_month", {})

    summary_lines = [
        f"Run ID: {get('run_id', 'unknown')}",
        f"Events indexed: {indexed_count}",
        f"Events quarantined: {quarantine_count} ({quarantine_rate:.1%})",
        f"Research loops: {loop_count}/{MAX_LOOPS}",
        f"Collection complete: {collection_complete}",
        f"Coverage gaps remaining: {len(coverage_gaps)}",
        f"Errors: {len(errors)}",
    ]

    if events_per_month:
        summary_lines.append("\nEvents by month:")
        for month, count in sorted(events_per_month.items()):
            bar = "█" * min(count, 50)
            summary_lines.append(f"  {month}: {count:>4} {bar}")

    if coverage_gaps:
        summary_lines.append(f"\nGaps: {', '.join(coverage_gaps[:5])}")
        if len(coverage_gaps) > 5:
            summary_lines.append(f"  ... and {len(coverage_gaps) - 5} more")

    if errors:
        summary_lines.append(f"\nErrors:")
        for err in errors[:5]:
            summary_lines.append(f"  ⚠ {str(err)[:120]}")

    summary_text = "\n".join(summary_lines)

    if console:
        from rich.panel import Panel
        console.print(Panel(
            summary_text,
            title="📊 Pipeline Complete",
            border_style="green" if collection_complete else "yellow",
        ))
    else:
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(summary_text)
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the CHRONOS research pipeline."""
    global MAX_LOOPS

    parser = argparse.ArgumentParser(
        description="CHRONOS — Autonomous Research Swarm for Temporal Knowledge Base",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=settings.collection_subject,
        help=f"Leader name (default: {settings.collection_subject})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=str(settings.collection_start),
        help=f"Start date YYYY-MM-DD (default: {settings.collection_start})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=str(date.today()),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=MAX_LOOPS,
        help=f"Maximum research loops (default: {MAX_LOOPS})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Resume from last checkpoint of this run ID",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny 1/100th-size smoke test (1 loop, 7-day window)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # List checkpoints mode
    if args.list_checkpoints:
        checkpoints = list_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
        else:
            print(f"{'Run ID':<20} {'Node':<20} {'Loop':<6} {'Timestamp':<20}")
            print("-" * 66)
            for cp in checkpoints:
                print(f"{cp['run_id']:<20} {cp['node']:<20} {cp['loop']:<6} {cp['timestamp']:<20}")
        sys.exit(0)

    # Override max loops if specified
    MAX_LOOPS = args.max_loops

    # Parse dates
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    # Smoke test: override to 1 loop, 7-day window
    if args.smoke:
        from datetime import timedelta
        MAX_LOOPS = 1
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        print(f"🔥 SMOKE TEST: 1 loop, {start_date} → {end_date}")

    # Run
    final_state = asyncio.run(
        run_pipeline(
            args.subject,
            start_date,
            end_date,
            resume_run_id=args.resume,
        )
    )

    # Exit code based on completion
    is_complete = final_state.get("collection_complete", False) if isinstance(final_state, dict) else getattr(final_state, "collection_complete", False)
    sys.exit(0 if is_complete else 1)


if __name__ == "__main__":
    main()
