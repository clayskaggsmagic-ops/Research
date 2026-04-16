"""CHRONOS checkpoint/resume system — survive crashes without losing progress.

After each pipeline node completes, the full SwarmState is serialized to
a JSON file in the `checkpoints/` directory. If the pipeline crashes,
you can resume from the last successful node.

Usage:
    # Save (called automatically by pipeline wrapper)
    save_checkpoint(state, node_name="extraction", run_id="20260416_142600")

    # Resume
    state, resume_node = load_latest_checkpoint(run_id="20260416_142600")
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from .models import SwarmState

logger = logging.getLogger(__name__)

# Checkpoint directory (relative to project root)
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# Pipeline node order — used to determine resume point
NODE_ORDER = [
    "loop_guard",
    "coordinator",
    "discovery",
    "extraction",
    "cleaning",
    "temporal_validator",
    "indexing",
    "coverage_auditor",
]


def _ensure_dir() -> Path:
    """Create checkpoint directory if it doesn't exist."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR


def _serialize_state(state: SwarmState) -> dict:
    """Convert SwarmState to a JSON-safe dict.

    Handles special types:
    - date → ISO string
    - set → list
    - Pydantic models → dict
    """
    data = state.model_dump(mode="json")
    # Ensure urls_visited is a list (Pydantic may already handle this)
    if isinstance(data.get("urls_visited"), set):
        data["urls_visited"] = list(data["urls_visited"])
    return data


def _deserialize_state(data: dict) -> SwarmState:
    """Reconstruct SwarmState from a checkpoint dict."""
    # Convert urls_visited back to set
    if isinstance(data.get("urls_visited"), list):
        data["urls_visited"] = set(data["urls_visited"])
    return SwarmState(**data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: SwarmState,
    node_name: str,
    run_id: str,
) -> Path:
    """Save a checkpoint after a node completes.

    File format: {run_id}_{node_name}_{timestamp}.json
    """
    dir_ = _ensure_dir()
    ts = datetime.now().strftime("%H%M%S")
    filename = f"{run_id}_{node_name}_{ts}.json"
    filepath = dir_ / filename

    payload = {
        "run_id": run_id,
        "node_name": node_name,
        "timestamp": datetime.now().isoformat(),
        "loop_count": state.loop_count,
        "state": _serialize_state(state),
    }

    filepath.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"[checkpoint] Saved: {filepath.name} (loop {state.loop_count}, node={node_name})")
    return filepath


def load_latest_checkpoint(run_id: str | None = None) -> tuple[SwarmState, str] | None:
    """Load the most recent checkpoint, optionally filtered by run_id.

    Returns:
        (SwarmState, last_completed_node_name) or None if no checkpoints found.
    """
    dir_ = _ensure_dir()

    # Find matching checkpoint files
    pattern = f"{run_id}_*.json" if run_id else "*.json"
    files = sorted(dir_.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)

    if not files:
        logger.info(f"[checkpoint] No checkpoints found{f' for run_id={run_id}' if run_id else ''}")
        return None

    latest = files[0]
    logger.info(f"[checkpoint] Loading: {latest.name}")

    payload = json.loads(latest.read_text())
    state = _deserialize_state(payload["state"])
    node_name = payload["node_name"]

    return state, node_name


def get_resume_node(last_completed_node: str) -> str:
    """Given the last completed node, return the NEXT node to run.

    If the last node was 'coverage_auditor', the pipeline completed that loop
    and should restart at 'loop_guard' for the next iteration.
    """
    if last_completed_node not in NODE_ORDER:
        logger.warning(f"[checkpoint] Unknown node '{last_completed_node}', starting from beginning")
        return NODE_ORDER[0]

    idx = NODE_ORDER.index(last_completed_node)
    if idx + 1 >= len(NODE_ORDER):
        # Completed the full loop — start next iteration
        return NODE_ORDER[0]

    return NODE_ORDER[idx + 1]


def list_checkpoints() -> list[dict]:
    """List all available checkpoints with metadata."""
    dir_ = _ensure_dir()
    checkpoints = []

    for f in sorted(dir_.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(f.read_text())
            checkpoints.append({
                "file": f.name,
                "run_id": payload.get("run_id", "unknown"),
                "node": payload.get("node_name", "unknown"),
                "timestamp": payload.get("timestamp", "unknown"),
                "loop": payload.get("loop_count", -1),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return checkpoints


def clean_checkpoints(run_id: str | None = None, keep_latest: int = 3) -> int:
    """Remove old checkpoints, keeping only the N most recent.

    Returns the number of files deleted.
    """
    dir_ = _ensure_dir()
    pattern = f"{run_id}_*.json" if run_id else "*.json"
    files = sorted(dir_.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)

    to_delete = files[keep_latest:]
    for f in to_delete:
        f.unlink()

    if to_delete:
        logger.info(f"[checkpoint] Cleaned {len(to_delete)} old checkpoints")
    return len(to_delete)
