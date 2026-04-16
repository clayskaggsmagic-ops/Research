"""Tests for the checkpoint/resume system."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from src.checkpoint import (
    CHECKPOINT_DIR,
    NODE_ORDER,
    clean_checkpoints,
    get_resume_node,
    list_checkpoints,
    load_latest_checkpoint,
    save_checkpoint,
)
from src.models import SwarmState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_checkpoint_dir():
    """Ensure a fresh checkpoint directory for each test."""
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)


def _make_state(**overrides) -> SwarmState:
    """Create a minimal SwarmState for testing."""
    defaults = dict(
        subject_name="Test Subject",
        collection_start=date(2024, 1, 1),
        collection_end=date(2024, 6, 1),
        run_id="test_run_001",
        loop_count=1,
    )
    defaults.update(overrides)
    return SwarmState(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSaveAndLoad:
    """Test save_checkpoint → load_latest_checkpoint round-trip."""

    def test_round_trip(self):
        """State survives serialization → deserialization."""
        state = _make_state(
            loop_count=2,
            indexed_count=42,
            errors=["test error"],
        )
        state.urls_visited = {"https://example.com/a", "https://example.com/b"}

        save_checkpoint(state, node_name="discovery", run_id="test_run_001")

        loaded = load_latest_checkpoint(run_id="test_run_001")
        assert loaded is not None

        restored, last_node = loaded
        assert last_node == "discovery"
        assert restored.loop_count == 2
        assert restored.indexed_count == 42
        assert restored.run_id == "test_run_001"
        assert "test error" in restored.errors
        assert "https://example.com/a" in restored.urls_visited

    def test_multiple_checkpoints_returns_latest(self):
        """Most recent checkpoint is loaded."""
        state1 = _make_state(loop_count=1)
        state2 = _make_state(loop_count=2)

        save_checkpoint(state1, node_name="coordinator", run_id="test_run_001")
        save_checkpoint(state2, node_name="extraction", run_id="test_run_001")

        loaded = load_latest_checkpoint(run_id="test_run_001")
        assert loaded is not None

        restored, last_node = loaded
        assert last_node == "extraction"
        assert restored.loop_count == 2

    def test_no_checkpoint_returns_none(self):
        """Returns None when no checkpoints exist."""
        result = load_latest_checkpoint(run_id="nonexistent")
        assert result is None


class TestGetResumeNode:
    """Test resume-point logic."""

    def test_resume_after_coordinator(self):
        assert get_resume_node("coordinator") == "discovery"

    def test_resume_after_discovery(self):
        assert get_resume_node("discovery") == "extraction"

    def test_resume_after_extraction(self):
        assert get_resume_node("extraction") == "cleaning"

    def test_resume_after_coverage_auditor_wraps(self):
        """After coverage_auditor, wrap back to loop_guard."""
        assert get_resume_node("coverage_auditor") == "loop_guard"

    def test_resume_unknown_node_starts_from_beginning(self):
        assert get_resume_node("unknown_node") == NODE_ORDER[0]

    def test_full_node_order_coverage(self):
        """Every node in NODE_ORDER has a valid successor."""
        for node in NODE_ORDER:
            resume = get_resume_node(node)
            assert resume in NODE_ORDER, f"Resume from {node} → {resume} not in NODE_ORDER"


class TestListAndClean:
    """Test listing and cleaning checkpoints."""

    def test_list_checkpoints(self):
        state = _make_state()
        save_checkpoint(state, node_name="indexing", run_id="test_run_001")
        save_checkpoint(state, node_name="coverage_auditor", run_id="test_run_001")

        items = list_checkpoints()
        assert len(items) == 2
        assert items[0]["node"] == "coverage_auditor"  # Most recent first

    def test_clean_keeps_latest(self):
        state = _make_state()
        for node in NODE_ORDER:
            save_checkpoint(state, node_name=node, run_id="test_run_001")

        deleted = clean_checkpoints(run_id="test_run_001", keep_latest=2)
        assert deleted == len(NODE_ORDER) - 2

        remaining = list(CHECKPOINT_DIR.glob("test_run_001_*.json"))
        assert len(remaining) == 2


class TestSerializationEdgeCases:
    """Test that special types survive the checkpoint round-trip."""

    def test_empty_state(self):
        """Minimal state can be checkpointed."""
        state = SwarmState()
        save_checkpoint(state, node_name="loop_guard", run_id=state.run_id)

        loaded = load_latest_checkpoint(run_id=state.run_id)
        assert loaded is not None

    def test_dates_survive_round_trip(self):
        """date fields are properly serialized/deserialized."""
        state = _make_state(
            collection_start=date(2023, 10, 1),
            collection_end=date(2025, 6, 15),
        )
        save_checkpoint(state, node_name="coordinator", run_id="test_run_001")

        loaded = load_latest_checkpoint(run_id="test_run_001")
        restored, _ = loaded
        assert restored.collection_start == date(2023, 10, 1)
        assert restored.collection_end == date(2025, 6, 15)

    def test_urls_visited_set_round_trip(self):
        """set[str] → list → set conversion works."""
        state = _make_state()
        state.urls_visited = {"a", "b", "c"}

        save_checkpoint(state, node_name="discovery", run_id="test_run_001")
        loaded = load_latest_checkpoint(run_id="test_run_001")
        restored, _ = loaded

        assert isinstance(restored.urls_visited, set)
        assert restored.urls_visited == {"a", "b", "c"}
