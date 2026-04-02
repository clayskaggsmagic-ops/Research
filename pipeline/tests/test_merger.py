"""
Tests for the merger / deduplication logic in Stage 1.
"""

from __future__ import annotations

from src.schemas import DecisionSeed, DomainType, Source
from src.stages.stage1_seeds import merge_and_dedup


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_seed(
    seed_id: str = "TEST-001",
    event_desc: str = "Trump signed an executive order on tariffs",
    decision: str = "Signed EO imposing 25% tariffs on steel",
    date: str = "2025-03-01",
    domain: DomainType = DomainType.TRADE_TARIFFS,
    sources: list[Source] | None = None,
    alternatives: list[str] | None = None,
    attribution: str = "Signed by the President",
) -> DecisionSeed:
    return DecisionSeed(
        seed_id=seed_id,
        event_description=event_desc,
        decision_taken=decision,
        decision_date=date,
        simulation_date="2025-02-20",
        domain=domain,
        plausible_alternatives=alternatives or ["Take no action", "Impose lower tariffs"],
        sources=sources or [Source(name="Reuters", url="https://reuters.com/test", date=date)],
        attribution_evidence=attribution,
        leader_attributable=True,
    )


# ── Deduplication Tests ────────────────────────────────────────────────────────


def test_dedup_identical_seeds():
    """Exact duplicate seeds should merge into one."""
    s1 = _make_seed(seed_id="A")
    s2 = _make_seed(seed_id="B")

    result = merge_and_dedup([s1, s2])
    assert len(result) == 1


def test_dedup_similar_events_same_date():
    """Similar descriptions on the same date should merge."""
    s1 = _make_seed(
        seed_id="A",
        event_desc="Trump signed executive order imposing tariffs on steel imports",
        decision="Signed EO imposing 25% tariffs on steel and aluminum",
        date="2025-03-01",
    )
    s2 = _make_seed(
        seed_id="B",
        event_desc="President Trump signed an executive order to impose tariffs on steel",
        decision="Imposed 25% tariffs on steel imports via executive order",
        date="2025-03-01",
    )

    result = merge_and_dedup([s1, s2])
    assert len(result) == 1


def test_dedup_different_events():
    """Different events should NOT merge."""
    s1 = _make_seed(
        seed_id="A",
        event_desc="Trump signed executive order on tariffs",
        decision="Imposed tariffs on steel",
        date="2025-03-01",
    )
    s2 = _make_seed(
        seed_id="B",
        event_desc="Trump fired the Secretary of Defense",
        decision="Fired Secretary Austin",
        date="2025-03-15",
        domain=DomainType.PERSONNEL,
    )

    result = merge_and_dedup([s1, s2])
    assert len(result) == 2


def test_dedup_same_event_different_dates():
    """Same event description but dates far apart should NOT merge."""
    s1 = _make_seed(seed_id="A", date="2025-03-01")
    s2 = _make_seed(seed_id="B", date="2025-06-01")

    result = merge_and_dedup([s1, s2])
    assert len(result) == 2


def test_dedup_close_dates_merge():
    """Same event within 3 days should merge."""
    s1 = _make_seed(seed_id="A", date="2025-03-01")
    s2 = _make_seed(seed_id="B", date="2025-03-03")

    result = merge_and_dedup([s1, s2])
    assert len(result) == 1


# ── Merge Quality Tests ───────────────────────────────────────────────────────


def test_merge_unions_sources():
    """Merged seeds should have sources from both originals."""
    s1 = _make_seed(
        seed_id="A",
        sources=[Source(name="Reuters", url="https://reuters.com/1", date="2025-03-01")],
    )
    s2 = _make_seed(
        seed_id="B",
        sources=[Source(name="AP", url="https://apnews.com/1", date="2025-03-01")],
    )

    result = merge_and_dedup([s1, s2])
    assert len(result) == 1
    assert len(result[0].sources) == 2
    urls = {s.url for s in result[0].sources}
    assert "https://reuters.com/1" in urls
    assert "https://apnews.com/1" in urls


def test_merge_unions_alternatives():
    """Merged seeds should have combined alternatives."""
    s1 = _make_seed(
        seed_id="A",
        alternatives=["Take no action", "Impose lower tariffs"],
    )
    s2 = _make_seed(
        seed_id="B",
        alternatives=["Take no action", "Negotiate instead"],
    )

    result = merge_and_dedup([s1, s2])
    assert len(result) == 1
    alt_lower = [a.lower() for a in result[0].plausible_alternatives]
    assert "negotiate instead" in alt_lower
    assert "impose lower tariffs" in alt_lower
    # "Take no action" should appear only once
    assert alt_lower.count("take no action") == 1


def test_merge_keeps_longer_description():
    """Merged seed should keep the more detailed description."""
    s1 = _make_seed(
        seed_id="A",
        event_desc="Short description",
    )
    s2 = _make_seed(
        seed_id="B",
        event_desc="A much longer and more detailed description of the same event with more context",
    )

    result = merge_and_dedup([s1, s2])
    assert len(result) == 1
    assert "much longer" in result[0].event_description


# ── Edge Cases ─────────────────────────────────────────────────────────────────


def test_empty_input():
    """Empty seed list should return empty."""
    assert merge_and_dedup([]) == []


def test_single_seed():
    """Single seed should pass through unchanged."""
    s = _make_seed()
    result = merge_and_dedup([s])
    assert len(result) == 1


def test_no_action_always_present():
    """'Take no action' should always be in alternatives after merge."""
    s = _make_seed(alternatives=["Impose tariffs", "Negotiate"])
    result = merge_and_dedup([s])
    assert any("no action" in a.lower() for a in result[0].plausible_alternatives)


def test_stable_seed_ids():
    """Seed IDs after merge should be deterministic (content-based hash)."""
    s1 = _make_seed(seed_id="A")
    s2 = _make_seed(seed_id="B")

    result1 = merge_and_dedup([s1])
    result2 = merge_and_dedup([s2])

    # Same content → same hash-based ID
    assert result1[0].seed_id == result2[0].seed_id
