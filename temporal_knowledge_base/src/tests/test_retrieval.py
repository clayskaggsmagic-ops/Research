"""CHRONOS E2E Verification — temporal sliding window correctness tests.

This script inserts known test events into the Neon database and verifies
that retrieval respects temporal constraints:
  1. Events after simulation_date are NEVER returned
  2. Events before model training cutoff are NEVER returned
  3. Quarantined records (DateConfidence.UNCERTAIN) are NEVER returned
  4. Multi-model switching returns different result sets

Usage:
    uv run python -m src.tests.test_retrieval
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from datetime import date

import numpy as np

from ..config import DateConfidence, ModelConfig
from ..database import (
    Base,
    EventRecordRow,
    async_session,
    engine,
    retrieve_events,
)
from ..models import EventRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test Data — 10 real events spread Oct 2023 – Mar 2025, + 2 quarantined
# ---------------------------------------------------------------------------

TEST_EVENTS: list[dict] = [
    # --- Oct-Dec 2023 (3 events) ---
    {
        "event_date": date(2023, 10, 3),
        "headline": "McCarthy ousted as House Speaker in historic vote",
        "summary": "Rep. Kevin McCarthy was removed as Speaker of the House in an unprecedented vote, with 8 Republicans joining Democrats.",
        "topics": ["legislative", "personnel"],
        "actors": ["Kevin McCarthy", "Matt Gaetz"],
    },
    {
        "event_date": date(2023, 11, 15),
        "headline": "Biden and Xi meet at APEC summit in San Francisco",
        "summary": "President Biden and Chinese President Xi Jinping held talks at the APEC summit, agreeing to resume military communications.",
        "topics": ["foreign_policy", "diplomacy"],
        "actors": ["Joe Biden", "Xi Jinping"],
    },
    {
        "event_date": date(2023, 12, 6),
        "headline": "Senate passes NDAA with $886 billion defense budget",
        "summary": "The Senate approved the annual defense authorization bill with bipartisan support, setting Pentagon spending at $886 billion.",
        "topics": ["legislative", "economic"],
        "actors": ["Senate"],
    },
    # --- Jan-Jun 2024 (3 events) ---
    {
        "event_date": date(2024, 1, 15),
        "headline": "Trump wins Iowa caucuses by record margin",
        "summary": "Former President Trump won the Iowa Republican caucuses with over 50% of the vote, his largest primary margin.",
        "topics": ["executive_actions", "personnel"],
        "actors": ["Donald J. Trump"],
    },
    {
        "event_date": date(2024, 3, 5),
        "headline": "Super Tuesday: Trump sweeps 14 of 15 states",
        "summary": "Donald Trump dominated Super Tuesday primaries, effectively securing the Republican nomination.",
        "topics": ["executive_actions"],
        "actors": ["Donald J. Trump", "Nikki Haley"],
    },
    {
        "event_date": date(2024, 6, 1),
        "headline": "Trump found guilty on 34 felony counts in hush money trial",
        "summary": "A Manhattan jury convicted former President Trump on all 34 counts of falsifying business records.",
        "topics": ["legal"],
        "actors": ["Donald J. Trump"],
    },
    # --- Jul-Dec 2024 (2 events) ---
    {
        "event_date": date(2024, 7, 21),
        "headline": "Biden withdraws from 2024 presidential race",
        "summary": "President Biden announced he would not seek re-election, endorsing Vice President Kamala Harris.",
        "topics": ["executive_actions", "personnel"],
        "actors": ["Joe Biden", "Kamala Harris"],
    },
    {
        "event_date": date(2024, 11, 5),
        "headline": "Trump wins 2024 presidential election",
        "summary": "Donald Trump defeated Kamala Harris in the general election, winning 312 electoral votes.",
        "topics": ["executive_actions"],
        "actors": ["Donald J. Trump", "Kamala Harris"],
    },
    # --- Jan-Mar 2025 (2 events) ---
    {
        "event_date": date(2025, 1, 20),
        "headline": "Trump inaugurated as 47th President of the United States",
        "summary": "Donald Trump was sworn in for his second term, immediately signing executive orders on immigration and energy.",
        "topics": ["executive_actions"],
        "actors": ["Donald J. Trump"],
    },
    {
        "event_date": date(2025, 2, 10),
        "headline": "Trump announces 25% tariffs on all steel and aluminum imports",
        "summary": "President Trump signed proclamations imposing 25% tariffs on steel and aluminum from all countries.",
        "topics": ["economic", "tariff", "trade"],
        "actors": ["Donald J. Trump"],
    },
]

# Quarantined events — should NEVER appear in results
QUARANTINED_EVENTS: list[dict] = [
    {
        "event_date": date(2024, 5, 1),
        "headline": "QUARANTINE: Unverified report of secret trade deal",
        "summary": "An unverified report claims a secret trade agreement was signed. Date is uncertain.",
        "topics": ["economic"],
        "actors": ["Donald J. Trump"],
        "date_confidence": DateConfidence.UNCERTAIN,
    },
    {
        "event_date": date(2025, 1, 5),
        "headline": "QUARANTINE: Rumored cabinet reshuffle before inauguration",
        "summary": "Unconfirmed reports of major cabinet changes. Event date cannot be verified.",
        "topics": ["personnel"],
        "actors": ["Donald J. Trump"],
        "date_confidence": DateConfidence.UNCERTAIN,
    },
]


def _make_record_id(headline: str) -> str:
    return f"test_{hashlib.md5(headline.encode()).hexdigest()[:12]}"


def _make_dummy_embedding() -> list[float]:
    """Generate a random 768-dim embedding for testing."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(768).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # Normalize
    return vec.tolist()


# ---------------------------------------------------------------------------
# Database setup/teardown
# ---------------------------------------------------------------------------

async def setup_test_data():
    """Insert all test events into the database."""
    logger.info("Inserting test events...")

    dummy_embedding = _make_dummy_embedding()

    async with async_session() as session:
        # Clean previous test data
        from sqlalchemy import delete
        await session.execute(
            delete(EventRecordRow).where(
                EventRecordRow.record_id.like("test_%")
            )
        )
        await session.commit()

    inserted = 0
    async with async_session() as session:
        # Insert regular events
        for evt in TEST_EVENTS:
            row = EventRecordRow(
                record_id=_make_record_id(evt["headline"]),
                event_date=evt["event_date"],
                event_date_precision="day",
                date_confidence=DateConfidence.HIGH.value,
                date_verification_method="test_fixture",
                headline=evt["headline"],
                summary=evt["summary"],
                key_facts=[],
                direct_quotes=[],
                topics=evt["topics"],
                actors=evt["actors"],
                sources=[],
                source_count=0,
                confidence=0.9,
                embedding=dummy_embedding,
            )
            session.add(row)
            inserted += 1

        # Insert quarantined events
        for evt in QUARANTINED_EVENTS:
            row = EventRecordRow(
                record_id=_make_record_id(evt["headline"]),
                event_date=evt["event_date"],
                event_date_precision="day",
                date_confidence=DateConfidence.UNCERTAIN.value,
                date_verification_method="failed_validation",
                headline=evt["headline"],
                summary=evt["summary"],
                key_facts=[],
                direct_quotes=[],
                topics=evt["topics"],
                actors=evt["actors"],
                sources=[],
                source_count=0,
                confidence=0.1,
                embedding=dummy_embedding,
            )
            session.add(row)
            inserted += 1

        await session.commit()

    logger.info(f"Inserted {inserted} test events ({len(TEST_EVENTS)} normal + {len(QUARANTINED_EVENTS)} quarantined)")
    return inserted


async def cleanup_test_data():
    """Remove all test events from the database."""
    async with async_session() as session:
        from sqlalchemy import delete
        result = await session.execute(
            delete(EventRecordRow).where(
                EventRecordRow.record_id.like("test_%")
            )
        )
        await session.commit()
        logger.info(f"Cleaned up {result.rowcount} test rows")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""

    def pass_(self, msg: str = ""):
        self.passed = True
        self.message = msg
        return self

    def fail(self, msg: str):
        self.passed = False
        self.message = msg
        return self


async def run_all_tests() -> list[TestResult]:
    """Run all E2E tests and return results."""
    results: list[TestResult] = []
    dummy_embedding = _make_dummy_embedding()

    # -----------------------------------------------------------------------
    # Test 1: simulation_date=2024-06-15, model=gpt-4o
    # gpt-4o cutoff = 2023-10-01
    # Window: [2023-10-01, 2024-06-15]
    # Expected: 6 events (Oct 2023 – Jun 2024)
    # -----------------------------------------------------------------------
    t1 = TestResult("Test 1: GPT-4o window [2023-10-01, 2024-06-15]")
    try:
        async with async_session() as session:
            result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2024, 6, 15),
                model_training_cutoff=ModelConfig.get_cutoff("gpt-4o"),
                top_k=20,
            )
        event_dates = sorted([e.event_date for e in result.events])
        # Should get events from Oct 2023 through Jun 2024
        if all(date(2023, 10, 1) <= d <= date(2024, 6, 15) for d in event_dates):
            if len(result.events) == 6:
                t1.pass_(f"Got {len(result.events)} events, dates: {event_dates}")
            else:
                t1.fail(f"Expected 6 events, got {len(result.events)}: {event_dates}")
        else:
            out_of_range = [d for d in event_dates if d < date(2023, 10, 1) or d > date(2024, 6, 15)]
            t1.fail(f"Events outside window: {out_of_range}")
    except Exception as e:
        t1.fail(f"Exception: {e}")
    results.append(t1)

    # -----------------------------------------------------------------------
    # Test 2: simulation_date=2024-06-15, model=gemini-3-pro
    # gemini-3-pro cutoff = 2025-06-01
    # Window: [2025-06-01, 2024-06-15] → EMPTY (cutoff > simulation_date)
    # Expected: 0 events
    # -----------------------------------------------------------------------
    t2 = TestResult("Test 2: Gemini-3-Pro empty window (cutoff > sim date)")
    try:
        async with async_session() as session:
            result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2024, 6, 15),
                model_training_cutoff=ModelConfig.get_cutoff("gemini-3-pro"),
                top_k=20,
            )
        if len(result.events) == 0:
            t2.pass_(f"Correctly returned 0 events for empty window")
        else:
            t2.fail(f"Expected 0 events, got {len(result.events)}")
    except Exception as e:
        t2.fail(f"Exception: {e}")
    results.append(t2)

    # -----------------------------------------------------------------------
    # Test 3: simulation_date=2025-03-01, model=gpt-4o
    # gpt-4o cutoff = 2023-10-01
    # Window: [2023-10-01, 2025-03-01]
    # Expected: ALL 10 events
    # -----------------------------------------------------------------------
    t3 = TestResult("Test 3: GPT-4o full range [2023-10-01, 2025-03-01]")
    try:
        async with async_session() as session:
            result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2025, 3, 1),
                model_training_cutoff=ModelConfig.get_cutoff("gpt-4o"),
                top_k=20,
            )
        if len(result.events) == 10:
            t3.pass_(f"Got all 10 events")
        else:
            t3.fail(f"Expected 10 events, got {len(result.events)}")
    except Exception as e:
        t3.fail(f"Exception: {e}")
    results.append(t3)

    # -----------------------------------------------------------------------
    # Test 4: simulation_date=2024-01-01, model=gpt-4o
    # gpt-4o cutoff = 2023-10-01
    # Window: [2023-10-01, 2024-01-01]
    # Expected: 3 events (Oct-Dec 2023)
    # -----------------------------------------------------------------------
    t4 = TestResult("Test 4: GPT-4o narrow [2023-10-01, 2024-01-01]")
    try:
        async with async_session() as session:
            result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2024, 1, 1),
                model_training_cutoff=ModelConfig.get_cutoff("gpt-4o"),
                top_k=20,
            )
        event_dates = sorted([e.event_date for e in result.events])
        if len(result.events) == 3:
            t4.pass_(f"Got 3 events: {event_dates}")
        else:
            t4.fail(f"Expected 3 events, got {len(result.events)}: {event_dates}")
    except Exception as e:
        t4.fail(f"Exception: {e}")
    results.append(t4)

    # -----------------------------------------------------------------------
    # Test 5: Quarantine exclusion
    # Verify UNCERTAIN events NEVER appear, even in full window
    # -----------------------------------------------------------------------
    t5 = TestResult("Test 5: Quarantined events never returned")
    try:
        async with async_session() as session:
            result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2025, 12, 31),  # Far future
                model_training_cutoff=date(2020, 1, 1),  # Far past
                top_k=50,  # Get everything
            )
        quarantine_headlines = [e.headline for e in result.events if "QUARANTINE" in e.headline]
        if len(quarantine_headlines) == 0:
            t5.pass_(f"0 quarantined events in {len(result.events)} results")
        else:
            t5.fail(f"Quarantined events leaked: {quarantine_headlines}")
    except Exception as e:
        t5.fail(f"Exception: {e}")
    results.append(t5)

    # -----------------------------------------------------------------------
    # Test 6: Multi-model switching — same simulation_date, different models
    # sim_date=2025-03-01
    # gpt-4o cutoff=2023-10-01 → 10 events
    # gemini-2.5-pro cutoff=2025-01-01 → 2 events (Jan-Feb 2025 only)
    # -----------------------------------------------------------------------
    t6 = TestResult("Test 6: Multi-model switching (gpt-4o vs gemini-2.5-pro)")
    try:
        async with async_session() as session:
            gpt4o_result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2025, 3, 1),
                model_training_cutoff=ModelConfig.get_cutoff("gpt-4o"),
                top_k=20,
            )
            gemini_result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2025, 3, 1),
                model_training_cutoff=ModelConfig.get_cutoff("gemini-2.5-pro"),
                top_k=20,
            )
        if len(gpt4o_result.events) == 10 and len(gemini_result.events) == 2:
            t6.pass_(
                f"gpt-4o→{len(gpt4o_result.events)} events, "
                f"gemini-2.5-pro→{len(gemini_result.events)} events"
            )
        else:
            t6.fail(
                f"Expected gpt-4o=10/gemini=2, got "
                f"gpt-4o={len(gpt4o_result.events)}/gemini={len(gemini_result.events)}"
            )
    except Exception as e:
        t6.fail(f"Exception: {e}")
    results.append(t6)

    # -----------------------------------------------------------------------
    # Test 7: Briefing format
    # -----------------------------------------------------------------------
    t7 = TestResult("Test 7: Intelligence briefing output format")
    try:
        async with async_session() as session:
            result = await retrieve_events(
                session=session,
                query_embedding=dummy_embedding,
                simulation_date=date(2025, 3, 1),
                model_training_cutoff=ModelConfig.get_cutoff("gpt-4o"),
                top_k=5,
            )
        result.query = "What has Trump done recently?"
        briefing = result.to_briefing(subject_name="Donald J. Trump")

        # Verify it has essential elements
        checks = {
            "has header": "INTELLIGENCE BRIEFING" in briefing or "BRIEFING" in briefing.upper(),
            "has date range": "2023" in briefing and "2025" in briefing,
            "has events": len(briefing) > 200,
            "has headlines": any(e.headline[:20] in briefing for e in result.events[:3]),
        }
        failures = [k for k, v in checks.items() if not v]
        if not failures:
            t7.pass_(f"Briefing is {len(briefing)} chars, passes all format checks")
        else:
            t7.fail(f"Briefing format issues: {failures}")

        # Print the briefing for visual inspection
        print("\n" + "=" * 70)
        print("SAMPLE BRIEFING OUTPUT (for visual inspection)")
        print("=" * 70)
        print(briefing[:2000])
        if len(briefing) > 2000:
            print(f"\n... [{len(briefing) - 2000} more characters]")
        print("=" * 70)

    except Exception as e:
        t7.fail(f"Exception: {e}")
    results.append(t7)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    """Run the full E2E verification suite."""
    print("\n" + "=" * 70)
    print("  CHRONOS — End-to-End Verification Suite")
    print("=" * 70)

    # Model cutoffs for reference
    print("\nModel cutoff dates:")
    for model, cutoff in sorted(ModelConfig.CUTOFFS.items(), key=lambda x: x[1]):
        print(f"  {model:20s} → {cutoff}")

    # Setup
    print("\n--- Setup ---")
    try:
        await setup_test_data()
    except Exception as e:
        print(f"❌ Failed to insert test data: {e}")
        sys.exit(1)

    # Run tests
    print("\n--- Running Tests ---")
    results = await run_all_tests()

    # Report
    print("\n" + "=" * 70)
    print("  TEST RESULTS")
    print("=" * 70)
    passed = 0
    failed = 0
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"  {status}  {r.name}")
        if r.message:
            print(f"          {r.message}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print(f"\n  {passed}/{len(results)} tests passed", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" — ALL PASS ✅")
    print("=" * 70)

    # Cleanup
    print("\n--- Cleanup ---")
    await cleanup_test_data()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
