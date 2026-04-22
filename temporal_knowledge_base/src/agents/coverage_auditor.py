"""CHRONOS Coverage Auditor — gap detection and follow-up research triggers.

The Coverage Auditor is the "quality assurance" layer. After events are
indexed, it checks whether the database has sufficient coverage across
all months and topic areas. Gaps are reported back to the Coordinator
for follow-up research, creating the autonomous feedback loop.

Completion criteria:
  1. Every month has >= 15 events
  2. No month has a topic category with 0 events (across core categories)
  3. Quarantine rate < 15% of total events
  4. At least 3 research rounds have been completed
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date

from dateutil.relativedelta import relativedelta
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import func, select, text as sql_text

from ..config import settings, DateConfidence
from ..database import EventRecordRow, async_session, get_event_count_by_month, get_quarantined_count
from ..models import SwarmState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_EVENTS_PER_MONTH = 15
HIGH_ACTIVITY_MONTHS_MIN = 25
RECENCY_BIAS_THRESHOLD = 3.0  # Flag if month has 3x+ average
MAX_QUARANTINE_RATE = 0.15
MIN_RESEARCH_ROUNDS = 3

# Core topic categories every month should have some coverage in
CORE_TOPIC_CATEGORIES = [
    "executive_actions",
    "foreign_policy",
    "economic",
    "personnel",
    "legislative",
    "legal",
]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.research_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
        timeout=120,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# 1. Monthly Coverage Analysis
# ---------------------------------------------------------------------------

async def analyze_monthly_coverage(
    collection_start: date,
    collection_end: date,
) -> tuple[dict[str, int], list[str]]:
    """Query the database for monthly event counts and identify sparse months.

    Returns:
        (events_per_month dict, list of gap descriptions)
    """
    async with async_session() as session:
        events_per_month = await get_event_count_by_month(
            session,
            start_date=collection_start,
            end_date=collection_end,
        )

    # Build expected months list
    expected_months: list[str] = []
    current = collection_start.replace(day=1)
    end = min(collection_end, date.today())
    while current <= end:
        expected_months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)

    gaps: list[str] = []

    for month in expected_months:
        count = events_per_month.get(month, 0)
        if count == 0:
            gaps.append(f"CRITICAL: {month} has 0 events — needs full research")
        elif count < MIN_EVENTS_PER_MONTH:
            gaps.append(
                f"SPARSE: {month} has {count}/{MIN_EVENTS_PER_MONTH} events — "
                f"needs {MIN_EVENTS_PER_MONTH - count} more"
            )

    return events_per_month, gaps


# ---------------------------------------------------------------------------
# 2. Topic Coverage Analysis
# ---------------------------------------------------------------------------

async def analyze_topic_coverage(
    collection_start: date,
    collection_end: date,
) -> dict[str, list[str]]:
    """Check topic distribution per month and identify gaps.

    Returns:
        dict: month → list of missing core topic categories
    """
    async with async_session() as session:
        # Query topics grouped by month
        stmt = (
            select(
                func.to_char(EventRecordRow.event_date, "YYYY-MM").label("month"),
                EventRecordRow.topics,
            )
            .where(EventRecordRow.date_confidence != DateConfidence.UNCERTAIN.value)
            .where(EventRecordRow.event_date >= collection_start)
            .where(EventRecordRow.event_date <= collection_end)
        )
        result = await session.execute(stmt)

    # Aggregate topics per month
    topics_by_month: dict[str, set[str]] = defaultdict(set)
    for row in result:
        month = row.month
        topics = row.topics or []
        for topic in topics:
            # Map topic to core category
            category = _map_to_core_category(topic)
            if category:
                topics_by_month[month].add(category)

    # Find missing categories per month
    missing_topics: dict[str, list[str]] = {}
    for month, topics in topics_by_month.items():
        missing = [c for c in CORE_TOPIC_CATEGORIES if c not in topics]
        if missing:
            missing_topics[month] = missing

    return missing_topics


def _map_to_core_category(topic: str) -> str | None:
    """Map a freeform topic tag to a core category."""
    topic = topic.lower().replace("-", "_")

    executive_keywords = {"executive", "order", "memo", "directive", "signing"}
    foreign_keywords = {"foreign", "diplomacy", "summit", "treaty", "nato", "un", "bilateral"}
    economic_keywords = {"economic", "tariff", "trade", "gdp", "inflation", "budget", "fiscal"}
    personnel_keywords = {"personnel", "nomination", "appointment", "cabinet", "fired", "resign"}
    legislative_keywords = {"legislative", "congress", "senate", "bill", "vote", "law", "ndaa"}
    legal_keywords = {"legal", "court", "lawsuit", "ruling", "indictment", "investigation"}

    for keyword in executive_keywords:
        if keyword in topic:
            return "executive_actions"
    for keyword in foreign_keywords:
        if keyword in topic:
            return "foreign_policy"
    for keyword in economic_keywords:
        if keyword in topic:
            return "economic"
    for keyword in personnel_keywords:
        if keyword in topic:
            return "personnel"
    for keyword in legislative_keywords:
        if keyword in topic:
            return "legislative"
    for keyword in legal_keywords:
        if keyword in topic:
            return "legal"

    return None


# ---------------------------------------------------------------------------
# 3. Recency Bias Detection
# ---------------------------------------------------------------------------

def detect_recency_bias(events_per_month: dict[str, int]) -> list[str]:
    """Detect months with disproportionately more events than average.

    Returns list of over-collected month descriptions.
    """
    if not events_per_month or len(events_per_month) < 3:
        return []

    counts = list(events_per_month.values())
    avg = sum(counts) / len(counts)

    if avg == 0:
        return []

    over_collected: list[str] = []
    for month, count in sorted(events_per_month.items()):
        ratio = count / avg
        if ratio >= RECENCY_BIAS_THRESHOLD:
            over_collected.append(
                f"OVER_COLLECTED: {month} has {count} events "
                f"({ratio:.1f}x average of {avg:.0f}) — likely recency bias"
            )

    return over_collected


# ---------------------------------------------------------------------------
# 4. Gap Report Generation
# ---------------------------------------------------------------------------

def generate_gap_report(
    events_per_month: dict[str, int],
    monthly_gaps: list[str],
    missing_topics: dict[str, list[str]],
    over_collected: list[str],
    total_events: int,
    quarantined_count: int,
    research_rounds: int,
) -> dict:
    """Generate a structured gap report."""
    sparse_months = [
        g.split(":")[1].strip().split(" ")[0]
        for g in monthly_gaps
        if g.startswith("SPARSE") or g.startswith("CRITICAL")
    ]

    quarantine_rate = quarantined_count / max(total_events + quarantined_count, 1)

    # Determine recommendation
    if not sparse_months and not missing_topics and quarantine_rate < MAX_QUARANTINE_RATE:
        if research_rounds >= MIN_RESEARCH_ROUNDS:
            recommendation = "Collection complete — all criteria met"
        else:
            recommendation = (
                f"Coverage adequate but only {research_rounds}/{MIN_RESEARCH_ROUNDS} "
                f"rounds completed — continue research"
            )
    else:
        issues = []
        if sparse_months:
            issues.append(f"{len(sparse_months)} sparse months")
        if missing_topics:
            issues.append(f"{len(missing_topics)} months with topic gaps")
        if quarantine_rate >= MAX_QUARANTINE_RATE:
            issues.append(f"quarantine rate {quarantine_rate:.0%} exceeds {MAX_QUARANTINE_RATE:.0%}")
        recommendation = f"Additional research needed: {', '.join(issues)}"

    return {
        "sparse_months": sparse_months,
        "missing_topics": missing_topics,
        "over_collected_months": [m.split(":")[1].strip().split(" ")[0] for m in over_collected],
        "total_events": total_events,
        "quarantined_events": quarantined_count,
        "quarantine_rate": f"{quarantine_rate:.1%}",
        "research_rounds": research_rounds,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 5. Completion Check
# ---------------------------------------------------------------------------

def check_completion(
    events_per_month: dict[str, int],
    expected_months: list[str],
    missing_topics: dict[str, list[str]],
    quarantine_rate: float,
    research_rounds: int,
) -> bool:
    """Check if all completion criteria are met.

    Criteria:
    1. Every expected month has >= 15 events
    2. No month has missing core topic categories
    3. Quarantine rate < 15%
    4. At least 3 research rounds completed
    """
    # Criterion 1: Monthly minimum
    for month in expected_months:
        if events_per_month.get(month, 0) < MIN_EVENTS_PER_MONTH:
            return False

    # Criterion 2: Topic coverage
    if missing_topics:
        return False

    # Criterion 3: Quarantine rate
    if quarantine_rate >= MAX_QUARANTINE_RATE:
        return False

    # Criterion 4: Minimum rounds
    if research_rounds < MIN_RESEARCH_ROUNDS:
        return False

    return True


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

async def coverage_auditor_node(state: SwarmState) -> SwarmState:
    """LangGraph node: analyze database coverage and set gaps / completion flag.

    Queries the database, identifies gaps, updates state.coverage_gaps,
    and sets state.collection_complete if all criteria are met.
    """
    logger.info("Coverage Auditor: analyzing database coverage...")

    # Track research rounds (each pass through the pipeline = 1 round)
    # We estimate rounds from how many times the auditor has been called
    # by checking if events_per_month was previously populated
    research_rounds = getattr(state, "_research_rounds", 0) + 1

    # --- 1. Monthly coverage ---
    events_per_month, monthly_gaps = await analyze_monthly_coverage(
        state.collection_start,
        state.collection_end,
    )
    state.events_per_month = events_per_month
    total_events = sum(events_per_month.values())

    # --- 2. Topic coverage ---
    try:
        missing_topics = await analyze_topic_coverage(
            state.collection_start,
            state.collection_end,
        )
    except Exception as e:
        logger.warning(f"Topic coverage analysis failed: {e}")
        missing_topics = {}

    # --- 3. Recency bias ---
    over_collected = detect_recency_bias(events_per_month)

    # --- 4. Quarantine stats ---
    try:
        async with async_session() as session:
            quarantined_count = await get_quarantined_count(session)
    except Exception as e:
        logger.warning(f"Quarantine count query failed: {e}")
        quarantined_count = len(state.quarantined_records)

    quarantine_rate = quarantined_count / max(total_events + quarantined_count, 1)

    # --- 5. Expected months ---
    expected_months: list[str] = []
    current = state.collection_start.replace(day=1)
    end = min(state.collection_end, date.today())
    while current <= end:
        expected_months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)

    # --- 6. Generate gap descriptions for the Coordinator ---
    all_gaps: list[str] = []
    all_gaps.extend(monthly_gaps)

    for month, topics in missing_topics.items():
        all_gaps.append(
            f"TOPIC_GAP: {month} is missing coverage of: {', '.join(topics)}"
        )

    all_gaps.extend(over_collected)

    state.coverage_gaps = all_gaps

    # --- 7. Completion check ---
    is_complete = check_completion(
        events_per_month, expected_months, missing_topics,
        quarantine_rate, research_rounds,
    )
    state.collection_complete = is_complete

    # --- 8. Generate report ---
    report = generate_gap_report(
        events_per_month, monthly_gaps, missing_topics, over_collected,
        total_events, quarantined_count, research_rounds,
    )

    logger.info(
        f"Coverage Auditor complete:\n"
        f"  Total events: {total_events}\n"
        f"  Quarantined: {quarantined_count} ({quarantine_rate:.1%})\n"
        f"  Sparse months: {len(report['sparse_months'])}\n"
        f"  Topic gaps: {len(missing_topics)} months\n"
        f"  Over-collected: {len(over_collected)} months\n"
        f"  Research rounds: {research_rounds}\n"
        f"  Collection complete: {is_complete}\n"
        f"  Recommendation: {report['recommendation']}"
    )

    return state
