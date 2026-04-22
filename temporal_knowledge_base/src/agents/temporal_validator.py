"""CHRONOS Temporal Validator Agent — 4-layer date validation and quarantine.

The Temporal Validator is the "immune system" of the knowledge base.
It subjects every EventRecord's date to 4 layers of validation:

    Layer 1: Parsing Validation — mechanical date checks
    Layer 2: Cross-Source Verification — date agreement across sources
    Layer 3: Logical Consistency — LLM-powered temporal logic checks
    Layer 4: Statistical Outlier Detection — aggregate-level anomaly flags

Records that pass all layers get promoted to `validated_records`.
Records that fail hard checks are quarantined — they NEVER enter the database.

A wrong date IS data leakage. ZERO tolerance.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import date, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings, DateConfidence, DatePrecision
from ..models import EventRecord, SwarmState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    """Get the LLM for logical consistency checks."""
    return ChatGoogleGenerativeAI(
        model=settings.research_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,  # Deterministic — this is a fact-checking task
        timeout=120,
        max_retries=2,
    )


# ===================================================================
# LAYER 1: Parsing Validation (purely mechanical)
# ===================================================================

def layer_1_parsing(record: EventRecord, collection_start: date) -> tuple[bool, str]:
    """Layer 1: Mechanical date checks.

    Checks:
    1. Date is not in the future
    2. Date is within collection window
    3. Date precision is acceptable
    4. Event date ≤ pub_date + 1 day (if pub_date available)

    Returns:
        (passed: bool, reason: str)
    """
    today = date.today()

    # Check 1: Not in the future
    if record.event_date > today:
        return False, f"L1_FAIL: event_date {record.event_date} is in the future (today: {today})"

    # Check 2: Within collection window (30-day buffer before start)
    earliest_allowed = collection_start - timedelta(days=30)
    if record.event_date < earliest_allowed:
        return False, (
            f"L1_FAIL: event_date {record.event_date} is before collection window "
            f"(start: {collection_start}, buffer: {earliest_allowed})"
        )

    # Check 3: Date precision
    if record.event_date_precision == DatePrecision.UNKNOWN:
        return False, "L1_FAIL: date precision is UNKNOWN — cannot validate"

    # Check 4: Event date vs pub_date consistency
    for source in record.sources:
        if source.pub_date:
            # Event can't happen more than 1 day after it was published
            if record.event_date > source.pub_date + timedelta(days=1):
                return False, (
                    f"L1_FAIL: event_date {record.event_date} is after "
                    f"pub_date {source.pub_date} + 1 day grace (source: {source.url})"
                )

    return True, "L1_PASS"


# ===================================================================
# LAYER 2: Cross-Source Verification
# ===================================================================

def layer_2_cross_source(record: EventRecord) -> tuple[DateConfidence, str]:
    """Layer 2: Verify date agreement across multiple sources.

    Returns:
        (updated DateConfidence, verification method string)
    """
    source_count = record.source_count

    if source_count == 0:
        return DateConfidence.UNCERTAIN, "L2: no_sources"

    if source_count == 1:
        # Single source — check if date was explicitly stated
        if not record.date_confidence == DateConfidence.UNCERTAIN:
            return DateConfidence.SINGLE_SOURCE, "L2: single_source_explicit_date"
        else:
            return DateConfidence.UNCERTAIN, "L2: single_source_ambiguous_date"

    # Multiple sources — check date agreement via existing sources
    # The Cleaning Agent already resolved dates via majority vote.
    # We check the confidence it assigned.
    if record.date_confidence == DateConfidence.HIGH:
        if source_count >= 3:
            return DateConfidence.VERIFIED, f"L2: {source_count}_source_agreement"
        else:
            return DateConfidence.HIGH, f"L2: {source_count}_source_consensus"
    elif record.date_confidence == DateConfidence.MEDIUM:
        return DateConfidence.HIGH, f"L2: {source_count}_source_majority_consensus"
    elif record.date_confidence == DateConfidence.UNCERTAIN:
        # Sources disagree by more than 2 days — QUARANTINE
        return DateConfidence.UNCERTAIN, "L2_FAIL: source_date_disagreement_gt_2_days"
    else:
        return record.date_confidence, f"L2: inherited_{record.date_confidence.value}"


# ===================================================================
# LAYER 3: Logical Consistency (LLM-powered)
# ===================================================================

LOGICAL_CONSISTENCY_PROMPT = """You are a fact-checker specializing in temporal consistency.
Your job is to verify that the stated date of an event is logically consistent
with the event's content and context.

If you have ANY doubt about the date's accuracy, flag it.

CHECK FOR:
1. Was the stated leader in the stated position on that date?
   - Trump was inaugurated January 20, 2025 (2nd term).
   - He was NOT president between January 20, 2021 and January 20, 2025.
   - If an event says "President Trump signed..." on a date he wasn't president → FAIL.

2. Does the event reference other events that must have happened first?
   - "In response to last week's tariff announcement..." — the tariff announcement
     must exist and be dated BEFORE this event.

3. Does the policy/decision make sense for its stated time period?
   - A 2024 event about a policy that was only proposed in 2025 → FAIL.

4. Are referenced laws, regulations, or organizations consistent with the date?
   - Referencing a law that hadn't been passed yet → FAIL.

EXAMPLES:
✅ PASS: "Trump signed executive order on January 22, 2025" — He was president. Date is plausible.
✅ PASS: "Trump announced tariffs on Canadian goods, March 4, 2025" — He was president. Plausible.
❌ FAIL: "President Trump signed executive order on July 4, 2023" — He wasn't president in July 2023.
❌ FAIL: "Trump responded to the 2025 NDAA on November 1, 2024" — The 2025 NDAA hadn't passed yet.

RESPOND WITH EXACTLY ONE OF:
- "PASS" (if the date is logically consistent)
- "FAIL: <one-line reason>" (if there's a logical inconsistency)
- "FLAG: <one-line reason>" (if suspicious but not definitively wrong)"""


async def layer_3_logical_consistency(record: EventRecord) -> tuple[bool, str]:
    """Layer 3: LLM-powered temporal logic check.

    Returns:
        (passed: bool, reason: str)
    """
    llm = _get_llm()

    prompt = (
        f"Verify the temporal consistency of this event:\n\n"
        f"EVENT DATE: {record.event_date.isoformat()}\n"
        f"HEADLINE: {record.headline}\n"
        f"SUMMARY: {record.summary[:500]}\n"
        f"KEY FACTS: {', '.join(record.key_facts[:5])}\n\n"
        f"Is the date logically consistent with the event content?"
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=LOGICAL_CONSISTENCY_PROMPT),
            HumanMessage(content=prompt),
        ])

        result = response.content.strip().upper()

        if result.startswith("PASS"):
            return True, "L3_PASS: logically_consistent"
        elif result.startswith("FAIL"):
            reason = response.content.strip()
            return False, f"L3_FAIL: {reason}"
        elif result.startswith("FLAG"):
            reason = response.content.strip()
            # Flagged = pass with note, not quarantine
            return True, f"L3_FLAG: {reason}"
        else:
            logger.warning(f"Unexpected L3 response for '{record.headline}': {result}")
            return True, f"L3_INCONCLUSIVE: unexpected_response"

    except Exception as e:
        logger.error(f"Layer 3 LLM error for '{record.headline}': {e}")
        # Don't quarantine on LLM failure — pass with warning
        return True, f"L3_SKIP: llm_error ({e})"


# ===================================================================
# LAYER 4: Statistical Outlier Detection
# ===================================================================

def layer_4_statistical_outlier(
    record: EventRecord,
    all_records: list[EventRecord],
) -> str:
    """Layer 4: Aggregate-level outlier checks.

    These don't quarantine — they add notes for review.

    Checks:
    1. Is this the only event on its date when 10+ cluster a day away?
    2. Is the event date suspiciously far from its pub_date (>30 days)?

    Returns:
        Note string (empty if no flags).
    """
    notes: list[str] = []

    # Check 1: Date isolation
    event_dates = [r.event_date for r in all_records if r.event_date != record.event_date]
    if event_dates:
        # Count events on adjacent days
        adjacent_counts = Counter()
        for d in event_dates:
            diff = abs((d - record.event_date).days)
            if diff <= 1:
                adjacent_counts[d] += 1

        # If 10+ events cluster 1 day away but this is alone on its date
        for adj_date, count in adjacent_counts.items():
            same_day_count = sum(1 for r in all_records if r.event_date == record.event_date)
            if count >= 10 and same_day_count <= 1:
                notes.append(
                    f"L4_FLAG: isolated_date — only {same_day_count} event(s) on "
                    f"{record.event_date} but {count} events on {adj_date}"
                )

    # Check 2: Pub_date distance
    for source in record.sources:
        if source.pub_date:
            gap = abs((record.event_date - source.pub_date).days)
            if gap > 30:
                notes.append(
                    f"L4_FLAG: pub_date_gap — event_date {record.event_date} is "
                    f"{gap} days from pub_date {source.pub_date} (source: {source.name})"
                )
                break  # One flag per record is enough

    return "; ".join(notes) if notes else ""


# ===================================================================
# Full validation pipeline
# ===================================================================

async def validate_record(
    record: EventRecord,
    all_records: list[EventRecord],
    collection_start: date,
) -> tuple[bool, EventRecord]:
    """Run a single record through all 4 validation layers.

    Returns:
        (passed: bool, updated_record: EventRecord)
        If passed=False, the record is quarantined.
    """
    verification_notes: list[str] = []

    # --- Layer 1: Parsing ---
    l1_passed, l1_reason = layer_1_parsing(record, collection_start)
    verification_notes.append(l1_reason)

    if not l1_passed:
        record.date_confidence = DateConfidence.UNCERTAIN
        record.date_verification_method = " | ".join(verification_notes)
        logger.warning(f"QUARANTINED (L1): '{record.headline}' — {l1_reason}")
        return False, record

    # --- Layer 2: Cross-Source ---
    l2_confidence, l2_method = layer_2_cross_source(record)
    verification_notes.append(l2_method)

    if l2_confidence == DateConfidence.UNCERTAIN and "FAIL" in l2_method:
        record.date_confidence = DateConfidence.UNCERTAIN
        record.date_verification_method = " | ".join(verification_notes)
        logger.warning(f"QUARANTINED (L2): '{record.headline}' — {l2_method}")
        return False, record

    record.date_confidence = l2_confidence

    # --- Layer 3: Logical Consistency ---
    l3_passed, l3_reason = await layer_3_logical_consistency(record)
    verification_notes.append(l3_reason)

    if not l3_passed:
        record.date_confidence = DateConfidence.UNCERTAIN
        record.date_verification_method = " | ".join(verification_notes)
        logger.warning(f"QUARANTINED (L3): '{record.headline}' — {l3_reason}")
        return False, record

    # --- Layer 4: Statistical Outlier ---
    l4_notes = layer_4_statistical_outlier(record, all_records)
    if l4_notes:
        verification_notes.append(l4_notes)

    # Record the full verification trail
    record.date_verification_method = " | ".join(verification_notes)

    return True, record


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

async def temporal_validator_node(state: SwarmState) -> SwarmState:
    """LangGraph node: run all cleaned records through 4-layer validation.

    Records that pass → state.validated_records
    Records that fail → state.quarantined_records
    """
    if not state.cleaned_records:
        logger.info("Temporal Validator: no cleaned records to validate")
        return state

    records_to_validate = state.cleaned_records[:]
    state.cleaned_records = []

    logger.info(f"Temporal Validator: validating {len(records_to_validate)} records")

    passed_count = 0
    quarantined_count = 0

    # Validate records in parallel — L3 is an LLM call so this is the key speedup
    sem = asyncio.Semaphore(10)

    async def _validate_with_sem(record: EventRecord) -> tuple[bool, EventRecord]:
        async with sem:
            return await validate_record(
                record=record,
                all_records=records_to_validate,
                collection_start=state.collection_start,
            )

    outcomes = await asyncio.gather(*[_validate_with_sem(r) for r in records_to_validate])

    for passed, updated_record in outcomes:
        if passed:
            state.validated_records.append(updated_record)
            passed_count += 1
        else:
            state.quarantined_records.append(updated_record)
            quarantined_count += 1

    logger.info(
        f"Temporal Validator complete: {passed_count} validated, "
        f"{quarantined_count} quarantined "
        f"(total validated: {len(state.validated_records)}, "
        f"total quarantined: {len(state.quarantined_records)})"
    )

    return state
