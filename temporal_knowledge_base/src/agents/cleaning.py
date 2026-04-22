"""CHRONOS Cleaning Agent — deduplication, normalization, and bias stripping.

The Cleaning Agent is the "editor" of the swarm. It:
1. Clusters ExtractionResults that describe the same real-world event
2. Merges clusters into single, authoritative EventRecords
3. Bias-strips summaries to neutral, AP-wire-copy tone
4. Normalizes actor names, topics, and dates
5. Computes confidence scores based on source count/quality/date agreement

It does NOT validate temporal integrity — that's the Temporal Validator's job.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import date, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings, DateConfidence, DatePrecision
from ..models import (
    DirectQuote,
    EventRecord,
    ExtractionResult,
    Source,
    SwarmState,
)
from ..agents.discovery import score_source_quality

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    """Get the LLM for cleaning tasks."""
    return ChatGoogleGenerativeAI(
        model=settings.research_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
        timeout=120,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# 1. Event Clustering — group extractions about the same event
# ---------------------------------------------------------------------------

def cluster_extractions(
    results: list[ExtractionResult],
) -> list[list[ExtractionResult]]:
    """Cluster ExtractionResults that describe the same real-world event.

    Clustering heuristic:
    - Same event_date (within 2 days) AND high headline word overlap (>50%)
    - OR same event_date (exact) AND overlapping topics

    Returns list of clusters (each cluster is a list of ExtractionResults).
    """
    if not results:
        return []

    used = set()
    clusters: list[list[ExtractionResult]] = []

    for i, r1 in enumerate(results):
        if i in used:
            continue

        cluster = [r1]
        used.add(i)

        for j, r2 in enumerate(results):
            if j in used or j <= i:
                continue

            if _should_cluster(r1, r2):
                cluster.append(r2)
                used.add(j)

        clusters.append(cluster)

    logger.info(
        f"Clustered {len(results)} extractions into {len(clusters)} event groups"
    )
    return clusters


def _should_cluster(a: ExtractionResult, b: ExtractionResult) -> bool:
    """Decide whether two extractions describe the same event."""
    # Both need dates to cluster by date proximity
    if a.event_date and b.event_date:
        date_diff = abs((a.event_date - b.event_date).days)

        # Within 2 days + high headline overlap
        if date_diff <= 2 and _word_overlap(a.headline, b.headline) > 0.5:
            return True

        # Exact date + overlapping topics
        if date_diff == 0 and a.topics and b.topics:
            topic_overlap = set(a.topics) & set(b.topics)
            if topic_overlap:
                return True

    # No dates but very high headline similarity
    if _word_overlap(a.headline, b.headline) > 0.75:
        return True

    return False


def _word_overlap(text_a: str, text_b: str) -> float:
    """Compute word overlap ratio between two strings."""
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    max_len = max(len(words_a), len(words_b))
    return overlap / max_len if max_len > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. Merge clusters into EventRecords
# ---------------------------------------------------------------------------

async def merge_cluster(cluster: list[ExtractionResult]) -> EventRecord:
    """Merge a cluster of ExtractionResults into a single EventRecord.

    Strategy:
    - Headline: pick the most specific (longest non-editorial) headline
    - Summary: LLM synthesizes from all summaries
    - Key facts: union, deduplicated
    - Quotes: union, deduplicated by quote text
    - Sources: all unique URLs
    - Date: majority vote; flag ambiguous if disagree
    """
    # --- Date resolution: majority vote ---
    event_date, date_confidence, date_precision = _resolve_date(cluster)

    # --- Headline: pick most specific ---
    headline = _pick_best_headline(cluster)

    # --- Key facts: union ---
    all_facts: set[str] = set()
    for r in cluster:
        for fact in r.key_facts:
            all_facts.add(fact.strip())

    # --- Quotes: union, dedup by quote text ---
    seen_quotes: set[str] = set()
    all_quotes: list[DirectQuote] = []
    for r in cluster:
        for q in r.quotes:
            normalized = q.quote.strip().lower()
            if normalized not in seen_quotes:
                seen_quotes.add(normalized)
                all_quotes.append(q)

    # --- Topics: union, normalized ---
    all_topics: set[str] = set()
    for r in cluster:
        for topic in r.topics:
            all_topics.add(topic.strip().lower().replace(" ", "_"))

    # --- Sources ---
    sources: list[Source] = []
    seen_urls: set[str] = set()
    for r in cluster:
        if r.url and r.url not in seen_urls:
            seen_urls.add(r.url)
            sources.append(Source(
                name=r.url.split("/")[2] if "/" in r.url else r.url,
                url=r.url,
                type="news",
                pub_date=r.pub_date,
            ))

    # --- Summary: LLM synthesis if multiple sources, else use best ---
    if len(cluster) > 1:
        summary = await _synthesize_summary(cluster, headline)
    else:
        summary = cluster[0].summary or cluster[0].headline

    # --- Confidence scoring ---
    confidence = _compute_confidence(
        source_count=len(sources),
        source_urls=[s.url for s in sources],
        dates_agree=all(
            r.event_date == event_date for r in cluster
            if r.event_date is not None
        ),
    )

    return EventRecord(
        event_date=event_date,
        event_date_precision=date_precision,
        date_confidence=date_confidence,
        headline=headline,
        summary=summary,
        key_facts=sorted(all_facts),
        direct_quotes=all_quotes,
        topics=sorted(all_topics),
        sources=sources,
        source_count=len(sources),
        confidence=confidence,
    )


def _resolve_date(
    cluster: list[ExtractionResult],
) -> tuple[date, DateConfidence, DatePrecision]:
    """Resolve the event date from a cluster of extractions.

    Strategy: majority vote. If no majority, use earliest date and flag uncertain.
    """
    date_counts: dict[date, int] = defaultdict(int)
    ambiguous_count = 0

    for r in cluster:
        if r.event_date and not r.event_date_ambiguous:
            date_counts[r.event_date] += 1
        elif r.event_date_ambiguous:
            ambiguous_count += 1

    if not date_counts:
        # No concrete dates — use preliminary dates or today
        for r in cluster:
            if r.pub_date:
                return r.pub_date, DateConfidence.UNCERTAIN, DatePrecision.DAY
        return date.today(), DateConfidence.UNCERTAIN, DatePrecision.UNKNOWN

    # Sort by count (majority vote)
    sorted_dates = sorted(date_counts.items(), key=lambda x: x[1], reverse=True)
    best_date, best_count = sorted_dates[0]

    # Determine confidence
    total_with_dates = sum(date_counts.values())
    if best_count == total_with_dates and total_with_dates >= 2:
        confidence = DateConfidence.HIGH
    elif best_count > total_with_dates / 2:
        confidence = DateConfidence.MEDIUM
    else:
        confidence = DateConfidence.UNCERTAIN

    return best_date, confidence, DatePrecision.DAY


def _pick_best_headline(cluster: list[ExtractionResult]) -> str:
    """Pick the most specific, factual headline from a cluster."""
    headlines = [r.headline for r in cluster if r.headline]
    if not headlines:
        return "Unknown Event"

    # Editorial "buzzwords" to penalize
    editorial_words = {
        "controversial", "divisive", "unprecedented", "bombshell",
        "slam", "shock", "outrage", "stunning", "breaking",
    }

    def score(h: str) -> float:
        words = h.lower().split()
        # Longer = more specific (up to a point)
        length_score = min(len(words) / 15.0, 1.0)
        # Penalize editorial buzzwords
        editorial_penalty = sum(1 for w in words if w in editorial_words) * 0.2
        return length_score - editorial_penalty

    return max(headlines, key=score)


async def _synthesize_summary(
    cluster: list[ExtractionResult],
    headline: str,
) -> str:
    """Use LLM to synthesize a neutral summary from multiple source summaries."""
    llm = _get_llm()

    source_summaries = "\n\n".join(
        f"Source {i+1} ({r.url}):\n{r.summary}"
        for i, r in enumerate(cluster)
        if r.summary
    )

    prompt = (
        f"Synthesize a single 100-200 word factual summary from these multiple "
        f"source descriptions of the same event.\n\n"
        f"EVENT: {headline}\n\n"
        f"SOURCE SUMMARIES:\n{source_summaries}\n\n"
        f"Rules:\n"
        f"- Neutral, AP/Reuters wire copy tone\n"
        f"- Include only verifiable facts\n"
        f"- No editorial language\n"
        f"- Combine unique facts from all sources\n"
        f"- Output the summary text only, no other text"
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=BIAS_STRIP_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Summary synthesis failed: {e}")
        # Fallback: use the longest summary
        best = max(cluster, key=lambda r: len(r.summary or ""))
        return best.summary or headline


# ---------------------------------------------------------------------------
# 3. Bias stripping
# ---------------------------------------------------------------------------

BIAS_STRIP_SYSTEM_PROMPT = """You are a news wire editor. Your job is to rewrite text
to AP/Reuters wire copy standards: neutral, factual, no editorial tone.

REMOVE:
- Editorializing: "controversial", "divisive", "unprecedented", "landmark"
- Attribution of intent: "in a power grab", "strategically", "shrewdly"
- Emotional language: "shock", "outrage", "bombshell", "stunning", "slammed"
- Partisan framing: "critics blast" / "supporters cheer" without both sides
- Superlatives without citation: "the biggest", "the worst", "historic"

KEEP:
- Specific facts, numbers, dates, names, places
- Direct quotes (verbatim, never paraphrased)
- Official titles and legal citations
- Factual cause and effect

BEFORE → AFTER EXAMPLES:

BEFORE: "In a stunning power grab, Trump controversially signed an executive order
that critics say will devastate the environment."
AFTER: "Trump signed an executive order directing the EPA to revise emissions standards.
Environmental groups opposed the measure."

BEFORE: "In an unprecedented and bombshell move, the administration slammed China with
massive tariffs that sent shockwaves through global markets."
AFTER: "The administration imposed 25% tariffs on $200 billion of Chinese goods.
Global equity markets declined following the announcement."

Output ONLY the cleaned text. No preamble."""


async def bias_strip(text: str) -> str:
    """Rewrite text to remove editorial bias and emotional language."""
    if not text or len(text.split()) < 10:
        return text

    llm = _get_llm()

    try:
        response = await llm.ainvoke([
            SystemMessage(content=BIAS_STRIP_SYSTEM_PROMPT),
            HumanMessage(content=f"Rewrite this to neutral wire copy:\n\n{text}"),
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Bias stripping failed: {e}")
        return text  # Return original if LLM fails


# ---------------------------------------------------------------------------
# 4. Normalization
# ---------------------------------------------------------------------------

# Common variant → canonical name mappings
ACTOR_NORMALIZATION = {
    "trump": "Donald J. Trump",
    "donald trump": "Donald J. Trump",
    "president trump": "Donald J. Trump",
    "the president": "Donald J. Trump",  # Context-dependent but default subject
    "biden": "Joseph R. Biden",
    "joe biden": "Joseph R. Biden",
    "xi": "Xi Jinping",
    "xi jinping": "Xi Jinping",
    "president xi": "Xi Jinping",
    "putin": "Vladimir Putin",
    "vladimir putin": "Vladimir Putin",
    "zelensky": "Volodymyr Zelensky",
    "zelenskyy": "Volodymyr Zelensky",
    "macron": "Emmanuel Macron",
    "netanyahu": "Benjamin Netanyahu",
    "modi": "Narendra Modi",
    "trudeau": "Justin Trudeau",
    "starmer": "Keir Starmer",
}


def normalize_actor_name(name: str) -> str:
    """Normalize an actor name to its canonical form."""
    lookup = name.strip().lower()
    return ACTOR_NORMALIZATION.get(lookup, name.strip())


def normalize_topics(topics: list[str]) -> list[str]:
    """Normalize topic tags: lowercase, underscore-separated, deduplicated."""
    normalized = set()
    for topic in topics:
        clean = topic.strip().lower().replace(" ", "_").replace("-", "_")
        if clean:
            normalized.add(clean)
    return sorted(normalized)


def validate_date_plausibility(
    event_date: date,
    collection_start: date,
    collection_end: date,
) -> bool:
    """Check if a date is plausible (within collection window, not future)."""
    today = date.today()
    if event_date > today:
        return False
    if event_date < collection_start - timedelta(days=30):  # Small buffer
        return False
    if event_date > collection_end + timedelta(days=1):
        return False
    return True


# ---------------------------------------------------------------------------
# 5. Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    source_count: int,
    source_urls: list[str],
    dates_agree: bool,
) -> float:
    """Compute confidence score for an EventRecord.

    Formula:
    - Base from source count: 1→0.4, 2→0.6, 3+→0.8
    - Adjust by avg source quality
    - +0.1 if dates agree, -0.2 if they disagree
    - Clamp to [0.0, 1.0]
    """
    # Base from source count
    if source_count >= 3:
        base = 0.8
    elif source_count == 2:
        base = 0.6
    else:
        base = 0.4

    # Average source quality adjustment
    if source_urls:
        avg_quality = sum(score_source_quality(url) for url in source_urls) / len(source_urls)
        # Scale quality adjustment: ±0.1
        quality_adj = (avg_quality - 0.5) * 0.2
    else:
        quality_adj = 0.0

    # Date agreement
    if source_count > 1:
        date_adj = 0.1 if dates_agree else -0.2
    else:
        date_adj = 0.0

    confidence = base + quality_adj + date_adj
    return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

async def cleaning_node(state: SwarmState) -> SwarmState:
    """LangGraph node: cluster, merge, clean, and normalize extraction results.

    Processes all successful ExtractionResults, producing cleaned EventRecords.
    """
    # Filter to successful extractions only
    successful = [r for r in state.extraction_results if r.extraction_success]

    if not successful:
        logger.info("Cleaning: no successful extractions to clean")
        return state

    logger.info(f"Cleaning: processing {len(successful)} successful extractions")

    # --- Step 1: Cluster ---
    clusters = cluster_extractions(successful)

    # --- Step 2-4: Merge + bias-strip + normalize, in parallel across clusters ---
    sem = asyncio.Semaphore(10)  # Cap concurrent LLM calls to avoid Gemini rate limits

    async def _process_cluster(cluster: list[ExtractionResult]) -> tuple[EventRecord | None, str | None]:
        async with sem:
            try:
                record = await merge_cluster(cluster)
                record.summary = await bias_strip(record.summary)
                record.topics = normalize_topics(record.topics)
                record.actors = [normalize_actor_name(a) for a in record.actors]

                if state.subject_name not in record.actors:
                    record.actors.insert(0, state.subject_name)

                if not validate_date_plausibility(
                    record.event_date,
                    state.collection_start,
                    state.collection_end,
                ):
                    record.date_confidence = DateConfidence.UNCERTAIN
                    logger.warning(
                        f"Date {record.event_date} outside collection window "
                        f"for '{record.headline}' — flagged uncertain"
                    )

                return record, None
            except Exception as e:
                return None, f"Failed to merge cluster (headlines: {[r.headline for r in cluster]}): {e}"

    merge_results = await asyncio.gather(*[_process_cluster(c) for c in clusters])

    new_records: list[EventRecord] = []
    for record, error in merge_results:
        if error:
            logger.error(error)
            state.errors.append(error)
        elif record is not None:
            new_records.append(record)

    # Clear processed extractions, add cleaned records
    state.extraction_results = []
    state.cleaned_records.extend(new_records)

    logger.info(
        f"Cleaning complete: {len(new_records)} EventRecords produced "
        f"(from {len(successful)} extractions in {len(clusters)} clusters)"
    )

    return state
