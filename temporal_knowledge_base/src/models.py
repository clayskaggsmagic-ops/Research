"""CHRONOS data models — the Event Record and related Pydantic schemas."""

from __future__ import annotations

from datetime import date, datetime
from time import strftime
from uuid import uuid4

from pydantic import BaseModel, Field

from .config import DateConfidence, DatePrecision


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DirectQuote(BaseModel):
    """A verbatim quote from a named speaker."""

    speaker: str
    quote: str
    context: str = ""


class Source(BaseModel):
    """Provenance for an event — where the information came from."""

    name: str
    url: str = ""
    type: str = ""  # "primary", "wire_service", "news", "government", "social_media"
    pub_date: date | None = None


# ---------------------------------------------------------------------------
# The Event Record — the core unit of the knowledge base
# ---------------------------------------------------------------------------

class EventRecord(BaseModel):
    """A single, timestamped, bias-stripped event in the CHRONOS knowledge base.

    Every piece of information is stored as an EventRecord. Each record
    represents ONE thing that happened on ONE date. No summaries spanning
    time ranges. No editorial opinion. Just facts.

    The `event_date` field is the most critical field in the entire system.
    It determines which side of the temporal sliding window this record
    falls on. A wrong date IS data leakage.
    """

    # --- Identity ---
    record_id: str = Field(default_factory=lambda: f"EVT-{uuid4().hex[:12]}")

    # --- Temporal (THE critical fields) ---
    event_date: date = Field(description="When this event happened (NOT when the article was published)")
    event_date_precision: DatePrecision = Field(default=DatePrecision.DAY)
    date_confidence: DateConfidence = Field(default=DateConfidence.HIGH)
    date_verification_method: str = Field(
        default="",
        description="How the date was verified: 'cross_referenced_3_sources', 'primary_document', etc.",
    )
    ingestion_date: datetime = Field(default_factory=datetime.utcnow)

    # --- Content (LLM-readable) ---
    headline: str = Field(description="One-line summary of the event")
    summary: str = Field(
        description="100-200 word factual summary, optimized for embedding quality and LLM consumption"
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Structured bullet points: specific numbers, dates, thresholds, legal citations",
    )
    direct_quotes: list[DirectQuote] = Field(
        default_factory=list,
        description="Verbatim quotes from the leader and other actors",
    )

    # --- Classification (freeform, no rigid taxonomy) ---
    topics: list[str] = Field(
        default_factory=list,
        description="Freeform topic tags: ['tariffs', 'canada', 'steel', 'section_232']",
    )
    actors: list[str] = Field(
        default_factory=list,
        description="Named actors involved: ['donald_trump', 'justin_trudeau']",
    )

    # --- Provenance ---
    sources: list[Source] = Field(default_factory=list)
    source_count: int = Field(default=0)
    confidence: float = Field(
        default=0.0,
        description="Overall confidence in this record (0.0-1.0), factoring in source count and date confidence",
    )

    def to_briefing_text(self) -> str:
        """Format this record as LLM-readable briefing text.

        This is the format that gets injected into the LLM's context window.
        Natural language, not structured data.
        """
        lines = [f"[{self.event_date.isoformat()}] {self.headline}"]
        lines.append(self.summary)

        if self.key_facts:
            lines.append("Key facts:")
            for fact in self.key_facts:
                lines.append(f"  • {fact}")

        if self.direct_quotes:
            for q in self.direct_quotes:
                ctx = f" ({q.context})" if q.context else ""
                lines.append(f'"{q.quote}" — {q.speaker}{ctx}')

        source_names = [s.name for s in self.sources]
        confidence_note = f"{self.source_count} source{'s' if self.source_count != 1 else ''}"
        if self.date_confidence == DateConfidence.VERIFIED:
            confidence_note += ", verified date"
        elif self.date_confidence == DateConfidence.APPROXIMATE:
            confidence_note += f", approximate date ({self.event_date_precision.value})"
        lines.append(f"Sources: {', '.join(source_names)} ({confidence_note})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrieval request / response models
# ---------------------------------------------------------------------------

class RetrievalRequest(BaseModel):
    """A request to retrieve events from the knowledge base."""

    query: str = Field(description="The question or search text")
    simulation_date: date = Field(description="The temporal upper bound — from the question")
    model_training_cutoff: date = Field(
        description="The model's training cutoff — the temporal lower bound"
    )
    top_k: int = Field(default=15, description="Number of events to retrieve")
    topic_filter: list[str] | None = Field(
        default=None, description="Optional: restrict to specific topics"
    )


class RetrievalResult(BaseModel):
    """The result of a retrieval query — formatted for LLM consumption."""

    events: list[EventRecord]
    query: str
    simulation_date: date
    model_training_cutoff: date
    total_events_in_window: int = Field(
        description="Total events available in the [cutoff, simulation_date] window"
    )

    def to_briefing(self, subject_name: str = "Donald J. Trump") -> str:
        """Format the full set of results as an LLM-readable intelligence briefing."""
        header = (
            f"{'═' * 60}\n"
            f" INTELLIGENCE BRIEFING — As of {self.simulation_date.isoformat()}\n"
            f" Subject: {subject_name}\n"
            f" Knowledge Window: {self.model_training_cutoff.isoformat()} → {self.simulation_date.isoformat()}\n"
            f" Events in window: {self.total_events_in_window} | Showing top {len(self.events)}\n"
            f"{'═' * 60}\n"
        )

        body_parts = []
        for event in self.events:
            body_parts.append(event.to_briefing_text())

        body = f"\n{'─' * 60}\n".join(body_parts)

        footer = (
            f"\n{'═' * 60}\n"
            f" END BRIEFING — {len(self.events)} events retrieved\n"
            f"{'═' * 60}"
        )

        return header + "\n" + body + footer


# ---------------------------------------------------------------------------
# Pipeline state models (for the LangGraph research swarm)
# ---------------------------------------------------------------------------

class RawEventCandidate(BaseModel):
    """An unprocessed event found by the Discovery Agent."""

    url: str = ""
    title: str = ""
    snippet: str = ""
    preliminary_date: date | None = None
    source_name: str = ""
    discovery_query: str = Field(default="", description="The search query that found this")


class ExtractionResult(BaseModel):
    """Output of the Extraction Agent — parsed article content."""

    url: str
    headline: str = ""
    summary: str = Field(default="", description="100-200 word factual summary of the event")
    full_text: str = ""
    pub_date: date | None = None
    event_date: date | None = None
    event_date_ambiguous: bool = False
    quotes: list[DirectQuote] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list, description="Specific numbers, thresholds, legal citations")
    topics: list[str] = Field(default_factory=list, description="Freeform tags relevant to the event")
    is_opinion: bool = Field(default=False, description="True if article is editorial/opinion")
    word_count: int = 0
    extraction_success: bool = True
    failure_reason: str = ""


class SwarmState(BaseModel):
    """Shared state for the LangGraph research swarm.

    This is the central state object that flows through the pipeline.
    Each agent reads from and writes to this state.
    """

    # --- Config ---
    subject_name: str = "Donald J. Trump"
    collection_start: date = Field(default_factory=lambda: date(2023, 10, 1))
    collection_end: date = Field(default_factory=date.today)

    # --- Checkpoint/Resume ---
    run_id: str = Field(
        default_factory=lambda: strftime("%Y%m%d_%H%M%S"),
        description="Unique run identifier for grouping checkpoints",
    )
    last_completed_node: str = Field(
        default="",
        description="Name of the last successfully completed pipeline node",
    )

    # --- Discovery ---
    research_plan: list[str] = Field(
        default_factory=list, description="Topics/time chunks to research"
    )
    raw_candidates: list[RawEventCandidate] = Field(default_factory=list)
    urls_visited: set[str] = Field(default_factory=set)

    # --- Extraction ---
    extraction_results: list[ExtractionResult] = Field(default_factory=list)

    # --- Cleaning ---
    cleaned_records: list[EventRecord] = Field(default_factory=list)

    # --- Validation ---
    validated_records: list[EventRecord] = Field(default_factory=list)
    quarantined_records: list[EventRecord] = Field(
        default_factory=list, description="Records with uncertain dates — not indexed"
    )

    # --- Indexing ---
    indexed_count: int = 0
    loop_count: int = 0

    # --- Coverage ---
    events_per_month: dict[str, int] = Field(default_factory=dict)
    coverage_gaps: list[str] = Field(
        default_factory=list, description="Months/topics with insufficient coverage"
    )
    collection_complete: bool = False

    # --- Errors ---
    errors: list[str] = Field(default_factory=list)
