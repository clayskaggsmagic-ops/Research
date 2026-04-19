"""CHRONOS database layer — PostgreSQL + pgvector for temporal event storage and retrieval."""

from __future__ import annotations

from datetime import date

import numpy as np
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from pgvector.sqlalchemy import Vector

from .config import DateConfidence, settings
from .models import EventRecord, RetrievalResult, Source, DirectQuote


# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class EventRecordRow(Base):
    """PostgreSQL table for event records — the core knowledge base table."""

    __tablename__ = "event_records"

    # Identity
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    # Temporal — THE critical fields
    event_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    event_date_precision: Mapped[str] = mapped_column(String(16), default="day")
    date_confidence: Mapped[str] = mapped_column(String(32), default="high", index=True)
    date_verification_method: Mapped[str] = mapped_column(Text, default="")
    ingestion_date: Mapped[date] = mapped_column(DateTime, server_default=func.now())

    # Content
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_facts = Column(ARRAY(Text), default=[])
    direct_quotes = Column(JSONB, default=[])

    # Classification
    topics = Column(ARRAY(Text), default=[])
    actors = Column(ARRAY(Text), default=[])

    # Provenance
    sources = Column(JSONB, default=[])
    source_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Vector embedding of the summary
    embedding = Column(Vector(settings.embedding_dimensions))

    def to_event_record(self) -> EventRecord:
        """Convert ORM row back to Pydantic model."""
        return EventRecord(
            record_id=self.record_id,
            event_date=self.event_date,
            event_date_precision=self.event_date_precision,
            date_confidence=DateConfidence(self.date_confidence),
            date_verification_method=self.date_verification_method,
            headline=self.headline,
            summary=self.summary,
            key_facts=self.key_facts or [],
            direct_quotes=[DirectQuote(**q) for q in (self.direct_quotes or [])],
            topics=self.topics or [],
            actors=self.actors or [],
            sources=[Source(**s) for s in (self.sources or [])],
            source_count=self.source_count,
            confidence=self.confidence,
        )


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create tables and enable pgvector extension."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

async def insert_event(session: AsyncSession, record: EventRecord, embedding: list[float]) -> str:
    """Insert an EventRecord into the database with its embedding vector."""
    row = EventRecordRow(
        record_id=record.record_id,
        event_date=record.event_date,
        event_date_precision=record.event_date_precision.value,
        date_confidence=record.date_confidence.value,
        date_verification_method=record.date_verification_method,
        headline=record.headline,
        summary=record.summary,
        key_facts=record.key_facts,
        direct_quotes=[q.model_dump() for q in record.direct_quotes],
        topics=record.topics,
        actors=record.actors,
        sources=[s.model_dump(mode="json") for s in record.sources],
        source_count=record.source_count,
        confidence=record.confidence,
        embedding=embedding,
    )
    session.add(row)
    await session.flush()
    return record.record_id


async def retrieve_events(
    session: AsyncSession,
    query_embedding: list[float],
    simulation_date: date,
    model_training_cutoff: date,
    top_k: int = 15,
    topic_filter: list[str] | None = None,
) -> RetrievalResult:
    """Retrieve events within the temporal window, ranked by vector similarity.

    This is THE core retrieval function. The temporal sliding window is
    enforced here via hard SQL filters:
      - event_date <= simulation_date   (upper bound — from the question)
      - event_date >= model_training_cutoff  (lower bound — from the model)
      - date_confidence != 'uncertain'  (quarantined records excluded)

    Records after simulation_date PHYSICALLY CANNOT appear in results.
    """
    # Build the base query
    stmt = (
        select(EventRecordRow)
        .where(EventRecordRow.event_date <= simulation_date)
        .where(EventRecordRow.event_date >= model_training_cutoff)
        .where(EventRecordRow.date_confidence != DateConfidence.UNCERTAIN.value)
    )

    # Optional topic filter
    if topic_filter:
        stmt = stmt.where(EventRecordRow.topics.overlap(topic_filter))

    # Order by vector similarity (cosine distance via pgvector <=> operator)
    embedding_vec = np.array(query_embedding, dtype=np.float32)
    stmt = stmt.order_by(EventRecordRow.embedding.cosine_distance(embedding_vec))
    stmt = stmt.limit(top_k)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    # Get total event count in window (for the briefing header)
    count_stmt = (
        select(func.count())
        .select_from(EventRecordRow)
        .where(EventRecordRow.event_date <= simulation_date)
        .where(EventRecordRow.event_date >= model_training_cutoff)
        .where(EventRecordRow.date_confidence != DateConfidence.UNCERTAIN.value)
    )
    total_count = await session.scalar(count_stmt) or 0

    events = [row.to_event_record() for row in rows]

    return RetrievalResult(
        events=events,
        query="",  # Filled by caller
        simulation_date=simulation_date,
        model_training_cutoff=model_training_cutoff,
        total_events_in_window=total_count,
    )


async def get_event_count_by_month(
    session: AsyncSession,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, int]:
    """Get event counts grouped by month — used by the Coverage Auditor."""
    stmt = select(
        func.to_char(EventRecordRow.event_date, "YYYY-MM").label("month"),
        func.count().label("count"),
    ).where(EventRecordRow.date_confidence != DateConfidence.UNCERTAIN.value)

    if start_date:
        stmt = stmt.where(EventRecordRow.event_date >= start_date)
    if end_date:
        stmt = stmt.where(EventRecordRow.event_date <= end_date)

    stmt = stmt.group_by(text("month")).order_by(text("month"))

    result = await session.execute(stmt)
    return {row.month: row.count for row in result}


async def get_quarantined_count(session: AsyncSession) -> int:
    """Count records in quarantine (uncertain dates)."""
    stmt = (
        select(func.count())
        .select_from(EventRecordRow)
        .where(EventRecordRow.date_confidence == DateConfidence.UNCERTAIN.value)
    )
    return await session.scalar(stmt) or 0
