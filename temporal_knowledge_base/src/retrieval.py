"""CHRONOS retrieval interface — the public API for querying the knowledge base."""

from __future__ import annotations

from datetime import date

from .config import ModelConfig, settings
from .database import async_session, retrieve_events
from .embeddings import embed_query
from .models import RetrievalRequest, RetrievalResult


async def retrieve(
    query: str,
    simulation_date: date,
    model_name: str,
    top_k: int | None = None,
    topic_filter: list[str] | None = None,
    subject_name: str | None = None,
) -> str:
    """Retrieve events and return an LLM-readable intelligence briefing.

    This is the main entry point for downstream consumers (the prediction
    pipeline, the question tester, etc.).

    Args:
        query: The question or context to search for.
        simulation_date: The date the question is "set in" — upper bound.
        model_name: The LLM model name — determines training cutoff.
        top_k: Number of events to retrieve (default from settings).
        topic_filter: Optional topic filter.
        subject_name: The leader's name for the briefing header.

    Returns:
        A formatted intelligence briefing string, ready for LLM consumption.
    """
    # Resolve parameters
    model_cutoff = ModelConfig.get_cutoff(model_name)
    k = top_k or settings.default_top_k
    subject = subject_name or settings.collection_subject

    # Embed the query
    query_embedding = await embed_query(query)

    # Retrieve with temporal filtering
    async with async_session() as session:
        result = await retrieve_events(
            session=session,
            query_embedding=query_embedding,
            simulation_date=simulation_date,
            model_training_cutoff=model_cutoff,
            top_k=k,
            topic_filter=topic_filter,
        )

    # Set the query text for the result
    result.query = query

    # Format as briefing
    return result.to_briefing(subject_name=subject)


async def retrieve_structured(
    request: RetrievalRequest,
    subject_name: str | None = None,
) -> RetrievalResult:
    """Retrieve events and return the structured result (for programmatic use).

    Use this when you need the raw EventRecord objects, not the formatted briefing.
    """
    query_embedding = await embed_query(request.query)

    async with async_session() as session:
        result = await retrieve_events(
            session=session,
            query_embedding=query_embedding,
            simulation_date=request.simulation_date,
            model_training_cutoff=request.model_training_cutoff,
            top_k=request.top_k,
            topic_filter=request.topic_filter,
        )

    result.query = request.query
    return result
