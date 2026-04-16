"""CHRONOS Indexing Agent — embedding generation and database insertion.

The Indexing Agent is the "librarian" of the swarm. It:
1. Takes validated EventRecords from the Temporal Validator
2. Generates vector embeddings (headline + summary combined for best retrieval)
3. Inserts records + embeddings into PostgreSQL/pgvector
4. Handles deduplication (upsert/skip on duplicate record_id)
5. Includes retry logic with exponential backoff for DB connection issues

This is the final step before an event becomes retrievable.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError

from ..config import settings
from ..database import EventRecordRow, async_session, init_db, insert_event
from ..embeddings import embed_texts
from ..models import EventRecord, SwarmState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 20  # Records per embedding batch
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0  # seconds


# ---------------------------------------------------------------------------
# Embedding text preparation
# ---------------------------------------------------------------------------

def prepare_embedding_text(record: EventRecord) -> str:
    """Combine headline + summary for optimal embedding quality.

    The headline provides specificity (who, what),
    the summary provides context (why, how).
    Combined = best retrieval performance.
    """
    return f"{record.headline}. {record.summary}"


# ---------------------------------------------------------------------------
# Duplicate checking
# ---------------------------------------------------------------------------

async def check_existing_record_ids(record_ids: list[str]) -> set[str]:
    """Check which record_ids already exist in the database.

    Returns set of record_ids that already exist (should be skipped).
    """
    if not record_ids:
        return set()

    try:
        async with async_session() as session:
            stmt = (
                select(EventRecordRow.record_id)
                .where(EventRecordRow.record_id.in_(record_ids))
            )
            result = await session.execute(stmt)
            existing = {row[0] for row in result}
            return existing
    except Exception as e:
        logger.warning(f"Could not check existing records: {e}")
        return set()  # Proceed and let IntegrityError handle dupes


# ---------------------------------------------------------------------------
# Batch insertion with retry
# ---------------------------------------------------------------------------

async def insert_batch_with_retry(
    records: list[EventRecord],
    embeddings: list[list[float]],
) -> tuple[int, int]:
    """Insert a batch of records + embeddings with retry on failure.

    Returns:
        (inserted_count, skipped_count)
    """
    inserted = 0
    skipped = 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with async_session() as session:
                async with session.begin():
                    for record, embedding in zip(records, embeddings):
                        try:
                            await insert_event(session, record, embedding)
                            inserted += 1
                            logger.debug(f"Indexed: {record.record_id} — {record.headline}")
                        except IntegrityError:
                            await session.rollback()
                            skipped += 1
                            logger.info(
                                f"Skipped duplicate: {record.record_id} — {record.headline}"
                            )
                            # Re-enter transaction after rollback
                            async with session.begin():
                                pass
                            continue
            return inserted, skipped

        except OperationalError as e:
            if attempt < MAX_RETRIES:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"DB connection error (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                inserted = 0
                skipped = 0
            else:
                logger.error(f"DB insertion failed after {MAX_RETRIES} retries: {e}")
                raise

    return inserted, skipped


# ---------------------------------------------------------------------------
# Core indexing pipeline
# ---------------------------------------------------------------------------

async def index_records(records: list[EventRecord]) -> tuple[int, int, list[str]]:
    """Index a list of validated EventRecords into the database.

    Steps:
    1. Filter out duplicates already in DB
    2. Generate embeddings in batches
    3. Insert records + embeddings in batches

    Returns:
        (total_inserted, total_skipped, errors)
    """
    if not records:
        return 0, 0, []

    # --- Step 1: Filter existing duplicates ---
    record_ids = [r.record_id for r in records]
    existing_ids = await check_existing_record_ids(record_ids)

    if existing_ids:
        logger.info(f"Skipping {len(existing_ids)} records already in database")
        new_records = [r for r in records if r.record_id not in existing_ids]
    else:
        new_records = records

    if not new_records:
        logger.info("All records already exist in database — nothing to index")
        return 0, len(existing_ids), []

    total_inserted = 0
    total_skipped = len(existing_ids)
    errors: list[str] = []

    # --- Step 2+3: Batch embed + insert ---
    total = len(new_records)
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = new_records[batch_start:batch_end]

        logger.info(f"Indexing batch {batch_start + 1}-{batch_end}/{total}...")

        # Generate embeddings
        try:
            texts = [prepare_embedding_text(r) for r in batch]
            embeddings = await embed_texts(texts)
        except Exception as e:
            error_msg = f"Embedding generation failed for batch {batch_start + 1}-{batch_end}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue

        # Insert into database
        try:
            inserted, skipped = await insert_batch_with_retry(batch, embeddings)
            total_inserted += inserted
            total_skipped += skipped
        except Exception as e:
            error_msg = f"DB insertion failed for batch {batch_start + 1}-{batch_end}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return total_inserted, total_skipped, errors


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

async def indexing_node(state: SwarmState) -> SwarmState:
    """LangGraph node: generate embeddings and insert validated records into the DB.

    Takes state.validated_records → embeds → inserts → updates state.indexed_count.
    """
    if not state.validated_records:
        logger.info("Indexing: no validated records to index")
        return state

    records_to_index = state.validated_records[:]
    state.validated_records = []

    logger.info(f"Indexing: processing {len(records_to_index)} validated records")

    # Ensure database is initialized
    try:
        await init_db()
    except Exception as e:
        error_msg = f"Database initialization failed: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        # Put records back — don't lose them
        state.validated_records.extend(records_to_index)
        return state

    # Run indexing pipeline
    inserted, skipped, errors = await index_records(records_to_index)

    state.indexed_count += inserted
    state.errors.extend(errors)

    logger.info(
        f"Indexing complete: {inserted} inserted, {skipped} skipped, "
        f"{len(errors)} errors (total indexed: {state.indexed_count})"
    )

    return state
