# Build Step 6: Indexing Agent

## What this builds
`src/agents/indexing.py` — The agent that takes validated EventRecords, generates embeddings, and inserts them into the PostgreSQL + pgvector database.

## Context
The Indexing Agent is the "librarian" — it takes validated, clean, date-verified events and commits them to the permanent knowledge base. This is the final step before an event becomes retrievable.

## Existing code it depends on
- `src/models.py` → `SwarmState`, `EventRecord`
- `src/database.py` → `insert_event`, `async_session`, `init_db`
- `src/embeddings.py` → `embed_text`, `embed_texts`
- Temporal Validator must be done first (it produces `validated_records`)

## What the output file must contain

### 1. Batch Embedding Generation
- Take validated records and generate embeddings for their `summary` field
- Use batch embedding (`embed_texts`) for efficiency — batch size of 20
- The embedding text should be: `"{headline}. {summary}"` — combining headline and summary gives the best retrieval quality

### 2. Database Insertion
- Insert each record + embedding into the database via `insert_event()`
- Handle duplicate `record_id` gracefully (upsert or skip)
- Track insertion count in `state.indexed_count`

### 3. Batch Processing
- Process records in batches to avoid memory issues
- After each batch, commit the transaction
- Log progress: "Indexed 20/150 records..."

### 4. LangGraph Node Function
A `indexing_node(state: SwarmState) -> SwarmState` function that:
- Ensures database is initialized (`init_db()`)
- Takes `state.validated_records`
- Generates embeddings in batches
- Inserts into database
- Updates `state.indexed_count`
- Returns updated state

## Quality bar
- Must handle database connection failures gracefully (retry with backoff)
- Must not insert duplicate records
- Batch size should be configurable
- Should log every insertion for auditability
