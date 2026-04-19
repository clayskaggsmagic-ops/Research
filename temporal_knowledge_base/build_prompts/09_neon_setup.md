# Build Step 9: Neon Database Setup

## What this builds
Database infrastructure using the Neon MCP server — create a project, enable pgvector, run the schema migration, and wire the connection string into the application.

## Context
The user has a Neon MCP available. Use it to set up a real PostgreSQL + pgvector database in the cloud. This replaces any local PostgreSQL dependency.

## What this step must do

### 1. Create Neon Project
- Use the Neon MCP `create_project` tool to create a new project named "chronos"
- Note the project ID and connection string

### 2. Enable pgvector
- Use `run_sql` to execute: `CREATE EXTENSION IF NOT EXISTS vector;`

### 3. Run Schema Migration
- Use `run_sql` to create the `event_records` table matching the SQLAlchemy ORM in `src/database.py`:
  ```sql
  CREATE TABLE IF NOT EXISTS event_records (
      id SERIAL PRIMARY KEY,
      record_id VARCHAR(64) UNIQUE NOT NULL,
      event_date DATE NOT NULL,
      event_date_precision VARCHAR(16) DEFAULT 'day',
      date_confidence VARCHAR(32) DEFAULT 'high',
      date_verification_method TEXT DEFAULT '',
      ingestion_date TIMESTAMP DEFAULT NOW(),
      headline TEXT NOT NULL,
      summary TEXT NOT NULL,
      key_facts TEXT[] DEFAULT '{}',
      direct_quotes JSONB DEFAULT '[]',
      topics TEXT[] DEFAULT '{}',
      actors TEXT[] DEFAULT '{}',
      sources JSONB DEFAULT '[]',
      source_count INTEGER DEFAULT 0,
      confidence FLOAT DEFAULT 0.0,
      embedding vector(768)
  );

  CREATE INDEX IF NOT EXISTS idx_event_date ON event_records (event_date);
  CREATE INDEX IF NOT EXISTS idx_date_confidence ON event_records (date_confidence);
  CREATE INDEX IF NOT EXISTS idx_record_id ON event_records (record_id);
  CREATE INDEX IF NOT EXISTS idx_topics ON event_records USING GIN (topics);
  CREATE INDEX IF NOT EXISTS idx_actors ON event_records USING GIN (actors);
  ```

### 4. Create Vector Index
- Use `run_sql` to create the pgvector index:
  ```sql
  CREATE INDEX IF NOT EXISTS idx_embedding ON event_records 
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
  ```
  Note: For < 1000 records, skip ivfflat and use exact search. Create the ivfflat index later when the dataset grows.

### 5. Update .env
- Write the Neon connection string to `temporal_knowledge_base/.env`
- Format: `DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname?sslmode=require`

### 6. Test Connection
- Use `run_sql` to verify: `SELECT COUNT(*) FROM event_records;` (should return 0)

## Quality bar
- pgvector extension must be enabled
- Table schema must exactly match the SQLAlchemy ORM
- Connection string must use `asyncpg` driver format
- SSL must be enabled (Neon requires it)
