# Build Step 8: LangGraph Pipeline

## What this builds
`src/pipeline.py` — The LangGraph StateGraph that wires all 7 agents into an autonomous research pipeline with a supervisor loop.

## Context
This is the orchestration layer. It connects all the agents into a directed graph with conditional routing. The key feature is the **coverage loop**: after indexing, the Coverage Auditor checks for gaps. If gaps exist, control returns to the Coordinator, which plans follow-up research, and the discovery→extraction→cleaning→validation→indexing cycle repeats.

## Existing code it depends on
- ALL agent files must be done first (01-07)
- `src/models.py` → `SwarmState`
- `src/config.py` → `settings`

## What the output file must contain

### 1. StateGraph Definition
A LangGraph `StateGraph` with the following nodes:
- `coordinator` → `coordinator_node`
- `discovery` → `discovery_node`
- `extraction` → `extraction_node`
- `cleaning` → `cleaning_node`
- `temporal_validator` → `temporal_validator_node`
- `indexing` → `indexing_node`
- `coverage_auditor` → `coverage_auditor_node`

### 2. Edge Routing
```
START → coordinator → discovery → extraction → cleaning → temporal_validator → indexing → coverage_auditor → CONDITIONAL
```

The conditional edge after `coverage_auditor`:
- If `state.collection_complete == True` → END
- If `state.collection_complete == False` → `coordinator` (loop back)

### 3. Safety Limits
- Maximum loop iterations: 5 (prevent infinite loops if coverage never converges)
- Track loop count in state
- If max loops reached, END with warning in state.errors

### 4. Pipeline Entry Point
A `run_pipeline(subject_name: str, start_date: date, end_date: date) -> SwarmState` async function that:
- Creates initial SwarmState
- Compiles and invokes the graph
- Returns the final state
- Prints progress via `rich` console

### 5. CLI Entry Point
A `__main__` block or `main()` function that:
- Parses command-line args (subject name, date range)
- Runs the pipeline
- Prints a summary report (events indexed, quarantined, coverage stats)

Example invocation:
```bash
uv run python -m src.pipeline --subject "Donald J. Trump" --start 2023-10-01 --end 2025-04-01
```

## Quality bar
- Pipeline must run end-to-end without manual intervention
- Coverage loop must correctly re-trigger research for sparse months
- Must handle agent failures gracefully (log error, skip to next step)
- Total runtime for a 12-month range should be < 2 hours with API rate limits
