# Build Step 7: Coverage Auditor

## What this builds
`src/agents/coverage_auditor.py` — The agent that analyzes the database for coverage gaps and triggers additional research rounds.

## Context
The Coverage Auditor is the "quality assurance" layer. After a research round completes and events are indexed, the Auditor checks whether the database has sufficient coverage across all months and topic areas. If gaps are found, it reports them back to the Coordinator, which plans follow-up research. This creates the autonomous feedback loop.

## Existing code it depends on
- `src/models.py` → `SwarmState`
- `src/database.py` → `get_event_count_by_month`, `async_session`
- `src/config.py` → `settings`
- Indexing must be done first (events must be in the database to audit)

## What the output file must contain

### 1. Monthly Coverage Analysis
- Query `get_event_count_by_month()` from the database
- Calculate the expected minimum events per month:
  - Baseline: at least 15 events per month for a major world leader
  - Months with known high activity (inauguration, state of the union, major summits) should have 25+
- Flag months below the minimum as coverage gaps

### 2. Topic Coverage Analysis
- For each month, check topic distribution
- A month with 20 events all about "tariffs" and 0 about "foreign_policy" has a topic gap
- Use an LLM to suggest which topic areas seem underrepresented compared to what's expected for that time period

### 3. Recency Bias Detection
- Check if recent months have disproportionately more events than older months
- This suggests search engines are returning recent results for historical queries
- Flag if any month has 3x+ the average — likely over-collected

### 4. Gap Report Generation
Output a structured gap report:
```
{
  "sparse_months": ["2024-03", "2024-07"],
  "missing_topics": {"2024-03": ["foreign_policy", "personnel"], "2024-07": ["economic"]},
  "over_collected_months": ["2025-03"],
  "total_events": 450,
  "quarantined_events": 23,
  "recommendation": "2 additional research rounds needed"
}
```

### 5. Completion Criteria
The collection is "complete" when:
- Every month has >= 15 events
- No month has a topic category with 0 events
- The quarantine rate is < 15% of total events
- At least 3 research rounds have been completed (prevents premature stop)

### 6. LangGraph Node Function
A `coverage_auditor_node(state: SwarmState) -> SwarmState` function that:
- Queries the database for coverage stats
- Updates `state.events_per_month`
- Sets `state.coverage_gaps` with specific gap descriptions
- If all criteria met → set `state.collection_complete = True`
- Returns updated state

## Quality bar
- Must correctly identify months with < 15 events
- Must not mark collection as complete if major gaps exist
- Gap descriptions must be specific enough for the Coordinator to generate targeted queries
