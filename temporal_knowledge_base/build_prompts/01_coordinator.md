# Build Step 1: Coordinator Agent

## What this builds
`src/agents/coordinator.py` — The supervisor agent that plans research campaigns, assigns work to other agents, and prevents redundant research.

## Context
The Coordinator is the "brain" of the research swarm. It does NOT do any searching or scraping itself. It:
1. Takes in a leader name + time range
2. Generates a research plan (list of topic/time chunks to investigate)
3. Dispatches work to the Discovery Agent
4. Tracks what's been covered to avoid duplicating effort
5. Receives coverage reports from the Coverage Auditor and plans follow-up research to fill gaps

## Existing code it depends on
- `src/models.py` → `SwarmState` (the shared state it reads/writes)
- `src/config.py` → `settings` (for `collection_subject`, `collection_start`, `research_model`)

## What the output file must contain

### 1. Research Plan Generation
An LLM-powered function that takes a leader name and date range and outputs a structured list of research queries. The plan should:
- Break the time range into monthly chunks
- For each month, generate 5-8 diverse search queries covering:
  - Executive actions (orders, memos, directives)
  - Foreign policy (summits, sanctions, diplomatic statements)
  - Domestic policy (legislation, vetoes, regulatory actions)
  - Economic decisions (tariffs, trade deals, fiscal policy)
  - Personnel (appointments, firings, nominations)
  - Crises and responses (natural disasters, security events)
  - Public statements (press conferences, social media, rallies)
- Tag each query with its target month and topic category

### 2. Deduplication Tracking
A mechanism to track which URLs and topics have already been researched, so the Coordinator doesn't re-dispatch redundant work. This should use the `SwarmState.urls_visited` set and implement fuzzy headline matching.

### 3. Gap-Filling Logic
When the Coverage Auditor reports sparse months or missing topics, the Coordinator generates NEW, more specific search queries targeting those gaps. This is the feedback loop that makes the swarm autonomous.

### 4. LangGraph Node Function
A `coordinator_node(state: SwarmState) -> SwarmState` function that can be wired into the LangGraph StateGraph. It should:
- Read the current state
- If `research_plan` is empty → generate initial plan
- If `coverage_gaps` is non-empty → generate follow-up queries for gaps
- If coverage is sufficient → set `collection_complete = True`
- Write updated plan back to state

## Prompt engineering requirements
- The LLM prompt for plan generation must explicitly instruct: "Generate search queries that will find events with SPECIFIC DATES. Avoid queries that will return undated opinion pieces or analysis."
- Include few-shot examples of good vs bad queries.

## Quality bar
- The coordinator should generate at least 50 initial queries for a 12-month range
- Follow-up queries should be more specific than initial ones (e.g., initial: "Trump tariff decisions January 2025", follow-up: "Trump steel tariff Section 232 January 2025 specific date")
