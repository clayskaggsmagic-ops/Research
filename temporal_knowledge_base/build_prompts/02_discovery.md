# Build Step 2: Discovery Agent

## What this builds
`src/agents/discovery.py` — The agent that executes web searches, queries structured APIs (GDELT), and returns raw event candidates.

## Context
The Discovery Agent takes search queries from the Coordinator and finds raw source material. It does NOT parse or validate — it just finds URLs and snippets. It is the "eyes" of the swarm.

## Existing code it depends on
- `src/models.py` → `SwarmState`, `RawEventCandidate`
- `src/config.py` → `settings` (for API keys, `research_model`)
- Coordinator must be done first (it generates the queries Discovery executes)

## What the output file must contain

### 1. Web Search Tool
A wrapper around a web search API (Serper.dev or Google Search via LangChain) that:
- Takes a search query string
- Returns structured results: `[{title, url, snippet, date_hint}]`
- Filters out known garbage domains (pinterest, quora answers, forums)
- Rate-limits to avoid API throttling (configurable delay between calls)
- Logs every query executed for auditability

### 2. GDELT Integration (Optional but preferred)
A client for the GDELT Event API that:
- Queries events by actor name + date range
- Returns structured event records with dates, actors, and source URLs
- Provides a "free tier" data source that doesn't require API keys

### 3. Source Quality Scoring
A simple heuristic scorer that ranks discovered sources:
- **Tier 1** (1.0): Government sites (.gov), wire services (AP, Reuters, AFP)
- **Tier 2** (0.8): Major newspapers (NYT, WaPo, WSJ, BBC, Guardian)
- **Tier 3** (0.6): Regional/specialty outlets
- **Tier 4** (0.3): Blogs, opinion sites, social media
- Unknown domains default to 0.5

### 4. Deduplication at Discovery
Before adding a candidate to the state, check:
- URL not in `state.urls_visited`
- Headline not a near-duplicate of existing candidates (simple fuzzy match — substring overlap > 80%)

### 5. LangGraph Node Function
A `discovery_node(state: SwarmState) -> SwarmState` function that:
- Pops the next N queries from `state.research_plan` (batch size configurable, default 5)
- Executes searches for each query
- Converts results to `RawEventCandidate` objects
- Adds new candidates to `state.raw_candidates`
- Adds visited URLs to `state.urls_visited`
- Returns updated state

## Quality bar
- Must handle search API failures gracefully (retry once, then skip with error log)
- Must not add duplicate URLs to candidates
- Should discover 10-30 candidates per query batch
