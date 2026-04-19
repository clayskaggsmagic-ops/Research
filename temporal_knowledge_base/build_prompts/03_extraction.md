# Build Step 3: Extraction Agent

## What this builds
`src/agents/extraction.py` — The agent that takes raw URLs/snippets from Discovery and extracts structured event data from the actual articles.

## Context
The Extraction Agent is the "hands" of the swarm. It fetches full article text, identifies the KEY EVENT described, extracts dates, quotes, and facts, and outputs structured `ExtractionResult` objects. It does NOT validate dates — that's the Temporal Validator's job.

## Existing code it depends on
- `src/models.py` → `SwarmState`, `RawEventCandidate`, `ExtractionResult`, `DirectQuote`
- `src/config.py` → `settings` (for `research_model`)
- Discovery must be done first (it produces the `RawEventCandidate` list)

## What the output file must contain

### 1. Article Fetcher
A robust web scraper that:
- Fetches article content from a URL using `httpx` + `BeautifulSoup` (for most sites)
- Falls back to `newspaper3k` for structured article extraction (title, text, publish date)
- Detects and handles: paywalls (skip with reason), 404s (skip), redirects, cookie walls
- Extracts: full text, publish date, author (if available)
- Timeout: 15 seconds per request
- Returns raw text + metadata or a failure reason

### 2. LLM-Powered Fact Extraction
An LLM call (Gemini Flash) that takes the article text and extracts:
- **Event headline**: One-line summary of what happened (not the article's headline — the EVENT)
- **Event date**: The date the event OCCURRED, not the article publish date. If the article says "the president signed an executive order on Tuesday" and the article was published Wednesday Jan 15, the event date is Tuesday Jan 14.
- **Event date ambiguity flag**: True if the article doesn't give a clear date
- **Summary**: 100-200 word factual summary of the event only, no editorial commentary
- **Key facts**: Specific numbers, thresholds, legal citations, dollar amounts — the hard data
- **Direct quotes**: Verbatim quotes with speaker attribution and context
- **Topics**: Freeform tags relevant to the event

The LLM prompt must:
- Explicitly instruct: "Extract the date the EVENT HAPPENED, not the date the article was published. These are often different."
- Explicitly instruct: "If you cannot determine a specific date, set event_date_ambiguous to true. Do not guess."
- Explicitly instruct: "Strip all editorial opinion. Include only verifiable facts."
- Include a worked example showing article text → correct extraction

### 3. Content Quality Filter
Before accepting an extraction:
- Word count must be > 100 (skip stubs)
- Summary must not be empty
- Reject if the article is clearly an opinion piece / editorial (LLM can flag this)

### 4. LangGraph Node Function
A `extraction_node(state: SwarmState) -> SwarmState` function that:
- Takes raw candidates from `state.raw_candidates` (batch of N at a time)
- Fetches and extracts each one
- Adds successful extractions to `state.extraction_results`
- Logs failures to `state.errors`
- Returns updated state

## Quality bar
- Must correctly distinguish event date from publish date (this is critical for temporal integrity)
- Must gracefully handle broken URLs, paywalls, and stubs
- Extraction success rate should be > 60% of attempted URLs
