# Build Step 4: Cleaning Agent

## What this builds
`src/agents/cleaning.py` — The agent that deduplicates, normalizes, and bias-strips extracted events before they reach the Temporal Validator.

## Context
The Cleaning Agent sits between Extraction and Validation. Multiple articles often describe the same event. This agent merges them, strips editorial bias, and produces a single clean `EventRecord` per real-world event. It is the "editor" of the swarm.

## Existing code it depends on
- `src/models.py` → `SwarmState`, `ExtractionResult`, `EventRecord`, `Source`, `DirectQuote`
- `src/config.py` → `settings`, `DateConfidence`
- Extraction must be done first (it produces the `ExtractionResult` list)

## What the output file must contain

### 1. Event Deduplication
Events are duplicated when multiple articles cover the same real-world happening. The deduplication system must:
- **Cluster extractions** that describe the same event using LLM-powered semantic matching
  - Input: list of ExtractionResult objects
  - Output: clusters of results that describe the same event
  - Use headline + event_date proximity (within 2 days) + topic overlap as clustering features
- **Merge clusters** into single EventRecords:
  - Headline: use the most factual/specific headline from the cluster
  - Summary: synthesize the best summary from all sources (LLM-powered)
  - Key facts: union of all key facts, deduplicated
  - Quotes: union of all direct quotes, deduplicated by quote text
  - Sources: all unique sources from the cluster
  - Source count: total unique sources
  - Event date: majority vote if sources agree; flag as ambiguous if they disagree

### 2. Bias Stripping
An LLM pass that rewrites summaries to remove:
- Editorializing ("controversial", "divisive", "unprecedented")
- Attribution of intent ("in a power grab", "strategically")
- Emotional language ("shock", "outrage", "bombshell")
- Partisan framing ("critics say" / "supporters praise" without balance)

The output must be neutral, factual, Wikipedia-tone prose. The LLM prompt must include explicit before/after examples of biased → cleaned text.

### 3. Normalization
- Actor names normalized to consistent format: "Donald J. Trump", "Xi Jinping" (not "Trump", "the president", "Biden's successor")
- Topic tags normalized: lowercase, underscore-separated, no duplicates
- Dates validated as plausible (not in the future, not before the collection window)

### 4. Confidence Scoring
Set the `confidence` field (0.0-1.0) based on:
- Source count: 1 source = 0.4, 2 = 0.6, 3+ = 0.8 base
- Source quality: average tier score from Discovery's quality scorer
- Date agreement: +0.1 if all sources agree on date, -0.2 if they disagree
- Cap at 1.0

### 5. LangGraph Node Function
A `cleaning_node(state: SwarmState) -> SwarmState` function that:
- Takes `state.extraction_results`
- Clusters, merges, cleans, and normalizes
- Outputs `EventRecord` objects to `state.cleaned_records`
- Returns updated state

## Quality bar
- Must correctly merge 3 articles about the same tariff announcement into 1 EventRecord
- Bias-stripped summaries must read like Reuters/AP wire copy, not NYT opinion
- Confidence scores must be deterministic given the same inputs
