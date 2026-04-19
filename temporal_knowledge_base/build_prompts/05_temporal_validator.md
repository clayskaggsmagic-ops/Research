# Build Step 5: Temporal Validator Agent

## What this builds
`src/agents/temporal_validator.py` — The 4-layer date validation system. This is the MOST CRITICAL agent in the entire pipeline. A wrong date = data leakage = the entire experiment is invalidated.

## Context
The Temporal Validator is the "immune system" of the knowledge base. It takes cleaned EventRecords and subjects each one's date to 4 layers of validation. Records that pass all 4 layers get promoted to `validated_records`. Records that fail are quarantined — they NEVER enter the database.

## Existing code it depends on
- `src/models.py` → `SwarmState`, `EventRecord`
- `src/config.py` → `settings`, `DateConfidence`, `DatePrecision`
- Cleaning must be done first (it produces the `cleaned_records` list)

## What the output file must contain

### Layer 1: Parsing Validation
Purely mechanical checks on the date itself:
- Date is a valid calendar date (not Feb 30, not month 13)
- Date is within the collection window [collection_start, today]
- Date is NOT in the future
- Date precision is reasonable (day-level preferred, month-level accepted with flag)
- If article pub_date is available: event_date should be <= pub_date + 1 day (can't have event after it was reported, with 1-day grace for timezone differences)

### Layer 2: Cross-Source Verification
Compare the event date across multiple sources:
- If 3+ sources agree on the date → `DateConfidence.VERIFIED`
- If 2 sources agree → `DateConfidence.HIGH`
- If only 1 source → `DateConfidence.SINGLE_SOURCE`
- If sources DISAGREE on the date:
  - Disagreement within 1-2 days → use the majority date, flag as `DateConfidence.HIGH` with a note
  - Disagreement > 2 days → `DateConfidence.UNCERTAIN` → **QUARANTINE**
- Record the verification method in `date_verification_method` (e.g., "3_source_agreement", "2_source_consensus", "single_source_explicit_date")

### Layer 3: Logical Consistency
LLM-powered checks for temporal logic:
- Does the event make sense in its stated time period? (e.g., "Trump signed executive order on January 20, 2025" — was he in office on that date? Yes → pass)
- Does the event reference other events? If so, do the dates of referenced events come BEFORE this event? (e.g., "In response to the tariffs announced last week..." — check that tariff announcement exists and is dated before this event)
- Does the event's content match the political/policy context of its stated date? (rough sanity check)

### Layer 4: Statistical Outlier Detection
Aggregate-level checks:
- If an event's date makes it the only event on a date where 10+ events cluster a day later, flag for review
- If an event's date is suspiciously far from its article's pub_date (> 30 days), flag for review
- These don't auto-quarantine; they add a note to `date_verification_method`

### Quarantine System
Events that fail validation go to `state.quarantined_records`. They:
- Are stored with a `DateConfidence.UNCERTAIN` tag
- Include a `quarantine_reason` in the `date_verification_method` field
- Are EXCLUDED from all retrieval queries (enforced in `database.py`'s WHERE clause)
- Can be manually reviewed and promoted later

### LangGraph Node Function
A `temporal_validator_node(state: SwarmState) -> SwarmState` function that:
- Takes `state.cleaned_records`
- Runs each record through all 4 layers
- Passes → `state.validated_records` with updated `DateConfidence`
- Fails → `state.quarantined_records` with reason
- Returns updated state with counts

## Prompt engineering requirements
- Layer 3's LLM prompt must include: "You are a fact-checker specializing in temporal consistency. Your job is to verify that the stated date of an event is logically consistent with the event's content and context. If you have ANY doubt, flag it."
- Include specific examples of events that should pass and should fail

## Quality bar
- ZERO tolerance for incorrect date confidence labels
- Records with dates disagreeing by > 2 days across sources MUST be quarantined
- Layer 3 should catch obvious anachronisms (e.g., an event about a policy that didn't exist yet)
- All validation decisions must be logged with reasoning
