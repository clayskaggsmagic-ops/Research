# Build Step 10: End-to-End Verification

## What this builds
A verification script and test suite that proves the entire system works: insertion, temporal filtering, multi-model retrieval, and quarantine.

## Context
This is the final validation step. We insert known test events and verify that the temporal sliding window works correctly — that events after the simulation date are NEVER returned, and that model training cutoffs correctly adjust the retrieval window.

## What this step must do

### 1. Create Test Script
`src/tests/test_retrieval.py` — A script that:

#### a. Insert Sample Events
Insert 10 test events with known dates spread across the timeline:
- 3 events in Oct-Dec 2023 (before GPT-4o cutoff? No, after — in the new knowledge window)
- 3 events in Jan-Jun 2024
- 2 events in Jul-Dec 2024
- 2 events in Jan-Mar 2025

Each event should be a realistic geopolitical event with proper fields filled.

#### b. Test Temporal Sliding Window
- **Test 1**: Query with `simulation_date = 2024-06-15`, `model = "gpt-4o"` → should return ONLY events between Oct 2023 and Jun 15 2024
- **Test 2**: Query with `simulation_date = 2024-06-15`, `model = "gemini-3-pro"` → should return ONLY events between Jun 2025 (cutoff) and Jun 15 2024. Wait — that's an EMPTY window (cutoff is AFTER simulation date). The system should handle this gracefully and return 0 events.
- **Test 3**: Query with `simulation_date = 2025-03-01`, `model = "gpt-4o"` → should return ALL 10 events (all fall between Oct 2023 and Mar 2025)
- **Test 4**: Query with `simulation_date = 2024-01-01`, `model = "gpt-4o"` → should return only the 3 Oct-Dec 2023 events

#### c. Test Quarantine Exclusion
- Insert 2 events with `DateConfidence.UNCERTAIN`
- Verify that NO retrieval query ever returns them, regardless of date window

#### d. Test Intelligence Briefing Output
- Run a retrieval and verify the output is formatted as a briefing (has header, events, footer)
- Verify it includes source citations and confidence notes

#### e. Test Multi-Model Switching
- Same query, same simulation_date, but switch between models
- Verify different models return different result sets based on their cutoffs

### 2. Run and Report
- Execute all tests
- Print pass/fail for each
- Print the actual briefing output for visual inspection

## Quality bar
- Test 1-4 must all pass — this is the core correctness guarantee
- Quarantined events must NEVER appear in results
- Empty window case must be handled without errors
- Briefing output must be human-readable and properly formatted
