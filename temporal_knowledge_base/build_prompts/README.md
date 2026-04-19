# CHRONOS Build Prompts

Feed these to the AI one at a time, in order. Each prompt builds one component of the research swarm pipeline.

## Order

| # | File | What it builds | Depends on |
|---|------|----------------|------------|
| 1 | `01_coordinator.md` | Coordinator Agent — research planner & supervisor | scaffolding (done) |
| 2 | `02_discovery.md` | Discovery Agent — web search, source finding | Coordinator |
| 3 | `03_extraction.md` | Extraction Agent — article parsing, fact extraction | Discovery |
| 4 | `04_cleaning.md` | Cleaning Agent — dedup, normalization, bias stripping | Extraction |
| 5 | `05_temporal_validator.md` | Temporal Validator — 4-layer date validation | Cleaning |
| 6 | `06_indexing.md` | Indexing Agent — embedding + DB insertion | Validator |
| 7 | `07_coverage_auditor.md` | Coverage Auditor — gap detection, loop triggers | Indexing |
| 8 | `08_pipeline.md` | LangGraph Pipeline — wire all agents into StateGraph | All agents |
| 9 | `09_neon_setup.md` | Neon Database — create project, enable pgvector, migrate | Pipeline |
| 10 | `10_verification.md` | End-to-end test — insert samples, test retrieval | Everything |

## Usage

Just say: **"Do prompt 1"** (or "build step 1", etc.)
