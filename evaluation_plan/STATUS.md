# Evaluation Experiments — Status

Track progress across the steps defined in [steps.md](steps.md). Update this file whenever a step is completed.

## Current branch: `eval-experiments`
## Pinned model: `claude-opus-4-7` (see [config.yaml](config.yaml))

---

## Steps

- [x] **Step 0 — Scaffolding**
  - [x] Branch `eval-experiments` created
  - [x] Output dirs (`pipeline/output/predictions/{e1..e5}`, `resolutions/`, `scores/`) exist
  - [x] Prompt files written (`trump_system.md`, `analyst_system.md`, `user_binary.md`, `user_action.md`)
  - [x] `config.yaml` written
  - [x] `src/schemas.py` written
  - [x] This file created

- [ ] **Step 1 — Ground-truth resolution** *(longest pole, 1–3 days)*
  - [x] 1.1 Smoke-test (Claude + WebSearch) on 3 questions → `pipeline/output/resolutions/smoke_test.json`
  - [x] 1.2 Pass A — **Claude Opus 4.7 + WebSearch** (via 10 parallel subagents + 3 smoke-test inline) on all 104 → [`pass_a.json`](../pipeline/output/resolutions/pass_a.json) (27 YES / 31 NO / 45 option / 1 ambiguous)
  - [x] ~~1.3 Pass B~~ — **skipped per user decision** (Pass A accepted as ground truth)
  - [x] ~~1.4 reconcile.py~~ — skipped (no Pass B to reconcile against)
  - [x] ~~1.5 User adjudicates disagreements~~ — skipped
  - [x] 1.6 [`resolutions.json`](../pipeline/output/resolutions/resolutions.json) assembled — 103 scorable (27 YES / 31 NO / 45 option), 1 excluded (Q-S-029-02 Hegseth conditional)
  - [ ] 1.7 Manifest update + git tag `questions-v1.0-locked` *(do after scoring module written)*

- [x] **Step 2 — Scoring module**
  - [x] [`src/score.py`](src/score.py) written — pure-stdlib: Brier (bin/multi), log-loss, ECE, Murphy decomposition, temperature scaling (grid search), top-k accuracy, baselines, `score_experiment` end-to-end, CLI
  - [x] [`src/test_score.py`](src/test_score.py) — 18/18 synthetic-data tests passing

**Plumbing for Steps 3–8 is complete but NOT EXECUTED** — CHRONOS (the temporal
knowledge base) is still under construction. E1/E1'/E2/E4 depend on it. E3/E5
can run independently but are gated on CHRONOS being finished first (user
directive: build pipelines now, execute later). Re-check this gate before
running any `python -m evaluation_plan.src.run_experiment ...`.

Shared plumbing:
- [x] [`src/io_utils.py`](src/io_utils.py) — config/manifest/prompts loaders, hashing, append-only JSONL
- [x] [`src/prompts.py`](src/prompts.py) — user-message renderer, option letters, prompt hashing
- [x] [`src/llm_client.py`](src/llm_client.py) — ChatAnthropic wrapper, JSON extraction, binary + action predict functions, error capture
- [x] [`src/briefings.py`](src/briefings.py) — `NoBriefing` / `ChronosBroad(top_k)` / `ChronosRefined()` with disk cache
- [x] [`src/refined_retrieval.py`](src/refined_retrieval.py) — two-stage: haiku expand + over-retrieve + rerank + group-by-label
- [x] [`src/web_search_tool.py`](src/web_search_tool.py) — Anthropic `web_search_20250305` spec + temporal-constraint preamble
- [x] [`src/run_experiment.py`](src/run_experiment.py) — unified CLI (`--experiment e1|e1p|e2|e3|e4|e5`)

- [ ] **Step 3 — E5 (Analyst + web search, answerability gate)** *(plumbing ready)*
  - [x] Runner spec wired; temporal constraint injected into system prompt
  - [ ] Predictions written to `pipeline/output/predictions/e5/`
  - [ ] Score logged; gate decision recorded below

- [ ] **Step 4 — E3 (Trump only, no context)** *(plumbing ready)*
  - [x] Runner spec wired
  - [ ] Predictions + score

- [ ] **Step 5 — E1 (Trump + CHRONOS broad)** *(plumbing ready, blocked on CHRONOS)*
  - [x] `ChronosBroad(top_k=15)` wired with disk cache under `_briefing_cache/chronos_broad_15/`
  - [ ] Briefings cached
  - [ ] Predictions + score

- [ ] **Step 6 — E2 (Trump + CHRONOS refined) + E1′ (top_k=8 control)** *(plumbing ready, blocked on CHRONOS)*
  - [x] Refined-retrieval module written (`refined_retrieval.py`)
  - [x] `ChronosBroad(top_k=8)` variant wired for E1′
  - [ ] Predictions + scores for both

- [ ] **Step 7 — E4 (Analyst + CHRONOS, reusing E1 briefings)** *(plumbing ready, blocked on CHRONOS)*
  - [x] E4 uses same `ChronosBroad(top_k=15)` provider → cache hits E1 briefings verbatim (same `briefing_hash`)
  - [ ] Predictions + score

- [x] **Step 8 — Analysis + summary** *(plumbing ready)*
  - [x] [`src/analyze.py`](src/analyze.py) — paired Wilcoxon (pure stdlib), reliability bins, contrasts, `summary.md` generator
  - [ ] `scores/summary.md` written (runs after experiments execute)

- [ ] **Step 9 — Write-up (optional)**

---

## Gate decisions

| Gate | Criterion | Status | Notes |
|------|-----------|--------|-------|
| Step 1 (post-1.4) | Disagreement rate ≤ 30% | pending | If >30%, resolver itself is broken — stop and fix before adjudicating. |
| Step 1 (post-1.6) | All 104 resolved or user-approved `ambiguous`/`annulled`/`excluded` | pending | Locked via `questions-v1.0-locked` tag. |
| Step 3 | E5 Brier ≤ 0.10 | pending | If it fails, pause and audit questions before proceeding. |

---

## Log

| Date | Step | Notes |
|------|------|-------|
| 2026-04-19 | 0 | Scaffolding complete. Pinned `claude-opus-4-7` as primary, `claude-haiku-4-5-20251001` as refiner. |
| 2026-04-19 | 1.1 | Smoke test on 3 questions validated flow. |
| 2026-04-19 | 1.2 | Pass A complete via 10 parallel Claude subagents. 104/104 resolved (27 YES / 31 NO / 45 option / 1 ambiguous). Q-S-029-02 flagged — Hegseth conditional, schema mismatch (should likely be `annulled`, not `ambiguous`). Pass A used Claude + WebSearch (not Gemini) — supersedes original plan. |
| 2026-04-19 | 2 | Scoring module complete: `src/score.py` + `src/test_score.py` (18/18 pass). Pure stdlib. |
| 2026-04-19 | 3–7 | Plumbing for all five experiments complete (unified runner). NOT EXECUTED — CHRONOS not ready. E3/E5 could execute standalone but held per user directive. Smoke-tested via `--dry-run`. |
| 2026-04-19 | 8 | `src/analyze.py` written: paired Wilcoxon signed-rank (pure stdlib), reliability bins, contrast table, summary.md generator. Wilcoxon smoke-tested on synthetic a<<b input (p≈0.002, W=0, z≈-3.1). |
