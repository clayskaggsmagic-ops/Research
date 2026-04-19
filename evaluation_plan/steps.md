# Evaluation Experiments — Step-by-Step Execution Plan

Companion to [experiment_design.md](experiment_design.md). Break the work into discrete, resumable chunks so each session stays well within a single context window. Feed one step at a time.

## Ground rules (apply to every step)

- **Branch:** do all work on `eval-experiments`. Do not touch `main` or the `chronos-research-swarm` branch.
- **Don't touch CHRONOS source.** We only *consume* [temporal_knowledge_base/src/retrieval.py](../temporal_knowledge_base/src/retrieval.py) via its `retrieve()` API. No edits to agents, DB, or ingestion.
- **Model pinning:** one frontier model ID logged per call; written into every prediction record. Pick it in Step 0 and don't change it mid-experiment.
- **Temporal leakage:** every CHRONOS retrieval must pass the question's `simulation_date` as the upper bound. Never use today's date.
- **Output locations:** predictions → [pipeline/output/predictions/](../pipeline/output/predictions/); scores → [pipeline/output/scores/](../pipeline/output/scores/); ground truth → [pipeline/output/resolutions/](../pipeline/output/resolutions/). These dirs exist and are empty — safe to write into.
- **Question set:** [evaluation_plan/output/final_manifest.json](output/final_manifest.json) — 104 questions, treat as frozen. Do not regenerate.
- **Sample count:** 10 runs per (question, experiment) at T=1 unless a step says otherwise.
- **Each step produces one artifact** and updates a `STATUS.md` checkbox so the next session can see what's done.

---

## Step 0 — Scaffolding

**Goal:** stand up the skeleton so later steps have somewhere to land.

**Do:**
- Create branch `eval-experiments`.
- Create dirs: `pipeline/output/predictions/{e1,e2,e3,e4,e5}/`, `pipeline/output/resolutions/`, `pipeline/output/scores/`, `evaluation_plan/prompts/`, `evaluation_plan/src/`.
- Write prompt files in `evaluation_plan/prompts/`:
  - `trump_system.md` (persona prompt for E1/E2/E3)
  - `analyst_system.md` (neutral persona for E4/E5)
  - `user_binary.md` (asks for probability + reasoning)
  - `user_action.md` (asks for probability distribution over options + reasoning)
- Decide + document the pinned model ID in `evaluation_plan/config.yaml` (model, temperature, sample_count, token budgets).
- Define the prediction-record JSON schema in `evaluation_plan/src/schemas.py` (question_id, experiment, sample_idx, model_id, prompt_hash, briefing_hash, probability or probability_dist, reasoning, tokens, timestamp).
- Create `evaluation_plan/STATUS.md` with an unchecked row per step.

**Output:** empty dirs, prompt files, schema module, STATUS.md. No runs yet.

---

## Step 1 — Ground-truth resolution

**Goal:** resolve every question in the manifest. Without this, nothing can be scored. **This is the longest pole — budget 1–3 days.**

**Context (read before starting):**
- A resolver already exists: [pipeline/src/stages/post_resolution.py](../pipeline/src/stages/post_resolution.py). It's an adversarial web-search agent that produces `correct_answer`, `resolution_evidence`, `resolution_derivation`, `resolution_weaknesses`, `search_queries_used`, and a `resolution_status` (`resolved_yes` | `resolved_no` | `resolved_option` | `ambiguous` | `annulled`). **Don't rebuild from scratch — run what's there.**
- Current state of [evaluation_plan/output/final_manifest.json](output/final_manifest.json): **104 questions, all `correct_answer: null`, all `resolution_status: pending`.** Never been resolved. (Note: user estimated ~115; actual is 104.)
- Existing resolver default: `gemini-2.5-flash` via [pipeline/run_all.py:178](../pipeline/run_all.py#L178) with `google_search` tools. Entry point: `run_ground_truth_resolver(state)` at [pipeline/src/stages/post_resolution.py:188](../pipeline/src/stages/post_resolution.py#L188).

**Do:**

**1.1 Smoke-test the existing resolver.** Before running on 104, confirm end-to-end on ~3 questions:
  - Load `final_manifest.json` into `PipelineState`.
  - Call `resolve_question(q, model_name="gemini-2.5-flash")` on three questions with different formats / resolution dates.
  - Confirm the output populates all five resolution fields and a non-pending `resolution_status`.

**1.2 Pass A — resolve all 104.**
  - Run `run_ground_truth_resolver` (or the batched loop in [pipeline/src/orchestrator.py:303](../pipeline/src/orchestrator.py#L303)) on the full manifest with `gemini-2.5-flash` + Google Search.
  - Persist to `pipeline/output/resolutions/pass_a.json` (don't clobber the manifest yet).
  - Log: `N resolved_yes / resolved_no / resolved_option / ambiguous / annulled / pending_after_failure`.

**1.3 Pass B — independent second pass.** Use a **different provider** so the failure modes are uncorrelated:
  - Port the resolver to call **Claude Opus 4.7** with the Anthropic `web_search_20250305` tool (same adversarial prompt).
  - Run on all 104. Write `pipeline/output/resolutions/pass_b.json`.
  - If Anthropic web search is unavailable in this environment, fall back to `gemini-3.1-pro-preview` at `temperature=0.0` — stronger Gemini model, lower temp — and document the fallback in STATUS.md.

**1.4 Compare + flag disagreements.**
  - Build `evaluation_plan/src/reconcile.py`. Read both pass files. For each question: `agreement = (pass_a.correct_answer == pass_b.correct_answer)`.
  - Emit `pipeline/output/resolutions/disagreements.md` — one entry per mismatch with both derivations side-by-side, for manual adjudication.
  - Expected disagreement rate: 10–20% (per user's estimate). If it's >30%, something is wrong with the resolver — stop and investigate before adjudicating.

**1.5 Manual adjudication.** User (not Claude) walks the disagreements.md file and picks a winner per row. Claude writes the chosen verdict + rationale into the final record. Items still ambiguous after adjudication stay as `ambiguous` and get excluded from scoring.

**1.6 Produce the final resolutions file.**
  - Merge into `pipeline/output/resolutions/resolutions.json` matching the [`QuestionResolution` schema](src/schemas.py). Each record includes both passes, the agreement flag, the `manual_review_required` bit, and the adjudicated outcome.
  - Report final counts: `N resolved / N ambiguous / N annulled / N excluded`.

**1.7 Lock the manifest.**
  - Write the adjudicated `correct_answer` + `resolution_status` back into `final_manifest.json`.
  - Commit. Tag: `git tag questions-v1.0-locked`.
  - After this tag, **no edits** to the manifest without incrementing the version. Downstream experiments (E1–E5) reference this tag.

**Output:** `resolutions.json` + locked, tagged manifest. Report counts and any excluded questions to user before proceeding to Step 2.

**Cost note:** this is the single biggest API spend of the project. Pass A on `gemini-2.5-flash` is cheap; Pass B on Opus is the expensive part. Confirm with user before kicking off Pass B.

---

## Step 2 — Scoring module

**Goal:** build the scorer once, use it for every experiment.

**Do:**
- `evaluation_plan/src/score.py`. Functions:
  - `brier_binary(p, y)`, `brier_multiclass(p_vec, y_idx)`
  - `log_loss(...)`, `ece(preds, bins=10)`, `murphy_decomposition(...)`
  - `top1_accuracy(...)`, `top2_accuracy(...)` for action_selection
  - `temperature_scale(predictions)` — fits one T on a held-out split, applies globally
  - `score_experiment(predictions_path, resolutions_path)` → writes `pipeline/output/scores/<experiment>.json` and appends to `pipeline/output/scores/per_question.csv`
- Unit test on 3–5 hand-built fake predictions. Don't run on real data yet.

**Output:** scoring module + tests. No experiment runs.

---

## Step 3 — E5: Analyst + web search (answerability gate)

**Goal:** sanity-check the question bank. If the analyst-with-web-search can't hit ~95%, the questions are suspect.

**Do:**
- `evaluation_plan/src/run_experiment.py --experiment e5`.
- System prompt: `analyst_system.md`. Tool: web search. **No CHRONOS.**
- For each question, 10 samples.
- **Important:** web search must be constrained to results dated ≤ `simulation_date`. This is the hardest implementation detail of this step — pick an approach (search query filters, post-hoc URL date filtering, or both) and document it.
- Write JSONL to `pipeline/output/predictions/e5/`.
- Run scorer. Report Brier + Top-1 per category.

**Output:** E5 predictions + E5 score. **Gate:** if E5 Brier > 0.10 overall, pause and investigate before continuing.

---

## Step 4 — E3: Trump persona, no context

**Goal:** cheapest real experiment. Establishes what the persona alone knows.

**Do:**
- `run_experiment.py --experiment e3`.
- System: `trump_system.md`. No tools, no briefing, no web search.
- 10 samples per question.
- Write predictions, run scorer.

**Output:** E3 predictions + E3 score.

---

## Step 5 — E1: Trump persona + CHRONOS (vanilla retrieval)

**Goal:** the intended system.

**Do:**
- Wire up `temporal_knowledge_base.src.retrieval.retrieve()`. For each question: `query = question_text`, `simulation_date = question.simulation_date`, `model_name = <pinned model>`, default `top_k`.
- Cache briefings by `(question_id, model_id)` — we'll reuse them in Step 7 for E4.
- 10 samples per question using the cached briefing.
- Write predictions, run scorer.

**Output:** E1 predictions + E1 score + cached briefings.

---

## Step 6 — E2: Trump persona + CHRONOS (refined retrieval)

**Goal:** test whether curation helps or hurts.

**Do:**
- Implement two-stage retrieval in `evaluation_plan/src/refined_retrieval.py`:
  1. LLM call: from `(question_text + background + resolution_criteria)` produce `{actors, topic_tags, date_subwindow, paraphrases[]}`.
  2. Over-retrieve top-50 candidates using paraphrases + topic filter.
  3. LLM re-ranker labels each event `supports-YES | supports-NO | background | irrelevant`.
  4. Keep ~8–12; group by label in the briefing.
- **Token-match to E1.** Also run an `E1′` variant (E1 with `top_k=8`) so we can separate *curation* from *compression*.
- 10 samples. Write, score.

**Output:** E2 predictions + E1′ predictions + both scores.

---

## Step 7 — E4: Analyst persona + CHRONOS

**Goal:** isolate the persona's contribution. Reuse E1 briefings verbatim, swap only the system prompt.

**Do:**
- Load cached briefings from Step 5. Run with `analyst_system.md`.
- 10 samples. Write, score.

**Output:** E4 predictions + E4 score.

---

## Step 8 — Temperature scaling + cross-experiment analysis

**Goal:** produce the headline result.

**Do:**
- Fit temperature scaling on a random 30% split per experiment; evaluate on the other 70%. Report raw + calibrated Brier side by side.
- `evaluation_plan/src/analyze.py` produces `pipeline/output/scores/summary.md`:
  - Headline table: experiment × (Brier, log-loss, ECE, Top-1) with baselines (coin-flip, `base_rate_estimate`, uniform).
  - Per-category breakdown (trade_tariffs, executive_orders, etc.).
  - Calibration plots (reliability diagrams) per experiment.
  - Pairwise Brier deltas with paired Wilcoxon signed-rank p-values: E1 vs E3 (CHRONOS value), E1 vs E4 (persona value), E1 vs E2 (refinement value), E1 vs E1′ (compression control).
- Read-out: one paragraph of findings at the top of `summary.md`.

**Output:** `summary.md` + reliability plots.

---

## Step 9 — Write-up pass (optional)

Turn `summary.md` into a results section in the main design doc or a standalone report. Only after Step 8 has been reviewed with the user.

---

## Dependency graph

```
0 → 1 → 2 → 3 (gate) → 4 ─┐
                5 ────────┼→ 8 → 9
                6 ────────┤
                7 ────────┘   (7 depends on 5's cached briefings)
```

## Sequencing tips for future-me

- **Always read [STATUS.md](STATUS.md) first.** It tells you what's done without re-reading outputs.
- **Don't re-run completed steps.** Predictions are expensive — only re-run if prompts, model, or retrieval logic changed.
- **If a step feels too big, stop and split it.** Better to hand back mid-way than blow the context window.
- **Keep predictions append-only.** If re-running, write to a new subdir (e.g., `e1_v2/`), never overwrite.
