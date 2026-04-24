# Pre-Re-Run Audit — Things That Are Wrong Before We Spend Another Dollar

Purpose: audit everything that could invalidate the experiment BEFORE re-running. Each item lists severity, evidence, and a proposed fix. Do not re-run until you sign off on the fix plan at the bottom.

---

## P0 — Temporal leakage (threatens validity of the whole study)

### P0.1 — Model training cutoff overlaps the simulation window

- **Ran with:** `gemini-2.5-flash` (config.yaml line 11). Official knowledge cutoff: **January 2025** (Google DeepMind model card, June 2025).
- **Simulation dates in manifest:** 26 distinct dates spanning **2025-01-02 → 2025-05-05**.
- **Direct answer leakage (event inside training window):** **20 / 104 questions** — the resolving event occurred on or before 2025-01-31, so Gemini can see the outcome in pretraining. Examples: `Q-S-016-01 sim=2025-01-15 event=2025-01-20`, `Q-S-024-01 sim=2025-01-15 event=2025-01-19`, etc.
- **Temporal-consistency leakage (sim inside window, event outside):** **31 / 104 questions** — sim_date is before training cutoff, so the model's pretraining "knows future" relative to its simulated present, even if the specific resolving event is later.
- **Clean (sim and event both post-cutoff):** only **42 / 104** under Gemini 2.5 Flash.

**Implication:** Current pre-registered model (Opus 4.7, Jan 2026 cutoff) is catastrophic here — near everything would leak. Gemini 2.5 Flash is the *better* choice of the two, but still leaks ~20% of the set outright and a further ~30% is borderline.

**Fix options (pick one):**
1. **Switch model to one with cutoff ≤ 2024-12-31.** Candidates: `gemini-2.0-flash` (cutoff ~Aug 2024), `claude-3-5-haiku` (July 2024), `claude-3-5-sonnet-20241022` (April 2024). Under a 2024-12-31 cutoff, **93 of 104 questions are clean** (10 undetermined, 0 leaking).
2. **Keep Gemini 2.5 Flash but restrict analysis set to the 42 post-cutoff questions.** Ugly — N is small, stratified sub-tables get thin.
3. **Split the bench into two cohorts** ("within-cutoff-subset, leak-expected" vs "post-cutoff-subset, clean"). The leak cohort becomes a sanity check (model should near-ace it); the clean cohort is where we draw conclusions.

My recommendation: **Option 1 + Option 3 combined** — run a model with December-2024-or-earlier cutoff, then report results on the full N=93 clean set. Compute a "leak canary" secondary on the 10 questions where our audit couldn't extract a date.

---

## P0.2 — Tavily web search silently bypasses the date filter

- File: `evaluation_plan/src/web_search_tool.py:106-110`. We call Tavily with `end_date=simulation_date`, but in post-processing we only drop items whose `published_date` parses AND is > sim_date. **Items with missing / unparseable published_date pass through unfiltered.**
- E5 uses this for its entire retrieval context. If Tavily returns any post-sim content without a date (common — news homepages, opinion pieces, updated articles), the model sees the future.
- Config also sets `enforce_temporal_filter: true` (config.yaml:44) — the code doesn't honor that.

**Fix:** Strict policy — drop any result with missing/unparseable `published_date` when `enforce_temporal_filter: true`. Also capture the URL and published_date of every kept item into the record so we can audit ex post.

---

## P1 — Experiment infrastructure bugs (generate spurious attrition)

### P1.1 — E5 attrition is all infra failure, not "condition characteristic"

Breakdown of 123 E5 error records:
- **72** `503 DNS resolution failed` / `No route to host` — cascading grpc/C-ares failures during the long E5 run. Machine's network went south mid-run; retry gave up after 10 min.
- **25** `No JSON object found in response` — Gemini returned prose on long-context structured-output. Current retry only fires once with a "STRICT JSON" suffix.
- **24** `504 DeadlineExceeded` — Gemini generation timeout on long contexts. Current `timeout=90, max_retries=2` → ~3 min cap, not enough.
- **2** pydantic `probabilities must sum to ~1.0, got 0.0000` — Gemini returned all-zeros JSON. **No retry path for this.** llm_client.py `_parse` returns an invalid object, which raises ValidationError, which isn't caught for retry.

### P1.2 — E4 has 1 identical failure: `Q-S-052-02 sample=4` also prose-instead-of-JSON.

**Fixes for P1.1 / P1.2 (before any re-run):**
1. **Retry on pydantic ValidationError** the same way we retry on `extract_json_object` failure. Catch `Exception` around `_parse(...)` (already does — but ValidationError is inside `_parse`, so the except path is hit; the issue is the retry suffix doesn't push the model off the all-zeros answer). Add explicit anti-degenerate instruction to the retry: *"Probabilities must be strictly positive and sum to 1. Do not return all zeros."*
2. **Tighten timeout+retries for Gemini.** Keep `timeout=90` but increase `max_retries=5` with exponential backoff, AND add a total wall-clock cap (e.g. ≤ 240s per sample). Add local `tenacity` retry around `chat.invoke` for transient `ResourceExhausted` / `DeadlineExceeded` / `ServiceUnavailable`.
3. **Truncate E5 Tavily context at a safe token budget.** Currently we format 20 results × up to 500 chars + metadata — feasible but combined with the question prompt pushes Gemini past its comfortable long-context JSON regime. Cap to 10 results × 300 chars, OR include raw_content opt-in only when the search returns ≤ 5 items.
4. **Detect network-down early.** Add a one-shot DNS/TLS preflight before each experiment starts; bail with clear error if grpc endpoint is unreachable, instead of burning 600s per call.
5. **Delete all error records and re-run only those** — the current predictions files will have residual error rows after fixes.

---

## P1.3 — Temperature 1.0 defeats sample-level variance interpretation

- config.yaml: `model.temperature: 1.0`. At T=1 the model is highly stochastic. With N=5 samples this is a reasonable draw from the posterior, but per-question sample stdev numbers in the summary (e.g. E5 mean stdev 0.1273) reflect mostly model noise, not a signal we care about.
- Pre-reg likely expected T=0 (deterministic) with N=1, or T=1 with N≥10 for proper Monte Carlo averaging.

**Fix:** Either
- (a) Drop to T=0, N=1 per question, rerun N_rep=3 for a tight variance estimate (cheap, clean), or
- (b) Keep T=1 and bump N to 10 as pre-registered (doubles cost, matches original plan).

Cost estimate before committing: re-quote to you before running.

---

## P1.4 — N=5 samples/question instead of pre-registered N=10

- config.yaml:27 explicitly states: "Reduced from 10 → 5 to stay within Gemini budget."
- This is a deviation from pre-registration. Either restore N=10 or amend the pre-reg doc to document the change (with $ cost rationale) before writing results.

---

## P2 — Pre-registration / paper mismatches (reporting issues, not data issues)

### P2.1 — Pre-reg model is Claude Opus 4.7, we ran Gemini 2.5 Flash

- paper.md Section 5 says Opus 4.7. Code says Gemini 2.5 Flash. Either amend the paper to document the deviation (with justification: cost, and now — given P0.1 — temporal-validity), or flip to a different model before re-running. Opus 4.7 is *worse* for this study (Jan 2026 cutoff → everything leaks).

### P2.2 — Paper condition labels (E1, E1′, E2, E3, E4) don't match code ids (e1, e1p, e2, e3, e4, e5)

- Paper: 5 conditions registered. Code: 6 (e2 = refined is post-reg).
- Mapping: code e1→paper E1, e1p→E1′, e2→E2′ (post-hoc), e3→E2, e4→E3, e5→E4.
- Fix: pick one naming scheme and propagate through paper + code (or add a glossary in the paper and leave code alone — lower risk).

### P2.3 — I previously wrote text in Section 8 framing E5's 23.9% attrition as "a genuine characteristic of the condition." That framing is wrong and will be removed — attrition is an infra failure.

---

## P3 — Lower-severity issues to resolve before final run

### P3.1 — CHRONOS SQL temporal bounds look correct but worth sanity-checking

- `temporal_knowledge_base/src/database.py:161-162` — `WHERE event_date <= simulation_date AND event_date >= model_training_cutoff`. Inclusive on both sides.
- The *lower bound* is the model's training cutoff, meaning CHRONOS is explicitly designed to "fill in the post-training gap." So if we change the primary model (P0.1 Option 1), we must update `model_training_cutoff` in CHRONOS config accordingly, or the retrieval window will be mis-sized.

### P3.2 — Resolution date vs simulation date

- One resolution (`Q-S-045-01`) has simulation_date `2025-02-15` and an event mention of `January 9, 2025` — means the event had already occurred at simulation time. That's a question resolvable from public news at sim date; shouldn't be in the eval, or it's a floor-test sanity question. Worth spot-checking.

### P3.3 — `no_ev_date` set (10 questions) — no date extractable from resolution evidence

- These can't be automatically classified as pre/post cutoff. Manually spot-check all 10 before including them in the clean cohort.

### P3.4 — Single-model family

- Only Gemini 2.5 Flash was run. Any result is conditional on that single model's behavior. For a stronger paper: run the primary condition (e1, CHRONOS broad-15) on at least one additional model (different family: Claude or GPT with a compatible cutoff) as a robustness check.

### P3.5 — Deterministic seeds

- `predict_*` has no explicit seed. Gemini doesn't expose a seed parameter. With T=0 this is moot. With T=1, replicability across re-runs is impossible. Document this in the paper; don't claim bitwise reproducibility.

---

## Proposed fix plan (awaiting sign-off)

**Phase A — stop & fix (no spend):**
1. Add pydantic-ValidationError retry + anti-degenerate retry instruction to llm_client.py.
2. Add tenacity-based retry for transient Gemini 504/503/DNS.
3. Tighten Tavily temporal filter (drop missing-date items).
4. Reduce Tavily context to 10 × 300 chars.
5. Add DNS/TLS preflight at experiment start.
6. Pick the primary model. My recommendation: **`gemini-2.0-flash`** (cutoff ~Aug 2024, 93 clean questions) — or confirm another model.
7. Pick N and T. My recommendation: **N=10, T=1** (matches original pre-reg; best statistical power).

**Phase B — re-quote cost to you, wait for approval.**

**Phase C — once approved:**
1. Delete all current `pipeline/output/predictions/e{1,1p,2,3,4,5}/predictions.jsonl` (the snapshotted copy is preserved in `evaluation_plan/output/snapshots/run_20260423T154710Z/`).
2. Rebuild briefings with the new model's training cutoff (CHRONOS lower bound).
3. Run all 6 conditions.
4. Verify 0 error records before analysis.
5. Re-run analysis + significance, rewrite Section 8.

Don't touch anything in Phase C until Phase B is approved.
