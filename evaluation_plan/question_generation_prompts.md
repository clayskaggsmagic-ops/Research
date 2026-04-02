# Question Generation Pipeline — Build Prompts
## Prompts to Give Your Coding Agent to Build Each Pipeline Component

> **How to use this**: Each prompt below is something you paste into a conversation with a coding AI (Gemini, Claude, etc.) to get it to BUILD one piece of the automated question generation pipeline. Run them roughly in order — later prompts depend on code from earlier ones.
>
> **Architecture**: Adapted from Bosse et al. (2026) arXiv:2601.22444 — a 5-stage multi-agent pipeline for automated forecasting question generation. Their pipeline produced 1,499 questions at 96% quality. We adapt it for retroactive evaluation of Trump presidential decisions. All agent prompts cross-referenced against the paper's Appendices A.8–A.21.
>
> **Stack**: Python, LangGraph, Pydantic schemas, LLM tool-calling. Each stage is a LangGraph node.

---

## Prompt 1 — Project Scaffolding & Shared Schemas

```
Build the project scaffolding for a LangGraph-based question generation pipeline. 
This pipeline automates the creation of forecasting evaluation questions about 
Trump's presidential decisions, following the Bosse et al. (2026) 5-stage 
architecture: Seeds → Proto-Questions → Refinement → Verification → Deduplication.

Set up:
1. A Python project with uv/pyproject.toml
2. LangGraph StateGraph as the orchestrator
3. Pydantic schemas for the shared pipeline state

The key data types the pipeline passes around:

- DecisionSeed: A real decision Trump made. Fields: seed_id, event_description, 
  decision_taken, decision_date, simulation_date (1-30 days before decision_date — 
  the "fake today" we tell the prediction model), domain (trade_tariffs, 
  executive_orders, personnel, foreign_policy, legislative, public_comms, 
  legal_judicial), plausible_alternatives (list of strings), sources (list of 
  {name, url, date}), attribution_evidence, leader_attributable (bool).

- Question: A prediction question generated from a seed. Fields: question_id, 
  seed_id, question_type (binary or action_selection), title, background, 
  question_text, options (list, for MC only), resolution_criteria, 
  resolution_source, fine_print, resolution_date, base_rate_estimate, 
  simulation_date, domain, difficulty (easy/medium/hard), correct_answer (null 
  until ground truth resolution), verification_verdict (approved/revision_needed/
  rejected), verification_notes, background_research (structured research brief).

- PipelineState: The LangGraph state object. Contains: training_cutoff_date, 
  today_date, seeds (list of DecisionSeed), proto_questions, researched_questions,
  refined_questions, verified_questions, rejected_questions, final_manifest 
  (all lists of Question).

Create the project structure with separate modules for each pipeline stage. 
Include a config file for API keys (Federal Register, Congress.gov, Kalshi, 
Polymarket) and model settings (which LLM for each stage, temperatures, etc.).
```

---

## Prompt 2 — Stage 1: Seed Harvesting (Autonomous Discovery Agent)

```
Build Stage 1 of the pipeline: a ReAct-style web research agent that 
AUTONOMOUSLY discovers post-training-cutoff Trump presidential decisions 
and extracts them as structured DecisionSeed objects.

This follows Bosse et al. (2026), where Stage 1 uses "arbitrary text" (news, 
reports) as inspiration, and the agent itself discovers and structures the 
relevant events — NOT a set of hardcoded API fetchers.

The agent:
1. Receives the pipeline config (training_cutoff_date, today_date, leader name) 
   and the 7 decision domains as seed topics:
   - trade_tariffs, executive_orders, personnel, foreign_policy, legislative, 
     public_comms, legal_judicial

2. For each domain, uses web search tools to autonomously discover real Trump 
   decisions that occurred between training_cutoff_date and today_date. The 
   agent decides what to search for, what sources to trust, and how deep to dig. 
   It should look for executive orders, tariff actions, personnel firings/
   nominations, vetoes, sanctions, military actions, diplomatic moves, public 
   statements — whatever it finds.

3. For each discovered decision, the agent extracts a structured DecisionSeed:
   - What happened (event_description, decision_taken)
   - When (decision_date)
   - A simulation_date 1-30 days BEFORE the decision — the "fake today" the 
     prediction model will see later
   - What alternatives existed (plausible_alternatives — always include 
     "take no action")
   - Why it's personally attributable to Trump (attribution_evidence)
   - Source citations with URLs

4. After discovery, a MERGER step deduplicates (same event found via different 
   searches → merge into one seed), tags domains, and flags seeds with 
   uncertain leader attribution.

Give the agent these tools:
- web_search: General web search
- web_scrape: Read a specific URL's content

The agent is FULLY AUTONOMOUS — no hardcoded API fetchers, no pre-wired data 
sources. The agent discovers everything itself through web search, exactly as 
Bosse et al. (2026) intended with "arbitrary text" seeds.

Output: deduplicated list of DecisionSeed objects → stored in state.seeds.
```

---

## Prompt 3 — Stage 2: Proto-Question Generator (ReAct Web Agent)

```
Build the proto-question generator node (Stage 2). This corresponds to Stage 2 
in Bosse et al. (2026): "A ReAct-style web research agent proposes 1-7 
forecasting questions per seed." (Paper Appendix A.8)

The node:
1. Takes the deduplicated seed list from Stage 1
2. For each seed, uses an LLM with web search tools to:
   a. Research the political context around the simulation_date
   b. Find what experts and prediction markets were saying at the time
   c. Generate 1-3 candidate prediction questions per seed

The LLM should generate a MIX of question types:

BINARY: "Will Trump [specific action] by [date]?" — clear yes/no
ACTION SELECTION: Scenario + 4-5 mutually exclusive options from most 
aggressive to most passive, always including "no action"

Critical constraint: Questions must be written AS IF asked on simulation_date. 
The LLM must NOT reveal the actual outcome in the question text, background, 
or options. Leave correct_answer as null.

Quality criteria from the paper's prompt (A.8):
- Questions should have "high entropy" — non-trivial, plausibly YES or NO
- A good forecast should be between 5% and 95% probability
- Two good forecasters should reasonably differ by at least 20 percentage points
- Doing more research should lead to a better forecast (avoid pure base-rate 
  or coin-flip questions)
- Questions should be maximally independent (can't just bet on one confounder)

Give the LLM these tools:
- web_search: Search the web for context (constrained to info available on 
  simulation_date where possible)
- prediction_market_lookup: Check Kalshi/Polymarket for analogous questions 
  and prices near the simulation date

Output: list of Question objects (question_text, background, options filled; 
resolution fields left null) → stored in state.proto_questions.
```

---

## Prompt 4 — Stage 2.5: Background Research Agent

```
Build a dedicated background research agent. In Bosse et al. (2026), this is 
a SEPARATE ReAct agent (Appendix A.10) that gathers comprehensive research 
for each proto-question BEFORE refinement. This step is critical because the 
paper found that without thorough research, LLMs generate plausible-but-invalid 
questions (e.g., asking about events that already happened or sources that 
don't exist).

The agent:
1. Takes each proto-question from Stage 2
2. Uses web search to produce a structured research brief covering:

   BACKGROUND AND CONTEXT:
   - Status quo as of simulation_date
   - Historical context and recent developments
   - Key stakeholders and players involved
   - Regulatory/legal framework if applicable

   DATA AND INFORMATION:
   - What data sources are relevant? How to access them?
   - When has past data been published? Is there a publication schedule?
   - Where should we expect new information to become available?

   RECENT NUMBERS AND EVENTS:
   - For threshold questions ("will X exceed Y"): what is the current number?
   - What would the outcome have been in prior periods?
   - Are there relevant projections, forecasts, or base rates?

   TRENDS:
   - Factors that make a YES resolution more likely
   - Factors that make a NO resolution more likely

3. Stores the research brief in question.background_research
4. Flags any proto-questions where research reveals problems:
   - The event already happened before simulation_date
   - The resolution source doesn't actually exist
   - The question is based on hallucinated facts

Give the agent these tools:
- web_search: General web search
- web_scrape: Read specific URLs for detail

This agent should be THOROUGH — the paper emphasizes that "all information 
should be easily digestible" and must include links and sources. "This is 
important work, and any mistakes could be hugely embarrassing."

Output: proto_questions enriched with background_research → stored in 
state.researched_questions.
```

---

## Prompt 5 — Stage 3: Refinement Agent

```
Build the refinement agent node (Stage 3). In Bosse et al. (2026), this is a 
SEPARATE agent from the question writer — it adds precise, objective resolution 
criteria without changing the question substance. (Paper Appendix A.9)

The node:
1. Takes researched_questions from Stage 2.5 (with background research attached)
2. For each question, the LLM adds:
   - resolution_criteria: Exact YES/NO conditions (binary) or how to match 
     each option (MC). Must be unambiguous enough that a stranger can resolve 
     it without judgment calls.
   - resolution_source: Specific database/site (Federal Register, USTR, OFAC 
     SDN List, Congressional Record, archived Truth Social, etc.)
   - fine_print: Edge cases — partial action, reversed action, done via 
     subordinate, resolution source unavailable
   - base_rate_estimate: Historical frequency with reasoning
   - resolution_date: After simulation_date, on/after actual decision_date

Per the paper's refinement prompt (A.9), key rules:
- ALL relevant terms must be clearly defined (with links to authoritative 
  sources like Wikipedia, official organizations, etc.)
- When December 31 comes, there ABSOLUTELY needs to be a resolution source 
  that unambiguously tells us the correct outcome
- Use "pars pro toto" — find specific measurable indicators as proxies for 
  broader concepts (e.g., "10+ missiles fired" instead of "war")
- Always include timezones on dates, always set start AND end dates
- The question should be resolvable by a human within ~10 minutes
- Avoid "will condition X happen before condition Y" — use fixed dates instead

Give the LLM web_search to verify resolution sources exist and compute base 
rates from historical data.

Output: list of Question objects with resolution fields populated → stored in 
state.refined_questions.

Ideally this runs on the SAME model as Stages 2-2.5 — the cross-model 
separation comes in Stage 4 (verification).
```

---

## Prompt 6 — Stage 4: Adversarial Verification (Cross-Model, Separate Agents)

```
Build the verification stage (Stage 4). This is the key quality gate. In Bosse 
et al. (2026), verification uses FOUR SEPARATE, INDEPENDENT ReAct agents — 
each specialized in one dimension — not a single agent doing all checks. They 
achieved 96% unambiguous rate by using a DIFFERENT model family for 
verification vs. drafting. (Paper Appendices A.12–A.15)

We extend the paper's 4 agents to 6 (adding leader-attribution and 
information-leakage checks specific to our retroactive evaluation design).

Build 6 INDEPENDENT verification agents. Each one:
- Receives the full list of refined_questions from Stage 3
- Evaluates ONLY its own dimension
- Returns a per-question verdict with structured reasoning
- Runs on a DIFFERENT LLM model family than Stages 2-3

AGENT 1 — QUALITY & MEANINGFULNESS (paper A.12)
"Is this a good forecasting question?"
- Is it somewhat difficult — does more research lead to better forecasts?
  (Bad: coin flip. OK: look up option prices. Good: interpret trends/polling.)
- Does it have high entropy — non-trivial, answer not almost certainly 
  true or false?
- Is there room for disagreement — two good forecasters could reasonably 
  differ by 20+ percentage points?
- Verdict: bad / meh / good / great (paper's exact 4-point scale)

AGENT 2 — AMBIGUITY (paper A.13)
"Can this be clearly and unambiguously resolved?"
- Are all key terms well-defined with links to authoritative sources?
- Is the resolution date unambiguous with timezone and year?
- Are numeric cutoffs explicitly defined (>= 50% vs > 50%)?
- Is it robust against unexpected technicalities or gotchas?
- "If 10 people look at the question and the source, will they all broadly 
  agree on the outcome?" Score 0-100.
- Verdict: bad / meh / good / great

AGENT 3 — AI-RESOLVABILITY (paper A.14 — NOT just "resolvability")
"Can an AI agent autonomously resolve this question?"
The paper specifically tests whether a future ReAct agent with web search 
can find the answer, not just whether it's theoretically resolvable.
- Will it be trivially possible to LOCATE the resolution source?
- Will the source exist with 99% probability at resolution time?
- If source is already specified: does it actually exist right now? Is it 
  freely accessible? Does the specific column/variable exist?
- Can you access the source and see what it currently says?
- Could a human resolve it within 10 minutes?
- Verdict: very certainly no / probably no / probably yes / very certainly yes

AGENT 4 — NON-TRIVIALITY VIA FORECASTING (paper A.15)
Instead of just checking base rates, this agent ACTUALLY MAKES A FORECAST 
on the question to assess triviality. This catches subtler forms of 
obvious-answer questions.
- Think about base rates for similar events
- Consider status quo bias (the world changes slowly)
- Consider seasonal effects, current trends, scope sensitivity
- Think about incentives and power of influential people involved
- Pre-mortem: how would you most likely be wrong?
- If the forecast is >95% or <5%, the question is trivially easy → flag
- If two reasonable forecasters couldn't differ by 20+ points → flag
- Verdict: probability estimate + bad/meh/good/great

AGENT 5 — LEADER ATTRIBUTION CHECKER (our addition)
Is this genuinely the President's personal decision?
- Auto-fail: Federal Reserve decisions, SCOTUS rulings, Congressional 
  vote outcomes, market movements, foreign government actions.
- The decision must be traceable to Trump's personal directive, not 
  a bureaucratic default or institutional process.

AGENT 6 — INFORMATION LEAKAGE DETECTOR (our addition)
Since we wrote these questions retroactively (knowing the outcome), 
does the wording accidentally reveal the answer?
- Check: Does the background narrative, option framing, or question 
  phrasing subtly telegraph what actually happened?
- This is unique to our retroactive evaluation design — Bosse et al. 
  didn't need this because they wrote questions prospectively.

Then build an AGGREGATOR node that:
1. Collects all 6 agents' verdicts per question
2. Computes a final verdict:
   - APPROVED: all 6 pass (or only minor warns)
   - REVISION_NEEDED: 1-2 fails with fixable issues (attach notes)
   - REJECTED: 3+ fails or any fatal fail
3. Routes:
   - APPROVED → state.verified_questions
   - REVISION_NEEDED → back to Stage 3 (refinement) with notes, then 
     re-verify. Max 2 revision loops per question.
   - REJECTED → state.rejected_questions (logged, not deleted)

Build the routing as a LangGraph conditional edge. The 6 agents should 
run IN PARALLEL for efficiency (they're independent).
```

---

## Prompt 7 — Stage 5: Deduplication & Batch Balancing

```
Build the deduplication and batch balancing node (Stage 5). Bosse et al. (2026) 
used LLM-based pairwise similarity scoring with mean intra-cluster similarity 
of 1.32/4.0. (Paper Appendix A.20 — uses Claude Haiku for dedup)

The node:
1. Takes state.verified_questions
2. Uses an LLM to score pairwise similarity (1-4 scale). Remove questions 
   scoring 3+ against any other — keep the better-written version.
3. Checks distribution against targets and flags imbalances:
   Domain: trade_tariffs ~25%, executive_orders ~20%, personnel ~15%, 
   foreign_policy ~15%, legislative ~10%, public_comms ~10%, legal_judicial ~5%
   Difficulty: easy ~20%, medium ~60%, hard ~20%
   Format: binary ~60%, action_selection ~40%

Output: state.final_manifest — the locked question set ready for pre-registration.

Also build the serialization: export the manifest as a versioned JSON file 
with metadata (pipeline version, model versions used at each stage, date 
generated, total questions, distribution stats).
```

---

## Prompt 8 — Post-Pipeline: Ground Truth Resolution & Difficulty Scoring

```
Build two post-pipeline nodes that run AFTER the question set is finalized 
and pre-registered, but BEFORE any predictions are collected.

NODE A: Ground Truth Resolver (based on paper Appendix A.21)
Since all events in our question set have ALREADY HAPPENED, this node 
determines the actual outcome for each question.

The resolver agent must:
- Use web search to find concrete evidence from the resolution sources 
  specified in each question
- Binary: sets correct_answer to YES or NO with citations
- MC: sets correct_answer to the matching option letter
- Marks AMBIGUOUS if evidence is contradictory, ANNULLED if question is invalid
- The answer key gets LOCKED after this step — no changes allowed

CRITICAL — adopt the paper's adversarial framing (A.21):
The resolution_derivation must be a "BULLET-PROOF argument that should 
convince even someone who just LOST A BUNCH OF MONEY betting on this 
question and tries to find loopholes to overturn its resolution." This means:
- Include ALL links and ALL search queries used, verbatim
- Make assumptions explicit and visible
- Add a resolution_weaknesses section: "imagine there was ONE subtle 
  mistake in your derivation — what would it be?"
- Be extremely literal in interpreting resolution criteria — they always 
  supersede "reasonable interpretations of the question"

NODE B: Difficulty Scorer
- Estimates difficulty using three methods:
  1. Historical base rate (how often Trump has done this before)
  2. Prediction market proxy (Kalshi/Polymarket prices near simulation_date)
  3. Editorial judgment (how constrained was the decision space)
- Assigns difficulty (easy/medium/hard) and time_horizon (short/medium/long)

These can run in parallel. Both need web search tools.
```

---

## Prompt 9 — Prediction Collection Engine

```
Build the prediction collection engine. This runs the finalized questions 
through the model under test to collect predictions.

The engine:
1. Takes the locked manifest (with ground truth hidden from the prediction model)
2. For each question, runs THREE prompts:

   A. TRUMP PERSONA (binary or MC): System prompt establishes Trump identity, 
      user message sets simulation_date as "today" and presents the question. 
      Model answers in first person with prediction + confidence + reasoning.

   B. GENERIC ANALYST BASELINE: Same question, no persona. Model acts as a 
      political analyst predicting Trump's behavior in third person.

   C. (Optional) HISTORICAL BASE RATE: Simple lookup of how often Trump has 
      done this type of thing — no LLM needed, just the base_rate_estimate.

3. Each prompt runs 10 TIMES per question (temperature > 0) to measure 
   variance and compute mean predictions.

4. The model under test must be VERSION-PINNED — record the exact model ID.

5. Output: For each question, store all 10 persona predictions, all 10 analyst 
   predictions, with confidence scores and reasoning.

Build this as a batch runner with progress tracking, retry logic, and output 
serialization to JSON.
```

---

## Prompt 10 — Scoring & Analysis

```
Build the scoring module. Takes predictions from Prompt 9 and ground truth 
from Prompt 8, computes all evaluation metrics.

Metrics to compute:

1. BRIER SCORE: (predicted_probability - outcome)² per question, then mean. 
   Compute separately for persona vs. analyst vs. base rate.

2. LOG LOSS: -[y*log(p) + (1-y)*log(1-p)] — punishes confident wrong answers.

3. CALIBRATION (ECE): Bin predictions by confidence decile, compare predicted 
   vs. actual frequency. Generate calibration curves.

4. ACCURACY: For binary, straightforward. For MC, top-1 and top-2 accuracy.

5. RESOLUTION: Variance of predicted probabilities — is the model actually 
   discriminating or hedging everything at ~50%?

Statistical tests:
- Paired t-test or Wilcoxon signed-rank: persona vs. analyst Brier scores
- Bootstrap confidence intervals (10,000 resamples) for all metrics
- Break down by domain, difficulty, time horizon

Output: Summary stats, calibration plots, head-to-head comparison tables, 
and a flag for whether the persona significantly outperforms the analyst baseline.

Also generate a human-readable evaluation report as markdown.
```

---

## Build Order

```
Prompt 1   → Project scaffolding, schemas, config
Prompt 2   → Seed harvesting (autonomous ReAct discovery agent)
Prompt 3   → Proto-question generator (ReAct + web search)         [A.8]
Prompt 4   → Background research agent (deep research per question) [A.10]
Prompt 5   → Refinement agent (resolution criteria)                 [A.9]
Prompt 6   → Adversarial verification (6 independent agents)        [A.12-A.15]
Prompt 7   → Deduplication & balancing                              [A.20]
Prompt 8   → Ground truth resolution + difficulty scoring           [A.21]
Prompt 9   → Prediction collection engine (persona + baseline)
Prompt 10  → Scoring & analysis module
```

Each prompt builds one layer. Test each layer independently before wiring 
them together in the full LangGraph StateGraph.

---

## Context to Include With Each Prompt

When you give any of these prompts to your coding agent, also provide:

1. **This file** — so it understands the full pipeline architecture
2. **`question_generation_framework.md`** — the 6-gate filter, domain weights, 
   and 4-agent architecture details
3. **`experiment_design.md`** — the 3 experiments, scoring metrics, baselines, 
   and question categories
4. **`hello.md`** — Mantic's specialized tools architecture (relevant for 
   prediction collection)
5. **The Bosse et al. paper** (arXiv:2601.22444) — or at minimum, our notes 
   on it in `domain4_benchmarks_superforecasting.md` Entry 11

---

## Appendix Cross-Reference

| Our Prompt | Paper Appendix | What It Does |
|---|---|---|
| Prompt 3 | A.8 | Proto-question generation from seeds |
| Prompt 4 | A.10 | Background research per question |
| Prompt 5 | A.9 | Refinement with resolution criteria |
| Prompt 6, Agent 1 | A.12 | Quality/meaningfulness check |
| Prompt 6, Agent 2 | A.13 | Ambiguity check |
| Prompt 6, Agent 3 | A.14 | AI-resolvability check |
| Prompt 6, Agent 4 | A.15 | Forecasting-based triviality check |
| Prompt 7 | A.20 | Deduplication |
| Prompt 8 | A.21 | Adversarial resolution |
| — | A.17 | Research for forecasting (used in Prompt 9) |
| — | A.18 | Probabilistic forecast (used in Prompt 9) |

## References

| Citation | What We Adapted |
|---|---|
| Bosse et al. (2026) arXiv:2601.22444 | 5-stage pipeline, cross-model verification, ReAct agents, all appendix prompts |
| ForecastBench (Zou et al., 2024) | Post-training-date methodology |
| Mantic (Karger et al., 2025) | Difficulty scoring, prediction market benchmarking |
| Katz et al. (2017) | Action-selection format |
| Payne et al. (2024) | Escalation bias mitigation |
