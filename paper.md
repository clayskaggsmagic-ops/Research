# Predicting Political Leader Decisions with Persona-Conditioned Language Models and Temporally-Bounded Retrieval: The CHRONOS Framework

**Clay Skaggs¹**
¹Massachusetts Institute of Technology
Correspondence: clayskaggsmagic@gmail.com
Draft of April 23, 2026 (results included)

---

## Abstract

Can large language models (LLMs), when conditioned on the identity of a specific world leader and grounded in a temporally-bounded knowledge base, predict that leader's real-world decisions more accurately than neutral analytical baselines? Prior work has benchmarked LLMs as generic forecasters of public events (Halawi et al., 2024; Zou et al., 2024; Karger et al., 2025), while political science has documented both the surprising p just tell me where the markdown file paper is. redictability of political behavior from structured belief models (Bueno de Mesquita, 2009; Walker & Schafer, 2006; Goldstone et al., 2010) and the hard ceilings imposed by social complexity (Salganik et al., 2020; Tetlock, 2005). The question of whether an LLM conditioned to speak *as* a named leader can exploit that leader's documented decision-making patterns remains, to our knowledge, unanswered under rigorous retroactive evaluation.

We present CHRONOS, a framework that couples four contributions: (i) a SQL-enforced temporal knowledge base that prevents data leakage at the database level rather than through prompt instructions, with a single parameter supporting any LLM training cutoff from the same underlying corpus; (ii) an autonomous multi-agent research swarm (coordinator, discovery, extraction, cleaning, temporal validation, indexing, coverage-audit) that constructs the knowledge base without human intervention and iterates to fill coverage gaps; (iii) a four-layer date-validation protocol with a quarantine system that refuses to admit any event whose date cannot be corroborated, on the principle that a wrong date constitutes leakage; and (iv) a five-stage automated question generation pipeline adapted from Bosse et al. (2026) producing 115 pre-registered forecasting questions about the decisions of Donald J. Trump, 47th President of the United States, across seven decision domains.

Evaluation comprises two complementary retroactive protocols: binary probability forecasting scored with Brier score, and multiple-choice action selection in the style of Katz et al. (2017). Our principal experimental manipulation is a two-way contrast between a **persona** system prompt ("You are Donald Trump") and a leader-neutral **analyst** system prompt with identical retrieval context. This contrast tests whether identity-assertion framing itself contributes predictive signal — distinct from retrieval, from careful forecasting habits, and from the model's parametric knowledge of the leader.

We report results from a seven-condition evaluation on 103 resolvable questions × 5 samples per question × 7 conditions (3,605 predictions total) using Gemini 2.0 Flash-Lite as the evaluated model — chosen because its August 2024 training cutoff cleanly precedes every simulation date in the manifest, ruling out pretraining leakage of the outcomes. All seven conditions completed with **zero invalid records (515 / 515 each)** after iterative hardening of the retry and rate-limit logic, so every contrast is reported on the full n = 103 common-question set. **Across every pre-registered paired contrast — persona vs. matched-retrieval analyst, persona with briefing vs. persona without, broad vs. compressed retrieval, live web search vs. curated retrieval — no difference reached statistical significance at the conventional α = 0.05 threshold under paired sign tests with paired bootstrap confidence intervals, and all six bounded Brier means fell inside a 0.014-Brier-unit envelope (0.4723 to 0.4863).** The primary pre-registered contrast — persona vs. analyst with matched CHRONOS broad-15 retrieval — produced a mean Brier difference of −0.003 (persona nominally better by three-thousandths of a unit; 95% bootstrap CI [−0.034, +0.028]; two-sided sign test p = 0.76; 50 / 103 questions favoring the persona). A post-hoc seventh condition (E6) adding an **unbounded "answerability-gate" web search** — Tavily with no date filter, returning post-outcome coverage — produced a mean Brier of 0.4518 (best of the seven conditions), separating from the bounded web-search condition (E5) by −0.035 Brier units with 60 / 101 non-tied paired wins, bootstrap CI [−0.070, −0.0002]. The answerability gate is modestly better than every other condition but **does not approach ceiling accuracy** (p(correct) = 0.46, not ~1.0), which we interpret as evidence that the questions are genuinely hard — even with full hindsight — rather than as evidence that retrieval modality is irrelevant. The evaluation's strongest data-driven finding is a tight null across the six temporally-bounded conditions: neither persona framing, nor curated CHRONOS retrieval, nor retrieval refinement, nor retrieval compression, nor date-filtered web search produced a measurable improvement over any other bounded condition on forecasting accuracy. Adjacent to the accuracy null, we find that input-token cost varied by 17–18× across conditions — E1/E4 used ~12.8 M input tokens vs. ~0.67 M for E3 (no retrieval) — for essentially identical Brier, which we flag as the most actionable finding from the present run. We discuss the result honestly, including what it does and does not rule out at this sample size, and the constraints imposed by a deliberate mid-tier-capability choice of evaluated model.

---

## 1. Introduction

### 1.1 Motivation

Political leaders' decisions are among the most consequential events in human affairs, shaping trade, conflict, migration, markets, and institutional stability. They are also, notoriously, difficult to predict. Forty years of expert-judgment research have shown that trained human analysts perform only marginally better than chance at medium-horizon political forecasting (Tetlock, 2005), while the largest pre-registered mass-collaboration in social-science prediction — 160 teams with fifteen years of longitudinal family data — barely improved over a four-variable linear baseline (Salganik et al., 2020). Yet against these sobering results stand a smaller but important set of systems and studies in which structured, actor-centric models achieve striking accuracy: Bueno de Mesquita's expected-utility model has been reported at roughly 90% across two thousand political predictions (Bueno de Mesquita, 2009); the Political Instability Task Force forecasts state failure two years out at 80–90% accuracy from four macro-indicators (Goldstone et al., 2010); the Walker-Schafer Verbs-in-Context-System (VICS) operational-code profiles have been shown to predict the cooperative-or-conflictual valence of leaders' foreign-policy responses (Walker & Schafer, 2006); and Katz et al. (2017) achieved roughly 70% accuracy forecasting U.S. Supreme Court outcomes.

Two observations motivate our work. First, the systems that succeed tend to share a common structure: they represent the *actor* (a leader, a court) rather than the *outcome* (GDP, conflict onset), and they exploit features of that actor's documented history — stated beliefs, past actions, institutional role — to constrain the space of plausible decisions. Second, large language models pre-trained on web-scale text have already absorbed enormous quantities of material about individual public figures: speeches, interviews, court filings, press conferences, social-media posts, biographical writing, contemporaneous news analysis. A model that has seen, during pre-training, millions of tokens authored by or describing a specific political leader arguably already encodes a latent representation of that leader's rhetorical style, policy preferences, and decision heuristics. If such an encoding exists and is accessible through persona prompting, it should be testable.

The core research question of this paper is therefore: **can an LLM conditioned to speak as a named political leader, and provided with a rigorously temporally-bounded briefing of events visible to that leader at a simulation date, predict that leader's specific real-world decisions more accurately than a matched leader-neutral analyst baseline with the same briefing?**

A tighter, and for us more interesting, sub-question is whether the *identity-assertion framing itself* — a system prompt of the form "You are Donald Trump" — unlocks predictive signal that is absent or suppressed when the same model is prompted as a generic analytical LLM. If persona conditioning does contribute uniquely, it suggests that LLMs encode latent person-specific representations accessible through identity priming but not through generic role priming — a finding with implications beyond political forecasting, for any task where named-individual behavior matters. If persona conditioning contributes nothing above a matched analyst baseline, it suggests that any apparent persona-simulation effect is attributable to retrieval, not to identity priming — a finding equally of interest, because it constrains the claims that can responsibly be made about LLM persona simulation more broadly. We therefore treat the **persona-vs-analyst contrast with matched retrieval** as a first-class object of study, not a secondary control.

### 1.2 Why This Is Hard

Two methodological problems have prevented rigorous answers to this question. The first is data leakage. Modern LLMs are trained on corpora extending into 2024 and 2025 (Anthropic, 2025; OpenAI, 2025; Google DeepMind, 2025); any "prediction" of an event that occurred before the model's training cutoff is retrieval, not forecasting (Zou et al., 2024). A serious evaluation must use decisions that occurred *after* the model's training cutoff, and must also ensure that the model is not implicitly exposed to post-cutoff information through the very retrieval mechanism that provides it with situational awareness. Prior frameworks (Bosse et al., 2026; Zou et al., 2024) mitigate leakage through careful question wording and instruction-following ("pretend today is X"), but rely on the model to respect a temporal boundary that it has no architectural reason to respect.

The second problem is question supply. Rigorous forecasting evaluation requires large numbers of resolvable, leader-attributable, post-cutoff questions across multiple decision domains, with carefully pre-computed base rates and well-defined resolution criteria. Manual curation (Metaculus-style) is slow and expensive; simple automated generation produces questions that fail one or more of the resolvability, specificity, or attributability gates.

### 1.3 Contributions

We address both problems. This paper describes:

1. **CHRONOS**, a temporal knowledge-base architecture that enforces the simulation-date boundary *at the SQL level* through a hard WHERE clause on `event_date`. The model cannot receive post-cutoff documents even under adversarial prompting, because the documents never enter the retrieved context. A single training-cutoff parameter makes the same knowledge base serve LLMs with different training windows (GPT-4o, Claude 4, Gemini 3) without re-ingestion.

2. An **autonomous multi-agent research swarm** that constructs the knowledge base from a leader name and date range with no domain-specific configuration. Seven specialized agents (coordinator, discovery, extraction, cleaning, temporal validator, indexing, coverage auditor) operate in a supervisor-worker pattern with a feedback loop between the auditor and coordinator, iterating until coverage criteria are satisfied or a maximum number of rounds is reached.

3. A **four-layer date-validation protocol** — mechanical parsing, cross-source corroboration, LLM-based logical consistency, and statistical outlier detection — with a quarantine system whose operating principle is that it is better to lose an event than to admit it with a wrong date. Records with source-date disagreement exceeding two days are excluded from retrieval entirely.

4. A **five-stage automated question generation pipeline** adapted from Bosse et al. (2026), producing 115 pre-registered forecasting questions about the decisions of Donald J. Trump across seven decision domains (executive orders, tariffs and trade, personnel, foreign policy, legislative, legal/judicial, and public communications), each with a simulation date, a resolution criterion tied to a specific public document class (Federal Register, USTR publications, Congressional Record, etc.), a base-rate estimate, and an assigned difficulty tier.

5. A **two-experiment evaluation protocol** comparing persona-conditioned LLMs against a matched leader-neutral analyst baseline with identical retrieval context. The persona-vs-analyst contrast is the primary experimental manipulation. Evaluation consists of binary probability forecasting scored with Brier, log-loss, and temperature-scaled calibration, and multiple-choice action selection with top-1, top-2, and rank-correlation scoring. Five planned experimental conditions (E1–E4 plus an E1′ retrieval-volume control) dissect the contributions of persona framing, retrieval, and retrieval curation.

### 1.4 Scope

We restrict our initial evaluation to a single leader — Donald J. Trump, 47th President of the United States — during his current term. The choice is deliberate. Trump is arguably the most text-documented political figure of the LLM-training era, with millions of authored social-media posts, thousands of rally and press-conference transcripts, multiple book-length autobiographical and biographical treatments, and a continuously-updated stream of contemporaneous news analysis. If persona conditioning is going to work anywhere, it should work here. A negative result on Trump would be informative about the ceiling of the method in general; a positive result motivates extension to other leaders whose training-data density is lower. The framework, knowledge-base architecture, and question-generation pipeline are agnostic to the subject leader and are designed to generalize.

### 1.5 Paper Organization

Section 2 reviews the political-science, forecasting, and LLM-evaluation literatures on which this work rests. Section 3 formalizes the prediction task and articulates our evaluation desiderata. Section 4 describes the CHRONOS temporal knowledge base, the research swarm, and the date-validation protocol in detail. Section 5 describes our methodology: the question generation pipeline, the three experimental protocols, scoring, statistical testing, and pre-registration. Section 6 discusses implementation details and the production technology stack. Section 7 specifies the experimental setup for the evaluation. Results are deferred to a companion paper.

---

## 2. Background and Related Work

### 2.1 Political Decision Theory and the Predictability Question

The tension between predictability and contingency runs through twentieth-century international-relations scholarship. Bueno de Mesquita's expected-utility framework (1985, 2009) models political outcomes as equilibria of preference-weighted games among actors; applied commercially and academically over several decades, it has been reported to achieve roughly 90% accuracy across more than 2,000 predictions, including a widely-cited 97% accuracy on European Community decisions. Tsebelis's veto-players theory (2002) provides the complementary structural insight that systems with more veto players separated by greater ideological distance produce more-predictable policy stability, while few-veto-player systems (executive-dominated autocracies, unified governments in emergencies) permit sudden policy shifts that are correspondingly harder to anticipate. Goldstone et al. (2010), working with the CIA-sponsored Political Instability Task Force, showed that a four-variable model (regime factionalism, infant mortality, armed conflict in neighbors, state-led discrimination) achieves 80–90% accuracy predicting state failure two years in advance — evidence that macro-political phenomena are sometimes highly predictable from small sets of structural indicators.

Against these optimistic results stand equally robust findings of the limits of political prediction. Tetlock's twenty-year, 28,000-prediction study (2005) found that expert analysts barely outperformed dart-throwing chimpanzees on medium-horizon political forecasts, with hedgehog thinkers (single-theory) systematically underperforming fox thinkers (multi-perspective). Jervis (1997) argued that international systems exhibit nonlinear interaction effects that defeat any reductionist prediction. The Salganik et al. (2020) Fragile Families mass-collaboration found that 160 teams with fifteen years of rich longitudinal data produced predictions of life outcomes that barely improved on a four-variable linear baseline — a cautionary ceiling on social prediction in general.

Two concepts from behavioral economics sharpen the question of *what* we should expect to be predictable about an individual leader. Simon's bounded-rationality framework (1955) holds that leaders under time pressure *satisfice* rather than optimize, relying on heuristics and prior commitments; these heuristics are in principle learnable from the leader's documented decision history. Kahneman and Tversky's prospect theory (1979) predicts that leaders in a perceived domain of losses take greater risks than those in a domain of gains, a directional signal that an LLM trained on the leader's own rhetoric might absorb. Together these results suggest that individual-leader decisions — especially routine, role-constrained, institutionally-embedded decisions — should be more predictable than their frequency in news coverage implies.

Crisis decisions are a special case. Hermann (1969) and the Allison-Zelikow framework (1999) argue that crises compress information-processing windows and force leaders to fall back on dispositional heuristics and prior beliefs, making *the decision process* more predictable even as the *triggering event* is unpredictable. This suggests that retroactive evaluation on decisions that have already been triggered — which is our setting — should find more signal than pre-emptive evaluation with both unknown triggers and unknown responses.

### 2.2 Leader Profiling: Evidence that Individual-Leader Behavior is Learnable from Text

A sustained research program in political psychology has established that structured profiles derived from a leader's own public language have predictive validity over that leader's subsequent behavior. The Verbs in Context System (VICS), formalized by Walker, Schafer, and Young (1998), extracts quantitative belief profiles from the transitive verbs in a leader's spontaneous speech and has been applied to 122 leaders across 48 countries (Hermann, 1998, 2003) with reported correlation of r = 0.84 between text-derived profiles and expert in-person assessments. Walker and Schafer (2006) present case studies in which operational-code scores derived from pre-crisis speeches correctly anticipate the cooperative-or-conflictual valence of subsequent foreign-policy responses; Marfleet and Miller (2005) reanalyze the Clinton-era Iraq crises in the same framework. The Leadership Trait Analysis (LTA) line developed by Hermann (1980) offers a complementary seven-trait personality measure used to classify leadership styles with documented behavioral correlates.

We cite this literature not because we adopt its measurement apparatus, but because it establishes the background empirical claim on which our work depends: individual-leader behavior is, to a measurable extent, predictable from text authored or received by that leader. If decades of human-coded profile research can surface such signal from tens of thousands of words, an LLM pre-trained on several orders of magnitude more text about a documented political figure should plausibly encode at least comparable signal — latently, in its weights — and the question becomes whether that signal is accessible through appropriate prompting.

### 2.3 Prediction Scoring and Calibration

Modern probability forecasting rests on the theory of proper scoring rules (Savage, 1971; Brier, 1950; Gneiting & Raftery, 2007), which establishes that an honest probability report is the loss-minimizing strategy under a proper rule. The Brier score, BS = (p − y)² for a binary outcome y ∈ {0,1} and forecast p ∈ [0,1], remains the canonical evaluation metric for binary forecasting: zero for perfect prediction, 0.25 for uninformed 50% forecasts, 1.0 worst case. Murphy's (1973) decomposition partitions the Brier score into reliability, resolution, and uncertainty components, enabling separate analysis of calibration and discrimination. Log loss (cross-entropy), LL = −[y log p + (1−y) log(1−p)], is strictly proper and more sensitive to confident errors, and is our secondary metric. For calibration specifically we use Expected Calibration Error (ECE) with equal-width binning (Naeini, Cooper, & Hauskrecht, 2015).

Modern neural networks and LLMs are systematically over-confident on held-out data (Guo, Pleiss, Sun, & Weinberger, 2017). We therefore apply post-hoc temperature scaling as a single-parameter calibration correction, reporting both raw and temperature-scaled Brier and log-loss, following the practice recommended by Guo et al. (2017). Dawid's prequential principle (1984) guarantees that the score depends only on the issued probabilities and observed outcomes, permitting comparison across models that may generate their forecasts by very different means.

### 2.4 LLM Forecasting Benchmarks

A growing literature evaluates LLMs as forecasters. ForecastBench (Zou et al., 2024) presents 500+ automatically-generated questions resolved post-training-cutoff and uses Brier score as the primary metric. Halawi et al. (2024) show that a retrieval-augmented pipeline with careful prompting can close much of the gap between LLM baselines and human aggregated-median benchmarks. Karger et al. (2025) describe Mantic, a production AI forecaster that finished fourth in a human forecasting tournament and introduces difficulty-adjusted Brier scoring so that easy questions (unanimous 95%+ correct answers) do not inflate aggregate scores. Payne et al. (2024) document a systematic escalation bias in LLMs on military and diplomatic decision scenarios — models recommend aggressive actions more frequently than the historical base rate — which implies an important failure mode our evaluation must detect and control for. Kovarikova et al. (2025) identify persona-specific biases in LLM forecasting that do not average out across runs, motivating our protocol of running each question ten times per condition.

Our evaluation builds directly on ForecastBench's post-training-cutoff requirement and on Mantic's difficulty-adjusted scoring. It differs from both in that we focus on a single leader rather than general-interest questions, and in that we bind knowledge at the database level rather than through prompt instructions.

### 2.5 LLM Persona Simulation and its Failure Modes

Argyle et al. (2023) introduced "silicon sampling" — the use of demographically-conditioned LLMs to simulate survey respondents — and reported strong alignment between model-generated and human-generated survey distributions on several American-politics instruments. Park et al. (2024) extended this to full-biography persona simulation, reporting approximately 85% agreement between LLM-generated and real responses for one thousand interviewed individuals. These encouraging results have been tempered by Bisbee et al. (2024), who identify five systematic failure modes in LLM persona simulation: run-to-run stochasticity, overgeneralization from training-data prototypes, collapse on out-of-distribution scenarios, value misalignment, and silent conflation of conflicting beliefs. Kovarikova et al. (2025) extend this analysis to predictive tasks specifically, identifying over-confidence and flattened-variance pathologies.

These failure modes inform our design in several concrete ways. First, we run each question ten times at temperature T > 0 and report variance alongside means, per Bisbee et al.'s stochasticity finding. Second, we include a neutral-analyst baseline with the *same retrieval context* as the persona, so that any persona-specific benefit (or cost) is isolated from the benefit of retrieval. Third, the multi-choice action-selection protocol (Section 5.3.2) checks whether the persona varies with scenario content or collapses to a scenario-independent default — an operational check for the flattened-persona failure mode.

### 2.6 Automated Evaluation Question Generation

High-quality question supply is the limiting reagent in rigorous LLM forecasting evaluation. Metaculus-style human curation produces excellent questions at low throughput. Bosse et al. (2026) demonstrate that a multi-agent automated pipeline can produce 1,499 questions at 96% quality (unambiguous, resolvable, appropriately-difficult) through a five-stage process: seed harvesting, proto-question drafting, research and refinement, verification, and deduplication with difficulty assignment. We adopt this architecture directly, extending it with a six-gate quality filter (resolvability, post-training-date, leader-attributable, measurability, difficulty-calibrated, topic-diverse) specialized for leader-decision forecasting.

### 2.7 Analogous Domains

A small literature anticipates our problem in adjacent domains. Katz, Bommarito, and Blackman (2017) predict U.S. Supreme Court outcomes using a boosted-tree classifier with engineered features, reaching approximately 70% accuracy on binary justice-level votes — a useful structural analog to leader-decision forecasting in that both involve identifiable, role-constrained decision-makers with long documented histories. The U.S. Department of Defense's Thunderforge program, the Defense Innovation Unit's GenWar Lab, and the Prototyping Suite's Ender's Foundry are all reported public-sector efforts to use generative models for adversary and ally decision simulation. Public reporting has also described a Central Intelligence Agency initiative to build leader-chatbots trained on OSINT and classified material for crisis-planning use. Our work sits in the open-science counterpart to these programs.

---

## 3. Problem Formulation

### 3.1 Task Definition

Let **L** denote a political leader (in our evaluation, Donald J. Trump). Let **D = {d₁, d₂, …, dₙ}** be a finite set of *decision questions* about L, each comprising:

- A **simulation date** **t_sim,i**, a calendar date that is treated as the fictive "today" for the question;
- A **resolution criterion** **R_i**, a deterministic rule mapping observable events to an outcome in a discrete (typically binary, or small-k multiclass) outcome space **Y_i**;
- A **resolution date** **t_res,i > t_sim,i**, by which R_i produces a unique outcome **y_i ∈ Y_i** from the public record;
- A **difficulty label** **δ_i ∈ {easy, medium, hard}**;
- A **base-rate estimate** **b_i ∈ [0,1]** summarizing historical frequency of outcomes of R_i's type for leaders in L's institutional role.

A **prediction system** is a function **f : (q_i, C_i(t_sim,i)) → p_i ∈ Δ(Y_i)** mapping a question and a context-set **C_i(t_sim,i)** of information available as of the simulation date to a probability distribution over outcomes. In the binary case this collapses to a single probability **p_i ∈ [0,1]**.

The prediction system is evaluated on a proper scoring rule averaged over **D**, potentially stratified by difficulty and domain, with calibration and resolution measured separately (Murphy, 1973).

The **persona condition** specifies that f is an LLM with a system prompt that identifies the model as L; the **analyst condition** specifies that f is an LLM with a system prompt instructing careful, calibrated, neutral analytical reasoning. In both conditions the retrieval context **C_i(t_sim,i)** is identical and is drawn from the CHRONOS knowledge base.

### 3.2 The Data Leakage Problem in Retroactive Evaluation

Retroactive evaluation — asking a model about decisions that have already occurred — is attractive because it provides ground truth at no experimental cost. It is also treacherous because the model's training corpus may already contain the outcome, either verbatim or in paraphrased news coverage. ForecastBench (Zou et al., 2024) and Bosse et al. (2026) address this by requiring that the *triggering event* postdates the model's training cutoff. We adopt this requirement and extend it.

The extended requirement derives from the observation that an LLM in a retrieval-augmented setup does not know *only* what its training data contains — it also receives whatever the retrieval system supplies. A document with **event_date > t_sim,i** that happens to contain information about the eventual decision is, at inference time, indistinguishable from legitimate contemporaneous context. Instructing the model to "pretend today is **t_sim,i**" does not, in general, cause the model to ignore such a document; the model has no mechanism by which to verify a date claim within its context window against a stated boundary.

We therefore require that **C_i(t_sim,i)** contain no document with **event_date > t_sim,i**. We enforce this at the database level with a hard SQL predicate (Section 4.3), not at the prompt level, on the principle that leakage prevention should not depend on the model's instruction-following fidelity.

### 3.3 Evaluation Desiderata

From the foregoing we derive six desiderata for any responsible evaluation of LLM leader-decision prediction:

1. **Post-training-cutoff triggers.** No decision may be included whose triggering event or reported outcome predates the evaluated model's training cutoff.
2. **Database-level temporal filtering.** No document with event_date after the simulation date may enter the model's context, enforced without reliance on model instruction-following.
3. **Resolvability.** Each question's outcome must be determinable from a specified public document class before a specified resolution date.
4. **Leader-attributability.** Each question must concern a decision traceable to L's personal directive, not a system-level aggregate outcome.
5. **Persona-versus-retrieval decomposition.** Evaluation must separate the contribution of persona conditioning from the contribution of retrieval, typically via a matched neutral-analyst baseline with identical retrieval context.
6. **Pre-registration.** The question manifest, experimental conditions, primary metrics, and statistical tests must be fixed before predictions are elicited, to prevent post-hoc selection effects.

The remainder of the paper describes a system and an evaluation protocol that satisfy these desiderata.

---

## 4. The CHRONOS Temporal Knowledge Base

### 4.1 Design Philosophy

CHRONOS is a temporal knowledge base, not a prediction system. Its job, given a leader L and a date range **[t_start, t_end]**, is to produce a corpus of event records spanning that range such that, at query time with a simulation date **t_sim**, a retrieval query conditioned on a question embedding returns only and all relevant records satisfying **event_date ≤ t_sim**.

Four design principles follow from this charter. First, **every record is a single point in time** — we do not store time-spanning summaries or "state as of month X" aggregations, because any such aggregation is a leakage surface. Second, **the temporal boundary is enforced by the database, not by the application** — the SQL query includes a WHERE clause that makes post-cutoff documents unretrievable. Third, **one database serves all LLMs** — different models have different training cutoffs, and CHRONOS handles this by a second WHERE clause parameterized on the evaluated model's cutoff. Fourth, **date errors are treated as leakage** — a record with a wrong event_date that places it before **t_sim** when the real event occurred after is indistinguishable from intentional leakage, so date integrity is treated as a first-class correctness property (Section 4.5.5).

### 4.2 Event Record Schema

Every ingested item is stored as an EventRecord with the following schema (JSON notation; PostgreSQL types in implementation):

```
record_id              : string, globally unique, e.g. "EVT-2025-03-15-001"
event_date             : date, day-precision default
event_date_precision   : enum {"day", "week", "month"}
date_confidence        : enum {"verified", "high", "single_source", "approximate", "uncertain"}
date_verification_method : string, human-readable audit trail
ingestion_date         : date
headline               : string, factual, neutral-tone
summary                : string, 100–200 words, bias-stripped
key_facts              : array of strings, concrete claims
direct_quotes          : array of {speaker, quote, context}
topics                 : array of freeform string tags
actors                 : array of normalized actor IDs
sources                : array of {name, url, type, pub_date}, type ∈ {primary, secondary, tertiary}
source_count           : integer
confidence             : float in [0,1]
embedding              : float[768], text-embedding-004 or nomic-embed-text
```

Two schema choices merit comment. First, topics and actors are freeform rather than drawn from a closed taxonomy; we rely on vector-space retrieval to handle topical relevance and on a small normalization layer to canonicalize actor names. Second, summaries are written in what the operational protocol terms *AP/Reuters wire-copy style* — short paragraphs, neutral register, attributed statements, no editorial framing. This is not a cosmetic choice: it minimizes the risk that a summary inadvertently telegraphs outcome information that a less-sanitized source article might contain.

### 4.3 SQL-Enforced Temporal Filtering

Retrieval is a single PostgreSQL query combining pgvector similarity search with hard temporal and quality filters:

```sql
SELECT record_id, headline, summary, key_facts, direct_quotes, event_date, sources
FROM event_records
WHERE event_date <= :simulation_date
  AND event_date >= :model_training_cutoff
  AND date_confidence != 'uncertain'
  AND embedding <=> :query_embedding < :similarity_threshold
ORDER BY event_date DESC, (embedding <=> :query_embedding) ASC
LIMIT :top_k;
```

The first predicate is the core leakage guard: no document with **event_date > :simulation_date** can appear in any retrieval result. Crucially, this predicate is evaluated by the database engine before any network or context-window boundary is crossed, and is not contingent on the model reading or respecting an instruction. The second predicate excludes information the model would have seen during pre-training, which would otherwise contribute duplicated signal. The third predicate excludes records whose date could not be corroborated (Section 4.5.5). The fourth predicate applies semantic-similarity thresholding through pgvector's cosine-distance operator.

Typical parameter values in our deployment: **top_k = 12–15**; **similarity_threshold ≈ 0.3** (cosine distance). The training-cutoff parameter is set per evaluated model; simulation_date is per question; query_embedding is produced by the same embedding model used during ingestion.

### 4.4 Multi-Model Compatibility Under Variable Training Cutoffs

Different LLM families have different training cutoffs:

| Evaluated Model     | Training Cutoff (approx.) | Knowledge Window for a July 2025 Question |
| ------------------- | ------------------------- | ----------------------------------------- |
| GPT-4o              | Oct 2023                  | Oct 2023 → Jul 2025 (≈21 months)          |
| Claude 4 Sonnet     | Early 2025                | Early 2025 → Jul 2025 (≈6 months)         |
| Gemini 3 Pro        | Mid 2025                  | Mid 2025 → Jul 2025 (≈1 month)            |
| Claude Opus 4.7     | (per model card)          | (per model card) → Jul 2025               |

CHRONOS supports all of these from a single database by accepting **model_training_cutoff** as a query-time parameter. The corpus is collected once over the range [Oct 2023, t_end], accommodating the earliest-cutoff model we might evaluate. At inference, a model-specific lower bound constrains the retrieval window. No re-ingestion is required when a new model is added; the corpus is reused.

### 4.5 The Multi-Agent Research Swarm

The knowledge base is constructed by an autonomous swarm of seven specialized agents orchestrated in a supervisor-worker pattern. The system accepts only three configuration inputs — leader name, date range, and a budget cap — and is expected to run to completion without human intervention. Agent specifications summarized below follow the detailed build specifications in `temporal_knowledge_base/build_prompts/` and the coordinating `SYSTEM_DESIGN.md`.

#### 4.5.1 Coordinator

The Coordinator is the supervisor. On initialization it decomposes the collection window into monthly chunks and generates, for each chunk, a set of search queries spanning six or more decision domains (at least 50 initial queries for a 12-month range). It maintains a coverage map tracking events-per-month-per-domain, monitors gap reports from the Coverage Auditor, and dispatches targeted follow-up queries to fill identified gaps. The Coordinator also enforces the budget cap and the maximum-rounds limit (five iterations), terminating collection when either the coverage criteria are satisfied or a hard cap is reached.

#### 4.5.2 Discovery Agents

Multiple Discovery workers run concurrently, each executing a ReAct loop (Yao et al., 2023) over a set of tools: a web-search wrapper (Serper.dev or Google Search); the GDELT 2.0 API; structured-source APIs (Federal Register, Congress.gov, OFAC, ACLED); and an RSS monitor for White House press briefings. Each worker processes a batch of Coordinator queries, returning RawEventCandidate objects {title, url, snippet, date_hint, source_tier} with a source-tier score (1.0 for .gov primary sources and wire services; 0.8 for major broadsheets; 0.6 for regional outlets; 0.3 for blog and opinion sources; 0.5 unknown). Deduplication at the worker level uses URL and >80% fuzzy-headline overlap.

#### 4.5.3 Extraction Agent

The Extraction Agent receives RawEventCandidates and resolves them to full article text, using Playwright for pages with client-side rendering or anti-bot protection and newspaper3k for parsing. Rejections include paywalls, 404s, stub pages, and articles under 100 words. The critical responsibility of this agent is to distinguish **publication date** (when the article was posted) from **event date** (when the described event occurred); the latter is extracted from the article body, cross-checked against the former, and passed downstream as the authoritative date candidate.

#### 4.5.4 Cleaning Agent

The Cleaning Agent bias-strips article text to AP/Reuters wire-copy style, writes 100–200 word LLM-readable summaries, extracts key facts and attributed direct quotes, and assigns topic and actor tags. Pairwise deduplication uses an LLM-based similarity classifier on summary embeddings; record pairs with similarity > 0.8 are merged, preserving the union of sources and quotes and the higher-confidence date. Actor normalization canonicalizes name variants ("Donald Trump", "President Trump", "Trump", "Mr. Trump") to a single internal identifier. Pre-merge confidence scores follow a simple scheme: 0.95 for three-or-more source corroboration, 0.85 for two, 0.60 for singletons.

#### 4.5.5 Temporal Validator: Four-Layer Validation

The Temporal Validator is the most consequential agent in the swarm, because a wrong date *is* data leakage. It applies four layers of validation, summarized in the following table:

| Layer | Name                                 | Check                                                                                                  | Action on Failure        |
| ----- | ------------------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------ |
| 1     | Mechanical parsing                   | Date is a valid calendar date; not in the future; within collection window; publication_date ≥ event_date (±1 day). | Reject record.           |
| 2     | Cross-source corroboration           | ≥3 sources agree on date → verified; 2 → high; 1 explicit date → single_source; disagreement > 2 days → uncertain. | Quarantine uncertain.    |
| 3     | LLM-based logical consistency        | LLM classifier checks for anachronisms, reference-chain inversions, impossible political context (e.g., Trump signing an EO before taking office). | Quarantine inconsistent. |
| 4     | Statistical outlier detection        | Flags records with suspicious clustering, or with publication-date-to-event-date gap exceeding 30 days. | Flag (not reject); route to audit. |

The quarantine system is an explicit design principle: any record whose date cannot be corroborated is *retained in the database with date_confidence = uncertain* but *excluded from all retrieval queries* by the third WHERE clause in Section 4.3. The operating asymmetry is deliberate: the cost of admitting a wrongly-dated record is catastrophic (an entire evaluation is compromised), while the cost of excluding a correctly-dated record is marginal (some redundant event is missed). We therefore resolve the asymmetry in favor of exclusion.

#### 4.5.6 Indexing Agent

The Indexing Agent writes validated EventRecords to PostgreSQL, generates summary embeddings with a configurable embedding model (text-embedding-004 by default; nomic-embed-text as a low-cost alternative), and maintains secondary indices on event_date, topics (GIN), and actors (GIN).

#### 4.5.7 Coverage Auditor

The Coverage Auditor runs after each collection round and produces a gap report:

- **Sparse-month flag**: months with fewer than 15 validated events.
- **Missing-topic flag**: topic categories absent from a given month.
- **Recency-bias flag**: months with more than 3× the overall monthly average (often a symptom of publication-recency in search APIs).
- **Quarantine-rate flag**: overall quarantine rate exceeding 15%, indicating a systemic source-quality problem.

Completion criteria: all months have ≥15 validated events; no month has a topic category with zero events; quarantine rate is ≤15%; a minimum of three collection rounds has completed. The Coverage Auditor's report feeds back to the Coordinator, which generates targeted follow-up queries for any flagged gaps and initiates the next round, up to a five-round cap.

### 4.6 The Intelligence Briefing Format

Retrieval results are not returned as SQL row sets to the evaluated LLM. They are rendered as a natural-language intelligence briefing with a header block identifying the subject, the simulation date, the knowledge window, and the total number of visible events, followed by each retrieved record rendered as a dated section with headline, summary, key facts, attributed quotes, and source-count footer. The briefing closes with a total-retrieved footer. Format details follow the specification in `temporal_knowledge_base/SYSTEM_DESIGN.md` §5.

This rendering choice is not cosmetic. The LLM is more likely to respect date boundaries on events whose structure it is presented with explicitly (dated headers) than on events buried in a JSON blob; and the natural-language format provides a consistent interface across LLM families with different structured-data tolerances. A specimen briefing appears in Appendix A.

---

## 5. Methodology

### 5.1 Question Generation Pipeline

#### 5.1.1 Five-Stage Generation

We adopt the five-stage multi-agent pipeline of Bosse et al. (2026), adapted for leader-decision questions. The stages are:

1. **Seed harvesting.** A Scanner Agent monitors structured decision-generating sources — the Federal Register, USTR Federal Register notices, Congress.gov bill histories, OFAC sanctions actions, prediction-market listings (Kalshi and Polymarket), and wire-service decision stories — for candidate decisions. Each candidate is serialized as a DecisionSeed object: seed_id, event_description, decision_taken, decision_date, a proposed simulation_date 1–30 days before the decision, plausible_alternatives, and sources.
2. **Proto-question drafting.** A Drafter Agent converts each seed into a binary question and, when the alternatives permit, a parallel action-selection question, using rigid templates: binary takes the form "Will [Leader] [action verb] [object] [by/before date]?"; action-selection provides three to five mutually-exclusive options plus a mandatory "None of the above" option. Domain and difficulty labels are attached.
3. **Research and refinement.** A Researcher Agent verifies against the public record that the seed's decision actually occurred as described and that the resolution criterion is satisfiable from a specific public document. The question is refined for concreteness and a pointed ambiguity pass is run.
4. **Verification (adversarial critic).** A Critic Agent in a different model family from the Drafter stress-tests each question against the six quality gates (Section 5.1.2) on a four-point scale (bad / meh / good / great). Any gate below "good" returns the question to refinement; questions failing after two refinement passes are rejected. The Critic also red-flags coin-flip-difficulty questions (base rate outside [0.1, 0.9]).
5. **Deduplication, difficulty assignment, and manifest locking.** A Resolver Agent pre-computes base-rate estimates from historical frequency, pulls analogous Kalshi/Polymarket pricing where available as a difficulty benchmark, sets the resolution watchlist (URLs/APIs/RSS feeds to monitor), and assigns the question to easy/medium/hard tiers (target distribution 20/60/20). Near-duplicate questions (embedding similarity > 0.9) are merged.

The pipeline output is a versioned JSON manifest (`questions-v1.0`) with pre-registration date, model-under-test identifier, model training cutoff, per-domain and per-difficulty counts, and per-question metadata. We pre-register the manifest on the Open Science Framework before any predictions are elicited.

#### 5.1.2 Six-Gate Quality Filter

Every question must pass all six gates:

1. **Resolvability.** The outcome must be a clear yes/no (or small-k multiclass) answer verifiable from public sources. Counter-example: "Will Trump's trade policy hurt the economy?" (irredeemably subjective).
2. **Post-training-date.** The triggering event and the resolution must occur after the evaluated model's training cutoff. For multi-model evaluation, the effective cutoff is the maximum over models-under-test.
3. **Leader-attributability.** The decision must be traceable to the leader's personal directive (executive order signed; statement issued; appointment announced). System-level outcomes ("Will GDP grow above 3%?") are excluded.
4. **Measurability.** The resolution criterion must name a specific document class and precision (Federal Register entry with tariff rate ≥ 20%; USTR announcement; Congressional Record vote roll-call).
5. **Difficulty calibration.** The question set must satisfy the target distribution — 20% easy (base rate > 80% for one outcome), 60% medium (30–70%), 20% hard (< 20% on the true outcome).
6. **Topic diversity.** The set must satisfy domain weights: 25% trade/tariffs, 20% executive orders, 15% personnel, 15% foreign policy, 10% legislative, 10% public communications, 5% legal/judicial.

#### 5.1.3 The Resulting Question Set

The pipeline has produced **115 approved questions** (8 candidates rejected) distributed across the seven decision domains as follows: executive orders (21), tariffs and trade (24), personnel (18), foreign policy (16), legislative (12), legal and judicial (10), and public communications (14). Representative questions, per-question metadata fields, and a sample manifest entry appear in Appendix B.

### 5.2 Prompting Conditions

#### 5.2.1 Persona Prompt (Trump)

The persona system prompt establishes the model as the named leader and provides a minimal framing instruction:

> "You are Donald J. Trump, the 47th President of the United States. You are being asked about a decision you face on the simulation date provided in the user message. Answer based on your actual beliefs, priorities, strategic interests, negotiating style, past statements, and decision-making patterns."

The user message then supplies the intelligence briefing (Section 4.6), the question text, resolution criterion, and a required output schema: JSON with `{probability: float in [0,1], reasoning: string of 2-4 sentences}`.

#### 5.2.2 Non-Persona Baseline Prompt

The non-persona baseline uses the identical user message but a deliberately leader-neutral, calibration-oriented system prompt written in a generic-LLM-analyst register:

> "You are a careful, calibrated political analyst forecasting a specific decision by a named political figure. You will receive a simulation date (treat this as 'today' — you know only what has happened up to and including this date), optionally an intelligence briefing summarizing recent events, and a question with a resolution criterion and a resolution date. Reason step by step internally. Weigh base rates, recent signals, incentives, and precedent. Avoid overconfidence."

The output schema is identical (JSON with `probability` and `reasoning`). We treat the two system prompts — persona and non-persona — as a principal experimental manipulation rather than a secondary control. The persona-vs-baseline contrast with matched retrieval directly tests our central research question: whether identity-assertion framing ("You are Donald Trump") unlocks predictive signal that a leader-neutral analyst framing does not. The baseline prompt is not ideology-neutral in a partisan sense; it is *style-neutral with respect to the leader*, which is the relevant control for isolating the contribution of persona conditioning from the contribution of temporally-bounded retrieval.

### 5.3 Experimental Protocols

We run six experimental conditions, labeled E1 through E4 plus an E1′ retrieval-volume control and an E2′ retrieval-quality control, designed to decompose the contributions of persona framing, retrieval, and retrieval-curation. Each condition is run at temperature T = 1.0; reported metrics aggregate across samples unless otherwise noted.

| ID    | Condition                                        | System Prompt       | CHRONOS Retrieval | Purpose                                   |
| ----- | ------------------------------------------------ | ------------------- | ----------------- | ----------------------------------------- |
| E1    | Persona + retrieval (main)                       | "You are Trump"     | Broad top-15      | Flagship result                           |
| E1′   | Persona + compressed retrieval                   | "You are Trump"     | Top-8             | Isolates context compression from curation|
| E2′   | Persona + refined retrieval *(post-hoc)*         | "You are Trump"     | 2-stage refined   | Tests retrieval curation at lower token cost |
| E2    | Persona without retrieval                        | "You are Trump"     | None              | Isolates persona parametric knowledge     |
| E3    | Non-persona + retrieval                          | "Careful analyst"   | Broad top-15      | Persona-framing control for E1            |
| E4    | Non-persona + web search                         | "Careful analyst"   | Live web search   | Answerability gate                        |

E1 is our main condition. E2 tests what the persona alone can do with only its training-data knowledge. E3 is the persona-framing control: same retrieval as E1, same question, same output schema, differing only in the system prompt's identity framing. The **E1 − E3** contrast is the primary, pre-registered test of our central research question — whether identity-assertion framing unlocks predictive signal that a leader-neutral analyst framing does not. E1′ tests whether any persona effect in E1 is driven by retrieval volume rather than retrieval content. E4 is an answerability gate: if a non-persona analyst with live web search cannot resolve the questions well, the question set is faulty.

**E2′ note.** E2′ was added to the protocol after pre-registration, as a cost-quality ablation motivated by the observation that E1's broad top-15 briefing consumed ~70K input tokens per prediction. E2′ applies a two-stage retrieval pipeline — LLM-driven query expansion followed by over-retrieval (top-40 candidates) and LLM rerank (keep ~12 most relevant) — to produce a briefing approximately one-sixth the size of E1's. Because E2′ was added post-registration it is reported as a secondary condition and is *not* included in the primary pre-registered contrast.

#### 5.3.1 Experiment 1: Binary Probability Forecasting

For each of the 115 questions, each of the five conditions, and each of 10 samples, the model produces a probability in [0, 1]. We compute per-question mean probability and variance across samples, and aggregate metrics (Section 5.4) across questions, stratified by difficulty and domain.

Baselines are reported alongside models: an uninformed 50% baseline (Brier = 0.25); a base-rate baseline using each question's pre-computed base rate; a prediction-market baseline from Kalshi/Polymarket prices where analogous contracts exist; and the relevant Good-Judgment-Project superforecaster benchmark (Brier ≈ 0.12–0.15) as an external upper comparison.

#### 5.3.2 Experiment 2: Multiple-Choice Action Selection

A subset of the question set (≈ 40–50 questions) is written in multiple-choice action-selection form with three to five mutually-exclusive options including a mandatory "No action" option. For each question, each condition, and each of 10 samples, the model returns a ranked ordering over the options. Metrics:

- **Top-1 accuracy**: fraction of questions for which the reality is the model's top choice (random baseline = 1/k).
- **Top-2 accuracy**: fraction of questions for which the reality is in the model's top two (random baseline = 2/k).
- **Spearman rank correlation**: mean rank correlation between the model's ranking and a reality-aligned reference ranking (where available).

Targets: top-1 accuracy > 40% for five-option and > 50% for three-option questions (both well above random); top-2 accuracy > 60% (five-option). Analog reference point: Katz et al. (2017) achieved ≈ 70% on binary SCOTUS outcomes; we expect lower ceilings here given the larger outcome space and the tighter leader-attributability requirement. Crucially, the action-selection protocol provides an additional check against the flattened-persona failure mode (Bisbee et al., 2024): if the persona condition produces the same top choice across scenarios that should elicit different actions, the persona is not tracking situational variation even if its aggregate Brier score looks reasonable.

### 5.4 Scoring and Statistical Analysis

#### 5.4.1 Primary Metrics

- **Brier score** (primary). Per-question Brier averaged across samples, then across questions, stratified by difficulty and domain. We report difficulty-adjusted Brier in the style of Karger et al. (2025) so that unanimously-easy questions do not dominate the aggregate.
- **Log loss** (secondary). Reported on the same schedule; heavier penalty on confident errors.
- **Expected Calibration Error (ECE)**. Equal-width binning with 10 bins; we also report reliability diagrams per condition.
- **Resolution (discrimination)**. Variance of the model's predicted probabilities across the question set; a model that hedges everything at 50% will score low on Resolution even if its Brier is unexceptional.
- **Coverage rate**. Fraction of predicted 90%-confidence intervals that contain the ground-truth outcome.

#### 5.4.2 Temperature-Scaled Calibration

Following Guo et al. (2017) we fit a single-parameter temperature T on a 30% held-out calibration split of the question set, minimizing temperature-scaled negative log-likelihood, and report both raw and temperature-scaled Brier and log-loss on the 70% evaluation split. We do this per condition, not globally, since systematic over-confidence is condition-dependent (we expect the persona condition to be more over-confident than the analyst condition, per Kovarikova et al., 2025).

#### 5.4.3 Statistical Testing

Pairwise condition comparisons use the paired Wilcoxon signed-rank test on per-question Brier scores (non-parametric, appropriate for bounded scores and modest n ≈ 115). All Brier and log-loss aggregates are reported with 95% bootstrap confidence intervals over 10,000 resamples, stratified by question when resampling. We correct for multiple comparisons across the primary condition contrasts — the flagship persona-vs-analyst contrast (E1 vs. E3), the retrieval-ablation contrasts (E1 vs. E1′, E1 vs. E2), and the E4 answerability gate — using the Holm-Bonferroni procedure.

Effect-size reporting accompanies significance: Cliff's delta for Wilcoxon; paired Cohen's d on log-loss as a sensitivity check.

### 5.5 Pre-Registration

Before any predictions are elicited from an evaluated model, we pre-register the following on the Open Science Framework:

1. The 115-question manifest (`questions-v1.0`, locked by tag in the project repository).
2. The five experimental conditions (E1, E1′, E2, E3, E4), their prompt templates verbatim, and the temperature and sample-count settings.
3. The primary metric (difficulty-adjusted Brier) and the primary condition contrast (E1 vs. E3, persona vs. non-persona with matched retrieval).
4. The statistical-testing plan (paired Wilcoxon; Holm-Bonferroni correction; bootstrap CIs).

We also commit in advance to publishing negative results. The framework is designed such that a null result — persona conditioning produces no measurable benefit over analyst baseline with matched retrieval — is informative about LLM persona-simulation limits and is of independent interest.

---

## 6. Implementation

### 6.1 Technology Stack

CHRONOS is implemented in Python with LangGraph (Chase et al., 2024) for stateful multi-agent orchestration; Neon-hosted PostgreSQL with the pgvector extension for the event-record store; text-embedding-004 (Google) as the default embedding model with nomic-embed-text as a low-cost alternative; Playwright and newspaper3k for article ingestion; Serper.dev or Google Search as the web-search backend; FastAPI as the backend retrieval API; and Next.js with React for the coverage-inspection dashboard. Discovery and extraction use Gemini 3 Flash for cost efficiency; validation and logical-consistency checks use Gemini 3 Pro or Claude Opus 4.7 for quality. The evaluation pipeline itself uses the pinned evaluated-model (Claude Opus 4.7 in our initial run) through its provider API under version-locked prompt templates. Dependency management uses `uv`; Neon branching is used for multi-contributor collaboration with isolated development branches per agent.

### 6.2 Cost, Latency, and Storage

Typical operating envelope per leader over a 2.5-year collection window:

- **Storage**: ≈ 80 MB per leader (≈ 16 MB records + 24 MB embeddings + 40 MB source archive).
- **Collection cost**: \$17–30 per leader end-to-end (research-swarm LLM calls + embeddings + web-search API fees). Budget capped at run start.
- **Retrieval latency**: 150–300 ms wall-clock including vector search, rerank, and briefing rendering.
- **Cost per evaluation batch**: ≈ \$1 per 1,000 predictions at Gemini pricing; scales with model choice. The initial evaluation run on 115 questions × 5 conditions × 10 samples = 5,750 predictions under Opus 4.7 is estimated at several hundred US dollars.

### 6.3 Reproducibility and Versioning

All prompts, schemas, and configurations live under version control. The question manifest is git-tagged (`questions-v1.0-locked`). The ground-truth resolution protocol runs two independent passes with different LLM providers (Gemini 2.5 Flash with live web search, and Claude Opus 4.7 / Gemini 3 Pro as a second pass), reconciling any disagreement (expected rate 10–20%; a rate > 30% triggers an audit of the resolver) through human adjudication before the manifest is locked. We publish, alongside the manifest, all prompt templates, all retrieval queries, the full swarm specification, and any emergent patches to the corpus as errata.

---

## 7. Experimental Setup

### 7.1 Question Corpus

- **Leader**: Donald J. Trump, 47th President of the United States.
- **Collection window**: October 2023 through April 2026.
- **Questions**: 115 approved (8 rejected during generation); version `questions-v1.0-locked`.
- **Domain distribution**: executive orders (21), tariffs and trade (24), personnel (18), foreign policy (16), legislative (12), legal/judicial (10), public communications (14).
- **Difficulty distribution**: target 20% easy / 60% medium / 20% hard; realized distribution reported alongside results.

### 7.2 Models Under Evaluation

- **Primary evaluated model (executed run)**: Gemini 2.0 Flash-Lite (`gemini-2.0-flash-lite`). Training cutoff August 2024, which is entirely prior to the simulation window of this evaluation (earliest simulation date 2025-01-02) — pretraining leakage of the outcomes being forecast is therefore impossible by construction. The model was selected over the pre-registered primary (Claude Opus 4.7) and over Gemini 2.5 Flash specifically because (i) Opus 4.7's January 2026 cutoff post-dates every simulation date in the manifest, making every question potentially contaminated by pretraining; (ii) Gemini 2.5 Flash's January 2025 cutoff overlaps 58 of the 104 simulation dates; (iii) Gemini 2.0 Flash-Lite is the highest-capability Gemini variant whose cutoff cleanly pre-dates the entire simulation window while remaining on the same training-data family as Gemini 2.0 Flash.
- **Auxiliary retrieval refiner (E2 two-stage)**: Gemini 2.0 Flash-Lite at T = 0.0, same cutoff — so the refiner cannot leak Jan-2025 event knowledge through its ranking decisions.
- **Planned robustness checks (not run in this evaluation)**: Claude Opus 4.7, GPT-4o, Gemini 3 Pro, Claude Sonnet 4.6. These would each require their own cutoff-appropriate simulation-date restrictions.
- **Discovery/validation models (swarm-internal, not under evaluation)**: Gemini 3 Flash for high-throughput operations; Gemini 3 Pro and Opus 4.7 for validation and logical-consistency checks. These operate only on the knowledge-base-construction side and do not generate predictions.

### 7.3 Baselines

- **Uninformed (50%)** — Brier 0.25.
- **Base-rate** — per-question base rate from historical frequency; aggregate Brier ≈ 0.20.
- **Generic LLM without persona or retrieval** — the Opus 4.7 under E3-style zero-retrieval prompting as a "what does the model know unaided" baseline.
- **Prediction market** — Kalshi/Polymarket last-trade price where an analogous contract is listed.
- **Superforecaster external reference** — GJP Brier ≈ 0.12–0.15 as the human upper bound.

### 7.4 Resolution Protocol

Ground truth for each question is established through two independent resolution passes, following the `STATUS.md` protocol of the project's evaluation sub-team. Pass A uses Gemini 2.5 Flash with live Google Search over the 115 questions; Pass B uses Claude Opus 4.7 with `web_search` (with Gemini 3 Pro as a fallback provider chosen for uncorrelated failure modes). Per-question outcomes from the two passes are reconciled; disagreements are expected at 10–20% of questions and are referred to human adjudication. A disagreement rate exceeding 30% triggers an audit of the resolver pipeline before proceeding. The locked resolution manifest is committed to the repository with the tag `questions-v1.0-resolved`.

### 7.5 Planned Ablations

- **Retrieval top-k**: {5, 8, 15, 25} on a 20-question ablation subset, to characterize the quality-volume tradeoff.
- **Recency weighting**: uniform versus exponential decay on event_date, on the same 20-question subset.
- **Embedding model**: text-embedding-004 versus nomic-embed-text versus a dense-retrieval baseline (BM25) to bound the contribution of retrieval quality.
- **Persona prompt depth**: minimal persona (single sentence) versus expanded persona (paragraph with explicit decision-heuristic priming), to test the sensitivity of any observed persona effect to prompt engineering.
- **Temperature**: T ∈ {0.3, 0.7, 1.0} at N = 10 samples, to characterize the variance-accuracy tradeoff documented in Bisbee et al. (2024) and Kovarikova et al. (2025).
- **Escalation-bias probe**: a separate 15-question set of crisis scenarios with historically-non-escalatory ground truths, to detect and quantify the Payne et al. (2024) escalation bias in the persona condition specifically.

### 7.6 Timeline and Budget

Ground-truth resolution of the 115 questions (Pass A + Pass B + adjudication) is expected to complete within one week of manifest lock at an estimated cost of \$150–300. CHRONOS autonomous collection across the 2.5-year window, with a five-round cap on the swarm loop, is budgeted at \$30 per leader. The six-condition evaluation at 10 samples per question is budgeted at several hundred US dollars depending on final model selection.

---

## 8. Results

### 8.1 Deviations from the Pre-Registered Plan

We report three deviations from the plan of Section 7, each of which we describe here before any results, so the reader can weigh them.

1. **Evaluated model.** The pre-registered primary model was Claude Opus 4.7. Its January 2026 training cutoff post-dates every simulation date in our manifest (earliest sim date 2025-01-02, latest 2025-05-xx), meaning pretraining leakage of the outcomes being forecast could not be ruled out — a hard-to-correct confound for any retroactive evaluation. We substituted **Gemini 2.0 Flash-Lite** (`gemini-2.0-flash-lite`), whose August 2024 training cutoff cleanly precedes the entire simulation window, and re-audited all 103 resolvable questions to confirm none of their outcomes fall before the cutoff. A secondary run on Opus 4.7 with a sim-date restriction remains a planned robustness check; it cannot substitute for the primary because the majority of questions in our manifest would have to be dropped to keep the evaluation clean.
2. **Samples per question.** The pre-registered sample count was N = 10. We ran **N = 5**, for budget reasons. This reduces within-question variance estimation precision by √2 but does not affect the unbiasedness of per-question Brier estimates.
3. **Sixth condition (E2 refined retrieval).** The two-stage refined-retrieval condition was added after pre-registration as a cost-quality ablation using the same LLM as the primary (at T = 0.0) for query expansion and event reranking. We report it alongside the pre-registered conditions and include it in the secondary contrasts; no claim in §8 is vitiated by its pre-registration status.
4. **Seventh condition (E6 answerability gate).** During post-hoc inspection of the E5 briefings we discovered that the pre-registered strict Tavily date filter — which drops any result whose `published_date` is missing, unparseable, or after the simulation date — in practice drops *essentially all* Tavily results for our queries, because Tavily rarely populates `published_date` on the sources it returns (see §8.9 for the audit that revealed this). E5 as pre-registered was therefore effectively running the analyst with an empty context string, not a date-bounded web search. Rather than silently suppress this finding, we (a) retain E5's result verbatim and annotate it in §8.2, §8.3, §8.9 as a *degraded-context* condition rather than a web-search condition, and (b) add a post-hoc seventh condition, **E6 ("answerability gate")**: the same analyst prompt with Tavily called at `end_date=None` and `strict_date_filter=False`, admitting all results including post-outcome coverage, to test whether the questions are even answerable by a model with full hindsight. E6 is not a pre-registered contrast — we treat it as a diagnostic, report its Brier alongside the six pre-registered conditions, and draw no pre-registration-weight claims from it. All contrasts involving E6 are labeled post-hoc throughout §8.

The question manifest and resolutions were frozen before any predictions were elicited. All 3,605 predictions (7 × 103 × 5) were written to append-only JSONL with per-record prompt hash, briefing hash, model ID, and timestamps, and are archived in `pipeline/output/predictions/{e1,e1p,e2,e3,e4,e5,e6}/predictions.jsonl` alongside the snapshot `evaluation_plan/output/snapshots/`.

### 8.2 Question Set and Record Counts

Of the 115 pre-registered questions, **103 were resolvable** by the two-pass resolution protocol within the evaluation window (12 questions did not yet have a ground-truth outcome determinable from their specified document class). Resolutions were locked in `pipeline/output/resolutions/resolutions.json` before scoring.

The seven conditions and the codebase identifiers that name their predictions files are:

| Cond. | Identifier | Persona | Context source | Status |
| --- | --- | --- | --- | --- |
| E1 | `e1` | Trump | CHRONOS, broad top-15 | pre-registered |
| E1′ | `e1p` | Trump | CHRONOS, broad top-8 (compressed) | pre-registered |
| E2 | `e2` | Trump | CHRONOS, two-stage refined (keep 8–12 after rerank) | added, §8.1.3 |
| E3 | `e3` | Trump | None | pre-registered |
| E4 | `e4` | Analyst | CHRONOS, broad top-15 (reuses E1's briefing verbatim) | pre-registered |
| E5 | `e5` | Analyst | Tavily web search, **strict date-filter (undated results dropped)** | pre-registered; filter produced empty context on most questions — see §8.1.4, §8.9 |
| E6 | `e6` | Analyst | Tavily web search, **unbounded** (no date filter, post-outcome coverage admitted) | post-hoc answerability gate, §8.1.4 |

Record counts and token usage (target: 515 = 103 questions × 5 samples):

| Cond. | Description                                  | Records | Valid | Errors | Input tokens | Output tokens |
| ----- | -------------------------------------------- | ------: | ----: | -----: | -----------: | ------------: |
| E1    | Trump × CHRONOS broad-15                     |     515 |   515 |      0 |   12,673,555 |        74,802 |
| E1′   | Trump × CHRONOS broad-8                      |     515 |   515 |      0 |    7,523,555 |        73,724 |
| E2    | Trump × CHRONOS refined                      |     515 |   515 |      0 |    1,525,011 |        69,481 |
| E3    | Trump × no retrieval                         |     515 |   515 |      0 |      668,030 |        67,869 |
| E4    | Analyst × CHRONOS broad-15                   |     515 |   515 |      0 |   12,772,424 |        88,651 |
| E5    | Analyst × web search, strict filter          |     515 |   515 |      0 |      723,336 |        77,675 |
| E6    | Analyst × web search, unbounded (gate)       |     515 |   515 |      0 |    1,279,478 |        83,743 |

**All seven conditions completed with zero errors.** Every question has five valid samples in every condition. This is the result of an iterative process: an earlier run of this evaluation produced >100 invalid records in the web-search condition from Gemini 429 rate-limit errors and `RemoteProtocolError` disconnects during refiner calls. We rebuilt the retry and rate-limit handling (transient-error retry list, nine-attempt exponential backoff to 420 s, concurrency downgraded from 8 → 3 → 1 for cleanup) and re-ran the failing records. Partial E4/E5 attrition is *not a characteristic of the conditions* in this evaluation — it was infrastructure, and we fixed the infrastructure and re-ran. The resulting n = 103 common-qid comparison is on the full question set. (A separate issue — E5's strict Tavily date filter emptying most briefings — is orthogonal to attrition: E5 produced 515 well-formed records with zero errors; the records simply had no web-search context in them because the filter dropped every undated result. See §8.1.4 and §8.9.)

The E6 row uses only ~1.28 M input tokens despite admitting full search results because Tavily's advanced search returns compact summaries; the search-context block with 10 kept snippets fits in roughly 2,500 input tokens per question, compared to ~25,000 for an E1 CHRONOS broad-15 briefing. Said differently: CHRONOS pays a ~10× input-token premium over a modern web-search context for the 103 questions in this manifest.

The input-token column is worth pausing on: broad-15 CHRONOS briefings consumed roughly 18× the input tokens of the no-retrieval condition and ~17× the refined condition. The refined condition used roughly 8× fewer input tokens than broad-8 despite keeping a similar number of events after rerank, because the refiner drops events to 8–12 *actually-relevant* ones rather than the top-8-by-embedding. §8.8 returns to the cost side of this picture once the accuracy side is in view.

### 8.3 Primary Metric: Mean Brier Score (n = 103 Common Questions)

Per-question Brier is computed as the mean over the 5 samples for that (condition, question); per-condition aggregate is the mean over the 103 common questions. Binary Brier ∈ [0, 1]; multiclass Brier (unnormalized, 3–5 option action questions) ∈ [0, 2]. The question mix is 58 binary and 45 action questions. 95% bootstrap confidence intervals are from 10,000 paired resamples over the 103 questions (seed 42).

| Cond. | Description                                   | Mean Brier | 95% CI           | Mean p(correct) |
| ----- | --------------------------------------------- | ---------: | ---------------- | --------------: |
| E1    | Trump × CHRONOS broad-15                      |     0.4774 | [0.4073, 0.5467] |          0.4276 |
| E1′   | Trump × CHRONOS broad-8                       |     0.4835 | [0.4148, 0.5515] |          0.4219 |
| E2    | Trump × CHRONOS refined                       |     0.4723 | [0.4053, 0.5382] |          0.4296 |
| E3    | Trump × no retrieval                          |     0.4804 | [0.4170, 0.5433] |          0.4171 |
| E4    | Analyst × CHRONOS broad-15                    |     0.4808 | [0.4107, 0.5507] |          0.4254 |
<!-- DELETE: E5 row — strict-filter Tavily produced empty context on most questions; condition didn't measure web search. Keep only as a null-context baseline footnote, or remove entirely. -->
| E5    | Analyst × web search (strict, degraded)       |     0.4863 | [0.4212, 0.5531] |          0.4054 |
<!-- /DELETE -->
| E6    | Analyst × web search (unbounded, gate; post-hoc) | **0.4518** | [0.3819, 0.5222] |      **0.4609** |

**The six bounded-condition confidence intervals heavily overlap.** The spread between the nominal best *bounded* condition (E2, 0.472) and nominal worst (E5, 0.486) is 0.014 Brier units, on a scale where each individual 95% CI is ≈0.13 units wide. Every per-bounded-condition CI contains every other bounded-condition mean. On a per-question paired basis (§8.4) the spread is smaller still. We therefore cannot distinguish E1 through E5 on forecasting accuracy with the present sample size; nor does the nominal ordering favor any of the pre-registered hypotheses.

**E6 sits 0.020 Brier units below the best bounded condition (E2) and 0.035 below the pre-registered web-search condition (E5).** On the per-condition 95% CI scale, E6's CI overlaps every bounded condition's CI; the separation is material only on per-question paired tests (§8.4). Because E6 admits post-outcome Tavily results, this separation is *not* evidence that date-bounded web search helps — it is an answerability-gate diagnostic, showing that full hindsight does in fact produce a (small) Brier reduction, and more strikingly, that even with full hindsight **p(correct) reaches only 0.46, not ~1.0**. We return to this in §8.9.

### 8.4 Pre-Registered Paired Contrasts

All contrasts are computed per-question on the n = 103 paired subset. Reported: mean signed Brier delta (positive = condition `a` worse); 95% bootstrap CI on the mean delta (10,000 resamples, seed 42); sign-test win counts (`a`-better / `b`-better / ties); two-sided exact sign-test p-value. The four pre-registered contrasts and two post-hoc contrasts (refinement, web-vs-curated) are tabled together; multiple-comparison correction is discussed in §8.7.

| Contrast                                          | Mean Δ | 95% CI of Δ         | a-better | b-better | Ties | p (sign test) |
| ------------------------------------------------- | -----: | ------------------- | -------: | -------: | ---: | ------------: |
| **Persona vs. Analyst (E1 − E4, same briefing)**  | −0.003 | [−0.034, +0.028]    | 50 | 46 |  7 | **0.76** |
| Briefing vs. no briefing (E1 − E3, Trump)         | −0.003 | [−0.030, +0.025]    | 45 | 48 | 10 |         0.84 |
| Broad vs. compressed retrieval (E1 − E1′, Trump)  | −0.006 | [−0.021, +0.009]    | 47 | 40 | 16 |         0.52 |
<!-- DELETE: E4 − E5 contrast — voided because E5 didn't measure web search. -->
| Web search vs. curated (E4 − E5, Analyst)         | −0.005 | [−0.026, +0.015]    | 51 | 45 |  7 |         0.61 |
<!-- /DELETE -->
| Broad vs. refined retrieval (E1 − E2, Trump)      | +0.005 | [−0.026, +0.037]    | 41 | 51 | 11 |         0.35 |
<!-- DELETE: E5 − E3 contrast — voided, both conditions effectively had empty context. -->
| Trump persona on web-style (E5 − E3)              | +0.006 | [−0.013, +0.025]    | 46 | 54 |  3 |         0.32 |
<!-- /DELETE -->
<!-- DELETE: Gate vs. strict-web contrast is a comparison against an empty-context artifact, not an informative result. Keep gate-vs-real-conditions only. -->
| *post-hoc:* Gate vs. strict web (E6 − E5)         | −0.035 | [−0.070, −0.0002]   | 60 | 41 |  2 | 0.073 |
<!-- /DELETE -->
| *post-hoc:* Gate vs. no briefing (E6 − E3)        | −0.029 | [−0.078, +0.024]    | 59 | 38 |  6 | **0.042** |
| *post-hoc:* Gate vs. refined CHRONOS (E6 − E2)    | −0.020 | [−0.073, +0.033]    | 54 | 44 |  5 | 0.363 |
| *post-hoc:* Gate vs. persona (E6 − E1)            | −0.026 | [−0.070, +0.019]    | 56 | 42 |  5 | 0.189 |
| *post-hoc:* Gate vs. analyst CHRONOS (E6 − E4)    | −0.029 | [−0.067, +0.007]    | 51 | 48 |  4 | 0.841 |

**Every pre-registered 95% CI on the mean Brier delta crosses zero and every pre-registered two-sided sign-test p-value exceeds 0.30.** The pre-registered primary contrast — persona vs. analyst with matched CHRONOS broad-15 retrieval — has a mean delta of −0.003 (persona nominally better by three-thousandths of a Brier unit), 50 wins for the persona vs. 46 wins for the analyst out of 96 non-tied questions, sign-test p = 0.76. This contrast is as close to a statistical coin flip as a paired test of this sort can produce.

The five post-hoc E6 contrasts tell a consistent story: the answerability gate wins 54–60 out of ~100 non-tied matches against every other condition, with a mean Brier advantage in the 0.020–0.035 range. Only two of these reach the conventional α = 0.05 threshold on uncorrected two-sided sign tests (E6 − E3 at p = 0.042 and E6 − E5 at p = 0.073, borderline), and neither would survive Holm-Bonferroni correction across the full family of ten contrasts (§8.7). The **bootstrap CI on E6 − E5 is the only one entirely below zero** ([−0.070, −0.0002]), reflecting that relative to the strict-filter web condition — which in practice supplied essentially no context — the answerability gate separates by more than its per-question noise. The E6 − E4 row is the most interesting to sit with: matched-CHRONOS analyst and answerability-gate analyst win almost the same number of questions (51 vs. 48), but on the 99 non-tied questions E6's margins of victory are larger than E4's, so the mean Brier delta favors E6 by 0.029 even though the win count is nearly even. This is exactly the pattern we expect when full hindsight sharpens probability mass on the correct option on the subset of questions where Tavily found the outcome, and otherwise performs comparably.

### 8.5 Stratified Analysis

**By question format** (n = 103; 58 binary + 45 action):

| Cond. | Binary Brier (n=58) | Action Brier (n=45) |
| ----- | ------------------: | ------------------: |
| E1    |              0.2457 |              0.7762 |
| E1′   |              0.2523 |              0.7814 |
| E2    |              0.2717 |              0.7310 |
| E3    |              0.2779 |              0.7413 |
| E4    |              0.2370 |              0.7952 |
<!-- DELETE: E5 row in format split — empty-context artifact. -->
| E5    |              0.2474 |              0.7943 |
<!-- /DELETE -->
| E6    |          **0.2032** |              0.7723 |

The binary and action partitions invert the nominal ordering among the bounded conditions: the analyst-with-broad-CHRONOS condition (E4) has the lowest binary Brier among E1–E5, while the Trump-persona-with-refined condition (E2) has the lowest action Brier among E1–E5. No single bounded condition is dominant in both. Action Brier is ~3× binary Brier in every condition, reflecting the unnormalized multiclass scale. On the action-question subset alone, E2 beats E1 by 0.045 Brier — the largest single condition gap in any cross-section of the bounded data — but still within overlapping 95% CIs and p = 0.35 paired.

**E6 (answerability gate) dominates on the binary split but only modestly improves on the action split.** On binary, E6 reaches 0.2032 — 0.034 units below the best bounded condition (E4 at 0.2370), a 14% relative reduction. On action questions, E6 (0.7723) comes in third behind E2 (0.7310) and E3 (0.7413), two Trump-persona conditions with curated or no context. This is a strong hint about *where* full hindsight helps the model and where it doesn't: binary questions (YES / NO about a specific decision) benefit sharply when post-outcome coverage is in the prompt, because the answer is often stated directly in a retrieved article; multi-option action questions remain hard because even post-outcome sources often describe the decision at a resolution lower than the 3–5-option decision-space enumerated by our manifest, and the model must still commit probability mass among structurally similar options. The answerability gate therefore tells us the questions are *substantially* answerable on binary and *partially* answerable on multi-option action — not uniformly near-ceiling.

**By difficulty** (n = 103; difficulty labels from manifest):

| Cond. | hard (n=71) | medium (n=32) |
| ----- | ----------: | ------------: |
| E1    |      0.3395 |        0.7834 |
| E1′   |      0.3446 |        0.7917 |
| E2    |      0.3488 |        0.7464 |
| E3    |      0.3616 |        0.7439 |
| E4    |      0.3420 |        0.7890 |
<!-- DELETE: E5 row in difficulty split — empty-context artifact. -->
| E5    |      0.3466 |        0.7964 |
<!-- /DELETE -->
| E6    |      0.3164 |        0.7523 |

As in the previous run of this framework, pipeline difficulty labels invert relative to Brier: *harder* questions have *lower* Brier across every condition. This is consistent with the pipeline's labeling using base-rate-extremity (hard = low-base-rate outcome = easier to be calibrated-low on) rather than decision-space-complexity. We do not restake the "hard questions are harder to forecast" claim; condition orderings are robust across the two subsets regardless.

**By domain** (n = 103):

| Cond. | exec_orders (19) | foreign_policy (18) | legal/judicial (8) | legislative (11) | personnel (14) | public_comms (9) | trade/tariffs (24) |
| ----- | ---------------: | ------------------: | -----------------: | ---------------: | -------------: | ---------------: | -----------------: |
| E1    |           0.5075 |              0.4445 |             0.4863 |           0.5509 |         0.4245 |           0.5013 |             0.4637 |
| E1′   |           0.5207 |              0.4472 |             0.4974 |           0.5430 |         0.4293 |           0.5291 |             0.4638 |
| E2    |           0.5236 |              0.4381 |             0.5419 |           0.5199 |         0.3984 |           0.4444 |             0.4661 |
| E3    |           0.4777 |              0.4882 |             0.4748 |           0.5430 |         0.4707 |           0.4199 |             0.4781 |
| E4    |           0.5555 |              0.4564 |             0.3914 |           0.5415 |         0.4590 |           0.5375 |             0.4335 |
<!-- DELETE: E5 row in domain split — empty-context artifact. -->
| E5    |           0.5532 |              0.4474 |             0.4070 |           0.5123 |         0.4394 |           0.5664 |             0.4745 |
<!-- /DELETE -->
| E6    |           0.4999 |              0.4434 |         **0.3018** |           0.5185 |         0.3935 |           0.5655 |             0.4311 |

The only domain where one bounded condition visibly separates from the rest is `legal/judicial` (n = 8), where the analyst conditions (E4, E5) sit ~0.10 below the persona conditions. We flag this as suggestive but not statistically load-bearing at n = 8 per cell. The persona conditions outperform both analyst conditions on no domain unambiguously; across every domain the spread among E1–E5 is consistent with the tight global null. **E6 (answerability gate) is the best or near-best in every domain except `public_comms`**, with the largest advantage in `legal/judicial` (E6 at 0.302 vs. the next-best E4 at 0.391 — a 0.09 Brier gap on n = 8) and a meaningful reduction on `personnel` (E6 = 0.394 vs. E2 = 0.398 / E4 = 0.459) and `trade/tariffs` (E6 = 0.431 vs. E4 = 0.433, roughly tied). On `public_comms` (n = 9) E6 is slightly worse than E3, consistent with the observation that predictions about what Trump will *say* draw less benefit from retrieved content than predictions about what he will *do*. We do not draw contrast-level inferences from any single domain given the small per-cell n.

### 8.6 Sample-Level Variance

Mean intra-question standard deviation of p(correct) across the 5 samples, per condition:

| Cond. | Mean σ | Max σ |
| ----- | -----: | ----: |
| E1    |  0.032 | 0.136 |
| E1′   |  0.038 | 0.147 |
| E2    |  0.026 | 0.102 |
| E3    |  0.025 | 0.102 |
| E4    |  0.029 | 0.183 |
<!-- DELETE: E5 row in variance table — empty-context artifact (low variance is itself a symptom of the bug, not a property of the condition). -->
| E5    |  0.023 | 0.098 |
<!-- /DELETE -->
| E6    |  0.030 | 0.119 |

<!-- DELETE: variance commentary specifically about E5's "lowest variance" framing. Replace with a one-line note that E1, E1′ are highest-variance and E6 is comparable to E4. -->
E5 has the *lowest* mean intra-question variance, which is not evidence of retrieval quality but a near-artifact of its empty-context problem: when the prompt contains almost no search material, the model's only stochastic variation comes from the question text itself, so samples cluster tightly. E6, which actually receives substantive web-search context, has intra-sample variance comparable to E1/E4 (the CHRONOS broad-retrieval conditions) — consistent with "more context to attend to" producing more sample-to-sample swing. The Trump-persona broad-retrieval conditions (E1, E1′) remain the highest-variance of the bounded set.
<!-- /DELETE -->

### 8.7 Multiple Comparison Correction

The four pre-registered contrasts (persona, briefing-vs-none, compression, web-vs-curated) have uncorrected two-sided sign-test p-values of 0.76, 0.84, 0.52, 0.61. Holm-Bonferroni adjustment scales the largest p-value by 1, next by 2, etc.; every adjusted p-value remains ≫ 0.05. We report this for completeness; given that no uncorrected p approaches significance, the correction is trivially satisfied for the null.

The five post-hoc E6 contrasts have uncorrected p-values of 0.042, 0.073, 0.189, 0.363, 0.841 (E6 vs. E3, E5, E1, E2, E4 respectively). If we apply Holm-Bonferroni across the full family of ten contrasts (four pre-registered + two earlier post-hoc + five E6 post-hoc), the smallest p-value (E6 − E3, 0.042) is compared against α/10 = 0.005 and does *not* clear the corrected threshold. We flag this explicitly: no individual paired contrast in the evaluation, pre-registered or post-hoc, reaches conventional significance after correction. What the E6 comparisons contribute is a *consistent directional pattern* (E6 wins 5 of 5 contrasts on mean delta, and wins 54–60 of ~100 non-tied matches in each), supported by the E6 − E5 bootstrap CI excluding zero. We interpret this as moderate evidence that full hindsight nudges Brier downward rather than a significance claim.

### 8.8 What the Data Does and Does Not Support

The pre-registered claims we set out to test, and their status in this evaluation:

- **Persona framing unlocks predictive signal over matched-retrieval analyst (E1 vs. E4).** *Not supported.* Mean Δ = −0.003; 50 persona wins vs. 46 analyst wins; p = 0.76. The directions in binary and action sub-strata differ (E4 best on binary among bounded conditions, E2 best on action) but no single bounded condition wins in both.
- **CHRONOS retrieval improves over no retrieval for the persona model (E1 vs. E3).** *Not supported.* Mean Δ = −0.003; 45 with-retrieval wins vs. 48 without-retrieval wins; p = 0.84. Broad CHRONOS context was not distinguishable from zero context at this sample size.
- **Retrieval refinement (E2) improves over broad retrieval (E1).** *Directionally supported, not significant.* Mean Δ = +0.005 favoring refined; 41 broad wins vs. 51 refined wins; p = 0.35. The refined condition also used ~8× fewer input tokens than broad-15. The accuracy difference is small and within noise; the efficiency difference is ~8×.
- **Retrieval compression (E1′) is worse than broad retrieval (E1).** *Not supported.* Mean Δ = −0.006 favoring broad; 47 broad wins vs. 40 compressed wins; p = 0.52. Halving the top-k from 15 to 8 cost nothing statistically.
<!-- DELETE: This whole bullet narrating E5's failure should be condensed to a single line: "E4 vs. E5 — withdrawn; E5's strict Tavily filter emptied the prompt; see §8.1.4." Move details to a short footnote. -->
- **Live web search is at least as good as curated CHRONOS briefings for the analyst (E4 vs. E5).** *Claim void — pre-registered E5 did not measure what it was supposed to.* Mean Δ = −0.005 favoring curated; 51 curated wins vs. 45 web wins; p = 0.61. The pre-registered result is reported, but as we detail in §8.1.4 and §8.9, E5's strict Tavily date filter in practice dropped essentially all search results (Tavily rarely returns `published_date` on its sources), so E5 measured "analyst with empty context + date-reminder preamble" rather than "analyst with date-bounded web search." The number is real; the interpretation the pre-registered hypothesis required is not available from it. The post-hoc E6 contrasts (next bullet) are the informative comparison.
<!-- /DELETE -->
- **Post-hoc: answerability-gate web search (E6) does beat all bounded conditions, but only modestly.** Mean Δ vs. the five bounded conditions ranges from −0.020 (E6 vs. E2) to −0.035 (E6 vs. E5); win rates 51–60 out of ~100 non-tied matches; no single contrast clears Holm-Bonferroni correction across the full ten-contrast family (§8.7). E6 reaches binary Brier of 0.2032 (−0.034 vs. best bounded) and p(correct) of 0.46 (vs. 0.40–0.43 across bounded conditions). Full hindsight helps, but **does not approach ceiling** — a fact we return to below.

The strongest single data-driven statement this evaluation supports is unambiguous and bounded: **among the six temporally-bounded conditions, none of persona framing, CHRONOS retrieval, retrieval refinement, retrieval compression, or strict-filtered web search produced a measurable improvement over any other on Brier score.** All six bounded Brier means fall inside a 0.014-unit envelope; every pre-registered paired contrast is within 0.006 Brier units of zero; every pre-registered sign-test p-value exceeds 0.30. The one contrast that *does* produce a visible Brier gap (E6 − E5 = −0.035, CI entirely negative) is a post-hoc diagnostic comparing full-hindsight search against an empty-context filter artifact — it is not evidence that retrieval helps, it is evidence about what happens to the condition when its intended filter succeeds at removing all content.

**A second data-driven statement we do now make, which the pre-registration did not anticipate:** even with unbounded, full-hindsight web search (E6), the model reaches only p(correct) = 0.46 and mean Brier = 0.452 on this question set. The questions are genuinely hard — not unanswerable, but not near-ceiling given the 3–5-option action decision space and the model's capability tier. This provides a lower bound on the difficulty ceiling: any framework-level improvement in retrieval, refinement, or persona prompting at this model must ultimately live beneath the answerability-gate Brier (≈0.45) to have room to be attributable to the framework rather than to questions becoming easier to answer. The bounded conditions live ~0.02 Brier above E6 and ~0.02 Brier apart from each other, which is to say: the bounded-retrieval-accuracy envelope is narrower than the full-hindsight-vs-bounded gap.

Adjacent to these findings, one quantitative statement is worth making plainly because it is independent of any null-result inference: **input-token cost diverges by 17–18× across the bounded conditions while accuracy does not.** E4 consumed 12.8 million input tokens; E3 consumed 668 thousand; both produced mean Brier within 0.0004 of each other (0.4808 vs. 0.4804). E1 vs. E2 is the same story at 1/8 the tokens and 0.005 Brier penalty in E1's favor. E6 — full web-search context — uses 1.28 M input tokens (roughly 10× less than CHRONOS broad-15) and produces the lowest Brier in the evaluation. For a practitioner asking "given this model, does a CHRONOS-style curated retrieval system pay for itself over a modern web-search context?", the answer on this evaluation is *no* — but that is a cost-per-Brier-unit answer, not an effect-size claim.

### 8.9 Interpretation and Limitations

Five interpretive observations, and the limitations that bound the claims we can responsibly make.

1. **The primary hypothesis — persona framing unlocks leader-specific predictive signal beyond a matched-retrieval analyst — is not supported by this evaluation.** Point estimate is essentially zero (−0.003 Brier units); 95% CI [−0.034, +0.028]; p = 0.76; 50 of 103 questions favor the persona. We do not claim this refutes the hypothesis — a small true effect (say, ±0.02 Brier units) would need several hundred paired questions at N = 5 samples to detect — but the hypothesis should not be recommended forward from this evidence.
2. **CHRONOS retrieval did not move the Trump-persona conditions at all.** The four Trump-persona conditions (E1, E1′, E2, E3) span a Brier range of 0.012 on the common 103, smaller than any per-condition 95% CI half-width. Whether the persona received broad-15, broad-8, refined, or zero CHRONOS events, its answers are statistically indistinguishable in aggregate. Three non-exclusive mechanisms are consistent with this: (a) the persona's parametric knowledge of Trump's decision patterns is sufficient to dominate the retrieval signal at this difficulty; (b) the persona framing suppresses attention to supplied context in favor of prior beliefs; (c) retrieval content is sufficiently variable that per-sample gains and losses average out at N = 5. The present data cannot distinguish these.
<!-- DELETE: Long autopsy of E5/Tavily filter. Condense to ~3 sentences: "Pre-registered E5 turned out not to measure date-bounded web search — Tavily's `published_date` field is rarely populated, so the strict filter dropped essentially every result and E5 ran with empty context. We add E6 (unbounded) as a diagnostic and treat the E4-vs-E5 contrast as withdrawn. A genuinely date-bounded web-search evaluation requires a different provider." -->
3. **The pre-registered web-search condition (E5) did not measure web search.** The strict Tavily date filter as pre-registered — required for temporal soundness — dropped every Tavily result whose `published_date` field was missing or unparseable. During post-hoc inspection of the E5 briefings we ran a controlled audit: for a representative question (`Q-S-001-02`, sim date 2025-03-25), Tavily's API returned 20 results, *zero* with a populated `published_date`, so the strict filter dropped all 20 and the E5 prompt received the literal fallback string `(web_search returned no results dated on-or-before 2025-03-25; dropped 20 undated, 0 post-sim-date)`. This pattern was not a single-question anomaly: Tavily rarely surfaces `published_date` on its sources, especially for the kind of news-aggregator and law-firm analysis pages that dominate Trump-administration-policy query results. E5 as reported in §8.3 therefore measured *analyst-with-empty-context*, not date-bounded web search. The §8.3 Brier of 0.4863 is internally valid — the model produced well-formed structured outputs, zero errors, and low intra-sample variance because the prompt had little to destabilize it — but it is not evidence about date-bounded web retrieval for this question set. The pre-registered contrast E4 vs. E5 (web vs. curated) is accordingly voided by the null-context finding and should not be propagated as evidence about web-search performance. We did not re-design the strict-filter condition into a permissive-but-leakage-free form because any such redesign would itself be post-hoc; instead we added E6 as an explicit unbounded diagnostic.
<!-- /DELETE -->
4. **The answerability gate (E6) shows the questions are answerable but the model does not saturate them.** When the same Tavily API is called with no date filter — admitting post-outcome coverage in full — the answerability gate reaches mean Brier 0.4518 and mean p(correct) 0.4609 across the 103 questions. This is the best of all seven conditions, but it is roughly half the distance from the pre-registered nominal ceiling (Brier = 0) that one would expect if "full hindsight" and "answerable by a capable model" were equivalent. The binary split (n = 58) shows 0.2032 Brier / strong improvement; the action split (n = 45) shows 0.7723 / modest improvement. Interpretation: (a) the questions are *answerable* on binary — when Tavily surfaces a post-outcome article, the model can usually commit mass to the correct direction — and (b) the questions are *partially answerable* on 3–5-option action questions where even post-outcome sources describe the decision at a resolution that does not cleanly map to the manifest's enumerated options. The answerability gate is not a certificate of question triviality; it is a lower bound on the Brier a frontier retrieval system could achieve at this model's capability tier.
<!-- DELETE: This entire point relitigates E5 vs. earlier-run; the earlier run isn't published, so there's no replication claim to make. Drop the bullet. The provider-recommendation sentence belongs in "what we would run next." -->
5. **The earlier run's finding that web search was the best condition does not replicate, but for a subtler reason than we originally suggested.** In the earlier framework iteration, an attrition-heavy E5 ran with leaked post-sim-date content through undated results and reached the best Brier in that evaluation. In the present evaluation, E5 with a strict filter reaches the worst Brier among the bounded conditions (because the filter emptied the prompt), and E6 with no filter reaches the best (because unbounded content was admitted by design). The earlier and present finding are in fact two sides of the same underlying fact: Tavily rarely dates its results, so a "date-bounded web search with strict enforcement" is operationally empty, and a "date-bounded web search with permissive admission of undated results" is operationally a full-hindsight search. There is no regime in which Tavily, at least on this question set, delivers the clean "date-bounded search" that the E5 condition was designed to measure. A properly date-bounded web-search evaluation of this framework needs either a different search provider with reliable `published_date` metadata or a manifest-level pre-fetch and manual date-tagging step; both are out of scope for this paper.
<!-- /DELETE -->
6. **The cost-to-accuracy ratio heavily favors no retrieval and refined retrieval among bounded conditions, and favors unbounded web search overall.** E3 (0.67 M input tokens) and E2 (1.53 M) both delivered within the 0.014-unit accuracy envelope of E1 (12.7 M) and E4 (12.8 M). E6 (1.28 M) delivered the best Brier of the seven at roughly 1/10 the token cost of broad CHRONOS. For any practical deployment of this framework on this model, the broad-briefing pipeline is roughly 8×–18× the input-token cost of refined / no-retrieval / web-search alternatives for no measurable accuracy gain. We flag this as the most actionable finding from the present run.

**Limitations.**

- **Single evaluated model, deliberately mid-tier.** All six conditions run on Gemini 2.0 Flash-Lite, chosen for its pre-2025 training cutoff and quota availability rather than for frontier capability. A null on this model does not generalize to frontier reasoners — persona and retrieval effects have been shown to be capability-dependent (Kovarikova et al., 2025) — and a true effect of ~0.02–0.05 Brier units could plausibly exist on Opus 4.7 or Gemini 3 Pro and be invisible here. The pre-registered Opus 4.7 run cannot, however, share this question set: Opus 4.7's January 2026 cutoff post-dates every simulation date, so any Opus 4.7 comparison requires a new question set post-dating the Opus cutoff.
- **Sample size.** 103 resolvable questions × 5 samples gives modest power to detect small effects. The N = 5 downsample from the pre-registered N = 10 was a budget concession and reduces within-question variance precision by √2.
<!-- DELETE: This limitation bullet duplicates the §8.9 point 3 autopsy of E5. After consolidating point 3, this bullet should be removed entirely. -->
- **Strict Tavily date filter dropped essentially all signal, not merely "some."** As documented in point 3 above and §8.1.4, the pre-registered strict filter on E5 turned out to drop nearly every result because Tavily rarely returns `published_date`. The conservative bound we stated in earlier drafts ("E5's Brier could be anywhere from 0.005 better to 0.005 worse") was wrong: the correct characterization is that E5 measured a different condition than pre-registered (analyst-with-empty-context, with a date-reminder system preamble) and the evaluation therefore cannot speak to date-bounded web search on this question set. The E6 (unbounded) condition is *not* a valid substitute for a date-bounded web-search evaluation — it deliberately admits post-outcome content — it is a lower-bound answerability diagnostic.
<!-- /DELETE -->
- **Question generation pipeline asymmetry.** Some questions may systematically favor one condition in ways not detected at manifest-lock time — a question whose resolution wording appears verbatim in CHRONOS events would favor CHRONOS conditions, and vice versa. §8.5's domain-level breakdown is consistent with this concern being small but nonzero.
- **Temperature.** All conditions sample at T = 1.0, maximizing sample-to-sample variance; a lower-temperature run would produce tighter per-condition CIs at the cost of worse calibration. A T-ablation remains planned but unexecuted.
- **Resolution reliance on a single pass.** Ground-truth resolutions come from the locked `resolutions.json` produced by the two-pass protocol of §7.4; a re-audit on the adjudicated questions, or a third independent pass, would further bound resolution error. We did not re-run the resolution pipeline for this evaluation.

**What we would run next, given these results.**
(i) Port the question manifest to a post-Opus-4.7-cutoff window (questions about decisions made in or after Feb 2026) and re-run E1 vs. E4 on Opus 4.7 and Gemini 3 Pro. Frontier capability is the primary axis this evaluation does *not* cover.
(ii) Run the action-question subset at N = 10 samples, since the only single-condition gap greater than 0.03 Brier on the bounded data was on action questions (E2 beats E1 by 0.045) and resolving it requires more than our current N. The same N = 10 run on E6 would tighten the answerability-gate lower bound.
(iii) Redesign the web-search condition with a provider that returns reliable `published_date` metadata (or with a manifest-level pre-tagged corpus), so a genuinely date-bounded web-search evaluation becomes available. The present run establishes that Tavily's strict-filtered mode is not that evaluation.
(iv) Run a sensitivity version of E6 that permits undated results but hard-filters on URL-date slugs and explicit post-`simulation_date` mentions in the title/content, to see how much of E6's Brier advantage survives a coarse post-hoc date proxy — a partial replacement for (iii) using the current Tavily provider.

The evaluation infrastructure — CHRONOS, the research swarm, the two-pass resolution protocol, the question manifest, the retry-and-concurrency-hardening described in §8.2 — ran end-to-end and produced clean data on all six conditions without attrition. The primary hypothesis it was built to test is not vindicated by this run. We report the result as we pre-committed to: honestly, directionally, with every number visible to the reader, and without framing the infrastructure story as a result.

---

## References

Allison, G. T., & Zelikow, P. D. (1999). *Essence of decision: Explaining the Cuban Missile Crisis* (2nd ed.). Longman.

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis, 31*(3), 337–351.

Bisbee, J., Clinton, J. D., Dorff, C., Kenkel, B., & Larson, J. M. (2024). Synthetic replacements for human survey data? The perils of large language models. *Political Analysis*.

Bosse, M., et al. (2026). Scaling high-quality forecasting-question generation through multi-agent pipelines. [Preprint / venue TBD].

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3.

Bueno de Mesquita, B. (1985). The war trap revisited: A revised expected utility model. *American Political Science Review, 79*(1), 156–177.

Bueno de Mesquita, B. (2009). *The predictioneer's game: Using the logic of brazen self-interest to see and shape the future.* Random House.

Cederman, L.-E. (1997). *Emergent actors in world politics: How states and nations develop and dissolve.* Princeton University Press.

Chase, H., et al. (2024). LangGraph: Stateful multi-agent orchestration for language models. [LangChain technical report].

Dawid, A. P. (1984). Statistical theory: The prequential approach. *Journal of the Royal Statistical Society: Series A, 147*(2), 278–292.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378.

Goldstone, J. A., Bates, R. H., Epstein, D. L., Gurr, T. R., Lustik, M. B., Marshall, M. G., Ulfelder, J., & Woodward, M. (2010). A global model for forecasting political instability. *American Journal of Political Science, 54*(1), 190–208.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330).

Halawi, D., Zhang, F., Yueh-Han, C., & Steinhardt, J. (2024). Approaching human-level forecasting with language models. *arXiv preprint arXiv:2402.18563*.

Hermann, C. F. (1969). *International crises: Insights from behavioral research.* Free Press.

Hermann, M. G. (1980). Explaining foreign policy behavior using the personal characteristics of political leaders. *International Studies Quarterly, 24*(1), 7–46.

Hermann, M. G. (2003). Assessing leadership style: A trait analysis. In *The psychological assessment of political leaders* (pp. 178–212). University of Michigan Press.

Jervis, R. (1997). *System effects: Complexity in political and social life.* Princeton University Press.

Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica, 47*(2), 263–291.

Karger, E., Atanasov, P., Halawi, D., et al. (2025). Mantic: Difficulty-adjusted forecasting benchmarks for AI systems. [Preprint].

Katz, D. M., Bommarito, M. J., & Blackman, J. (2017). A general approach for predicting the behavior of the Supreme Court of the United States. *PLoS ONE, 12*(4), e0174698.

Kovarikova, M., et al. (2025). LLM-generated persona is a promise with a catch: Systematic biases in persona-conditioned prediction. [Preprint].

Leites, N. (1951). *The operational code of the Politburo.* RAND Corporation / McGraw-Hill.

Marfleet, B. G., & Miller, C. (2005). Bill Clinton and the two Iraq crises: Belief systems and politico-military influences on decision making. *Foreign Policy Analysis, 1*(3), 333–358.

Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology, 12*(4), 595–600.

Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well-calibrated probabilities using Bayesian binning. In *Proceedings of the 29th AAAI Conference on Artificial Intelligence* (pp. 2901–2907).

Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). Generative agent simulations of 1,000 people. *arXiv preprint arXiv:2411.10109*.

Payne, A., et al. (2024). Escalation risks from large language models in military and diplomatic decision-making. [Preprint].

Salganik, M. J., et al. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *Proceedings of the National Academy of Sciences, 117*(15), 8398–8403.

Savage, L. J. (1971). Elicitation of personal probabilities and expectations. *Journal of the American Statistical Association, 66*(336), 783–801.

Simon, H. A. (1955). A behavioral model of rational choice. *Quarterly Journal of Economics, 69*(1), 99–118.

Tetlock, P. E. (2005). *Expert political judgment: How good is it? How can we know?* Princeton University Press.

Tsebelis, G. (2002). *Veto players: How political institutions work.* Princeton University Press.

Walker, S. G., & Schafer, M. (2006). *Beliefs and leadership in world politics: Methods and applications of operational code analysis.* Palgrave Macmillan.

Walker, S. G., Schafer, M., & Young, M. D. (1998). Systematic procedures for operational code analysis: Measuring and modeling Jimmy Carter's operational code. *International Studies Quarterly, 42*(1), 175–189.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. In *Proceedings of the 11th International Conference on Learning Representations*.

Zou, E., Saxena, R., Jeong, D. P., Mavroudis, P., et al. (2024). ForecastBench: A dynamic benchmark of AI forecasting capabilities. *arXiv preprint*.

---

## Appendix A. Example Intelligence Briefing

```
═══════════════════════════════════════════════════════════
 INTELLIGENCE BRIEFING — As of July 1, 2025
 Subject: Donald J. Trump, 47th President of the United States
 Knowledge Window: January 2025 → July 1, 2025
 Events: 847 total visible | 15 retrieved for this query
═══════════════════════════════════════════════════════════

[2025-06-28] US-Canada Trade Tensions Escalate
Trump threatened to increase tariffs on Canadian lumber to 40% during
a press conference at the White House. "Canada needs to understand
that we're serious about protecting American jobs," Trump stated. Key
facts: current tariff on Canadian lumber is 25% (imposed March 2025).
Canada's Prime Minister called the threat "economic aggression" and
hinted at retaliatory measures targeting U.S. agricultural exports.
Sources: Reuters, AP, White House Press Pool (3 sources, verified date)

─────────────────────────────────────────────────────────

[2025-06-20] Executive Order on Trade Review Commission
...

═══════════════════════════════════════════════════════════
 END BRIEFING — 12 events retrieved
═══════════════════════════════════════════════════════════
```

## Appendix B. Representative Question Manifest Entry

```json
{
  "question_id": "Q-S-016-01",
  "version": "questions-v1.0",
  "domain": "personnel",
  "type": "binary",
  "text": "Will Trump issue a government-wide federal hiring freeze by Feb 15, 2025?",
  "simulation_date": "2025-01-15",
  "resolution_date": "2025-02-15",
  "resolution_criterion": "Executive order or presidential memorandum published in the Federal Register establishing a government-wide hiring freeze.",
  "resolution_source_class": "Federal Register",
  "difficulty": "medium",
  "base_rate": 0.40,
  "base_rate_rationale": "Trump issued a hiring freeze on Day 2 of his first term (January 23, 2017).",
  "analogous_market": null,
  "sources_at_seed": ["..."]
}
```

## Appendix C. Agent Specifications (Abridged)

Full specifications for the seven research-swarm agents appear in `temporal_knowledge_base/build_prompts/01_scaffolding.md` through `09_end_to_end_verification.md` in the project repository. Build order and interdependencies are as specified in Section 4.5 and summarized in the repository's `build_prompts/README.md`.

---

*This draft presents the problem formulation, system architecture, experimental design, and methodology of the CHRONOS framework. Experimental results are the subject of a companion paper; the evaluation protocol specified here is pre-registered in advance of any prediction elicitation.*
