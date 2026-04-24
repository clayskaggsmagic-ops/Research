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

We report results from a six-condition evaluation on 103 resolvable questions × 5 samples per question × 6 conditions (3,090 predictions total) using Gemini 2.5 Flash as the evaluated model. **Across every pre-registered paired contrast — persona vs. matched-retrieval analyst, persona with briefing vs. persona without, broad vs. refined retrieval, broad vs. compressed retrieval — no difference reached statistical significance at the conventional α = 0.05 threshold under paired sign tests with sample-size-appropriate bootstrap confidence intervals.** The primary pre-registered contrast — persona vs. analyst with matched CHRONOS retrieval — produced a mean Brier difference of +0.028 (persona worse; 95% bootstrap CI [-0.013, +0.071]; two-sided sign test p = 0.59), in the *opposite* direction from the persona-helps hypothesis. The lowest raw mean Brier was obtained by the Analyst × live web search condition (0.44; 95% CI [0.34, 0.54]), which also suffered 23.9% attrition from API timeouts and malformed outputs — a genuine characteristic of the condition, not a sampling accident. The direction of effects and the tight clustering of Brier means across the four persona conditions (range: 0.51–0.54) is consistent with the strong null: identity-assertion framing and curated retrieval did not, in this evaluation, measurably improve forecasting of Donald Trump's real-world decisions over a careful-analyst baseline with matched context. We discuss the result honestly, including its implications for the feasibility of leader-as-persona forecasting, the interpretation of partial E4 attrition, and the constraints imposed by a single-model family, modest sample size, and mid-tier model evaluation.

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

- **Primary evaluated model**: Claude Opus 4.7 (anthropic.com/claude-opus-4-7), pinned in the project configuration file.
- **Secondary models (planned robustness checks)**: GPT-4o, Gemini 3 Pro, Claude Sonnet 4.6.
- **Discovery/validation models (swarm-internal, not under evaluation)**: Gemini 3 Flash for high-throughput operations; Gemini 3 Pro and Opus 4.7 for validation and logical-consistency checks.

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

1. **Evaluated model.** The pre-registered primary evaluated model was Claude Opus 4.7. Due to a hard budget ceiling on the present run we instead evaluated **Gemini 2.5 Flash** (`gemini-2.5-flash`). This is a mid-tier cost-optimized model rather than a frontier reasoner. Any null result reported below is accordingly a null result *on Gemini 2.5 Flash*, not on all frontier LLMs. A secondary evaluation on Opus 4.7 remains a planned robustness check.
2. **Samples per question.** The pre-registered sample count was N = 10 per (condition, question). We ran **N = 5**, also for budget reasons. This reduces within-question variance estimation precision by √2 but does not affect the unbiasedness of per-question Brier estimates.
3. **Sixth condition (E2′).** The refined-retrieval condition E2′ was added after pre-registration as a cost-quality ablation. We report it alongside the pre-registered conditions but exclude it from the primary contrast.

The question manifest and resolutions were frozen before any predictions were elicited. All 3,090 predictions were written to append-only JSONL with per-record prompt hash, briefing hash, model ID, and timestamps, and are archived in the snapshot `evaluation_plan/output/snapshots/run_20260423T154710Z/`.

### 8.2 Question Set and Attrition

Of the 115 pre-registered questions, **103 were resolvable** by the two-pass resolution protocol within the evaluation window (12 questions did not yet have a ground-truth outcome determinable from their specified document class). Resolutions were locked in `pipeline/output/resolutions/resolutions.json` before scoring.

Record counts per condition (target: 515 = 103 questions × 5 samples):

| Cond.  | Description                         | Records | Valid | Errors | Input tokens | Output tokens |
| ------ | ----------------------------------- | ------: | ----: | -----: | -----------: | ------------: |
| E1     | Persona × CHRONOS broad-15          |     515 |   515 |      0 |   10,317,240 |        73,010 |
| E1′    | Persona × CHRONOS broad-8           |     515 |   515 |      0 |    6,208,970 |        71,542 |
| E2′    | Persona × CHRONOS refined           |     515 |   515 |      0 |    1,252,595 |        70,956 |
| E2     | Persona × no retrieval              |     515 |   515 |      0 |      659,050 |        69,389 |
| E3     | Analyst × CHRONOS broad-15          |     515 |   514 |      1 |   11,373,489 |        99,678 |
| E4     | Analyst × live web search (Tavily)  |     515 |   392 |    123 |    2,033,852 |        81,285 |

Five of the six conditions completed with zero or one errors. E4 (live web search) produced **123 invalid records (23.9%)** — 92 Gemini `504 Deadline Exceeded` timeouts on long Tavily-augmented contexts and 31 pydantic validation failures arising from Gemini returning all-zero probability vectors on multiple-choice questions. These are genuine characteristics of the E4 condition (long noisy web-retrieved context appears to destabilize Gemini 2.5 Flash's structured-output behavior), not transient infrastructure failures: when we verified DNS health and rechecked mid-run, the errors continued to appear on specific questions. We accordingly report E4 on its 88-question subset where at least one sample succeeded and flag the attrition prominently.

Fifteen questions have zero valid E4 samples. Scoring all other conditions on *just those 15 questions* yields mean Brier within ±0.03 of their global mean (E1: +0.034; E2: −0.012; E3: −0.013), so the E4-missing set is not systematically easy or hard for the other conditions — the survivorship-bias contamination of E4's apparent advantage is small but nonzero. For fair comparisons we restrict all cross-condition tests to the **88 questions for which every condition has at least one valid sample**.

### 8.3 Primary Metric: Mean Brier Score (Common 88 Questions)

Per-question Brier is computed as the mean over surviving samples for that (condition, question); per-condition aggregate is the mean over the 88 common questions. Binary Brier ∈ [0, 1]; multiclass Brier (unnormalized, 3–5 option action questions) ∈ [0, 2]; the mix in the common subset is 50 binary and 38 action questions. 95% bootstrap confidence intervals are from 10,000 paired resamples (seed 42).

| Cond. | Description                 | Mean Brier | 95% CI           | Mean p(correct) |
| ----- | --------------------------- | ---------: | ---------------- | --------------: |
| E1    | Persona × CHRONOS broad-15  |     0.5114 | [0.4163, 0.6093] |          0.4669 |
| E1′   | Persona × CHRONOS broad-8   |     0.5332 | [0.4379, 0.6331] |          0.4497 |
| E2′   | Persona × CHRONOS refined   |     0.5208 | [0.4280, 0.6178] |          0.4579 |
| E2    | Persona × no retrieval      |     0.5377 | [0.4449, 0.6336] |          0.4423 |
| E3    | Analyst × CHRONOS broad-15  |     0.4830 | [0.4007, 0.5702] |          0.4681 |
| E4    | Analyst × web search        | **0.4409** | [0.3430, 0.5427] |      **0.5750** |

All six confidence intervals overlap. The separation between the nominal best (E4, 0.441) and nominal worst (E2, 0.538) is ~0.10 in Brier units on a multiclass scale that runs to 2.0 — small relative to within-condition CI widths (~0.20 each).

### 8.4 Pre-Registered Paired Contrasts

All contrasts are computed per-question on the common 88-qid subset. Reported: mean signed Brier delta (positive = condition `a` worse); 95% bootstrap CI on the mean delta; sign-test win counts (`a`-better / `b`-better / ties); two-sided exact sign-test p-value. No multiple-comparison correction is applied to the table (Holm-Bonferroni was pre-registered; see §8.7 for the corrected version).

| Contrast                                 | Mean Δ | 95% CI of Δ           | a-better | b-better | Ties | p (sign test) |
| ---------------------------------------- | -----: | --------------------- | -------: | -------: | ---: | ------------: |
| **Persona vs. Analyst (E1 − E3, same briefing)**    | +0.028 | [−0.013, +0.071]      | 41 | 47 | 0 | **0.59** |
| Briefing vs. no briefing (E1 − E2, Trump persona)   | −0.026 | [−0.064, +0.004]      | 47 | 39 | 2 |         0.45 |
| Broad vs. refined retrieval (E1 − E2′, Trump)       | −0.009 | [−0.040, +0.020]      | 40 | 45 | 3 |         0.66 |
| Broad vs. compressed retrieval (E1 − E1′, Trump)    | −0.022 | [−0.058, +0.007]      | 40 | 43 | 5 |         0.83 |
| Web search vs. CHRONOS (E3 − E4, analyst)           | +0.042 | [−0.038, +0.123]      | 37 | 51 | 0 |         0.17 |

**Every contrast's 95% CI on the mean Brier delta crosses zero.** Every two-sided sign-test p-value exceeds 0.15. The pre-registered primary contrast — persona vs. analyst with matched retrieval, E1 − E3 — not only does not reach significance, its direction is the opposite of the persona-helps hypothesis: the analyst condition (E3) produced a *lower* mean Brier than the matched-retrieval persona condition (E1), 0.483 vs. 0.511, with 47/88 questions favoring the analyst and 41/88 favoring the persona.

### 8.5 Stratified Analysis

**By question format** (common 88):

| Cond. | Binary Brier (n=50) | Action Brier (n=38) |
| ----- | ------------------: | ------------------: |
| E1    |              0.3139 |              0.7712 |
| E1′   |              0.3342 |              0.7951 |
| E2′   |              0.3255 |              0.7777 |
| E2    |              0.3231 |              0.8200 |
| E3    |              0.2774 |              0.7535 |
| E4    |          **0.2191** |          **0.7326** |

Rankings are consistent across the two question-format partitions: E4 best, E3 second, Trump-persona conditions cluster below. Action-question Brier is ~2.3× binary-question Brier in every condition, reflecting the (unnormalized) multiclass scale difference.

**By difficulty** (common 88; difficulty labels from manifest):

| Cond. | hard (n=61) | medium (n=27) |
| ----- | ----------: | ------------: |
| E1    |      0.4116 |        0.7369 |
| E1′   |      0.4228 |        0.7828 |
| E2′   |      0.4078 |        0.7761 |
| E2    |      0.4124 |        0.8208 |
| E3    |      0.3698 |        0.7387 |
| E4    |      0.3171 |        0.7205 |

The difficulty labels assigned by the question-generation pipeline invert relative to Brier: *harder* questions have *lower* Brier across every condition. This is consistent with the pipeline's labeling using base-rate-extremity (hard = low base rate for the true outcome = easier to be calibrated-low on) rather than decision-space-complexity. We do not restake the "hard questions are harder" claim on the data; the per-condition ordering is robust across the two subsets regardless.

**By domain** (common 88):

| Cond. | executive_orders (18) | foreign_policy (12) | legal/judicial (8) | legislative (11) | personnel (6) | public_comms (9) | trade/tariffs (24) |
| ----- | --------------------: | ------------------: | -----------------: | ---------------: | ------------: | ---------------: | -----------------: |
| E1    |                0.5655 |              0.6280 |             0.5089 |           0.4781 |        0.4057 |           0.3831 |             0.5031 |
| E1′   |                0.5786 |              0.6027 |             0.5402 |           0.5466 |        0.4320 |           0.3874 |             0.5360 |
| E2′   |                0.5838 |              0.6315 |             0.5459 |           0.5138 |        0.3902 |           0.3494 |             0.5099 |
| E2    |                0.5651 |              0.6323 |             0.5052 |           0.5317 |        0.4920 |           0.3396 |             0.5691 |
| E3    |                0.5009 |              0.5100 |             0.4777 |           0.4422 |        0.4624 |           0.5085 |             0.4720 |
| E4    |                0.4839 |              0.5156 |         **0.2499** |           0.4492 |        0.3075 |           0.5315 |             0.4304 |

E4's largest single advantage is in `legal/judicial` (0.25 vs. E1's 0.51), where live web search presumably surfaces post-decision case names and rulings that CHRONOS had not yet harvested for those questions. Conversely, `public_comms` is the only domain where the persona conditions collectively outperform both analyst conditions — a domain where stylistic and rhetorical priors might plausibly help the persona. We flag these as suggestive cross-domain patterns without making statistical claims about them at n = 6–24 per cell.

### 8.6 Sample-Level Variance

Mean intra-question standard deviation of p(correct) across the 5 samples, per condition:

| Cond. | Mean σ | Max σ |
| ----- | -----: | ----: |
| E1    | 0.068 | 0.271 |
| E1′   | 0.058 | 0.265 |
| E2′   | 0.064 | 0.337 |
| E2    | 0.062 | 0.287 |
| E3    | 0.086 | 0.311 |
| E4    | 0.127 | 0.424 |

E4's intra-question variance is roughly twice the other conditions'. Inspecting the within-question sample trajectory for high-variance cases shows that Tavily's retrieved results change subtly across samples even at fixed query, and Gemini responds with strongly sample-dependent probabilities. This is a known failure mode of web-grounded forecasting and it cuts both ways: high variance means E4's apparent Brier advantage is less robust to sample size than the persona-conditions' Brier.

### 8.7 Multiple Comparison Correction

Applying Holm-Bonferroni to the four pre-registered contrasts (persona, briefing, compression, web-vs-curated) — excluding E2′ as post-hoc — yields no change in substantive conclusion: every uncorrected p-value already exceeds 0.15, so Holm-Bonferroni-adjusted α is trivially satisfied for the null. We do not report adjusted p-values individually because, in the absence of any uncorrected p < 0.05, the correction is vacuous.

### 8.8 What the Data Does and Does Not Support

The pre-registered claims we set out to test, and their status in this evaluation:

- **Persona framing unlocks predictive signal over matched-retrieval analyst.** *Not supported.* Point estimate was in the opposite direction; CI crossed zero; p = 0.59.
- **CHRONOS retrieval improves over no retrieval for the persona model.** *Directionally supported, not significant.* Mean Brier was lower with retrieval by 0.026 Brier units; p = 0.45; 47/88 questions favored retrieval.
- **Retrieval refinement (E2′) improves over broad retrieval.** *Not supported.* Refined retrieval was statistically indistinguishable from broad top-15 on forecasting quality despite using ~1/8 the input tokens. The cost-efficiency case for refinement survives; the accuracy case does not.
- **Retrieval compression (E1′) is worse than broad retrieval.** *Directionally supported, not significant.* Compressed top-8 had Brier 0.022 higher than broad top-15; p = 0.83.
- **Live web search is at least as good as curated CHRONOS briefings for the analyst model.** *Directionally supported, not significant, with 23.9% attrition.* E4's nominal advantage over E3 is 0.042 Brier units favoring web search; p = 0.17.

The strongest single statement the data support is the converse of our originating hypothesis: **in this evaluation, on this question set, with this model, a leader-neutral careful-analyst framing tied 41–47 against the full persona framing on matched retrieval, and the analyst with live web search produced the lowest nominal mean Brier of any condition.**

### 8.9 Interpretation and Limitations

We draw four interpretive observations from these results, and list the limitations that bound the claims we can responsibly make from them.

1. **The main hypothesis — that LLM persona conditioning unlocks leader-specific predictive signal beyond what a careful analyst framing achieves with matched retrieval — is not supported by this evaluation.** We do not claim it is refuted, because the effect's 95% CI spans [−0.013, +0.071] and sample size (N = 88 questions × 5 samples = 440 per-sample observations, or n = 88 at the per-question level) leaves the door open to a small true effect. But the direction of the point estimate favors analyst, not persona, and if there is a persona effect it must be small enough to be invisible at our power. This was a pre-registered possible outcome; we pre-committed to publishing it.
2. **CHRONOS retrieval did not meaningfully separate the persona conditions from each other.** The four persona conditions (E1, E1′, E2′, E2) had Brier means in a range of 0.026 — smaller than within-condition sample variance. Whether the persona model received broad-15, broad-8, refined, or zero CHRONOS events, its answers were approximately the same, on average. Either (a) the persona's parametric knowledge of Trump's decision patterns is sufficient to dominate the retrieval signal, or (b) the persona framing suppresses attention to supplied context in favor of prior beliefs, or (c) retrieval content is sufficiently variable that per-sample gains and losses average out — we cannot distinguish these hypotheses with the present data.
3. **The live-web-search condition's apparent advantage is fragile.** E4 had the lowest raw Brier but also (i) 23.9% attrition from genuine condition-specific failures, (ii) 2× the intra-question sample variance of other conditions, and (iii) a CI on its advantage over E3 that spans [−0.038, +0.123]. The web-search-helps story is a plausible hypothesis the data are *consistent with*, not one the data *establish*.
4. **On the E1 vs. E3 pre-registered contrast, the persona condition and the analyst condition behaved similarly across the question set** — 41 questions favored the persona, 47 favored the analyst, 0 ties. The paired distribution is close to a coin flip on which framing wins a given question. This does not support either framing as the better choice; it supports neither framing as reliably better than the other with matched retrieval, at this sample size.

**Limitations.**

- **Single evaluated model.** All six conditions use Gemini 2.5 Flash. Effects of persona framing have been shown to be model-family-dependent (Kovarikova et al., 2025). A null on Gemini 2.5 Flash does not generalize to frontier reasoners; a larger effect could plausibly exist on Opus 4.7 or GPT-5 and be worth re-running the evaluation to detect.
- **Sample size.** 103 resolvable questions × 5 samples gives modest statistical power to detect small effects. A true persona effect of ~0.01–0.02 Brier units would require several hundred questions to detect reliably. The pre-registered plan was N = 10 samples; we ran N = 5.
- **E4 attrition.** 15 questions (14.6%) have no valid E4 sample; E4 conclusions rest on an 88-qid subset. While our sensitivity check (§8.2) suggests the E4-missing set is not systematically harder, we cannot rule out small survivorship-bias effects on the E4 rankings.
- **E4 temporal leakage (Tavily).** The live-web-search condition filters Tavily results by `published_date ≤ simulation_date` where that field is present, but many Tavily results lack a parseable published_date and are therefore not filtered. Manual inspection of E4 reasoning text on individual questions confirmed occasional references to events post-dating the simulation date. The effect of this on E4 is not a one-way advantage: we observed E4 both (a) correctly citing post-sim-date events that would have revealed the ground truth, and (b) confabulating plausible-sounding but incorrect post-sim-date events that led it to the wrong answer. The net direction on mean Brier is ambiguous.
- **Question generation pipeline asymmetry.** Some questions may systematically favor one condition (e.g., questions whose resolution appears verbatim in web search results but not in the CHRONOS knowledge base) in ways not detected at manifest-lock time.
- **Temperature.** Both persona and analyst were sampled at T = 1.0, which is the default temperature for Gemini 2.5 Flash and maximizes sample-to-sample variance. A temperature ablation is planned.

**What we would run next, given these results.** First, re-run E1 vs. E3 specifically on a frontier reasoner (Claude Opus 4.7) to test whether the null on Gemini 2.5 Flash generalizes or is model-specific. Second, expand the action-selection question set, since per-format scores cluster tightly in binary questions and diverge more in action questions — suggesting that measurable condition effects may be concentrated in the richer outcome space. Third, instrument E4 with a strict sim-date filter that drops Tavily results lacking a published_date rather than admitting them, and re-score to bound the temporal-leakage effect.

The system — CHRONOS as implemented, the research swarm, the two-pass resolution protocol, the question manifest — ran end-to-end without methodological problems at the level of the evaluation infrastructure. The hypothesis it was built to test was not vindicated by this run. We report the result as we pre-committed to: honestly, directionally, with every number visible to the reader, in advance of any re-run on a frontier model.

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
