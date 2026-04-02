# Question Generation Framework

> **Purpose**: A systematic methodology for generating, validating, and scoring evaluation questions for the LLM leader prediction experiments. This document answers: *How do we find the right questions to ask?*Th

---

## The Problem

The hardest part of evaluating LLM leader prediction isn't scoring — it's *question selection*. Bad questions produce misleading results. A question that's too easy ("Will Trump continue being President tomorrow?") inflates accuracy. A question that's too hard or ambiguous ("Will Trump's tariff policy succeed?") is unresolvable. A biased question set (all tariff questions, no personnel questions) tests one narrow domain and tells you nothing about general predictive ability.

The existing literature solved this problem three different ways:

| System | Method | Strengths | Weaknesses |
|---|---|---|---|
| **ForecastBench** (Zou et al., 2024) | Template-based auto-generation from time-series data + scraping prediction markets | Produces 500+ questions per round, fully automated, reproducible | Questions are generic ("Will X metric exceed Y by Z date?"), not leader-specific |
| **Bosse et al.** (2026) | Multi-agent LLM pipeline: agents suggest → refine → critique questions, with web-grounded resolution | 96% unambiguous rate, 3.9% annulment (better than Metaculus's 8%), scales to 1500+ questions | Expensive to run, requires careful prompt engineering |
| **Metaculus** (human curation) | Expert humans write questions with detailed resolution criteria, peer review | Highest quality individual questions, rich context | Doesn't scale, human bottleneck, 8% annulment rate |

Our system needs to combine the best of all three: **the scalability of ForecastBench, the multi-agent validation of Bosse et al., and the resolution rigor of Metaculus** — adapted specifically for leader-decision prediction.

---

## Part 1: The Philosophy — What Makes a Good Question?

Every question we ask must pass a **6-gate filter** before it enters the evaluation set. If it fails any gate, it's rejected.

### Gate 1: Resolvability

**The question must have a clear, unambiguous yes/no answer or a definite action taken, verifiable from public sources.**

- ✅ "Will Trump sign an executive order imposing tariffs above 20% on Brazilian goods before September 1, 2027?"
  - *Resolution source*: Federal Register, USTR tariff schedule
- ❌ "Will Trump's trade policy hurt the economy?"
  - *Why it fails*: "Hurt the economy" is subjective and unmeasurable within a fixed timeframe

**Resolution criteria must specify:**
1. The exact outcome being predicted (what happens)
2. The resolution date (by when)
3. The authoritative source that will determine the answer (who says so)
4. Edge case handling (what if the action is partially taken, or the source is unavailable?)

This follows the Metaculus standard: questions that are "underspecified or circumvented" resolve as "ambiguous" and are excluded from scoring.

### Gate 2: Post-Training-Date

**The event being asked about must occur after the LLM's training data cutoff.**

This is the single most important methodological constraint, taken directly from ForecastBench (Zou et al., 2024). If the LLM has already "seen" the event in training data, it's doing *retrieval*, not *prediction*. We're testing forecasting ability, not memory.

**Practical implication**: Before finalizing any question, check the model's knowledge cutoff date. For current frontier models:
- GPT-4o: ~October 2023 (with web search potentially more recent)
- Gemini 3.0: ~Early 2025
- Claude 4: ~Early 2025

Questions about events in the LLM's training window must be excluded, no matter how good they are.

### Gate 3: Leader-Attributable

**The outcome must be a decision made by the specific leader, not by a system, institution, or external force.**

- ✅ "Will Trump publicly criticize the Federal Reserve Chair within 7 days of the next rate decision?"
  - *Trump makes this decision personally*
- ❌ "Will the US GDP grow above 3% in Q3 2027?"
  - *Why it fails*: GDP growth is determined by millions of actors, not by Trump's individual decision

This gate is what distinguishes our experiment from generic forecasting. Every question must be traceable to an individual decision or action by the named leader.

### Gate 4: Measurability

**The outcome must be clearly distinguishable as having occurred or not. No partial credit, no judgment calls.**

This is where we specify what counts as "evidence":

| Decision Type | Acceptable Evidence Sources |
|---|---|
| Executive orders signed | Federal Register (federalregister.gov) |
| Tariff changes | USTR announcements, HTS schedule updates |
| Personnel changes | Official White House announcements |
| Public statements | Video/audio transcript from official events, Truth Social posts (archived) |
| Military actions | DoD press briefings, official orders |
| Vetoes / signatures | Congressional Record |
| Sanctions | OFAC SDN List updates |

### Gate 5: Difficulty Calibration

**The question should not be trivially predictable or essentially random.**

Questions exist on a difficulty spectrum. We need a balanced mix:

| Difficulty Level | Base Rate | Example | Role in Evaluation |
|---|---|---|---|
| **Easy** (base rate >80%) | Very likely to happen | "Will Trump make a public statement about trade within 30 days?" | Tests that the system doesn't fail on obvious predictions. Floor check. |
| **Medium** (base rate 30-70%) | Genuinely uncertain | "Will Trump impose new tariffs on a country not currently under tariff threat in the next 90 days?" | The sweet spot — where real predictive value shows up. |
| **Hard** (base rate <20%) | Unlikely but possible | "Will Trump withdraw the US from a major international agreement in the next 6 months?" | Tests for overconfidence, false positive rate. |

The target distribution is approximately: **20% easy, 60% medium, 20% hard.**

This mirrors the difficulty-adjustment methodology from Mantic (Karger et al., 2025), which weights scores by question difficulty so that getting a hard question right counts more than getting an easy question right.

### Gate 6: Topic Diversity

**The question set must cover all major decision domains, not just the leader's most publicized actions.**

For Trump, the decision domains are:

| Domain | Weight | Why This Weight |
|---|---|---|
| Trade & Tariffs | 25% | Highest-frequency measurable decisions, extensive public record |
| Executive Orders & Regulatory Actions | 20% | Formally published, unambiguous resolution |
| Personnel & Appointments | 15% | Frequent, clearly resolvable, reveals management style |
| Foreign Policy & Diplomacy | 15% | High stakes, tests strategic reasoning |
| Legislative Interactions (veto/sign/lobby) | 10% | Less frequent but clearly resolvable |
| Public Communications (Truth Social, rallies) | 10% | Tests persona accuracy more than policy prediction |
| Legal & Judicial Responses | 5% | Reactive decisions, tests crisis response modeling |

**Why weights matter**: If 80% of our questions are about tariffs, we're measuring "can the LLM predict tariff decisions" — not "can the LLM predict *Trump.*" The weights ensure we test across the full range of presidential decision-making.

---

## Part 2: The Pipeline — How to Generate Questions at Scale

### Architecture: The Question Swarm

Inspired by the Bosse et al. (2026) multi-agent pipeline and adapted for leader-specific prediction. The swarm consists of four specialized agents, each with a distinct role:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   SCANNER   │────▶│  DRAFTER    │────▶│   CRITIC    │────▶│  RESOLVER   │
│             │     │             │     │             │     │             │
│ Finds raw   │     │ Turns events│     │ Stress-tests│     │ Writes final│
│ decision    │     │ into well-  │     │ questions   │     │ resolution  │
│ opportunities│    │ formed      │     │ for all 6   │     │ criteria &  │
│ from data   │     │ questions   │     │ gates       │     │ scores      │
│ sources     │     │             │     │             │     │ difficulty  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

#### Agent 1: The Scanner

**Job**: Continuously monitor structured data sources to identify *upcoming decision points* — moments where the leader will have to make a choice.

**Data sources the Scanner watches:**

| Source | What It Provides | Update Frequency | URL / API |
|---|---|---|---|
| Federal Register | New executive orders, proclamations, rules | Daily | federalregister.gov/api |
| USTR.gov | Tariff announcements, trade investigations, Section 301/232 actions | Weekly | ustr.gov |
| Congressional schedule | Bills approaching the President's desk (veto/sign decisions) | Weekly | congress.gov |
| White House briefing schedule | Press conferences, bilateral meetings, state visits | Daily | whitehouse.gov |
| OFAC sanctions list | New sanctions designations | As-needed | ofac.treasury.gov |
| Prediction markets | Existing questions about Trump decisions (as difficulty benchmarks) | Real-time | Kalshi API, Polymarket API |
| Wire services (AP, Reuters) | Breaking events that will demand a presidential response | Continuous | News APIs |

**How the Scanner generates "decision seeds":**

The Scanner doesn't write questions — it identifies *decision opportunities*. A decision opportunity is a situation where:
1. An external event has occurred (or is scheduled to occur)
2. That event falls within the leader's decision authority
3. The leader has not yet acted
4. There are at least 2 plausible responses

**Example Scanner output:**
```json
{
  "seed_id": "SCAN-2027-0142",
  "event": "EU announces retaliatory tariffs on US agricultural exports, effective June 1",
  "source": "Reuters wire, March 15 2027",
  "decision_domain": "trade_tariffs",
  "leader": "Trump",
  "plausible_actions": [
    "Escalate with counter-tariffs on EU goods",
    "Threaten tariffs but delay implementation",
    "Negotiate directly with EU leadership",
    "Take no immediate action",
    "Issue public statement but no policy change"
  ],
  "expected_resolution_window": "30-90 days",
  "data_availability": "USTR announcements, Federal Register"
}
```

#### Agent 2: The Drafter

**Job**: Takes Scanner seeds and converts them into properly formatted questions — one binary (yes/no) question and one multiple-choice (action selection) question per seed.

**The Drafter follows a rigid template:**

For binary questions:
```
TITLE: [Clear, specific claim about a future action]
QUESTION: Will [Leader] [specific action] [specific target] [by/before specific date]?
BACKGROUND: [2-3 sentences of factual context from the Scanner's source]
RESOLUTION: Resolves YES if [exact condition]. Resolves NO if [exact condition]. 
SOURCE: [Authoritative resolution source]
FINE PRINT: [Edge cases — what if only partially done? What if the action is 
taken but then reversed? What if the leader acts through a subordinate?]
```

For action selection:
```
SCENARIO: On [date], [event description with factual context].
OPTIONS:
A) [Specific action 1 — concrete, verifiable]
B) [Specific action 2]
C) [Specific action 3]
D) [Specific action 4]
E) None of the above / no public action within [timeframe]
NOTE: Options must be mutually exclusive and collectively exhaustive.
```

**Drafting rules:**
- Every binary question must have a specific date or timeframe
- Every option in action selection must be independently verifiable
- Every question must include an "Option E: None of the above" to prevent forced choice artifacts
- The Drafter must label each question with its expected difficulty (easy/medium/hard) and domain category

#### Agent 3: The Critic

**Job**: Adversarial validation. The Critic tries to *break* every question the Drafter produces by testing it against all 6 gates.

**The Critic asks:**

| Gate Check | The Critic's Test | Red Flag |
|---|---|---|
| Resolvability | "Can I imagine a scenario where reasonable people would disagree on whether this question resolved YES or NO?" | If yes → send back for revision or reject |
| Post-training | "Could the LLM possibly have seen the answer to this in training data?" | If plausible → reject |
| Leader-attribution | "Is this really *the leader's decision*, or is it determined by a bureaucracy, market, or external actor?" | If attribution is unclear → reject |
| Measurability | "Can I find the resolution source within 24 hours of the event occurring?" | If evidence is hard to find → send back for revision |
| Difficulty | "Is the answer obvious to anyone who reads the news?" or "Is this essentially a coin flip with no signal?" | If floor or ceiling → rebalance batch |
| Diversity | "Does this question overlap with existing questions in the batch on this topic?" | If too similar → defer or merge |

**Key innovation from Bosse et al. (2026)**: The Critic is a *different LLM instance* (or ideally a different model entirely) from the Drafter. This prevents "self-agreement bias" — the tendency for an LLM to approve its own output. Bosse et al. achieved a 96% unambiguous rate with this adversarial structure vs. ~92% for human-curated Metaculus questions.

#### Agent 4: The Resolver

**Job**: Pre-computes resolution metadata for validated questions. Does *not* resolve the question (that happens after the event) — instead, it:

1. **Sets the resolution watchlist**: Specific URLs, APIs, or RSS feeds that will be checked for resolution evidence
2. **Scores difficulty**: Computes an estimated base rate using historical frequency analysis
3. **Assigns domain tags**: Maps to the 7-domain taxonomy
4. **Generates the "prediction market benchmark"**: Searches Kalshi and Polymarket for analogous questions and records their current price as a baseline

### Pipeline Output: The Question Manifest

Every question that passes all 4 agents gets added to a **Question Manifest** — a structured JSON dataset that gets version-controlled and pre-registered before any LLM predictions are collected.

```json
{
  "manifest_version": "1.0",
  "pre_registration_date": "2027-04-01",
  "model_under_test": "gemini-3.0-flash-preview",
  "training_cutoff": "2025-03-01",
  "leader": "Donald J. Trump",
  "questions": [
    {
      "id": "Q-001",
      "type": "binary",
      "domain": "trade_tariffs",
      "difficulty": "medium",
      "base_rate_estimate": 0.55,
      "title": "Will Trump impose tariffs above 20% on Brazilian goods before September 1, 2027?",
      "background": "On July 30, 2025, Trump signed an executive order imposing a 40% tariff on certain Brazilian products, citing threats to US interests...",
      "resolution_criteria": "Resolves YES if USTR tariff schedule shows a tariff rate above 20% on any category of Brazilian goods. Resolves NO if no such tariff is published by the resolution date.",
      "resolution_source": "ustr.gov, Federal Register",
      "resolution_date": "2027-09-01",
      "prediction_market_benchmark": {
        "source": "Kalshi",
        "price_at_creation": 0.42,
        "recorded_date": "2027-04-01"
      },
      "created_by": "Drafter v1.2",
      "validated_by": "Critic v1.1",
      "validation_date": "2027-04-01"
    }
  ]
}
```

---

## Part 3: How to Generate VICS Questions (Experiment 2)

VICS questions are fundamentally different from binary/action-selection questions. We're not asking "will X happen?" — we're asking "what will the leader *say* about X, and will the cognitive fingerprint of that speech match reality?"

### The VICS Question Selection Criteria

A good VICS evaluation event must satisfy three conditions:

1. **The leader gave a substantial public response** (≥200 words of spontaneous speech, not a prepared written statement read from a teleprompter — those are drafted by speechwriters and don't reflect the leader's personal operational code)
2. **The response is available in transcript form** (video/audio transcripts from press conferences, rally speeches, interviews, or spontaneous remarks)
3. **The event triggered a *change* in the leader's VICS profile** — i.e., the specific VICS scores for that speech differ meaningfully from the leader's historical average

Point 3 is crucial: if every speech produces the same VICS scores as the average, then the "baseline = average" strategy always wins, and the experiment is pointless. We specifically want events that *push the leader off their baseline* — crises, surprises, confrontations, victories.

### A Taxonomy of VICS-Triggering Events

Based on VICS research (Walker & Schafer) and the existing Trump profiling literature, these event types are most likely to produce distinctive VICS shifts:

| Event Type | Expected VICS Shift | Why | Example |
|---|---|---|---|
| **Direct personal attack** (foreign leader insults Trump personally) | P-1 drops sharply (world becomes hostile), I-1 shifts conflictual, I-5 shifts to "threaten" | Trump's VICS profile is known to become more hostile when personally attacked | A foreign leader mocks Trump at a press conference |
| **Trade victory** (trade deal signed, concessions won) | P-1 rises (world is friendly), P-2 rises (values are being realized), I-1 shifts cooperative | Success validates Trump's deal-making self-concept | Country agrees to reduce tariffs after Trump's threats |
| **Military crisis** (attack on US assets, ally under threat) | P-3 drops (world becomes unpredictable), I-3 rises (risk tolerance increases), I-5 shifts toward "punish"/"threaten" | Crises force recalibration of worldview (VICS P-scores) | Drone attack on US base, naval confrontation |
| **Legal setback** (court blocks executive order, investigation escalation) | P-4 drops (loss of control), P-1 drops (political universe is hostile), I-5 shifts to "oppose" | Legal constraints directly challenge Trump's sense of control | Supreme Court strikes down tariff authority (actually happened Feb 2026) |
| **Congressional cooperation** (bipartisan bill passes, ally confirms) | I-1 shifts cooperative, I-4a increases (flexibility), P-2 rises | Success through the system validates cooperative tactics | "One Big Beautiful Bill" passage |
| **Natural disaster / humanitarian crisis** | I-1 shifts cooperative (rallying mode), I-2 shifts positive (help-oriented), lower I-3 (risk-averse — don't want to make things worse) | "Rally around the flag" effect shifts toward unity language | Major hurricane, earthquake |

### The VICS Question Pipeline

Unlike binary questions, VICS events are **selected retrospectively** (the event already happened, the speech already exists) but the *synthetic generation* happens **prospectively** (the LLM hasn't seen the speech):

```
1. IDENTIFY: Find post-training-cutoff events where Trump gave spontaneous public remarks
2. TRANSCRIBE: Get the official transcript of Trump's actual response
3. CODE REAL: Run VICS analysis on the real transcript → produces 10 real VICS scores
4. CHECK DEVIATION: Calculate how far the real VICS scores are from Trump's historical average
   → If deviation < threshold → reject (event didn't push Trump off baseline, uninformative)
   → If deviation ≥ threshold → proceed
5. BLIND GENERATE: Give the LLM the event description (but NOT Trump's actual response) 
   and prompt it to generate Trump's speech
6. CODE SYNTHETIC: Run VICS analysis on the synthetic speech → produces 10 synthetic scores
7. COMPARE: Euclidean distance (synthetic vs. real) vs. (average vs. real)
```

**Step 4 is the key filter.** We only keep events where the real Trump speech shows VICS scores that are meaningfully different from his average — because that's where the real test is. Can the LLM predict not just "what Trump generally sounds like" but "how Trump *specifically reacts* to *this specific event*"?

### How Many VICS Events Do We Need?

Using a simple **power analysis** for a paired comparison of two distance measures:

- We need the synthetic-vs-real distance to be *smaller* than the average-vs-real distance in a statistically significant majority of cases
- With a medium effect size (Cohen's d = 0.5), a paired t-test at α=0.05, power=0.80 requires **n ≈ 27 events**
- With a large effect size (d = 0.8), **n ≈ 15 events**

**Target: 20 VICS events minimum**, which gives us enough statistical power for a medium effect size while remaining feasible for manual quality control.

---

## Part 4: Swarm Implementation Notes

### Option A: Full Autonomous Swarm (Recommended for Scale)

Uses the existing LangGraph agent architecture (from the RED CHAMBER project) to run all four agents autonomously:

- **Scanner** runs continuously, monitoring RSS feeds and APIs
- **Drafter** processes new seeds in batches (10-20 seeds per run)
- **Critic** reviews each batch adversarially (different model from Drafter)
- **Resolver** finalizes and records questions in the manifest

**Models**: Use different models for Drafter and Critic to prevent self-agreement:
- Drafter: Gemini 3 Pro (or whatever the test model is)
- Critic: Claude 4 (or GPT-4o) — must be a *different* model family

### Option B: Human-in-the-Loop (Recommended for Initial Validation)

Run the full swarm pipeline but add a human review step between the Critic and Resolver:

```
Scanner → Drafter → Critic → [HUMAN REVIEW] → Resolver
```

The human reviewer:
- Checks resolution criteria for real-world ambiguity that LLMs might miss
- Validates difficulty estimates against their own intuition
- Ensures topic diversity across the full batch
- Flags questions with potential "training data contamination" (events the LLM might have seen)

**Recommendation**: Start with Option B for the first 50 questions. Once the pipeline achieves ≥95% acceptance rate at human review, switch to Option A for scale.

---

## Part 5: Anti-Bias Controls

### Selection Bias

- **Problem**: Unconscious tendency to pick questions you think the LLM will get right (confirmation bias in question selection)
- **Solution**: Pre-register the full question manifest before collecting any LLM predictions. Use the Open Science Framework (osf.io). Once pre-registered, you cannot add or remove questions.

### Temporal Bias

- **Problem**: Asking disproportionately about near-term vs. far-term decisions
- **Solution**: Enforce a distribution across time horizons:
  - 30% questions with resolution in 1-30 days
  - 40% in 31-180 days
  - 30% in 181-365 days

### Difficulty Bias

- **Problem**: Questions that are all too easy or all too hard
- **Solution**: Use prediction market prices as difficulty proxies. If a Kalshi/Polymarket question on a similar topic exists, record its price. Target distribution: 20% easy, 60% medium, 20% hard.

### Domain Bias

- **Problem**: All questions from one domain (e.g., only tariffs)
- **Solution**: Enforce the 7-domain weights from Part 1, Gate 6. The Resolver checks batch diversity before finalizing.

---

## References

| Short Citation | Full Reference | Contribution to This Framework |
|---|---|---|
| Zou et al. (2024) | "ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities." arXiv:2409.09839 | Template-based question generation from time series; post-training-date methodology; bi-weekly auto-generation |
| Bosse et al. (2026) | "Automating Forecasting Question Generation and Resolution for AI Evaluation." arXiv, 2026 | Multi-agent suggest/refine/critique pipeline; 96% unambiguous rate; 3.9% annulment rate; ReAct-style web agents for resolution |
| Metaculus | Question Writing Standards (metaculus.com/help/question-writing-guide) | Resolution criteria best practices; "ambiguous" resolution for underspecified questions; editorial standards |
| Karger et al. (2025) | Mantic AI forecasting methodology (mantic.com) | Difficulty-adjusted Brier scoring; geopolitical question categories; top-ranked AI forecasting system |
| Walker & Schafer | "Operational Code Analysis." Various publications | VICS coding methodology; 10-index system for leader belief profiles |
| Trump VICS Literature | Multiple studies via Profiler Plus (ir-journal.com, lu.se) | Trump-specific VICS baselines; correlation between campaign speeches and policy behavior |
| Tetlock (2015) | "Superforecasting: The Art and Science of Prediction." | Question decomposition, calibration training, difficulty spectrum |
