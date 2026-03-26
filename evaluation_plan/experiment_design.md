# LLM Leader Prediction — Experiment Design

> **Purpose**: A practical blueprint for testing whether LLMs can predict the decisions of specific world leaders. Defines three evaluation experiments, the exact questions we'll ask, and how we'll score them.

---

## The Core Question

> Can an LLM, prompted as a specific world leader, predict that leader's actual decisions more accurately than a simple baseline (e.g., the leader's historical average behavior)?

---

## Why Trump First

Trump is the ideal first subject for three reasons:

1. **Maximum training data**: More publicly available text by and about Trump than arguably any other living person — millions of tweets, thousands of rally transcripts, books, interviews, legal filings, executive orders, press conferences, and media coverage. The LLM "knows" Trump better than any other world leader.
2. **Distinctive and consistent style**: Trump's communication patterns are highly recognizable and consistent, making both generation and evaluation easier.
3. **Measurable decisions**: Trump makes frequent, public, binary-ish decisions (tariff on/off, executive order signed/not, nominee named/withdrawn, deal accepted/rejected) that resolve clearly and quickly.

After Trump, the next candidates in order of expected LLM accuracy:
- **Biden** — extensive public record, but less stylistically distinctive
- **Macron / Modi / Erdogan** — large public profiles, English-language coverage
- **Xi Jinping** — harder due to CCP information filtering and Mandarin-first corpus

---

## Experiment 1: Binary Resolution Questions (Mantic-Style)

### What It Is

Adapted from Mantic's difficulty-adjusted forecasting methodology (Karger et al., 2025) and ForecastBench (Zou et al., 2024). We ask the LLM-persona a series of yes/no questions about *its own future actions*, each tied to a specific resolution date and verifiable outcome.

### The Setup

```
SYSTEM PROMPT:
You are Donald J. Trump, the 47th President of the United States. 
You are being asked about decisions you will make in your current term.
Answer based on your actual beliefs, priorities, strategic interests, 
and decision-making style. For each question, provide:
1. Your answer: YES or NO
2. Your confidence: a probability from 0% to 100%
3. Your reasoning: 2-3 sentences explaining why

USER PROMPT:
"Will you impose tariffs of 25% or higher on Canadian goods before July 1, 2027?"
```

### How We Score It

| Metric | Formula | What It Measures | Citation |
|---|---|---|---|
| **Brier Score** | BS = (probability − outcome)² | Overall accuracy + calibration in one number. 0 = perfect, 1 = worst. *(Domain 3, Entry 1: Brier 1950)* | Brier (1950) |
| **Log Loss** | LL = −[y·log(p) + (1−y)·log(1−p)] | Punishes confident wrong answers severely. *(Domain 3, Entry 2)* | Good (1952) |
| **Calibration (ECE)** | Bin predictions by confidence, compare predicted vs. actual frequency | When the LLM says "80% confident," is it right ~80% of the time? *(Domain 3, Entry 7: Naeini et al. 2015)* | Naeini et al. (2015) |
| **Resolution** | Variance of predicted probabilities | Is the LLM actually discriminating between likely and unlikely events, or hedging everything at ~50%? *(Domain 3, Entry 4: Murphy 1973)* | Murphy (1973) |

### Baselines to Beat

| Baseline | What It Is | Expected Brier | Why It Matters |
|---|---|---|---|
| **Coin flip** | 50% on everything | 0.250 | Absolute floor — if we can't beat this, the experiment failed |
| **Base rate** | Historical frequency of similar Trump decisions (e.g., "Trump imposes tariffs 70% of the time when threatened") | ~0.20 | Does the persona add signal beyond "Trump usually does X"? |
| **Generic LLM** | Same question, no persona prompt ("Will the US impose tariffs...?") | ~0.18-0.22 | Does the persona prompt *improve* accuracy over vanilla LLM? |
| **Prediction market** | Kalshi/Polymarket price at question creation time | ~0.12-0.15 | The gold standard — prediction markets aggregate thousands of informed opinions |
| **Superforecaster** | If available via tournament data | ~0.15 | Human expert ceiling *(Domain 4, Entry 16: Tetlock 2015)* |

### The Question Bank — 50 Candidate Questions for Trump

Organized by **decision category** (the types of decisions Trump makes frequently that are measurable and resolve clearly):

#### Category 1: Trade & Tariffs (Most Frequent, Most Measurable)

| # | Question | Resolution Criteria | Resolution Date |
|---|---|---|---|
| 1 | Will you impose tariffs above 20% on Chinese goods in Q3 2027? | Official tariff schedule published by USTR | End of Q3 2027 |
| 2 | Will you grant a tariff exemption to any EU country before year-end 2027? | Executive order or USTR announcement | Dec 31, 2027 |
| 3 | Will you threaten tariffs on a new country (not currently under tariff threat) in the next 90 days? | Public statement (tweet, press conference, official statement) | 90 days from question date |
| 4 | Will you raise tariffs on Canadian lumber specifically? | USTR tariff schedule | 6 months from question date |
| 5 | Will you use tariffs as leverage in a non-trade negotiation (e.g., immigration, defense spending) in the next 6 months? | Public statement linking tariff action to non-trade demands | 6 months |

#### Category 2: Executive Orders & Policy Actions

| # | Question | Resolution Criteria | Resolution Date |
|---|---|---|---|
| 6 | Will you sign an executive order on [topic X] in the next 60 days? | Federal Register publication | 60 days |
| 7 | Will you reverse or modify [specific Biden-era executive order] in the next 6 months? | Executive order rescinding or amending | 6 months |
| 8 | Will you invoke executive privilege in [specific investigation] in the next 90 days? | Public assertion or legal filing | 90 days |
| 9 | Will you declare a national emergency related to [specific issue] before [date]? | Official declaration | Specified date |
| 10 | Will you issue a presidential pardon to [specific person]? | Official pardon announcement | 6 months |

#### Category 3: Personnel & Appointments

| # | Question | Resolution Criteria | Resolution Date |
|---|---|---|---|
| 11 | Will you fire or accept the resignation of [specific cabinet member] in the next 6 months? | Official announcement | 6 months |
| 12 | Will you nominate a new [specific position] in the next 90 days? | White House nomination announcement | 90 days |
| 13 | Will you publicly criticize [specific member of your own party] in the next 30 days? | Twitter/Truth Social post or press conference transcript | 30 days |

#### Category 4: Foreign Policy & Diplomacy

| # | Question | Resolution Criteria | Resolution Date |
|---|---|---|---|
| 14 | Will you hold a bilateral summit with [specific leader] in the next 6 months? | Official meeting confirmation | 6 months |
| 15 | Will you withdraw from or renegotiate [specific international agreement]? | Official notice or public statement of intent | 1 year |
| 16 | Will you impose new sanctions on [specific country/entity] in the next 90 days? | OFAC sanctions list update | 90 days |
| 17 | Will you publicly threaten military action against [specific country] in the next 6 months? | Public statement | 6 months |
| 18 | Will you reduce US troop presence in [specific region] below [specific number]? | DoD reporting | 1 year |

#### Category 5: Domestic & Social Media

| # | Question | Resolution Criteria | Resolution Date |
|---|---|---|---|
| 19 | Will you post about [specific topic] on Truth Social within 48 hours of [specific event]? | Truth Social post | 48 hours post-event |
| 20 | Will you veto [specific bill] if passed by Congress? | Official veto or signature | Within 10 days of bill presentation |

> **Note**: Questions should be written *after* the model's training data cutoff date, using events the LLM could not have seen. This ensures we're testing prediction, not memory retrieval.

### Post-Hoc Calibration

After collecting ~50+ predictions, apply **Temperature Scaling** (Guo et al., 2017 — Domain 3, Entry 8) to correct for LLM overconfidence. Report both raw and calibrated Brier scores.

---

## Experiment 2: VICS Score Comparison (Synthetic Speech Matching)

### What It Is

The idea: for a post-training-date event, generate what Trump *would say* about it, compute the VICS (Verbs in Context System) operational code scores on that synthetic speech, then compare against:
- **The real speech** Trump actually gave about the event (ground truth)
- **Trump's historical average VICS scores** (baseline)

If the synthetic VICS scores are closer to the *real speech* scores than to the *historical average*, the LLM is capturing Trump's *event-specific* reasoning — not just his general personality.

### VICS Scores — Quick Refresher

VICS produces 10 scores from coded verbs in a leader's speech *(Domain 2, Entries 3-5: Walker & Schafer)*:

| Code | What It Measures | Scale | Trump's Approximate Historical Average |
|---|---|---|---|
| **P-1** | Nature of the political universe (friendly vs. hostile?) | -1 (hostile) to +1 (friendly) | ~0.25 (moderately friendly — sees conflict but believes in deal-making) |
| **P-2** | Prospects for realizing fundamental values | -1 to +1 | ~0.15 (cautiously optimistic about own ability) |
| **P-3** | Predictability of the political universe | 0 (unpredictable) to 1 (predictable) | ~0.10 (sees world as chaotic, unpredictable) |
| **P-4** | Control over historical development | 0 (no control) to 1 (total control) | ~0.30 (believes in personal agency — "only I can fix it") |
| **I-1** | Best strategy: cooperative vs. conflictual | -1 to +1 | ~0.40 (leans cooperative — prefers deals to war) |
| **I-2** | Intensity of tactics | -1 to +1 | ~0.25 (uses rhetorical intensity but prefers economic over military) |
| **I-3** | Risk orientation | 0 (risk-averse) to 1 (risk-acceptant) | ~0.40 (moderate-high risk tolerance) |
| **I-4a** | Flexibility (cooperative-conflictual shifts) | 0 (rigid) to 1 (flexible) | ~0.50 (Trump pivots frequently — "deal" to "threat" and back) |
| **I-4b** | Flexibility (word-deed consistency) | 0 (rigid) to 1 (flexible) | ~0.55 (high gap between rhetoric and action) |
| **I-5** | Utility of means (punish, threaten, oppose, appeal, promise, reward) | Distribution across 6 categories | Skews toward "threaten" and "promise" |

> These averages are illustrative — the real experiment will compute them from Trump's actual coded speeches.

### The Protocol — Step by Step

1. **Select post-training-date events** (5-10 initially) where Trump gave a public response (speech, press conference, Truth Social thread ≥200 words).
   
   Example events:
   | Event | Date | Trump's Response Format |
   |---|---|---|
   | [Country X] announces retaliatory tariffs | Post-cutoff | Press conference, 20 min |
   | Major Supreme Court ruling on [topic] | Post-cutoff | Truth Social thread, 500+ words |
   | Natural disaster in [state] | Post-cutoff | Prepared remarks, 10 min |
   | Foreign leader summit meeting | Post-cutoff | Joint press conference |
   | Congressional vote on [specific bill] | Post-cutoff | Tweet storm + rally remarks |

2. **Generate synthetic response**: Prompt the LLM:
   ```
   SYSTEM: You are Donald J. Trump, the 47th President of the United States.
   Respond to the following event exactly as you would in a press conference.
   Match your typical speaking style, vocabulary, rhetorical patterns, and 
   policy positions. Speak for approximately [X] minutes / [Y] words.
   
   USER: [Description of the event, with factual context]
   ```

3. **Code both texts with VICS**: Run automated VICS coding (using the LLM-based VICS coder from Domain 2, Entry 3) on:
   - The **real** Trump speech (ground truth)
   - The **synthetic** Trump speech (LLM-generated)
   - Trump's **historical corpus** (average across 50+ previous speeches — the baseline)

4. **Compute distances**:
   
   | Comparison | Formula | What It Measures |
   |---|---|---|
   | Synthetic vs. Real | Euclidean distance across all 10 VICS scores | How close the LLM got to Trump's actual reasoning on this specific event |
   | Historical Average vs. Real | Same distance metric | How close the "just use his average" baseline gets |
   | **The key test** | Is Synthetic-vs-Real < Average-vs-Real? | Does the persona prompt capture *event-specific* variation, or just the general personality? |

5. **Statistical test**: Over 5-10 events, use a **paired t-test** or **Wilcoxon signed-rank test** (non-parametric, better for small samples) to determine if the synthetic-vs-real distance is *significantly* smaller than the average-vs-real distance.

### Why This Is Hard (and Why It's Valuable)

- **Hard**: VICS scores are noisy even for real speeches (inter-coder reliability ~0.85), so the ground truth itself has measurement error. Small sample sizes (5-10 events initially) limit statistical power.
- **Valuable**: If it works, this is the strongest possible evidence that the LLM is doing more than surface mimicry — it's capturing the leader's *situational reasoning*, not just their general personality. This would be a publishable result.

### What Success Looks Like

| Outcome | Interpretation | Significance |
|---|---|---|
| Synthetic closer to real than average for ≥7/10 events | The persona prompt captures event-specific reasoning | Strong evidence — LLM personas have real predictive value |
| Synthetic closer for ~5/10 events | Mixed — persona adds some signal but inconsistently | Moderate — worth continuing with more data |
| Synthetic no closer than average | The LLM is just reproducing the leader's general personality, not adapting to specific events | Negative result — persona prompts don't add value for prediction |

---

## Experiment 3: Action Selection (Multiple Choice)

### What It Is

A simpler, higher-volume version of Experiment 1. Instead of open-ended yes/no questions, we present the LLM-persona with a scenario and a menu of 3-5 possible actions, one of which Trump *actually took*. This is exactly analogous to the SCOTUS prediction setup (Katz et al., 2017 — Domain 5, Entry 1).

### The Setup

```
SYSTEM: You are Donald J. Trump, 47th President of the United States.

USER: On [date], [event description]. You are deciding how to respond.
Which of the following actions will you take?

A) Issue a public statement condemning [X] and threaten sanctions
B) Impose immediate tariffs on [country]  
C) Arrange a phone call with [foreign leader] to negotiate
D) Take no immediate public action
E) Sign an executive order addressing [issue]

Answer with the letter of your most likely action and explain your reasoning.
```

### Why This Is Useful

| Advantage | Why |
|---|---|
| **Higher volume** | Faster to score — no need for VICS coding or calibration curves. Can do 50+ scenarios quickly. |
| **Clear baseline** | Random chance = 1/N (20-33% for 3-5 options). Very easy to determine if you're beating chance. |
| **Captures reasoning** | The reasoning explanation lets us evaluate *why* the LLM chose as it did, not just *what* it chose. |
| **Directly comparable to SCOTUS** | Katz et al. achieved 70% on binary SCOTUS outcomes. Can we match that on leader decisions with 3-5 options? |

### Scoring

| Metric | Formula | Target |
|---|---|---|
| **Top-1 accuracy** | % of scenarios where the LLM's first choice matches reality | >40% (3 options), >30% (5 options) — must beat random |
| **Top-2 accuracy** | % where reality is in the LLM's top 2 choices | >60% |
| **Rank correlation** | Spearman correlation between LLM's ranking and actual action | Positive and significant |

### Event Categories for Action Selection

| Category | Example Scenario | Typical Options |
|---|---|---|
| **Trade retaliation** | EU announces counter-tariffs. How does Trump respond? | Escalate tariffs / Negotiate / Threaten but delay / Back down |
| **Congressional standoff** | Bill passes Senate, reaches Trump's desk. | Sign / Veto / Pocket veto / Sign with signing statement |
| **Foreign crisis** | Ally requests military support. | Deploy forces / Send weapons / Offer diplomatic support / Refuse |
| **Personnel** | Cabinet member contradicts Trump publicly. | Fire immediately / Publicly rebuke / Ignore / Support privately |
| **Legal/judicial** | Court blocks executive order. | Appeal / Issue new EO / Attack court publicly / Comply quietly |

---

## Experiment Design Summary

| Experiment | What It Tests | Questions Needed | Difficulty | Publication Value |
|---|---|---|---|---|
| **1. Binary Resolution** | Can the LLM-persona predict specific yes/no outcomes better than baselines? | 50+ | Medium | High — directly comparable to ForecastBench and Mantic |
| **2. VICS Comparison** | Does the persona capture *event-specific* reasoning, not just general personality? | 5-10 speeches | Hard | Very high — novel contribution, publishable in political science journals |
| **3. Action Selection** | Can the LLM-persona pick the right action from a menu? | 50+ | Easy | Medium — useful for demonstrating concept, less methodologically novel |

### Recommended Sequence

1. **Start with Experiment 3** (Action Selection) — lowest overhead, fastest results, establishes feasibility.
2. **Then Experiment 1** (Binary Resolution) — builds the calibration dataset, establishes Brier score baseline.
3. **Then Experiment 2** (VICS Comparison) — hardest but most publication-worthy. Only attempt after Experiments 1 and 3 demonstrate that the persona has *some* predictive value.

---

## Key Design Rules (From the Literature)

These rules are non-negotiable constraints drawn from the literature review:

| Rule | Why | Source |
|---|---|---|
| **Pin the model version** | Same prompt on GPT-4 vs. GPT-4o vs. GPT-4.5 produces different results. All experiments must log exact model ID. | Bisbee et al. (2024) — Domain 5, Entry 10 |
| **Version-control all prompts** | Minor rephrasing changes output distributions. Use Git-tracked prompt templates. | Bisbee et al. (2024) |
| **Use post-training-date events only** | If the event happened before the training cutoff, the LLM might be "remembering" not "predicting." | ForecastBench (Zou et al., 2024) — Domain 4, Entry 1 |
| **Apply Temperature Scaling** | LLMs are overconfident. Post-hoc calibrate all probabilities before scoring. | Guo et al. (2017) — Domain 3, Entry 8 |
| **Check for escalation bias** | LLMs systematically overpredict aggressive actions. Explicitly weight diplomatic options. | Payne et al. (2024) — Domain 5, Entry 16 |
| **Report both raw + calibrated scores** | Transparency — readers need to see both the LLM's raw output and the corrected version. | Standard best practice |
| **Pre-register the question set** | Commit to questions *before* seeing results to prevent cherry-picking. | Open Science Framework |
| **Multiple prompt runs** | Run each question 10+ times to measure consistency. Report mean, standard deviation, and agreement rate. | Kovarikova et al. (2025) — Domain 5, Entry 11 |

---

## References

| Short Citation | Full Reference | Used For |
|---|---|---|
| Brier (1950) | Brier, G.W. "Verification of forecasts expressed in terms of probability." *Monthly Weather Review*, 78(1). | Brier Score — primary accuracy metric |
| Murphy (1973) | Murphy, A.H. "A new vector partition of the probability score." *Journal of Applied Meteorology*, 12(4). | Murphy Decomposition — separating calibration from resolution |
| Naeini et al. (2015) | Naeini, M.P., Cooper, G., Hauskrecht, M. "Obtaining well calibrated probabilities using Bayesian binning into quantiles." *AAAI 2015*. | Expected Calibration Error (ECE) |
| Guo et al. (2017) | Guo, C. et al. "On calibration of modern neural networks." *ICML 2017*. | Temperature Scaling — post-hoc calibration |
| Katz et al. (2017) | Katz, D.M. et al. "A general approach for predicting the behavior of the Supreme Court." *PLoS ONE*, 12(4). | SCOTUS prediction — the closest structural analogy |
| Argyle et al. (2023) | Argyle, L.P. et al. "Out of One, Many: Using language models to simulate human samples." *Political Analysis*, 31(3). | Silicon sampling — baseline for persona simulation |
| Bisbee et al. (2024) | Bisbee, J. et al. "Synthetic Replacements for Human Survey Data?" *Political Analysis*. | Critique — five failure modes of persona simulation |
| Payne et al. (2024) | Payne, J. et al. "Escalation Risks from Language Models in Military and Diplomatic Decision-Making." | Escalation bias — the #1 failure mode to control for |
| Park et al. (2024) | Park, J.S. et al. "Generative Agent Simulations of 1,000 People." arXiv:2411.10109. | 85% accuracy benchmark for individual persona simulation |
| Zou et al. (2024) | Zou, A. et al. "ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities." arXiv:2409.09839. | Post-training-date question methodology |
| Karger et al. (2025) | Karger, E. et al. Mantic / Thinking Machines forecasting methodology. | Difficulty-adjusted binary resolution scoring |
| Kovarikova et al. (2025) | Kovarikova et al. "LLM Generated Persona Is a Promise with a Catch." arXiv, 2025. | Systematic bias in persona predictions |
| Walker & Schafer | Walker, S.G. & Schafer, M. "Operational Code Analysis." Various publications. | VICS scoring methodology |
