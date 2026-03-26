# Domain 4: LLM Forecasting Benchmarks, Prediction Arenas & Superforecasting — Literature Review

> **Question this review answers**: What platforms, benchmarks, and competitions exist for testing whether artificial intelligence (AI) — specifically Large Language Models (LLMs) — can predict future events? How do these AI systems compare to the best human forecasters? And what is the current state-of-the-art?
>
> **Why this matters for our project**: Before we can test whether LLMs can predict world leader decisions, we need to know (1) how other researchers have tested LLM prediction ability, (2) what scoring methods they use, (3) how they prevent cheating (data leakage), and (4) what the current accuracy benchmarks are. This domain maps the competitive landscape our work enters.

---

## Key Terms Defined Up Front

| Term | Plain-Language Definition |
|---|---|
| **Benchmark** | A standardized test designed to measure how well an AI performs on a specific task — like a final exam that every model takes so you can compare scores fairly |
| **Brier Score** | A number between 0 and 1 that measures how accurate a probability prediction was. 0 = perfect, 0.25 = random guessing, 1 = perfectly wrong. See Domain 3 for full mathematical definition |
| **Brier Skill Score (BSS)** | How much better (or worse) a forecaster is compared to a baseline. BSS = 1 − (your Brier / baseline Brier). Positive = you beat the baseline. Negative = the baseline beat you |
| **Calibration** | Whether your confidence matches reality. If you say "70% chance" for 100 different events, calibration means roughly 70 of them should actually happen |
| **Contamination / Data Leakage** | When an AI has already "seen" the answer during training, making its prediction artificially accurate — like a student who stole the answer key before an exam |
| **ECE (Expected Calibration Error)** | A single number measuring how far off a model's confidence is from reality, averaged across all confidence levels. Lower = better calibrated. See Domain 3 for details |
| **Ensemble** | Combining predictions from multiple models (or people) and averaging them, which typically produces more accurate results than any single predictor |
| **Log Loss** | Another scoring rule (like Brier) that penalizes confident wrong predictions very harshly. A model that says "99% sure" and is wrong gets devastated by log loss |
| **Prediction Market** | A marketplace where people buy and sell contracts on future events. The market price reflects the crowd's collective probability estimate. Example: if a contract for "Will X happen?" trades at $0.70, the crowd thinks there's a 70% chance |
| **RAG (Retrieval-Augmented Generation)** | A technique where an LLM first searches for relevant documents/data, then uses that retrieved information to generate its answer — like an open-book exam instead of a closed-book exam |
| **RLVR (Reinforcement Learning with Verifiable Rewards)** | Training an AI by rewarding it when its predictions match reality. "Verifiable" means the reward is based on objective outcomes (did the event happen or not?), not human opinions |
| **Superforecaster** | A person in the top 2% of prediction accuracy, as identified by the IARPA-funded Good Judgment Project. They consistently outperform intelligence analysts, domain experts, and prediction markets |

---

## Thread 1: Benchmarks & Platforms — How Researchers Test LLM Forecasting

> **What this thread covers**: The standardized tests (benchmarks) that researchers have built to measure whether LLMs can predict future events. Each benchmark has different rules, different types of questions, and different scoring methods.

---

### Entry 1: ForecastBench — The Primary Benchmark for LLM Prediction

**Citation**: Karger, E., Bastani, H., Chen, Y., Garimella, K., Jacobsen, J., Leclerc, M., Lester, J., Tetlock, P., & Mellers, B. (2024). "ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities." arXiv:2409.xxxxx.
**URL**: [forecastbench.org](https://forecastbench.org)

**What problem does this solve?**
Most AI benchmarks test knowledge that already exists in training data — like answering trivia questions or solving math problems. This means an AI can score well simply by memorizing answers, not by actually reasoning. ForecastBench solves this by asking questions that *have no answer yet*. Every question is about a future event that hasn't happened. No amount of memorization can help — the AI must actually *predict*.

**How it works — step by step**:
1. **Automatic question generation**: Software programs pull data from public sources — FRED (the Federal Reserve Economic Database, which tracks economic indicators like inflation, unemployment, and GDP), ACLED (the Armed Conflict Location & Event Data Project, which tracks every reported conflict event worldwide), and Yahoo Finance (stock prices, commodity prices, corporate earnings) — and automatically generate questions like "Will U.S. CPI exceed 3.5% in March 2026?"
2. **Human-curated questions**: Additional questions are pulled from Metaculus (a reputation-based forecasting platform where thousands of people predict events) and Polymarket (a cryptocurrency-based prediction market where people bet real money on outcomes).
3. **Question format**: All questions are binary — they resolve to either Yes (1.0) or No (0.0). The LLM outputs a probability between 0 and 1.
4. **Dynamic rotation**: Questions expire when they resolve, and new ones are automatically generated. At any given time, approximately 1,000 questions are active. This prevents models from memorizing a fixed question set.
5. **What the LLM receives**: The question text, precise resolution criteria (exactly how "yes" or "no" will be determined), a timestamp, and in some variants, the Metaculus community median forecast (called the "crowd-augmented" condition — this tells the LLM what thousands of humans collectively predict).

**Why the "contamination-free guarantee" matters**:
- **The fundamental problem with AI benchmarks**: If a benchmark asks about events that have already happened, the AI might have seen news articles about the outcome during training. Its "prediction" is really just recall.
- **ForecastBench's solution**: Every question is about an event that *has not yet occurred* at the time the LLM is queried. No ground-truth label exists in any training corpus. This is the gold standard for preventing data leakage.
- **Dynamic regeneration**: Even if someone published all current ForecastBench questions online, the questions would rotate out within weeks, making any leaked answers obsolete.

**How predictions are scored**:
- **Brier Score**: The standard metric (see Domain 3, Entry 2). Calculated as (predicted probability − actual outcome)². Range: 0 (perfect) to 1 (perfectly wrong). 0.25 = random guessing.
- **Difficulty-adjusted Brier Score**: Questions that most forecasters get wrong are weighted more heavily. This rewards models that can predict hard events, not just easy ones.
- **Brier Index**: The Brier Score rescaled to 0–100% where higher = better, for easier readability. Brier Index ≈ (1 − Brier Score) × 100, with difficulty adjustment.

**Current results** (ForecastBench leaderboard, March 14, 2026):

| Forecaster | Brier Index | Brier Score (approx.) | What this means |
|---|---|---|---|
| Superforecaster median | **70.6** | ~0.081 | The best humans — top 2% of all forecasters |
| Cassi ensemble (crowd-adjusted) | 68.0 | ~0.095 | AI system that also sees human crowd predictions |
| Grok 4.20 (Preview) | 67.9 | ~0.096 | xAI's frontier model |
| Gemini 3 Pro (zero-shot + crowd) | 67.8 | ~0.096 | Google's frontier model, given crowd data |
| GPT-5 (zero-shot + crowd) | 67.4 | ~0.098 | OpenAI's frontier model, given crowd data |
| Foresight-32B | 67.4 | ~0.098 | Lightning Rod Labs' RL-trained model (see Entry 9) |

**How the gap has closed over time**:

| Date | Best LLM Brier | Superforecaster Brier | Gap | Gap as % |
|---|---|---|---|---|
| 2023 (GPT-3 era) | >0.25 (worse than random) | ~0.08 | >0.17 | >210% |
| Oct 2024 (Karger paper) | 0.124 | 0.071 | 0.053 | 75% |
| Oct 2025 | 0.101 (GPT-4.5) | 0.081 | 0.020 | 25% |
| Jan 2026 | 0.098 | 0.081 | 0.017 | 21% |
| **Projected parity** | — | — | 0.000 | ~Nov 2026 |

**Limitations you should know about**:
- Questions skew economic and quantitative (FRED, financial data). There are fewer purely political or geopolitical questions, which is exactly the domain we care about.
- The "crowd-augmented" variants give LLMs access to the Metaculus community median. This is essentially telling the model what humans think before it predicts. The boost is significant (~17–28% improvement per Halawi et al., Entry 8). When citing results, always note whether the model had access to crowd data.
- Difficulty-adjusted scoring can mask absolute calibration problems — a model might score well on the index but still be poorly calibrated.

**Why it matters for our project**: ★★★★★ — This is *the* primary benchmark for LLM forecasting. Our experiments must use comparable methodology (Brier scoring, contamination-free design) and report BSS against ForecastBench baselines so our results are directly comparable to the field.

---

### Entry 2: Prophet Arena — Live Forecasting with Economic Value Scoring

**Citation**: Xu, H. et al. (2024). "Prophet Arena: Benchmarking LLM Forecasting with Live Events." UChicago SIGMA Lab.
**URL**: [prophetarena.co](https://prophetarena.co)

**What problem does this solve?**
Most benchmarks only measure statistical accuracy (was the probability close to the outcome?). But in the real world, what matters is whether a prediction is *useful* — does it tell you something the market doesn't already know? Prophet Arena adds an economic value metric: if you bet $1 every time the model disagreed with the market, would you make money?

**How it works — the 3-stage pipeline**:
1. **Information sourcing**: The LLM autonomously gathers context — searching news articles, data sources, and background information relevant to the question. This tests whether the model can *find* the right information, not just process information handed to it.
2. **Prediction**: The model outputs a probability distribution (e.g., "72% chance this happens").
3. **Resolution**: The event resolves against Kalshi's official settlement. Kalshi is a CFTC-regulated prediction market (meaning it's overseen by the U.S. Commodity Futures Trading Commission, the federal agency that regulates futures and options markets), so resolution criteria are legally binding and unambiguous.

**Three complementary scoring metrics**:

| Metric | What It Measures | How to Interpret |
|---|---|---|
| **Brier Score** | Statistical accuracy of probabilities | Reported as "1 − Brier" (higher = better). Same math as ForecastBench but inverted for readability |
| **Simulated Average Return (Market ROI)** | Economic value of predictions | Simulates placing $1 bets whenever the model's probability diverges from the Kalshi market price by a threshold. Positive ROI = the model knows something the market doesn't |
| **IRT (Item Response Theory)** | Difficulty-adjusted skill rating | Borrowed from educational testing (the same math behind SAT/GRE scoring). Treats each question as a "test item" with its own difficulty and discrimination parameters. Weights hard, informative questions more heavily |

**Why Market ROI matters**:
- A model that simply agrees with the prediction market adds no value, even if it's correct. The market already knows what it knows.
- Market ROI rewards *contrarianism that turns out to be right*. If the market says 30% and the model says 70%, and the event happens, the model would have profited.
- This directly tests whether the AI has *private information* (information the market hasn't priced in) — which is exactly what we want to know for predicting leader decisions.

**Limitations**:
- Limited to questions with Kalshi markets, which skew U.S.-centric and finance-heavy.
- IRT scoring requires large question sets (100+) to produce stable skill estimates.

**Why it matters for our project**: ★★★★☆ — The ROI metric is exactly the right question: does the LLM add value *beyond* what existing information sources already provide? The 3-stage pipeline (source → predict → resolve) maps directly onto our experimental flow.

---

### Entry 3: PredictionArena.ai — Real-Money AI Trading on Prediction Markets

**URL**: [predictionarena.ai](https://predictionarena.ai)

**What this is**: The most aggressive test of AI forecasting — each AI model starts with a **virtual $10,000 portfolio** and trades on *actual* Kalshi prediction markets with real pooled capital. Every ~15 minutes, AI agents receive current market data, review their portfolio, research opportunities, and decide whether to trade or pass.

**Why real money matters**: When real money is on the line, every trade has consequences. The AI can't just output probabilities and walk away — it must decide how much to bet, when to bet, and when to sit out. This tests *decision-making under uncertainty*, not just probability estimation.

**Scoring**: Mark-to-Market Account Value — calculated using current bid prices (what you'd actually get if you sold everything right now) plus cash balance. This is a conservative measure that reflects real-world liquidation value.

**Limitations**: Only tests markets where Kalshi contracts exist (U.S. political, economic, event markets). Real-money stakes limit participation to well-funded teams.

**Why it matters for our project**: ★★★☆☆ — Proof-of-concept that AI can trade profitably on predictions, but less directly relevant since we're scoring predictions, not trading.

---

### Entry 4: MIRAI — Intelligence Analyst Simulation for Conflict Forecasting

**Citation**: Jin, Y. et al. (2024). "MIRAI: Evaluating LLM Agents for Event Forecasting." *NeurIPS 2024 Datasets and Benchmarks*.
**URL**: [arxiv.org/abs/2407.01231](https://arxiv.org/abs/2407.01231)

**What problem does this solve?**
Most benchmarks test LLMs as passive oracle — you give it a question, it gives you a probability. But real-world intelligence analysis requires *actively finding information*, writing queries, and synthesizing across multiple data sources. MIRAI tests whether LLMs can act as autonomous intelligence analysts.

**How it works — simulating an intelligence analyst**:
The LLM must perform three tasks that mirror a real analyst's workflow:
1. **Search a database**: The model queries a SQL database of ~500,000 structured event records from GDELT (Global Database of Events, Language, and Tone) and ICEWS (Integrated Crisis Early Warning System). These databases classify every international interaction on a cooperation-conflict scale called the **Goldstein Scale** (ranging from -10 for "military attack" to +10 for "extend military aid").
2. **Read news articles**: The model queries a REST API to retrieve from ~180,000 news articles from international wire services covering January–November 2023.
3. **Predict future events**: Using both structured data and unstructured news, the model predicts: Will a specific type of event (protest, diplomatic meeting, military action) occur between two actors (country pairs) next week? Next month? Will the relationship be cooperative or conflictual? What intensity?

**Geographic focus**: Conflict-prone regions — Horn of Africa (Ethiopia, Somalia, Sudan) and Middle East (Israel, Palestine, Lebanon, Iraq, Syria).

**What makes this unique — the tool-use requirement**:
Unlike other benchmarks where LLMs just answer questions, MIRAI requires the model to *write Python code* to query databases and APIs. This evaluates three capabilities simultaneously:
1. **Information retrieval** — can the model find the right data?
2. **Code generation** — can it write working SQL queries and API calls?
3. **Joint reasoning** — can it synthesize structured event data with unstructured news articles to make predictions?

**Key findings**:
- **Retrieval quality is the key bottleneck**: Better RAG (retrieval) → better predictions. The model's ability to find the right information matters more than its reasoning ability.
- **Long-term predictions are hard for everyone**: All LLMs struggled significantly with next-month predictions compared to next-week.
- **Code generation is a major failure mode**: Models frequently wrote incorrect SQL queries or API calls, meaning they never even accessed the relevant data.

**Limitations**: Uses categorical predictions only (will this event type occur? yes/no) — does not evaluate probabilistic calibration. Limited to the GDELT/ICEWS event ontology and data ending November 2023.

**Why it matters for our project**: ★★★★☆ — Directly relevant to our RAG-based prediction pipeline. The lesson is clear: *how well your model retrieves information matters more than how smart your model is*. We should invest heavily in retrieval quality for our leader decision prediction task.

---

### Entry 5: UNBench — The Most Directly Relevant Benchmark (UN Security Council Predictions)

**Citation**: "UNBench: Benchmarking LLMs on UN Security Council Decision-Making." arXiv:2024. Published at AAAI 2025.

**What problem does this solve?**
No benchmark specifically tested whether LLMs could predict how *specific governments and diplomats* would vote on *specific policy proposals*. UNBench is the first to do exactly this, using 30 years of UN Security Council records.

**The four tasks** (each progressively harder):

| Task | What the LLM Must Do | Why It's Hard | Key Metric |
|---|---|---|---|
| **1. Co-penholder Judgment** | Predict which countries led the drafting of a resolution | Requires understanding informal, behind-the-scenes diplomatic negotiations that aren't in public records | Accuracy |
| **2. Representative Voting Simulation** | Act as a specific country and predict its vote (Yes / No / Abstain) on a draft resolution | Must internalize a country's foreign policy positions, alliances, and historical patterns | F1 per class, macro-F1 |
| **3. Draft Adoption Prediction** | Predict whether a resolution will pass or fail | Must understand veto dynamics — any of the 5 permanent members (US, UK, France, Russia, China) can block any resolution | Binary accuracy |
| **4. Statement Generation** | Generate a country-specific speech justifying its vote | Must capture diplomatic tone, policy priorities, and rhetorical style | BLEU, BERTScore |

**The dataset — 30 years of UN Security Council records**:
- **Source**: UN Digital Library — every UNSC draft resolution, voting record, and verbatim transcript from **1994–2024**.
- **Scale**: ~2,500 draft resolutions × 15 UNSC members = ~37,500 individual vote labels.
- **Class imbalance**: ~80% of votes are "Yes" (most resolutions pass by consensus). The interesting predictions are the ~20% that involve "No" votes or "Abstain" — and especially the rare vetoes by the P5 (the five permanent members with veto power).
- **Context provided to LLMs**: Draft resolution text, the target country's historical voting patterns, geopolitical context (news summaries), and the country's prior UNSC statements on related topics.

**Key findings**:
- LLMs struggle most with **co-penholder judgment** (requires insider knowledge of informal negotiations that doesn't appear in public records).
- LLMs perform best on **draft adoption prediction** (more structured, pattern-based — the P5 veto pattern is historically regular).
- **No Brier scoring** — predictions are categorical (Yes/No/Abstain), not probabilistic. This is a limitation we should address by adding probabilistic scoring on top.

**Contamination risk**: Uses historical data (1994–2024), so LLMs may have seen some of this in training. A temporal holdout design (train on 1994-2020, test on 2021-2024) would be more rigorous.

**Why it matters for our project**: ★★★★★ — **The single most relevant benchmark to our work.** The voting simulation task is exactly what we want to evaluate. We should adopt UNBench's temporal holdout design, add probabilistic Brier scoring on top of their categorical metrics, and compare our leader decision predictions against their results.

---

### Entry 6: Metaculus FutureEval — The Continuously Updated Meta-Benchmark

**URL**: [metaculus.com/futureeval](https://metaculus.com)

**What this is**: Metaculus (a platform where thousands of people predict real-world events for reputation points) launched its own continuously updated benchmark in February 2026 specifically to track how fast AI forecasting is improving relative to humans.

**Current projections**:
- AI projected to surpass **Metaculus community** performance (aggregated predictions from thousands of non-expert forecasters) by **April 2026**.
- AI projected to surpass **professional forecasters** (people who forecast for a living, including some superforecasters) by **June 2027**.
- 80% confidence interval for AI reaching superforecaster parity: **October 2025 to 2030**.

**Key tournament results (Metaculus Cup)**:

| Tournament | Top AI Result | Human Comparison |
|---|---|---|
| Summer 2024 | AI systems first entered | Humans dominated |
| Summer 2025 | Mantic (AI) placed **8th** | Beat most humans |
| Fall 2025 | Mantic placed **4th** out of 539 participants | Outperformed 535 humans including 6 professionals |
| Spring 2026 | Bot-only tournament launched | Separate AI-vs-AI track |

**Why it matters for our project**: ★★★☆☆ — Useful as a meta-tracker of the field's progress, less useful as a specific evaluation tool (we need more control over question design).

---

### Entry 7 (New): Pratt et al. (2024) — "Can Language Models Use Forecasting Strategies?"

**Citation**: Pratt, S., Blumberg, E., Carolino, R., & Morris, A. (2024). "Can Language Models Use Forecasting Strategies?" arXiv:2406.xxxxx.

**What they tested**: Whether explicitly instructing LLMs to use the strategies that make human superforecasters successful (see Thread 3 below) actually improves LLM forecasting accuracy.

**The strategies they tested** (all borrowed from Tetlock's superforecasting research):
1. **Base-rate anchoring**: Start with the historical frequency of similar events, then adjust. ("How often have trade wars escalated to military conflict historically?" → use that as your starting point)
2. **Decomposition**: Break complex questions into simpler sub-questions, estimate each, and recombine. ("Will X happen?" → "What's the chance of A? If A, what's the chance of B? If A and B, what's the chance of X?")
3. **Outside view**: Explicitly consider base rates and reference classes before looking at case-specific details.
4. **Multiple perspectives**: Consider arguments for and against the prediction before committing.

**Key finding**: The superforecasting strategies **did not consistently improve LLM performance** compared to simpler prompting approaches. This is a cautionary result — just telling an LLM to "think like a superforecaster" doesn't make it one.

**Why the strategies might not transfer well**:
- Human superforecasters internalize these strategies over years of practice and feedback. LLMs receive them as one-shot instructions.
- LLMs showed a tendency to default to predicting "unlikely" for most events (since most specific events are indeed unlikely), which inflates accuracy on datasets with low base rates but doesn't reflect genuine forecasting skill.
- The strategies may interact with LLM biases (overconfidence, anchoring on salient details) in unpredictable ways.

**Why it matters for our project**: ★★★★☆ — Important negative result. Simply prompting LLMs to use superforecasting strategies is not a shortcut. We need RL-based training (see Entries 9-10) or structural pipeline improvements (retrieval, ensemble, calibration) to close the accuracy gap.

---

## Thread 2: AI Forecasting Systems — Production-Grade AI Prediction

> **What this thread covers**: AI systems that go beyond one-off experiments to provide continuous, real-time forecasting. These are the closest analogs to what our project aims to build.

---

### Entry 8: Halawi et al. (2024) — "Approaching Human-Level Forecasting with Language Models" (NeurIPS 2024)

**Citation**: Halawi, D., Zhang, F., Yueh-Han, C., & Steinhardt, J. (2024). "Approaching Human-Level Forecasting with Language Models." *NeurIPS 2024*, Main Conference Track.
**URL**: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a5acfd0876c940d81619c1dc60e7748-Abstract-Conference.html)

**What problem does this solve?**
Before this paper, claims about LLM forecasting ability were anecdotal or based on small samples. Halawi et al. provided the first rigorous, large-scale comparison of LLM predictions against human forecasters using proper scoring rules.

**Experimental setup**:
- **900 binary forecasting questions** from competitive prediction platforms (primarily Metaculus), covering geopolitics, economics, science, technology, and domestic policy.
- **Custom news retrieval system**: For each question, the system queries Google News and Bing for relevant articles published *before* the prediction date. The top-k articles are concatenated into the LLM's prompt context. This simulates an "open-book" exam rather than testing pure recall.
- **Models tested**: GPT-3.5-turbo, GPT-4, GPT-4-turbo, Claude 2, Claude 3, and various ensemble configurations.
- **Human comparison**: Metaculus community forecasts (thousands of forecasters) and a superforecaster cohort recruited from the Good Judgment Project.

**The "Wisdom of the Silicon Crowd" finding**:
Just as aggregating predictions from many humans produces a "wisdom of the crowd" that beats most individuals, Halawi et al. discovered the same effect with LLMs:
- **Ensemble method**: Take the median (middle value) of predictions from 5–12 diverse LLMs (different architectures, different training data).
- **Result**: The LLM ensemble ("silicon crowd") achieved accuracy **comparable to the aggregated human crowd** — not superforecasters, but the collective prediction of thousands of non-expert humans.

**The human prior boost**:
- When LLMs were shown the **median human prediction** before making their own forecast, their accuracy improved by **17–28%**.
- This works bidirectionally: human forecasters given LLM reasoning also improved their accuracy by 24–28% (see Entry 13).

**Brier Score results across the field** (compiled from Halawi et al. and subsequent benchmarks):

| Forecaster | Brier Score | Interpretation |
|---|---|---|
| GPT-3-level models | >0.25 | *Worse than random* due to extreme overconfidence |
| Early GPT-4 | ~0.25 | Roughly random — better reasoning but still poorly calibrated |
| Human crowd (Halawi) | 0.149 | Aggregated predictions from thousands of Metaculus forecasters |
| o3 (OpenAI, 2025) | 0.135 | Better than human crowd |
| Claude 3.5 Sonnet | 0.117 | Strong frontier model performance |
| Superforecasters (Karger) | 0.121 | Top 2% of human forecasters |

**Critical lesson — why GPT-3 was *worse than random***:
A random predictor always says "50%" and gets a Brier score of 0.25. GPT-3-level models scored *above* 0.25 because they were **overconfident on wrong answers**. They would say things like "95% chance" when the event didn't happen, getting severely penalized. This is the **calibration problem** discussed in Domain 3 — a model can be smart (good at discrimination) but unreliable (poorly calibrated).

**Why it matters for our project**: ★★★★★ — Provides hard numbers for the LLM-human accuracy gap. Key design implications: (1) always use ensembles, (2) consider giving LLMs access to human base-rate estimates, (3) post-hoc calibration (Domain 3, Entries 17-19) is essential to fix overconfidence.

---

### Entry 9: Turtel et al. (2025) — "Outcome-based Reinforcement Learning to Predict the Future"

**Citation**: Turtel, B., Franklin, D., Skotheim, K., Hewitt, L., & Schoenegger, P. (2025). arXiv:2505.17989. Lightning Rod Labs.
**Dataset**: [HuggingFace](https://huggingface.co/datasets/LightningRodLabs/outcome-rl-test-dataset)

**What problem does this solve?**
Standard LLMs are trained to predict the *next word* in text. This objective has nothing to do with predicting *future events*. Turtel et al. asked: what if we trained an LLM specifically to minimize prediction error on real-world events, using reinforcement learning (RL)?

**The core insight — RLVR for prediction**:
- **RLVR (Reinforcement Learning with Verifiable Rewards)** is the technique that dramatically improved LLM performance on math and coding problems (e.g., DeepSeek-R1, OpenAI o1). The "verifiable" part means the reward is based on an objective outcome — did the model get the right answer? — not on a human rater's subjective judgment.
- Turtel et al. applied the same idea to forecasting: the "reward" is the negative Brier score (a strictly proper scoring rule — see Domain 3). When an event resolves, the model receives a reward proportional to how close its probability was to the actual outcome.

**Training data — how they built the dataset**:
1. **10,000 resolved questions** scraped from Polymarket prediction markets. For each question, a random prediction date is sampled between market open and close. Relevant news headlines are retrieved from Exa.ai for dates *strictly before* the prediction date.
2. **100,000 synthetic questions** generated via "Foresight Learning" — the LLM reads streams of news articles and automatically generates questions that are hard to predict at time T but verifiable at time T+Δ. *Zero human annotation required.*
3. **Test set**: 1,265 held-out Polymarket questions with a **3-step temporal leakage prevention protocol**:
   - Step 1: All test prediction-dates occur *after* the latest resolution date of any training question.
   - Step 2: Only questions originally scheduled to close by dataset construction time are included (prevents cherry-picking questions that resolved favorably).
   - Step 3: OpenAI o3 is used as a "leak detector" — it flags any question where retrieved news contains information that shouldn't have been known at the prediction date.

**Base model**: DeepSeek-R1-Distill-Qwen-14B — a 14 billion parameter model (small by frontier standards — GPT-4 and o1 are estimated at 100x+ larger).

**Results**:

| Model | Brier Score | ECE | Key Takeaway |
|---|---|---|---|
| **ReMax (10K, ensemble-7)** | **0.190** | **0.062** | Best fine-tuned model — small model, big improvement |
| Modified GRPO (100K) | 0.194 | 0.069 | Close second using synthetic data |
| DeepSeek-R1-14B (base, no RL) | 0.219 | 0.120 | Same model before RL training — significantly worse |
| OpenAI o1 | 0.208 | 0.092 | ~100x more parameters but worse than the RL-trained 14B |
| **Polymarket crowd** | **0.165** | — | Prediction market still best overall |

**The "small model beats giant model" finding**:
A 14B parameter model trained with RL (ReMax) achieved Brier 0.190 — better than OpenAI o1 (0.208), which has roughly 100x more parameters but was not specifically trained for forecasting. **Training objective matters more than model size.**

**Simulated trading performance**:
- ReMax earned **$52 profit on $433 of bets** (~10% ROI) across all 1,265 questions.
- **Edge concentration**: Profit was driven almost entirely by questions where the market was most uncertain (40-60% probability). On these uncertain questions, the model's bets succeeded **11.8 percentage points** more often than market prices predicted — a ~20% ROI.
- **No edge on confident markets**: When Polymarket was highly confident (>80% or <20%), the model could not profitably bet against the market. The market's collective wisdom prevailed for "easy" questions.

**Why it matters for our project**: ★★★★★ — Five critical lessons:
1. RLVR works for prediction — the same technique that improved math/code transfers to open-world forecasting.
2. Training objective > model size. A small, well-trained model beats a giant general-purpose one.
3. Proper scoring rules (Brier) work directly as RL reward functions — this cleanly integrates our Domain 3 evaluation framework with training.
4. The 3-step leakage prevention protocol is a gold standard we should adopt.
5. "Foresight Learning" (auto-generating training questions from news) is directly applicable to our pipeline.

---

### Entry 10: Turtel et al. (2026) — "Future-as-Label: Scalable Supervision from Real-World Outcomes"

**Citation**: Turtel, B., Wilczewski, P., Franklin, D., & Skothiem, K. (2026). arXiv:2601.06336. Lightning Rod Labs.

**What problem does this solve?**
The 2025 paper (Entry 9) relied on Polymarket questions for training data. But prediction markets don't cover most topics, and their questions are written by market makers, not researchers. "Future-as-Label" asks: can you train a forecasting model using *only news articles and the passage of time* — with zero human annotation?

**How "Foresight Learning" works** — the key innovation:
1. **Freeze the news corpus** at a cutoff time T (e.g., July 1, 2024).
2. **Auto-generate binary questions** about events expected to resolve after T, using only pre-T information. For example: "Will the UK Labour Party win the general election by July 15?"
3. **Wait for events to resolve** — the passage of time provides the labels. A separate, frozen Gemini-2.5-Flash model checks post-T news to determine whether the event happened. The resolver never sees model outputs, so resolution errors introduce noise but not systematic bias.
4. **Train the model** using the (question, prediction, outcome) triplets with Brier score as the RL reward.

**Dataset**:
- **5,620 total examples** — 5,120 training + 500 temporally disjoint test questions.
- Training data: predictions as of July 1, 2024 through January 30, 2025.
- Test data: predictions on or after February 1, 2025 — *strict temporal separation*.
- **Second test set**: 293 human-written forecasting questions from Metaculus (completely independent of the training pipeline).
- Topics: politics, economics, and corporate actions.
- Base model: **Qwen3-32B** (32 billion parameters).

**The "training objective > model size" result, reinforced**:

| Model | Brier Improvement | Calibration Improvement | Key Finding |
|---|---|---|---|
| Qwen3-32B (base, no RL) | baseline | baseline | Just a general-purpose LLM |
| Qwen3-32B (ensemble-7) | modest | modest | Sampling multiple answers helps but doesn't change fundamental capability |
| **Qwen3-32B + Foresight Learning** | **−27% vs baseline** | **−50% vs baseline** | RL training fundamentally changes what the model can do |
| Qwen3-235B (base, no RL) | *worse than Foresight-32B* | *worse than Foresight-32B* | 7× more parameters but no RL training — inferior |

**The critical finding**: A 32B model trained with Foresight Learning outperformed a 235B model on *both* the synthetic test set *and* the independently-authored Metaculus questions. **A model 7× smaller beats a model 7× larger because of how it was trained, not how big it is.**

**Generalization**: Performance gains transferred to Metaculus questions (a completely different distribution from the news-based training data), suggesting the model learned *general forecasting skill*, not just memorization of the training domain.

**Why it matters for our project**: ★★★★★ — The strongest evidence that training objective matters more than model size for prediction. The entire training pipeline requires *zero human annotation* — news articles + passage of time provide all supervision. Directly applicable to geopolitical prediction: we could train a "Foresight" model on geopolitical news and evaluate it on leader decisions.

---

### Entry 11: Bosse et al. (2026) — "Automating Forecasting Question Generation and Resolution"

**Citation**: Bosse, N.I., Mühlbacher, P., Wildman, J., Phillips, L., & Schwarz, D. (2026). arXiv:2601.22444. ICLR 2026 Workshop.

**What problem does this solve?**
AI forecasting research has a data bottleneck: every forecasting question needs to be (1) written by a human expert, (2) given clear resolution criteria, (3) checked for ambiguity, and (4) resolved months later by a human judge. Bosse et al. automated all four steps.

**The 5-stage question generation pipeline**:

| Stage | What Happens | Example |
|---|---|---|
| 1. **Seeds** | Arbitrary text (news articles, company reports) provides inspiration | A Reuters article about EU tariff negotiations |
| 2. **Proto-questions** | A ReAct-style web research agent proposes 1-7 forecasting questions per seed | "Will the EU impose retaliatory tariffs on US goods by June 2026?" |
| 3. **Refined questions** | Another agent adds precise, objective resolution criteria | "Resolution: Yes if the European Commission officially announces tariff implementation. Source: Official EU press release or Journal entry." |
| 4. **4-stage verification** | Four separate agents evaluate each question for: (a) quality, (b) ambiguity, (c) resolvability, (d) non-triviality | A question with 99% base rate ("Will the sun rise tomorrow?") gets filtered out at stage (d) |
| 5. **Deduplication** | LLM-based deduplication removes near-identical questions | Mean intra-cluster similarity = 1.32/4.0 (70% of pairs rated "completely different") |

**(ReAct** stands for "Reasoning + Acting" — a prompting technique where the LLM alternates between thinking through a problem and taking actions like web searches, rather than trying to answer in one shot.)

**Quality of generated questions**:
- **1,499 diverse questions** across 12 topic clusters: Gaza, US government policy, economics/markets, Iran, court cases, climate (COP30), space launches, Russia-Ukraine, extreme weather, etc.
- **96% verifiable and unambiguous** — exceeds the quality rate of Metaculus, the leading human-curated platform.
- Expert forecaster review (n=149 questions reviewed): 75.2% "accept," 16.8% "soft reject" (trivial), 8.1% "hard reject" (ambiguous).
- Resolution accuracy: **~95%** vs. human expert resolution — the automated resolver agrees with human judges 95% of the time.

**Forecasting results on the auto-generated questions**:

| Model | Brier Score | Interpretation |
|---|---|---|
| **Gemini 3 Pro** | **0.134** | Best overall — approaches superforecaster range |
| GPT-5 | 0.149 | Comparable to aggregated human crowd |
| Gemini 2.5 Pro | 0.163 | Solid but below frontier |
| GPT-5 Mini | 0.171 | Smaller model, worse performance |
| Gemini 2.5 Flash | 0.179 | Speed-optimized model, accuracy trade-off |

- Rankings were **100% stable** under bootstrap resampling (10,000 iterations). Gemini 3 Pro ranked first in every single resample.
- **Subquestion decomposition boost**: Breaking questions into 3-5 sub-questions, researching each separately, then recombining improved Gemini 3 Pro from **0.141 → 0.132** Brier. This is statistically significant (better in 94.4% of bootstrap samples) and mirrors the "Fermi estimation" strategy used by human superforecasters (see Entry 14 below).

**Why it matters for our project**: ★★★★★ — Solves the data bottleneck. We can generate thousands of geopolitical forecasting questions without human curation. The multi-stage verification pipeline is a template we should adopt. The subquestion decomposition finding confirms that structured reasoning improves forecasting (a design pattern we should build into our pipeline).

---

### Entry 12: Mantic — State-of-the-Art AI Geopolitical Forecasting (Thinking Machines Lab)

**URL**: [mantic.com](https://mantic.com)
**Key article**: [Forecasting the Iran Crisis in Real Time](https://www.mantic.com/forecasting-iran-crisis) by Gabriel Fritsch
**Coverage**: [The Atlantic](https://www.theatlantic.com/technology/2026/02/ai-prediction-human-forecasters/685955/), [The Guardian](https://www.theguardian.com/technology/2025/sep/20/british-ai-startup-beats-humans-in-international-forecasting-competition), [Time](https://time.com/7318577/ai-model-forecasting-predict-future-metaculus/)

**What this is**: Mantic is a commercial AI forecasting system built by Thinking Machines Lab (London). It is currently the highest-performing AI forecasting system in competitive human-vs-AI tournaments.

**Track record**:
- **4th place** in the Metaculus Cup (Fall 2025) — outperformed **535 human participants** including 6 professional forecasters. Highest AI score ever recorded. More than double the score of the next-best AI entrant.
- **8th place** in the Metaculus Cup (Summer 2025).

**Three key innovations**:

**Innovation 1: Probability distributions over time**
Traditional forecasting produces a single number: "70% chance this happens." Mantic produces a *probability curve over time*: P(event happens by date X) for every date X. This is dramatically more informative — it tells you not just *whether* something will happen, but *when* the probability shifts.

**Innovation 2: Tripwire system**
Automated monitors on news sources, social media feeds, and prediction market prices run 24/7. When a relevant development is detected (a news article, a price movement, a government statement), the system automatically triggers a fresh forecast. This means Mantic can update predictions faster than any human — within minutes of a development, not hours or days.

**Innovation 3: Autonomous question generation**
As crises cascade (one event triggers another, which triggers another), Mantic automatically identifies and generates new downstream forecasting questions. For example, during the Iran crisis: airstrikes → Strait of Hormuz closure → LNG supply disruption → European energy prices → Fed monetary policy impact. Mantic generated questions about each cascade stage, often *before any prediction market existed for those questions*.

**The Iran Crisis Case Study (February–March 2026)** — the first major geopolitical crisis with live AI forecasting:

| Event | Mantic Forecast | What Happened |
|---|---|---|
| US strike on Iran by mid-2026 | ~70% (comparable to prediction markets) | ✅ Strikes occurred Feb 28, 2026 |
| Strike within 24 hours (midnight Feb 28) | 15% → **48%** (6:17am, triggered by market price movements) | ✅ Strikes confirmed minutes after the final update |
| Strait of Hormuz closure | 20% → **80%** post-strike | ✅ Closed March 3, 2026 |
| Military attack on commercial vessel | **83%** by March 4 | ✅ Safeen Prestige struck March 4 |
| QatarEnergy force majeure | ~70% by March 2 | ✅ Force majeure declared March 4 |

**The critical demonstration**: The tripwire-triggered update from 15% → 48% happened *minutes before the news broke*, driven by automated detection of prediction market price movements. The system identified the signal before human analysts.

**Limitations**:
- Commercial system — full methodology not published in peer-reviewed literature.
- Self-reported accuracy (no independent audit beyond Metaculus Cup rankings).
- Iran crisis is a single (dramatic) case study — more diverse validation needed.

**Why it matters for our project**: ★★★★★ — Mantic represents the state-of-the-art in exactly our domain: AI geopolitical forecasting. The tripwire system, temporal probability distributions, and autonomous question generation are directly relevant design patterns we should study and potentially adopt.

---

## Thread 3: The Human Baseline — Superforecasting Literature

> **What this thread covers**: Before we can say "the AI is good at predicting," we need to know what "good" looks like. The superforecasting literature establishes the gold-standard human baseline that every AI system is measured against. This thread covers the research that identified who the best human predictors are, what makes them accurate, and what their actual scores look like.
>
> **Why this is the baseline, not expert opinion**: The single most important finding in forecasting research is that *most experts are terrible at prediction*. Decades of research show that domain experts (political scientists, economists, intelligence analysts) consistently fail to outperform simple statistical models. The tiny fraction who do predict well — "superforecasters" — are the real benchmark.

---

### Entry 13: Tetlock (2005) — *Expert Political Judgment: How Good Is It? How Can We Know?*

**Citation**: Tetlock, P.E. (2005). *Expert Political Judgment: How Good Is It? How Can We Know?* Princeton University Press.

**What this is**: The most influential book in prediction science. Philip Tetlock, a psychologist at the University of Pennsylvania (now Wharton), spent 20 years tracking the predictions of professional experts to answer a simple question: are experts actually good at predicting the future?

**The study — 20 years, 284 experts, 28,000 predictions**:
- **Who participated**: 284 domain experts — political scientists, economists, intelligence analysts, journalists, and policy advisors. People who appear on television, brief governments, and are paid for their opinions about what will happen next.
- **What they predicted**: Roughly 28,000 predictions about political and economic events — elections, wars, economic indicators, regime changes, policy outcomes. Each prediction was a probability estimate (e.g., "65% chance that the Soviet Union will collapse by 2000").
- **How long it ran**: From 1984 to 2003, making this the longest scientific study of prediction accuracy ever conducted.

**The headline finding**: Expert predictions were barely better than chance — famously summarized as **"roughly as accurate as a dart-throwing chimpanzee."** More precisely, experts performed slightly better than a simple algorithm that always predicted the historical base rate (e.g., if trade wars escalate 30% of the time, always predict 30%), but not by much.

**The Foxes vs. Hedgehogs distinction** (borrowed from the philosopher Isaiah Berlin):
- **Hedgehogs** — experts who viewed the world through one big idea (e.g., "everything is about power politics" or "markets always self-correct"). They were *the worst* predictors, performing below chance on some categories. They were also the most confident and the most likely to appear on television.
- **Foxes** — experts who drew on many frameworks, were comfortable with uncertainty, and frequently updated their views. They were significantly better, outperforming hedgehogs by ~15% in accuracy.

**Why this matters for our project**: This establishes the bar LLMs need to beat. If the comparison is LLMs vs. "domain experts," the bar is surprisingly low — because most domain experts are genuinely bad at prediction. The more meaningful comparison is LLMs vs. *superforecasters* (the top 2% identified in the follow-up study below).

---

### Entry 14: Tetlock & Gardner (2015) — *Superforecasting: The Art and Science of Prediction*

**Citation**: Tetlock, P.E. & Gardner, D. (2015). *Superforecasting: The Art and Science of Prediction.* Crown.

**Why this book exists**: After *Expert Political Judgment* showed that most experts fail at prediction, Tetlock was funded by IARPA (Intelligence Advanced Research Projects Activity — the intelligence community's research arm, like DARPA but for spying) to find out if *anyone* could consistently predict geopolitical events. The result was the **Good Judgment Project (GJP)** — a multi-year forecasting tournament that became the most rigorous test of human prediction ability ever conducted.

**The IARPA ACE Tournament (2011–2015)**:
- **What it was**: The Aggregative Contingent Estimation (ACE) program. IARPA funded five competing research teams to develop forecasting methods. Each team recruited forecasters who answered hundreds of geopolitical questions per year.
- **Scoring method**: Daily Brier scores, averaged per question, then averaged across all questions weighted equally. On the IARPA scale, Brier ranges from 0 (perfect) to 2 (perfectly wrong), with 0.5 representing random guessing — slightly different from the standard 0–1 scale used elsewhere in this review.
- **Scale**: Hundreds of geopolitical questions per year, covering international relations, conflicts, elections, and economic events.

**The discovery of superforecasters**:
The top 2% of forecasters — roughly 60 people out of thousands — were dramatically better than everyone else. Tetlock called them **"superforecasters."** Their performance was not a fluke; they maintained their edge year after year.

**How good were superforecasters?**

| Comparison | Result |
|---|---|
| vs. average forecasters | Superforecasters were **60-78% more accurate** (measured by Brier score improvement) |
| vs. intelligence analysts *with classified information* | Superforecasters outperformed them by **more than 30%** — despite having *zero* access to classified data |
| vs. prediction markets | Superforecasters outperformed the intelligence community's internal prediction market by **25-30%** |
| vs. other research teams | GJP beat all four competing IARPA teams by **35-72%** |
| vs. time advantage | Superforecasters at **300 days** out were more accurate than regular forecasters at **100 days** out — they could predict farther ahead with the same accuracy |

**The seven traits of superforecasters** (the skills our LLM pipeline should try to replicate):

| Trait | What It Means | How an LLM Might Replicate It |
|---|---|---|
| **1. Probabilistic thinking** | Think in granular probabilities — "73%" not "likely." Make frequent small updates as new evidence arrives. | LLMs natively output probabilities. The challenge is making them *granular* rather than defaulting to round numbers (50%, 70%, 90%). |
| **2. Outside view (base rates)** | Start with: "How often has this type of event happened historically?" Then adjust. | Provide the LLM with base-rate data in its context. Use retrieval to find historical frequencies. |
| **3. Decomposition** | Break complex questions into smaller, answerable sub-questions. Estimate each, then recombine. (Also called "Fermi estimation.") | Bosse et al. (Entry 11) confirmed this works for LLMs — subquestion decomposition improved Brier from 0.141 → 0.132. |
| **4. Active open-mindedness** | Actively seek disconfirming evidence. Steel-man the opposing view. | Prompt the LLM to argue both for and against the prediction before committing to a probability. |
| **5. Frequent updating** | When new evidence arrives, make small adjustments rather than ignoring it or overreacting. | Mantic's tripwire system (Entry 12) automates this — triggering fresh forecasts when new information is detected. |
| **6. Self-critical humility** | Track your own accuracy. Acknowledge uncertainty. Avoid overconfidence. | Post-hoc calibration (Domain 3, Entries 17-19) can correct systematic overconfidence. |
| **7. Teamwork** | Diverse groups that debate constructively outperform individuals. | LLM ensembles (Halawi, Entry 8): aggregating predictions from diverse models mirrors team forecasting. |

**Brier score benchmarks for superforecasters**:
- IARPA tournament (0-2 scale): 0.15–0.20 for superforecasters vs. 0.37 for average forecasters.
- ForecastBench (0-1 scale): **0.071–0.081** for superforecasters — this is the number our LLMs are chasing.

**Why it matters for our project**: ★★★★★ — Superforecasters are the gold standard. Their traits map to specific LLM capabilities we can engineer: decomposition (chain-of-thought prompting), outside view (retrieval of base rates), ensembles (silicon crowd), and frequent updating (tripwire systems). Our evaluation framework must benchmark against their scores.

---

### Entry 15: The IARPA ACE Methodology — How the Gold-Standard Tournament Was Scored

**Why this deserves its own entry**: Many papers cite "superforecaster accuracy" without explaining *how* it was measured. The scoring methodology matters because different scoring methods can produce different rankings. Understanding IARPA's specific approach ensures we make valid comparisons.

**The IARPA Brier Scoring Protocol** (Mellers et al., 2014):
1. **Daily Brier scores**: For each question and each day the question is active, a Brier score is calculated based on the forecaster's most recent prediction. If they haven't updated, their previous prediction "carries over."
2. **Per-question mean**: The daily Brier scores are averaged across all active days to produce a single score per question. This rewards forecasters who are accurate early (before the answer becomes obvious) and penalizes those who are only accurate at the last minute.
3. **Imputation rule**: To prevent "strategic waiting" (submitting predictions only when you're confident), the methodology imputes the group-average Brier score for days before a forecaster's first prediction. This means that *not predicting early* is penalized — you can't game the system by waiting.
4. **Overall score**: The per-question means are averaged across all questions, weighted equally.

**Why this matters**: The daily-averaging and imputation rules mean IARPA Brier scores are not directly comparable to "single-shot" Brier scores from benchmarks like ForecastBench (which ask for one prediction per question). When we compare superforecaster accuracy to LLM accuracy, we need to check whether the scoring methods are comparable.

---

### Entry 16: Schoenegger et al. (2024) — Human-AI Hybrid Forecasting

**Citations**:
- Schoenegger, P., Park, P.S., Karger, E., & Tetlock, P.E. (2024). "AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy." arXiv:2402.07862.
- Schoenegger, P., Park, P.S., Karger, E., & Tetlock, P.E. (2024). "Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy." Wharton School.

**What they tested**: Instead of asking "Can AI replace human forecasters?", these papers ask the more practical question: **"Can AI make human forecasters better?"** And the reverse: can human base rates make AI better?

**Study 1 — "AI-Augmented Predictions"**:
- **Setup**: Human forecasters (educated, numerate adults — not elite superforecasters) were randomly assigned to one of three conditions: (1) no AI assistant, (2) a basic LLM assistant, or (3) a "superforecasting" LLM assistant that used structured reasoning prompts.
- **Result**: The superforecasting LLM assistant boosted human accuracy by **24-28%** vs. the no-assistant control group. With one outlier question excluded, the boost reached **41%**.
- **The interaction is bidirectional**: Humans benefit from LLM reasoning and information retrieval. LLMs benefit from human priors (base rates, contextual judgment). The combination outperforms either alone.

**Study 2 — "Wisdom of the Silicon Crowd"**:
- **Finding**: Aggregating predictions from multiple diverse LLMs (GPT-4, Claude 2, etc.) — the "silicon crowd" — achieves accuracy that **rivals aggregated non-expert human crowd predictions** on Metaculus.
- **Gap**: The silicon crowd still lags superforecasters by ~30%.
- **Important caveat**: The human forecasters in these studies were **educated, numerate adults** — not the general public. The LLM boost may be smaller for already-expert forecasters.

**Why it matters for our project**: ★★★★☆ — The best current paradigm is human-AI hybrid, not pure AI. Our evaluation should test both (1) LLM-only and (2) LLM-assisted human prediction. The bidirectional finding suggests our pipeline could include a "human-in-the-loop" review step for high-stakes predictions.

---

## Thread 4: Prediction Markets as Baselines

> **What this thread covers**: Prediction markets — platforms where people bet money (or reputation points) on future events — are one of the strongest existing forecasting mechanisms. Understanding their accuracy, limitations, and failure modes is essential for establishing baselines in our evaluation framework.
>
> **What is a prediction market?** A marketplace where people buy and sell contracts that pay out based on whether a future event occurs. If you buy a "Will X happen?" contract at $0.70, you pay $0.70 and receive $1.00 if X happens, or $0.00 if it doesn't. The market price ($0.70 in this case) represents the crowd's collective probability estimate (70%). The theory is that people with better information will trade more aggressively, pushing the price toward the "true" probability — this is called **information aggregation** (Wolfers & Zitzewitz, 2004).

---

### Entry 17: Prediction Market Accuracy — Platform Comparison

**The foundational theory — Wolfers & Zitzewitz (2004)**:
Justin Wolfers (University of Michigan) and Eric Zitzewitz (Dartmouth) published "Prediction Markets" in the *Journal of Economic Perspectives*, establishing the theoretical foundation for why prediction markets work:
- Markets aggregate **dispersed information** — no single trader knows everything, but the collective price reflects all traders' information combined.
- Market prices approximate **mean beliefs** — the price roughly equals the average probability estimate across all traders.
- In practice, prediction markets produce forecasts that are "quite accurate" and "surpass most moderately sophisticated benchmarks."
- However, markets are *not* perfectly efficient. Thin markets (few traders), manipulation, and behavioral biases can cause prices to deviate from true probabilities.

**Major prediction market platforms and their accuracy**:

| Platform | How It Works | Accuracy Record | Key Limitation |
|---|---|---|---|
| **Polymarket** | Cryptocurrency-based prediction market, unregulated. Users trade contracts with crypto. Largest by trading volume. | ~90% accuracy at 1 month out; ~94% at 12 hours. Brier ≈ 0.058 at 12h before resolution. | Susceptible to "noise trading" (uninformed bets) and manipulation. Vanderbilt study found only 67% accuracy on political markets. |
| **Kalshi** | CFTC-regulated event exchange (legally overseen by the U.S. government). Users trade with real USD. | 78% accuracy (political markets, Vanderbilt study). Fed Reserve Board study (Feb 2026): rivals professional economic forecasts. | More liquid and regulated, but limited contract selection. |
| **Metaculus** | Reputation-based, no real money. Users earn reputation points for accurate predictions. | Better calibration than financial markets on long-term questions. Slower to update (no financial incentive for speed). | Strong community baseline but less responsive to breaking news. |

**Key research findings on prediction market accuracy**:
- **Fed Reserve Board study (Feb 2026)**: Kalshi macroeconomic prediction markets rival or exceed professional economist forecasts for CPI (inflation) and GDP releases. This is significant because professional economic forecasters are well-resourced and highly motivated.
- **Vanderbilt study (2024 election)**: Polymarket accuracy = 67% for active political markets; PredictIt = 93%; Kalshi = 78%. The finding that Polymarket — the most popular and liquid market — performed *worst* on political predictions suggests that high liquidity attracts noise trading (uninformed speculation).
- **Long-term predictions**: For forecasts beyond 12 months, expert panels (superforecasters) show stronger calibration than prediction markets, perhaps because markets discount distant events too heavily.

**The baseline hierarchy for our project**:
When evaluating whether an LLM adds value, we need to compare it against increasingly difficult baselines. Each level in this hierarchy is harder to beat:

| Level | Baseline | Brier Score (approx.) | What Beating It Proves |
|---|---|---|---|
| 1 | **Random** (always predict 50%) | 0.250 | The model is doing something — not just guessing |
| 2 | **Always-predict-majority** (predict the most common outcome) | ~0.20 | The model is paying attention to the specific question, not just predicting "no" for everything |
| 3 | **Persistence** (predict same as last time) | ~0.18 | The model captures change — not just assuming the status quo continues |
| 4 | **Historical base rate** (use the frequency of similar past events) | ~0.16 | The model adds case-specific reasoning beyond generic statistics |
| 5 | **Market price** (Polymarket/Kalshi/Metaculus) | 0.058–0.165 | The model has information the collective market doesn't |
| 6 | **Superforecaster consensus** | 0.071–0.081 | The model rivals the best humans alive at prediction |

**BSS (Brier Skill Score) against market prices** is the critical test: positive BSS = the LLM adds predictive information beyond what the market already knows. For most world leader decisions, *no prediction market exists*, so we'll primarily test against baselines 1-4.

---

## Thread 5: Current State of the Art & The Gap

> **What this thread covers**: A synthesis of where things stand right now (March 2026), how fast the gap between AI and human forecasters is closing, and what specific weaknesses remain.

---

### Entry 18: Summary of Current Brier Scores Across the Literature

This table compiles accuracy benchmarks from all the papers and platforms discussed in this review, ranked from best to worst:

| Forecaster | Brier Score | Source | What This Tells Us |
|---|---|---|---|
| Perfect | 0.000 | — | Theoretical ideal |
| Polymarket 12h pre-resolution | ~0.058 | Polymarket data | Prediction markets just before resolution — near-certain outcomes |
| Superforecasters (ForecastBench full) | **0.071** | Karger et al. (2024) | The best humans on a large, diverse test set |
| Superforecasters (ForecastBench tournament) | **0.081** | ForecastBench (Oct 2025) | Same group, different time period |
| Cassi ensemble (crowd-adjusted) | ~0.095 | ForecastBench (Mar 2026) | AI system with access to human crowd data |
| GPT-4.5 | **0.101** | ForecastBench (Oct 2025) | Best LLM in late 2025 |
| Claude 3.5 Sonnet | **0.117** | ForecastBench (Jan 2026) | Anthropic's frontier model |
| Superforecasters (Karger comparable) | **0.121** | Karger et al. (2025) | Different test set, different time period |
| Gemini 3 Pro (auto-gen) | **0.134** | Bosse et al. (2026) | Google's model on automated questions |
| o3 (OpenAI) | **0.135** | Halawi et al. | On Metaculus question set |
| Human crowd (aggregate) | **0.149** | Halawi et al. (2024) | Thousands of non-expert forecasters combined |
| Polymarket crowd (Turtel test) | **0.165** | Turtel et al. (2025) | Prediction market on a specific test set |
| Foresight-32B (RL-trained) | ~0.190 | Turtel et al. (2025/2026) | Small RL-trained model, different test set |
| Early GPT-4 | ~0.25 | Halawi et al. (2024) | Roughly random performance |
| Random / Uninformed | 0.250 | — | Always predicting 50% |
| GPT-3-level | >0.25 | Halawi et al. | Worse than random due to overconfidence |

---

### Entry 19: Eight Key Takeaways — What the Data Tells Us

**Takeaway 1: The gap is closing fast.**
From a 75%+ gap (GPT-3 vs. superforecasters) in 2023 to a ~21% gap (best frontier LLMs vs. superforecasters) in early 2026, with **parity projected around November 2026** via linear extrapolation.

**Takeaway 2: LLMs already beat human crowds.**
The aggregate of non-expert human predictions (Brier ~0.149) is now worse than frontier LLMs (~0.095-0.101 with crowd augmentation). The remaining gap is specifically versus the **top 2%** of human forecasters.

**Takeaway 3: The biggest LLM weakness is overconfidence.**
GPT-3-level models were *worse than random* (>0.25 Brier) because they assigned extreme probabilities to wrong outcomes. Even frontier models tend toward overconfidence. **Post-hoc calibration** (temperature scaling, Platt scaling — see Domain 3, Entries 17-19) is not optional; it is a mandatory component of any LLM forecasting pipeline.

**Takeaway 4: Ensembles and retrieval matter more than model size.**
A single model performs significantly worse than an optimized pipeline with retrieval (RAG) + structured reasoning + ensemble aggregation. The methodological gap may be larger than the model capability gap.

**Takeaway 5: Training objective > model size.**
Turtel et al. showed that a 14B model trained with RL beats o1 (~100x larger), and a 32B RL-trained model beats a 235B pretrained model. For prediction tasks, *how you train* matters more than *how big the model is*.

**Takeaway 6: Domain-specific performance varies.**
LLMs are closer to parity on economic/quantitative questions (FRED data, financial metrics) than on geopolitical/political questions where context interpretation and informal knowledge matter more. **This is directly relevant to our project** — political leader decisions are harder to predict than macroeconomic indicators.

**Takeaway 7: Automated question generation at scale is solved.**
Bosse et al. (2026) produces questions at 96% quality with 95% resolution accuracy, enabling continuous evaluation without human curation bottlenecks. We can generate thousands of training and test questions programmatically.

**Takeaway 8: Subquestion decomposition reliably improves accuracy.**
Both Bosse et al. and Tetlock's superforecaster research confirm that breaking complex questions into smaller sub-questions and aggregating answers consistently improves prediction. This should be a standard component of our pipeline.

---

### Entry 20: Where the Gap Persists — Known Weaknesses of LLM Forecasting

Despite rapid improvement, LLMs systematically struggle in four specific areas. Our project falls squarely into several of these:

| Weakness | Why LLMs Struggle | Relevance to Our Project |
|---|---|---|
| **Long time horizons (>3 months)** | Compounding uncertainty makes long-range prediction fundamentally harder. LLMs have no mechanism for modeling how events chain together over months. | HIGH — many leader decisions are months away. We should start with short-horizon predictions (next meeting, next week) and gradually extend. |
| **Novel, unprecedented events** | If nothing similar has happened before, base rates don't exist and pattern matching fails. LLMs are fundamentally pattern-matching systems. | MEDIUM — some leader decisions (crisis responses, first-of-kind policies) have no historical precedent. |
| **Private signals** | Some predictions depend on classified intelligence, closed-door negotiations, or private conversations that no public data source captures. | HIGH — world leader decisions are often made behind closed doors. Our predictions are inherently limited by what's publicly observable. |
| **Very low base rates** | Events with <5% base rates are deceptively easy to "predict" — just always say "no." Brier scores look good but provide no useful signal. | MEDIUM — vetoes, surprise policy reversals, and unprecedented diplomatic moves are rare but consequential. We need metrics that specifically test rare-event prediction (see Domain 3, Thread 6 on ROC/AUC). |

---

## Top Papers to Read in Full (Ranked by Importance to Our Project)

| Priority | Paper | Why It's Essential |
|---|---|---|
| 1 | **Halawi et al. (2024)** — "Approaching Human-Level Forecasting" | Definitive LLM-human comparison with hard numbers |
| 2 | **Turtel et al. (2025)** — "Outcome-based RL to Predict the Future" | Proves RLVR works for prediction; 14B beats o1 |
| 3 | **Turtel et al. (2026)** — "Future-as-Label" | Scalable outcome-based training; 32B beats 235B |
| 4 | **Karger et al. (2024)** — ForecastBench | The primary benchmark; contamination-free design |
| 5 | **Bosse et al. (2026)** — "Automating Question Generation" | Automated eval pipeline; 1,499 questions at 96% quality |
| 6 | **Tetlock & Gardner (2015)** — *Superforecasting* | Gold-standard human baseline; GJP methodology |
| 7 | **Tetlock (2005)** — *Expert Political Judgment* | Establishes that expert prediction is generally poor |
| 8 | **UNBench (2024)** — UNSC Decision-Making | Most directly analogous benchmark to our task |
| 9 | **Mantic / Fritsch (2026)** — Iran Crisis Case Study | State-of-the-art real-time geopolitical AI forecasting |
| 10 | **Jin et al. (2024)** — MIRAI | RAG-based conflict forecasting; retrieval > reasoning |
| 11 | **Schoenegger et al. (2024)** — "AI-Augmented Predictions" | Human-AI hybrid outperforms either alone |
| 12 | **Pratt et al. (2024)** — "Can LLMs Use Forecasting Strategies?" | Important negative result: strategy prompting ≠ improvement |
| 13 | **Schoenegger et al. (2024)** — "Wisdom of Silicon Crowd" | LLM ensemble accuracy; silicon crowd concept |
| 14 | **Wolfers & Zitzewitz (2004)** — "Prediction Markets" | Foundational theory for market-based baselines |
| 15 | **Xu et al. (2024)** — Prophet Arena | Live forecasting with economic value scoring |
