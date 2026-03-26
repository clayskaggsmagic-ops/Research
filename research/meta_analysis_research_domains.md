# Literature Review Prompts

> **Instructions**: Run each prompt in a separate deep-research session (Gemini Deep Research, ChatGPT Deep Research, or Perplexity). Each prompt covers one domain. Save each output as its own document. Together they form the complete literature review.

---

## Prompt 1: Political Decision Theory & Predictability

> I am researching whether AI can predict the decisions of world leaders. Before testing anything, I need to understand what political science says about **which political decisions are even theoretically predictable**.
>
> Conduct a comprehensive literature review covering:
>
> **Sub-questions to answer:**
> - What makes a political decision non-stochastic (logically structured vs. random)?
> - Which decision types have the highest ex-ante predictability (votes, budgets, sanctions, military deployments)?
> - What does political science say about the limits of prediction in complex adaptive systems?
> - How do scholars categorize decisions by frequency, observability, and boundedness?
>
> **Research threads to cover:**
> - **Bargaining vs. Procedural Models**: Different model types predict political outcomes with varying accuracy; bargaining models emphasize informal negotiations, procedural models focus on institutional rules
> - **Game Theory / Expected Utility Models**: Bruce Bueno de Mesquita's work on predicting political outcomes using game-theoretic models of actor interests, priorities, and influence — his claimed accuracy rates, methodology, and criticisms
> - **Bounded Rationality in IR**: Kahneman/Tversky-influenced work on how cognitive biases constrain leader decision-making into predictable patterns
> - **Political Instability Forecasting**: Computational models (e.g., Political Instability Task Force) that predict instability based on macro factors — what "predictable" means in this context
> - **Limits of Social Prediction**: The "Fragile Families" study (Salganik et al., PNAS 2020) showing that even with rich data + ML, life outcome predictions hit practical limits. What does this imply for political prediction?
> - **Crisis vs. Routine Decision-Making**: Are routine decisions (scheduled votes, budget approvals) fundamentally more predictable than crisis decisions (military strikes, emergency sanctions)?
> - **Institutional Constraints on Predictability**: Do democracies produce more predictable decisions than autocracies? How do veto players, coalition politics, and bureaucratic politics affect predictability?
>
> **Search terms to use:**
> - "predictability political decision making"
> - "computational models political outcomes"
> - "game theory international relations prediction Bueno de Mesquita"
> - "bounded rationality foreign policy decision making"
> - "limits of prediction complex social systems"
> - "crisis vs routine foreign policy decision making"
> - "Political Instability Task Force forecasting"
>
> **For every paper or system you find, provide**: (1) full citation with author, year, and title, (2) a link or DOI if available, (3) a 2-3 sentence summary of the key finding, and (4) why it matters for predicting world leader decisions with AI. Organize your output by research thread. Aim for 10-20 key papers total.

---

## Prompt 2: Leader Profiling & Operational Code Analysis

> I am researching whether AI can predict the decisions of specific named world leaders. I need a comprehensive literature review on **how political psychology has modeled individual leader decision-making** — the methods used to profile leaders "at a distance" and predict their behavior from public data.
>
> **Sub-questions to answer:**
> - How has political psychology quantified individual leader belief systems and decision styles?
> - What is Operational Code Analysis and how has it been used to predict leader behavior?
> - What automated/computational methods exist for profiling leaders from text?
> - How accurate have these profiling methods been at actually predicting decisions?
>
> **Research threads to cover:**
> - **VICS Operational Code Analysis** (Walker, Schafer): How it quantifies leader beliefs via verb coding — philosophical beliefs (view of political universe) and instrumental beliefs (preferred strategies). How this has been automated with NLP (Social Science Automation). Specific studies where VICS predictions were tested against real outcomes.
> - **Leadership Trait Analysis** (Margaret Hermann): The 7 personality dimensions (belief in control, need for power, conceptual complexity, distrust, in-group bias, self-confidence, task focus). How these traits are coded from speech at a distance. Validation studies.
> - **Integrative Complexity** (Peter Suedfeld): How cognitive complexity measured in leader speech correlates with escalation, crisis behavior, and decision quality. Studies linking IC scores to actual outcomes.
> - **At-a-Distance Assessment**: The broader field of profiling leaders without direct access — methods, validity, criticisms.
> - **Computational / Automated Profiling**: Any work using NLP, machine learning, or LLMs to automate leader profiling (automated VICS, sentiment analysis of speeches, etc.)
> - **Connecting Profiles to Predictions**: Has anyone actually used these profiles as inputs to predictive models? What worked, what didn't?
>
> **Search terms to use:**
> - "VICS Verbs in Context System operational code analysis"
> - "leadership trait analysis Hermann political psychology"
> - "integrative complexity political leaders Suedfeld"
> - "at-a-distance assessment foreign policy leaders"
> - "automated operational code analysis NLP"
> - "predicting leader behavior political psychology"
>
> **For every paper or system you find, provide**: (1) full citation with author, year, and title, (2) a link or DOI if available, (3) a 2-3 sentence summary of the key finding, and (4) why it matters for using LLMs to predict specific leader decisions. Organize by research thread. Aim for 10-20 key papers.

---

## Prompt 3: Prediction Scoring, Calibration & Evaluation Methodology

> I am designing an evaluation framework to test whether LLMs can predict world leader decisions. Before I run any experiments, I need a comprehensive literature review on **how predictions are measured and scored across all of science** — the mathematical tools for saying "this prediction was good" or "this prediction was bad."
>
> **Sub-questions to answer:**
> - What is a proper scoring rule and why does it matter for honest probability reporting?
> - What is the difference between accuracy, calibration, resolution, and discrimination?
> - How do you diagnose *why* a model is good or bad at prediction (not just *that* it is)?
> - When should you use probabilistic metrics vs. categorical metrics?
> - How do you fix a model that discriminates well but is poorly calibrated?
>
> **Research threads to cover:**
> - **Proper Scoring Rules**: Brier score, log loss (negative log likelihood), CRPS — what they are, mathematical definitions, when to use each, and why "properness" matters (Gneiting & Raftery 2007)
> - **Brier Score Decomposition** (Murphy 1973): Breaking the Brier score into Reliability (calibration), Resolution (discrimination), and Uncertainty — what each component tells you diagnostically about a forecaster
> - **Calibration Assessment**: Reliability diagrams, Expected Calibration Error (ECE), Maximum Calibration Error (MCE) — how to visualize and quantify whether predicted probabilities match observed frequencies
> - **Post-Hoc Calibration Methods**: Platt scaling, temperature scaling, isotonic regression — what they do, when to use which, tradeoffs (Guo et al. 2017, Platt 1999)
> - **Classification Metrics**: Accuracy, F1, macro-F1, precision, recall, AUC-ROC — when each is appropriate, especially for imbalanced outcomes (e.g., UNSC vetoes are rare events)
> - **Forecast Verification in Practice**: How the weather forecasting community (WMO standards), medical diagnostics (sensitivity/specificity), and ML research each approach this differently
> - **Comparing Forecasters**: Statistical tests for comparing two forecasters' performance (Diebold-Mariano test, permutation tests, confidence intervals on Brier scores)
>
> **Search terms to use:**
> - "proper scoring rules forecasting Gneiting Raftery"
> - "Brier score decomposition reliability resolution uncertainty Murphy"
> - "calibration reliability diagram machine learning"
> - "Expected Calibration Error ECE"
> - "temperature scaling calibration neural networks Guo"
> - "forecast verification methodology"
> - "Diebold-Mariano test forecast comparison"
>
> **For every paper, method, or metric you cover, provide**: (1) full citation, (2) link or DOI, (3) 2-3 sentence summary, (4) when you would and would NOT use it for evaluating political decision prediction specifically. Organize by research thread. Aim for 15-25 key papers/concepts.

---

## Prompt 4: LLM Forecasting Benchmarks, Prediction Arenas & Superforecasting

> I am evaluating whether LLMs can predict world leader decisions. I need a comprehensive literature review on **every existing benchmark, platform, and competition that tests LLM prediction ability**, plus the human forecasting literature that establishes the baseline LLMs are measured against.
>
> **Sub-questions to answer:**
> - What platforms exist for testing LLM forecasting ability, and how do they score predictions?
> - How do current LLMs compare to human superforecasters on prediction tasks?
> - How do these benchmarks prevent data leakage and contamination?
> - What is the current state-of-the-art in LLM forecasting, and where is the gap?
>
> **Research threads to cover:**
> - **ForecastBench** (Karger et al.): How it works, scoring (Brier + log loss), contamination-free guarantee, auto-generated questions from FRED/ACLED/Yahoo Finance + prediction market questions from Metaculus/Polymarket, current leaderboard results
> - **Prophet Arena** (University of Chicago): 3-stage pipeline (info sourcing → prediction → resolution), Brier scoring, simulated betting ROI using Kalshi prices, IRT scoring, human-AI collaboration
> - **PredictionArena.ai**: Real Kalshi market trading with $10K portfolios, mark-to-market ranking
> - **MIRAI**: RAG-based international event forecasting — how retrieval quality affects prediction
> - **UNBench**: UN Security Council prediction (voting, co-penholder, draft adoption) — 1994-2024
> - **WORLDREP**: Country relationship prediction (cooperate/conflict) from news articles
> - **Forecaster Arena / Polymarket AI competitions**: AI competing directly in prediction markets
> - **Mantic** (mantic.com): AI superforecaster that placed 4th against human forecasters in a major tournament (covered by The Atlantic, Feb 2026). Live-tested during 2026 Iran crisis — generated real-time probability updates for US military strikes, Strait of Hormuz closure, oil supply disruptions. Uses "tripwires" to auto-trigger forecast updates from news/market signals. Forecasts events as probability distributions over time, not just point estimates. Key case study: updated Iran strike probability from 15% → 48% minutes before news broke. See: https://www.mantic.com/forecasting-iran-crisis
> - **Superforecasting Literature**: Tetlock (2005) *Expert Political Judgment*, Tetlock & Gardner (2015) *Superforecasting*, the Good Judgment Project — traits of superforecasters, how they're scored, what accuracy they achieve
> - **Human + AI Hybrid Forecasting**: Tetlock et al. (2025 Wharton study) — does LLM assistance improve human forecasters?
> - **Halawi et al. (2024) "Approaching Human-Level Forecasting"**: Current Brier score gap between LLMs and superforecasters, projected parity timeline
> - **Prediction Markets as Baselines**: Polymarket, Kalshi, Metaculus accuracy — how market prices serve as probability baselines
>
> **Search terms to use:**
> - "ForecastBench LLM forecasting benchmark"
> - "Prophet Arena live forecasting AI"
> - "LLM forecasting accuracy superforecasters"
> - "Tetlock superforecasting Good Judgment Project"
> - "human AI hybrid forecasting accuracy"
> - "prediction market accuracy geopolitical events"
> - "UNBench UN Security Council prediction"
> - "MIRAI international event forecasting LLM"
> - "approaching human-level forecasting LLM Halawi"
>
> **For every benchmark, platform, or paper you find, provide**: (1) full citation or URL, (2) how it scores predictions (which metrics), (3) how it prevents cheating/leakage, (4) current results (numbers — Brier scores, accuracy percentages), (5) its limitations. Organize by platform/paper. Aim for 15-25 entries.

---

## Prompt 5: Analogous Prediction Domains & LLM Persona Simulation

> I am researching whether LLMs can predict the decisions of specific named world leaders. Before narrowing to political leaders, I need a comprehensive literature review on **every analogous domain where researchers predict the decisions of named individuals or small identifiable bodies**, plus research on LLMs role-playing as specific personas and simulating political actors.
>
> **Sub-questions to answer:**
> - What other fields predict the behavior of specific named decision-makers, and how accurate are they?
> - Can LLMs faithfully simulate a specific person's beliefs and decision patterns?
> - What has been tried with LLMs simulating political actors, and what are the known failure modes?
> - What military/government programs are working on this problem at institutional scale?
>
> **Research threads to cover:**
>
> *Analogous Prediction Domains:*
> - **Supreme Court / Judicial Prediction**: ML models predicting how specific justices will vote (Katz et al., Martin-Quinn scores, Pre/Dicta) — accuracy rates (70-75%), what features drive predictions, failure on novel cases
> - **Central Bank Rate Decisions**: Predicting Fed/ECB/BoE decisions — fed funds futures baseline accuracy (~85-90%), how forward guidance affects predictability, market vs. model vs. expert accuracy
> - **Corporate CEO Decision Prediction**: Any research predicting M&A, strategic pivots, or executive behavior from profiles + context
> - **Election Prediction with AI**: LLM-based election forecasting accuracy, comparison to polls and prediction markets
>
> *LLM Persona Simulation:*
> - **Silicon Sampling / Synthetic Participants**: LLMs given persona profiles to simulate human survey responses (Argyle et al., Bisbee et al.) — algorithmic fidelity, demographic conditioning, limitations, "random silicon sampling"
> - **LLM Political Actor Simulation**: WarAgent (historical conflict simulation), DiplomacyAgent (collective decision-making), Richelieu (self-evolving AI diplomacy), AgentTorch (MIT, million-agent simulation)
> - **Known Failure Modes**: Geopolitical bias from Western training data (CSIS 2025), escalation bias in crisis scenarios, persona shallowness, training cutoff decay
> - **CIA Foreign Leader Chatbots**: Reported initiative to build LLMs that predict foreign leader reactions from OSINT + intel
>
> *Military/Government AI Programs:*
> - **Ender's Foundry** (Dept. of War/PSP): AI-enabled simulation, AI vs. AI training for doctrine
> - **Thunderforge** (Defense Innovation Unit): AI wargaming for INDOPACOM & EUCOM
> - **GenWar Lab** (DoD): GenAI for defense wargaming
> - **USAF Wargaming Acceleration**: AI for 10,000x real-time simulation
>
> **Search terms to use:**
> - "predicting Supreme Court decisions machine learning accuracy"
> - "federal reserve rate decision prediction accuracy baseline"
> - "silicon sampling LLM synthetic participants algorithmic fidelity"
> - "WarAgent DiplomacyAgent LLM geopolitical simulation"
> - "LLM persona simulation political leaders"
> - "LLM bias escalation crisis international relations"
> - "Ender's Foundry AI simulation Department of Defense"
> - "Thunderforge Defense Innovation Unit wargaming"
> - "CIA foreign leader chatbot AI prediction"
>
> **For every paper, system, or program you find, provide**: (1) full citation or URL, (2) what was predicted or simulated, (3) what methods were used, (4) what accuracy was achieved (if applicable), (5) what baselines were compared against, (6) key lessons for our project. Organize by research thread. Aim for 15-25 entries.
