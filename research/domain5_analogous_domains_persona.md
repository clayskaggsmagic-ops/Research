# Domain 5: Analogous Prediction Domains & LLM Persona Simulation — Literature Review

> **Question this review answers**: What analogous domains predict the decisions of specific named individuals, can LLMs faithfully simulate specific personas, what are the known failure modes, what evaluation frameworks exist, and what military/government programs are working on this at institutional scale?

---

## Glossary of Key Terms

Before diving in, here are definitions of technical terms used throughout this review. Every term is explained in plain language the first time it appears, but this glossary serves as a quick reference.

| Term | Plain-Language Definition |
|---|---|
| **Silicon sampling** | Using an LLM to generate simulated survey responses from virtual people, instead of surveying real humans. The LLM is given a demographic backstory (age, race, education, etc.) and asked to answer as that "person" would. |
| **Persona agent** | An LLM that has been instructed to role-play as a specific character — either a real person (e.g., "Answer as Donald Trump would") or a fictional character with defined traits. |
| **Persona fidelity** | How accurately the LLM stays "in character" — does it consistently respond the way the assigned persona would, or does it slip back to generic LLM behavior? |
| **Big Five personality traits** | The most widely used scientific framework for measuring personality. Five dimensions: Openness (curiosity), Conscientiousness (organization), Extraversion (sociability), Agreeableness (cooperativeness), Neuroticism (emotional instability). Each person scores somewhere on a spectrum for each trait. |
| **General Social Survey (GSS)** | A long-running U.S. survey (since 1972) that asks Americans about their attitudes, behaviors, and demographics — politics, religion, social issues, etc. Used as a benchmark for whether simulated survey responses match real ones. |
| **Escalation bias** | The tendency of LLMs to recommend or predict more aggressive, confrontational actions than real humans typically choose — especially in crisis scenarios. |
| **Public figure data advantage** | The hypothesis (central to our project) that LLMs can simulate public figures more accurately than private individuals because vastly more text by and about public figures exists in training data. |
| **SCOTUS** | Supreme Court of the United States — the highest court, with 9 justices who vote on cases. A key analogy because predicting how a *named* justice votes is structurally identical to predicting how a *named* leader decides. |
| **FOMC** | Federal Open Market Committee — the Federal Reserve body that sets U.S. interest rates. Another key analogy because its decisions are structured, public, and have strong existing baselines. |
| **Generative agents** | AI software agents that can simulate believable human behavior — not just answering questions, but planning, reflecting on memories, and interacting with each other over time. |
| **PersonaScore** | A metric from the PersonaGym framework that measures how faithfully an LLM maintains an assigned persona across different situations. |
| **MACM** | Multi-Agent Cognitive Mechanism — a technique from the Human Simulacra benchmark that uses multiple LLM agents working together to simulate human memory and cognition. |

---

## Section A: Analogous Prediction Domains

> **What this section covers**: Before attempting to predict world leader decisions, we need to know: has anyone successfully predicted the decisions of *specific named individuals* before? This section surveys every domain where researchers predict how a particular person or small group will decide — not statistical trends about populations, but "What will Justice Roberts do?" or "Will the Fed raise rates?" These analogies establish what's achievable and what accuracy ceilings look like.

---

### Entry 1: Supreme Court Prediction — Katz et al. (2017)

**Citation**: Katz, D.M., Bommarito, M.J., & Blackman, J. (2017). "A general approach for predicting the behavior of the Supreme Court of the United States." *PLoS ONE*, 12(4).

**What they predicted**: How each individual U.S. Supreme Court justice would vote on each case (affirm or reverse the lower court), and what the overall case outcome would be.

**Why this is the closest analogy to our project**: Predicting how a *named individual* (Justice X) will vote on a *specific question* (Case Y) is structurally identical to predicting how a named leader (President X) will decide on a policy question (Action Y). If it works for SCOTUS, it might work for world leaders.

**How they did it — step by step**:
1. **Collected nearly 200 years of data** — every SCOTUS case from 1816 to 2015 (over 240,000 justice votes across 28,000+ cases).
2. **Extracted predictive features** — not the legal text of the cases, but structural metadata: which circuit court the case came from, what area of law it concerned (civil rights, criminal, economic, etc.), who the petitioner and respondent were (individual vs. government vs. corporation), how the lower court ruled, and each justice's historical ideological score.
3. **Trained a time-evolving random forest classifier** — a type of machine learning model that makes predictions by building many simple decision trees and averaging their answers. "Time-evolving" means the model is retrained each year, so it can adapt to changes in the Court's composition and ideological drift.
4. **Tested out-of-sample** — predictions were made for each year using only data from *prior* years, preventing the model from "seeing" the future.

**Accuracy achieved**:

| Metric | Accuracy |
|---|---|
| Case outcome (affirm/reverse) | **70.2%** |
| Individual justice votes | **71.9%** |
| Consistency across eras | Maintained performance even on historically ambiguous periods |

**What baselines did they compare against?**

| Baseline | Accuracy | What It Means |
|---|---|---|
| Random chance | 50% | Coin flip — the absolute floor |
| Always-reverse | ~60% | SCOTUS reverses more often than it affirms, so always guessing "reverse" gets you 60% |
| Expert prediction | ~75-80% | Legal scholars who study the Court achieve ~75-80%, but *inconsistently* year-over-year |

**Key insight**: The model's 70.2% accuracy is lower than expert accuracy in *good* years but higher than expert accuracy in *bad* years. The model is more *consistent* than human experts — it doesn't have off years.

**Why this matters for our project**: ★★★★★ — This is the strongest direct evidence that predicting individual decision-makers is feasible. The ~70% accuracy ceiling on structured decisions with extensive historical data gives us a realistic target. But note: SCOTUS decisions are highly constrained (binary choice, institutional norms, public record) — world leader decisions are less constrained, suggesting we should expect *lower* accuracy.

---

### Entry 2: Martin-Quinn Scores — Latent Ideology as a Predictor

**Citation**: Martin, A.D. & Quinn, K.M. (2002). "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo." *Political Analysis*, 10(2), 134-153.

**What they measure**: Each Supreme Court justice's ideological position on a liberal-conservative scale, estimated from their voting history over time.

**How it works — in plain language**: Imagine you could place each justice on a number line from very liberal (left) to very conservative (right). Martin-Quinn scores do exactly this, but statistically. They use a technique called **Bayesian item-response modeling** (similar to how standardized tests estimate student ability from test answers) to infer each justice's "ideal point" (true ideological position) from the pattern of cases where they vote with or against other justices. The scores change over time, capturing ideological drift (e.g., Justice Kennedy moving moderately rightward over his tenure).

**Accuracy when used for prediction**: Martin-Quinn scores, when fed into prediction models, match actual justice voting behavior with **75-80%** accuracy.

**Why this is directly analogous to our approach**: Martin-Quinn scores extract a *latent psychological variable* (ideology) from *observable behavior* (votes). This is exactly what the VICS operational code analysis does for world leaders (Domain 2) — extracting latent belief systems from observable speech patterns. Both assume that a person's past behavior reveals a stable internal disposition that predicts future behavior.

---

### Entry 3: Pre/Dicta & SCOTUSbot — Commercial and AI-Powered Judicial Prediction

**Pre/Dicta** ([pre-dicta.com](https://pre-dicta.com)):
- **What it predicts**: Outcomes of dispositive motions (decisions that end a case) in federal courts — not just SCOTUS but thousands of lower court judges.
- **Accuracy**: Claims **85%** on dispositive motions, using judicial profiles, case history, and biographical data.
- **Why it matters**: Demonstrates *commercial viability* — there's a market for predicting named decision-makers. Law firms pay for this.

**SCOTUSbot** (The Economist, 2025):
- **What it is**: An AI tool using LLMs with case-specific context to predict SCOTUS 2024-2025 term rulings.
- **Results**: Algorithmic approaches achieve **~70% vote-level accuracy**; human experts scored **~80% vote accuracy** and **76.3% case accuracy** in comparative 2024 testing.
- **Why it matters**: Even with LLMs, humans still edge out algorithms on SCOTUS — but the gap is narrow (~10 percentage points). This mirrors the pattern from Domain 4 where frontier LLMs approach but haven't yet matched superforecasters.

---

### Entry 4: FOMC / Central Bank Rate Decision Prediction

**What is predicted**: Whether the Federal Reserve's Federal Open Market Committee (FOMC) will raise, cut, or hold interest rates at each of its 8 annual meetings.

**Why this is our ideal pilot experiment**: FOMC decisions have every property that makes prediction tractable:
- **Structured**: Binary/ternary choice (raise/cut/hold), made at fixed intervals
- **Public reasoning**: The Fed publishes detailed minutes, press conferences, and "dot plots" (each member's rate prediction)
- **Strong baselines**: Multiple independent prediction methods already exist, so we can benchmark rigorously
- **High stakes but verifiable**: Markets move billions on these decisions, creating natural incentive alignment

**Accuracy by prediction method**:

| Method | Accuracy | Time Horizon | How It Works |
|---|---|---|---|
| **Fed funds futures** | ~85-90% | < 6 months | Financial contracts that pay based on the actual rate. Market price = implied probability. Most accurate financial indicator. |
| **Prediction markets (Kalshi)** | Rivals/exceeds futures | Eve of meeting | CFTC-regulated exchange where people bet real money. Fed Reserve Board study (Feb 2026) found Kalshi rivals professional economic forecasts. |
| **Survey of Professional Forecasters** | Comparable to Fed | < 1 quarter | Economists from banks and research firms submit quarterly forecasts. Accuracy degrades sharply beyond 3 months. |
| **Random walk / persistence** | Competitive short-term | < 3 months | Simply predict "no change." Surprisingly hard to beat because the Fed *usually* doesn't change rates. |
| **Hybrid models (text + macro)** | Best overall | Varies | Combining NLP analysis of FOMC statements with macroeconomic data (inflation, unemployment) outperforms either alone. |

**The "public figure data advantage" applies here**: The Federal Reserve is one of the most analyzed institutions in the world. Every speech by every Fed governor is transcribed, analyzed, and debated. LLMs have seen enormous amounts of Fed-related text, which should make FOMC prediction a favorable domain for LLM-based forecasting.

**Why this matters for our project**: ★★★★★ — FOMC is our planned first experiment. The 85-90% accuracy ceiling with existing methods tells us what's achievable, and the rich baselines (futures, Kalshi, persistence, professional forecasters) give us a rigorous comparison framework.

---

### Entry 5: Election Prediction with LLMs — The Bias Warning

**What was tested**: Could LLMs predict the 2024 U.S. presidential election outcome?

**Key findings that matter for us**:

| Finding | Source | What It Means |
|---|---|---|
| LLMs overpredicted Kamala Harris favorability by **10-40%** | Multi-model study, 2024 | LLMs have systematic liberal/Democratic bias from training data |
| MIT study: 12 LLMs showed high sensitivity to identity cues | MIT CSAIL, 2024 | Telling the LLM it's "a conservative voter" vs. "a progressive voter" dramatically shifts responses — personas are shallow stereotypes |
| Prediction markets outperformed polls and LLMs | Multiple sources | Polymarket showed better accuracy than traditional polls, especially in swing states |
| Polls underestimated Trump for the third consecutive cycle | 2016, 2020, 2024 | Both humans and algorithms have a persistent blind spot for populist mobilization |

**The critical lesson**: LLMs have **systematic directional biases** in political prediction. They reflect their training data's ideological distribution, which skews pro-Democrat/liberal for English-language models. For our project, this means: if we ask a Western-trained LLM to predict Putin's or Xi's decisions, the model's Western-centric training data will bias predictions toward what *Western analysts expect*, not what the leader *actually does*.

---

### Entry 6: The Public Figure Data Advantage — A Core Hypothesis

> **This entry introduces a concept not covered in any single paper but is central to our project's design.**

**The hypothesis**: An LLM's ability to simulate a specific person should scale with the volume and diversity of publicly available text *by* and *about* that person.

**Why this matters**: Supreme Court justices and FOMC governors are somewhat predictable partly because their public records are extensive. But consider the spectrum:

| Person | Public Data Volume | Expected LLM Simulation Quality | Why |
|---|---|---|---|
| **Donald Trump** | Extreme — millions of tweets, thousands of rally transcripts, books, interviews, court filings, media coverage | **Very high** — the LLM has seen more Trump text than almost any other individual on Earth | His communication style is distinctive and consistent, his positions are publicly stated, his decision patterns are extensively analyzed |
| **Xi Jinping** | Moderate — official speeches, state media coverage, diplomatic transcripts, but much is in Mandarin and filtered through CCP messaging | **Medium** — good for official positions, poor for private reasoning and internal debate | Language barrier + state media filtering reduce data quality; inner circle dynamics are opaque |
| **Average SCOTUS justice** | Moderate — published opinions, oral argument transcripts, some media appearances | **Medium-high** — enough data to establish ideological patterns | Opinions are detailed and analytically rich, but personal reasoning is not fully public |
| **Average Fortune 500 CEO** | Low-moderate — earnings calls, annual letters, some interviews | **Low-medium** — enough for communication style, not enough for strategic reasoning | Key strategic discussions happen behind closed doors |
| **Unknown bureaucrat** | Minimal | **Very low** — no individual signal in training data | The LLM can only apply generic role-based stereotypes |

**The implication for our project**: We should **start with the most data-rich leaders** — figures like Trump, Biden, Macron, Modi, and Erdogan who have extensive public records — before attempting leaders with less public data (e.g., Xi's internal reasoning, Putin's strategic deliberations).

**Evidence supporting this hypothesis**:
- Park et al. (2024 — Entry 8 below) found that *more interview data* per participant produced *more accurate* simulations.
- The MIT 2024 election study found that LLM persona quality directly correlates with the distinctiveness and volume of the person's public discourse.
- SCOTUSbot accuracy is higher for justices with longer tenures (more published opinions), supporting the data-volume-matters hypothesis.

---

## Section B: LLM Persona Simulation — Can LLMs Be Someone?

> **What this section covers**: The "prediction through simulation" approach asks: instead of building a statistical model to predict what a leader will do, can we build an LLM that *thinks like* that leader? This section reviews the research on whether LLMs can faithfully simulate specific individuals' beliefs, reasoning styles, and decision patterns — from generic demographic personas to specific named people.

---

### Entry 7: Generative Agents — Park, O'Brien, Cai et al. (2023)

**Citation**: Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., & Bernstein, M.S. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023* (ACM Symposium on User Interface Software and Technology).

**What they built**: A virtual town (like "The Sims") populated by 25 AI-controlled characters, each with a backstory, job, relationships, and daily routines. The characters wake up, go to work, have conversations, form opinions, and make plans — all autonomously, driven by LLMs.

**Why this paper is foundational**: It introduced the architecture that makes persona simulation possible — not just answering one question "in character," but maintaining a coherent personality over *extended interactions and time*.

**The architecture — three key innovations**:

| Component | What It Does | Why It's Important |
|---|---|---|
| **Memory stream** | Records everything the agent experiences — conversations, observations, events — in natural language, with timestamps. | Without memory, LLMs can't maintain consistency. Each interaction would be independent, producing contradictions. The memory stream gives agents a "life history" they can reference. |
| **Reflection** | Periodically, the agent pauses and generates higher-level insights from its recent memories. Example: after attending several parties, the agent might reflect, "I enjoy socializing but find large groups exhausting." | Raw memories are too granular to guide behavior. Reflections create *abstract self-knowledge* — exactly what personality is. |
| **Planning** | The agent creates and revises multi-step plans based on its goals, personality, and current situation. Plans are hierarchical (day → hour → minute). | Without planning, agents are purely reactive. Planning creates proactive, goal-directed behavior that looks intentional and human-like. |

**Key results**:
- Agents spontaneously organized a Valentine's Day party without being told to — one agent mentioned the date, others coordinated, and the event emerged from individual decisions.
- Human evaluators rated the agents' behavior as "believable" — they could not reliably distinguish agent behavior from scripted human behavior in blind tests.
- Agents maintained personality consistency across days of simulated time.

**What this does NOT prove**: The agents are believable *as fictional characters*. They are NOT simulations of real people. They have made-up backstories, not real life histories. The question of whether this architecture can simulate a *specific real person* — which is what our project requires — is addressed in Entry 8.

**Why it matters for our project**: ★★★★★ — The memory + reflection + planning architecture is the foundation for any serious persona simulation. If we build an "LLM Xi Jinping," we need these components: a memory stream of his speeches and actions, reflections that capture his strategic worldview, and planning that models his decision-making process.

---

### Entry 8: Generative Agent Simulations of 1,000 People — Park et al. (2024)

**Citation**: Park, J.S., Zou, C., Shaw, A., Hill, B., Cai, C., Morris, M., Bernstein, M.S., et al. (2024). "Generative Agent Simulations of 1,000 People." arXiv:2411.10109. Stanford University & Google DeepMind.

**What they did**: Took the generative agent architecture from the 2023 paper and asked: can we simulate *real people* — not fictional characters, but actual individuals — accurately enough to predict their survey responses and personality traits?

**The methodology — step by step**:
1. **Recruited 1,052 real people** from diverse demographic backgrounds.
2. **Conducted 2-hour qualitative interviews** with each participant — in-depth conversations about their lives, beliefs, values, experiences, and decision-making patterns. Each interview produced a transcript averaging **6,400+ words**.
3. **Created an AI agent for each person** by feeding their interview transcript into the generative agent architecture. The agent's "personality" is grounded in the actual person's own words about themselves.
4. **Added a reflection module** — the system generated "expert reflections" on each interview, as if a psychologist, demographer, and sociologist had independently analyzed the transcript. These reflections gave the agent deeper self-understanding than the raw interview alone.
5. **Tested the agents** by having them take the same surveys and personality tests as the real person, then comparing the agent's answers to the real person's answers.

**Accuracy results — the numbers**:

| Test | Accuracy | What This Means |
|---|---|---|
| General Social Survey (GSS) — attitudes on politics, religion, social issues | **85%** match with real person's responses | The agent correctly predicted how the real person would answer 85 out of 100 survey questions |
| Big Five Personality Traits | **0.80 normalized correlation** | On a 0-to-1 scale, the agent's personality profile correlated at 0.80 with the real person's — very strong |
| Test-retest comparison | Comparable to human self-replication | The agent's accuracy was roughly as good as when the same human re-took the survey 2 weeks later — humans don't even replicate their own answers perfectly |
| Experimental study replication | **4 out of 5** successfully replicated | When the agents were put through behavioral experiments (e.g., economic games), their behavior matched real human patterns 80% of the time |
| Economic games (Dictator Game) | Strong correlation with real behavior | Agents' decisions about sharing money mirrored genuine human generosity/selfishness patterns |

**Critical nuances**:
- The **85% accuracy requires 2-hour interviews** — not just demographic data. When agents were given only demographic descriptions (age, race, education) without the interview, accuracy dropped significantly. The *quality and depth* of personal data matters enormously.
- Agents showed **reduced variance** compared to real people — they tended to cluster toward modal (most common) responses, underestimating human quirks and extremes. This is the same "overconfidence" / "convergence to the mean" problem identified in Domain 4.
- Agents showed **positivity bias** — they overestimated the tendency to give positive ratings, make generous choices, and express agreement.

**The public figure data advantage, revisited**: For our project, we don't have 2-hour interviews with Xi Jinping or Trump. But we have something potentially better for *public figures*: thousands of hours of their speeches, tweets, interviews, press conferences, books, and media coverage. The Trump corpus alone likely exceeds what any single interview participant provided. The question is whether *public text* (which is often performative, calculated, and audience-aware) is as informative for simulation as *private interview* (which is more candid and reflective).

**Why it matters for our project**: ★★★★★ — This is the strongest evidence yet that LLMs can simulate specific individuals, not just demographic archetypes. The 85% / 0.80 numbers are our benchmark for persona fidelity. The key gap: this study used *private individuals* with interview data — we need to test whether the same approach works for *public figures* using their public record.

---

### Entry 9: Silicon Sampling — Argyle et al. (2023)

**Citation**: Argyle, L.P., Busby, E.C., Fulda, N., Gubler, J.R., Rytting, C., & Wingate, D. (2023). "Out of One, Many: Using Language Models to Simulate Human Samples." *Political Analysis*, 31(3), 337-351.

**What they did**: Coined the term **"silicon sampling."** Conditioned GPT-3 on socio-demographic backstories from real survey participants (age, race, gender, education, party ID) and generated synthetic survey responses.

**The method — in plain language**: Instead of surveying 1,000 real Americans, they told GPT-3: "You are a 45-year-old Black woman with a college degree who identifies as a Democrat living in Atlanta." Then they asked the LLM to answer the same survey questions the real person answered, to see how well the LLM's answers matched.

**Key results**:
- **Group-level patterns were accurately reproduced** — the gender gap on gun control, the racial gap on policing attitudes, and the partisan gap on climate change all appeared in the synthetic data, matching real survey distributions.
- **At the aggregate level**, silicon samples match real survey distributions reasonably well for well-represented U.S. demographic groups.

**Critical limitation**: This works for **group-level patterns** but NOT individual-level prediction. The LLM captures what a *typical* person in a demographic group thinks — not what a *specific* person thinks. Our project requires individual-level accuracy (what *Xi Jinping specifically* will do), which is a fundamentally harder problem.

---

### Entry 10: The Critique — Bisbee et al. (2024)

**Citation**: Bisbee, J. et al. (2024). "Synthetic Replacements for Human Survey Data? The Perils of Large Language Models." *Political Analysis*, Cambridge University Press.

**What they found** — five specific failure modes that directly affect our project:

| Failure Mode | What Happens | Impact on Leader Prediction |
|---|---|---|
| **Reduced variance** | LLM responses show less diversity than real human survey data — the model converges toward the most common answer | The LLM "Xi Jinping" may predict the *most likely* action but underestimate the probability of surprising moves |
| **Prompt sensitivity** | Minor rephrasing of the same question produces significantly different response distributions | We must use *exactly* standardized, version-controlled prompts. Any ad-hoc rephrasing invalidates comparisons |
| **Model update fragility** | The same prompt on GPT-3 vs. GPT-3.5 vs. GPT-4 produces different distributions | We must pin model versions and report model ID in all experiments. Results are NOT transferable across model versions |
| **Overconfidence** | LLMs assign more extreme probabilities than real human populations | Requires post-hoc calibration (Temperature Scaling from Domain 3) |
| **NOT reliable for statistical inference** | Average scores align with baselines, but you can't use synthetic data for rigorous hypothesis testing | We should use LLM predictions for *directional* guidance, not as substitutes for actual evidence |

**Why this matters**: ★★★★☆ — This is the essential counterpoint to the optimistic findings in Entries 7-9. Persona simulation *looks* impressive on averages but is fragile underneath. Every failure mode identified here maps directly to a design constraint for our pipeline.

---

### Entry 11: "LLM Generated Persona Is a Promise with a Catch" — Kovarikova et al. (2025)

**Citation**: Kovarikova et al. (2025). "LLM Generated Persona is a Promise with a Catch." arXiv, March 2025.

**What they tested**: Through large-scale experiments — including presidential election forecasts — they tested whether current LLM-generated personas contain systematic biases that could lead to substantial deviations from real-world outcomes.

**Key findings**:
- LLM personas exhibit **significant systematic biases** — they don't just add noise, they push predictions in consistent wrong directions.
- These biases are large enough to produce **substantial deviations from real-world outcomes** — not just small errors, but qualitatively wrong predictions.
- The authors call for "a rigorous science of persona generation" — meaning the field currently lacks it.

**The implication**: You can't just prompt an LLM with "You are Xi Jinping" and trust the output. Without rigorous validation against ground truth (what the real person actually did), persona predictions can be confidently wrong.

---

### Entry 12: Human Simulacra Benchmark — Xie et al. (ICLR 2025)

**Citation**: Xie, Q., Feng, Q., Zhang, T., Li, Q., Yang, L., Zhang, Y., Feng, R., He, L., Gao, S., & Zhang, Y. (2025). "Human Simulacra: A Step toward the Personification of Large Language Models." *ICLR 2025*.

**What it is**: A formal benchmark — the first of its kind — for evaluating how well LLMs simulate individual human personalities. Published at ICLR (International Conference on Learning Representations), one of the top-3 machine learning conferences.

**The methodology — three innovations**:

1. **Detailed life story generation**: Instead of just giving the LLM a demographic profile, Human Simulacra creates detailed fictional life stories for 11 virtual characters — childhood, education, career, relationships, beliefs, turning points. Each character has a unique biography designed to produce distinct personality traits.

2. **Multi-Agent Cognitive Mechanism (MACM)**: Inspired by human memory theories (Atkinson-Shiffrin's sensory/short-term/long-term memory model; Baddeley-Hitch's working memory model), MACM uses multiple LLM agents working together to simulate how humans process experiences, form memories, and use those memories to make decisions. One agent handles "sensory input" (new information), another manages "short-term memory" (current context), and a third maintains "long-term memory" (accumulated knowledge and personality).

3. **Dual evaluation — self-reported + observed**: The benchmark evaluates personas from two perspectives:
   - **Self-reported**: Does the character describe itself consistently with its designed personality? (Like a personality questionnaire)
   - **Observational**: Does the character *behave* consistently with its designed personality when put in novel situations? (Like observing someone in daily life)

**Dataset**: ~129,000 texts spanning 11 characters.

**Key finding**: The MACM-generated simulacra produce responses that align with their target characters — demonstrating that cognitive architecture matters for persona fidelity, not just prompting.

**Why it matters for our project**: ★★★★☆ — This benchmark shows us *how* to evaluate persona fidelity rigorously. Our own evaluation of an "LLM Xi Jinping" should test both self-reported consistency (does it describe Xi's views correctly?) and behavioral consistency (does it make decisions Xi would make in novel situations?).

---

## Section C: Persona Evaluation Frameworks — How Do You Measure If It's Working?

> **What this section covers**: If we build an "LLM Xi Jinping," how do we know if it's any good? You can't just eyeball it and say "yeah, that sounds like something Xi would say." You need rigorous, reproducible evaluation frameworks. This section reviews the best existing approaches.

---

### Entry 13: PersonaGym & PersonaScore — Jansen et al. (2024-2025)

**Citation**: Jansen, T. et al. (2024). "PersonaGym: Evaluating Persona Agents and LLMs." ACL 2025 / arXiv 2407.18416.

**What it is**: The first *dynamic* evaluation framework for LLM persona agents. "Dynamic" means the evaluation adapts to the specific persona being tested, rather than using the same generic questions for everyone.

**Why existing evaluation methods are insufficient**: Before PersonaGym, researchers evaluated personas by asking the same set of generic questions to every persona and checking for consistency. The problem: if you ask "What's your favorite food?" to an LLM playing a medieval knight vs. an LLM playing a Silicon Valley CEO, both questions are relevant — but "What's your stance on cryptocurrency?" is only relevant to the CEO persona. Generic questions produce generic evaluations that miss persona-specific fidelity.

**The PersonaGym framework — three stages**:

| Stage | What Happens | Why It's Necessary |
|---|---|---|
| **1. Dynamic environment selection** | The system automatically chooses scenarios that are *relevant* to the specific persona. A military general gets battlefield scenarios; an economist gets market scenarios. | Ensures the evaluation tests knowledge and reasoning the persona *should* have, not irrelevant trivia. |
| **2. Persona-task generation** | For each environment, the system generates persona-specific questions that test whether the LLM knows what this specific character would know and how they would respond. | Generic questions only test surface-level role-play. Persona-specific questions test depth — does the LLM *understand* this person, or just mimic their speech patterns? |
| **3. Agent response evaluation via PersonaScore** | Responses are scored using PersonaScore, an automatic metric grounded in decision theory that measures persona adherence. PersonaScore is calibrated against human judgments to ensure it matches what humans consider "in character." | Human evaluation doesn't scale. You need an automatic metric that correlates with human judgment so you can evaluate thousands of responses efficiently. |

**Scale of evaluation**: 200 diverse personas tested across 10,000 questions, measuring 5 dimensions:

| Dimension | What It Measures | Example |
|---|---|---|
| **Action Justification** | Can the persona explain *why* it would take a specific action, in a way consistent with its character? | "I chose diplomacy because, as Xi Jinping, maintaining regional stability protects China's economic interests." |
| **Expected Action** | Does the persona choose the action its character would most likely choose? | When presented with a Taiwan scenario, does the LLM-Xi choose the same action the real Xi typically signals? |
| **Linguistic Habits** | Does the persona speak the way its character speaks? | Trump uses short declarative sentences with superlatives ("tremendous," "the best"). Does the persona maintain this? |
| **Persona Consistency** | Does the persona stay in character across different topics and situations, or does it "break character"? | If the LLM-Xi is dovish on Taiwan in one prompt but hawkish in a slightly reworded prompt, that's a consistency failure. |
| **Toxicity Control** | Does the persona avoid generating harmful content even when the character might realistically use such language? | The LLM should simulate a leader's views without generating genuinely dangerous content (e.g., specific military plans). |

**Critical finding**: Even state-of-the-art models (Claude 3.5 Sonnet, GPT-4.5) showed significant persona consistency failures. Larger, more advanced models do NOT automatically produce better personas. Some smaller models maintained personas more faithfully than frontier models on certain dimensions.

**Why this matters for our project**: ★★★★★ — PersonaGym gives us a ready-made evaluation framework we can adapt. For our "LLM world leader" project, we should evaluate along all 5 dimensions, with special emphasis on **Expected Action** (predicting what the leader will do) and **Persona Consistency** (maintaining the persona across different scenarios and prompt formulations).

---

### Entry 14: CharacterBot / "Beyond Profile" — Deep Persona Simulation (ACL 2025)

**Citation**: "Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs." *ACL 2025*, July 2025.

**What problem it solves**: Most persona simulation stays at the surface level — the LLM knows a character's biographical facts (born in X, studied Y, works at Z) and mimics their speech patterns. But it doesn't capture how the person *thinks* — their deeper reasoning patterns, ideological framework, and decision-making logic. "Beyond Profile" tries to close this gap.

**The distinction it makes**:

| Level | What It Captures | Example (Trump) |
|---|---|---|
| **Surface-level (basic)** | Biographical facts, speech patterns, catchphrases | Uses "tremendous," "believe me," "many people are saying." Grew up in Queens, went to Wharton, was a real estate developer. |
| **Deep persona** | Thought processes, ideological reasoning, decision frameworks | Thinks in terms of bilateral deals rather than multilateral institutions. Evaluates policy through lens of perceived personal loyalty. Views international relations as zero-sum transactions. |

**CharacterBot's approach**: Instead of just training on what a person *says*, CharacterBot trains on the *reasoning process* behind what they say. It decomposes each public statement into:
1. The **surface output** (the actual words)
2. The **underlying opinion/ideology** that motivated those words
3. The **logical chain** connecting beliefs to statements

**Evaluation results**: CharacterBot significantly outperformed baseline models on:
- **Linguistic accuracy** — reproducing the character's communication style
- **Style preservation** — maintaining consistency across different topics
- **Opinion comprehension** — correctly predicting the character's stance on new, unseen topics

**Why this matters for our project**: ★★★★★ — For our world leader simulation, we need the "deep persona" level. An LLM that can mimic Xi Jinping's speech patterns is interesting but not useful for prediction. We need an LLM that captures Xi's *strategic reasoning* — his operational code (Domain 2 concepts: VICS philosophical and instrumental beliefs). CharacterBot's approach of decomposing text into surface statements + underlying ideology + logical chains maps directly onto our VICS-based pipeline.

---

### Entry 15: CharacterEval — Chinese Role-Playing Assessment (2024)

**Citation**: CharacterEval. (2024). "CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Assessment."

**What it is**: A benchmark specifically for evaluating role-playing agents that converse in Chinese — relevant because our project includes Chinese-speaking leaders (Xi Jinping).

**Framework**: 13 metrics across 4 dimensions:
1. **Knowledge consistency** — Does the agent know what it should know?
2. **Persona consistency** — Does the agent stay in character?
3. **Role-playing attractiveness** — Is the agent engaging and realistic?
4. **CharacterRM** — A dedicated reward model (a separate neural network trained to score quality) for evaluating subjective metrics.

**Key finding**: Chinese LLMs (e.g., Baichuan, Qwen) showed promising role-playing capabilities in Chinese-language scenarios — sometimes outperforming GPT-4 on Chinese character simulation. This suggests that **language-specific models matter** — a Chinese-trained model may simulate Chinese leaders better than an English-first model.

**Why this matters**: ★★★☆☆ — Reinforces that we should consider Chinese-language models (or at least bilingual evaluation) when simulating Chinese leaders.

---

## Section D: Political Simulation & Crisis Prediction — What Happens When You Apply This to Geopolitics?

> **What this section covers**: This section reviews research where LLMs have been used specifically to simulate political actors, predict geopolitical outcomes, and model crisis scenarios. This is the most directly relevant literature for our project — and it reveals both promising results and alarming failure modes.

---

### Entry 16: Escalation Bias — Payne et al. (2024)

**Citation**: Payne, J. et al. (2024). "Escalation Risks from Language Models in Military and Diplomatic Decision-Making." Georgia Tech, Stanford, Northeastern. Also: Rivera et al. (2024), "Escalation with LLMs."

**What they tested**: Placed frontier LLMs (GPT-4, Claude, Llama-2) into wargame simulations — realistic military and diplomatic scenarios with multiple rounds of decision-making — and measured the tendency of each model to recommend aggressive actions.

**The experimental setup — step by step**:
1. **Designed 8 geopolitical crisis scenarios** — e.g., border tensions between nuclear states, maritime disputes, cyberattacks on critical infrastructure.
2. **Assigned the LLM a role** — "You are the National Security Advisor to the President of the United States." Or: "You are a senior advisor to the Russian Ministry of Defense."
3. **Presented escalation ladders** — at each turn, the LLM chose from a set of actions ranging from diplomatic (send ambassador, issue statement) to military (deploy troops, conduct strikes, use nuclear weapons). Each action had a numerical escalation score.
4. **Tracked escalation trajectories** over 15-round wargames — did the LLM consistently choose increasingly aggressive actions, or did it de-escalate?

**The results — the key finding**:

| Model | Escalation Tendency | First-Strike Nuclear Use? |
|---|---|---|
| **GPT-4** | Moderate escalation bias — consistently chose actions 1-2 levels more aggressive than human control groups | No first-use in most scenarios, but *justified* nuclear use when prompted with sufficiently severe scenarios |
| **Claude** | Lowest escalation bias among tested models — generally preferred diplomatic options | No first-use |
| **Llama-2** | Highest escalation bias — recommended military action significantly earlier than GPT-4 or Claude | Yes — in some scenarios, recommended preemptive nuclear strikes |
| **GPT-3.5** | Moderate-high escalation bias | Occasionally justified nuclear use |
| **Human control** | De-escalated more often than any LLM | No first-use in any test scenario |

**Why this happens** — three mechanisms:

1. **Training data bias**: LLM training data over-represents crisis scenarios (news, fiction, wargaming literature) relative to routine diplomacy. The model has "read" more about wars than about successful negotiations.
2. **Narrative completion tendency**: LLMs are trained to produce coherent narratives. Escalation makes for a better story than de-escalation, so the model gravitates toward dramatic outcomes.
3. **Lack of consequence modeling**: Real decision-makers feel the weight of casualties and costs. LLMs don't experience consequences — they optimize for plausible-sounding responses.

**Why this matters for our project**: ★★★★★ — This is the single most important failure mode to address. If we build an "LLM Xi Jinping" and ask it what China will do in a Taiwan crisis, the model will *systematically overpredict aggressive actions*. Our pipeline MUST include explicit debiasing:
- **Weight accommodation/negotiation** actions to counterbalance escalation bias.
- **Compare against diplomacy-weighted baselines** — what would a random draw from historical diplomatic outcomes predict?
- **Report calibrated probabilities** — not just "China will/won't invade" but "probability of military action = X%, with escalation bias adjustment factor Y."

---

### Entry 17: WarAgent — Multi-Agent Wargaming (2024)

**Citation**: Hua, W. et al. (2024). "WarAgent: AI-Driven Multi-Agent Simulation of World Wars." arXiv:2311.17227.

**What they built**: A multi-agent simulation where multiple LLMs play different countries simultaneously, making decisions about alliances, trade, military mobilization, and war declarations. Tested on stylized recreations of the lead-up to World War I and II.

**The architecture**:
- Each country is controlled by a separate LLM agent with a national profile (resources, alliances, strategic interests, historical behavior patterns).
- Agents interact with each other through diplomatic channels (messages, treaties, threats).
- Each round covers a simulated time period (days to weeks), and agents must decide their actions given other agents' moves.

**Key results**:
- The simulation reproduced the alliance structures and mobilization patterns of WWI with surprising accuracy — the web of mutual defense treaties created cascading declarations of war, just as in reality.
- However, the models consistently escalated *faster* than historical actors did — compressing months of diplomatic maneuvering into days.
- Confirmed the **escalation bias** from Entry 16 in a multi-agent setting.

**Why it matters**: ★★★☆☆ — Shows that multi-agent LLM simulations can capture structural dynamics (alliance cascades, security dilemmas) but suffer from compressed timelines and escalation bias.

---

### Entry 18: FlockVote — Simulating Voters with Demographic Profiles (2025)

**Citation**: FlockVote. (2025). "FlockVote: LLM Agents as a Computational Laboratory for Political Simulation." arXiv, November 2025.

**What it does**: Creates a "computational laboratory" of LLM agents, each instantiated with a high-fidelity demographic profile (age, race, education, income, geographic location, media consumption, party identification, religious affiliation), and simulates how they would vote on specific candidates or ballot measures.

**How it works — step by step**:
1. **Profile construction**: Each agent gets a detailed demographic backstory derived from real census and survey microdata — not generic categories but specific combinations (e.g., "38-year-old Latina woman, community college education, household income $52,000, lives in suburban Phoenix, watches Fox News and reads local newspaper, Catholic, registered independent").
2. **Contextual injection**: Each agent receives current news context — recent campaign events, policy proposals, debate performances — to simulate how voters update their preferences.
3. **Vote elicitation**: The agent is asked to explain its voting reasoning and cast a vote. The system captures both the vote and the reasoning chain.

**Key advantage over silicon sampling (Entry 9)**: FlockVote adds *dynamic contextual information* — agents don't just reflect static demographics but update their preferences based on recent events, like real voters do.

**Why it matters**: ★★★☆☆ — Demonstrates the state of the art in population-level simulation. For our project, the profile construction methodology could be adapted for leader simulation — instead of demographic profiles, we'd use VICS operational codes, policy positions, and historical decision patterns.

---

### Entry 19: ElectionSim — AI-Driven Voter Simulation Framework (2024-2025)

**Citation**: ElectionSim. (2024-2025). LLM-powered simulation for elections.

**What it is**: An end-to-end LLM-based election simulation framework designed for:
- Simulating individual voter behavior using AI-generated personas
- Customizing demographic distributions to match actual electoral districts
- Interactive dialogue where researchers can "interview" simulated voters about their reasoning

**Accuracy results**: When calibrated against the 2024 U.S. election, ElectionSim's aggregate predictions aligned with actual voting patterns better than some traditional polling methods — but still underperformed prediction markets (Polymarket, PredictIt).

**Key limitation**: The system works for *aggregate* vote prediction (how will Maricopa County vote?) but NOT for predicting individual voter behavior. The same limitation as silicon sampling — good at group patterns, poor at individual decisions.

---

### Entry 20: European Parliament Voting Prediction (2025)

**Citation**: "Persona-Driven Simulation of Voting Behavior in the European Parliament Using LLMs." arXiv, 2025.

**What they tested**: Whether LLMs could predict how individual Members of European Parliament (MEPs) would vote on specific legislative proposals, by giving the LLM a persona of each MEP.

**The method**: For each MEP, the researchers constructed a persona prompt including:
- Party affiliation and political group
- National background
- Committee memberships
- Historical voting record on related issues
- Public statements on the topic

**Results**: The LLM achieved moderate accuracy on party-line votes (where the MEP votes with their party) but poor accuracy on cross-party or "free votes" (where the individual's personal judgment matters most). This reinforces the pattern: LLMs can predict *group-level* patterns (party discipline) but struggle with individual-level deviations.

**Why this matters**: ★★★★☆ — This is the closest existing analog to our project — predicting named legislators' votes using persona prompts. The finding that persona prompts work for party-line votes but not individual deviations is a critical insight for our pipeline: the harder and more important predictions (when will a leader *break* from expected patterns?) are precisely the ones LLMs are worst at.

---

## Section E: Government & Military Programs — Who Is Already Working on This at Institutional Scale?

> **What this section covers**: Predicting the decisions of world leaders isn't just an academic exercise. Intelligence agencies and military organizations have been working on this problem for decades — first with human analysts, now increasingly with AI. This section surveys the institutional programs that are most relevant to our project, from the CIA's at-a-distance profiling tradition to DARPA's current AI forecasting initiatives.

---

### Entry 21: CIA & Intelligence Community — At-a-Distance Leader Profiling

**What it is**: The CIA and broader U.S. Intelligence Community (IC) have conducted "at-a-distance" psychological assessments of foreign leaders since at least the Cold War. These assessments predict leaders' likely behavior based on public information — speeches, interviews, biographical data, observed decision patterns — without direct access to the individual.

**Key methodologies** (connecting back to Domain 2):

| Method | Origin | What It Does |
|---|---|---|
| **Leadership Trait Analysis (LTA)** | Prof. Margaret Hermann, 1970s-present | Scores leaders on 7 personality traits from speech analysis: belief in ability to control events, need for power, conceptual complexity, self-confidence, task vs. relationship focus, distrust of others, in-group bias. Used by CIA and DoD for foreign leader profiling. |
| **Operational Code Analysis (VICS)** | Alexander George / Stephen Walker; formalized by Mark Schafer & Stephen Walker | Scores leaders on 4 philosophical beliefs (about the nature of conflict, the role of chance, etc.) and 5 instrumental beliefs (about preferred tactics, risk propensity, etc.) by coding the verbs in their public statements. |
| **Profiling from a Distance** | IC-wide practice | Combines LTA, VICS, and biographical analysis. Published assessments exist for Castro, Hussein, Gaddafi, Kim Jong-il, and many others. |

**Scale**: The IC maintains ongoing profiles of dozens of world leaders, updated as new speech data becomes available. These profiles inform presidential daily briefings and National Intelligence Estimates.

**Key limitation**: Human-coded profiles are expensive, slow, and limited by analyst capacity. A single VICS analysis of one speech requires a trained coder to classify ~50 verbs, taking several hours. Automating this with LLMs is a central goal of our project (see Domain 2, Entry 3: automated VICS coding).

---

### Entry 22: DARPA / IARPA Forecasting Programs

**What they are**: The Defense Advanced Research Projects Agency (DARPA) and the Intelligence Advanced Research Projects Activity (IARPA) have funded multiple large-scale forecasting research programs. The most relevant:

**IARPA ACE (Aggregative Contingent Estimation) Program, 2011-2015**:
- **Purpose**: Find the best methods for forecasting geopolitical events.
- **Scale**: 5 research teams, thousands of forecasters, hundreds of questions about international events over 4 years.
- **Key output**: Discovered that a small elite group of "superforecasters" (top 2% of all participants) consistently outperformed CIA analysts with access to classified intelligence. This finding, published in Philip Tetlock's *Superforecasting* (2015), is the foundation for the baseline hierarchy in Domain 4.
- **Scoring methodology**: Used Brier scores (Domain 3, Entry 1) as the primary accuracy metric, establishing the standard our project follows.

**DARPA KAIROS (Knowledge-directed Artificial Intelligence Reasoning Over Schemas), 2019-2024**:
- **Purpose**: Build AI systems that can predict complex geopolitical event sequences (e.g., "regime instability → civil unrest → military coup → refugee crisis").
- **Approach**: Used schema-based event modeling — defining common patterns of event sequences and training AI to recognize when a real-world situation matches one of these patterns.
- **Relevance**: KAIROS explicitly frames prediction as *pattern matching* — which is exactly what LLMs do (matching current situations to patterns in training data). But KAIROS used structured schemas, while LLMs use implicit pattern matching, making LLM predictions harder to audit and explain.

**DIU (Defense Innovation Unit) — Donovan Group**:
- **Focus**: Applying commercial AI capabilities to intelligence analysis problems.
- **Notable projects**: Automated open-source intelligence (OSINT) analysis, satellite imagery interpretation, and — most relevant to us — NLP-based analysis of foreign leader statements for intent prediction.
- **Status**: Largely classified, but public statements from DIU leadership emphasize the use of LLMs for "rapid sense-making" from large volumes of foreign-language text.

**DARPA's current AI forecasting portfolio (2024-2026)**:
- **TRIAGE (Targeting and Realizing Improved Analysis through Generative Encounter)**: A program exploring LLM-augmented intelligence analysis, where LLMs generate hypotheses that human analysts then evaluate.
- **Emphasis on human-AI teaming**: Unlike our project (which explores fully autonomous LLM prediction), DARPA's current approach treats LLMs as *tools for human analysts* rather than standalone predictors. This "LLM-in-the-loop" approach sacrifices speed for reliability — human oversight catches LLM errors (hallucination, escalation bias) at the cost of slower, more expensive analysis.

**Why this matters for our project**: ★★★★☆ — The IC's decades of leader profiling validate our approach: predicting leaders from public data is a real, funded intelligence activity. IARPA ACE gives us our evaluation standard (Brier scores, superforecaster baselines). DARPA's shift toward human-AI teaming suggests that fully autonomous LLM prediction is not yet trusted for high-stakes decisions — which is the gap our research aims to address.

---

## Section F: Synthesis — The Evaluation Framework for Our Project

> **What this section provides**: Drawing on everything above, here is the integrated evaluation framework for testing whether LLMs can predict world leader decisions.

---

### The Evaluation Stack (What to Measure)

| Layer | Metric | Source | What It Tells Us |
|---|---|---|---|
| **1. Accuracy** | Brier Score, AUC-ROC | Domain 3 | How often is the prediction correct, and how well-calibrated are the probabilities? |
| **2. Calibration** | Expected Calibration Error (ECE), reliability diagram | Domain 3 | Do predicted probabilities match actual frequencies? (If we say "80% likely," does it happen 80% of the time?) |
| **3. Persona Fidelity** | PersonaScore, 5-dimension PersonaGym | Entry 13 | Does the LLM actually think like the leader, or is it generating generic responses with the leader's vocabulary? |
| **4. Deep Persona** | CharacterBot decomposition (surface + ideology + logic chain) | Entry 14 | Does the LLM capture the leader's strategic reasoning, not just their communication style? |
| **5. Escalation Bias** | Escalation trajectory analysis, diplomacy-weighted baselines | Entry 16 | Is the LLM systematically over-predicting aggressive actions? |
| **6. Consistency** | Cross-prompt agreement, version-pinned reproducibility | Entries 10, 11 | Same question rephrased → same answer? Same question on same model version → same answer? |

### The Baseline Hierarchy (What to Beat)

Based on the analogous domains reviewed above, here is the accuracy we should expect and what constitutes success:

| Level | Baseline | Expected Accuracy | Source |
|---|---|---|---|
| 1 | Random chance | 50% (binary) | Floor |
| 2 | Always-majority / persistence | ~55-65% | SCOTUS always-reverse; FOMC always-hold |
| 3 | Simple base-rate model | ~65% | Historical frequency of the most common outcome |
| 4 | Traditional ML on structural features | ~70% | Katz et al. SCOTUS model |
| 5 | Expert human judgment | ~75-80% | Legal scholars on SCOTUS; professional forecasters |
| 6 | Prediction markets | ~80-90% | Fed funds futures; Polymarket |
| 7 | Superforecasters | ~80-85% (Brier ~0.15-0.20) | Tetlock's IARPA ACE results |

**Our target**: Level 4-5 (70-80% on structured decisions like FOMC, with well-calibrated probabilities) is a realistic near-term goal. Matching or exceeding prediction markets (Level 6) would be a breakthrough result.

### The Pilot Experiment Protocol (FOMC)

Based on the review above, here is the recommended protocol for our first experiment:

1. **Decision to predict**: FOMC rate decision (raise / hold / cut) at each of the 8 annual meetings.
2. **Persona construction**: Create an "LLM FOMC" using:
   - Fed Chair speeches and press conference transcripts (last 2 years)
   - FOMC meeting minutes (last 2 years)
   - Dot plot data (individual member projections)
   - Current macroeconomic indicators (CPI, unemployment, GDP growth)
3. **Prediction format**: Probability distribution over {raise, hold, cut}, not a single deterministic prediction.
4. **Baselines to beat**:
   - Persistence ("no change from last meeting")
   - Fed funds futures implied probability
   - Survey of Professional Forecasters consensus
   - Kalshi prediction market price
5. **Evaluation metrics**:
   - Brier Score (primary)
   - Log loss (secondary)
   - Calibration (ECE + reliability diagram)
6. **Debiasing**: Check for and correct escalation bias (are we overpredicting rate hikes?), recency bias (are we overfitting to the last meeting?), and positivity bias.
7. **Version control**: Pin model version, pin prompt template, log everything.

---

## Summary — Top 10 Papers for Our Project

| Rank | Paper | Key Contribution | Relevance |
|---|---|---|---|
| 1 | Park et al. (2024) — 1,000 People | Simulated real individuals at 85% accuracy | ★★★★★ |
| 2 | Payne et al. (2024) — Escalation Bias | Identified the #1 failure mode for political simulation | ★★★★★ |
| 3 | PersonaGym (2024-2025) | Provided a ready-made evaluation framework (PersonaScore, 5 dimensions) | ★★★★★ |
| 4 | Park et al. (2023) — Generative Agents | Invented the memory + reflection + planning architecture | ★★★★★ |
| 5 | CharacterBot / Beyond Profile (2025) | Demonstrated deep persona simulation (ideology + reasoning, not just speech) | ★★★★★ |
| 6 | Katz et al. (2017) — SCOTUS | Proved specific-person prediction at 70% with ML | ★★★★★ |
| 7 | Bisbee et al. (2024) — Critique | Identified 5 failure modes of silicon sampling | ★★★★☆ |
| 8 | Argyle et al. (2023) — Silicon Sampling | Established group-level prediction baseline | ★★★★☆ |
| 9 | Human Simulacra (ICLR 2025) | First formal benchmark for persona personification | ★★★★☆ |
| 10 | EP Voting Prediction (2025) | Closest analog — named legislator vote prediction with persona prompts | ★★★★☆ |
