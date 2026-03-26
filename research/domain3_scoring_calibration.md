# Domain 3: Prediction Scoring, Calibration & Evaluation Methodology — Literature Review

> **The big question this review answers**: You've built a model that predicts what world leaders will do. Now what? How do you *measure* whether those predictions are actually good? How do you tell whether your model is calibrated (its "70% confident" predictions actually happen 70% of the time), or just spitting out plausible-sounding numbers? How do you compare your model to a baseline, and how do you *fix* it when its confidence is off?

> **Why this matters for our project**: A prediction without a rigorous scoring method is just a guess with extra steps. This domain reviews 75+ years of mathematical tools — from Glenn Brier's 1950 weather formula to 2023 LLM calibration techniques — that let us say precisely *how good* a prediction is, *why* it's good or bad, and *how to fix it*. Every experiment we run will use these tools.

---

## Thread 1: The Philosophical Foundation — Why Scoring Rules Must Be "Proper"

### What is a scoring rule?

A **scoring rule** is a mathematical formula that takes two inputs and produces one output:
- **Input 1**: The prediction (a probability, like "70% chance the Fed raises rates")
- **Input 2**: The outcome (what actually happened — the Fed raised rates, or it didn't)
- **Output**: A numerical score representing how good the prediction was

Think of it like a grading rubric for predictions. The question is: what should the rubric look like?

### What makes a scoring rule "proper"?

Here's the problem: if you pick the wrong rubric, forecasters learn to *game* it. Imagine a scoring rule that gives the best score when you predict 90% for everything. A strategic forecaster would always say "90% confident" regardless of what they actually believe — because that's what the rubric rewards.

A **proper scoring rule** is one where the forecaster gets the best expected score by reporting their *actual, honest belief*. There's no way to game it. If you truly believe there's a 70% chance of something, saying "70%" gives you a better expected score than saying anything else.

A **strictly proper scoring rule** goes one step further: not only is your true belief the best strategy, it's the *unique* best strategy. There's exactly one optimal answer (your true belief), and any deviation from it makes your expected score worse.

**Why this matters in plain English**: Without proper scoring rules, you can't trust that predictions are honest. A forecaster might say "80% confident" not because they believe it, but because the scoring system rewards confident-sounding predictions. Proper scoring rules make honesty the optimal strategy.

### Key Papers

---

#### 1. Savage (1971) — "Elicitation of Personal Probabilities and Expectations"

**Citation**: Savage, L.J. (1971). "Elicitation of Personal Probabilities and Expectations." *Journal of the American Statistical Association*, 66(336), 783–801.

**Who was Savage?** Leonard Jimmie Savage was a mathematician and statistician who is considered one of the founding fathers of **subjective probability theory** — the idea that probabilities represent *personal degrees of belief*, not just objective frequencies. His 1954 book *The Foundations of Statistics* is one of the most important works in 20th-century statistics.

**What this paper did**: Savage asked a deceptively simple question: if you want someone to honestly tell you their probability estimate for an event, what reward system should you use? He proved mathematically that the class of scoring rules that incentivize honesty (proper scoring rules) corresponds exactly to the class of **convex functions** on the probability simplex. This means:
- The **Brier score** (quadratic) is proper because the quadratic function is convex.
- The **logarithmic score** is proper because the negative log function is convex.
- Simple percentage correct (accuracy) is *not* proper because it can incentivize extreme predictions.

**Key insight**: Savage showed that proper scoring rules don't just measure prediction quality — they *define* what it means to elicit honest probability judgments. If you use an improper scoring rule, you literally cannot trust the probabilities people give you, because they have incentives to distort them.

**Why it matters for our project**: This is the theoretical bedrock. Every metric we use (Brier score, log loss, CRPS) derives its validity from Savage's proof. When we ask an LLM "what's the probability that Xi Jinping does X?", we need to score the answer with a proper scoring rule, or we can't interpret the probability as meaningful.

---

#### 2. Gneiting & Raftery (2007) — "Strictly Proper Scoring Rules, Prediction, and Estimation"

**Citation**: Gneiting, T. & Raftery, A.E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association*, 102(477), 359–378. DOI: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)

**What this paper is**: The definitive modern treatment of proper scoring rules. Savage (1971) laid the conceptual foundation; Gneiting and Raftery built the complete mathematical cathedral.

**What they did in plain English**: They characterized *every possible* strictly proper scoring rule — not just the famous ones (Brier, log), but the entire infinite family. They showed that any strictly proper scoring rule can be represented through a convex function, and they proved key relationships between scoring rules, calibration, and sharpness.

**The key principle they established**: **"Maximize sharpness, subject to calibration."** This deceptively simple sentence is the guiding philosophy for all probabilistic forecasting:
- **Calibration** means your probabilities match reality (when you say 70%, it happens 70% of the time).
- **Sharpness** means your probabilities are *informative* — you make confident predictions (close to 0% or 100%), not wishy-washy ones (always near 50%).
- The goal is to be as sharp (confident) as possible while remaining calibrated (correct). A forecaster who says "50/50" every time is perfectly calibrated if the base rate is 50% — but completely useless. A forecaster who says "95%!" every time is very sharp — but disastrously wrong if it only happens 70% of the time.

**Concrete example**: Imagine two LLMs predicting FOMC decisions:
- **LLM A** always says "60% chance of a rate hold." Over 100 meetings, 60 are holds. LLM A is perfectly calibrated but not sharp — it's barely more informative than saying "it's a coin flip."
- **LLM B** says "95% hold" for some meetings and "20% hold" for others. Over 100 meetings, the events it called 95% happen 93% of the time, and the events it called 20% happen 22% of the time. LLM B is both sharp AND calibrated — it tells you something useful AND it's correct.

**Why it matters for our project**: This paper gives us the framework for interpreting *every* evaluation we run. Good predictions require *both* calibration (are the probabilities honest?) and sharpness (do the probabilities vary enough to be useful?). We need to measure and report both.

---

#### 3. Dawid (1984) — "The Prequential Approach"

**Citation**: Dawid, A.P. (1984). "Present Position and Potential Developments: Some Personal Views: Statistical Theory: The Prequential Approach." *Journal of the Royal Statistical Society, Series A*, 147(2), 278–292.

**What "prequential" means**: Dawid coined this term by combining "**pre**dictive" and "se**quential**." The idea: you should evaluate a forecasting method *solely on the quality of its predictions*, issued one at a time as new data comes in — not on how well it fits past data, and not on the elegance of its underlying theory.

**Why this matters (in plain English)**: Before Dawid, the standard way to evaluate a statistical model was to check how well it fit the observed data (goodness of fit). Dawid argued this was backwards. What matters isn't whether your model explains the past — what matters is whether it *predicts the future*. A model should be judged by running it forward in real time, watching its predictions, and scoring them against what actually happens.

**The prequential principle**: *The quality of a forecasting system should depend only on the forecasts it actually issued, not on the reasoning or model behind those forecasts.* Two forecasters who issue identical probability forecasts should receive identical scores, even if one uses a sophisticated model and the other flips a coin (and happened to get lucky).

**Connection to proper scoring rules**: Dawid showed that only proper scoring rules are consistent with the prequential approach. Using an improper scoring rule violates the prequential principle because it evaluates something other than the forecast itself.

**Why it matters for our project**: This gives us the evaluation philosophy. We don't care *how* the LLM generates its prediction (what architecture, what prompting strategy, what retrieval pipeline). We care *only* about the quality of the predictions it outputs. This means we evaluate the LLM the same way we'd evaluate a human superforecaster — by the accuracy of their issued forecasts, period.

---

## Thread 2: The Three Primary Scoring Rules (The Actual Formulas)

### Overview

There are three scoring rules that dominate practice. Each is strictly proper (honesty-incentivizing), but they penalize errors differently:

| Scoring Rule | Mathematical Name | How It Penalizes Errors | Best For |
|---|---|---|---|
| **Brier Score** | Quadratic | Proportional to the *square* of the error | Binary outcomes (yes/no) |
| **Log Loss** | Logarithmic | Proportional to the *negative log* — harsh on overconfidence | Multi-class outcomes; punishing confident wrong predictions |
| **CRPS** | Continuous Ranked Probability Score | Generalizes Brier to continuous outcomes | Predicting continuous values (how much, when) |

---

#### 4. Brier (1950) — "Verification of Forecasts Expressed in Terms of Probability"

**Citation**: Brier, G.W. (1950). "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*, 78(1), 1–3.

**Who was Brier?** Glenn W. Brier was a meteorologist at the U.S. Weather Bureau. In 1950, weather forecasters were starting to give probabilistic forecasts ("40% chance of rain") instead of just yes/no forecasts ("it will rain"). But nobody had a good way to *score* probabilistic forecasts. Brier invented one.

**The formula (with plain-English explanation)**:

```
Brier Score = (1/N) × Σ (forecast_probability − actual_outcome)²
```

Where:
- **N** = the total number of predictions
- **forecast_probability** = the probability you assigned (a number between 0 and 1)
- **actual_outcome** = what actually happened (1 if the event occurred, 0 if it didn't)
- The sum (Σ) is over all N predictions

**How to read it**: The Brier score is the **average squared error** of your probability forecasts. Every prediction gets penalized by the *square* of the distance between your probability and the truth.

**Interpreting Brier scores (the practical benchmarks)**:

| Brier Score | What It Means | Analogy |
|---|---|---|
| **0.000** | Perfect — you predicted 100% for events that happened and 0% for events that didn't | You knew the future |
| **0.100** | Excellent — highly skilled probabilistic forecasting | Top superforecasters achieve this |
| **0.150** | Good — meaningfully better than coin-flipping | Average Good Judgment Project forecasters |
| **0.200** | Moderate — some skill, room for improvement | Typical informed forecaster |
| **0.250** | No skill — equivalent to always saying "50/50" | Coin flip / "I have no idea" |
| **0.330** | Bad — actively worse than saying 50/50 for events with ~50% base rate | Overconfident wrong predictions |
| **1.000** | Worst possible — you predicted 100% for everything that didn't happen | Perfectly wrong |

**Concrete example**: The Fed has a meeting. You predict "85% chance they hold rates." They hold rates.
- Your Brier score for this prediction: (0.85 − 1)² = (−0.15)² = **0.0225** — excellent!
- If they had *cut* rates instead: (0.85 − 0)² = (0.85)² = **0.7225** — terrible! You were very confident and very wrong.

**Why the Brier score is our primary metric**: It's intuitive (lower = better), decomposable (we can diagnose *why* it's good or bad — see Thread 3), and the standard used by ForecastBench, the Good Judgment Project, Metaculus, and every major forecasting platform.

---

#### 5. Logarithmic Score (Log Loss / Negative Log-Likelihood)

**What it is**: An alternative scoring rule that punishes overconfident wrong predictions much more harshly than the Brier score.

**The formula**:

```
Log Loss = −(1/N) × Σ [outcome × log(forecast) + (1 − outcome) × log(1 − forecast)]
```

Or equivalently: for each prediction, the score for that prediction is **−log(probability assigned to what actually happened)**.

**How to read it**: If you assigned probability *p* to the thing that actually happened, your log loss for that prediction is **−log(p)**. The key property: log loss goes to **infinity** as *p* approaches zero. This means:

| Your Prediction | What Happened | Brier Penalty | Log Loss Penalty |
|---|---|---|---|
| 95% yes | Yes | 0.0025 (tiny) | 0.05 (small) |
| 70% yes | Yes | 0.09 (small) | 0.36 (moderate) |
| 50% yes | Yes | 0.25 (moderate) | 0.69 (large) |
| 10% yes | Yes | 0.81 (large) | 2.30 (huge) |
| 1% yes | Yes | 0.98 (very large) | 4.61 (enormous) |
| 0.1% yes | Yes | 0.998 (near max) | 6.91 (catastrophic) |
| 0.01% yes | Yes | 0.9998 (~max) | 9.21 (approaching ∞) |

**The crucial difference**: Brier score's maximum penalty is 1.0. Log loss has *no* maximum — if you assign near-zero probability to something that happens, log loss punishes you infinitely. This means:
- **Use Brier** when you want a forgiving, bounded metric that doesn't blow up from a single bad prediction.
- **Use Log Loss** when you want to severely punish a model that says "this will almost certainly not happen" right before it happens (e.g., predicting 1% probability of a Russian invasion the day before the invasion).

**Practical note**: Because log(0) = −∞, you must always **clip** probabilities in practice. The standard approach: never let any probability go below 0.01 or above 0.99 (or 0.001/0.999 for a more permissive range). ForecastBench uses clipping at 0.01/0.99.

---

#### 6. CRPS — Continuous Ranked Probability Score

**Citation**: Matheson, J.E. & Winkler, R.L. (1976). "Scoring Rules for Continuous Probability Distributions." *Management Science*, 22(10), 1087–1096.

**What it is**: A scoring rule for when the thing you're predicting is a continuous number (like "what will the GDP growth rate be?" or "how many sanctions will be imposed?"), not a binary yes/no outcome.

**The intuition (no formula needed)**: Imagine you predict a full probability distribution — like a bell curve centered on 2.5% GDP growth. The CRPS measures how far your predicted distribution is from the single observed number (say, actual GDP growth was 2.3%). It integrates the squared difference between your predicted cumulative distribution and a step function at the true value.

**Key property**: When applied to binary outcomes, CRPS reduces to the Brier score. When applied to deterministic (point) forecasts, CRPS reduces to **MAE** (Mean Absolute Error). So CRPS is a generalization that unifies these metrics.

**When to use it for our project**: If we predict continuous outcomes like "what percentage tariff will be imposed?" or "how many days until a ceasefire?", CRPS is the right metric. For binary predictions (will they or won't they?), just use Brier.

---

## Thread 3: Breaking Open the Brier Score — The Murphy Decomposition

### Why you need to decompose the Brier score

A Brier score of 0.18 tells you "your predictions are moderately good." But it doesn't tell you *why* they're moderately good — or more importantly, *how to improve*. The Murphy decomposition cracks open that single number into three components that each tell you something different about what's going on.

---

#### 7. Murphy (1973) — "A New Vector Partition of the Probability Score"

**Citation**: Murphy, A.H. (1973). "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology*, 12(4), 595–600. DOI: [10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)

**Who was Murphy?** Allan H. Murphy was a meteorologist at Oregon State University who spent his career developing the science of forecast verification. His Brier score decomposition is one of the most cited papers in all of forecasting.

**The decomposition (in plain English)**:

```
Brier Score = Reliability − Resolution + Uncertainty

Where:
• Reliability (REL) = How well your probabilities match reality
• Resolution (RES) = How much your predictions vary based on the situation
• Uncertainty (UNC) = How hard the prediction task is (you can't control this)
```

Let's unpack each:

**Reliability (REL)** — "Are your probabilities honest?"
- Take all the times you predicted "70% probability." Did the event actually happen 70% of those times? If yes, REL is low (good). If the event happened 90% of the time when you said 70%, your probabilities are dishonest — you're **underconfident**, and REL is high (bad).
- REL = 0 means perfect calibration. Higher = worse.
- Think of it as: **trust.** Can I take your stated probability at face value?

**Resolution (RES)** — "Do your predictions actually vary?"
- If you always predict 50%, you have zero resolution — you're never saying anything useful. If sometimes you predict 90% and sometimes 10%, and those predictions correlate with what actually happens, you have high resolution.
- RES = 0 means you're uninformative (always predicting the base rate). Higher = better.
- Think of it as: **information.** Do your predictions tell me something I didn't already know from the base rate?

**Uncertainty (UNC)** — "How hard is this?"
- If the event happens 50% of the time (maximum uncertainty), UNC = 0.25 (its maximum). If the event almost always happens (99%) or almost never happens (1%), UNC is near 0.
- You can't change UNC — it's a property of the problem, not your predictions.
- UNC = base_rate × (1 − base_rate).
- Think of it as: **difficulty.** Don't expect low Brier scores on inherently unpredictable tasks.

**The four diagnostic scenarios**:

| REL | RES | Diagnosis | What To Do |
|---|---|---|---|
| Low (good) | High (good) | **Excellent forecaster** — well-calibrated AND informative | Keep doing what you're doing |
| High (bad) | High (good) | **Knowledgeable but miscalibrated** — your predictions vary meaningfully, but your stated probabilities don't match reality | Apply post-hoc calibration (Temperature scaling, Platt scaling — see Thread 5). Your model "knows something" but needs its confidence adjusted. |
| Low (good) | Low (bad) | **Well-calibrated but useless** — you always predict near the base rate | Your model doesn't actually know anything. Improve the model. |
| High (bad) | Low (bad) | **Terrible** — wrong AND uninformative | Start over |

**Concrete example for our project**: Suppose our LLM predicts 100 FOMC decisions with overall Brier score = 0.15. The decomposition gives REL = 0.08, RES = 0.18, UNC = 0.25.
- REL = 0.08 (moderate) — The LLM's probabilities don't perfectly match reality. When it says 80%, things happen about 70% of the time. It's somewhat overconfident.
- RES = 0.18 (high) — The LLM's predictions vary a lot across meetings. It distinguishes "easy" meetings (95% hold) from "close call" meetings (55% hold), and those distinctions correlate with what happens.
- UNC = 0.25 — FOMC decisions are genuinely uncertain (roughly 50/50 between hold and change in our sample).
- **Diagnosis**: The model knows something (high RES) but is overconfident (moderate REL). **Solution**: Apply temperature scaling to reduce overconfidence while preserving the model's discrimination ability.

---

## Thread 4: Measuring Calibration — Is Your Model Honest?

### What is calibration?

**Calibration** answers a simple question: *When your model says "70% probability," does the thing actually happen about 70% of the time?*

A perfectly calibrated model's stated probabilities match observed frequencies exactly. In practice, no model is perfectly calibrated, so we need tools to measure *how far off* the calibration is and *in which direction* (overconfident or underconfident).

### Key Papers & Metrics

---

#### 8. Reliability Diagram (Calibration Curve) — The Most Important Visualization

**What it is**: A plot that shows whether your model's probabilities match reality.

**How to read it**:
1. Take all your predictions and sort them into **bins** by predicted probability (e.g., 0-10%, 10-20%, ..., 90-100%).
2. For each bin, calculate the **actual frequency** — how often the event really happened when you predicted that probability range.
3. Plot **predicted probability** (x-axis) versus **actual frequency** (y-axis).
4. The **diagonal line** (y = x) represents perfect calibration.

**Interpreting the diagram**:
- **Points ON the diagonal** → perfect calibration in that range
- **Points BELOW the diagonal** → **overconfident** — the model says "80%" but it only happens 60% of the time. The model thinks it knows more than it does.
- **Points ABOVE the diagonal** → **underconfident** — the model says "30%" but it actually happens 50% of the time. The model is excessively cautious.

**Why this is the #1 visualization**: A single reliability diagram tells you instantly whether your model is trustworthy, in which probability ranges it fails, and whether the failure is overconfidence or underconfidence. Every results section in our research should include one.

---

#### 9. Naeini, Cooper & Hauskrecht (2015) — Expected Calibration Error (ECE)

**Citation**: Naeini, M.P., Cooper, G.F. & Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2015)*.

**What ECE is**: A single number that summarizes how miscalibrated your model is. It's the **weighted average** of the gap between predicted confidence and actual accuracy across all probability bins.

**The formula (plain English)**:

```
ECE = Σ (fraction of predictions in bin) × |accuracy in bin − average confidence in bin|
```

For each bin:
- **Accuracy** = fraction of predictions in that bin where the model was correct
- **Confidence** = average predicted probability in that bin
- **Gap** = |accuracy − confidence|
- Weight each bin by how many predictions it contains

**Interpreting ECE**:

| ECE | Interpretation |
|---|---|
| < 0.02 | Excellent calibration — near-perfect |
| 0.02 – 0.05 | Good calibration — trustworthy probabilities |
| 0.05 – 0.10 | Moderate miscalibration — probabilities are off but usable |
| 0.10 – 0.15 | Poor calibration — stated probabilities are unreliable |
| > 0.15 | Very poor — don't trust the model's confidence at all |

**Important limitation**: ECE is sensitive to how you choose bins. With few predictions (< 100), the bins have very few data points each, making ECE noisy and unreliable. For small samples, rely more on the reliability diagram (visual inspection) than on ECE (a single number).

**What "Bayesian Binning into Quantiles" (BBQ) is**: Naeini proposed an improvement: instead of using fixed-width bins (0-10%, 10-20%, etc.), use **equal-count bins** (each bin has the same number of predictions). This is more robust when predictions cluster in certain ranges (as LLMs often do — they tend to predict near 50% or near 90%).

---

#### 10. Maximum Calibration Error (MCE)

**What it is**: While ECE tells you the *average* calibration error, MCE tells you the **worst-case** calibration error — the single bin where your model is most miscalibrated.

**Formula**: MCE = max over all bins of |accuracy in bin − confidence in bin|

**Why it matters**: A model might have low ECE (good average calibration) but terrible MCE (horrible calibration in one specific probability range). For example, the model might be well-calibrated everywhere except the 80-90% range, where it says "85%" but the event only happens 55% of the time. ECE would average this out; MCE would flag it.

**Use for our project**: After computing ECE, always check MCE to identify specific "danger zones" where the model's confidence can't be trusted. These zones often correspond to the model being asked about topics where it has less training data or knowledge.

---

#### 11. Guo et al. (2017) — "On Calibration of Modern Neural Networks"

**Citation**: Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017). "On Calibration of Modern Neural Networks." *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*. arXiv: [1706.04599](https://arxiv.org/abs/1706.04599).

**What they discovered (the landmark finding)**: Modern deep neural networks — the same type of AI technology that underlies ChatGPT, GPT-4, and all LLMs — are **systematically overconfident**. Their predicted probabilities are consistently higher than their actual accuracy.

**In plain English**: If a modern neural network says "I'm 90% confident this is a cat," it's actually correct only about 70-80% of the time. The model *thinks* it knows more than it does. This is a universal property of modern deep learning — it's not a bug in one particular model, it's a feature (or rather, a bug) of how these models are trained.

**What causes the overconfidence**:
- **Model depth**: Deeper networks (more layers) are more overconfident. Modern LLMs have hundreds of layers.
- **Model width**: Wider networks are more overconfident.
- **Batch normalization**: A common training technique that improves accuracy but worsens calibration.
- **Weight decay**: Regularization that affects calibration in complex ways.

Before ~2016, simpler networks (shallow, narrow) were actually quite well-calibrated. As networks got bigger and more powerful (better at classification), they got *worse* at calibration. **Accuracy and calibration have diverged** — modern networks are more accurate than ever, but their confidence scores are less trustworthy than ever.

**The fix they proposed — Temperature Scaling** (see entry 14 below): A surprisingly simple fix. Learn a single number (a "temperature" parameter T) on a validation set, and divide all the model's internal confidence scores (called "logits") by T before converting them to probabilities. That's it — one number fixes most of the overconfidence.

**Why it matters for our project**: LLMs are extremely deep neural networks. Per Guo et al., they are almost certainly overconfident by default. When our LLM says "90% chance Putin escalates," the true probability might be closer to 70-75%. We *must* apply post-hoc calibration (temperature scaling or better) to every LLM's predictions before trusting its probabilities. This is not optional — it's a prerequisite for honest evaluation.

---

#### 12. Kadavath et al. (2022) — "Language Models (Mostly) Know What They Know"

**Citation**: Kadavath, S. et al. (2022). "Language Models (Mostly) Know What They Know." arXiv: [2207.05221](https://arxiv.org/abs/2207.05221). (Anthropic research.)

**What they studied**: A foundational question for our project — can LLMs accurately assess how confident they *should* be? Do they "know what they know"?

**How the experiments worked**:
1. **Multiple choice calibration**: They asked large language models (specifically Anthropic's Claude predecessors) thousands of multiple-choice and true/false questions across diverse domains.
2. **P(True) — Self-evaluation**: After the model generated an answer to an open-ended question, they asked it: "Is the above answer correct? Give the probability that it is true." They called this probability estimate **P(True)**.
3. **P(IK) — Self-knowledge**: They trained models to predict whether they would be able to answer a question correctly *before seeing any answer*. They called this **P(I Know)** or P(IK).

**What they found**:
- **Larger models are better calibrated**: As models get bigger (more parameters), their confidence scores on multiple-choice questions become more aligned with their actual accuracy. This is encouraging — it means scaling up models improves calibration, not just accuracy.
- **Few-shot prompting improves calibration**: Giving the model a few examples of questions with correct answers (few-shot prompting) improved calibration more than zero-shot prompting (no examples).
- **P(True) works**: When asked "is your answer correct?", LLMs give reasonably calibrated probability estimates. This means we can use the LLM's own self-assessment as a confidence score.
- **P(IK) generalizes somewhat**: Models could predict which questions they'd get right even without seeing any candidate answer. However, calibrating P(IK) on new task types (domains it hasn't seen) is still challenging.

**The key limitation**: Calibration improves with scale, but it's not perfect. Large models are still somewhat overconfident, especially on questions they find difficult. The "mostly" in the paper title is doing real work.

**Why it matters for our project**: This tells us that when we ask an LLM to predict a leader's decision, we *can* also ask it "how confident are you?" and get a *somewhat* meaningful answer — especially with larger models. But we shouldn't take that confidence at face value. We should collect these self-assessments, plot them on a reliability diagram, and apply post-hoc calibration.

---

#### 13. Tian et al. (2023) — "Just Ask for Calibration"

**Citation**: Tian, K., Mitchell, E., Yao, H., Manning, C.D., & Finn, C. (2023). "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback." *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)*. arXiv: [2305.14975](https://arxiv.org/abs/2305.14975).

**The problem they address**: Modern LLMs like ChatGPT and GPT-4 are fine-tuned with **RLHF** (Reinforcement Learning from Human Feedback) — a process where human raters grade the model's responses and the model learns to generate responses humans prefer. But this process creates a calibration problem: RLHF-trained models learn to sound confident and authoritative (because humans prefer confident-sounding answers), even when they're wrong.

**What "verbalized confidence" means**: Instead of looking at the raw probability the model assigns to each token (which requires access to the model's internals), you simply *ask the model* to state its confidence: "On a scale of 0 to 100%, how confident are you in this answer?"

**What they found**:
- **Verbalized confidences are often better calibrated** than the model's internal token probabilities for RLHF-fine-tuned models. This is surprising — asking the model "how sure are you?" gives you a more honest signal than looking at its internal math.
- **Prompting the model to consider multiple answers first** (before stating confidence) significantly improves calibration. If you ask the model "consider several possible answers: [answer A], [answer B], [answer C]. Now, how confident are you that [answer A] is correct?", the resulting confidence score is more calibrated.
- **GPT-4 and Claude are better calibrated than GPT-3.5** when using verbalized confidence — larger, newer models have better self-awareness.

**Why it matters for our project**: This tells us *how* to extract probabilities from LLMs for our forecasting tasks. The best approach:
1. Ask the LLM "What will leader X do? Generate several scenarios."
2. Ask the LLM "Of these scenarios, estimate the probability of each."
3. Use these verbalized probabilities as the forecast.
4. Apply post-hoc calibration to correct any remaining miscalibration.

This is likely more calibrated than just asking the model for a single probability directly.

---

#### 14. Desai & Durrett (2020) — "Calibration of Pre-trained Transformers"

**Citation**: Desai, S. & Durrett, G. (2020). "Calibration of Pre-trained Transformers." *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)*. arXiv: [2003.07892](https://arxiv.org/abs/2003.07892).

**What they studied**: Whether the popular pre-trained transformer models (BERT, RoBERTa — the foundation architectures that later evolved into GPT and other LLMs) are well-calibrated when applied to natural language tasks.

**Key findings**:
- **Pre-trained transformers are surprisingly well-calibrated in-domain** — when tested on the same type of data they were fine-tuned on, their confidence scores are reasonably trustworthy.
- **Out-of-domain calibration is where they fail** — when tested on data from a different domain or distribution, calibration degrades significantly. However, *pre-trained* models still maintain much better out-of-domain calibration than models trained from scratch.
- **Temperature scaling works** — even for transformer models, the simple temperature scaling fix (Guo 2017) effectively reduces calibration error.
- **Label smoothing helps out-of-domain** — a training technique called label smoothing (deliberately introducing some artificial uncertainty during training) improves out-of-domain calibration by preventing the model from becoming too confident.

**Why it matters for our project**: Our LLM will be making predictions in a domain (political leader decisions) that is very different from its training data (mostly web text, code, and dialogue). This paper tells us to *expect poor calibration out-of-domain*. The good news: temperature scaling can still help, and the pre-training itself provides some calibration robustness. We should fine-tune our calibration correction on political prediction data specifically.

---

## Thread 5: Post-Hoc Calibration — Fixing a Miscalibrated Model

### The core idea

Your model is trained and deployed. Its predictions discriminate well (it can tell hard cases from easy cases) but its stated probabilities are off — usually overconfident. You don't want to retrain the model (expensive, might hurt accuracy). Instead, you apply a **post-hoc calibration** method: a mathematical transformation applied *after* the model produces its predictions, to make the probabilities more honest.

Think of it like adjusting the speedometer on a car. The engine runs fine (discrimination), but the speedometer reads too high (overconfidence). You calibrate the speedometer without touching the engine.

---

#### 15. Temperature Scaling (Guo et al., 2017)

**Citation**: Same as entry 11 above. Guo, C. et al. (2017). "On Calibration of Modern Neural Networks." ICML 2017.

**How it works (step by step)**:

1. **Start with logits**: Before a neural network produces probabilities, it produces raw scores called **logits** (think of these as "un-normalized confidence scores"). The standard process converts logits to probabilities using the **softmax function**, which turns scores into numbers between 0 and 1 that sum to 1.

2. **Divide by temperature**: Instead of applying softmax directly, divide all logits by a single number T (the "temperature") before applying softmax:
   - If T > 1: probabilities are "softened" — pushed toward 50/50. This reduces overconfidence.
   - If T < 1: probabilities are "sharpened" — pushed toward 0% or 100%. This increases confidence.
   - If T = 1: nothing changes — this is the default.

3. **Learn T on validation data**: Take a held-out dataset (predictions the model hasn't been trained on), and find the value of T that minimizes the calibration error (typically measured by negative log-likelihood / log loss on the validation set).

**Why temperature scaling is remarkable**: It's a single-parameter fix. You learn *one number* (T), apply it to *all* predictions, and calibration dramatically improves. Guo et al. showed it works as well as or better than more complex methods (Platt scaling, isotonic regression, Bayesian approaches).

**Typical temperature values in practice**:
- For modern neural networks: T ≈ 1.5 to 2.5 (networks are overconfident, so you cool them down)
- For LLMs specifically: T ≈ 1.2 to 2.0 depending on the domain

**Limitation**: Temperature scaling assumes the miscalibration is *uniform* — the model is equally overconfident in all probability ranges. If it's well-calibrated at 50% but overconfident at 90%, temperature scaling can't fix this differential miscalibration. For that, you need Platt scaling or isotonic regression.

---

#### 16. Platt (1999) — "Probabilistic Outputs for Support Vector Machines"

**Citation**: Platt, J.C. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." In *Advances in Large Margin Classifiers*, MIT Press, pp. 61–74.

**What Platt scaling is**: Instead of learning just one parameter (temperature), Platt scaling learns *two* parameters (A and B) and fits a logistic function to the model's outputs:

```
calibrated_probability = 1 / (1 + exp(A × logit + B))
```

**In plain English**: Platt scaling is like adjusting both the *sensitivity* and the *offset* of your confidence. Temperature scaling only adjusts sensitivity (how spread out the probabilities are). Platt scaling also adjusts where the "center point" is — useful when the model is systematically biased toward predicting one class.

**When to use Platt over Temperature**:
- Use **Temperature** when you have limited validation data (only 1 parameter to learn)
- Use **Platt** when you have moderate validation data (~50+ predictions) and suspect asymmetric miscalibration (overconfident on "yes" predictions but well-calibrated on "no" predictions)

---

#### 17. Isotonic Regression — The Non-Parametric Option

**What it is**: A calibration method that makes *no assumptions* about the shape of the miscalibration. It simply learns a monotonically increasing mapping from predicted probability to calibrated probability using a step function.

**How it works (plain English)**:
1. Take all your predictions and their outcomes
2. Sort them by predicted probability
3. Fit a step function that maps predicted probabilities to the observed frequencies — but with the constraint that the function must be **monotonically increasing** (higher predictions must map to higher or equal calibrated probabilities)

**Strengths**: Can fix any shape of miscalibration — not just uniform overconfidence. If the model is overconfident at 80% but underconfident at 30%, isotonic regression handles this.

**Weaknesses**: Needs much more validation data (~200+ predictions) because it has many more effective parameters. With too little data, it overfits and makes calibration *worse*.

**When to use for our project**: If we have extensive pilot data (200+ predictions with known outcomes) and the reliability diagram shows complex, non-uniform miscalibration.

---

## Thread 6: Discrimination — Can Your Model Tell Cases Apart?

### What is discrimination?

**Calibration** asks: "Are your probabilities honest?"
**Discrimination** asks: "Can your model tell the difference between cases where the event happens and cases where it doesn't?"

A model can have *good discrimination but bad calibration* — it reliably predicts higher probabilities for events that happen than for events that don't, but the specific numbers are wrong. This is actually the easier problem to fix (apply calibration methods from Thread 5).

A model with *bad discrimination* — it assigns similar probabilities regardless of what actually happens — is fundamentally broken. No calibration method can fix it.

---

#### 18. ROC Curve and AUC — The Standard Discrimination Measure

**What ROC stands for**: **R**eceiver **O**perating **C**haracteristic. The name comes from World War II radar operators who needed to distinguish real enemy aircraft (signal) from noise on their screens.

**How to read a ROC curve**:
1. Imagine sweeping a threshold from 0% to 100% across your model's predictions
2. At each threshold, calculate:
   - **True Positive Rate (TPR)** = sensitivity = fraction of actual events correctly identified
   - **False Positive Rate (FPR)** = fraction of non-events incorrectly flagged as events
3. Plot TPR (y-axis) vs. FPR (x-axis)

**Interpreting the ROC curve**:
- The **diagonal line** (from bottom-left to top-right) = random guessing
- A curve **above** the diagonal = your model does better than random
- A curve in the **top-left corner** = near-perfect discrimination
- **AUC** (Area Under the Curve) summarizes the whole curve as one number: 0.5 = random, 1.0 = perfect

| AUC | Interpretation |
|---|---|
| 0.50 – 0.60 | Random / useless |
| 0.60 – 0.70 | Poor discrimination |
| 0.70 – 0.80 | Acceptable discrimination |
| 0.80 – 0.90 | Good discrimination |
| 0.90 – 1.00 | Excellent discrimination |

**Why AUC is *not* sufficient for our project**: AUC measures discrimination but ignores calibration. A model with AUC = 0.85 could have horrible calibration (its stated probabilities are meaningless). For forecasting tasks, we need *both* AUC (does the model know which events are more likely?) *and* calibration (are its stated probabilities trustworthy?). Report AUC as a complement to Brier score, never as a replacement.

---

## Thread 7: Forecast Aggregation & Recalibration

### Why aggregate forecasts?

A core insight from decades of forecasting research: **combining multiple forecasts almost always outperforms any single forecast**. This is the quantitative version of "wisdom of crowds." For our project, this means combining predictions from multiple LLMs, or from multiple prompting strategies applied to the same LLM, or combining LLM predictions with prediction market odds.

---

#### 19. Baron et al. (2014) — "Two Reasons to Make Aggregated Probability Forecasts More Extreme"

**Citation**: Baron, J., Mellers, B.A., Tetlock, P.E., Stone, E., & Ungar, L.H. (2014). "Two Reasons to Make Aggregated Probability Forecasts More Extreme." *Decision Analysis*, 11(2), 133–145.

**The problem they identified**: When you average multiple probability forecasts together, the average is almost always **too close to 50%**. If three forecasters predict 80%, 90%, and 85%, the average is 85% — but the "correct" aggregate might be 92%.

**Why averaging pushes toward 50% (the two reasons)**:

1. **Mathematical compression**: Probabilities are bounded between 0 and 1. Random errors in individual forecasts, when averaged, tend to push the aggregate toward the midpoint (50%). This is a purely mathematical artifact — it would happen even if all forecasters were perfectly calibrated individually.

2. **Individual hedging**: Each forecaster accounts for their own ignorance by hedging toward 50%. But when you combine forecasters who have *different information*, the group collectively "knows" more than any individual. The aggregate should reflect this collective knowledge by being more extreme.

**The fix — Extremization**: After averaging the forecasts, apply a transformation that pushes the aggregate *away* from 50%. The standard method uses the **Linear Log-Odds (LLO) transformation**:

```
Step 1: Convert the average probability p to log-odds: logit(p) = log(p / (1-p))
Step 2: Multiply by a factor γ (gamma): extremized_logit = γ × logit(p)
Step 3: Convert back: extremized_p = 1 / (1 + exp(-extremized_logit))
```

If γ > 1, the transformation pushes probabilities away from 50% (extremizes). If γ < 1, it pushes toward 50% (de-extremizes). The Good Judgment Project found γ ≈ 2.5 optimal for their superforecaster data.

**Why it matters for our project**: If we average predictions from multiple LLMs (or multiple runs of the same LLM), we should **extremize** the aggregate. Without extremization, our combined forecast will be systematically too moderate. The extremization parameter γ should be tuned on validation data.

---

#### 20. Satopää et al. (2014) — "Combining Multiple Probability Predictions Using a Simple Logit Model"

**Citation**: Satopää, V.A., Baron, J., Foster, D.P., Mellers, B.A., Tetlock, P.E., & Ungar, L.H. (2014). "Combining Multiple Probability Predictions Using a Simple Logit Model." *International Journal of Forecasting*, 30(2), 344–356.

**What this paper does**: Develops a formal statistical model for combining probability predictions that naturally handles the extremization problem.

**The key insight**: Instead of averaging probabilities directly (which causes compression toward 50%), convert all probabilities to **log-odds space** first, average in log-odds space, then convert back. This naturally handles the mathematical compression problem because log-odds space is unbounded (−∞ to +∞), so averaging doesn't push toward any particular value.

**Practical implementation**: The "logit average" is:
1. For each forecaster's probability p_i, compute logit(p_i)
2. Compute the mean of the logit values
3. Convert the mean logit back to a probability

This method consistently outperforms simple probability averaging across multiple forecasting datasets, including the Aggregative Contingent Estimation (ACE) program and the Good Judgment Project.

**Why it matters for our project**: This is the method we should use when combining multiple LLM predictions. Average in log-odds space, not probability space.

---

## Thread 8: The Comprehensive Evaluation Toolkit

### Overview

This section covers additional metrics, reference frameworks, and advanced tools that complete the evaluation pipeline.

---

#### 21. Brier Skill Score (BSS) — Measuring Improvement Over a Baseline

**What it is**: The raw Brier score tells you how good your predictions are in absolute terms. The **Brier Skill Score** tells you how good they are *relative to a baseline* — typically climatology (the historical base rate) or a naive model (always predict 50%).

**The formula**:

```
BSS = 1 − (Brier_model / Brier_reference)
```

**Interpreting BSS**:

| BSS | What It Means |
|---|---|
| 1.0 | Perfect predictions — you've eliminated all error |
| > 0 | Your model beats the reference |
| 0 | Your model is exactly as good as the reference |
| < 0 | Your model is WORSE than the reference — you'd be better off with the naive baseline |

**Why BSS > 0 is hard**: In many forecasting domains (weather beyond 10 days, political events, economic crises), even achieving BSS = 0.05 is considered meaningful. Superforecasters in the IARPA ACE program achieved BSS ≈ 0.15–0.25 relative to the uniform prior. LLMs in ForecastBench achieve BSS ≈ 0.03–0.10 depending on the domain.

**Our baseline hierarchy**:
1. **Random** (Brier = 0.33 for 3-class, 0.25 for binary with 50% base rate)
2. **Majority class** (always predict the most common outcome)
3. **Persistence** (predict that tomorrow = today — things don't change)
4. **Base rate** (use the historical frequency)
5. **Prediction market** (what Metaculus/Polymarket says)
6. **Human superforecaster** (Tetlock's Good Judgment Project)

Our model must beat baseline 4 (base rate) to be minimally interesting, and baseline 5 (prediction markets) to be genuinely useful.

---

#### 22. Wilks (2011) — *Statistical Methods in the Atmospheric Sciences* (Textbook)

**Citation**: Wilks, D.S. (2011). *Statistical Methods in the Atmospheric Sciences*, Third Edition. Academic Press. ISBN: 978-0-12-385022-5.

**Why this textbook is in a political science literature review**: Meteorology is the field that has taken prediction scoring most seriously, for the longest time. Weather forecasters have been making and evaluating probabilistic predictions since the 1950s — that's 75+ years of refining the tools. Every other field that evaluates predictions (political forecasting, financial forecasting, epidemiology) borrows from meteorology.

**What this book covers (relevant chapters)**:
- **Chapter 8: Forecast Verification** — The most comprehensive single-chapter treatment of how to evaluate probabilistic forecasts, covering all the metrics in this review plus dozens more.
- **Reliability diagrams**, **ROC curves**, **skill scores**, **value scores**, **multi-category evaluation**, and **ensemble verification** — all presented with worked examples on real weather data.

**Key contribution to our framework**: Wilks provides the **skill score framework** that we adapt for all our baselines. His formulation of the Brier Skill Score, along with its decomposition into reliability, resolution, and uncertainty skill components, is the standard used in ForecastBench and we adopt it directly.

---

#### 23. Vovk & Shafer (2005) — Conformal Prediction

**Citation**: Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer. ISBN: 978-0-387-00152-4.

**What conformal prediction is**: A method for attaching **guaranteed coverage intervals** to any prediction model. Instead of asking "what's the probability of this outcome?", conformal prediction asks "what set of outcomes is this model confident about, such that I can *mathematically guarantee* the true outcome falls in this set at least 90% (or 95%, or 99%) of the time?"

**Why "distribution-free" matters (in plain English)**: Most statistical methods assume your data follows a specific pattern (often a bell curve / Gaussian distribution). Real-world data — especially political events — rarely follows clean mathematical patterns. Conformal prediction says: "I don't care what pattern your data follows. I will still give you valid uncertainty guarantees, period." This is called being **distribution-free**.

**How it works (simplified)**:
1. Take a calibration set of past predictions with known outcomes
2. For each past prediction, compute a "nonconformity score" — how unusual was the actual outcome relative to the prediction?
3. Use the distribution of these nonconformity scores to set a threshold
4. For new predictions, declare: "the outcome is in this set" where the set includes all outcomes with nonconformity scores below the threshold
5. **Guarantee**: with probability 1 − α (e.g., 95%), the true outcome falls in the set

**The key property**: This guarantee holds with **exactly the claimed probability** (95%), regardless of the model, regardless of the data distribution, regardless of how many samples you have (as long as the data points are **exchangeable** — roughly meaning each data point is equally likely to appear in any position in the sequence).

**Why it matters for our project**: Conformal prediction could give us a mathematically rigorous "uncertainty wrapper" around LLM predictions. Instead of saying "the model is 80% confident," we could say "with 95% statistical guarantee, the true decision falls within this set of outcomes." This is a much stronger claim than any calibration curve can provide.

**Limitation for our use case**: Conformal prediction requires **exchangeability** — the assumption that past data points and future data points come from the same process. Political decisions may not satisfy this (the world changes). This limits the guarantee but the method may still be useful as a heuristic.

---

#### 24. Ferro & Fricker (2012) — "A Bias-Corrected Decomposition of the Brier Score"

**Citation**: Ferro, C.A.T. & Fricker, T.E. (2012). "A Bias-Corrected Decomposition of the Brier Score." *Quarterly Journal of the Royal Meteorological Society*, 138(668), 1954–1960.

**The problem they solve**: Murphy's original Brier decomposition (entry 7) has a known statistical bias — when you have a small sample of predictions (< 200), the decomposition systematically overestimates resolution and underestimates reliability. This means: with few predictions, the decomposition might flatter your model (make it look more skillful than it is).

**Why this matters for our project**: We will likely have fewer than 200 predictions in our initial pilot studies (FOMC meetings happen 8 times per year; major foreign policy decisions are even rarer). With such small samples, the standard Murphy decomposition is unreliable. Ferro & Fricker provide a **bias-corrected** version that gives accurate decomposition even with small samples.

**The fix**: They derive analytical correction terms for each component of the Brier decomposition that correct for the finite-sample bias. Their corrected decomposition satisfies:

```
Corrected_Brier = Corrected_REL − Corrected_RES + UNC
```

Where the corrections asymptotically vanish as sample size increases (with large samples, the corrected and uncorrected versions converge).

**Practical recommendation for our project**: Always use the Ferro-Fricker bias-corrected decomposition for our small-sample political prediction evaluations. Only switch to the standard Murphy decomposition when N > 500.

---

#### 25. Bröcker (2009) — "Reliability, Sufficiency, and the Decomposition of Proper Scores"

**Citation**: Bröcker, J. (2009). "Reliability, Sufficiency, and the Decomposition of Proper Scores." *Quarterly Journal of the Royal Meteorological Society*, 135(643), 1512–1519.

**What this paper does**: Extends the Murphy decomposition (which only works for the Brier score) to work for *any* proper scoring rule — including log loss and CRPS. This means you can diagnose whether miscalibration or lack of resolution is the main problem regardless of which scoring rule you're using.

**Why this is useful**: If we report log loss (because it punishes overconfident wrong predictions more harshly), we still want to decompose it into reliability and resolution components. Bröcker provides the mathematical framework to do this.

---

#### 26. Gneiting, Balabdaoui & Raftery (2007) — "Probabilistic Forecasts, Calibration, and Sharpness"

**Citation**: Gneiting, T., Balabdaoui, F. & Raftery, A.E. (2007). "Probabilistic Forecasts, Calibration, and Sharpness." *Journal of the Royal Statistical Society, Series B*, 69(2), 243–268.

**What this paper adds beyond Gneiting & Raftery 2007 (entry 2)**: While entry 2 focused on the mathematical theory of scoring rules, this paper focuses on the *operational* question: how do you actually check if a probabilistic forecast is calibrated and sharp in practice?

**Key contributions**:
- **PIT histograms**: The **Probability Integral Transform** (PIT) is a tool for checking calibration of probabilistic forecasts. If the forecast is perfectly calibrated, the PIT values follow a uniform distribution (flat histogram). Deviations from flatness indicate miscalibration:
  - **U-shaped** PIT histogram → forecast is overdispersed (too uncertain)
  - **∩-shaped** (hump-shaped) histogram → forecast is underdispersed (too confident/overconfident)
  - **Skewed** histogram → forecast is biased (systematically too high or too low)

- **The calibration-sharpness paradigm operationalized**: They provide concrete statistical tests for calibration and visual tools for assessing sharpness.

**Why it matters for our project**: PIT histograms are another diagnostic tool we can use alongside reliability diagrams. They're particularly useful when predictions are for continuous outcomes (CRPS context), while reliability diagrams are better for binary outcomes (Brier context).

---

#### 27. Tetlock (2005) — *Expert Political Judgment* (Reference Text)

**Citation**: Tetlock, P.E. (2005). *Expert Political Judgment: How Good Is It? How Can We Know?* Princeton University Press. ISBN: 978-0-691-12302-8.

**What this book is**: The landmark study that launched modern forecasting research. Philip Tetlock tracked 28,000+ predictions from 284 political experts over 20 years and found that, on average, experts predicted no better than simple baselines ("dart-throwing chimpanzees," in his memorable phrase).

**The quantitative findings (with proper scoring)**:
- Experts achieved **Brier scores averaging ~0.24–0.26** for binary predictions — barely better than the no-skill baseline of 0.25 (always predicting 50/50).
- For three-category predictions (things get better / stay the same / get worse), experts slightly outperformed a "each outcome is equally likely" baseline (33% each), but only slightly.
- **Foxes** (experts who drew on many frameworks and were comfortable with uncertainty) performed better than **hedgehogs** (experts with one big idea who were very confident), but neither group excelled.

**Why this is our benchmark**: This establishes the *floor* that LLMs must beat. If our LLM cannot beat the expert Brier scores from Tetlock's study (~0.24), it provides no value over just asking a political analyst. If it achieves Brier scores in the 0.15–0.20 range (superforecaster territory from the later Good Judgment Project), it would represent a genuine advance.

---

#### 28. Tetlock & Gardner (2015) — *Superforecasting: The Art and Science of Prediction*

**Citation**: Tetlock, P.E. & Gardner, D. (2015). *Superforecasting: The Art and Science of Prediction.* Crown. ISBN: 978-0-804-13667-8.

**What changed between 2005 and 2015**: After Tetlock's devastating 2005 findings, IARPA (the Intelligence Advanced Research Projects Activity — the U.S. intelligence community's research funding arm) launched the **ACE program** (Aggregative Contingent Estimation) to find out if *anyone* could consistently outperform base rates.

**What they found**: Yes — a small fraction of forecasters (dubbed **"superforecasters"**) consistently outperformed. Key stats:
- The top 2% of forecasters (superforecasters) achieved **Brier scores ~0.15** across diverse geopolitical questions
- They outperformed the simple aggregate by ~30%
- Their advantage was **persistent** — the same people remained on top year after year, ruling out luck
- Superforecasters beat professional intelligence analysts who had access to classified information

**What makes superforecasters different** (the traits we want LLMs to emulate):
1. **Break problems into sub-components** (Fermi estimation)
2. **Start from the base rate and adjust** (Bayesian updating)
3. **Consider multiple perspectives** (foxes, not hedgehogs)
4. **Update frequently** as new evidence arrives
5. **Distinguish between levels of "maybe"** — they use precise probabilities, not vague language

**Why this is our gold standard**: If an LLM can match superforecaster performance (Brier ~0.15), it would represent a breakthrough — a machine matching the top 2% of human forecasters. Our evaluation framework should specifically compare LLM Brier scores to the superforecaster benchmark from Tetlock's research.

---

## Thread 9: Integrating Everything — The Complete Evaluation Protocol

### The evaluation stack we will use (in order of importance)

| Priority | Metric | What It Tells You | When to Report |
|---|---|---|---|
| 1 | **Brier Score** | Overall prediction quality | Always |
| 2 | **Murphy Decomposition** (bias-corrected per Ferro 2012) | Whether the problem is calibration or discrimination | Always |
| 3 | **Reliability Diagram** | Visual calibration check | Always |
| 4 | **ECE / MCE** | Numerical calibration summary | Always |
| 5 | **Brier Skill Score** | Improvement over each baseline | Always |
| 6 | **Log Loss** | Penalty for confident wrong predictions | When overconfidence is a concern |
| 7 | **AUC / ROC** | Discrimination ability | When comparing models |
| 8 | **CRPS** | For continuous outcomes only | When predicting magnitudes/timing |
| 9 | **Conformal Prediction sets** | Distribution-free coverage guarantee | When rigorous uncertainty bounds are needed |

### The calibration pipeline

1. **Raw LLM predictions** → collect them (verbalize confidence per Tian 2023, or use P(True) per Kadavath 2022)
2. **Reliability diagram** → diagnose the type of miscalibration (Guo 2017 showed modern networks are overconfident)
3. **Temperature scaling** → apply as first-pass fix (learn T on held-out political prediction validation data)
4. **Check residual miscalibration** → if reliability diagram still shows problems, try Platt scaling or isotonic regression
5. **Evaluate calibrated predictions** → compute Brier, decomposition, BSS against all baselines
6. **Extremize aggregated predictions** → if combining multiple models, use log-odds aggregation (Satopää 2014 + Baron 2014 extremization)

### Baseline hierarchy for our project

| Baseline | Expected Brier Score | BSS = 0 Threshold |
|---|---|---|
| Random (50/50) | 0.250 | Must beat this to claim ANY skill |
| Persistence (status quo) | ~0.10–0.15 (if status quo is strong, as in FOMC) | Must beat this for practical utility |
| Base rate (historical frequency) | Variable (depends on event frequency) | Must beat this for scientific contribution |
| Prediction market (Metaculus/Polymarket) | ~0.12–0.18 (well-calibrated, crowd wisdom) | Must beat this to claim LLM adds value over markets |
| Superforecaster (Tetlock) | ~0.15 | Must match this to claim LLM matches elite humans |

---

## Summary Statistics

| Metric | Count |
|---|---|
| Total papers/entries covered | 28 |
| Research threads | 9 |
| New sources added in enrichment | 10+ (Savage 1971, Dawid 1984, Kadavath 2022, Tian 2023, Desai & Durrett 2020, Baron 2014, Satopää 2014, Wilks 2011, Vovk & Shafer 2005, Ferro & Fricker 2012) |
| Date range of literature | 1950–2023 |

---

*Last updated: 2026-03-26. Part of the Evaluation Framework Development for the LLM World Leader Prediction Project.*
