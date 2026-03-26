# Domain 1: Political Decision Theory & Predictability — Literature Review

> **Question this review answers**: Which political decisions are theoretically predictable, what models have been used to predict them, what accuracy do they achieve, and where are the hard limits?

---

## Thread 1: Game Theory & Expected Utility Models

### Key Papers

**1. Bueno de Mesquita, B. (1985–2009). "The Predictioneer's Game" / Expected Utility Model**
- **Link**: [Hoover Institution overview](https://www.hoover.org/research/predictioneers-game)
- **Summary**: Bueno de Mesquita developed a game-theoretic model that takes four inputs per actor (stated position, influence/power, salience, risk propensity) and solves for Bayesian Perfect Equilibria to predict the most likely political outcome. Inputs are gathered from area experts.
- **Claimed accuracy**: ~90% across 2,000+ predictions, CIA-tested on real-time policy questions. A study of 21 European Community decisions showed 97% accuracy.
- **Why it matters for us**: This is the closest precedent to what we're doing — predicting specific political outcomes from structured actor profiles. The key question is whether LLMs can replicate the "area expert" input step automatically.

**2. Scholz, J. et al. (2011). "Re-examining predictions of the expected utility model" — [SCIRP](https://www.scirp.org)**
- **Summary**: Independent replication effort that confirmed >90% accuracy on a subset of Bueno de Mesquita's predictions, but raised questions about selection bias and the degree to which standard expert analysis already achieves similar accuracy.
- **Why it matters**: Establishes that the model works but the *incremental value* over expert intuition may be modest — the same question we must answer for LLMs.

**3. Criticisms of Expected Utility Models**
- **Rationality assumption**: Critics (e.g., Hafner-Burton et al., 2017, *Annual Review of Political Science*) argue that the model assumes perfect rationality, but leaders often act under stress, emotional states, and incomplete information — contradicting the model's foundations.
- **Data quality dependence**: Predictions are only as good as the expert-sourced inputs. Proprietary code made full replication difficult, raising transparency concerns.
- **Selection bias**: Some argue the 90% figure is inflated by predicting outcomes that were already likely, or by selecting problems amenable to the model (Strategicscience.org critique).
- **Why it matters**: Any LLM prediction system will face the same criticisms. We need to demonstrate performance on *surprising* outcomes, not just consensus cases.

---

## Thread 2: Bounded Rationality & Cognitive Bias in Foreign Policy

### Key Papers

**4. Simon, H. (1955). "A Behavioral Model of Rational Choice" — *Quarterly Journal of Economics***
- **DOI**: 10.2307/1884852
- **Summary**: Introduced the concept of "bounded rationality" — decision-makers don't optimize, they "satisfice" (settle for good enough). Under uncertainty and time pressure, they rely on heuristics.
- **Why it matters**: If leaders satisfice rather than optimize, their decisions follow *patterns* dictated by their heuristics — patterns that an LLM trained on their past behavior could learn.

**5. Kahneman, D. & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk" — *Econometrica***
- **DOI**: 10.2307/1914185
- **Summary**: People are loss-averse — they take more risks to avoid losses than to achieve gains. Applied to foreign policy: leaders in a "domain of losses" (e.g., perceived national decline) are predictably more aggressive.
- **Why it matters**: Prospect theory creates *directional predictions* — if we can assess whether a leader perceives themselves in a domain of gains or losses, we can predict their risk tolerance.

**6. Jervis, R. (1976). "Perception and Misperception in International Politics" — *Princeton University Press***
- **Summary**: Systematic analysis of how cognitive biases (mirror imaging, cognitive dissonance, belief perseverance) distort threat perception in foreign policy. These biases create "predictable departures from ideal rationality."
- **Why it matters**: Biases are systematic and measurable. Cognitive dissonance (rejecting evidence that contradicts beliefs) and the availability heuristic (overweighting recent vivid events) produce predictable patterns that could be modeled.

**7. Mintz, A. & DeRouen, K. (2010). "Understanding Foreign Policy Decision Making" — *Cambridge University Press***
- **DOI**: 10.1017/CBO9780511757761
- **Summary**: Introduces the "poliheuristic theory" which combines cognitive-psychological constraints (leaders first eliminate unacceptable options using heuristics) with rational choice (then optimize among remaining options). Two-stage process.
- **Why it matters**: This two-stage model suggests foreign policy decisions are *partially* predictable — the elimination stage is driven by known political constraints (no leader will pick an option that guarantees electoral loss), while the optimization stage is harder to predict.

---

## Thread 3: Political Instability Forecasting

### Key Papers

**8. Goldstone, J. et al. (2010). "A Global Model for Forecasting Political Instability" — *American Journal of Political Science***
- **DOI**: 10.1111/j.1540-5907.2009.00426.x
- **Summary**: The Political Instability Task Force (PITF), funded by CIA, achieved 80–90% accuracy in predicting state failure and instability 2 years in advance. Used only 4 variables: regime type (specifically factionalism), infant mortality, armed conflict in neighbors, and state-led discrimination.
- **Why it matters**: Demonstrates that some macro political outcomes are highly predictable with surprisingly few variables. Establishes the 80%+ accuracy bar for computational political forecasting. The parsimony of the model (4 variables!) is a key finding.

**9. Ward, M. et al. (2010). "The Perils of Policy by P-Value" — *Journal of Peace Research***
- **DOI**: 10.1177/0022343309356491
- **Summary**: Re-analysis of PITF showing that statistical significance and predictive accuracy are different things. Models that look good in-sample can fail out-of-sample. Argued for separating explanatory models from predictive models.
- **Why it matters**: Directly relevant to our project — we must evaluate *predictive* accuracy (out-of-sample), not just fit to historical data. This paper is the methodological warning label.

**10. ACLED Conflict Alert System (CAST)**
- **Link**: [acleddata.com](https://acleddata.com)
- **Summary**: ML-based early warning system that forecasts political violence events globally, using real-time event data. Publishes accuracy metrics for its forecasts.
- **Why it matters**: A live, operational prediction system with published accuracy — a direct comparison point for any AI-based political prediction.

---

## Thread 4: Limits of Social Prediction

### Key Papers

**11. Salganik, M. et al. (2020). "Measuring the Predictability of Life Outcomes with a Scientific Mass Collaboration" — *PNAS***
- **DOI**: 10.1073/pnas.1915006117
- **Summary**: 160 research teams used ML on 15 years of rich longitudinal data (Fragile Families study) to predict 6 life outcomes (GPA, grit, material hardship, eviction, job training, layoff). **Even the best models barely beat a simple 4-variable baseline.** Prediction error was driven more by which family was being predicted than by which algorithm was used.
- **Why it matters**: **This is the single most important cautionary finding for our project.** It shows that even with rich data and sophisticated methods, social prediction has hard limits. If 160 teams with 15 years of data can barely beat a simple baseline, we should temper expectations for political prediction.

**12. Tetlock, P. (2005). "Expert Political Judgment: How Good Is It? How Can We Know?" — *Princeton University Press***
- **Summary**: 20-year study tracking ~28,000 predictions from 284 political experts. **Experts barely beat chance.** "Hedgehog" thinkers (one big theory) were systematically worse than "fox" thinkers (many small ideas). Foxes showed better calibration — their stated probabilities matched reality more closely.
- **Why it matters**: (1) Establishes that human experts are a *low* baseline — beating them is necessary but not sufficient. (2) The hedgehog/fox distinction maps directly to LLM prompting strategies: a persona-conditioned LLM that can weigh multiple factors (fox-like) should outperform one locked into a single ideology (hedgehog-like). (3) Calibration matters — a well-calibrated predictor is more useful than an overconfident one.

---

## Thread 5: Crisis vs. Routine Decision-Making

### Key Papers

**13. Hermann, C.F. (1969). "International Crises as a Situational Variable" — in *International Politics and Foreign Policy***
- **Summary**: Defined crises along three dimensions: (1) high threat to values, (2) short time for response, (3) element of surprise. These dimensions constrain the decision space and change how leaders process information.
- **Why it matters**: Crises *reduce* the information-processing capacity of leaders, making them fall back on heuristics and prior beliefs — which are more predictable. But the triggering event itself (the surprise) is inherently unpredictable.

**14. Brecher, M. & Wilkenfeld, J. (1997). "A Study of Crisis" — *University of Michigan Press***
- **Summary**: Developed comprehensive crisis theory through the International Crisis Behavior (ICB) project, analyzing 470+ international crises from 1918–2015. Created structured coding of crisis triggers, actors, responses, and outcomes.
- **Why it matters**: The ICB dataset provides structured historical data on crisis decisions — potentially usable as training/evaluation data for LLM crisis prediction. The coding scheme could inform how we structure context for the LLM.

**15. Allison, G. & Zelikow, P. (1999). "Essence of Decision" (2nd ed.) — *Longman***
- **Summary**: Three models of crisis decision-making: Rational Actor (unitary state optimizing), Organizational Process (standard operating procedures), Governmental Politics (bureaucratic bargaining). Each model makes different predictions about the same event.
- **Why it matters**: The key insight is that *which model best predicts* depends on the type of decision. Routine decisions follow Organizational Process (SOPs). Crisis decisions involve Governmental Politics (harder to predict). This informs our "easiest first" strategy: start with routine decisions governed by SOPs.

---

## Thread 6: Institutional Constraints & Veto Players

### Key Papers

**16. Tsebelis, G. (2002). "Veto Players: How Political Institutions Work" — *Princeton University Press***
- **DOI**: 10.1515/9781400831456
- **Summary**: Defines "veto players" as actors whose agreement is necessary to change the status quo. Core finding: more veto players + greater ideological distance between them = more policy stability = more predictable outcomes (because the status quo persists). Fewer veto players = faster, less predictable policy shifts.
- **Why it matters**: Directly maps to leader selection for our project. Leaders in systems with many veto players (US Congress, EU Council) produce more predictable outcomes because the status quo is hard to change. Leaders in systems with few veto players (autocracies with concentrated power) can produce surprise decisions.

**Implication for leader ranking**:
| System Type | Veto Players | Policy Stability | Predictability |
|---|---|---|---|
| US (Congressional gridlock) | Many | High | High (status quo persists) |
| Russia/China (executive dominance) | Few formal | Low | Lower (leader can act unilaterally) |
| EU (consensus-based) | Very many | Very high | Very high (change is glacial) |
| Argentina (Milei deregulation) | Reduced | Low | Lower (rapid policy shifts) |

---

## Thread 7: Complex Adaptive Systems & Fundamental Limits

### Key Papers

**17. Jervis, R. (1997). "System Effects: Complexity in Political and Social Life" — *Princeton University Press***
- **Summary**: International politics is a complex adaptive system where interconnectedness, nonlinear dynamics, and emergence create fundamental unpredictability. Small causes can have large effects (and vice versa). Feedback loops can be delayed and non-obvious. The whole exceeds the sum of its parts.
- **Why it matters**: Establishes the *theoretical ceiling* on prediction. Some political phenomena are computationally irreducible — knowing the rules actors follow is insufficient to predict system-level behavior. This means our project should focus on *individual actor decisions* (more predictable) rather than *system-level outcomes* (less predictable).

**18. Cederman, L-E. (1997). "Emergent Actors in World Politics" — *Princeton University Press***
- **Summary**: Used agent-based modeling to show how macro-level international phenomena (alliances, wars, system polarity) emerge from micro-level interactions in non-predictable ways.
- **Why it matters**: Reinforces the distinction between *actor-level prediction* (feasible) and *system-level prediction* (much harder). Our project is correctly scoped at the actor level.

---

## Thread 8: Computational Political Event Prediction

### Key Papers

**19. ICEWS — Integrated Crisis Early Warning System (Lockheed Martin / DARPA)**
- **Link**: [dataverse.harvard.edu](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075)
- **Summary**: Automated coding of millions of news reports into structured political events (who did what to whom, when). Used for early warning of political instability. Provides the CAMEO event coding scheme.
- **Why it matters**: A major data source for political event prediction. The CAMEO coding scheme could inform how we structure ground truth events for our evaluation.

**20. ACLED — Armed Conflict Location & Event Data Project**
- **Link**: [acleddata.com](https://acleddata.com)
- **Summary**: Real-time global dataset of conflict events with date, location, actors, fatalities, and event type. Their CAST system uses ML to forecast political violence events.
- **Why it matters**: An operational prediction system with published accuracy — unlike academic models, this one runs in production and makes real forecasts about real events.

---

## Synthesis: What This Means for Our Project

### What's theoretically predictable
| Decision Type | Predictability | Evidence |
|---|---|---|
| Scheduled institutional votes (UNGA, FOMC) | **High** | Procedural models + historical patterns, PITF 80-90% |
| Policy persistence (status quo) | **High** | Tsebelis: more veto players = more stability |
| Leader behavior under known ideology | **Moderate-High** | Bueno de Mesquita 90%, bounded rationality patterns |
| Crisis response | **Low-Moderate** | Hermann: leaders fall back on heuristics, but trigger is unpredictable |
| System-level emergent outcomes | **Low** | Jervis: nonlinear, computationally irreducible |

### What we should take from each thread

1. **Start with institutional/routine decisions** (FOMC, UNGA votes) — these are governed by SOPs and have the highest base predictability
2. **Actor-level prediction is feasible** — system-level is not our problem
3. **Expect modest gains over baselines** — Fragile Families shows even rich data barely beats simple models
4. **The "surprise" dimension matters most** — beating base rates on obvious predictions proves nothing; finding signal on surprising decisions is the entire thesis
5. **Veto player analysis informs leader selection** — high-veto-player systems (US Congress, EU) are more predictable but less interesting; low-veto-player decisions (executive orders, UNSC vetoes, tariff actions) are harder but where value lies
6. **Foxes over hedgehogs** — our LLM prompting should encourage multi-perspective reasoning, not single-ideology prediction

### Papers to read in full (top 5)
1. Salganik et al. (2020) — Fragile Families Challenge (PNAS)
2. Tetlock (2005) — Expert Political Judgment
3. Goldstone et al. (2010) — PITF global model (AJPS)
4. Tsebelis (2002) — Veto Players
5. Bueno de Mesquita (2009) — The Predictioneer's Game
