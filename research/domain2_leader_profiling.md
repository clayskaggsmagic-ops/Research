# Domain 2: Leader Profiling & Operational Code Analysis — Literature Review

> **The big question this review answers**: Can we build a psychological profile of a world leader — just from their speeches, interviews, and public statements — and then use that profile to *predict what they'll do next*? What methods exist to do this? How accurate are they? And can AI do it better than humans?

> **Why this matters for our project**: If political psychologists have already figured out how to measure a leader's personality and beliefs from text, and shown that those measurements predict real decisions, then an LLM might be able to do the same thing faster and at scale. This domain reviews 50+ years of research that tried exactly this — profiling leaders "at a distance" (without ever meeting them) and testing whether those profiles correctly predicted what happened next.

---

## Thread 1: VICS — The Verb-Based System for Reading a Leader's Mind

### What is VICS?

**VICS** stands for **Verbs in Context System**. It's a method for figuring out what a leader *believes about the world* by analyzing the *verbs* they use in speeches, interviews, and press conferences.

The core idea is simple: **the verbs a leader uses reveal how they think.** If a president constantly uses words like "threaten," "demand," "attack," and "punish" when talking about other countries, that tells you something very different than a president who says "cooperate," "negotiate," "agree," and "reward."

VICS doesn't just count positive vs. negative words. It specifically looks at **transitive verbs** (action words that have a target — like "I *threatened* them" or "They *attacked* us") and codes each verb on two dimensions:

1. **Direction**: Is the verb positive/cooperative (promise, reward, agree) or negative/conflictual (threaten, punish, oppose)?
2. **Intensity**: How strong is the action? A verbal threat is less intense than a military strike. VICS uses a 6-level intensity scale: appeal → promise → reward → oppose → threaten → punish.

From this coding, VICS generates two types of belief scores (each on a scale from –1.0 to +1.0):

- **Philosophical beliefs** (P-scores): How the leader sees the political world. Is the universe fundamentally hostile (score near –1) or cooperative (score near +1)? Are other leaders basically threats or potential partners?
- **Instrumental beliefs** (I-scores): How the leader prefers to act. Do they favor cooperation and diplomacy (score near +1) or conflict and coercion (score near –1)?

Together, these produce **17 quantitative indices** that form a leader's **"operational code"** — a numerical fingerprint of their worldview and preferred strategies.

### Key Papers

---

#### 1. Walker, Schafer & Young (1998) — "Systematic Procedures for Operational Code Analysis"

**Citation**: Walker, S.G., Schafer, M. & Young, M.D. (1998). "Systematic Procedures for Operational Code Analysis: Measuring and Modeling Jimmy Carter's Operational Code." *International Studies Quarterly*, 42(1), 175–189.

**What they did (in plain English)**: These three researchers invented VICS. Before this paper, "operational code analysis" (the study of what leaders believe about politics and how they prefer to act) was done by hand — a single researcher would read everything a leader said and write a subjective essay about their worldview. This was slow, inconsistent, and couldn't be replicated by other researchers. Walker, Schafer, and Young automated it.

**How the method works in practice**:
- **Input**: Take any speech, interview transcript, or press conference by a leader.
- **Step 1**: Identify every transitive verb (an action word with a subject and object — "I warned them," "they cooperated with us").
- **Step 2**: For each verb, determine: Is the actor "Self" (the leader or their country) or "Other" (foreign leaders/countries)? Is the action positive (cooperative) or negative (conflictual)?
- **Step 3**: Rate the intensity of each verb on a 6-point scale (from mild "appeal" to extreme "punish/reward").
- **Step 4**: Calculate 17 numerical indices from these codings. Five are "philosophical" (beliefs about the world) and five are "instrumental" (beliefs about the best strategy), plus seven utility-of-means indices.
- **Output**: A numerical profile of the leader — a set of scores that can be compared across leaders, across time, and across issues.

**Test case — Jimmy Carter**: They applied VICS to Carter's foreign policy speeches across his entire presidency (1977–1981). The results showed Carter's worldview shifted measurably after the Soviet invasion of Afghanistan in December 1979 — his philosophical beliefs became significantly more hostile (the world is a more dangerous place) and his instrumental beliefs shifted toward conflict (force is more necessary). This belief shift *preceded* his actual policy changes (the Carter Doctrine, the Olympic boycott, increased defense spending), suggesting the beliefs *caused* the policy, not the other way around.

**Why it matters for our project**: VICS is the closest existing system to what we want an LLM to do — read a leader's words and extract structured, numerical psychological variables. If an LLM can replicate VICS coding, we skip the entire manual analysis pipeline. The 17 indices also give us concrete, measurable numbers to compare against (not just vague impressions).

---

#### 2. Walker & Schafer (2006) — *Beliefs and Leadership in World Politics*

**Citation**: Walker, S.G. & Schafer, M. (2006). *Beliefs and Leadership in World Politics: Methods and Applications of Operational Code Analysis*. New York: Palgrave Macmillan.

**What this is**: This is the definitive textbook/edited volume collecting all the evidence that leader beliefs (as measured by VICS) actually predict real-world decisions. It's not a single study — it's a collection of case studies by different researchers.

**What the case studies showed**:
- **U.S. presidents**: Operational codes varied significantly between presidents and predicted their foreign policy orientations. Hawks had different VICS profiles than doves, and those profiles were stable within a president's term but different across presidents.
- **Middle Eastern leaders**: VICS was applied to translated speeches of leaders like Saddam Hussein, showing that the method works cross-culturally (not just for English-speaking Western leaders).
- **European heads of state**: Different European leaders showed distinct operational codes that predicted their positions on issues like EU integration and NATO expansion.
- **Belief changes preceding behavioral changes**: The most important finding across all case studies was that when a leader's beliefs changed (measurable through VICS), their *actions* changed afterward. The beliefs were a *leading indicator* — like a turn signal before a car changes lanes.

**Why it matters for our project**: This book is the strongest evidence that the profiling-to-prediction pipeline works in principle. If beliefs (measurable from text) predict actions (observable in the real world), then an LLM that can measure beliefs can potentially predict actions.

---

#### 3. Marfleet & Miller (2005) — "Failure after 1441: Bush and Chirac in the UN Security Council"

**Citation**: Marfleet, B.G. & Miller, C. (2005). "Failure after 1441: Bush and Chirac in the UN Security Council." *Foreign Policy Analysis*, 1(3), 365–385.

**What they did**: Applied VICS to the public statements of George W. Bush and Jacques Chirac during the buildup to the Iraq War — specifically the period around **UN Security Council Resolution 1441** (November 2002), which unanimously demanded Iraq comply with weapons inspections. The question was: why did unanimous agreement on Resolution 1441 collapse into total disagreement about whether to invade?

**What they found**:
- **Bush's operational code**: His philosophical beliefs showed a relatively hostile view of the political universe (other actors are threats), and his instrumental beliefs favored coercive, confrontational strategies. His VICS scores predicted he would push for military action.
- **Chirac's operational code**: His philosophical beliefs showed a more cooperative view (other actors can be negotiated with), and his instrumental beliefs favored diplomatic solutions. His VICS scores predicted he would resist military action.
- **The critical finding**: Both leaders had an **"elevated sense of control"** — each believed they could shape the outcome. And both had **inaccurate perceptions of their opponent's preferences** — Bush didn't realize how firmly Chirac would oppose war, and Chirac didn't realize how committed Bush was to invasion. The diplomatic failure wasn't just about Iraq — it was about two leaders misreading each other.

**Why it matters for our project**: This is one of the cleanest tests of VICS as a prediction tool. The researchers used VICS profiles to predict which side each leader would take on a specific question (invade Iraq or not), and VICS got the *direction* right for both leaders. It couldn't predict the *timing* or exact diplomacy, but it correctly predicted who would be hawkish and who would be dovish.

---

#### 4. Feng (2005) — "The Operational Code of Mao Zedong"

**Citation**: Feng, H. (2005). "The Operational Code of Mao Zedong: Defensive or Offensive Realist?" *Political Psychology*, 26(5), 751–779.

**What they did**: Applied VICS to Mao Zedong's speeches across different periods of his leadership — from the early People's Republic through the Cultural Revolution and beyond.

**What they found**: Mao's operational code was *not* static. His philosophical and instrumental beliefs shifted systematically based on China's domestic political situation:
- During periods of domestic consolidation, Mao's operational code was more moderate and cooperative externally.
- During the Cultural Revolution (1966–1976), his operational code became significantly more conflictual — he saw the world as more hostile and preferred confrontational strategies.
- These shifts preceded corresponding changes in China's foreign policy behavior.

**Why it matters for our project**: Two critical validations. First, VICS works **cross-culturally** — it can be applied to translated Chinese speeches, not just English text. This matters because our project aims to profile leaders like Xi Jinping whose primary language isn't English. Second, it shows that leader beliefs **change over time**, and VICS can track those changes. We can't just build a single static profile of a leader and assume it holds forever — we need to update the profile as new speeches come in.

---

## Thread 2: LTA — The 7-Trait Personality Fingerprint

### What is LTA?

**LTA** stands for **Leadership Trait Analysis**. It was created by political psychologist **Margaret Hermann** and is the most widely used framework in academic political psychology for profiling world leaders.

The idea: You can figure out a lot about how a leader will behave by measuring 7 specific personality traits from their spontaneous speech (interviews, Q&A sessions, press conferences — *not* prepared speeches, which are usually written by speechwriters and don't reflect the leader's actual thinking patterns).

### The 7 Traits (explained in plain English)

| # | Trait | What It Measures | How It's Coded |
|---|---|---|---|
| 1 | **Belief in ability to control events** | Does the leader think they can shape what happens, or do they see themselves as at the mercy of forces beyond their control? | Count verbs that imply the leader's action can directly cause outcomes |
| 2 | **Need for power** | How much does the leader want to dominate, influence, and control others? | Count references to asserting authority, making demands, issuing ultimatums |
| 3 | **Conceptual complexity** | Does the leader see the world in black-and-white (low complexity) or in shades of gray (high complexity)? | Count "high-complexity" words (e.g., "approximately," "sometimes," "possibly") vs. "low-complexity" words (e.g., "always," "never," "absolutely") |
| 4 | **Self-confidence** | How certain is the leader of their own judgment? | Count self-referential statements expressing certainty or conviction |
| 5 | **Distrust of others** | Does the leader generally suspect the motives of other people/countries? | Count statements expressing suspicion, doubt about others' intentions |
| 6 | **In-group bias** | Does the leader strongly favor "us" over "them"? | Count positive references to own group vs. negative references to out-groups |
| 7 | **Task vs. relationship orientation** | Is the leader focused on getting things done (task) or on maintaining relationships and group harmony? | Count attention to instrumental goals vs. attention to group dynamics and feelings |

### How These 7 Traits Combine into Leadership Styles

Hermann showed that combinations of these traits predict distinct **leadership styles** — recurring patterns of how leaders respond to challenges:

| Style | Trait Pattern | Predicted Behavior |
|---|---|---|
| **Crusader** | High belief in control + low conceptual complexity + high in-group bias | Challenges constraints, refuses compromise, doubles down under pressure, sees issues in black-and-white moral terms |
| **Strategist** | High conceptual complexity + moderate distrust | Plans carefully, considers multiple angles, adapts tactics to circumstances |
| **Pragmatist** | Low ideology + high relationship orientation | Works within constraints, builds coalitions, compromises when necessary |
| **Opportunist** | Reactive + high self-confidence | Responds to situations as they arise, confident in improvising, less driven by ideology |

### Key Papers

---

#### 5. Hermann (1980) — "Explaining Foreign Policy Behavior Using the Personal Characteristics of Political Leaders"

**Citation**: Hermann, M.G. (1980). "Explaining Foreign Policy Behavior Using the Personal Characteristics of Political Leaders." *International Studies Quarterly*, 24(1), 7–46.

**What she did**: Created the entire LTA framework. She analyzed the spontaneous verbal material (unscripted interviews, press conferences, parliamentary Q&A) of **122 national leaders from 48 countries** and established baseline scores for each of the 7 traits. This created a "norming group" — a population of leaders that any new leader can be compared against (is leader X above or below average on "need for power" compared to the 122-leader baseline?).

**Data details**:
- **Sample**: 122 leaders from 48 countries across all global regions, spanning 1945–1999. Included heads of state, revolutionary leaders, cabinet ministers, opposition leaders, and even terrorist leaders.
- **Text requirement**: Minimum **50 pieces of text** or **10,000 words** of spontaneous material per leader for a reliable profile.
- **Validation**: Hermann compared LTA-generated profiles with assessments from people who had **directly interacted** with the leaders — journalists, former diplomats, government officials. The average correlation between the "at-a-distance" LTA profile and the "in-person" assessment was **r = 0.84** (on a 0-to-1 scale, where 1.0 = perfect agreement). This is extremely high for a psychological method.
- **Intercoder reliability** (do different researchers get the same results when coding the same text?): Ranged from **0.78 to 1.00** across the 7 traits. New coders must achieve **≥ 0.90 reliability** before their analyses are included in the dataset.

**Why it matters for our project**: This is the foundational dataset. The 7-trait framework gives us a concrete, measurable set of personality variables that an LLM could potentially extract from leader text. The 0.84 validation correlation is the benchmark to beat — if an LLM-generated profile correlates with real-world assessments at 0.84 or higher, it matches human expert performance.

---

#### 6. Hermann (2003) — "Assessing Leadership Style: A Trait Analysis"

**Citation**: Hermann, M.G. (2003). "Assessing Leadership Style: A Trait Analysis." In J.M. Post (ed.), *The Psychological Assessment of Political Leaders*. University of Michigan Press, pp. 178–212.

**What she did**: Updated and refined the LTA framework. The key addition was the **leadership style typology** — showing that combinations of traits produce distinct, predictable behavioral patterns (the crusader/strategist/pragmatist/opportunist types described above).

**The critical insight**: It's not just individual traits that matter — it's how they *combine*. A leader who is high on "belief in ability to control events" AND low on "conceptual complexity" (sees things in black and white) is a **crusader** who will challenge constraints, refuse compromise, and see their cause as a moral imperative. But a leader who is high on "belief in ability to control events" AND high on "conceptual complexity" (sees shades of gray) is a **strategist** who will also try to shape events but through flexible, adaptive means.

**Why it matters for our project**: The leadership style typology gives us testable predictions. If we profile a leader as a "crusader" using LTA traits, we can predict they'll challenge constraints and refuse compromise when faced with opposition — and then check whether that's what actually happened. This is a concrete evaluation criterion for our LLM: does a leader persona prompted with "crusader" traits behave like a crusader?

---

#### 7. Profiler/ProfilerPlus — Social Science Automation

**Citation**: Social Science Automation, Inc. ProfilerPlus software. [socialscience.net](https://socialscience.net) / [profilerplus.org](https://profilerplus.org)

**What this is**: A computer program (not an AI — it uses hand-crafted rules, not machine learning) that automates both LTA and VICS coding. It's been developed over 10+ years and is the current state-of-the-art tool that political psychologists actually use in practice.

**How it works**:
- **Input**: Raw text (speech transcript, interview, etc.)
- **Processing**: Rule-based Natural Language Processing (NLP) — the software applies hundreds of hand-written rules and dictionaries to identify relevant verbs, adjectives, and phrases.
- **Output**: Numerical scores for all 7 LTA traits and all VICS operational code indices.
- **Languages supported**: English, Arabic, Spanish, Russian, and Chinese.
- **Availability**: Free for academic use at profilerplus.org.

**Current accuracy**:
- VICS automated coding accuracy improved from **6% to 59%** agreement with expert human coders over the tool's development lifetime.
- The target is **80%+ agreement** with human coders — currently unmet.
- LTA coding accuracy is higher than VICS because the traits are simpler to code (counting specific word types rather than interpreting verb context).

**Why it matters for our project**: ProfilerPlus is the bar we need to clear. It achieves about 59% agreement with human experts on VICS coding. If an LLM can exceed that — and we expect modern LLMs can, since they understand context much better than rule-based systems — it would be an immediate, measurable improvement over the current state-of-the-art. The 80% target is our goalpost.

---

## Thread 3: Integrative Complexity (IC) — The Escalation Early Warning System

### What is Integrative Complexity?

**Integrative Complexity (IC)** measures how *complex* or *simple* a leader's thinking is, as reflected in their public communications. It was developed by psychologist **Peter Suedfeld** starting in the 1970s.

The concept has two components:

1. **Differentiation**: Can the leader see multiple sides of an issue? A person with low differentiation sees everything as "good vs. evil" or "us vs. them." A person with high differentiation acknowledges that an issue has multiple dimensions, that the enemy might have legitimate concerns, or that a policy has both benefits and costs.

2. **Integration**: Can the leader see *connections* between those different perspectives? It's not enough to see multiple sides — integration means you can synthesize them, see trade-offs, and understand how different factors interact.

IC is scored on a **1-to-7 scale**:
- **1** = No differentiation at all. Black-and-white thinking. "They are evil and we must stop them."
- **3** = Moderate differentiation but no integration. "There are multiple factors at play, but our course of action is clear."
- **5** = High differentiation with moderate integration. "The situation is complex, these factors are interconnected, and we need to balance competing concerns."
- **7** = Maximum complexity. "The problem has multiple interrelated dimensions, previous solutions have created new problems, and any approach involves irreducible trade-offs."

### The Critical Discovery: IC Drops Before Wars

The most important finding about IC — and the reason it's relevant to our project — is that **IC scores in government communications consistently *drop* before wars and *rise* during peaceful resolutions.** This has been replicated across dozens of crises spanning over a century.

### Key Papers

---

#### 8. Suedfeld & Tetlock (1977) — "Integrative Complexity of Communications in International Crises"

**Citation**: Suedfeld, P. & Tetlock, P.E. (1977). "Integrative Complexity of Communications in International Crises." *Journal of Conflict Resolution*, 21(1), 169–184.

**What they did**: This was the original study that discovered the IC-conflict link. They took diplomatic communications (letters, cables, official statements) from multiple international crises in the 20th century and scored each document for IC on the 1-7 scale.

**The crises they studied** (among others):
- **World War I (1914)**: Diplomatic communications between the major powers (Austria-Hungary, Germany, Russia, France, Britain) in the weeks before war broke out.
- **Cuban Missile Crisis (1962)**: Communications between the U.S. and Soviet Union during the 13-day confrontation over Soviet missiles in Cuba.
- **Multiple Arab-Israeli conflicts**: Communications before and during the 1956, 1967, and 1973 wars.

**What they found**:
- **WWI**: IC in diplomatic communications **declined sharply** in the weeks before war broke out. Leaders' thinking became simpler, more black-and-white, and more hostile as the crisis escalated toward war. By the time war was declared, IC was near the bottom of the scale.
- **Cuban Missile Crisis**: The opposite pattern. IC **stayed moderate or increased** throughout the crisis, even at its most dangerous moment (when Soviet ships approached the U.S. naval blockade). Kennedy and Khrushchev maintained complex, multi-dimensional thinking, which the researchers argued helped them find a face-saving compromise.
- **Arab-Israeli wars**: Crises that led to war showed the WWI pattern (IC decline); crises that were resolved peacefully showed the Cuban Missile pattern (IC maintained or increased).

**The conclusion**: When leaders' public communications show *declining* integrative complexity, it's an early warning that they are moving toward conflict. When IC *stays stable or rises*, they are more likely to find a peaceful resolution.

**Why it matters for our project**: IC decline is a **leading indicator of escalation** — it happens *before* the actual military action, not just during it. If we can measure IC from LLM-generated leader speech or from real-time analysis of leader communications, we have a predictive signal for escalation with over 40 years of empirical backing.

---

#### 9. Suedfeld (2010) — "The Cognitive Processing of Politics and Politicians"

**Citation**: Suedfeld, P. (2010). "The Cognitive Processing of Politics and Politicians: Archival Studies of Conceptual and Integrative Complexity." *Journal of Personality*, 78(6), 1669–1702.

**What this is**: A 30-year retrospective — Suedfeld looked back at three decades of IC research and synthesized all the findings.

**Crises where IC predicted the outcome**:
- WWI (1914) — IC decline predicted war ✓
- Munich Crisis (1938) — IC decline predicted appeasement-then-war ✓  
- Cuban Missile Crisis (1962) — IC stability predicted peaceful resolution ✓
- Arab-Israeli Wars (1956, 1967, 1973) — IC decline predicted each war ✓
- Multiple Cold War crises — IC pattern held repeatedly ✓

**Other findings**:
- **U.S. presidents' IC** tends to be highest at the beginning of their first term (when they are most intellectually engaged and open to information) and declines over time (as they become more ideologically committed and less flexible).
- **The "strategic judgment hypothesis"**: Leaders with higher IC are better at resolving coordination problems (situations where both sides need to find a solution) and avoiding violence. This isn't because they're smarter — it's because they can see more options and are less likely to back themselves into a corner.

**Why it matters for our project**: This paper confirms that IC is one of the most empirically validated predictive signals in political psychology. It's not just one study — it's dozens of studies across dozens of crises over 30+ years. The pattern holds.

---

#### 10. Conway et al. (2014) — "Automated Integrative Complexity"

**Citation**: Conway, L.G. III, Conway, K.R., Gornick, L.J., & Houck, S.C. (2014). "Automated Integrative Complexity." *Political Psychology*, 35(5), 603–624.

**What they did**: Built a computer program called **AutoIC** that tries to score IC automatically (without human coders), using a **dictionary-based approach** — essentially, a list of words and phrases that signal high vs. low complexity.

**How AutoIC works**:
- It scans text for words that signal differentiation ("on the other hand," "alternatively," "however," "but") and integration ("therefore," "because," "as a result," "interconnected").
- Each word/phrase is weighted based on how strongly it predicts IC as scored by human coders.
- The weighted scores are combined to produce an automated IC score.

**How accurate is it?**:
- **Alpha reliability**: 0.72 on the standard IC coding test (acceptable but not great).
- **Correlation with human-coded IC**: Moderate — roughly r ≈ 0.5–0.6. This means AutoIC captures about 25–36% of the variance that human coders capture.
- For comparison, **human inter-rater reliability** (two human coders scoring the same text) is about r ≈ 0.85.

**The gap**: AutoIC gets the general direction right (high complexity vs. low complexity) but misses a lot of nuance that human coders catch. Dictionary-based methods can't understand context — the word "but" might signal differentiation in one sentence and be irrelevant in another.

**A follow-up validation** (Conway et al., 2020, "Validating Automated Integrative Complexity: The Donald Trump Test") showed AutoIC could replicate known findings from human-scored studies, but the overlap between automated and human scores remained modest.

**Why it matters for our project**: This is the existing automated IC benchmark. Its accuracy (~50-60% of human performance) is the bar we need to clear. LLMs understand context far better than dictionary lookup, so an LLM-based IC scorer should significantly outperform AutoIC. If we can get an LLM to score IC at r ≈ 0.80+ with human coders, that would be a genuine advance.

---

## Thread 4: At-a-Distance Assessment — Profiling Leaders You Can't Interview

### What is "at-a-distance" assessment?

In clinical psychology, you assess someone by interviewing them, giving them tests, and observing them directly. But you can't do that with a sitting president or dictator — they won't take your personality test. **At-a-distance assessment** means profiling a leader using only publicly available information: their speeches, interviews, biographies, press conference behavior, and decisions.

### Key Papers

---

#### 11. Post (2003) — *The Psychological Assessment of Political Leaders*

**Citation**: Post, J.M. (2003). *The Psychological Assessment of Political Leaders: With Profiles of Saddam Hussein and Bill Clinton*. Ann Arbor: University of Michigan Press.

**What this is**: The Bible of at-a-distance profiling, written by **Jerrold Post** — a psychiatrist who spent **21 years at the CIA** and founded the agency's **Center for the Analysis of Personality and Political Behavior** in the 1970s. This center's entire job was to psychologically profile foreign leaders for U.S. policymakers.

**Key case studies**:

1. **Begin and Sadat (Camp David Accords, 1978)**: Post profiled both Israeli Prime Minister Begin and Egyptian President Sadat for President Carter before the Camp David negotiations. Carter later credited Post's profiles as **"crucial to the success of the Camp David Accords."** The profiles helped Carter understand what each leader needed psychologically (Begin needed to feel he hadn't surrendered; Sadat needed a grand gesture of peace) and structured the negotiations accordingly.

2. **Saddam Hussein (Gulf War, 1990)**: Post profiled Saddam as **"rational but driven by narcissistic compensatory dreams of glory"** — meaning Saddam wasn't crazy, but his decisions were driven by a deep need for status and legacy. Post's specific prediction: **Saddam would not voluntarily withdraw from Kuwait** because withdrawal would be perceived (by Saddam) as a humiliating defeat that threatened his self-image. Post testified before Congress in 1990 with this prediction. **He was correct** — Saddam did not withdraw, leading to the Gulf War.

**Why it matters for our project**: **This is the gold standard for our problem.** A human expert profiled a leader using only public data, made a specific, falsifiable prediction about what that leader would do in a high-stakes situation, and the prediction came true. Our goal is to see if an LLM can do something analogous — use the same inputs (public data about a leader) to make the same kinds of predictions (what will the leader do next?).

---

#### 12. Winter (2005) — "Things I've Learned About Personality From Studying Political Leaders"

**Citation**: Winter, D.G. (2005). "Things I've Learned About Personality From Studying Political Leaders." *Journal of Personality*, 73(3), 557–584.

**What he did**: Synthesized decades of at-a-distance research and proposed a simplified 3-variable model of leader motivation:

| Motive | What It Drives | Behavioral Prediction |
|---|---|---|
| **Achievement** | Need to accomplish goals, solve problems, meet standards of excellence | Leaders take **calculated risks** — not reckless, but not cautious. They seek concrete accomplishments they can point to. |
| **Affiliation** | Need for warm relationships, belonging, approval | Leaders **seek consensus**, avoid confrontation, prioritize group harmony. May make concessions to maintain relationships. |
| **Power** | Need to influence, dominate, control others | Leaders take **aggressive action**, assert authority, are willing to use force. View negotiations as competitions to be won. |

These three motives are measured from text using standard content analysis (counting specific types of imagery and references in spontaneous speech).

**Key findings**:
- Presidents with high **power motivation** were significantly more likely to enter wars.
- Presidents with high **affiliation motivation** were more likely to negotiate and compromise.
- Presidents with high **achievement motivation** were more likely to pursue ambitious but calculated policy changes.

**Why it matters for our project**: This gives us an alternative, simpler profiling model (3 variables instead of Hermann's 7). For quick LLM-based profiling, a 3-variable model might be more practical — fewer dimensions to estimate means less noise and faster calibration. We could use Winter's 3 motives as a "quick-and-dirty" profiling layer on top of the more detailed VICS/LTA analysis.

---

#### 13. The Goldwater Rule (APA, 1973 / updated 2017)

**What it is**: An ethical guideline from the **American Psychiatric Association (APA)** that says psychiatrists should **not** offer professional opinions about the mental health of public figures they haven't personally examined. It's named after the 1964 U.S. presidential election, when a magazine polled psychiatrists about whether candidate Barry Goldwater was mentally fit for office — most responded without ever meeting him.

**Why it matters for our project**: This is a framing issue, not a methodological one. Our work is **not** diagnostic — we're not claiming any leader is mentally ill. We're using content analysis to predict decision patterns from public behavior. This is what political psychologists (not psychiatrists) do, and the Goldwater Rule explicitly does **not** apply to political psychologists or content-analysis researchers. Understanding this distinction matters for credibility when publishing or presenting our work.

---

## Thread 5: Computational & Automated Profiling

### Key Papers / Systems

---

#### 14. Schafer & Lambert (2022) — PsyCL: "Psychological Characteristics of Leaders" Dataset

**Citation**: Schafer, M. & Lambert, J.E. (2022). "Psychological Characteristics of Leaders (PsyCL): A New Data Set." *Foreign Policy Analysis*, 18(2), orac008. DOI: [10.1093/fpa/orac008](https://doi.org/10.1093/fpa/orac008)
**Website**: [psycldataset.com](https://psycldataset.com)
**Data**: [GitHub (speech-level xlsx)](https://raw.githubusercontent.com/JELambert/Psych_Agg/master/data/csv/speech_level.xlsx)
**Python package**: [github.com/JELambert/Psych_Agg](https://github.com/JELambert/Psych_Agg)

**What this is (in plain English)**: The largest dataset of psychological profiles of world leaders ever compiled. It contains the raw data needed to calculate LTA personality traits and VICS operational code indices for dozens of leaders.

**Dataset details**:
- **Size**: Over **54 million words** of spoken material — speeches, press conferences, interviews, parliamentary Q&A.
- **Format**: The data is at the **speech level** — each row is one speech act by one leader, with raw counts of the components needed to calculate LTA and VICS variables. Some leaders have multiple speeches in a single day.
- **Leaders included**: U.S. presidents and other state leaders. The website provides pre-aggregated data at multiple time scales:
  - **Leader-Months**: Scores aggregated to monthly averages
  - **Leader-Quarters**: Scores aggregated to quarterly averages
  - **Leader-Years**: Scores aggregated to yearly averages
  - **Leaders**: Overall career averages
- **What the numbers represent**: The raw file contains **counts** of linguistic features (e.g., number of cooperative verbs, number of self-referential statements). To get actual VICS/LTA scores, you need to apply mathematical formulas to convert these counts into indices. The Python package (`Psych_Agg`) does this automatically.
- **Important note**: Individual speeches often don't have enough verbal material to produce meaningful scores. The researchers recommend aggregating across multiple speeches (e.g., all speeches in a month) to get stable, reliable scores.

**Why it matters for our project**: **PsyCL is our ground-truth validation dataset.** We can:
1. Give an LLM a leader's speeches and ask it to generate VICS/LTA scores.
2. Compare the LLM-generated scores against the PsyCL scores (calculated by the established method from the same speeches).
3. Measure how well the LLM replicates the "gold standard" profiling method.

If the LLM's scores correlate highly with PsyCL's scores, that's evidence the LLM can do automated leader profiling. If we can additionally show that those LLM-generated scores predict *actual decisions* (e.g., voting patterns, alliance choices, escalation), that's the full pipeline.

---

#### 15. Transformer-based Political Text Analysis (2020–2025)

**Key work**: Rheault, L. & Cochrane, C. (2020). "Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora." *Political Analysis*, 28(1), 112–133.

**What this research thread is about**: Over the last 5 years, researchers in Natural Language Processing (NLP) have started applying modern AI text-analysis tools — specifically **transformer models** like BERT and RoBERTa — to political text. These are the same technology underlying ChatGPT and similar AI systems, but used for analysis rather than generation.

**What "transformers" are (simple explanation)**: A transformer is a type of AI model that reads text and understands meaning in context. Unlike older methods that treated each word independently (a "dictionary" approach — the word "bank" always means the same thing), transformers understand that "bank" means something different in "river bank" vs. "bank account." This contextual understanding makes them much better at analyzing nuanced political language.

**What researchers have done with transformers**:
- **Ideology detection**: Given a speech in Parliament, predict whether the speaker is left-wing or right-wing. Transformers significantly outperform dictionary-based methods at this task.
- **Sentiment analysis**: Measure how positive or negative a leader's statements are about specific topics (trade, immigration, foreign policy).
- **Policy position extraction**: Determine where a leader stands on specific policy dimensions from their speeches.

**The gap**: Despite these advances, **nobody has systematically applied transformers to VICS or LTA coding.** The political psychology community still uses ProfilerPlus (rule-based, 2000s technology), while the NLP community has moved far ahead with transformers. Our project could bridge this gap — using LLMs for VICS/LTA coding should dramatically improve accuracy over ProfilerPlus's 59%.

---

#### 16. Argyle et al. (2023) — "Out of One, Many: Using Language Models to Simulate Human Samples"

**Citation**: Argyle, L.P., Busby, E.C., Fulda, N., Gubler, J.R., Rytting, C., & Wingate, D. (2023). "Out of One, Many: Using Language Models to Simulate Human Samples." *Political Analysis*, 31(3), 337–351. DOI: [10.1017/pan.2023.2](https://doi.org/10.1017/pan.2023.2)

**What they did (the "Silicon Sampling" experiment)**: These researchers asked: can an AI model pretend to be a specific type of person and give survey responses that match how that type of person would actually respond?

**How the experiment worked**:
1. They took real demographic data from the **American National Election Studies (ANES)** — large surveys of thousands of Americans conducted in **2012, 2016, and 2020**. These surveys ask about political attitudes, voting behavior, policy preferences, and demographics.
2. For each real survey respondent, they created a "backstory" from 11 survey questions — things like race/ethnicity, gender, age, political party, ideology (liberal/conservative), church attendance, and how often they discuss politics.
3. They fed this backstory into **GPT-3** (OpenAI's large language model, the predecessor to ChatGPT) as a prompt: "You are a [age]-year-old [race] [gender] who identifies as [party] and [ideology]..."
4. Then they asked GPT-3 a **12th survey question** that wasn't in the backstory — like "Who did you vote for?" or "Do you support gun control?"
5. They compared GPT-3's simulated answers to the **real answers** given by actual humans with the same demographics.

**What they found — "Algorithmic Fidelity"**:
They defined 4 criteria for whether the AI simulation was any good:
1. **Text quality**: Are the generated responses indistinguishable from human-written text? ✓ Yes.
2. **Backward continuity**: Are the responses consistent with the demographic backstory? ✓ Yes — a prompt conditioned as a conservative Republican generated conservative Republican responses.
3. **Forward continuity**: Do the responses naturally follow from the demographic context? ✓ Yes — the model didn't just repeat the backstory, it generated *new* responses consistent with it.
4. **Pattern correspondence**: Do the response distributions match real human populations? ✓ Mostly yes — GPT-3 silicon samples reproduced the distribution of vote choice across demographic subgroups for 2012, 2016, and 2020.

**Key limitations**:
- GPT-3 simulates *average* responses for demographic groups, not specific *individuals*. It can tell you how a typical 65-year-old white Republican man would respond — but not how *your specific uncle* would respond.
- The simulated responses are only as good as the demographic conditioning. If you leave out an important variable (like regional culture), the simulation will be less accurate.

**Why it matters for our project**: **This is the closest existing precedent to what we're trying to do.** Argyle showed that LLMs can simulate the *average* opinion of a demographic group. Our question is the logical next step: can an LLM simulate the *specific* decision of a **named individual** (like Xi Jinping) given sufficient context about their beliefs, personality, and situation? If silicon sampling works for demographics, it might work for individual leaders — but we need to test it.

---

## Thread 6: Connecting Profiles to Predictions — Empirical Tests

### What this thread is about

All the methods above (VICS, LTA, IC) are tools for *profiling* leaders. But the critical question for our project is: **do those profiles actually predict real decisions?** This thread reviews the studies that directly tested whether profiles-based-on-text correctly predicted what leaders actually did.

### Key Papers

---

#### 17. Lazarevska et al. (2004) — "Clinton's Operational Code and Bosnia"

**Citation**: Lazarevska, E., Sholl, J.M., & Young, M.D. (2004). "Clinton and the Kosovo Crisis: A Study of the Operational Code." *Foreign Policy Analysis* (and related work on Clinton's operational code and Bosnia).

**What they did**: Used VICS to construct Bill Clinton's operational code — his philosophical beliefs (how he saw the political world) and instrumental beliefs (how he preferred to act) — and then tested whether those beliefs predicted U.S. policy during the Bosnian crisis (1992–1995).

**What they found**:
- Clinton's operational code showed moderately cooperative philosophical beliefs (the world can be negotiated with) but instrumental beliefs that shifted over time from cooperative to confrontational as the Bosnian crisis worsened.
- **The operational code largely predicted the direction of U.S. policy** — Clinton's shift from diplomatic pressure to military intervention (NATO airstrikes in 1995) was preceded by a measurable shift in his operational code toward more conflictual instrumental beliefs.
- **What it got wrong**: The operational code predicted the *direction* of policy (cooperative → conflictual) but not the precise *timing* (when exactly the shift to military action would happen). The timing was driven by external events (the Srebrenica massacre) that the operational code model couldn't anticipate.

**Why it matters for our project**: This is one of the cleanest "profile → prediction" tests available. The takeaway: operational code analysis predicts **policy direction** (will the leader lean toward force or diplomacy?) with good accuracy, but does **not** predict exact timing. This means our prediction tasks should target directional questions ("Will leader X cooperate or escalate?") rather than precise timing questions ("Will leader X authorize a strike on Tuesday?").

---

#### 18. Özdamar et al. (2019) — Trump's Foreign Policy: Operational Code and LTA Analysis

**Citation**: Özdamar, Ö. et al. (2019). Various publications applying VICS and LTA to Trump's public statements, published across multiple political science journals.

**What they did**: Multiple research teams applied both VICS and LTA to Donald Trump's campaign speeches and early presidency statements to predict his foreign policy orientation.

**What they found — mixed results**:
- ✓ **Correct predictions**: Trump's operational code correctly predicted his general hawkishness, transactional approach to alliances, and preference for bilateral over multilateral negotiations. His LTA traits (high belief in ability to control events, low conceptual complexity, high distrust of others) correctly predicted his confrontational style.
- ✗ **Incorrect predictions**: VICS showed **poor correlation with Trump's specific actions toward North Korea**. His operational code predicted a confrontational approach, but he was actually *more cooperative* than predicted — holding unprecedented summits with Kim Jong-un in 2018–2019. His approach to Iran matched VICS predictions better (confrontational — withdrawal from the nuclear deal, maximum pressure campaign).

**The critical lesson**: Operational codes predict **general orientation** (hawkish vs. dovish, cooperative vs. confrontational) but can fail on **specific actions** when leaders face novel situations or have strategic incentives to deviate from their baseline beliefs. Trump's North Korea engagement was a strategic gamble — he calculated (correctly or not) that personal diplomacy with Kim could succeed where confrontation had failed. His operational code didn't anticipate this because operational codes model *habitual patterns*, not *strategic improvisations*.

**Why it matters for our project**: This is the most important limitation we need to acknowledge. Profile-based prediction works well for routine decisions and general orientation, but it can fail when:
1. The situation is genuinely novel (no precedent for how this leader handles this type of crisis).
2. The leader deliberately acts against their instincts for strategic reasons.
3. External events force a response that doesn't match the leader's baseline preferences.

---

#### 19. Dyson (2006) — "Personality and Foreign Policy: Tony Blair's Iraq Decisions"

**Citation**: Dyson, S.B. (2006). "Personality and Foreign Policy: Tony Blair's Iraq Decisions." *Political Psychology*, 27(6), 911–931. DOI: [10.1111/j.1467-9221.2006.00523.x](https://doi.org/10.1111/j.1467-9221.2006.00523.x)

**What he did**: Applied Hermann's LTA framework to Tony Blair — coding Blair's spontaneous responses to foreign policy questions in the UK House of Commons (not his prepared speeches, since those are written by staff and don't reflect Blair's actual thinking patterns).

**Blair's LTA profile**:

| Trait | Blair's Score | Relative to 87-Leader Baseline | What This Means |
|---|---|---|---|
| Belief in ability to control events | **High** | Above average | Blair believed he could personally shape outcomes |
| Need for power | **High** | Above average | Strong desire to influence and dominate |
| Conceptual complexity | **Low** | Below average | Saw issues in black-and-white terms, not shades of gray |
| Self-confidence | High | Above average | Very certain of his own judgment |
| In-group bias | **High** | Above average | Strong "us vs. them" framing |

**Hermann's typology classification**: These traits combined to classify Blair as a **"crusader"** — a leader who:
- Challenges constraints (will push back against Parliament, the UN, public opinion)
- Refuses to compromise on core positions
- Doubles down under opposition rather than reconsidering
- Frames issues in moral terms ("this is the right thing to do") rather than strategic terms ("this serves our interests")

**What the profile predicted vs. what actually happened**:
- ✓ Predicted Blair would **challenge constraints** (Parliament, the UN) → He did, going to war despite massive domestic opposition and no second UN resolution.
- ✓ Predicted Blair would **refuse compromise** → He did, rejecting multiple off-ramps and alternatives to full military commitment.
- ✓ Predicted Blair would **double down under pressure** → He did, becoming more assertive about the rightness of the Iraq decision even as the post-invasion situation deteriorated.
- ✓ Predicted Blair would **frame the decision in moral terms** → He did, repeatedly presenting the Iraq war as a moral obligation to remove a dictator, not just a strategic calculation.

**Why it matters for our project**: This is one of the strongest validation cases in the entire LTA literature. The profile was derived entirely from text analysis (House of Commons transcripts), classified Blair as a specific leadership type ("crusader"), and then all four behavioral predictions associated with that type were confirmed by his actual behavior during the Iraq crisis. This is the kind of clean, testable prediction our LLM system needs to replicate.

---

#### 20. Rivera, Mukobi et al. (2024) — "Escalation Risks from Language Models in Military and Diplomatic Decision-Making"

**Citation**: Rivera, J.-P., Mukobi, G., Reuel, A., Lamparth, M., Smith, C., & Schneider, J. (2024). "Escalation Risks from Language Models in Military and Diplomatic Decision-Making." *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)*. arXiv: [2401.03408](https://arxiv.org/abs/2401.03408). DOI: [10.1145/3630106.3658942](https://doi.org/10.1145/3630106.3658942)

**What they did**: Set up a **wargame simulation** where large language models (LLMs) played the role of leaders making military and diplomatic decisions, and measured whether the AI "leaders" escalated conflicts more or less than real human wargamers would.

**How the simulation worked**:
- **Setup**: Multiple AI "nations," each controlled by a separate LLM instance. No human oversight — the AI made all decisions autonomously.
- **World model**: A separate LLM summarized the consequences of each nation's actions and fed that summary back into the next round (like a game master narrating what happens).
- **Turn-based**: Each simulated "day," all nations took actions simultaneously. Actions and consequences were revealed to all nations, and then the next round began.
- **Three starting scenarios**:
  1. **Neutral**: No prior conflict. A blank slate to see if LLMs escalate from nothing.
  2. **Invasion**: One nation has already invaded another. How do the LLM "leaders" respond?
  3. **Cyberattack**: One nation has launched a cyberattack. How do the LLM "leaders" respond?

**Models tested**: Five "off-the-shelf" (meaning commercially available, not specially modified) LLMs:
- OpenAI GPT-4
- OpenAI GPT-3.5
- OpenAI GPT-4-Base (the raw model before safety fine-tuning)
- Anthropic Claude 2.0
- Meta Llama-2-Chat (70 billion parameters)

**What they found**:
- **All five models showed escalation tendencies.** Every model, in every scenario, tended to increase military spending, build up forces, and take increasingly aggressive actions over time — even in the neutral scenario where there was no initial conflict.
- **Arms-race dynamics**: The models developed self-reinforcing escalation cycles. If one AI nation built up its military, the others responded by building up theirs, which triggered more buildup, and so on — exactly the dynamic that led to WWI.
- **Nuclear escalation in rare cases**: In some simulation runs, models escalated all the way to nuclear strikes. The most infamous case: **GPT-4-Base** (the version without safety training) recommended a preemptive nuclear strike with the justification **"We have it! Let's use it."**
- **The model without safety training was worst**: GPT-4-Base (which had not undergone "Reinforcement Learning with Human Feedback" — the process that teaches AI to refuse harmful requests) was the most escalatory and unpredictable, confirming that safety training matters.
- **Worrying justifications**: When asked to explain their decisions, the models cited **deterrence theory** and **first-strike advantage** — reasoning that sounds strategic but leads to catastrophic outcomes.
- **Human comparison**: Real human wargamers in similar scenarios are typically much more reluctant to initiate nuclear conflict than the LLMs were.

**Why it matters for our project**: This paper is a critical **calibration warning**. It proves that LLMs *can* simulate leader decision-making — they take actions, explain their reasoning, and respond to changing situations. But they simulate it **badly by default**: they escalate faster than real humans, they develop arms-race dynamics, and they occasionally resort to nuclear weapons with flimsy justifications. This means:
1. Raw LLM persona simulation ≠ accurate prediction. The persona must be grounded in empirical profiles (VICS/LTA) and calibrated against historical decisions.
2. LLM predictions will likely need **de-escalation correction** — systematically adjusting for the model's tendency to over-escalate.
3. Any prediction system we build must compare LLM behavior to known baselines (How did real leaders behave in analogous situations? What does the historical operational code say?).

---

#### 21. Persona-Based Voting Prediction (2024, multiple arxiv papers)

**What this research is**: A cluster of 2024 studies (various research groups, published on arXiv) that tested whether LLMs conditioned with politician personas could predict individual voting behavior in legislatures.

**How the experiments typically worked**:
1. Create an LLM prompt with a politician's background information (party, ideology, committee assignments, previous voting record, state/district demographics).
2. Present the LLM with a specific bill or resolution.
3. Ask the LLM to predict how the politician would vote (yes, no, abstain).
4. Compare the LLM's prediction to the politician's actual vote.

**What they found**:
- **Moderate accuracy**: LLMs predicted individual votes with moderate accuracy — better than random, but not perfect.
- **Strongest predictor**: National party affiliation was by far the strongest signal. The LLM basically learned "Republicans vote for this, Democrats vote against it" and got most predictions right just from that.
- **Systematic left-leaning bias**: LLMs exhibited a consistent left-leaning bias in their persona simulations. When uncertain about how a politician would vote, the model defaulted to politically left-leaning positions. This means the raw LLM output is *not* an unbiased simulation of the leader — it's a simulation filtered through the model's own political tendencies.
- **Better for institutionalized decisions**: Voting in legislatures is highly constrained (party discipline, procedural rules, public records of past votes). LLMs performed better in these structured environments than in less predictable ones.

**Why it matters for our project**: Partial validation meets calibration warning. The good news: LLM personas *can* predict institutionalized decisions (legislative votes) with moderate accuracy. The bad news: systematic political bias in the model means raw LLM output shouldn't be treated as ground truth. Every prediction must be benchmarked against baselines, and we should explicitly measure and correct for any systematic bias in the model's simulations.

---

## Synthesis: What This All Means for Our Project

### The profiling-to-prediction pipeline (explained step by step)

```
Step 1: Collect public text
    Speeches, interviews, press conferences, parliamentary Q&A
    (The more spontaneous/unscripted, the better — prepared speeches
    are written by staff and don't reflect the leader's actual thinking)
              ↓
Step 2: Build a psychological profile
    ├── VICS → Philosophical beliefs (hostile vs. cooperative worldview)
    │          Instrumental beliefs (preferred strategies)
    │          = What the leader believes about how the world works
    │            and what strategies they think are effective
    │
    ├── LTA → 7 personality traits → Leadership style type
    │         = What kind of leader they are (crusader, strategist,
    │           pragmatist, opportunist) and how they respond to pressure
    │
    ├── IC → Integrative Complexity score (1-7 scale)
    │        = How complex or simple their current thinking is
    │          (declining IC = escalation warning sign)
    │
    └── Winter Motives → Power / Affiliation / Achievement scores
                         = What fundamentally drives the leader
              ↓
Step 3: Generate predictions
    ├── Direction (cooperative vs. confrontational) — STRONG evidence
    │   "Will leader X lean toward negotiation or force?"
    │   → SUCCESS RATE: Moderate-to-High across dozens of case studies
    │
    ├── Domain (which issues they'll engage on) — MODERATE evidence
    │   "Which issues will leader X prioritize?"
    │   → SUCCESS RATE: Moderate
    │
    └── Timing & specific actions — WEAK evidence
        "Exactly when will leader X do exactly what?"
        → SUCCESS RATE: Low — too many external factors
```

### What's validated (summary table)

| Method | What It Predicts | Accuracy | Evidence Base |
|---|---|---|---|
| VICS Operational Code | Policy direction (hawk vs. dove on specific issues) | Moderate-High | Case studies: Carter (belief shift → policy shift), Bush/Chirac (opposite positions predicted), Clinton (intervention predicted), Mao (cross-cultural validation) |
| Hermann LTA (7 traits) | How leader responds to constraints; leadership style | Moderate-High | 122 leaders from 48 countries; Blair "crusader" confirmed with 4/4 predictions correct; 0.84 correlation with in-person assessments |
| Integrative Complexity | Whether a crisis will escalate (war) or de-escalate (peace) | High | 30+ years, dozens of crises from WWI through modern era; replicated consistently |
| Post-style profiling | Specific decisions in high-stakes situations | Case-dependent | Saddam non-withdrawal prediction correct; Camp David profiling credited as crucial; but small N (few cases) |
| LLM persona simulation | Legislative voting behavior | Moderate | Multiple 2024 studies; systematic left-leaning bias is a confound |

### 6 key implications for our project

1. **LLMs could replace ProfilerPlus** — LLMs understand language context far better than the rule-based ProfilerPlus system (which currently achieves only 59% agreement with human coders). If an LLM can exceed 80% agreement on VICS/LTA coding, it's an immediate improvement over the state-of-the-art.

2. **PsyCL is our validation dataset** — 54 million words of coded speech, available as a free download with a Python aggregation package. We can benchmark LLM-generated profiles against PsyCL ground truth.

3. **Profile → Direction prediction is validated; Profile → Specific action is not** — We should target directional predictions ("Will leader X cooperate or defect on issue Y?") rather than specific-action predictions ("Will leader X sign this exact treaty on this date?").

4. **IC as a real-time escalation signal** — If we can measure Integrative Complexity from real-time leader communications, a decline in IC should predict escalation. This has 30+ years of replicated empirical backing.

5. **Systematic LLM biases must be calibrated** — LLMs escalate faster than real humans (Rivera et al.) and lean politically left (persona voting studies). Raw LLM output is not ground truth. Every prediction must be benchmarked against baselines (base rate → market prediction → expert → historical operational code).

6. **Start with leaders who have the most public speech data** — VICS/LTA accuracy depends on text volume (minimum 10,000 words for a reliable profile). Leaders with transcribed press conferences, interviews, and spontaneous Q&A produce better profiles than those who only read prepared speeches.

### Top papers to read in full

| Priority | Paper | Why |
|---|---|---|
| 1 | Post (2003) — *Psychological Assessment of Political Leaders* | The Bible of at-a-distance profiling. Saddam prediction case study. |
| 2 | Schafer & Lambert (2022) — PsyCL dataset | Our validation data. 54M words. Free download + Python package. |
| 3 | Walker & Schafer (2006) — *Beliefs and Leadership* | VICS theory + case studies proving beliefs predict behavior. |
| 4 | Argyle et al. (2023) — Silicon Sampling | Closest LLM precedent. Can GPT simulate real human survey responses. |
| 5 | Rivera et al. (2024) — LLM War Simulations | Critical escalation bias warning. Raw LLM personas are unreliable without calibration. |
| 6 | Hermann (1980, 2003) — LTA framework | Foundational 7-trait system, 122-leader baseline, 0.84 validation. |
| 7 | Dyson (2006) — Blair Iraq decisions | Strongest single-leader predictive validation. Crusader profile, 4/4 predictions correct. |
| 8 | Suedfeld (2010) — IC 30-year retrospective | Most empirically validated escalation signal in political psychology. |
