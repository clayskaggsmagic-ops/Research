# Mantic's "Specialized Tools" — Deep Investigation

From [Training LLMs to Predict World Events](https://thinkingmachines.ai/news/training-llms-to-predict-world-events/) (Thinking Machines Lab × Mantic, 2026)

---

## What "Specialized Tools" Actually Means

The phrase "specialized tools" refers to **LLM function-calling / tool-use** — the same mechanism that lets ChatGPT call a calculator or code interpreter. Except instead of a calculator, the "tools" are **probability distribution constructors**.

Here's the concrete mechanism, reconstructed from the article + the cited papers:

### Step 1: The LLM Gets a Tool Schema

The prediction LLM is given access to tool definitions (JSON Schema-style function specs) that let it construct mixture model components. Based on the article's description and standard function-calling patterns, the tool schema looks approximately like this:

```json
{
  "name": "set_mixture_component",
  "description": "Add a component to the temporal mixture model. Each component represents a scenario for when the event might occur.",
  "parameters": {
    "type": "object",
    "properties": {
      "distribution_type": {
        "type": "string",
        "enum": ["lognormal", "exponential", "normal", "uniform"],
        "description": "The type of probability distribution for this component"
      },
      "weight": {
        "type": "number",
        "description": "The relative probability of this scenario (all weights will be normalized to sum to 1)"
      },
      "params": {
        "type": "object",
        "description": "Distribution-specific parameters",
        "properties": {
          "meanlog": { "type": "number", "description": "For lognormal: log-scale mean" },
          "sdlog": { "type": "number", "description": "For lognormal: log-scale standard deviation" },
          "rate": { "type": "number", "description": "For exponential: rate parameter (1/mean)" },
          "mean": { "type": "number", "description": "For normal: mean in days from now" },
          "std": { "type": "number", "description": "For normal: standard deviation in days" }
        }
      }
    },
    "required": ["distribution_type", "weight", "params"]
  }
}
```

### Step 2: The LLM Calls the Tool Multiple Times

During its chain-of-thought reasoning, the LLM decides how many "scenarios" (mixture components) to create and calls the tool for each one. For example, for "Will Iran close the Strait of Hormuz before 2027?":

```
LLM reasoning: "There are roughly two scenarios:
1. A military escalation cycle leads to partial closure within 6-12 months (low probability)
2. The status quo continues indefinitely (high probability)"

Tool call 1: set_mixture_component(
  distribution_type="lognormal", 
  weight=0.15, 
  params={"meanlog": 5.5, "sdlog": 0.8}   // peaks around ~245 days
)

Tool call 2: set_mixture_component(
  distribution_type="exponential", 
  weight=0.85, 
  params={"rate": 0.0005}   // very slow decay = probably never happens
)
```

### Step 3: The System Computes the CDF

The system (not the LLM) takes the tool call outputs and mathematically constructs the mixture distribution:

```
CDF(t) = w₁ · CDF_lognormal(t; μ=5.5, σ=0.8) + w₂ · CDF_exponential(t; λ=0.0005)

where w₁ = 0.15 / (0.15 + 0.85) = 0.15
      w₂ = 0.85 / (0.15 + 0.85) = 0.85
```

### Step 4: Read Off the Probability

The question asks "before 2027" — say that's 365 days from now. The system evaluates CDF(365) and gets, say, 0.08. That's the final prediction: **8% probability**.

But crucially, that same CDF can answer "before 2028?" (CDF at 730 days), "before 2030?" (CDF at 1460 days), etc. — all from a single LLM inference.

---

## Why This Architecture Is Clever (And Why It Matters for Us)

| Property | Flat Probability ("65% yes") | Mixture Model CDF |
|---|---|---|
| **Information density** | 1 number per question | Full curve — answers infinite time horizons from one call |
| **RL-trainable** | Brier score on single point | Brier score at any/all temporal resolution points — much richer gradient signal |
| **Interpretable** | No insight into *why* 65% | Each component = a named scenario with a timeline |
| **Calibratable** | Need many questions to check calibration | Can check calibration *within* a single question across time windows |
| **Composable** | Can't combine predictions | Can overlay multiple agent predictions as additional mixture components |

### The Training Trick

From the Mantic article: they used **Brier score as the RL reward function** (not log loss). Why?

> *"We found that the Brier score leads to more stable training than the log score, even though the log score is also strictly proper. This could be because the Brier score is bounded in [0, 1] and so produces lower-variance policy gradient estimates."*

The reward works like this:
1. The LLM parameterizes a mixture model via tool calls
2. The system computes the CDF
3. The CDF is evaluated at the question's resolution date → probability `p`
4. The actual outcome is known (the question resolved YES or NO) → `y ∈ {0, 1}`
5. **Reward = 1 - (p - y)²** (inverted Brier score — higher is better)
6. This reward signal backpropagates through the RL algorithm (GRPO) to update the LLM's policy

The crucial point: the LLM learns to set better mixture model parameters over time. It's not just learning "what probability to assign" — it's learning **how to decompose uncertain futures into weighted scenarios**.

---

## How the Three Key Papers Relate

| Paper | Architecture | Output Format | Training Method |
|---|---|---|---|
| **Halawi et al. (2024)** — "Approaching Human-Level Forecasting" (NeurIPS) | Retrieval → Reasoning → Aggregation | **Single probability per question**, aggregated via trimmed mean across multiple runs | Supervised fine-tuning on 50k+ questions from forecasting platforms. No mixture model. |
| **Turtel et al. (2025)** — "Outcome-based RL to Predict the Future" | Simple prompt → prediction | **Single probability**, but trained with RL | GRPO + Brier score reward on a 14B model. 100k synthetic questions. No mixture model. Median prediction sampling at inference. |
| **Mantic (2026)** — "Training LLMs to Predict World Events" | Research agents → Prediction LLM with tool-use | **Mixture model via function calling** → CDF → probability | GRPO + Brier score on `gpt-oss-120b` via Tinker. 10k questions. **Uses tool calls to parameterize mixture.** |

### The Evolution

```
Halawi (2024): LLM outputs one number → average many runs
    ↓
Turtel (2025): RL-train LLM to output one better number
    ↓  
Mantic (2026): RL-train LLM to output a full distribution via tool calls
```

Each step adds more structure to the output format, giving the RL algorithm a richer signal to optimize against.

---

## What Mantic's Code Probably Looks Like

While their code isn't open-sourced, based on the Thinking Machines / Tinker infrastructure and standard practices, the prediction phase likely looks like:

```python
# Pseudo-code reconstruction of Mantic's prediction pipeline

from scipy.stats import lognorm, expon, norm

# 1. Define the tool schema for the LLM
FORECAST_TOOLS = [{
    "type": "function",
    "function": {
        "name": "add_mixture_component",
        "description": "Add a scenario to your temporal forecast. "
                       "Each component represents a distinct scenario "
                       "for when the event might occur.",
        "parameters": {
            "type": "object",
            "properties": {
                "scenario_name": {"type": "string"},
                "distribution": {"type": "string", "enum": ["lognormal", "exponential", "normal"]},
                "weight": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                "param1": {"type": "number", "description": "meanlog/rate/mean depending on dist"},
                "param2": {"type": "number", "description": "sdlog/None/std depending on dist"},
            },
            "required": ["scenario_name", "distribution", "weight", "param1"]
        }
    }
}]

# 2. Call the LLM with research context + tools
response = llm.chat(
    messages=[
        {"role": "system", "content": FORECASTING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research context:\n{research_summary}\n\n"
                                     f"Question: {question}\n"
                                     f"Resolution date: {resolution_date}"}
    ],
    tools=FORECAST_TOOLS,
    tool_choice="required"  # LLM MUST call the tool
)

# 3. Parse the tool calls into a mixture model
components = []
for tool_call in response.tool_calls:
    args = tool_call.function.arguments
    if args["distribution"] == "lognormal":
        dist = lognorm(s=args["param2"], scale=np.exp(args["param1"]))
    elif args["distribution"] == "exponential":
        dist = expon(scale=1/args["param1"])
    elif args["distribution"] == "normal":
        dist = norm(loc=args["param1"], scale=args["param2"])
    components.append({"weight": args["weight"], "dist": dist})

# 4. Normalize weights
total_weight = sum(c["weight"] for c in components)
for c in components:
    c["weight"] /= total_weight

# 5. Compute mixture CDF at resolution date
days_until_resolution = (resolution_date - datetime.now()).days
probability = sum(
    c["weight"] * c["dist"].cdf(days_until_resolution) 
    for c in components
)

# 6. During training: compute Brier score reward
actual_outcome = 1  # event happened (or 0 if it didn't)
brier_reward = 1 - (probability - actual_outcome) ** 2
```

---

## Relevance to Our Leader Prediction Project

We could adapt this exact architecture but swap the question format:

| Mantic's Format | Our Format |
|---|---|
| "Will [event] occur before [date]?" | "Will [leader] take [action] before [date]?" |
| Research agents gather general context | Research agents gather leader-specific context (speeches, VICS profile, policy history) |
| LLM reasons about world events | LLM reasons *as the leader persona* about their likely actions |
| Mixture components = different world scenarios | Mixture components = different decision scenarios the leader might pursue |

The temporal CDF format is especially useful for our action-timing questions: "When will Trump respond to the EU tariff announcement?" — the mixture model can express "60% chance within 48 hours via Truth Social, 30% chance within 2 weeks via formal policy, 10% chance he ignores it entirely."
