# CHRONOS — Temporal Knowledge Base for Leader Decision Prediction

> **Purpose**: Build a comprehensive, slidable temporal database of everything relevant to a world leader's decision-making — so that LLM personas can be grounded with *only* the information available up to a given date, enabling rigorous, leakage-free retroactive evaluation.  
> **First Subject**: Donald J. Trump  
> **Date**: April 16, 2026 (v2)

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [How Others Handle Data Leakage](#2-how-others-handle-data-leakage)
3. [Architecture: Temporally-Indexed Vector Store](#3-architecture-temporally-indexed-vector-store)
4. [Multi-Model Support & Variable Training Cutoffs](#4-multi-model-support--variable-training-cutoffs)
5. [Data Model](#5-data-model)
6. [Multi-Agent Research Swarm](#6-multi-agent-research-swarm)
7. [Temporal Integrity: Multi-Layer Date Validation](#7-temporal-integrity-multi-layer-date-validation)
8. [The Temporal Sliding Window](#8-the-temporal-sliding-window)
9. [LLM-Readable Output](#9-llm-readable-output)
10. [Efficiency](#10-efficiency)
11. [UI Design](#11-ui-design)
12. [Technology Stack](#12-technology-stack)
13. [Open Questions & Risks](#13-open-questions--risks)

---

## 1. The Problem

We have a separate question-generation pipeline (built on the Bosse et al. methodology, in another repository) that produces questions like:

- **Binary**: "Will Trump impose tariffs above 20% on Canadian goods before July 1, 2027?"
- **Action Selection**: "EU announces retaliatory tariffs. Which action does Trump take? A) Escalate B) Negotiate C) Threaten but delay D) Back down E) No action"

Each question arrives with a **simulation date** — the "fake today" — built into the question itself. The LLM persona is prompted as: *"Imagine you are Donald Trump. Today is [simulation_date]."* The persona must answer knowing only what was publicly known up to that date.

### What CHRONOS Is

CHRONOS is **the database**. That's it. It's a comprehensive, time-indexed knowledge base covering everything relevant to the leader's decision-making. When a question comes in (from the Bosse pipeline), the system filters this database down to only events visible at the question's simulation date, retrieves the most relevant records, and hands them to the LLM.

CHRONOS does NOT generate questions. It does NOT run predictions. It builds and serves the knowledge.

### Core Requirements

| Requirement | Description |
|:---|:---|
| **Zero Data Leakage** | The persona must have absolutely no access to information after the simulation date. This is the single most critical constraint. |
| **Time-Slidable** | Given any date, the system must instantly produce a snapshot of all knowledge available up to that date. Moving the date is changing one parameter. |
| **Comprehensive** | The database should be wide — covering ALL topics relevant to the leader's decision-making, not just one domain. We don't know what questions will be asked when we build this. |
| **Multi-Model Compatible** | Different LLMs have different training cutoff dates. The dataset must span wide enough that any model can be plugged in with its own cutoff without rebuilding. |
| **Autonomous Construction** | A multi-agent research swarm builds this database with zero human intervention. |
| **Perfectly Date-Tagged** | Every single record must have a verified, correct event date. A wrong date IS data leakage. |

---

## 2. How Others Handle Data Leakage

### ForecastBench (Zou et al., 2024)

**Approach**: Avoid the problem entirely — only ask about the future. Since the ground truth doesn't exist at prediction time, data leakage is impossible by construction.

**Why We Can't Use This**: We're evaluating retroactively — the events have already happened.

### Bosse et al. (2026) — The Paper Our Question Pipeline Is Based On

**Approach**: Retroactive evaluation with temporal discipline in the *question wording*:
1. Simulation dates on each question
2. Web research agents constrained to simulation date
3. Information Leakage Detector (adversarial agent checking if question wording reveals the outcome)
4. Cross-model verification

**Limitation**: The leakage prevention is in the question wording, not in the knowledge available to the model. The model itself still knows everything up to its training cutoff. Their approach trusts the LLM to "pretend" it doesn't know the future.

### The Gap CHRONOS Fills

None of these systems build **the actual knowledge infrastructure** that makes the simulation date real. CHRONOS does: the retrieval layer enforces a hard temporal cutoff at the database level. The model physically cannot receive information it shouldn't have.

---

## 3. Architecture: Temporally-Indexed Vector Store

### The Design

PostgreSQL stores every event as a timestamped record. pgvector stores embeddings of those records in the same database. When a question comes in with a simulation date, the retrieval query includes a hard SQL filter — `WHERE event_date <= simulation_date` — and the model only sees what passes that filter.

```
┌──────────────────────────────────────────────────────────────┐
│                    CHRONOS ARCHITECTURE                       │
│                                                              │
│  ┌───────────────┐     ┌──────────────────────────────────┐  │
│  │  Multi-Agent   │────▶│  PostgreSQL + pgvector            │  │
│  │  Research       │     │                                  │  │
│  │  Swarm          │     │  event_records table             │  │
│  │  (builds the    │     │  ├── structured fields (JSON)    │  │
│  │   database)     │     │  ├── event_date (temporal key)   │  │
│  │                 │     │  └── embedding (vector)          │  │
│  └───────────────┘     └───────────┬──────────────────────┘  │
│                                     │                        │
│           ┌─────────────────────────┘                        │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────┐            │
│  │  Retrieval Layer (called at question time)    │            │
│  │                                               │            │
│  │  1. Receive question + simulation_date        │            │
│  │  2. WHERE event_date <= simulation_date       │            │
│  │     AND event_date >= model_training_cutoff   │            │
│  │  3. Vector similarity search on question      │            │
│  │  4. Rerank → top K results                    │            │
│  │  5. Format as LLM-readable briefing           │            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### Why This Works

| Requirement | How |
|:---|:---|
| **Zero Leakage** | Hard SQL filter. Documents after the cutoff don't appear in results. Period. |
| **Time-Slidable** | Sliding the date = changing one WHERE clause parameter. No re-indexing. |
| **Comprehensive** | The research swarm casts a wide net. All domains, all topics. |
| **Multi-Model** | The lower bound (`model_training_cutoff`) is a parameter passed at query time. Different model = different lower bound. Same database. |
| **Autonomous** | The research swarm handles everything. |

---

## 4. Multi-Model Support & Variable Training Cutoffs

This is a critical design consideration. We will test multiple LLMs, each with a different training cutoff:

| Model | Approximate Training Cutoff |
|:---|:---|
| GPT-4o | Oct 2023 |
| GPT-4.1 | Jun 2024 |
| Claude 3.5 Sonnet | Apr 2024 |
| Claude 4 Sonnet | Early 2025 |
| Gemini 2.5 Pro | Jan 2025 |
| Gemini 3 Pro | Mid 2025 (estimated) |

### Implications for the Database

The knowledge window for each model is: `[model_training_cutoff, simulation_date]`

- **GPT-4o** tested on a question with simulation_date = July 2025 → needs events from **Oct 2023 to Jul 2025** (21 months of context)
- **Gemini 3 Pro** tested on the same question → needs events from **Mid 2025 to Jul 2025** (1 month of context)

The database must therefore contain events going back **as far as the earliest model's training cutoff**. This means comprehensive coverage starting from approximately **October 2023** through the present.

### How This Works at Query Time

```sql
-- For GPT-4o (training cutoff: Oct 2023)
SELECT * FROM event_records
WHERE event_date <= '2025-07-01'        -- simulation date (from question)
  AND event_date >= '2023-10-01'        -- GPT-4o's training cutoff
ORDER BY similarity(embedding, query_vec) DESC;

-- For Gemini 3 Pro (training cutoff: mid 2025)
SELECT * FROM event_records
WHERE event_date <= '2025-07-01'        -- same simulation date
  AND event_date >= '2025-06-01'        -- Gemini 3's training cutoff
ORDER BY similarity(embedding, query_vec) DESC;
```

Same database. Same question. Different lower bound. The model's training cutoff is passed as a parameter at query time — it's not baked into the data.

### Practical Impact

The dataset needs to be **~2.5 years wide** (Oct 2023 → present) to accommodate the oldest model. This is a broader collection effort than if we only supported one model, but it's a one-time cost. The research swarm collects everything; the sliding window handles the rest.

---

## 5. Data Model

### The Event Record

Every piece of information in the knowledge base is an **Event Record** — a timestamped, bias-stripped document representing a single thing that happened on a single date:

```json
{
  "record_id": "EVT-2025-03-15-001",

  "event_date": "2025-03-15",
  "event_date_precision": "day",
  "date_confidence": "verified",
  "date_verification_method": "cross_referenced_3_sources",
  "ingestion_date": "2026-04-16",

  "headline": "Trump signs executive order imposing 25% tariff on Canadian steel",
  "summary": "President Trump signed Executive Order 14XXX imposing a 25% tariff on all Canadian steel imports, effective April 1, 2025. The order cites Section 232 national security authority. Canadian PM responded with retaliatory tariff threats within 24 hours.",
  "key_facts": [
    "Tariff rate: 25% on all Canadian steel imports",
    "Legal authority: Section 232",
    "Effective date: April 1, 2025",
    "Canadian PM announced retaliatory measures within 24 hours"
  ],
  "direct_quotes": [
    {
      "speaker": "Donald Trump",
      "quote": "Canada has been taking advantage of us for years...",
      "context": "Signing ceremony remarks"
    }
  ],

  "topics": ["canada", "steel", "tariffs", "section_232", "trade"],
  "actors": ["donald_trump", "justin_trudeau"],

  "sources": [
    {"name": "Federal Register", "url": "...", "type": "primary", "pub_date": "2025-03-15"},
    {"name": "Reuters", "url": "...", "type": "wire_service", "pub_date": "2025-03-15"},
    {"name": "AP", "url": "...", "type": "wire_service", "pub_date": "2025-03-15"}
  ],
  "source_count": 3,
  "confidence": 0.95
}
```

### Key Design Decisions

- **`event_date` is THE critical field.** Everything depends on it being correct. See Section 7 for the multi-layer validation system.
- **`date_confidence`** tracks whether the date has been verified (`verified`, `high`, `approximate`, `uncertain`). Records marked `uncertain` get flagged for additional review.
- **No domain taxonomy.** We don't pre-categorize into rigid "decision domains." The `topics` array is freeform — the research swarm tags whatever is relevant. The vector search handles the rest.
- **No pre-computed summaries spanning time ranges.** Every record is a single point-in-time event. Summaries that span ranges would bake in temporal information and compromise the sliding window.
- **Every record is LLM-readable.** The `headline`, `summary`, `key_facts`, and `direct_quotes` are written in natural language that an LLM can directly consume. No parsing required.

### What We Don't Store

- No editorial opinion or analysis (only factual actions, statements, outcomes)
- No VICS operational code scores (out of scope)
- No inferred reasoning or psychological profiles (the LLM persona does its own reasoning)

---

## 6. Multi-Agent Research Swarm

This is designed as a fully autonomous deep-research system. You give it a name; it produces a comprehensive, time-indexed database of everything relevant to that person's decision-making.

### 6.1 Swarm Architecture

The swarm has **6 agents** organized in a supervisor-worker pattern with oversight layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH SWARM                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  COORDINATOR (Supervisor Agent)                                  │   │
│  │                                                                  │   │
│  │  • Maintains the global research plan                            │   │
│  │  • Tracks coverage across topics and time periods                │   │
│  │  • Identifies gaps and dispatches targeted follow-up research    │   │
│  │  • Decides when a topic area is "sufficiently covered"           │   │
│  │  • Resolves conflicts between agents                             │   │
│  └──────────┬───────────────────────────────┬───────────────────────┘   │
│             │                               │                           │
│             ▼                               ▼                           │
│  ┌────────────────────┐        ┌────────────────────┐                  │
│  │  DISCOVERY AGENT    │        │  DISCOVERY AGENT    │   ← multiple    │
│  │  (Topic Researcher) │        │  (Topic Researcher) │     concurrent  │
│  │                     │        │                     │     workers     │
│  │  • Deep web search  │        │  • Deep web search  │                  │
│  │  • Multi-step ReAct │        │  • Multi-step ReAct │                  │
│  │  • Source chaining  │        │  • Source chaining  │                  │
│  └────────┬───────────┘        └────────┬───────────┘                  │
│           │                              │                              │
│           └──────────┬───────────────────┘                              │
│                      ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  EXTRACTION AGENT                                                │   │
│  │                                                                  │   │
│  │  • Scrapes URLs from Discovery                                   │   │
│  │  • Parses article content (Playwright + newspaper3k)             │   │
│  │  • Extracts: headline, full text, pub date, quotes               │   │
│  │  • Determines EVENT DATE (when it happened, not when published)  │   │
│  │  • Rejects stubs, paywalls, 404s, content < 100 words            │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  CLEANING AGENT                                                  │   │
│  │                                                                  │   │
│  │  • Bias-strips content (RED CHAMBER methodology)                 │   │
│  │  • Writes LLM-readable summaries (100-200 words)                 │   │
│  │  • Extracts key facts, direct quotes                             │   │
│  │  • Tags topics and actors (freeform, no rigid taxonomy)          │   │
│  │  • Generates the structured Event Record                         │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  TEMPORAL VALIDATOR (Critical Oversight Agent)                    │   │
│  │                                                                  │   │
│  │  • DEDICATED agent whose only job is verifying dates             │   │
│  │  • Cross-references event_date against multiple sources          │   │
│  │  • Checks: does the pub_date match the event_date?               │   │
│  │  • Checks: do all sources agree on when this happened?           │   │
│  │  • Flags ambiguous dates ("last month", "recently", "this week") │   │
│  │  • Assigns date_confidence score                                 │   │
│  │  • REJECTS records where date cannot be verified                 │   │
│  │  • See Section 7 for full validation methodology                 │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  DEDUP & MERGE                                                   │   │
│  │                                                                  │   │
│  │  • Pairwise similarity scoring (LLM-based)                       │   │
│  │  • Events scoring > 0.8 similarity → merged                      │   │
│  │  • Keep best summary, combine source lists, keep all quotes      │   │
│  │  • Merged records get higher confidence (more sources)           │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  INDEXING AGENT                                                  │   │
│  │                                                                  │   │
│  │  • Stores Event Record in PostgreSQL                             │   │
│  │  • Generates vector embedding of summary                        │   │
│  │  • Stores embedding in pgvector                                  │   │
│  │  • Indexes event_date, topics, actors as filterable metadata     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  COVERAGE AUDITOR (runs after each collection cycle)             │   │
│  │                                                                  │   │
│  │  • Computes events-per-month across the full time range          │   │
│  │  • Flags suspiciously sparse months (< threshold)                │   │
│  │  • Checks topic distribution — are we missing entire areas?      │   │
│  │  • Generates a "gap report" for the Coordinator                  │   │
│  │  • Coordinator dispatches targeted research for gaps             │   │
│  │  • Loop continues until coverage meets threshold                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Agent Details

#### The Coordinator

The Coordinator is the brain. It doesn't do research itself — it plans, delegates, and audits.

**At startup**, the Coordinator:
1. Takes the leader's name (e.g., "Donald Trump") and the collection window (e.g., Oct 2023 → Apr 2026)
2. Generates a broad research plan: what are the major areas of this person's decision-making? What are the key relationships, ongoing conflicts, policy areas?
3. Breaks the time window into manageable chunks (e.g., quarter by quarter)
4. Dispatches Discovery Agents to each chunk/topic combination

**During collection**, the Coordinator:
- Monitors the event count per time period and per topic
- If a Discovery Agent returns too few results for a period, dispatches a follow-up with more specific search queries
- If two Discovery Agents seem to be finding the same events, redirects one to a different angle
- Tracks the overall progress and decides when collection is "complete enough"

**The Coordinator is what prevents the simple 4-agent pipeline from producing a shallow, redundant database.** Without it, you'd get 50 articles about the same tariff announcement and zero coverage of personnel changes.

#### Discovery Agents (Workers)

These are ReAct-style web research agents. Each one:

1. Receives a research directive from the Coordinator (e.g., "Find all significant Trump trade policy actions from Q1 2025")
2. Generates search queries iteratively — starting broad, then following leads
3. Follows citation chains: Article A mentions "the March executive order" → searches for the actual EO
4. Cross-references across source types: news → official government records → press briefings
5. Returns a list of raw event candidates with URLs and preliminary timestamps

**Supplementary data sources** (non-LLM, pure API calls for guaranteed coverage):
- **GDELT API** — 15-minute update frequency, global event database
- **Federal Register API** — executive orders, proclamations, rules
- **Congress.gov API** — bills, resolutions
- **ACLED API** — human-coded political events
- **White House press briefings** — official statements

These structured sources provide a **coverage floor** — even if the ReAct agents miss something, the APIs catch it.

#### The Temporal Validator

This is the agent that makes or breaks the entire system. **A wrong date IS data leakage.** See Section 7 for its full methodology.

#### The Coverage Auditor

Runs after each collection cycle and produces a gap report:

```
COVERAGE REPORT — Trump, Oct 2023 → Apr 2026
─────────────────────────────────────────────
Total events: 4,823

Monthly distribution:
  Oct 2023: 142  ✓
  Nov 2023: 156  ✓
  Dec 2023: 98   ✓
  ...
  Feb 2025: 23   ⚠ SPARSE — below threshold (50)
  ...

Topic distribution:
  Trade/tariffs:     1,247 events  ✓
  Foreign policy:     892 events  ✓
  Personnel:          634 events  ✓
  Executive orders:   445 events  ✓
  Legislative:        312 events  ✓
  Legal/judicial:     287 events  ✓
  Domestic policy:    256 events  ✓
  Military/defense:   198 events  ✓
  Immigration:        178 events  ✓
  Economic policy:    145 events  ✓
  Other:              229 events  ✓

GAP DETECTED: Feb 2025 only 23 events.
RECOMMENDATION: Dispatch targeted research for Feb 2025.
```

The Coordinator reads this report and dispatches additional Discovery Agents to fill gaps. This loop continues until the Coordinator is satisfied with coverage.

---

## 7. Temporal Integrity: Multi-Layer Date Validation

**This is the most important section of this document.** If any event has the wrong date, the sliding window becomes unreliable and every downstream experiment is compromised.

### The Problem

Dates are harder than they look:
- An article **published** March 16 might **describe** an event that happened March 14
- A government document might have an **effective date** of April 1 but was **signed** March 15
- Wire services sometimes say "last week" or "earlier this month"
- A tweet thread might span multiple hours across a date boundary
- Different time zones create ambiguity for same-day events

### The Multi-Layer Validation System

Every event date goes through **4 independent checks** before it enters the database:

#### Layer 1: Extraction-Time Date Parsing

The Extraction Agent determines the initial `event_date` by:
- Parsing the article's explicit date references ("On March 15, Trump signed...")
- Distinguishing between publication date and event date
- Looking for official timestamps (executive order signing dates, press briefing dates)
- Flagging any relative dates ("last week", "recently") for further investigation

#### Layer 2: Cross-Source Verification (Temporal Validator)

The Temporal Validator agent:
1. Takes the proposed `event_date` and the source URLs
2. Searches for the same event in at least 2 additional sources
3. Compares the dates across all sources
4. If all sources agree → `date_confidence: "verified"`
5. If sources disagree by 1 day → `date_confidence: "high"`, picks the most authoritative source (government record > wire service > news article)
6. If sources disagree by >1 day → `date_confidence: "uncertain"`, flags for review
7. If only 1 source exists → `date_confidence: "single_source"`, acceptable but noted

#### Layer 3: Logical Consistency Checks

Automated checks that catch obvious errors:
- Event date cannot be in the future
- Event date cannot be before the leader took office (for official actions)
- If event B is described as a "response to event A", event B's date must be >= event A's date
- Publication date should be >= event date (you can't publish about something before it happens)
- If the event date is suspiciously far from the publication date (>30 days), flag for review

#### Layer 4: Statistical Outlier Detection

After a batch of events is collected:
- Flag any events whose dates cluster suspiciously (e.g., 50 events on one day, 0 on surrounding days — might indicate a date parsing bug)
- Flag any events that are chronological outliers within their topic (e.g., a "tariff negotiation update" dated 6 months before any other tariff events)

### Date Confidence Classification

| Level | Meaning | Action |
|:---|:---|:---|
| `verified` | 3+ sources agree on the date | ✅ Include in database |
| `high` | 2 sources agree, or authoritative primary source | ✅ Include in database |
| `single_source` | Only 1 source, but date is explicit and logical | ✅ Include, note the limitation |
| `approximate` | Date is known to week/month but not exact day | ✅ Include, set `event_date_precision: "week"` or `"month"` |
| `uncertain` | Sources disagree or date is ambiguous | ⚠️ Quarantine — do not include until resolved |

**Records with `uncertain` dates are quarantined**, not included in the main database. They sit in a separate review queue until a human or follow-up research resolves the ambiguity. Better to miss an event than to include it with a wrong date.

---

## 8. The Temporal Sliding Window

### The Mechanism

Every retrieval query includes a hard temporal filter:

```sql
SELECT record_id, headline, summary, key_facts, direct_quotes, event_date
FROM event_records
WHERE event_date <= $simulation_date                    -- ← THE UPPER BOUND (from the question)
  AND event_date >= $model_training_cutoff              -- ← THE LOWER BOUND (from the model config)
  AND embedding <=> $query_embedding < $similarity_threshold
ORDER BY similarity DESC, event_date DESC
LIMIT $top_k;
```

Moving the simulation date is changing `$simulation_date`. Switching models is changing `$model_training_cutoff`. Same database, different windows.

### Why This Is Leakage-Proof

> **No document with `event_date > simulation_date` can ever appear in the retrieval results.**

This is enforced at the database level. The model doesn't need to "pretend" it doesn't know something — it literally doesn't receive the information. The retrieval layer is the firewall.

### The Three Knowledge Zones

For any given model + question combination:

```
│ ZONE 1: Before training cutoff │ ZONE 2: Cutoff → Sim Date │ ZONE 3: After sim date │
│ Model knows from training      │ Model knows ONLY from     │ Model knows NOTHING     │
│ (no retrieval needed)           │ CHRONOS retrieval          │ (guaranteed)            │
├─────────────────────────────────┼───────────────────────────┼─────────────────────────┤
│          Oct 2023               │        ...                │        Jul 2025 →       │
│     (GPT-4o cutoff)             │   CHRONOS provides this   │     INVISIBLE           │
```

Zone 2 is where CHRONOS operates. It's the gap between what the model learned in training and what it needs to know to answer the question. Different models have different-sized Zone 2s.

### Comparison to Other Approaches

| Approach | Temporal Sliding | Leakage Risk |
|:---|:---|:---|
| "Pretend today is X" (prompt only) | Free | **HIGH** — model's parametric knowledge isn't erased by instructions |
| Bosse et al. Agent 6 | Free | **MEDIUM** — catches leakage in question wording but not in model knowledge |
| Rebuild the graph per date (GraphRAG) | Extremely expensive | **LOW** — but computationally infeasible at scale |
| **CHRONOS (metadata filter)** | **Free — one SQL parameter** | **NONE** — information after cutoff physically cannot enter the context |

---

## 9. LLM-Readable Output

The entire point of this database is to be read by an LLM. The output format must be immediately consumable — no parsing, no conversion, no SQL results tables.

### The Briefing Format

When the retrieval layer returns results, they're formatted as a **natural-language intelligence briefing**:

```
═══════════════════════════════════════════════════════════
 INTELLIGENCE BRIEFING — As of July 1, 2025
 Subject: Donald J. Trump, 47th President of the United States
 Knowledge Window: January 2025 → July 1, 2025
═══════════════════════════════════════════════════════════

[2025-06-28] US-Canada Trade Tensions Escalate
Trump threatened to increase tariffs on Canadian lumber to 40% during
a press conference at the White House. "Canada needs to understand that
we're serious about protecting American jobs," Trump stated. Key facts:
current tariff on Canadian lumber is 25% (imposed March 2025). Canada's
PM called the threat "economic aggression" and hinted at retaliatory
measures targeting US agricultural exports.
Sources: Reuters, AP, White House Press Pool (3 sources, verified date)

─────────────────────────────────────────────────────────

[2025-06-20] Executive Order on Trade Review Commission
Trump signed EO 14XXX establishing a "Fair Trade Review Commission"
tasked with evaluating all bilateral trade agreements. Commission has
180 days to report. Chair: Robert Lighthizer. Key facts: covers all
trade agreements, not just USMCA. Bipartisan commission structure
(2 Dem, 2 GOP, 1 independent appointee).
Sources: Federal Register, Reuters (2 sources, verified date)

─────────────────────────────────────────────────────────

[... additional records, ordered by relevance ...]

═══════════════════════════════════════════════════════════
 END BRIEFING — 12 events retrieved
═══════════════════════════════════════════════════════════
```

### Design Principles for LLM Readability

1. **Natural language, not structured data.** The summary and key facts are written as prose/bullets, not database rows. An LLM reads English, not SQL result sets.
2. **Date on every record.** The event date is prominently displayed so the LLM knows the temporal ordering.
3. **Source attribution inline.** The LLM can assess credibility: "3 sources, verified date" vs "1 source, approximate date."
4. **Direct quotes preserved.** The leader's actual words are critical for the persona to "think like" the leader.
5. **Header includes the knowledge window.** The LLM knows exactly what time period it's seeing.

---

## 10. Efficiency

### Storage

| Component | Estimated Size (per leader, ~2.5 years) |
|:---|:---|
| Event Records (PostgreSQL) | ~8,000 records × ~2KB = ~16 MB |
| Vector embeddings (pgvector) | ~8,000 × 768 dims × 4 bytes = ~24 MB |
| Full text archive | ~8,000 × ~5KB = ~40 MB |
| **Total** | **~80 MB per leader** |

### Query Latency

| Operation | Latency |
|:---|:---|
| Vector search + metadata filter | ~10–50ms |
| Cross-encoder reranking (15 candidates) | ~100–200ms |
| Briefing formatting | ~5ms |
| **Total retrieval** | **~150–300ms** |

### Token Efficiency (Per Prediction)

| Component | Tokens |
|:---|:---|
| Context (12 Event Records × ~200 words) | ~3,600 tokens |
| System prompt (persona + instructions) | ~500 tokens |
| Question | ~100 tokens |
| **Total input** | **~4,200 tokens** |

For 1,000 predictions: ~4.2M input tokens → **~$1 total** at Gemini 3 Pro pricing.

### Collection Pipeline Cost

| Stage | Cost |
|:---|:---|
| Discovery Agents (LLM + web search) | ~$8–15 per leader |
| Extraction (scraping, minimal LLM) | ~$1 |
| Cleaning Agent (LLM for summaries) | ~$3–5 |
| Temporal Validator (LLM for cross-referencing) | ~$2–4 |
| Coverage Auditor + gap-filling cycles | ~$3–5 |
| Embedding generation | ~$0.10 |
| **Total per leader** | **~$17–30** |

Higher than v1 estimate because the research swarm is deeper — more validation, more coverage checking, more follow-up research. Worth it.

---

## 11. UI Design

### 11.1 The Timeline Explorer

```
┌─────────────────────────────────────────────────────────────┐
│  CHRONOS — Temporal Knowledge Base                          │
│                                                             │
│  Subject: Donald J. Trump       Events: 8,234              │
│                                                             │
│  Model: [GPT-4o ▼]  Cutoff: Oct 2023                       │
│                                                             │
│  Model Cutoff          Simulation Date             Today    │
│  Oct 2023                    ◆                    Apr 2026  │
│  ├──────────────────────────●───────────────────────┤      │
│                          Jun 15, 2025                       │
│                                                             │
│  ┌─ Visible Knowledge (5,892 events) ──────────────────┐   │
│  │                                                      │   │
│  │  ● Jun 14 — Trump threatens 50% tariff on EU autos  │   │
│  │  ● Jun 12 — Senate passes defense spending bill      │   │
│  │  ● Jun 10 — Trump-Zelenskyy phone call readout       │   │
│  │  ● Jun 8  — USTR announces Section 301 review...    │   │
│  │  ● Jun 5  — Trump fires Deputy AG Sarah Chen         │   │
│  │  [Show more...]                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ After Simulation Date (2,342 events) ──────────────┐   │
│  │  🔒 REDACTED — invisible to model                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Topic Distribution (visible window):                       │
│  Trade ████████████░░ 24%    Foreign Pol █████████░ 18%    │
│  Personnel ███████░░ 13%     Exec Orders ██████░░ 11%     │
│  Legislative █████░░ 9%      Legal ████░░░ 8%              │
│                                                             │
│  Model Selector: switching to "Claude 4" moves the left    │
│  boundary → more/fewer events visible depending on cutoff  │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Features

**A. Timeline Slider** — drag to any date. Event counts, topic distribution, everything updates instantly.

**B. Model Selector** — dropdown selects the LLM under test. Changes the left boundary (training cutoff) of the visible window. Immediately shows how many events that model would "see."

**C. Event Inspector** — click any event → full record view with summary, key facts, quotes, all sources, date confidence level, date verification method.

**D. Question Tester** — paste a question from the Bosse pipeline. CHRONOS retrieves the relevant records, formats the briefing, and shows exactly what the model would receive.

**E. Collection Pipeline Monitor** — real-time dashboard of the research swarm. Events discovered / extracted / validated / indexed. Error log. Coverage heatmap.

**F. Date Audit View** — shows all events by date_confidence level. Filter to see only `approximate` or `single_source` records. Drill into the quarantine queue for `uncertain` records.

### 11.3 Tech Stack

- **Framework**: Next.js (React)
- **Styling**: Dark mode, data-dense, monospace data fields
- **Charts**: Recharts for distributions and coverage heatmaps
- **Timeline**: Custom React component with drag interaction
- **Backend**: FastAPI (Python) — shares code with the research swarm

---

## 12. Technology Stack

| Component | Technology | Why |
|:---|:---|:---|
| **Orchestration** | LangGraph (Python) | Stateful multi-agent workflows, cyclic graphs for research loops |
| **Database** | PostgreSQL + pgvector | Single DB for structured records AND vector search. Temporal filtering via SQL WHERE clause. |
| **Embedding Model** | `text-embedding-004` (Google) or `nomic-embed-text` | High quality, cheap |
| **Research LLM** | Gemini 3 Flash | Fast, cheap — good for extraction/summarization |
| **Validation LLM** | Gemini 3 Pro | Higher quality — used for temporal validation and coverage auditing |
| **Web Scraping** | Playwright + newspaper3k | Playwright for dynamic pages, newspaper3k for article parsing |
| **Web Search** | Google Search API or Serper | ReAct agent tool |
| **Frontend** | Next.js + React | Dashboard and timeline explorer |
| **Backend API** | FastAPI | Serves retrieval queries |
| **Package Management** | uv | Fast Python deps |

---

## 13. Open Questions & Risks

### Questions

1. **How many events to retrieve per question?** 10? 15? 20? Should we run ablations on retrieval count?

2. **Recency weighting**: Events closer to the simulation date are probably more relevant. How aggressively do we weight them? Should weighting be topic-dependent? (A grudge from 6 months ago might matter more than yesterday's trade announcement.)

3. **"No retrieval" baseline**: Should one experiment condition be the LLM persona with zero CHRONOS context? This tells us whether retrieval adds value over the model's parametric knowledge alone.

4. **Collection depth vs. breadth**: With the multi-model support, we're covering ~2.5 years. Should we prioritize depth (more events per month in recent periods) or uniform coverage?

5. **Embedding model selection**: Should we evaluate multiple embedding models on a validation set of question-to-event matches before committing?

### Risks

| Risk | Severity | Mitigation |
|:---|:---|:---|
| **Event date errors** | **CRITICAL** — wrong date = data leakage | 4-layer validation system (Section 7). Quarantine uncertain dates. |
| **Incomplete coverage** | HIGH — missing events = missing context | Coverage Auditor + gap-filling loop. Structured API sources as floor. |
| **Scraping failures** | MEDIUM — paywalls, rate limits, dynamic content | Fallback to GDELT/API sources. Retry with Playwright. Graceful degradation. |
| **Embedding quality** | MEDIUM — poor embeddings = irrelevant retrieval | Test on validation set before committing. |
| **Model ignoring context** | LOW–MEDIUM — LLM might rely on parametric knowledge | Post-cutoff questions ensure no parametric knowledge of events. Strong system prompt. |
| **Cost overrun on deep research** | LOW — more agents = more API calls | Budget caps per collection cycle. Monitor token usage. |

---

## Summary

**CHRONOS is a temporal database.** You give it a name, it builds a comprehensive, time-indexed knowledge base of everything relevant to that person's decisions. When a question arrives with a simulation date, CHRONOS filters its database to only events before that date, retrieves the most relevant records, and formats them as an LLM-readable intelligence briefing.

The key design choices:

- **Metadata-filtered vector search** — because temporal sliding is a WHERE clause, not a graph rebuild
- **PostgreSQL + pgvector** — one database for structure and search
- **Point-in-time Event Records** — no pre-baked summaries that leak temporal info
- **4-layer date validation** — because a wrong date defeats the entire system
- **Multi-model support** — different LLMs, different cutoffs, same database
- **Autonomous research swarm** with coordinator, coverage auditing, and gap-filling — because a shallow database is as useless as no database
- **LLM-readable output** — because the consumer is a language model, not a human staring at SQL results
