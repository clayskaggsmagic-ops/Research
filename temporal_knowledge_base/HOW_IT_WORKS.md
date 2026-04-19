# CHRONOS — How It Works (Plain English)

## The Problem

You want to ask an AI: *"What would Trump do if China invaded Taiwan on March 15, 2025?"*

The AI has a problem: it was trained on data up to some point in time (say, October 2023). It doesn't know what happened between October 2023 and March 2025. It's missing ~18 months of context — tariffs, cabinet changes, diplomatic meetings, executive orders — all the stuff that would shape how Trump would actually respond.

**CHRONOS fills that gap.** It builds a "memory" of everything that happened in that missing window, so you can hand it to the AI and say: *"Here's what you missed. Now answer the question."*

---

## Why "Temporal"?

"Temporal" just means "related to time." The whole system is obsessed with one idea:

> **An AI should only know what it *would* have known on a given date.**

If you're simulating Trump making a decision on March 15, 2025, the AI should know about things that happened *before* March 15 — but **never** things that happened *after*. That would be like giving someone tomorrow's newspaper. It would ruin the simulation.

---

## The 7 Workers

Think of CHRONOS like a newsroom with 7 specialized employees working together.

### 1. 🧠 Coordinator (Editor-in-Chief)
The boss. Doesn't do any research himself. He:
- Looks at the date range ("October 2023 through April 2025")
- Breaks it into chunks ("First cover October 2023, then November...")
- Creates a research plan ("For each month, search for executive orders, foreign policy, economic news...")
- Hands out assignments to the Discovery Agent
- When the Coverage Auditor says "you're missing foreign policy in February 2025," the Coordinator plans follow-up research

### 2. 🔍 Discovery Agent (Reporter)
Goes out and *finds* stuff. Given a topic + time range:
- Searches Google News, GDELT (a giant global news database), and news APIs
- Finds URLs to relevant articles
- Doesn't read the articles — just finds them and passes the list along
- Like a reporter who makes phone calls: "I found 30 articles about Trump tariffs in January 2025"

### 3. 📄 Extraction Agent (Fact Checker / Summarizer)
Takes the raw articles from Discovery and actually *reads* them. For each one:
- Pulls out the **headline**, **date**, **key facts**, **direct quotes**, and **who was involved**
- Tags it with topics (economic, foreign policy, legal, etc.)
- Creates a structured "event card" — like an index card with all the essential info

### 4. 🧹 Cleaning Agent (Copy Editor)
Quality control. Takes extracted events and:
- **Removes duplicates** — 5 articles about the same tariff announcement? Merged into one event with higher confidence
- **Strips bias** — Rewrites summaries to sound like AP/Reuters wire copy (neutral, factual, no opinions)
- **Normalizes names** — "Donald Trump," "President Trump," and "Trump" all map to the same person
- **Scores confidence** — Many sources = high confidence. Single-source = lower confidence

### 5. 🛡️ Temporal Validator (Date Police)
The most paranoid employee. Only job: **make sure every date is correct.**

Why? Because the whole system depends on dates. If an event from April 2025 accidentally gets labeled as March 2025, it would appear in simulations where it shouldn't. That's "data leakage" — the cardinal sin of this system.

It runs 4 checks:
1. **Mechanical** — Is the date valid? In the future? Within our window?
2. **Cross-source** — Do multiple articles agree on the date? If 3 say "January 15" and 1 says "January 20," something's off
3. **Logic** — Uses AI to sanity-check ("This says Trump signed an executive order on Jan 5, but he wasn't inaugurated until Jan 20 — impossible")
4. **Statistical** — If 50 events cluster in January and 1 random event sits in August alone, that's suspicious

If an event fails? **Quarantined.** Stored but NEVER shown in results. Better to miss something than include wrong data.

### 6. 📦 Indexing Agent (Librarian)
Stores clean, validated events permanently:
- Converts each event into a **vector embedding** — a mathematical fingerprint of its meaning (so you can later search by meaning, not just keywords)
- Puts everything in the database with all its metadata
- Checks for duplicates before inserting

### 7. 📊 Coverage Auditor (Quality Inspector)
Steps back after indexing and asks: "Did we do a good enough job?"

Checks:
- **Monthly coverage** — Every month needs at least 15 events. If February only has 3, that's a gap
- **Topic coverage** — Every month should have events in all 6 core categories (executive actions, foreign policy, economic, personnel, legislative, legal)
- **Recency bias** — Are we accidentally collecting way more recent events than older ones?
- **Quarantine rate** — If >15% of events are quarantined, our sources might be bad

If gaps exist, it tells the Coordinator: *"February has 5 events and zero foreign policy — do more research."* The cycle restarts.

---

## The Loop

The pipeline runs autonomously in a loop:

```
Coordinator → Discovery → Extraction → Cleaning → Validation → Indexing → Auditor
     ↑                                                                        |
     └──────────────── "Gaps found, do more research!" ───────────────────────┘
```

It loops up to 5 times. Each pass finds fewer gaps. Eventually:
- ✅ All criteria met → **Done!**
- ⚠️ 5 loops exhausted → **Stop anyway** (prevents running forever)

---

## How Retrieval Works (The Payoff)

Once the database is full, you query it:

> *"Tell me about Trump's trade policy"*
> **Simulation date:** March 15, 2025
> **Model:** GPT-4o (training cutoff: October 2023)

Two things happen:
1. **Time filter** — Only events between October 2023 and March 15, 2025 (the "knowledge gap")
2. **Meaning search** — Among those, find the ones most relevant to "trade policy" using vector similarity

Output: an **intelligence briefing** — a formatted document an AI reads before answering your question. Like handing someone a cheat sheet.

---

## The Critical Rule

> **Events after the simulation date can NEVER appear in results. Period.**

Enforced at the database level (SQL `WHERE` clause). There's no way to accidentally bypass it.

---

## Why This Matters

**Without CHRONOS:** AI simulating Trump in February 2025 is guessing from October 2023 knowledge. Doesn't know about tariffs imposed, cabinet members fired, or executive orders signed.

**With CHRONOS:** AI gets a complete, verified, bias-stripped briefing of everything in that window. Like giving the AI 18 months of reading the newspaper — compressed into one clean document.

---

## Running It

```bash
uv run python -m src.pipeline --subject "Donald J. Trump" --start 2023-10-01 --end 2025-04-01
```

---

## Deep Dive: What's Actually In The Database?

This is where people get confused, so let's go slow.

### It's a normal SQL table — with one extra column

Each event is one row in a regular PostgreSQL table called `event_records`. Here's what a single row looks like, in plain terms:

| Column | Example | What it is |
|--------|---------|------------|
| `id` | 42 | Auto-incrementing row number |
| `record_id` | `evt_2024_01_15_iowa` | Unique string ID we generate |
| `event_date` | `2024-01-15` | **A normal SQL date.** Stored as a date, compared as a date. Nothing fancy. |
| `date_confidence` | `verified` | Did we trust this date? ("verified", "approximate", or "uncertain") |
| `headline` | "Trump wins Iowa caucuses" | Plain text |
| `summary` | "Former President Trump won the Iowa Republican caucuses with over 50%..." | Plain text, 100-200 words |
| `key_facts` | `["Won with 51%", "30 points ahead of DeSantis"]` | A list of strings |
| `direct_quotes` | `[{"quote": "This is a big victory", "speaker": "Donald Trump"}]` | Structured JSON |
| `topics` | `["domestic_politics", "elections"]` | A list of category tags |
| `actors` | `["Donald Trump", "Ron DeSantis"]` | A list of people involved |
| `sources` | `[{"name": "AP News", "url": "https://..."}, {"name": "Reuters"}]` | Where the info came from |
| `source_count` | 3 | How many sources confirmed this |
| `confidence` | 0.92 | Overall trust score (0 to 1) |
| `embedding` | `[0.023, -0.418, 0.112, ...]` (768 numbers) | **The vector. This is the "meaning fingerprint."** |

So 16 of the 17 columns are completely normal SQL data — text, dates, numbers, lists. The 17th column (`embedding`) is a list of 768 decimal numbers that captures the *meaning* of the event's summary.

### How the embedding works (no math, promise)

When the Indexing Agent stores an event, it takes the summary text and sends it to Google's embedding model. That model reads the text and returns 768 numbers. These numbers are like GPS coordinates, but instead of telling you *where* something is on Earth, they tell you *where* it sits in "meaning space."

Events about similar topics end up with similar numbers. So:
- "Trump imposes 25% tariffs on Chinese steel" → some set of 768 numbers
- "New trade barriers on Chinese manufacturing" → a *very similar* set of 768 numbers
- "Supreme Court hears abortion case" → a *very different* set of 768 numbers

That's all an embedding is: a way to turn text into numbers so you can mathematically compare how similar two pieces of text are.

### How retrieval works — SQL and vectors working together

This is the key insight: **dates are filtered with normal SQL. Relevance is ranked with vectors. They work together in one query.**

Here's what happens when you ask: "What's Trump's trade policy?" with a simulation date of March 15, 2025, using GPT-4o (cutoff: October 1, 2023):

**Step 1: Your question becomes a vector**

Your question "What's Trump's trade policy?" gets sent to the same embedding model. It returns 768 numbers representing the *meaning* of your question.

**Step 2: One SQL query does everything**

The database runs a single query that does two things at once:

```
Give me rows from event_records
WHERE event_date >= '2023-10-01'           ← after the model's training cutoff
  AND event_date <= '2025-03-15'           ← before the simulation date
  AND date_confidence != 'uncertain'       ← not quarantined
ORDER BY similarity(embedding, question)   ← most relevant first
LIMIT 15                                   ← only take top 15
```

The `WHERE` clauses are **hard filters**. They eliminate rows before any similarity math happens. An event from April 2025 doesn't just rank low — it's physically excluded from the search. It cannot appear no matter what.

The `ORDER BY` is the vector similarity part. Among the surviving rows, it compares each event's 768-number embedding to your question's 768-number embedding, and sorts by how close they are. The closest matches — the events most *about* trade policy — float to the top.

The `LIMIT 15` takes only the top 15 results.

**So to directly answer your question:** The date comparison has *nothing* to do with embeddings. The date is a normal date column compared with normal `>=` and `<=` operators. The embedding is a separate column used only for ranking relevance. They live in the same table but serve completely different purposes.

### Step 3: SQL rows → Python objects

The database returns 15 rows. Each row gets converted into a Python object called an `EventRecord`. This is a structured data object with named fields:

```
EventRecord(
    event_date = 2024-01-15,
    headline = "Trump wins Iowa caucuses by record margin",
    summary = "Former President Trump won the Iowa Republican...",
    key_facts = ["Won with 51%", "30 points ahead of DeSantis"],
    direct_quotes = [Quote(quote="This is a big victory", speaker="Donald Trump")],
    topics = ["domestic_politics", "elections"],
    sources = [Source(name="AP News"), Source(name="Reuters")],
    confidence = 0.92,
    ...
)
```

At this point it's just a well-organized Python object. Not a string. Not text. Just structured data with fields you can access.

### Step 4: Python objects → plain text briefing

Each `EventRecord` has a method called `to_briefing_text()` that converts it into a human-readable (and LLM-readable) paragraph:

```
[2024-01-15] Trump wins Iowa caucuses by record margin
Former President Trump won the Iowa Republican caucuses with over 50% of the vote.
Key facts:
  • Won with 51% of the vote, 30 points ahead of DeSantis
  • Largest margin in Iowa caucus history
"This is a big victory for our movement" — Donald Trump (victory speech)
Sources: AP News, Reuters (3 sources, verified date)
```

Then all 15 of those paragraphs get wrapped in a briefing envelope:

```
════════════════════════════════════════════════════════════
 INTELLIGENCE BRIEFING — As of 2025-03-15
 Subject: Donald J. Trump
 Knowledge Window: 2023-10-01 → 2025-03-15
 Events in window: 847 | Showing top 15
════════════════════════════════════════════════════════════

[2024-01-15] Trump wins Iowa caucuses by record margin
...

──────────────────────────────────────────────────────────
[2024-03-05] Super Tuesday: Trump sweeps 14 of 15 states
...

──────────────────────────────────────────────────────────
(... 13 more events ...)

════════════════════════════════════════════════════════════
 END BRIEFING — 15 events retrieved
════════════════════════════════════════════════════════════
```

### Step 5: Briefing → LLM

That entire text block (one big string) gets injected into the LLM's prompt. Typically it would go into the system message or a context section before the user's question. Something like:

```
SYSTEM: You are an analyst simulating decisions as of March 15, 2025.
The following intelligence briefing contains verified events between
the model's training cutoff and the simulation date. Use ONLY this
information (plus your training data) to answer the question.

[THE BRIEFING TEXT GOES HERE]

USER: What would Trump do if China invaded Taiwan?
```

The LLM reads the briefing as context, combines it with its pre-training knowledge, and answers the question with awareness of events it otherwise wouldn't know about.

### The full chain, summarized

```
Your question (text)
    ↓
Embed it (text → 768 numbers)
    ↓
SQL query:  filter by dates (normal SQL)
            rank by similarity (vector math)
            take top 15
    ↓
15 database rows
    ↓
15 Python EventRecord objects
    ↓
15 plain-text paragraphs
    ↓
Wrapped in a briefing header/footer
    ↓
One big string injected into LLM prompt
    ↓
LLM reads it and answers your question
```
