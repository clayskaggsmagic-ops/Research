"""
Two-stage refined retrieval for E2.

Pipeline:
  1. LLM call (refiner_model) on (question_text + background + resolution_criteria)
     → JSON {actors, topic_tags, date_subwindow, paraphrases[]}
  2. Over-retrieve top-N candidate events using each paraphrase, merged and
     deduplicated. Topic filter applied when CHRONOS supports it.
  3. LLM call (refiner_model) labels each event one of
     `supports-YES | supports-NO | background | irrelevant`.
  4. Drop `irrelevant`, rank by label priority, keep ~8-12, group by label.

The output is a single briefing string ready to be fed into the prediction
prompt, with a short preamble explaining the grouping.

Note: this module depends on `temporal_knowledge_base` for retrieval and
`langchain_anthropic` for the refiner LLM calls. Both are imported lazily so
the module is importable in isolation.
"""

from __future__ import annotations

import json
from datetime import date


EXPAND_SYSTEM = """\
You expand a forecasting question into retrieval hints for a temporal knowledge
base. Output strict JSON only:

{
  "actors": ["list of named entities central to the question"],
  "topic_tags": ["short slugs, e.g. 'tariffs', 'iran', 'scotus'"],
  "date_subwindow": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "paraphrases": ["3-5 distinct natural-language rephrasings of the question"]
}

The date_subwindow must end on or before the simulation_date provided."""


RERANK_SYSTEM = """\
You are labeling retrieved events for a forecasting question. For each event
assign exactly one label:
  - supports-YES        : event materially raises the probability the question resolves YES (or its primary option)
  - supports-NO         : event materially lowers that probability
  - background          : relevant context, but not directly probative
  - irrelevant          : not about the question

Output strict JSON only:
{"labels": [{"event_id": "...", "label": "supports-YES|supports-NO|background|irrelevant"}]}
"""


LABEL_PRIORITY = {"supports-YES": 0, "supports-NO": 1, "background": 2, "irrelevant": 3}


async def refine_briefing(
    *,
    question: dict,
    model_id: str,
    refiner_model_id: str,
    over_retrieve_k: int,
    keep_min: int,
    keep_max: int,
) -> str:
    """End-to-end refinement. Returns a single briefing string."""
    from temporal_knowledge_base.src.retrieval import retrieve_structured  # type: ignore
    from temporal_knowledge_base.src.models import RetrievalRequest  # type: ignore

    sim_date = date.fromisoformat(question["simulation_date"])

    # ── Stage 1: expand ──
    hints = await _expand_query(question, refiner_model_id, sim_date)
    paraphrases: list[str] = hints.get("paraphrases") or [question["question_text"]]
    topic_tags: list[str] = hints.get("topic_tags") or []

    # ── Stage 2: over-retrieve per paraphrase, merge + dedupe ──
    per_query_k = max(1, over_retrieve_k // max(1, len(paraphrases)))
    merged: dict[str, dict] = {}
    from temporal_knowledge_base.src.config import ModelConfig  # type: ignore

    model_cutoff = ModelConfig.get_cutoff(model_id)
    for q in paraphrases:
        req = RetrievalRequest(
            query=q,
            simulation_date=sim_date,
            model_training_cutoff=model_cutoff,
            top_k=per_query_k,
            topic_filter=topic_tags or None,
        )
        result = await retrieve_structured(req)
        for ev in _iter_events(result):
            key = _event_key(ev)
            if key not in merged:
                merged[key] = ev

    candidates = list(merged.values())[:over_retrieve_k]
    if not candidates:
        return _empty_briefing(question, hints)

    # ── Stage 3: rerank / label each event ──
    labels = await _rerank_events(question, candidates, refiner_model_id)

    # ── Stage 4: filter, sort by (priority, original rank), keep [keep_min, keep_max] ──
    labeled = []
    for ev in candidates:
        lab = labels.get(_event_key(ev), "background")
        if lab == "irrelevant":
            continue
        labeled.append((LABEL_PRIORITY.get(lab, 2), lab, ev))
    labeled.sort(key=lambda row: row[0])
    kept = labeled[:keep_max]
    if len(kept) < keep_min:
        kept = labeled[: min(keep_min, len(labeled))]

    return _format_refined_briefing(question, kept, hints)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _iter_events(result) -> list[dict]:
    """Normalize the CHRONOS RetrievalResult.events into plain dicts."""
    events = getattr(result, "events", None) or []
    out = []
    for ev in events:
        d = ev if isinstance(ev, dict) else getattr(ev, "model_dump", lambda: ev.__dict__)()
        out.append(d)
    return out


def _event_key(ev: dict) -> str:
    return str(ev.get("event_id") or ev.get("id") or ev.get("url") or ev.get("title") or hash(json.dumps(ev, sort_keys=True, default=str)))


def _refiner_invoke(chat, messages):
    """Invoke a refiner chat with transient-error retry (429, 5xx, disconnects)."""
    import time
    from evaluation_plan.src.llm_client import _is_transient
    delays = [2, 8, 30, 60, 120]
    last_exc = None
    for attempt in range(len(delays) + 1):
        try:
            return chat.invoke(messages)
        except Exception as e:
            last_exc = e
            if attempt == len(delays) or not _is_transient(e):
                raise
            time.sleep(delays[attempt])
    raise last_exc  # unreachable


def _build_refiner(refiner_model_id: str, max_tokens: int):
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore  # noqa: F401

    if refiner_model_id.lower().startswith(("gemini", "models/gemini")):
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        return ChatGoogleGenerativeAI(
            model=refiner_model_id,
            temperature=0.0,
            max_output_tokens=max_tokens,
            timeout=120,
            max_retries=3,
        )
    from langchain_anthropic import ChatAnthropic  # type: ignore

    return ChatAnthropic(model=refiner_model_id, temperature=0.0, max_tokens=max_tokens)


async def _expand_query(question: dict, refiner_model_id: str, sim_date: date) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    user = (
        f"SIMULATION DATE: {sim_date.isoformat()}\n"
        f"QUESTION: {question['question_text']}\n"
        f"BACKGROUND: {question.get('background', '')}\n"
        f"RESOLUTION CRITERIA: {question.get('resolution_criteria', '')}"
    )
    chat = _build_refiner(refiner_model_id, max_tokens=512)
    resp = _refiner_invoke(chat, [SystemMessage(content=EXPAND_SYSTEM), HumanMessage(content=user)])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        from evaluation_plan.src.llm_client import extract_json_object
        return extract_json_object(text)


async def _rerank_events(question: dict, events: list[dict], refiner_model_id: str) -> dict[str, str]:
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    enumerated = []
    for ev in events:
        enumerated.append({
            "event_id": _event_key(ev),
            "date": str(ev.get("date") or ev.get("event_date") or ""),
            "summary": ev.get("summary") or ev.get("title") or "",
        })
    user = (
        f"QUESTION: {question['question_text']}\n"
        f"RESOLUTION CRITERIA: {question.get('resolution_criteria', '')}\n\n"
        f"EVENTS (JSON):\n{json.dumps(enumerated, indent=2)}"
    )
    chat = _build_refiner(refiner_model_id, max_tokens=2048)
    resp = _refiner_invoke(chat, [SystemMessage(content=RERANK_SYSTEM), HumanMessage(content=user)])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        from evaluation_plan.src.llm_client import extract_json_object
        obj = extract_json_object(text)
    labels: dict[str, str] = {}
    for entry in obj.get("labels", []):
        labels[str(entry.get("event_id"))] = entry.get("label", "background")
    return labels


def _format_refined_briefing(question: dict, kept: list[tuple[int, str, dict]], hints: dict) -> str:
    groups: dict[str, list[dict]] = {"supports-YES": [], "supports-NO": [], "background": []}
    for _, label, ev in kept:
        groups.setdefault(label, []).append(ev)

    lines = [
        "REFINED BRIEFING (events grouped by probative direction).",
        f"Retrieval hints: actors={hints.get('actors')}, topics={hints.get('topic_tags')}.",
        "",
    ]
    for label in ("supports-YES", "supports-NO", "background"):
        bucket = groups.get(label) or []
        if not bucket:
            continue
        lines.append(f"## {label.upper()} ({len(bucket)})")
        for ev in bucket:
            d = ev.get("date") or ev.get("event_date") or "?"
            s = ev.get("summary") or ev.get("title") or ""
            lines.append(f"- [{d}] {s}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _empty_briefing(question: dict, hints: dict) -> str:
    return (
        "REFINED BRIEFING — no qualifying events retrieved.\n"
        f"Retrieval hints: actors={hints.get('actors')}, topics={hints.get('topic_tags')}."
    )
