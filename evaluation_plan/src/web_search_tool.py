"""
Web-search helper for E5 (Analyst + web search, answerability gate).

Implementations:
  1. Anthropic server-side web_search tool (original).
  2. Gemini google_search built-in grounding — NOT wired via bind_tools (langchain-google-genai
     rejects built-in tool dicts); call Gemini directly if you want this.
  3. Tavily pre-retrieval (default for Gemini runs): call Tavily with a date-bounded
     query, format top results as a search-context block, and hand that to the model
     as part of the user prompt. Deterministic, no tool-calling required.

Enforcement is two-layered, because search APIs are imperfect:
  1. In-prompt: the system injects "Only use results dated on-or-before
     {simulation_date}".
  2. Post-hoc: if Tavily returns a `published_date`, we filter results whose date
     is after the simulation_date.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any


WEB_SEARCH_TOOL_TYPE = "web_search_20250305"


def web_search_tool_spec(max_uses: int = 5, provider: str = "anthropic") -> dict | None:
    """Returns a tool spec for model-side tool use. For Gemini runs we return
    None — E5 uses Tavily pre-retrieval instead (see `tavily_search_context`)."""
    if provider == "google":
        return None
    return {
        "type": WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        "max_uses": max_uses,
    }


def augment_system_with_temporal_constraint(system_text: str, simulation_date: str | date) -> str:
    sim = simulation_date if isinstance(simulation_date, str) else simulation_date.isoformat()
    preamble = (
        "TEMPORAL CONSTRAINT: This is a historical-forecasting task. "
        f"You must only consider information dated on or before {sim}. "
        "If a web search result is clearly dated after this cutoff, treat it "
        "as unseen and do not use it in your analysis.\n\n"
    )
    return preamble + system_text


# ── Tavily pre-retrieval ──────────────────────────────────────────────────────


def _parse_date(val: Any) -> date | None:
    if not val:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    try:
        # Tavily usually returns ISO-like strings; be permissive.
        if isinstance(val, str):
            return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
    except Exception:
        pass
    return None


def tavily_search_context(
    query: str,
    simulation_date: date | None,
    max_results: int = 10,
    strict_date_filter: bool = True,
    max_kept: int = 10,
    snippet_chars: int = 300,
) -> str:
    """Run a Tavily search and return a formatted context block.

    Two modes:
      - Date-bounded (simulation_date set, strict_date_filter=True): asks Tavily
        with end_date=simulation_date and post-hoc drops items whose
        published_date is missing, unparseable, or after simulation_date.
        Tight leakage control but usually drops most results (Tavily rarely
        populates published_date).
      - Unbounded / answerability gate (simulation_date=None OR
        strict_date_filter=False): no end_date, no post-hoc drop. Returns
        everything Tavily gives us — including post-outcome coverage. Use this
        to verify questions are answerable given full hindsight.

    Returns an empty string (caller should handle gracefully) if Tavily isn't
    configured or the call fails.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return ""

    # Lazy import so the module stays importable without the dep.
    try:
        import requests  # type: ignore
    except ImportError:
        return ""

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
    }
    if simulation_date is not None and strict_date_filter:
        payload["end_date"] = simulation_date.isoformat()
    try:
        r = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"(web_search unavailable: {type(e).__name__}: {e})"

    results = data.get("results", []) or []
    kept = []
    dropped_nodate = 0
    dropped_postsim = 0
    for item in results:
        if simulation_date is None or not strict_date_filter:
            kept.append(item)
            if len(kept) >= max_kept:
                break
            continue
        pd = _parse_date(item.get("published_date"))
        if pd is None:
            dropped_nodate += 1
            continue
        elif pd > simulation_date:
            dropped_postsim += 1
            continue
        kept.append(item)
        if len(kept) >= max_kept:
            break

    if not kept:
        if simulation_date is None or not strict_date_filter:
            return "(web_search returned no results)"
        return (
            f"(web_search returned no results dated on-or-before {simulation_date.isoformat()}; "
            f"dropped {dropped_nodate} undated, {dropped_postsim} post-sim-date)"
        )

    if simulation_date is None or not strict_date_filter:
        header = (
            f"WEB SEARCH RESULTS (unbounded; strict_date_filter={strict_date_filter}; "
            f"kept={len(kept)}):"
        )
    else:
        header = (
            f"WEB SEARCH RESULTS (date ≤ {simulation_date.isoformat()}; "
            f"strict_date_filter={strict_date_filter}; "
            f"kept={len(kept)}, dropped_undated={dropped_nodate}, dropped_post_sim={dropped_postsim}):"
        )
    lines = [header, ""]
    for i, item in enumerate(kept, 1):
        title = item.get("title", "").strip()
        url = item.get("url", "")
        pd = item.get("published_date", "") or "unknown"
        content = (item.get("content") or "").strip()
        if len(content) > snippet_chars:
            content = content[:snippet_chars] + "…"
        lines.append(f"[{i}] {title}")
        lines.append(f"    date: {pd}   url: {url}")
        lines.append(f"    {content}")
        lines.append("")
    return "\n".join(lines).rstrip()
