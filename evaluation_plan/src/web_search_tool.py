"""
Web-search tool binding for E5 (Analyst + web search, answerability gate).

Uses Anthropic's server-side web search tool, but requires the caller to
enforce a temporal upper bound equal to the question's simulation_date.

Enforcement is two-layered, because search APIs are imperfect:
  1. In-prompt: the system injects "Only use results dated on-or-before
     {simulation_date}" into the analyst system prompt.
  2. Post-hoc: if the model cites URLs, we (optionally) check the page's
     publish date and drop cites that post-date the simulation.

This module exposes `web_search_tool_spec(simulation_date)` which returns the
tool definition to pass into `ChatAnthropic.bind_tools(...)` and an
augmented-system-prompt wrapper.
"""

from __future__ import annotations

from datetime import date


WEB_SEARCH_TOOL_TYPE = "web_search_20250305"


def web_search_tool_spec(max_uses: int = 5) -> dict:
    """Anthropic server-side web search tool spec."""
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
