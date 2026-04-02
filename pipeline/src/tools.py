"""
Agent Tools — web_search and web_scrape for the ReAct discovery agent.

Uses langchain-tavily (the current official package, NOT the deprecated
langchain_community.tools.tavily_search).
"""

from __future__ import annotations

from langchain_tavily import TavilyExtract, TavilySearch


def build_search_tool(
    max_results: int = 5,
    search_depth: str = "advanced",
    topic: str = "news",
) -> TavilySearch:
    """
    Build a TavilySearch tool configured for political news discovery.

    Requires TAVILY_API_KEY in environment variables.
    """
    return TavilySearch(
        name="web_search",
        description=(
            "Search the web for recent news, government actions, and political events. "
            "Returns summarized content from top results with source URLs. "
            "Use this to discover presidential decisions, executive orders, tariff "
            "actions, personnel changes, and other government actions."
        ),
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
    )


def build_extract_tool(
    extract_depth: str = "advanced",
) -> TavilyExtract:
    """
    Build a TavilyExtract tool for reading full page content from URLs.

    Requires TAVILY_API_KEY in environment variables.
    """
    return TavilyExtract(
        name="web_scrape",
        description=(
            "Extract and read the full text content from a specific URL. "
            "Use this when you find a promising source from web_search and "
            "need to read the full article for detailed information about "
            "a presidential decision — dates, specifics, attribution."
        ),
        extract_depth=extract_depth,
    )


def get_stage1_tools() -> list:
    """Return the tool set for the Stage 1 seed harvesting agent."""
    return [
        build_search_tool(),
        build_extract_tool(),
    ]
