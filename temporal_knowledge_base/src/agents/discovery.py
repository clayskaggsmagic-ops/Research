"""CHRONOS Discovery Agent — web search, GDELT integration, and source finding.

The Discovery Agent is the "eyes" of the research swarm. It:
1. Takes search queries from the Coordinator's research plan
2. Executes web searches via Tavily
3. Optionally queries GDELT for structured event data
4. Scores source quality by domain tier
5. Deduplicates candidates before passing downstream

It NEVER parses articles or validates dates — that's downstream agents' work.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from urllib.parse import urlparse

import httpx

from ..config import settings
from ..models import RawEventCandidate, SwarmState
from ..agents.coordinator import parse_query_string, _fuzzy_headline_match

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DISCOVERY_BATCH_SIZE = 30  # Queries to process per node invocation
SEARCH_RESULTS_PER_QUERY = 10
RATE_LIMIT_DELAY = 1.0  # Seconds between API calls


# ---------------------------------------------------------------------------
# Domain quality tiers
# ---------------------------------------------------------------------------

TIER_1_DOMAINS = {
    # Government
    "whitehouse.gov", "state.gov", "defense.gov", "treasury.gov",
    "congress.gov", "federalregister.gov", "govinfo.gov",
    # Wire services
    "apnews.com", "reuters.com",
}

TIER_2_DOMAINS = {
    # Major newspapers
    "nytimes.com", "washingtonpost.com", "wsj.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "politico.com", "thehill.com", "axios.com",
    "bloomberg.com", "ft.com", "cnn.com", "nbcnews.com", "cbsnews.com",
    "abcnews.go.com", "pbs.org", "npr.org", "usatoday.com",
    "aljazeera.com", "france24.com",
}

TIER_3_DOMAINS = {
    # Regional / specialty
    "foreignaffairs.com", "foreignpolicy.com", "defenseone.com",
    "lawfaremedia.org", "brookings.edu", "cfr.org", "csis.org",
    "rollcall.com", "theintercept.com", "propublica.org",
    "cnbc.com", "marketwatch.com", "economist.com",
}

GARBAGE_DOMAINS = {
    # Sites that won't have dateable factual events
    "pinterest.com", "quora.com", "reddit.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "youtube.com", "linkedin.com", "medium.com", "substack.com",
    "wikipedia.org",  # We want primary sources, not encyclopedia
}


def score_source_quality(url: str) -> float:
    """Score a URL's source quality by domain tier.

    Returns:
        0.0-1.0 quality score. Higher = more authoritative.
    """
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return 0.5

    if domain in TIER_1_DOMAINS or domain.endswith(".gov"):
        return 1.0
    elif domain in TIER_2_DOMAINS:
        return 0.8
    elif domain in TIER_3_DOMAINS:
        return 0.6
    elif domain in GARBAGE_DOMAINS:
        return 0.0  # Will be filtered out
    else:
        return 0.5  # Unknown domain — neutral


def is_garbage_domain(url: str) -> bool:
    """Check if a URL is from a blacklisted domain."""
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return True
    return domain in GARBAGE_DOMAINS


# ---------------------------------------------------------------------------
# Web search via Tavily
# ---------------------------------------------------------------------------

async def search_tavily(
    query: str,
    num_results: int = SEARCH_RESULTS_PER_QUERY,
) -> list[dict]:
    """Execute a web search via Tavily API.

    Returns:
        List of dicts with keys: title, url, snippet, date_hint
    """
    if not settings.tavily_api_key:
        logger.warning("No TAVILY_API_KEY set — skipping web search")
        return []

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": num_results,
        "include_raw_content": False,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily API error {e.response.status_code} for query '{query}'")
            return []
        except httpx.RequestError as e:
            logger.error(f"Tavily request error for query '{query}': {e}")
            return []

    results = []
    for item in data.get("results", []):
        result_url = item.get("url", "")

        # Filter garbage domains
        if is_garbage_domain(result_url):
            continue

        results.append({
            "title": item.get("title", ""),
            "url": result_url,
            "snippet": item.get("content", ""),
            "date_hint": item.get("published_date", ""),
        })

    logger.info(f"Tavily: '{query}' → {len(results)} results (filtered from {len(data.get('results', []))})")
    return results


# ---------------------------------------------------------------------------
# GDELT integration (free, no API key required)
# ---------------------------------------------------------------------------

async def search_gdelt(
    actor_name: str,
    start_date: date,
    end_date: date,
    max_results: int = 20,
) -> list[dict]:
    """Query the GDELT DOC 2.0 API for articles mentioning an actor.

    GDELT provides free access to a massive database of global news events.
    We use the DOC API (not the Event API) because it returns article URLs
    and metadata, which is what we need for downstream extraction.

    Returns:
        List of dicts with keys: title, url, snippet, date_hint
    """
    # GDELT DOC 2.0 API
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Format dates as GDELT expects (YYYYMMDDHHMMSS)
    start_str = start_date.strftime("%Y%m%d") + "000000"
    end_str = end_date.strftime("%Y%m%d") + "235959"

    params = {
        "query": f'"{actor_name}"',
        "mode": "ArtList",
        "maxrecords": str(max_results),
        "format": "json",
        "startdatetime": start_str,
        "enddatetime": end_str,
        "sort": "DateDesc",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"GDELT API error {e.response.status_code} for '{actor_name}'")
            return []
        except httpx.RequestError as e:
            logger.error(f"GDELT request error for '{actor_name}': {e}")
            return []
        except Exception as e:
            logger.error(f"GDELT unexpected error: {e}")
            return []

    results = []
    for article in data.get("articles", []):
        article_url = article.get("url", "")

        if is_garbage_domain(article_url):
            continue

        # Parse GDELT date format (YYYYMMDDTHHMMSS)
        date_hint = ""
        raw_date = article.get("seendate", "")
        if raw_date:
            try:
                dt = datetime.strptime(raw_date[:8], "%Y%m%d")
                date_hint = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        results.append({
            "title": article.get("title", ""),
            "url": article_url,
            "snippet": article.get("title", ""),  # GDELT doesn't give snippets
            "date_hint": date_hint,
        })

    logger.info(f"GDELT: '{actor_name}' ({start_date} → {end_date}) → {len(results)} articles")
    return results


# ---------------------------------------------------------------------------
# Candidate deduplication
# ---------------------------------------------------------------------------

def deduplicate_candidates(
    new_candidates: list[RawEventCandidate],
    existing_candidates: list[RawEventCandidate],
    visited_urls: set[str],
) -> list[RawEventCandidate]:
    """Remove candidates that are duplicates of existing work.

    Checks:
    1. URL not already visited
    2. Title not a near-duplicate of existing candidates (80% word overlap)
    """
    existing_titles = [c.title for c in existing_candidates if c.title]
    seen_urls = set(visited_urls)

    deduplicated = []
    for candidate in new_candidates:
        # URL dedup
        if candidate.url in seen_urls:
            continue

        # Title dedup (fuzzy)
        if candidate.title and _fuzzy_headline_match(candidate.title, existing_titles):
            continue

        deduplicated.append(candidate)
        seen_urls.add(candidate.url)
        if candidate.title:
            existing_titles.append(candidate.title)

    skipped = len(new_candidates) - len(deduplicated)
    if skipped > 0:
        logger.info(f"Candidate dedup: removed {skipped} duplicates")

    return deduplicated


# ---------------------------------------------------------------------------
# Convert search results to RawEventCandidate
# ---------------------------------------------------------------------------

def _parse_date_hint(date_str: str) -> date | None:
    """Try to parse a date hint string from search results."""
    if not date_str:
        return None

    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def results_to_candidates(
    results: list[dict],
    discovery_query: str,
) -> list[RawEventCandidate]:
    """Convert raw search results into RawEventCandidate objects."""
    candidates = []
    for r in results:
        if not r.get("url"):
            continue

        candidates.append(RawEventCandidate(
            url=r["url"],
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            preliminary_date=_parse_date_hint(r.get("date_hint", "")),
            source_name=urlparse(r["url"]).netloc.lstrip("www."),
            discovery_query=discovery_query,
        ))

    return candidates


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

SEARCH_CONCURRENCY = 5  # Max concurrent search API calls


async def _run_single_search(
    query_str: str,
    sem: asyncio.Semaphore,
) -> tuple[list[RawEventCandidate], str | None]:
    """Execute a single search query behind a semaphore.

    Returns (candidates, error_msg_or_None).
    """
    async with sem:
        parsed = parse_query_string(query_str)
        search_text = parsed["query"]
        try:
            tavily_results = await search_tavily(search_text)
            candidates = results_to_candidates(tavily_results, search_text)
            return candidates, None
        except Exception as e:
            error_msg = f"Tavily search failed for '{search_text}': {e}"
            logger.error(error_msg)
            return [], error_msg


async def _run_single_gdelt(
    month_str: str,
    subject_name: str,
    sem: asyncio.Semaphore,
) -> list[RawEventCandidate]:
    """Execute a single GDELT query behind a semaphore."""
    async with sem:
        try:
            year, month = int(month_str.split("-")[0]), int(month_str.split("-")[1])
            month_start = date(year, month, 1)
            if month == 12:
                month_end = date(year + 1, 1, 1)
            else:
                month_end = date(year, month + 1, 1)

            gdelt_results = await search_gdelt(
                actor_name=subject_name,
                start_date=month_start,
                end_date=month_end,
                max_results=15,
            )
            return results_to_candidates(
                gdelt_results, f"GDELT:{subject_name}:{month_str}"
            )
        except Exception as e:
            logger.warning(f"GDELT query failed for {month_str}: {e}")
            return []


async def discovery_node(state: SwarmState) -> SwarmState:
    """LangGraph node: execute search queries and produce raw event candidates.

    Runs up to SEARCH_CONCURRENCY queries in parallel for ~5x speedup.
    Pops the next batch of queries from the research plan, searches for each,
    and adds unique candidates to the state.
    """
    if not state.research_plan:
        logger.info("Discovery: no queries in research plan — nothing to do")
        return state

    # Pop the next batch
    batch_size = min(DISCOVERY_BATCH_SIZE, len(state.research_plan))
    batch = state.research_plan[:batch_size]
    state.research_plan = state.research_plan[batch_size:]

    logger.info(
        f"Discovery: processing {len(batch)} queries CONCURRENTLY (max {SEARCH_CONCURRENCY}) "
        f"({len(state.research_plan)} remaining in plan)"
    )

    all_new_candidates: list[RawEventCandidate] = []
    sem = asyncio.Semaphore(SEARCH_CONCURRENCY)

    # --- Parallel Tavily searches ---
    search_tasks = [_run_single_search(q, sem) for q in batch]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for result in search_results:
        if isinstance(result, BaseException):
            state.errors.append(f"Search task exception: {result}")
            continue
        candidates, error = result
        all_new_candidates.extend(candidates)
        if error:
            state.errors.append(error)

    # --- Parallel GDELT sweeps ---
    batch_months = set()
    for query_str in batch:
        parsed = parse_query_string(query_str)
        month = parsed["month"]
        if month != "unknown":
            batch_months.add(month)

    if batch_months:
        gdelt_tasks = [
            _run_single_gdelt(m, state.subject_name, sem) for m in batch_months
        ]
        gdelt_results = await asyncio.gather(*gdelt_tasks, return_exceptions=True)
        for result in gdelt_results:
            if isinstance(result, BaseException):
                logger.warning(f"GDELT task exception: {result}")
                continue
            all_new_candidates.extend(result)

    # --- Deduplicate against existing candidates ---
    unique_candidates = deduplicate_candidates(
        new_candidates=all_new_candidates,
        existing_candidates=state.raw_candidates,
        visited_urls=state.urls_visited,
    )

    # --- Update state ---
    state.raw_candidates.extend(unique_candidates)
    for c in unique_candidates:
        if c.url:
            state.urls_visited.add(c.url)

    logger.info(
        f"Discovery complete: {len(unique_candidates)} new candidates added "
        f"(total: {len(state.raw_candidates)}, URLs visited: {len(state.urls_visited)})"
    )

    return state
