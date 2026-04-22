"""CHRONOS Coordinator Agent — research planner, supervisor, and gap-filler.

The Coordinator is the "brain" of the research swarm. It:
1. Generates initial research plans (query lists by month × topic)
2. Tracks what's been covered to prevent redundant work
3. Receives coverage gap reports and generates targeted follow-up queries
4. Decides when collection is complete

It NEVER searches or scrapes — it only plans and dispatches.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from dateutil.relativedelta import relativedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings
from ..models import SwarmState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic categories — used for structured plan generation
# ---------------------------------------------------------------------------

TOPIC_CATEGORIES = [
    "executive_actions",     # Executive orders, memos, directives, proclamations
    "foreign_policy",        # Summits, sanctions, diplomatic statements, treaties
    "domestic_policy",       # Legislation, vetoes, regulatory actions, court nominations
    "economic_decisions",    # Tariffs, trade deals, fiscal policy, budgets
    "personnel",             # Appointments, firings, nominations, resignations
    "crises_and_responses",  # Natural disasters, security events, public health
    "public_statements",     # Press conferences, social media posts, rally speeches
]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLAN_GENERATION_SYSTEM_PROMPT = """You are a research planning specialist for the CHRONOS temporal knowledge base.
Your job is to generate highly targeted web search queries that will find news articles
about specific, dateable events involving a world leader.

CRITICAL RULES:
1. Every query must be designed to find articles about events with SPECIFIC DATES.
   We need to know WHEN things happened, not just WHAT happened.
2. Avoid queries that will return undated opinion pieces, analysis, or commentary.
3. Include the month and year in every query to anchor results temporally.
4. Prefer queries that will find primary sources: official announcements, signing
   ceremonies, press briefings, government press releases.
5. Each query should target ONE specific type of event, not broad overviews.

OUTPUT FORMAT:
Return a JSON array of query objects. Each object has:
- "query": the search string (50-80 chars, specific and temporal)
- "month": target month in "YYYY-MM" format
- "topic": one of the allowed topic categories
- "specificity": "initial" or "followup"

EXAMPLES OF GOOD QUERIES:
✅ "Trump executive orders signed January 2025 official"
✅ "Trump tariff announcement steel imports March 2025 date"
✅ "Trump cabinet nominations Senate confirmation February 2025"
✅ "Trump NATO summit statements June 2024 specific date"
✅ "Trump federal reserve interest rate comments December 2024"

EXAMPLES OF BAD QUERIES (DO NOT GENERATE THESE):
❌ "Trump presidency analysis" (no date anchor, will return opinion)
❌ "Trump controversial decisions" (vague, editorialized)
❌ "Trump news" (too broad, no temporal anchor)
❌ "Trump economy impact" (analysis, not events)
❌ "What did Trump do in 2024" (too broad, will return summaries)

ALLOWED TOPIC CATEGORIES:
- executive_actions
- foreign_policy
- domestic_policy
- economic_decisions
- personnel
- crises_and_responses
- public_statements"""


GAP_FILLING_SYSTEM_PROMPT = """You are a research gap analyst for the CHRONOS temporal knowledge base.
You have been given a list of coverage gaps — months and/or topics where we have
insufficient data. Your job is to generate HIGHLY SPECIFIC follow-up search queries
to fill these gaps.

Follow-up queries must be MORE SPECIFIC than initial queries:
- Initial: "Trump tariff decisions January 2025"
- Follow-up: "Trump steel tariff Section 232 January 2025 announcement date"
- Follow-up: "Trump 25% Canada tariff executive order January 2025"

CRITICAL: Each query must include the specific month/year and be designed to find
articles about events with concrete dates.

OUTPUT FORMAT:
Return a JSON array of query objects. Each object has:
- "query": the search string (specific, temporal, targeted at the gap)
- "month": target month in "YYYY-MM" format
- "topic": one of the allowed topic categories
- "specificity": "followup"

ALLOWED TOPIC CATEGORIES:
- executive_actions
- foreign_policy
- domestic_policy
- economic_decisions
- personnel
- crises_and_responses
- public_statements"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    """Get the research LLM for plan generation."""
    return ChatGoogleGenerativeAI(
        model=settings.research_model,
        google_api_key=settings.google_api_key,
        temperature=0.3,  # Low temp for structured output
        timeout=120,
        max_retries=2,
    )


def _generate_month_chunks(start: date, end: date) -> list[str]:
    """Break a date range into monthly chunks as 'YYYY-MM' strings."""
    months = []
    current = start.replace(day=1)
    end_month = end.replace(day=1)
    while current <= end_month:
        months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)
    return months


def _fuzzy_headline_match(headline: str, existing_headlines: list[str], threshold: float = 0.8) -> bool:
    """Check if a headline is a near-duplicate of any existing headline.

    Uses simple word overlap ratio. Not perfect, but fast and good enough
    to catch obvious duplicates like:
      "Trump Signs Executive Order on AI" vs "Trump Signs AI Executive Order"
    """
    if not existing_headlines:
        return False

    headline_words = set(headline.lower().split())
    if not headline_words:
        return False

    for existing in existing_headlines:
        existing_words = set(existing.lower().split())
        if not existing_words:
            continue
        overlap = len(headline_words & existing_words)
        max_len = max(len(headline_words), len(existing_words))
        if max_len > 0 and overlap / max_len >= threshold:
            return True

    return False


async def generate_initial_plan(
    subject_name: str,
    start_date: date,
    end_date: date,
) -> list[str]:
    """Generate the initial research plan — a list of search queries.

    Breaks the time range into monthly chunks and generates 5-8 queries
    per month across all topic categories.

    Returns:
        List of search query strings, tagged with month and topic metadata.
    """
    months = _generate_month_chunks(start_date, end_date)
    llm = _get_llm()

    all_queries: list[str] = []

    # Process in batches of 4 months to stay within context limits
    batch_size = 4
    for i in range(0, len(months), batch_size):
        batch_months = months[i : i + batch_size]

        prompt = (
            f"Generate search queries for the following leader and time periods.\n\n"
            f"Leader: {subject_name}\n"
            f"Months to cover: {', '.join(batch_months)}\n"
            f"Topics to cover per month: {', '.join(TOPIC_CATEGORIES)}\n\n"
            f"Generate 5-8 queries per month. That's {5 * len(batch_months)}-{8 * len(batch_months)} "
            f"queries total for this batch.\n\n"
            f"Return ONLY the JSON array, no other text."
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(content=PLAN_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            # Parse the JSON response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]

            queries_data = json.loads(content)

            for q in queries_data:
                # Store as a structured string: "query|||month|||topic|||specificity"
                query_str = f"{q['query']}|||{q['month']}|||{q['topic']}|||{q.get('specificity', 'initial')}"
                all_queries.append(query_str)

            logger.info(
                f"Generated {len(queries_data)} queries for months {batch_months[0]}-{batch_months[-1]}"
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response for months {batch_months}: {e}")
            # Fallback: generate simple template queries
            for month in batch_months:
                for topic in TOPIC_CATEGORIES:
                    fallback_query = f"{subject_name} {topic.replace('_', ' ')} {month}"
                    all_queries.append(f"{fallback_query}|||{month}|||{topic}|||initial")

    logger.info(f"Total initial plan: {len(all_queries)} queries across {len(months)} months")
    return all_queries


async def generate_gap_filling_queries(
    subject_name: str,
    coverage_gaps: list[str],
) -> list[str]:
    """Generate targeted follow-up queries based on coverage gaps.

    Args:
        subject_name: The leader being researched.
        coverage_gaps: List of gap descriptions from the Coverage Auditor,
            e.g. ["2024-03: sparse (5 events, need 15)", "2024-07: missing foreign_policy"]

    Returns:
        List of more specific search query strings.
    """
    if not coverage_gaps:
        return []

    llm = _get_llm()

    prompt = (
        f"We have the following coverage gaps in our knowledge base about {subject_name}.\n\n"
        f"GAPS TO FILL:\n"
        + "\n".join(f"  - {gap}" for gap in coverage_gaps)
        + "\n\n"
        f"Generate 3-5 HIGHLY SPECIFIC follow-up search queries for each gap.\n"
        f"These must be MORE specific than initial queries — target exact policies,\n"
        f"specific meetings, named legislation, etc.\n\n"
        f"Return ONLY the JSON array, no other text."
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=GAP_FILLING_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        queries_data = json.loads(content)

        followup_queries = []
        for q in queries_data:
            query_str = f"{q['query']}|||{q['month']}|||{q['topic']}|||followup"
            followup_queries.append(query_str)

        logger.info(f"Generated {len(followup_queries)} follow-up queries for {len(coverage_gaps)} gaps")
        return followup_queries

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse gap-filling response: {e}")
        # Fallback: generate simple queries from gap descriptions
        fallback = []
        for gap in coverage_gaps:
            # Parse gap string like "2024-03: sparse (5 events)"
            month = gap.split(":")[0].strip() if ":" in gap else "unknown"
            fallback_query = f"{subject_name} major events decisions {month}"
            fallback.append(f"{fallback_query}|||{month}|||executive_actions|||followup")
        return fallback


def parse_query_string(query_str: str) -> dict:
    """Parse a structured query string back into its components.

    Format: "query_text|||month|||topic|||specificity"
    """
    parts = query_str.split("|||")
    return {
        "query": parts[0] if len(parts) > 0 else query_str,
        "month": parts[1] if len(parts) > 1 else "unknown",
        "topic": parts[2] if len(parts) > 2 else "unknown",
        "specificity": parts[3] if len(parts) > 3 else "initial",
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_queries(
    new_queries: list[str],
    existing_queries: list[str],
    visited_urls: set[str],
) -> list[str]:
    """Remove queries that are redundant with existing work.

    Checks for:
    1. Exact duplicate query strings
    2. Near-duplicate queries (high word overlap)
    """
    existing_texts = {parse_query_string(q)["query"].lower() for q in existing_queries}
    existing_headlines = [parse_query_string(q)["query"] for q in existing_queries]

    deduplicated = []
    for query_str in new_queries:
        query_text = parse_query_string(query_str)["query"].lower()

        # Skip exact duplicates
        if query_text in existing_texts:
            continue

        # Skip near-duplicates
        if _fuzzy_headline_match(query_text, existing_headlines):
            continue

        deduplicated.append(query_str)
        existing_texts.add(query_text)
        existing_headlines.append(parse_query_string(query_str)["query"])

    skipped = len(new_queries) - len(deduplicated)
    if skipped > 0:
        logger.info(f"Deduplication removed {skipped} redundant queries")

    return deduplicated


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

MAX_RESEARCH_ROUNDS = 5


async def coordinator_node(state: SwarmState) -> SwarmState:
    """LangGraph node: plans research, fills gaps, or signals completion.

    Decision logic:
    1. If research_plan is empty AND no gaps → generate initial plan
    2. If coverage_gaps is non-empty → generate follow-up queries for gaps
    3. If collection_complete is True → no-op, pass through

    This node is re-entered after each coverage audit. The coverage loop is:
    coordinator → discovery → extraction → cleaning → validation → indexing → auditor → coordinator
    """
    # Track research rounds to prevent infinite loops
    round_count = state.events_per_month.get("_round_count", 0)

    if state.collection_complete:
        logger.info("Collection already marked complete — passing through")
        return state

    if round_count >= MAX_RESEARCH_ROUNDS:
        logger.warning(f"Max research rounds ({MAX_RESEARCH_ROUNDS}) reached — forcing completion")
        state.collection_complete = True
        state.errors.append(
            f"Collection stopped after {MAX_RESEARCH_ROUNDS} rounds. "
            f"Coverage may be incomplete. Remaining gaps: {state.coverage_gaps}"
        )
        return state

    # Increment round counter
    state.events_per_month["_round_count"] = round_count + 1

    if not state.research_plan and not state.coverage_gaps:
        # --- INITIAL PLAN ---
        logger.info(
            f"Generating initial research plan for {state.subject_name} "
            f"({state.collection_start} → {state.collection_end})"
        )

        queries = await generate_initial_plan(
            subject_name=state.subject_name,
            start_date=state.collection_start,
            end_date=state.collection_end,
        )

        state.research_plan = queries
        logger.info(f"Initial plan: {len(queries)} queries generated")

    elif state.coverage_gaps:
        # --- GAP-FILLING ---
        logger.info(
            f"Round {round_count + 1}: Generating follow-up queries for "
            f"{len(state.coverage_gaps)} coverage gaps"
        )

        followup_queries = await generate_gap_filling_queries(
            subject_name=state.subject_name,
            coverage_gaps=state.coverage_gaps,
        )

        # Deduplicate against existing queries and visited URLs
        deduplicated = deduplicate_queries(
            new_queries=followup_queries,
            existing_queries=state.research_plan,
            visited_urls=state.urls_visited,
        )

        state.research_plan.extend(deduplicated)
        state.coverage_gaps = []  # Clear gaps — auditor will re-assess after next round
        logger.info(
            f"Added {len(deduplicated)} follow-up queries "
            f"(filtered from {len(followup_queries)} candidates)"
        )

    else:
        # research_plan exists but no gaps — this means the auditor hasn't run yet
        # or discovery still has queries to process. Pass through.
        logger.info("Research plan exists with no gaps — passing through to Discovery")

    return state
