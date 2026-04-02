"""
Stage 1 — Seed Harvesting (Fully Autonomous Discovery Agent)

Architecture (per Bosse et al. 2026, adapted):
  1. ReAct agents (one per domain) autonomously search the web for Trump
     presidential decisions between training_cutoff_date and today_date
  2. A merger step deduplicates, tags domains, and flags uncertain attribution

The agents are fully autonomous — they decide what to search, which sources
to trust, and how deep to dig. No hardcoded data sources.

Entry point: run_stage1(state: PipelineState) -> PipelineState
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


from src.config import SEEDS_DIR
from src.schemas import DecisionSeed, DomainType, PipelineState, Source
from src.tools import get_stage1_tools

logger = logging.getLogger(__name__)


# ── Structured Output Schema for Agent ─────────────────────────────────────────


class DiscoveredSeed(BaseModel):
    """A single decision discovered by the research agent."""

    event_description: str = Field(description="What happened — full context around the decision")
    decision_taken: str = Field(description="The specific action Trump took")
    decision_date: str = Field(description="YYYY-MM-DD when the action was taken")
    simulation_date: str = Field(
        description="YYYY-MM-DD, 1-30 days BEFORE decision_date — the 'fake today'"
    )
    plausible_alternatives: list[str] = Field(
        description="What else he could have done — MUST include 'Take no action'"
    )
    attribution_evidence: str = Field(
        description="Why this is personally attributable to Trump, not bureaucratic default"
    )
    source_urls: list[str] = Field(description="URLs of sources confirming this decision")
    source_names: list[str] = Field(description="Names of each source (e.g. 'Reuters', 'AP')")
    confidence: str = Field(description="high / medium / low — how confident are you this happened")


class AgentResponse(BaseModel):
    """Structured output from the seed harvesting agent for one domain."""

    domain: str = Field(description="The domain being researched")
    seeds: list[DiscoveredSeed] = Field(description="All decisions discovered")
    search_summary: str = Field(description="Brief summary of what was searched and found")


# ── Agent Prompt ───────────────────────────────────────────────────────────────


DISCOVERY_SYSTEM_PROMPT = """\
You are an expert political research agent. Your job is to discover REAL \
presidential decisions made by {leader} between {cutoff_date} and {today_date} \
in the domain: **{domain_name}**.

## Domain: {domain_description}

## Your Task
1. Use web_search to find real decisions, actions, and orders by {leader} in \
this domain during the specified time period.
2. For each decision you find, verify it with at least 2 sources.
3. Use web_scrape to read full articles when you need specific dates, details, \
or attribution evidence.
4. Extract structured information for each decision.

## What to Search For
{domain_search_guidance}

## Quality Standards
- Only include decisions that ACTUALLY HAPPENED — no speculation or rumors
- Each decision must be personally attributable to {leader} (not Congress, courts, \
  agencies acting independently, or foreign governments)
- Include the specific date (not just "March 2026")
- The simulation_date must be 1-30 days BEFORE the decision_date
- plausible_alternatives MUST always include "Take no action"
- Cross-reference at least 2 sources before including a decision

## Output
Return ALL decisions you discover. Aim for thoroughness — search multiple \
angles, try different search queries, and look at both .gov sources and \
major news outlets (AP, Reuters, NYT, WSJ, Politico).\
"""

DOMAIN_SEARCH_GUIDANCE: dict[DomainType, str] = {
    DomainType.TRADE_TARIFFS: (
        "Search for: new tariffs imposed, tariff rates changed, trade agreements "
        "signed or withdrawn, Section 301/232 actions, retaliatory tariff threats, "
        "USTR announcements, Commerce Department trade actions directed by the President."
    ),
    DomainType.EXECUTIVE_ORDERS: (
        "Search for: executive orders signed, presidential memoranda, presidential "
        "proclamations, emergency declarations, regulatory freezes. Check the Federal "
        "Register and White House statements."
    ),
    DomainType.PERSONNEL: (
        "Search for: cabinet firings/hirings, agency head appointments, ambassador "
        "nominations, acting officials installed, inspector generals removed, "
        "White House staff changes, military leadership changes directed by the President."
    ),
    DomainType.FOREIGN_POLICY: (
        "Search for: sanctions imposed/lifted, diplomatic meetings, treaty withdrawals, "
        "UN votes directed, military deployments, foreign aid halted/resumed, recognition "
        "of governments, embassy moves, phone calls with foreign leaders."
    ),
    DomainType.LEGISLATIVE: (
        "Search for: bills signed into law, vetoes, pocket vetoes, signing statements, "
        "budget proposals submitted, government shutdowns, debt ceiling actions, "
        "calls for specific legislation."
    ),
    DomainType.PUBLIC_COMMS: (
        "Search for: major policy announcements via Truth Social or press conferences, "
        "public threats against companies/individuals/countries, rally statements with "
        "policy implications, public feuds that led to policy action."
    ),
    DomainType.LEGAL_JUDICIAL: (
        "Search for: pardons, commutations, judicial nominations, Supreme Court picks, "
        "DOJ directives, declassification orders, invocations of executive privilege, "
        "responses to court orders."
    ),
}

DOMAIN_DESCRIPTIONS: dict[DomainType, str] = {
    DomainType.TRADE_TARIFFS: "Trade policy, tariffs, import duties, trade agreements",
    DomainType.EXECUTIVE_ORDERS: "Executive orders, presidential memoranda, proclamations",
    DomainType.PERSONNEL: "Hiring, firing, and appointing government officials",
    DomainType.FOREIGN_POLICY: "Diplomacy, sanctions, military, foreign relations",
    DomainType.LEGISLATIVE: "Bill signings, vetoes, legislative engagement",
    DomainType.PUBLIC_COMMS: "Public statements with direct policy implications",
    DomainType.LEGAL_JUDICIAL: "Pardons, judicial nominations, legal directives",
}


# ── Run Single Domain Agent ───────────────────────────────────────────────────


async def run_domain_agent(
    domain: DomainType,
    leader: str,
    cutoff_date: str,
    today_date: str,
    model_name: str = "gemini-3-flash",
    temperature: float = 0.3,
) -> list[DecisionSeed]:
    """
    Run a single ReAct discovery agent for one domain.

    Returns a list of DecisionSeed objects discovered by the agent.
    """
    tools = get_stage1_tools()

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        response_format=AgentResponse,
    )

    prompt = DISCOVERY_SYSTEM_PROMPT.format(
        leader=leader,
        cutoff_date=cutoff_date,
        today_date=today_date,
        domain_name=domain.value,
        domain_description=DOMAIN_DESCRIPTIONS[domain],
        domain_search_guidance=DOMAIN_SEARCH_GUIDANCE[domain],
    )

    logger.info("Starting discovery agent for domain: %s", domain.value)

    try:
        result = await agent.ainvoke(
            {"messages": [("system", prompt), ("human", "Begin your research now.")]},
        )

        # Extract structured response
        structured: AgentResponse | None = result.get("structured_response")
        if not structured:
            logger.warning("Domain %s: no structured response from agent", domain.value)
            return []

        logger.info(
            "Domain %s: agent discovered %d seeds. Summary: %s",
            domain.value,
            len(structured.seeds),
            structured.search_summary[:200],
        )

        # Convert DiscoveredSeed → DecisionSeed
        seeds: list[DecisionSeed] = []
        for i, ds in enumerate(structured.seeds):
            if ds.confidence == "low":
                logger.info("Skipping low-confidence seed: %s", ds.decision_taken[:80])
                continue

            # Build sources list
            sources = []
            for url, name in zip(ds.source_urls, ds.source_names):
                sources.append(Source(name=name, url=url, date=ds.decision_date))

            seed_id = f"AGENT-{domain.value.upper()}-{i + 1:03d}"
            seed = DecisionSeed(
                seed_id=seed_id,
                event_description=ds.event_description,
                decision_taken=ds.decision_taken,
                decision_date=ds.decision_date,
                simulation_date=ds.simulation_date,
                domain=domain,
                plausible_alternatives=ds.plausible_alternatives,
                sources=sources,
                attribution_evidence=ds.attribution_evidence,
                leader_attributable=(ds.confidence == "high"),
            )
            seeds.append(seed)

        return seeds

    except Exception:
        logger.exception("Domain %s: agent failed", domain.value)
        return []


# ── Merger / Deduplicator ──────────────────────────────────────────────────────


def _similarity(a: str, b: str) -> float:
    """Fuzzy string similarity between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _date_close(d1: str, d2: str, max_days: int = 3) -> bool:
    """Check if two date strings are within max_days of each other."""
    try:
        dt1 = datetime.strptime(d1, "%Y-%m-%d")
        dt2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((dt1 - dt2).days) <= max_days
    except ValueError:
        return False


def _merge_two_seeds(primary: DecisionSeed, secondary: DecisionSeed) -> DecisionSeed:
    """Merge two duplicate seeds — keep the richer one, union sources and alternatives."""
    # Keep the longer (more detailed) description
    if len(secondary.event_description) > len(primary.event_description):
        primary.event_description = secondary.event_description
    if len(secondary.decision_taken) > len(primary.decision_taken):
        primary.decision_taken = secondary.decision_taken

    # Union sources (dedup by URL)
    existing_urls = {s.url for s in primary.sources}
    for s in secondary.sources:
        if s.url and s.url not in existing_urls:
            primary.sources.append(s)
            existing_urls.add(s.url)

    # Union alternatives (dedup by lowercase)
    existing_alts = {a.lower() for a in primary.plausible_alternatives}
    for a in secondary.plausible_alternatives:
        if a.lower() not in existing_alts:
            primary.plausible_alternatives.append(a)
            existing_alts.add(a.lower())

    # Use the strongest attribution evidence
    if len(secondary.attribution_evidence) > len(primary.attribution_evidence):
        primary.attribution_evidence = secondary.attribution_evidence

    # leader_attributable = True if either says True
    primary.leader_attributable = primary.leader_attributable or secondary.leader_attributable

    return primary


def merge_and_dedup(
    all_seeds: list[DecisionSeed],
    description_threshold: float = 0.65,
    decision_threshold: float = 0.70,
) -> list[DecisionSeed]:
    """
    Deduplicate seeds from multiple sources.

    Two seeds are considered duplicates if:
      - Their decision_date is within 3 days AND
      - Their event_description OR decision_taken similarity exceeds threshold

    When merging: keep richest description, union sources, union alternatives.
    """
    if not all_seeds:
        return []

    merged: list[DecisionSeed] = []

    for seed in all_seeds:
        found_match = False
        for i, existing in enumerate(merged):
            if not _date_close(seed.decision_date, existing.decision_date):
                continue

            desc_sim = _similarity(seed.event_description, existing.event_description)
            dec_sim = _similarity(seed.decision_taken, existing.decision_taken)

            if desc_sim >= description_threshold or dec_sim >= decision_threshold:
                merged[i] = _merge_two_seeds(existing, seed)
                found_match = True
                logger.debug(
                    "Merged duplicate: '%s' (desc_sim=%.2f, dec_sim=%.2f)",
                    seed.decision_taken[:60],
                    desc_sim,
                    dec_sim,
                )
                break

        if not found_match:
            merged.append(seed.model_copy(deep=True))

    # Reassign stable seed IDs based on content
    for i, seed in enumerate(merged):
        content_hash = hashlib.sha256(
            f"{seed.decision_taken}:{seed.decision_date}".encode()
        ).hexdigest()[:8]
        seed.seed_id = f"SEED-{seed.domain.value.upper()}-{content_hash}"

    # Ensure "Take no action" is always in alternatives
    for seed in merged:
        has_no_action = any("no action" in a.lower() for a in seed.plausible_alternatives)
        if not has_no_action:
            seed.plausible_alternatives.append("Take no action")

    logger.info(
        "Merger: %d input seeds → %d deduplicated seeds",
        len(all_seeds),
        len(merged),
    )
    return merged


# ── Serialization ──────────────────────────────────────────────────────────────


def save_seeds(seeds: list[DecisionSeed], output_dir: Path) -> Path:
    """Save seeds to a timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"seeds_{timestamp}.json"

    data = {
        "generated_at": datetime.now().isoformat(),
        "total_seeds": len(seeds),
        "domains": {
            d.value: sum(1 for s in seeds if s.domain == d) for d in DomainType
        },
        "seeds": [s.model_dump() for s in seeds],
    }

    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved %d seeds to %s", len(seeds), filepath)
    return filepath


# ── Entry Point ────────────────────────────────────────────────────────────────


async def run_stage1(state: PipelineState) -> PipelineState:
    """
    Stage 1 entry point — run as a LangGraph node.

    1. Run ReAct discovery agents (one per domain, 7 parallel)
    2. Merge + deduplicate all seeds
    3. Serialize to disk
    4. Return updated state
    """
    config = state.config
    cutoff = config.training_cutoff_date
    today = config.today_date
    leader = config.leader

    logger.info(
        "Stage 1 starting — leader=%s, range=[%s, %s]",
        leader, cutoff, today,
    )

    # 1. Run ReAct agents in parallel (one per domain)
    agent_tasks = [
        run_domain_agent(
            domain=domain,
            leader=leader,
            cutoff_date=cutoff,
            today_date=today,
        )
        for domain in DomainType
    ]
    agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

    all_seeds: list[DecisionSeed] = []
    for domain, result in zip(DomainType, agent_results):
        if isinstance(result, list):
            all_seeds.extend(result)
            logger.info("Agent %s: %d seeds", domain.value, len(result))
        else:
            logger.error("Agent %s failed: %s", domain.value, result)

    logger.info("Agents produced %d seeds total", len(all_seeds))

    # 2. Merge + dedup
    state.seeds = merge_and_dedup(all_seeds)

    # 3. Serialize
    save_seeds(state.seeds, SEEDS_DIR)

    logger.info(
        "Stage 1 complete — %d final seeds across %d domains",
        len(state.seeds),
        len({s.domain for s in state.seeds}),
    )

    return state


# ── Debug / Single-Domain Runner ──────────────────────────────────────────────


async def run_stage1_single_domain(
    domain_name: str,
    cutoff_date: str = "2025-01-20",
    today_date: str = "2026-04-02",
    leader: str = "Donald J. Trump",
) -> list[DecisionSeed]:
    """
    Run the discovery agent for a single domain — for testing and debugging.

    Usage:
        python -c "
        import asyncio
        from src.stages.stage1_seeds import run_stage1_single_domain
        seeds = asyncio.run(run_stage1_single_domain('executive_orders'))
        for s in seeds: print(f'{s.seed_id}: {s.decision_taken[:80]}')
        "
    """
    logging.basicConfig(level=logging.INFO)

    domain = DomainType(domain_name)
    seeds = await run_domain_agent(
        domain=domain,
        leader=leader,
        cutoff_date=cutoff_date,
        today_date=today_date,
    )

    save_seeds(seeds, SEEDS_DIR)
    return seeds
