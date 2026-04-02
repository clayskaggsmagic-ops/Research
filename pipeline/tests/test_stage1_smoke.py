"""
Smoke test for the full Stage 1 pipeline.

Mocks the LLM agent to test orchestration logic end-to-end
without real API calls. The pipeline is fully autonomous —
no hardcoded data sources.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.schemas import (
    DecisionSeed,
    DomainType,
    PipelineConfig,
    PipelineState,
    Source,
)
from src.stages.stage1_seeds import (
    AgentResponse,
    DiscoveredSeed,
    merge_and_dedup,
    run_stage1,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_agent_response(domain: str, n_seeds: int = 2) -> AgentResponse:
    """Create a mock agent response with n seeds."""
    seeds = []
    for i in range(n_seeds):
        seeds.append(
            DiscoveredSeed(
                event_description=f"Test event {i} in {domain}",
                decision_taken=f"Test decision {i} in {domain}",
                decision_date=f"2025-0{(i % 9) + 1}-15",
                simulation_date=f"2025-0{(i % 9) + 1}-10",
                plausible_alternatives=["Take no action", f"Alternative for {domain}"],
                attribution_evidence="Signed by the President",
                source_urls=[f"https://example.com/{domain}/{i}"],
                source_names=["Test Source"],
                confidence="high",
            )
        )
    return AgentResponse(
        domain=domain,
        seeds=seeds,
        search_summary=f"Found {n_seeds} decisions in {domain}",
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_stage1_produces_seeds():
    """Verify that run_stage1 produces a non-empty, deduplicated seed list."""
    state = PipelineState(
        config=PipelineConfig(
            training_cutoff_date="2025-01-20",
            today_date="2026-04-01",
        )
    )

    # Mock the domain agent to return a fixed response (unique per domain)
    domain_idx = {d: i for i, d in enumerate(DomainType)}

    async def mock_domain_agent(domain, leader, cutoff_date, today_date, **kwargs):
        idx = domain_idx.get(domain, 0)
        month = f"{(idx % 9) + 2:02d}"  # 02 through 08 — unique per domain
        return [
            DecisionSeed(
                seed_id=f"AGENT-{domain.value}-001",
                event_description=f"Unique {domain.value} event: specific action #{idx}",
                decision_taken=f"Unique decision for {domain.value} domain #{idx}",
                decision_date=f"2025-{month}-15",
                simulation_date=f"2025-{month}-10",
                domain=domain,
                plausible_alternatives=["Take no action", "Other"],
                sources=[Source(name="Reuters", url=f"https://reuters.com/{domain.value}", date=f"2025-{month}-15")],
                attribution_evidence=f"Directed by the President — {domain.value} specific",
                leader_attributable=True,
            )
        ]

    with patch(
        "src.stages.stage1_seeds.run_domain_agent",
        side_effect=mock_domain_agent,
    ), patch(
        "src.stages.stage1_seeds.save_seeds",
        return_value=None,
    ):
        result = await run_stage1(state)

    # Should have seeds from agents
    assert len(result.seeds) > 0

    # All seeds should be valid DecisionSeed objects
    for seed in result.seeds:
        assert isinstance(seed, DecisionSeed)
        assert seed.decision_date
        assert seed.simulation_date < seed.decision_date
        assert any("no action" in a.lower() for a in seed.plausible_alternatives)

    # Should have multiple domains represented (7 agents = 7 domains)
    domains = {s.domain for s in result.seeds}
    assert len(domains) >= 2


@pytest.mark.asyncio
async def test_run_stage1_handles_agent_failure():
    """Stage 1 should still produce seeds from surviving agents if some fail."""
    state = PipelineState(
        config=PipelineConfig(
            training_cutoff_date="2025-01-20",
            today_date="2026-04-01",
        )
    )

    call_count = 0
    surviving_domains = []

    async def mock_domain_agent_partial_failure(domain, leader, cutoff_date, today_date, **kwargs):
        nonlocal call_count
        call_count += 1
        # Fail on the first 2 domains, succeed on the rest
        if call_count <= 2:
            raise Exception("Agent crashed")
        surviving_domains.append(domain)
        month = f"{call_count + 1:02d}"  # unique month per surviving agent
        return [
            DecisionSeed(
                seed_id=f"AGENT-{domain.value}-001",
                event_description=f"Unique {domain.value} event for partial test #{call_count}",
                decision_taken=f"Unique decision in {domain.value} #{call_count}",
                decision_date=f"2025-{month}-01",
                simulation_date=f"2025-{month.replace(month, f'{int(month)-1:02d}')}-25",
                domain=domain,
                plausible_alternatives=["Take no action"],
                sources=[Source(name="AP", url=f"https://ap.com/{domain.value}", date=f"2025-{month}-01")],
                attribution_evidence=f"Presidential action — {domain.value}",
                leader_attributable=True,
            )
        ]

    with patch(
        "src.stages.stage1_seeds.run_domain_agent",
        side_effect=mock_domain_agent_partial_failure,
    ), patch(
        "src.stages.stage1_seeds.save_seeds",
        return_value=None,
    ):
        result = await run_stage1(state)

    # Should still have seeds from the 5 surviving agents
    assert len(result.seeds) > 0
    assert len(result.seeds) >= 3  # at least 3 of the 5 surviving domains survive dedup
