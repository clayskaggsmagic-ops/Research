"""
Smoke test for the full Stage 1 pipeline.

Mocks both the LLM agent and HTTP accelerators to test the orchestration
logic end-to-end without real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_accel_seed(seed_id: str, domain: DomainType) -> DecisionSeed:
    """Create a mock accelerator seed."""
    return DecisionSeed(
        seed_id=seed_id,
        event_description=f"Accelerator event {seed_id}",
        decision_taken=f"Accelerator decision {seed_id}",
        decision_date="2025-02-15",
        simulation_date="2025-02-10",
        domain=domain,
        plausible_alternatives=["Take no action", "Other option"],
        sources=[Source(name="Federal Register", url=f"https://gov/{seed_id}", date="2025-02-15")],
        attribution_evidence="Published in Federal Register",
        leader_attributable=True,
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

    # Mock accelerators
    mock_fed_seeds = [
        _make_accel_seed("FED-001", DomainType.EXECUTIVE_ORDERS),
        _make_accel_seed("FED-002", DomainType.TRADE_TARIFFS),
    ]
    mock_congress_seeds = [
        _make_accel_seed("CONG-001", DomainType.LEGISLATIVE),
    ]

    # Mock the domain agent to return a fixed response
    async def mock_domain_agent(domain, leader, cutoff_date, today_date, **kwargs):
        return [
            DecisionSeed(
                seed_id=f"AGENT-{domain.value}-001",
                event_description=f"Agent discovered event in {domain.value}",
                decision_taken=f"Decision in {domain.value}",
                decision_date="2025-04-01",
                simulation_date="2025-03-25",
                domain=domain,
                plausible_alternatives=["Take no action", "Other"],
                sources=[Source(name="Reuters", url=f"https://reuters.com/{domain.value}", date="2025-04-01")],
                attribution_evidence="Directed by the President",
                leader_attributable=True,
            )
        ]

    with patch(
        "src.stages.stage1_seeds.fetch_federal_register",
        new_callable=AsyncMock,
        return_value=mock_fed_seeds,
    ), patch(
        "src.stages.stage1_seeds.fetch_congress_bills",
        new_callable=AsyncMock,
        return_value=mock_congress_seeds,
    ), patch(
        "src.stages.stage1_seeds.run_domain_agent",
        side_effect=mock_domain_agent,
    ), patch(
        "src.stages.stage1_seeds.save_seeds",
        return_value=None,
    ):
        result = await run_stage1(state)

    # Should have seeds from accelerators + agents
    assert len(result.seeds) > 0

    # All seeds should be valid DecisionSeed objects
    for seed in result.seeds:
        assert isinstance(seed, DecisionSeed)
        assert seed.decision_date
        assert seed.simulation_date < seed.decision_date
        assert any("no action" in a.lower() for a in seed.plausible_alternatives)

    # Should have multiple domains represented
    domains = {s.domain for s in result.seeds}
    assert len(domains) >= 2  # accelerators + agents cover multiple domains


@pytest.mark.asyncio
async def test_run_stage1_handles_accelerator_failure():
    """Stage 1 should still produce agent seeds even if accelerators fail."""
    state = PipelineState(
        config=PipelineConfig(
            training_cutoff_date="2025-01-20",
            today_date="2026-04-01",
        )
    )

    async def mock_domain_agent(domain, leader, cutoff_date, today_date, **kwargs):
        return [
            DecisionSeed(
                seed_id=f"AGENT-{domain.value}-001",
                event_description=f"Agent event in {domain.value}",
                decision_taken=f"Decision in {domain.value}",
                decision_date="2025-05-01",
                simulation_date="2025-04-25",
                domain=domain,
                plausible_alternatives=["Take no action"],
                sources=[Source(name="AP", url="https://ap.com/test", date="2025-05-01")],
                attribution_evidence="Presidential action",
                leader_attributable=True,
            )
        ]

    with patch(
        "src.stages.stage1_seeds.fetch_federal_register",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    ), patch(
        "src.stages.stage1_seeds.fetch_congress_bills",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    ), patch(
        "src.stages.stage1_seeds.run_domain_agent",
        side_effect=mock_domain_agent,
    ), patch(
        "src.stages.stage1_seeds.save_seeds",
        return_value=None,
    ):
        result = await run_stage1(state)

    # Should still have seeds from the agents
    assert len(result.seeds) > 0
