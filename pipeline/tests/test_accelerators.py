"""
Tests for Stage 1 data source accelerators (Federal Register + Congress.gov).

Uses httpx mock responses — no real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.accelerators import fetch_congress_bills, fetch_federal_register
from src.schemas import DomainType


# ── Fixtures ───────────────────────────────────────────────────────────────────


SAMPLE_FED_REG_RESPONSE = {
    "count": 2,
    "results": [
        {
            "title": "Imposing Duties To Address the Synthetic Opioid Supply Chain",
            "type": "Presidential Document",
            "subtype": "Executive Order",
            "presidential_document_type_id": "executive_order",
            "executive_order_number": 14195,
            "signing_date": "2025-02-04",
            "publication_date": "2025-02-07",
            "html_url": "https://www.federalregister.gov/documents/2025/02/07/example",
            "abstract": "The President signed an executive order imposing additional duties on imports from China to address the flow of synthetic opioids.",
            "document_number": "2025-02345",
        },
        {
            "title": "Reorganizing the National Security Council",
            "type": "Presidential Document",
            "subtype": "Presidential Memorandum",
            "presidential_document_type_id": "memorandum",
            "executive_order_number": None,
            "signing_date": "2025-01-28",
            "publication_date": "2025-01-30",
            "html_url": "https://www.federalregister.gov/documents/2025/01/30/example2",
            "abstract": "Presidential memorandum reorganizing the structure of the National Security Council.",
            "document_number": "2025-01234",
        },
    ],
    "next_page_url": None,
}


SAMPLE_CONGRESS_RESPONSE = {
    "bills": [
        {
            "title": "Laken Riley Act",
            "type": "HR",
            "number": "29",
            "congress": 119,
            "updateDate": "2025-01-29",
            "url": "https://congress.gov/bill/119th-congress/house-bill/29",
            "latestAction": {
                "text": "Signed by President.",
                "actionDate": "2025-01-29",
            },
        },
        {
            "title": "National Defense Authorization Act",
            "type": "S",
            "number": "100",
            "congress": 119,
            "updateDate": "2025-03-15",
            "url": "https://congress.gov/bill/119th-congress/senate-bill/100",
            "latestAction": {
                "text": "Referred to committee.",
                "actionDate": "2025-03-15",
            },
        },
    ],
    "pagination": {"count": 2},
}


# ── Federal Register Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_federal_register_parses_results():
    """Verify correct parsing of Federal Register API response."""
    mock_response = httpx.Response(
        status_code=200,
        json=SAMPLE_FED_REG_RESPONSE,
        request=httpx.Request("GET", "https://test"),
    )

    with patch("src.accelerators.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        seeds = await fetch_federal_register("2025-01-20", "2025-12-31")

    assert len(seeds) == 2

    # First seed — executive order
    eo_seed = seeds[0]
    assert eo_seed.seed_id == "FED-REG-EO-14195"
    assert eo_seed.decision_date == "2025-02-04"
    assert eo_seed.domain == DomainType.EXECUTIVE_ORDERS
    assert eo_seed.leader_attributable is True
    assert len(eo_seed.sources) == 1
    assert "federalregister.gov" in eo_seed.sources[0].url
    assert len(eo_seed.plausible_alternatives) >= 2

    # Simulation date should be before decision date
    assert eo_seed.simulation_date < eo_seed.decision_date


@pytest.mark.asyncio
async def test_fetch_federal_register_handles_api_error():
    """Verify graceful handling of API errors."""
    mock_response = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("GET", "https://test"),
    )

    with patch("src.accelerators.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        seeds = await fetch_federal_register("2025-01-20", "2025-12-31")

    assert seeds == []


# ── Congress.gov Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_congress_filters_presidential_actions():
    """Verify that only bills with presidential actions are included."""
    mock_response = httpx.Response(
        status_code=200,
        json=SAMPLE_CONGRESS_RESPONSE,
        request=httpx.Request("GET", "https://test"),
    )

    with patch("src.accelerators.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        with patch("src.accelerators.CONGRESS_API_KEY", "test-key"):
            seeds = await fetch_congress_bills("2025-01-20", "2025-12-31")

    # Only the signed bill should be included, not the committee referral
    assert len(seeds) == 1
    assert seeds[0].seed_id == "CONGRESS-119-HR29"
    assert seeds[0].domain == DomainType.LEGISLATIVE
    assert "Signed into law" in seeds[0].decision_taken


@pytest.mark.asyncio
async def test_fetch_congress_skips_without_api_key():
    """Verify graceful skip when no API key is configured."""
    with patch("src.accelerators.CONGRESS_API_KEY", ""):
        seeds = await fetch_congress_bills("2025-01-20", "2025-12-31")

    assert seeds == []


# ── Domain Auto-Tagging Tests ─────────────────────────────────────────────────


def test_auto_tag_executive_order():
    """Executive order doc type should tag as EXECUTIVE_ORDERS."""
    from src.accelerators import _auto_tag_domain

    assert _auto_tag_domain("Anything", "executive_order") == DomainType.EXECUTIVE_ORDERS


def test_auto_tag_tariff_keyword():
    """Tariff-related titles should tag as TRADE_TARIFFS."""
    from src.accelerators import _auto_tag_domain

    assert _auto_tag_domain("Imposing Tariffs on Steel Imports", "") == DomainType.TRADE_TARIFFS


def test_auto_tag_personnel_keyword():
    """Personnel-related titles should tag as PERSONNEL."""
    from src.accelerators import _auto_tag_domain

    assert _auto_tag_domain("Appointing John Doe as Ambassador", "") == DomainType.PERSONNEL
