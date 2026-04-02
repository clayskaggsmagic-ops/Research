"""
Structured Data Source Accelerators — pure code (no LLM) fetchers.

These supplement the autonomous ReAct discovery agent with guaranteed-complete
structured data from official government APIs. Their output feeds into the
same merger/dedup step as the agent-discovered seeds.

Sources:
  - Federal Register API (executive orders, presidential proclamations)
  - Congress.gov API (bills presented to the president)
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta

import httpx

from src.config import CONGRESS_API_KEY, CONGRESS_API, FEDERAL_REGISTER_API
from src.schemas import DecisionSeed, DomainType, Source

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _random_simulation_date(decision_date_str: str) -> str:
    """Generate a simulation_date 1-30 days before the decision_date."""
    dt = datetime.strptime(decision_date_str, "%Y-%m-%d")
    offset = random.randint(1, 30)
    sim = dt - timedelta(days=offset)
    return sim.strftime("%Y-%m-%d")


def _auto_tag_domain(title: str, doc_type: str) -> DomainType:
    """Heuristic domain tagging based on document title and type."""
    title_lower = title.lower()

    if doc_type == "executive_order":
        return DomainType.EXECUTIVE_ORDERS

    # Keyword-based fallback
    trade_keywords = ["tariff", "trade", "import", "export", "duties", "commerce", "usmca"]
    personnel_keywords = ["appoint", "nominat", "firing", "resign", "dismiss", "vacancy"]
    foreign_keywords = [
        "sanction", "diplomat", "treaty", "nato", "foreign", "military",
        "deploy", "withdraw", "ambassador", "bilateral",
    ]
    legislative_keywords = ["sign", "veto", "bill", "act of congress", "legislation"]
    legal_keywords = ["pardon", "commut", "judicial", "court", "judge", "clemency"]
    comms_keywords = ["statement", "remarks", "press conference", "truth social", "tweet"]

    for kw in trade_keywords:
        if kw in title_lower:
            return DomainType.TRADE_TARIFFS
    for kw in personnel_keywords:
        if kw in title_lower:
            return DomainType.PERSONNEL
    for kw in foreign_keywords:
        if kw in title_lower:
            return DomainType.FOREIGN_POLICY
    for kw in legislative_keywords:
        if kw in title_lower:
            return DomainType.LEGISLATIVE
    for kw in legal_keywords:
        if kw in title_lower:
            return DomainType.LEGAL_JUDICIAL
    for kw in comms_keywords:
        if kw in title_lower:
            return DomainType.PUBLIC_COMMS

    # Default for presidential documents
    return DomainType.EXECUTIVE_ORDERS


# ── Federal Register Fetcher ──────────────────────────────────────────────────


async def fetch_federal_register(
    cutoff_date: str,
    today_date: str,
    per_page: int = 100,
) -> list[DecisionSeed]:
    """
    Fetch executive orders and presidential proclamations from the Federal
    Register API. No API key required.

    Endpoint: https://www.federalregister.gov/api/v1/documents.json
    Filters: type=PRESDOCU, president=donald-trump, date range
    """
    seeds: list[DecisionSeed] = []
    page = 1

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            params = {
                "conditions[type]": "PRESDOCU",
                "conditions[president]": "donald-trump",
                "conditions[publication_date][gte]": cutoff_date,
                "conditions[publication_date][lte]": today_date,
                "fields[]": [
                    "title",
                    "type",
                    "subtype",
                    "presidential_document_type_id",
                    "executive_order_number",
                    "signing_date",
                    "publication_date",
                    "html_url",
                    "abstract",
                    "document_number",
                ],
                "per_page": per_page,
                "page": page,
                "order": "newest",
            }

            try:
                resp = await client.get(
                    f"{FEDERAL_REGISTER_API}.json",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                logger.error("Federal Register API error (page %d): %s", page, e)
                break

            results = data.get("results", [])
            if not results:
                break

            for doc in results:
                title = doc.get("title", "Untitled")
                doc_subtype = doc.get("presidential_document_type_id") or ""
                signing_date = doc.get("signing_date") or doc.get("publication_date", "")
                eo_number = doc.get("executive_order_number")
                abstract = doc.get("abstract", "")

                # Build seed ID
                if eo_number:
                    seed_id = f"FED-REG-EO-{eo_number}"
                else:
                    seed_id = f"FED-REG-{doc.get('document_number', page)}"

                decision_date = signing_date[:10] if signing_date else today_date

                seed = DecisionSeed(
                    seed_id=seed_id,
                    event_description=abstract or title,
                    decision_taken=title,
                    decision_date=decision_date,
                    simulation_date=_random_simulation_date(decision_date),
                    domain=_auto_tag_domain(title, doc_subtype),
                    plausible_alternatives=[
                        "Take no action",
                        "Issue a different executive order",
                        "Delegate to agency rulemaking",
                        "Pursue legislative route instead",
                    ],
                    sources=[
                        Source(
                            name="Federal Register",
                            url=doc.get("html_url", ""),
                            date=decision_date,
                        )
                    ],
                    attribution_evidence=(
                        "Presidential document published in the Federal Register, "
                        "signed by the President."
                    ),
                    leader_attributable=True,
                )
                seeds.append(seed)

            # Pagination
            next_url = data.get("next_page_url")
            if not next_url or page * per_page >= data.get("count", 0):
                break
            page += 1

    logger.info("Federal Register: fetched %d seeds", len(seeds))
    return seeds


# ── Congress.gov Fetcher ───────────────────────────────────────────────────────


async def fetch_congress_bills(
    cutoff_date: str,
    today_date: str,
    limit: int = 100,
) -> list[DecisionSeed]:
    """
    Fetch bills presented to the president (enrolled/signed/vetoed) from
    the Congress.gov API.

    Requires CONGRESS_API_KEY in environment.
    Endpoint: https://api.congress.gov/v3/bill
    """
    if not CONGRESS_API_KEY:
        logger.warning("CONGRESS_API_KEY not set — skipping Congress.gov fetcher")
        return []

    seeds: list[DecisionSeed] = []
    offset = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            params = {
                "api_key": CONGRESS_API_KEY,
                "format": "json",
                "fromDateTime": f"{cutoff_date}T00:00:00Z",
                "toDateTime": f"{today_date}T23:59:59Z",
                "limit": limit,
                "offset": offset,
                "sort": "updateDate+desc",
            }

            try:
                # Search for enrolled bills (presented to president)
                resp = await client.get(CONGRESS_API, params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                logger.error("Congress.gov API error (offset %d): %s", offset, e)
                break

            bills = data.get("bills", [])
            if not bills:
                break

            for bill in bills:
                title = bill.get("title", "Untitled Bill")
                bill_type = bill.get("type", "")
                bill_number = bill.get("number", "")
                congress = bill.get("congress", "")
                update_date = bill.get("updateDate", "")[:10]

                # Check if bill has reached the president
                latest_action = bill.get("latestAction", {})
                action_text = latest_action.get("text", "").lower()
                action_date = latest_action.get("actionDate", update_date)

                presidential_actions = [
                    "became public law",
                    "signed by president",
                    "vetoed",
                    "pocket vetoed",
                    "presented to president",
                ]
                is_presidential = any(pa in action_text for pa in presidential_actions)
                if not is_presidential:
                    continue

                # Determine the decision taken
                if "vetoed" in action_text:
                    decision = f"Vetoed {bill_type}.{bill_number}: {title}"
                elif "signed" in action_text or "became public law" in action_text:
                    decision = f"Signed into law {bill_type}.{bill_number}: {title}"
                else:
                    decision = f"Presented with {bill_type}.{bill_number}: {title}"

                decision_date = action_date[:10] if action_date else update_date

                seed = DecisionSeed(
                    seed_id=f"CONGRESS-{congress}-{bill_type}{bill_number}",
                    event_description=(
                        f"Congress presented {bill_type}.{bill_number} ({title}) "
                        f"to President Trump. Latest action: {action_text}"
                    ),
                    decision_taken=decision,
                    decision_date=decision_date,
                    simulation_date=_random_simulation_date(decision_date),
                    domain=DomainType.LEGISLATIVE,
                    plausible_alternatives=[
                        "Take no action",
                        "Sign the bill into law",
                        "Veto the bill",
                        "Pocket veto (let it expire unsigned)",
                        "Sign with a signing statement expressing reservations",
                    ],
                    sources=[
                        Source(
                            name="Congress.gov",
                            url=bill.get("url", f"https://congress.gov/bill/{congress}"),
                            date=decision_date,
                        )
                    ],
                    attribution_evidence=(
                        "Bill presented to the President for signature or veto — "
                        "a constitutionally defined presidential action."
                    ),
                    leader_attributable=True,
                )
                seeds.append(seed)

            # Pagination
            pagination = data.get("pagination", {})
            total = pagination.get("count", 0)
            if offset + limit >= total:
                break
            offset += limit

    logger.info("Congress.gov: fetched %d seeds", len(seeds))
    return seeds
