"""CHRONOS Extraction Agent — article fetching and LLM-powered fact extraction.

The Extraction Agent is the "hands" of the swarm. It:
1. Fetches full article text from URLs found by Discovery
2. Uses an LLM to extract structured event data (dates, quotes, facts)
3. Filters out opinion pieces and stubs
4. Outputs ExtractionResult objects for downstream validation

It NEVER validates temporal integrity — that's the Temporal Validator's job.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import settings
from ..models import DirectQuote, ExtractionResult, RawEventCandidate, SwarmState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXTRACTION_CHUNK_SIZE = 60  # Candidates per concurrent chunk
EXTRACTION_MAX_CHUNKS_PER_STEP = 6  # Up to 6 chunks per node invocation (= 360 candidates max)
EXTRACTION_DRAIN_WATERMARK = 80  # Stop draining when queue falls below this
FETCH_TIMEOUT = 15.0  # Seconds timeout for HTTP requests
MIN_ARTICLE_WORDS = 100  # Minimum word count to not be a stub
EXTRACTION_CONCURRENCY = 15  # Max concurrent fetch+LLM extractions per chunk


# ---------------------------------------------------------------------------
# Article fetcher
# ---------------------------------------------------------------------------

# Common user-agent to avoid bot detection
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Patterns that indicate a paywall or access block
_PAYWALL_SIGNALS = [
    "subscribe to continue reading",
    "you've reached your free article limit",
    "this content is for subscribers",
    "sign in to read",
    "create a free account",
    "premium content",
    "to continue reading, please",
]


async def fetch_article(url: str) -> dict:
    """Fetch and parse article content from a URL.

    Returns a dict with:
        - text: cleaned article text
        - pub_date: detected publish date (or None)
        - author: detected author (or "")
        - success: bool
        - failure_reason: str if failed
    """
    headers = {"User-Agent": _USER_AGENT}

    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        try:
            response = await client.get(url)
        except httpx.TimeoutException:
            return {"text": "", "pub_date": None, "author": "", "success": False, "failure_reason": "timeout"}
        except httpx.RequestError as e:
            return {"text": "", "pub_date": None, "author": "", "success": False, "failure_reason": str(e)}

    if response.status_code == 404:
        return {"text": "", "pub_date": None, "author": "", "success": False, "failure_reason": "404_not_found"}
    if response.status_code >= 400:
        return {"text": "", "pub_date": None, "author": "", "success": False, "failure_reason": f"http_{response.status_code}"}

    # --- Parse HTML with BeautifulSoup ---
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, nav, footer elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    # Try to find article body (common selectors)
    article_body = (
        soup.find("article")
        or soup.find("div", class_="article-body")
        or soup.find("div", class_="story-body")
        or soup.find("div", {"role": "article"})
        or soup.find("main")
    )

    if article_body:
        text = article_body.get_text(separator="\n", strip=True)
    else:
        # Fallback: get all paragraph text
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)

    # Check for paywall signals
    text_lower = text.lower()
    for signal in _PAYWALL_SIGNALS:
        if signal in text_lower and len(text.split()) < 200:
            return {"text": "", "pub_date": None, "author": "", "success": False, "failure_reason": "paywall_detected"}

    # --- Extract publish date from meta tags ---
    pub_date = None
    date_meta_names = [
        "article:published_time", "datePublished", "date",
        "DC.date.issued", "sailthru.date", "pubdate",
    ]
    for meta_name in date_meta_names:
        meta = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
        if meta and meta.get("content"):
            pub_date = _parse_meta_date(meta["content"])
            if pub_date:
                break

    # Also check time tag
    if not pub_date:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            pub_date = _parse_meta_date(time_tag["datetime"])

    # --- Extract author ---
    author = ""
    author_meta = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
    if author_meta and author_meta.get("content"):
        author = author_meta["content"]

    return {
        "text": text,
        "pub_date": pub_date,
        "author": author,
        "success": True,
        "failure_reason": "",
    }


def _parse_meta_date(date_str: str) -> date | None:
    """Try to parse a date from HTML meta tag content."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(date_str.strip()[:25], fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# LLM-powered fact extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a precision fact extractor for the CHRONOS temporal knowledge base.
Your job is to read a news article and extract STRUCTURED EVENT DATA.

CRITICAL RULES FOR DATES:
1. Extract the date the EVENT HAPPENED, not the date the article was published.
   These are often different. If article says "signed on Tuesday" and was published
   Wednesday Jan 15, the event date is Tuesday Jan 14.
2. If you cannot determine a specific event date, set "event_date_ambiguous" to true
   and set "event_date" to null. DO NOT GUESS.
3. Dates from phrases like "last week", "recently", "in recent days" are AMBIGUOUS.
   Only non-ambiguous if the article states a specific date like "January 14" or "on March 3rd".

CRITICAL RULES FOR CONTENT:
1. Strip ALL editorial opinion. Include only verifiable facts.
2. The "headline" should describe WHAT HAPPENED, not the article's actual headline.
   Example: "Trump Signs Executive Order Banning TikTok" not "The Controversial Decision That Shook Silicon Valley"
3. The "summary" must be 100-200 words of pure facts. No commentary.
4. "key_facts" are specific, hard data points: dollar amounts, percentages, vote counts,
   legal citation numbers, named individuals, etc.
5. "topics" are categorical tags for the event.
6. "is_opinion" should be true if the article is primarily opinion, analysis, or editorial.

OUTPUT FORMAT (JSON):
{
  "headline": "One-line factual description of the event",
  "event_date": "YYYY-MM-DD" or null,
  "event_date_ambiguous": true/false,
  "summary": "100-200 word factual summary",
  "key_facts": ["fact 1", "fact 2"],
  "quotes": [
    {"speaker": "Name", "quote": "Exact quote", "context": "Setting/occasion"}
  ],
  "topics": ["tag1", "tag2"],
  "is_opinion": false
}

WORKED EXAMPLE:
Article (published Jan 15, 2025): "President Trump on Tuesday signed an executive order
directing the Department of Energy to expedite permits for new nuclear power plants.
The order, EO 14178, sets a 90-day deadline for the Nuclear Regulatory Commission to
streamline its review process. 'We're going to build the most beautiful power plants
you've ever seen,' Trump said at the White House signing ceremony. The order affects
approximately 30 pending applications worth an estimated $45 billion."

Correct extraction:
{
  "headline": "Trump Signs Executive Order Expediting Nuclear Power Plant Permits",
  "event_date": "2025-01-14",
  "event_date_ambiguous": false,
  "summary": "President Trump signed Executive Order 14178 directing the Department of Energy to expedite permits for new nuclear power plants. The order sets a 90-day deadline for the Nuclear Regulatory Commission to streamline its review process for pending applications. The signing took place at a White House ceremony.",
  "key_facts": [
    "Executive Order 14178",
    "90-day deadline for NRC review streamlining",
    "30 pending applications affected",
    "Estimated $45 billion in affected projects"
  ],
  "quotes": [
    {
      "speaker": "Donald Trump",
      "quote": "We're going to build the most beautiful power plants you've ever seen",
      "context": "White House signing ceremony for EO 14178"
    }
  ],
  "topics": ["executive_actions", "energy_policy", "nuclear"],
  "is_opinion": false
}"""


def _get_llm() -> ChatGoogleGenerativeAI:
    """Get the LLM for extraction."""
    return ChatGoogleGenerativeAI(
        model=settings.research_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,  # Very low — we want deterministic factual extraction
        timeout=120,
        max_retries=2,
    )


async def extract_facts_from_article(
    article_text: str,
    article_url: str,
    pub_date: date | None = None,
) -> ExtractionResult:
    """Use an LLM to extract structured event data from article text.

    Args:
        article_text: The full text of the article.
        article_url: The source URL (for logging).
        pub_date: The article's publish date if detected from meta tags.

    Returns:
        ExtractionResult with extracted facts or failure info.
    """
    word_count = len(article_text.split())

    # Reject stubs
    if word_count < MIN_ARTICLE_WORDS:
        return ExtractionResult(
            url=article_url,
            word_count=word_count,
            extraction_success=False,
            failure_reason=f"article_too_short ({word_count} words, need {MIN_ARTICLE_WORDS})",
        )

    # Truncate very long articles to save context
    truncated = article_text[:12000] if len(article_text) > 12000 else article_text

    llm = _get_llm()

    context_note = ""
    if pub_date:
        context_note = f"\n\nIMPORTANT CONTEXT: This article was published on {pub_date.isoformat()}. Use this to resolve relative dates like 'yesterday', 'last Tuesday', etc."

    prompt = (
        f"Extract structured event data from the following article.\n\n"
        f"SOURCE URL: {article_url}{context_note}\n\n"
        f"---ARTICLE TEXT---\n{truncated}\n---END ARTICLE TEXT---\n\n"
        f"Return ONLY the JSON object, no other text."
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        data = json.loads(content)

    except json.JSONDecodeError as e:
        logger.error(f"Extraction JSON parse error for {article_url}: {e}")
        return ExtractionResult(
            url=article_url,
            full_text=truncated,
            word_count=word_count,
            extraction_success=False,
            failure_reason=f"llm_json_parse_error: {e}",
        )
    except Exception as e:
        logger.error(f"Extraction LLM error for {article_url}: {e}")
        return ExtractionResult(
            url=article_url,
            full_text=truncated,
            word_count=word_count,
            extraction_success=False,
            failure_reason=f"llm_error: {e}",
        )

    # --- Parse the LLM output into ExtractionResult ---

    # Parse event date
    event_date = None
    raw_date = data.get("event_date")
    if raw_date:
        try:
            event_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

    # Parse quotes
    quotes = []
    for q in data.get("quotes", []):
        if isinstance(q, dict) and q.get("quote"):
            quotes.append(DirectQuote(
                speaker=q.get("speaker", "Unknown"),
                quote=q["quote"],
                context=q.get("context", ""),
            ))

    # Build result
    summary = data.get("summary", "")
    is_opinion = data.get("is_opinion", False)

    # Final quality filter: reject opinion pieces
    if is_opinion:
        return ExtractionResult(
            url=article_url,
            headline=data.get("headline", ""),
            summary=summary,
            full_text=truncated,
            pub_date=pub_date,
            event_date=event_date,
            event_date_ambiguous=data.get("event_date_ambiguous", False),
            quotes=quotes,
            key_facts=data.get("key_facts", []),
            topics=data.get("topics", []),
            is_opinion=True,
            word_count=word_count,
            extraction_success=False,
            failure_reason="opinion_piece_rejected",
        )

    # Reject if summary is empty
    if not summary:
        return ExtractionResult(
            url=article_url,
            word_count=word_count,
            extraction_success=False,
            failure_reason="empty_summary_after_extraction",
        )

    return ExtractionResult(
        url=article_url,
        headline=data.get("headline", ""),
        summary=summary,
        full_text=truncated,
        pub_date=pub_date,
        event_date=event_date,
        event_date_ambiguous=data.get("event_date_ambiguous", False),
        quotes=quotes,
        key_facts=data.get("key_facts", []),
        topics=data.get("topics", []),
        is_opinion=False,
        word_count=word_count,
        extraction_success=True,
        failure_reason="",
    )


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

async def _extract_single_candidate(
    candidate: RawEventCandidate,
    sem: asyncio.Semaphore,
) -> tuple[ExtractionResult, str | None]:
    """Fetch and extract a single candidate behind a semaphore.

    Returns (ExtractionResult, error_msg_or_None).
    """
    async with sem:
        # --- Step 1: Fetch article ---
        fetch_result = await fetch_article(candidate.url)

        if not fetch_result["success"]:
            error = f"Fetch failed for {candidate.url}: {fetch_result['failure_reason']}"
            return ExtractionResult(
                url=candidate.url,
                extraction_success=False,
                failure_reason=f"fetch_failed: {fetch_result['failure_reason']}",
            ), error

        # --- Step 2: LLM extraction ---
        pub_date = fetch_result["pub_date"] or candidate.preliminary_date

        result = await extract_facts_from_article(
            article_text=fetch_result["text"],
            article_url=candidate.url,
            pub_date=pub_date,
        )

        if result.extraction_success:
            return result, None
        else:
            error = (
                f"Extraction failed for {candidate.url}: {result.failure_reason}"
                if result.failure_reason
                else None
            )
            return result, error


async def extraction_node(state: SwarmState) -> SwarmState:
    """LangGraph node: fetch articles and extract structured event data.

    Runs up to EXTRACTION_CONCURRENCY candidates in parallel for ~8x speedup.
    Takes a batch of RawEventCandidates, fetches each article, runs LLM
    extraction, and deposits ExtractionResults into the state.
    """
    if not state.raw_candidates:
        logger.info("Extraction: no raw candidates to process")
        return state

    logger.info(
        f"Extraction: draining candidate queue ({len(state.raw_candidates)} pending, "
        f"up to {EXTRACTION_MAX_CHUNKS_PER_STEP} chunks × {EXTRACTION_CHUNK_SIZE})"
    )

    sem = asyncio.Semaphore(EXTRACTION_CONCURRENCY)
    total_success = 0
    total_fail = 0

    for chunk_idx in range(EXTRACTION_MAX_CHUNKS_PER_STEP):
        if not state.raw_candidates:
            break

        chunk_size = min(EXTRACTION_CHUNK_SIZE, len(state.raw_candidates))
        chunk = state.raw_candidates[:chunk_size]
        state.raw_candidates = state.raw_candidates[chunk_size:]

        logger.info(
            f"  Chunk {chunk_idx + 1}/{EXTRACTION_MAX_CHUNKS_PER_STEP}: "
            f"{len(chunk)} candidates (remaining after: {len(state.raw_candidates)})"
        )

        # --- Parallel fetch + extract for this chunk ---
        tasks = [_extract_single_candidate(c, sem) for c in chunk]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for outcome in results:
            if isinstance(outcome, BaseException):
                total_fail += 1
                state.errors.append(f"Extraction task exception: {outcome}")
                continue

            result, error = outcome
            state.extraction_results.append(result)

            if result.extraction_success:
                total_success += 1
            else:
                total_fail += 1

            if error:
                state.errors.append(error)

        # Stop draining once queue is below watermark — let downstream catch up
        if len(state.raw_candidates) < EXTRACTION_DRAIN_WATERMARK:
            logger.info(
                f"  Queue below watermark ({len(state.raw_candidates)} < {EXTRACTION_DRAIN_WATERMARK}) "
                f"— stopping drain"
            )
            break

    rate = (total_success / (total_success + total_fail) * 100) if (total_success + total_fail) > 0 else 0
    logger.info(
        f"Extraction complete: {total_success} succeeded, {total_fail} failed "
        f"({rate:.0f}% success rate, total results: {len(state.extraction_results)}, "
        f"queue remaining: {len(state.raw_candidates)})"
    )

    return state
