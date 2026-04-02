"""
Pipeline configuration — API keys, model settings, date ranges.

Reads from environment variables (.env file) with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


# ── API Keys ───────────────────────────────────────────────────────────────────


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY", "")  # developer key from congress.gov
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # for web search tools
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # optional, for news fallback


# ── Stage 1 Settings ──────────────────────────────────────────────────────────

SEED_AGENT_MODEL = os.getenv("SEED_AGENT_MODEL", "gemini-3-flash")
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))


# ── Data Source URLs ───────────────────────────────────────────────────────────

FEDERAL_REGISTER_API = "https://www.federalregister.gov/api/v1/documents"
CONGRESS_API = "https://api.congress.gov/v3/bill"
USTR_PRESS_URL = "https://ustr.gov/press-office/press-releases"
OFAC_SDN_URL = "https://sanctionssearch.ofac.treas.gov/"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_API = "https://clob.polymarket.com"


# ── Output Paths ───────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
SEEDS_DIR = OUTPUT_DIR / "seeds"
QUESTIONS_DIR = OUTPUT_DIR / "questions"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
SCORES_DIR = OUTPUT_DIR / "scores"


# ── Ensure output dirs exist ──────────────────────────────────────────────────

for _dir in [SEEDS_DIR, QUESTIONS_DIR, MANIFEST_DIR, PREDICTIONS_DIR, SCORES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Ensure Tavily key is set in env (langchain-tavily reads from os.environ) ─

if TAVILY_API_KEY:
    os.environ.setdefault("TAVILY_API_KEY", TAVILY_API_KEY)
