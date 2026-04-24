"""
Briefing providers — the pluggable source of intelligence-briefing text
shown to the model.

A briefing provider takes a question dict + model_id and returns
(briefing_text_or_None, briefing_hash_or_None). Implementations:

  * NoBriefing           — E3 (and E5 when web search substitutes)
  * ChronosBroad(top_k)  — E1 (top_k=15) and E1′ (top_k=8)
  * ChronosRefined()     — E2 (two-stage retrieval; see refined_retrieval.py)

Each provider also caches to disk under `briefing_cache_dir/<variant>/<qid>.json`
so runs are resumable and E4 can literally reuse E1's briefings via the hash.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from evaluation_plan.src.io_utils import sha256_short


# ── Base class ────────────────────────────────────────────────────────────────


class BriefingProvider(ABC):
    name: str  # "none", "chronos_broad_15", "chronos_broad_8", "chronos_refined"

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @abstractmethod
    def get(self, question: dict, model_id: str) -> tuple[str | None, str | None]:
        """Return (briefing_text, briefing_hash). Either may be None (no briefing)."""
        ...

    async def aget(self, question: dict, model_id: str) -> tuple[str | None, str | None]:
        """Async variant. Default: reuse sync `get()`. Subclasses that touch the
        database MUST override so a single event loop can batch them without
        cross-loop asyncpg pool errors."""
        return self.get(question, model_id)

    # ── Disk cache ──
    def _cache_path(self, question_id: str) -> Path | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / self.name / f"{question_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_cached(self, question_id: str) -> tuple[str, str] | None:
        p = self._cache_path(question_id)
        if p and p.exists():
            data = json.loads(p.read_text())
            return data["text"], data["hash"]
        return None

    def _save_cached(self, question_id: str, text: str, h: str, meta: dict | None = None) -> None:
        p = self._cache_path(question_id)
        if p is None:
            return
        p.write_text(json.dumps({"text": text, "hash": h, "meta": meta or {}}, indent=2))


# ── NoBriefing ────────────────────────────────────────────────────────────────


class NoBriefing(BriefingProvider):
    name = "none"

    def get(self, question: dict, model_id: str) -> tuple[str | None, str | None]:
        return None, None


# ── Chronos broad ─────────────────────────────────────────────────────────────


class ChronosBroad(BriefingProvider):
    """Vanilla CHRONOS retrieval: single query = question_text, top_k events."""

    def __init__(self, top_k: int, cache_dir: str | Path | None = None) -> None:
        super().__init__(cache_dir=cache_dir)
        self.top_k = top_k
        self.name = f"chronos_broad_{top_k}"

    def get(self, question: dict, model_id: str) -> tuple[str | None, str | None]:
        cached = self._load_cached(question["question_id"])
        if cached:
            return cached
        briefing = asyncio.run(self._retrieve(question, model_id))
        h = sha256_short(briefing)
        self._save_cached(
            question["question_id"],
            briefing,
            h,
            meta={"variant": self.name, "top_k": self.top_k, "model_id": model_id},
        )
        return briefing, h

    async def aget(self, question: dict, model_id: str) -> tuple[str | None, str | None]:
        cached = self._load_cached(question["question_id"])
        if cached:
            return cached
        briefing = await self._retrieve(question, model_id)
        h = sha256_short(briefing)
        self._save_cached(
            question["question_id"],
            briefing,
            h,
            meta={"variant": self.name, "top_k": self.top_k, "model_id": model_id},
        )
        return briefing, h

    async def _retrieve(self, question: dict, model_id: str) -> str:
        # Lazy import so this module is importable without CHRONOS deps.
        from temporal_knowledge_base.src.retrieval import retrieve  # type: ignore

        sim_date = date.fromisoformat(question["simulation_date"])
        return await retrieve(
            query=question["question_text"],
            simulation_date=sim_date,
            model_name=model_id,
            top_k=self.top_k,
        )


# ── Chronos refined (delegates to refined_retrieval.py) ───────────────────────


class ChronosRefined(BriefingProvider):
    """Two-stage retrieval: query expansion + over-retrieve + LLM rerank."""

    name = "chronos_refined"

    def __init__(
        self,
        refiner_model_id: str,
        over_retrieve_k: int,
        keep_min: int,
        keep_max: int,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.refiner_model_id = refiner_model_id
        self.over_retrieve_k = over_retrieve_k
        self.keep_min = keep_min
        self.keep_max = keep_max

    def get(self, question: dict, model_id: str) -> tuple[str | None, str | None]:
        cached = self._load_cached(question["question_id"])
        if cached:
            return cached
        from evaluation_plan.src.refined_retrieval import refine_briefing  # local import

        briefing = asyncio.run(
            refine_briefing(
                question=question,
                model_id=model_id,
                refiner_model_id=self.refiner_model_id,
                over_retrieve_k=self.over_retrieve_k,
                keep_min=self.keep_min,
                keep_max=self.keep_max,
            )
        )
        return self._save_and_hash(question, model_id, briefing)

    async def aget(self, question: dict, model_id: str) -> tuple[str | None, str | None]:
        cached = self._load_cached(question["question_id"])
        if cached:
            return cached
        from evaluation_plan.src.refined_retrieval import refine_briefing  # local import

        briefing = await refine_briefing(
            question=question,
            model_id=model_id,
            refiner_model_id=self.refiner_model_id,
            over_retrieve_k=self.over_retrieve_k,
            keep_min=self.keep_min,
            keep_max=self.keep_max,
        )
        return self._save_and_hash(question, model_id, briefing)

    def _save_and_hash(self, question: dict, model_id: str, briefing: str) -> tuple[str, str]:
        h = sha256_short(briefing)
        self._save_cached(
            question["question_id"],
            briefing,
            h,
            meta={
                "variant": self.name,
                "refiner_model_id": self.refiner_model_id,
                "over_retrieve_k": self.over_retrieve_k,
                "keep_range": [self.keep_min, self.keep_max],
                "model_id": model_id,
            },
        )
        return briefing, h
