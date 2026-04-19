"""
Thin LLM-client wrapper used by every experiment runner.

Responsibilities:
- Call Anthropic via langchain-anthropic's ChatAnthropic.
- Parse the JSON block from the model's text response.
- Return a PredictionRecord-ready dict (probability / probabilities + reasoning).
- Capture telemetry (token counts, latency, raw response, errors).

The wrapper is format-aware: binary vs action_selection dispatch through
`predict_binary()` and `predict_action()`. Errors are captured on the record,
not raised — a failed sample still appears in the JSONL so scoring can see it.

Web-search variant (for E5) is a separate helper that uses Anthropic's
server-side web search tool with an "only use results on-or-before
{simulation_date}" instruction.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from evaluation_plan.src.schemas import ActionPrediction, BinaryPrediction


# ── JSON extraction ───────────────────────────────────────────────────────────


_JSON_BLOCK = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Pull the first plausible top-level JSON object out of a model response.

    Strategy:
    1. Try strict `json.loads(text)`.
    2. Strip ``` fences and retry.
    3. Find the first balanced {...} block with a regex and parse that.
    Raises ValueError if nothing parses.
    """
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to first {...} match
    m = _JSON_BLOCK.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No JSON object found in response: {s[:200]!r}")


# ── Record assembly ───────────────────────────────────────────────────────────


@dataclass
class RawModelCall:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


def _base_record(
    *,
    question_id: str,
    experiment: str,
    sample_idx: int,
    question_format: str,
    model_id: str,
    temperature: float,
    prompt_hash: str,
    briefing_hash: str | None,
    raw: RawModelCall | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "experiment": experiment,
        "sample_idx": sample_idx,
        "question_format": question_format,
        "model_id": model_id,
        "temperature": temperature,
        "prompt_hash": prompt_hash,
        "briefing_hash": briefing_hash,
        "binary": None,
        "action": None,
        "raw_response": raw.text if raw else "",
        "tokens_in": raw.tokens_in if raw else 0,
        "tokens_out": raw.tokens_out if raw else 0,
        "latency_ms": raw.latency_ms if raw else 0,
        "created_at": datetime.utcnow().isoformat(),
        "error": error,
    }


# ── Anthropic call ────────────────────────────────────────────────────────────


def _build_chat(model_id: str, temperature: float, max_tokens: int, tools: list | None = None):
    """
    Lazy-import langchain-anthropic so this module is importable without the
    SDK installed (tests etc.).
    """
    from langchain_anthropic import ChatAnthropic  # type: ignore

    kwargs = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    chat = ChatAnthropic(**kwargs)
    if tools:
        chat = chat.bind_tools(tools)
    return chat


def _invoke_chat(
    *,
    model_id: str,
    system_text: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    tools: list | None = None,
) -> RawModelCall:
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    chat = _build_chat(model_id, temperature, max_tokens, tools=tools)
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]

    t0 = time.perf_counter()
    resp = chat.invoke(messages)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # langchain Message: .content can be str or list of content blocks.
    content = resp.content
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    else:
        text = str(content)

    usage = getattr(resp, "response_metadata", {}).get("usage", {}) or {}
    tokens_in = usage.get("input_tokens", 0) or 0
    tokens_out = usage.get("output_tokens", 0) or 0
    return RawModelCall(text=text, tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms)


# ── Public prediction helpers ─────────────────────────────────────────────────


def predict_binary(
    *,
    question_id: str,
    experiment: str,
    sample_idx: int,
    model_id: str,
    temperature: float,
    max_tokens: int,
    system_text: str,
    user_text: str,
    prompt_hash: str,
    briefing_hash: str | None = None,
    tools: list | None = None,
) -> dict[str, Any]:
    try:
        raw = _invoke_chat(
            model_id=model_id,
            system_text=system_text,
            user_text=user_text,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
    except Exception as e:  # capture, don't raise
        return _base_record(
            question_id=question_id, experiment=experiment, sample_idx=sample_idx,
            question_format="binary", model_id=model_id, temperature=temperature,
            prompt_hash=prompt_hash, briefing_hash=briefing_hash,
            raw=None, error=f"api_error: {type(e).__name__}: {e}",
        )

    rec = _base_record(
        question_id=question_id, experiment=experiment, sample_idx=sample_idx,
        question_format="binary", model_id=model_id, temperature=temperature,
        prompt_hash=prompt_hash, briefing_hash=briefing_hash,
        raw=raw, error=None,
    )
    try:
        obj = extract_json_object(raw.text)
        pred = BinaryPrediction(**obj)
        rec["binary"] = {"probability": pred.probability, "reasoning": pred.reasoning}
    except Exception as e:
        rec["error"] = f"parse_error: {type(e).__name__}: {e}"
    return rec


def predict_action(
    *,
    question_id: str,
    experiment: str,
    sample_idx: int,
    model_id: str,
    temperature: float,
    max_tokens: int,
    system_text: str,
    user_text: str,
    prompt_hash: str,
    option_letters: list[str],
    briefing_hash: str | None = None,
    tools: list | None = None,
) -> dict[str, Any]:
    try:
        raw = _invoke_chat(
            model_id=model_id,
            system_text=system_text,
            user_text=user_text,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
    except Exception as e:
        return _base_record(
            question_id=question_id, experiment=experiment, sample_idx=sample_idx,
            question_format="action_selection", model_id=model_id, temperature=temperature,
            prompt_hash=prompt_hash, briefing_hash=briefing_hash,
            raw=None, error=f"api_error: {type(e).__name__}: {e}",
        )

    rec = _base_record(
        question_id=question_id, experiment=experiment, sample_idx=sample_idx,
        question_format="action_selection", model_id=model_id, temperature=temperature,
        prompt_hash=prompt_hash, briefing_hash=briefing_hash,
        raw=raw, error=None,
    )
    try:
        obj = extract_json_object(raw.text)
        probs = obj.get("probabilities") or {}
        # Restrict to valid letters and 0-fill missing ones
        filtered = {L: float(probs.get(L, 0.0)) for L in option_letters}
        # Re-normalize if the model produced slightly-off sums
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v / total for k, v in filtered.items()}
        pred = ActionPrediction(probabilities=filtered, reasoning=obj.get("reasoning", ""))
        rec["action"] = {"probabilities": pred.probabilities, "reasoning": pred.reasoning}
    except Exception as e:
        rec["error"] = f"parse_error: {type(e).__name__}: {e}"
    return rec
