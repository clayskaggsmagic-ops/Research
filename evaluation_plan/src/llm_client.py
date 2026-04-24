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


def _scan_balanced_object(s: str, start: int) -> str | None:
    """Walk forward from '{' at index `start`, tracking string escapes, and
    return the balanced {...} substring (or None if unbalanced)."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first plausible top-level JSON object out of a model response.

    Strategy:
    1. Try strict `json.loads(text)`.
    2. Strip ``` fences and retry.
    3. Find the outermost balanced {...} via a manual scanner (handles truncated
       reasoning strings that still include valid probabilities blocks).
    4. Regex fallback for small one-level-nested blocks.
    5. If the response is clearly truncated mid-reasoning, try to repair by
       closing the string + object and parsing that.
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
    # Manual balanced scanner — outermost object first.
    # Strip any leading ```json / ``` fence.
    body = re.sub(r"^```(?:json)?\s*", "", s).rstrip("`").strip()
    first = body.find("{")
    if first != -1:
        balanced = _scan_balanced_object(body, first)
        if balanced:
            try:
                return json.loads(balanced)
            except json.JSONDecodeError:
                pass
        # Truncated? Attempt a simple repair: close any open string, then
        # add the required closing braces based on depth.
        depth = 0
        in_str = False
        esc = False
        for c in body[first:]:
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
        repaired = body[first:]
        if in_str:
            repaired += '"'
        repaired += "}" * max(depth, 0)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    # Final regex fallback (one-level-nested) — worst case returns inner dict.
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


def _is_gemini(model_id: str) -> bool:
    return model_id.lower().startswith(("gemini", "models/gemini"))


def _build_chat(model_id: str, temperature: float, max_tokens: int, tools: list | None = None):
    """
    Lazy-import the chat provider. Dispatches on model_id prefix:
      gemini-*  → langchain_google_genai.ChatGoogleGenerativeAI
      otherwise → langchain_anthropic.ChatAnthropic
    """
    if _is_gemini(model_id):
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        # Gemini 2.5 thinking tokens count against max_tokens. Disable thinking
        # to keep output tight and predictable for downstream JSON parsing.
        # timeout=90 + max_retries=2 caps a stuck call at ~3min instead of 9min,
        # which matters when Gemini service emits transient 504 DeadlineExceeded.
        chat_kwargs = {
            "model": model_id,
            "temperature": temperature,
            "max_tokens": max(4096, max_tokens),
            "timeout": 120,
            "max_retries": 2,
        }
        # thinking_config.thinking_budget is a Gemini 2.5+ kwarg; 2.0 rejects it.
        if "2.5" in model_id or "3" in model_id:
            chat_kwargs["model_kwargs"] = {
                "generation_config": {"thinking_config": {"thinking_budget": 0}}
            }
        chat = ChatGoogleGenerativeAI(**chat_kwargs)
        if tools:
            # Gemini supports built-in search via a dict spec, e.g. {"google_search": {}}.
            # bind_tools works for function-calling; for the builtin grounding tool we
            # pass it through on invoke via generation_config. We keep it simple here:
            # if a tool dict has "google_search", use Gemini search grounding.
            search_tool = None
            fn_tools = []
            for t in tools:
                if isinstance(t, dict) and ("google_search" in t or t.get("type") == "google_search"):
                    search_tool = {"google_search": {}}
                else:
                    fn_tools.append(t)
            if search_tool is not None:
                chat = chat.bind(tools=[search_tool])
            if fn_tools:
                chat = chat.bind_tools(fn_tools)
        return chat

    from langchain_anthropic import ChatAnthropic  # type: ignore

    chat = ChatAnthropic(model=model_id, temperature=temperature, max_tokens=max_tokens)
    if tools:
        chat = chat.bind_tools(tools)
    return chat


_TRANSIENT_MARKERS = (
    "DeadlineExceeded",
    "ResourceExhausted",
    "RESOURCE_EXHAUSTED",
    "ServiceUnavailable",
    "UNAVAILABLE",
    "DNS resolution failed",
    "C-ares",
    "No route to host",
    "Network is unreachable",
    "Connection reset",
    "EOF occurred in violation",
    "429",
    "503",
    "504",
    "500 Internal",
    "RemoteProtocolError",
    "Server disconnected",
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
    "ProtocolError",
)


def _is_transient(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}"
    return any(m in s for m in _TRANSIENT_MARKERS)


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

    # Outer tenacity-style loop for transient network / 5xx failures. The inner
    # langchain max_retries handles sub-call retries, but some failure modes
    # (DNS cache poisoning, gRPC channel death) survive that; we retry the
    # whole invoke with exponential backoff.
    t0 = time.perf_counter()
    resp = None
    _delays = [2, 4, 8, 16, 32, 60, 120, 180]  # 9 attempts total
    for attempt in range(len(_delays) + 1):
        try:
            resp = chat.invoke(messages)
            break
        except Exception as e:
            if not _is_transient(e) or attempt == len(_delays):
                raise
            time.sleep(_delays[attempt])
            # Rebuild chat — some grpc failure modes leave the channel in a
            # permanently bad state; a fresh client recovers.
            chat = _build_chat(model_id, temperature, max_tokens, tools=tools)
    assert resp is not None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # langchain Message: .content can be str or list of content blocks.
    content = resp.content
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    else:
        text = str(content)

    # Unified usage extraction across providers.
    meta = getattr(resp, "response_metadata", {}) or {}
    usage_meta = getattr(resp, "usage_metadata", None) or {}
    if _is_gemini(model_id):
        um = meta.get("usage_metadata") or usage_meta or {}
        tokens_in = um.get("prompt_token_count") or um.get("input_tokens") or 0
        tokens_out = (
            um.get("candidates_token_count")
            or um.get("output_tokens")
            or 0
        )
        # Gemini 2.5 thinking tokens are billed as output — include them.
        tokens_out += um.get("thoughts_token_count", 0) or 0
    else:
        usage = meta.get("usage", {}) or {}
        tokens_in = usage.get("input_tokens", 0) or 0
        tokens_out = usage.get("output_tokens", 0) or 0
    return RawModelCall(text=text, tokens_in=int(tokens_in), tokens_out=int(tokens_out), latency_ms=latency_ms)


# ── Public prediction helpers ─────────────────────────────────────────────────


_STRICT_JSON_SUFFIX = (
    "\n\nSTRICT OUTPUT FORMAT: Return ONLY one valid JSON object. "
    "Do NOT include any prose, preamble, markdown fences, or trailing commentary. "
    "The response must begin with `{` and end with `}`."
)

_STRICT_JSON_ACTION_SUFFIX = (
    "\n\nSTRICT OUTPUT FORMAT: Return ONLY one valid JSON object with keys "
    "`probabilities` (object mapping each option letter to a probability in [0,1], "
    "strictly positive values that sum to 1) and `reasoning` (string). "
    "DO NOT return all zeros. Every option must have a positive probability. "
    "Begin with `{` and end with `}` — no prose, no markdown fences."
)


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
        return rec
    except Exception:
        pass
    # Parse / validation failed — retry up to 3 times with escalating strictness.
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw_retry = _invoke_chat(
                model_id=model_id,
                system_text=system_text + _STRICT_JSON_SUFFIX,
                user_text=user_text,
                temperature=max(0.0, temperature - 0.5) if attempt == 0 else 0.0,
                max_tokens=max_tokens,
                tools=tools,
            )
            obj = extract_json_object(raw_retry.text)
            pred = BinaryPrediction(**obj)
            rec["raw_response"] = raw_retry.text
            rec["tokens_in"] += raw_retry.tokens_in
            rec["tokens_out"] += raw_retry.tokens_out
            rec["latency_ms"] += raw_retry.latency_ms
            rec["binary"] = {"probability": pred.probability, "reasoning": pred.reasoning}
            rec["error"] = None
            return rec
        except Exception as e:
            last_err = e
    rec["error"] = f"parse_error: {type(last_err).__name__}: {last_err}"
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
    def _parse(text: str):
        obj = extract_json_object(text)
        probs = obj.get("probabilities") or {}
        filtered = {L: float(probs.get(L, 0.0)) for L in option_letters}
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: v / total for k, v in filtered.items()}
        return ActionPrediction(probabilities=filtered, reasoning=obj.get("reasoning", ""))

    try:
        pred = _parse(raw.text)
        rec["action"] = {"probabilities": pred.probabilities, "reasoning": pred.reasoning}
        return rec
    except Exception:
        pass
    # Parse / validation failed (incl. all-zero probabilities) — retry up to 3
    # times with an anti-degenerate instruction.
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw_retry = _invoke_chat(
                model_id=model_id,
                system_text=system_text + _STRICT_JSON_ACTION_SUFFIX,
                user_text=user_text,
                temperature=max(0.0, temperature - 0.5) if attempt == 0 else 0.0,
                max_tokens=max_tokens,
                tools=tools,
            )
            pred = _parse(raw_retry.text)
            rec["raw_response"] = raw_retry.text
            rec["tokens_in"] += raw_retry.tokens_in
            rec["tokens_out"] += raw_retry.tokens_out
            rec["latency_ms"] += raw_retry.latency_ms
            rec["action"] = {"probabilities": pred.probabilities, "reasoning": pred.reasoning}
            rec["error"] = None
            return rec
        except Exception as e:
            last_err = e
    rec["error"] = f"parse_error: {type(last_err).__name__}: {last_err}"
    return rec
