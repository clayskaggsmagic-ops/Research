"""
Unified experiment runner for E1 / E1' / E2 / E3 / E4 / E5.

Given `--experiment <id>`, this script:
  1. Loads the pinned config (`evaluation_plan/config.yaml`).
  2. Loads the question manifest and the locked resolutions file.
  3. Selects the right (persona, briefing, tool) combo for the experiment.
  4. For each question × sample_idx, renders prompts, calls the LLM,
     captures the prediction record, and appends JSONL.

Experiment → (persona prompt, briefing provider, tools):

  e1   → trump_system.md     + ChronosBroad(top_k=15)                 + no tools
  e1p  → trump_system.md     + ChronosBroad(top_k=8)                  + no tools
  e2   → trump_system.md     + ChronosRefined()                       + no tools
  e3   → trump_system.md     + NoBriefing                             + no tools
  e4   → analyst_system.md   + ChronosBroad(top_k=15)  [REUSE E1]     + no tools
  e5   → analyst_system.md   + NoBriefing                             + web_search

E4 *reuses* E1's cached briefing verbatim so the persona-delta is clean:
  the briefing_hash written to E4 records matches the E1 hash for that qid.

DO NOT EXECUTE YET. CHRONOS (the temporal knowledge base) is still under
construction — E1 / E1' / E2 / E4 depend on it. E3 and E5 can run independently
once the Anthropic key is present, but per user direction we are currently
building pipelines only.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation_plan.src.briefings import (
    BriefingProvider,
    ChronosBroad,
    ChronosRefined,
    NoBriefing,
)
from evaluation_plan.src.io_utils import (
    already_predicted,
    append_prediction,
    load_config,
    load_manifest,
    repo_path,
)
from evaluation_plan.src.llm_client import predict_action, predict_binary
from evaluation_plan.src.prompts import option_letters_for, render_messages
from evaluation_plan.src.web_search_tool import (
    augment_system_with_temporal_constraint,
    tavily_search_context,
    web_search_tool_spec,
)


# ── Experiment descriptor ─────────────────────────────────────────────────────


def build_experiment_spec(
    experiment_id: str,
    config: dict,
) -> dict:
    """
    Return a dict describing how to run this experiment:
      system_prompt_name, briefing_provider, uses_web_search
    """
    cache_dir = repo_path(config["paths"]["briefing_cache"])
    chr_cfg = config["chronos"]

    specs: dict[str, dict] = {
        "e1": {
            "system_prompt_name": "trump_system.md",
            "briefing_provider": ChronosBroad(top_k=chr_cfg["top_k_broad"], cache_dir=cache_dir),
            "uses_web_search": False,
        },
        "e1p": {
            "system_prompt_name": "trump_system.md",
            "briefing_provider": ChronosBroad(top_k=chr_cfg["top_k_compressed"], cache_dir=cache_dir),
            "uses_web_search": False,
        },
        "e2": {
            "system_prompt_name": "trump_system.md",
            "briefing_provider": ChronosRefined(
                refiner_model_id=config["refiner_model"]["id"],
                over_retrieve_k=chr_cfg["over_retrieve_k"],
                keep_min=chr_cfg["keep_after_rerank_min"],
                keep_max=chr_cfg["keep_after_rerank_max"],
                cache_dir=cache_dir,
            ),
            "uses_web_search": False,
        },
        "e3": {
            "system_prompt_name": "trump_system.md",
            "briefing_provider": NoBriefing(),
            "uses_web_search": False,
        },
        "e4": {
            # E4 explicitly reuses the same broad-15 provider so its cache hits E1's.
            "system_prompt_name": "analyst_system.md",
            "briefing_provider": ChronosBroad(top_k=chr_cfg["top_k_broad"], cache_dir=cache_dir),
            "uses_web_search": False,
        },
        "e5": {
            "system_prompt_name": "analyst_system.md",
            "briefing_provider": NoBriefing(),
            "uses_web_search": True,
        },
    }
    if experiment_id not in specs:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")
    return specs[experiment_id]


# ── Resolvable-question filter ────────────────────────────────────────────────


def _scorable_qids(resolutions_path: str | Path) -> set[str]:
    import json
    data = json.loads(Path(resolutions_path).read_text())
    return {r["question_id"] for r in data["resolutions"]}


# ── Main loop ─────────────────────────────────────────────────────────────────


def run(
    experiment_id: str,
    *,
    config_path: str = "evaluation_plan/config.yaml",
    out_path: str | None = None,
    limit: int | None = None,
    question_ids: list[str] | None = None,
    samples_override: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
    concurrency: int = 8,
) -> Path:
    cfg = load_config(config_path)
    model_id = cfg["model"]["id"]
    temperature = cfg["model"]["temperature"]
    max_tokens = cfg["model"]["max_tokens"]
    n_samples = samples_override or cfg["samples_per_question"]

    spec = build_experiment_spec(experiment_id, cfg)
    briefing: BriefingProvider = spec["briefing_provider"]
    uses_web_search = spec["uses_web_search"]

    manifest = load_manifest(cfg["paths"]["manifest"])
    scorable = _scorable_qids(repo_path(cfg["paths"]["resolutions"]))
    questions = [q for q in manifest if q["question_id"] in scorable]
    if question_ids:
        questions = [q for q in questions if q["question_id"] in set(question_ids)]
    if limit is not None:
        questions = questions[:limit]

    out = Path(out_path) if out_path else repo_path(
        f"{cfg['paths']['predictions_dir']}/{experiment_id}/predictions.jsonl"
    )
    print(f"[{experiment_id}] model={model_id} questions={len(questions)} "
          f"samples={n_samples} out={out} dry_run={dry_run}")

    provider = cfg["model"].get("provider", "anthropic")
    tools = None
    if uses_web_search:
        spec_tool = web_search_tool_spec(
            max_uses=cfg['web_search']['max_results_per_question'],
            provider=provider,
        )
        # For Gemini we pre-retrieve via Tavily; no model-side tool needed.
        tools = [spec_tool] if spec_tool is not None else None

    # Build all briefings in a SINGLE event loop so the asyncpg pool
    # (module-level in temporal_knowledge_base.database) only sees one loop.
    # Multiple asyncio.run() calls across questions would leak pool connections
    # into subsequent closed loops → "Future attached to a different loop".
    async def _gather_briefings() -> dict[str, tuple[str | None, str | None]]:
        sem = asyncio.Semaphore(4)  # cap concurrent DB+embedding+refiner calls

        async def _one(q):
            async with sem:
                return q["question_id"], await briefing.aget(q, model_id)

        results = await asyncio.gather(*[_one(q) for q in questions])
        return dict(results)

    briefings_map: dict[str, tuple[str | None, str | None]] = (
        {} if dry_run else asyncio.run(_gather_briefings())
    )

    call_plan: list[tuple[dict, int, str, str, str, str | None]] = []
    n_skipped = 0
    for q in questions:
        qid = q["question_id"]
        brief_text, brief_hash = briefings_map.get(qid, (None, None)) if not dry_run else (None, None)

        if uses_web_search and provider == "google" and not dry_run:
            from datetime import date as _date
            ws_cfg = cfg["web_search"]
            search_ctx = tavily_search_context(
                query=q["question_text"],
                simulation_date=_date.fromisoformat(q["simulation_date"]),
                max_results=ws_cfg.get("max_results_per_question", 20),
                strict_date_filter=ws_cfg.get("enforce_temporal_filter", True),
                max_kept=ws_cfg.get("max_kept_results", 10),
                snippet_chars=ws_cfg.get("snippet_chars", 300),
            )
            brief_text = search_ctx or None
            if brief_text:
                from evaluation_plan.src.io_utils import sha256_short as _sha
                brief_hash = _sha(brief_text)

        system_name = spec["system_prompt_name"]
        system_text, user_text, prompt_hash = render_messages(q, system_name, brief_text)
        if uses_web_search:
            system_text = augment_system_with_temporal_constraint(system_text, q["simulation_date"])

        for i in range(n_samples):
            if resume and already_predicted(out, qid, i):
                n_skipped += 1
                continue
            if dry_run:
                print(f"  would call: qid={qid} sample={i} prompt_hash={prompt_hash} "
                      f"briefing_hash={brief_hash}")
                continue
            call_plan.append((q, i, system_text, user_text, prompt_hash, brief_hash))

    t_start = time.perf_counter()
    n_calls = 0
    write_lock = threading.Lock()

    def _do_call(item):
        q, i, system_text, user_text, prompt_hash, brief_hash = item
        qid = q["question_id"]
        fmt = q["question_type"]
        if fmt == "binary":
            rec = predict_binary(
                question_id=qid, experiment=experiment_id, sample_idx=i,
                model_id=model_id, temperature=temperature, max_tokens=max_tokens,
                system_text=system_text, user_text=user_text,
                prompt_hash=prompt_hash, briefing_hash=brief_hash, tools=tools,
            )
        else:
            rec = predict_action(
                question_id=qid, experiment=experiment_id, sample_idx=i,
                model_id=model_id, temperature=temperature, max_tokens=max_tokens,
                system_text=system_text, user_text=user_text,
                prompt_hash=prompt_hash, briefing_hash=brief_hash,
                option_letters=option_letters_for(q), tools=tools,
            )
        with write_lock:
            append_prediction(out, rec)
        return rec

    if call_plan and not dry_run:
        workers = max(1, min(concurrency, len(call_plan)))
        print(f"[{experiment_id}] dispatching {len(call_plan)} calls "
              f"across {workers} worker threads")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_do_call, item) for item in call_plan]
            for f in as_completed(futures):
                f.result()  # re-raise any uncaught exception (rare — predict_* capture)
                n_calls += 1
                if n_calls % 10 == 0 or n_calls == len(call_plan):
                    rate = n_calls / max(0.1, time.perf_counter() - t_start)
                    print(f"[{experiment_id}]   {n_calls}/{len(call_plan)} done "
                          f"({rate:.1f} calls/s)")

    elapsed = time.perf_counter() - t_start
    print(f"[{experiment_id}] done. calls={n_calls} skipped={n_skipped} elapsed={elapsed:.1f}s "
          f"→ {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["e1", "e1p", "e2", "e3", "e4", "e5"])
    ap.add_argument("--config", default="evaluation_plan/config.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N questions (smoke test)")
    ap.add_argument("--qid", action="append", default=None, help="Restrict to specific question_id(s)")
    ap.add_argument("--samples", type=int, default=None, help="Override samples_per_question")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print call plan without calling APIs")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Thread-pool size for parallel LLM calls (default 8)")
    args = ap.parse_args()
    run(
        experiment_id=args.experiment,
        config_path=args.config,
        out_path=args.out,
        limit=args.limit,
        question_ids=args.qid,
        samples_override=args.samples,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
