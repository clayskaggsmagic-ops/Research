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
import sys
import time
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

    tools = [web_search_tool_spec(max_uses=cfg['web_search']['max_results_per_question'])] if uses_web_search else None

    t_start = time.perf_counter()
    n_calls = 0
    n_skipped = 0
    for q in questions:
        qid = q["question_id"]
        brief_text, brief_hash = briefing.get(q, model_id) if not dry_run else (None, None)

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
            append_prediction(out, rec)
            n_calls += 1

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
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
