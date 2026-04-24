"""Full post-run analysis: Brier scoring, per-experiment means, deltas, per-domain
breakdown, per-question format, token cost, variance, and E5 attrition handling.

Outputs a JSON report at evaluation_plan/output/analysis/summary.json and a
human-readable Markdown table at evaluation_plan/output/analysis/summary.md.

Scoring conventions:
  * Binary Brier: (p(YES) - y)^2 with y ∈ {0,1}
  * Multiclass Brier: Σ over options (p_i - 1[i==correct])^2 (unnormalized;
    range [0, 2] for multiclass, [0, 1] for binary — keep separate in reporting)
  * Per-question aggregate: mean Brier across samples for that (exp, qid)
  * Per-experiment aggregate: mean over per-question aggregates (so each
    question contributes equally regardless of how many samples succeeded)

E5 attrition handling: we report two numbers side-by-side —
  (a) complete-case analysis — drop error rows, report mean over remaining
  (b) available-case restriction — restrict *all* experiments to the intersection
    of qids where *every* experiment has ≥1 valid sample, so comparisons are fair
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "pipeline/output/predictions"
RES_PATH = ROOT / "pipeline/output/resolutions/resolutions.json"
MANIFEST_PATH = ROOT / "evaluation_plan/output/final_manifest.json"
OUT_DIR = ROOT / "evaluation_plan/output/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["e1", "e1p", "e2", "e3", "e4", "e5"]
EXP_LABELS = {
    "e1":  "Trump × CHRONOS broad-15",
    "e1p": "Trump × CHRONOS broad-8 (compressed)",
    "e2":  "Trump × CHRONOS refined (2-stage)",
    "e3":  "Trump × no briefing",
    "e4":  "Analyst × CHRONOS broad-15",
    "e5":  "Analyst × Tavily web search",
}

# ── Load truth + manifest ────────────────────────────────────────────────────

resolutions = {r["question_id"]: r for r in json.loads(RES_PATH.read_text())["resolutions"]}
manifest = {q["question_id"]: q for q in json.loads(MANIFEST_PATH.read_text())["questions"]}

# ── Scoring helpers ──────────────────────────────────────────────────────────


def brier_binary(p_yes: float, y: int) -> float:
    return (p_yes - y) ** 2


def brier_multiclass(probs: dict, correct: str) -> float:
    letters = sorted(set(list(probs.keys()) + [correct]))
    return sum((probs.get(L, 0.0) - (1.0 if L == correct else 0.0)) ** 2 for L in letters)


def score_record(rec: dict) -> float | None:
    qid = rec["question_id"]
    if qid not in resolutions:
        return None
    r = resolutions[qid]
    if rec.get("error"):
        return None
    if r["question_type"] == "binary":
        p = (rec.get("binary") or {}).get("probability")
        if p is None:
            return None
        y = 1 if r["correct_answer"] == "YES" else 0
        return brier_binary(p, y)
    probs = (rec.get("action") or {}).get("probabilities") or {}
    if not probs:
        return None
    return brier_multiclass(probs, r["correct_answer"])


def prob_on_correct(rec: dict) -> float | None:
    qid = rec["question_id"]
    if qid not in resolutions:
        return None
    r = resolutions[qid]
    if rec.get("error"):
        return None
    if r["question_type"] == "binary":
        p = (rec.get("binary") or {}).get("probability")
        if p is None:
            return None
        return p if r["correct_answer"] == "YES" else 1.0 - p
    probs = (rec.get("action") or {}).get("probabilities") or {}
    return probs.get(r["correct_answer"])


# ── Load all predictions ──────────────────────────────────────────────────────


def load_preds() -> dict:
    preds: dict[str, list[dict]] = {}
    for exp in EXPERIMENTS:
        f = PRED_DIR / exp / "predictions.jsonl"
        preds[exp] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return preds


# ── Aggregate ────────────────────────────────────────────────────────────────


def per_qid_brier(records: list[dict]) -> dict[str, float]:
    """Return {qid: mean_brier_over_samples}. Error rows excluded."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        b = score_record(r)
        if b is not None:
            buckets[r["question_id"]].append(b)
    return {qid: mean(vs) for qid, vs in buckets.items() if vs}


def per_qid_p_on_correct(records: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        p = prob_on_correct(r)
        if p is not None:
            buckets[r["question_id"]].append(p)
    return {qid: mean(vs) for qid, vs in buckets.items() if vs}


def variance_across_samples(records: list[dict]) -> dict[str, float]:
    """Per-qid stdev of p-on-correct across samples. Proxy for sampling noise."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        p = prob_on_correct(r)
        if p is not None:
            buckets[r["question_id"]].append(p)
    out = {}
    for qid, vs in buckets.items():
        if len(vs) >= 2:
            out[qid] = pstdev(vs)
    return out


# ── Report builder ───────────────────────────────────────────────────────────


def build_report():
    preds = load_preds()

    # Per-experiment counts
    counts = {}
    tokens = {}
    for e, recs in preds.items():
        errs = sum(1 for r in recs if r.get("error"))
        tokens[e] = {
            "in": sum((r.get("tokens_in") or 0) for r in recs),
            "out": sum((r.get("tokens_out") or 0) for r in recs),
        }
        counts[e] = {"total": len(recs), "errors": errs, "ok": len(recs) - errs}

    # Per-qid Brier + p-on-correct
    brier_by_exp = {e: per_qid_brier(preds[e]) for e in EXPERIMENTS}
    pcorr_by_exp = {e: per_qid_p_on_correct(preds[e]) for e in EXPERIMENTS}
    var_by_exp = {e: variance_across_samples(preds[e]) for e in EXPERIMENTS}

    # Full set of scorable qids
    all_qids = set(resolutions.keys()) & set(manifest.keys())

    # Restrict to qids present in ALL experiments for fair comparison
    common = all_qids.copy()
    for e in EXPERIMENTS:
        common &= set(brier_by_exp[e].keys())
    common = sorted(common)

    # Complete-case (per exp, over its own surviving qids)
    complete_case = {}
    for e in EXPERIMENTS:
        qids = sorted(brier_by_exp[e].keys())
        bs = [brier_by_exp[e][q] for q in qids]
        ps = [pcorr_by_exp[e][q] for q in qids]
        complete_case[e] = {
            "n_questions": len(qids),
            "mean_brier": mean(bs) if bs else None,
            "mean_p_correct": mean(ps) if ps else None,
        }

    # Available-case (restricted to common qids)
    available_case = {}
    for e in EXPERIMENTS:
        bs = [brier_by_exp[e][q] for q in common]
        ps = [pcorr_by_exp[e][q] for q in common]
        available_case[e] = {
            "n_questions": len(common),
            "mean_brier": mean(bs) if bs else None,
            "mean_p_correct": mean(ps) if ps else None,
        }

    # Split by question format (binary vs action) using common set only
    bin_qids = sorted(q for q in common if resolutions[q]["question_type"] == "binary")
    act_qids = sorted(q for q in common if resolutions[q]["question_type"] != "binary")
    fmt_split = {}
    for e in EXPERIMENTS:
        bbs = [brier_by_exp[e][q] for q in bin_qids if q in brier_by_exp[e]]
        aas = [brier_by_exp[e][q] for q in act_qids if q in brier_by_exp[e]]
        fmt_split[e] = {
            "binary": {"n": len(bbs), "mean_brier": mean(bbs) if bbs else None},
            "action": {"n": len(aas), "mean_brier": mean(aas) if aas else None},
        }

    # Split by difficulty (using manifest.difficulty)
    diff_groups: dict[str, list[str]] = defaultdict(list)
    for q in common:
        d = manifest[q].get("difficulty", "unknown")
        diff_groups[d].append(q)
    diff_split = {}
    for e in EXPERIMENTS:
        diff_split[e] = {}
        for d, qs in diff_groups.items():
            bs = [brier_by_exp[e][q] for q in qs if q in brier_by_exp[e]]
            diff_split[e][d] = {"n": len(bs), "mean_brier": mean(bs) if bs else None}

    # Split by domain
    dom_groups: dict[str, list[str]] = defaultdict(list)
    for q in common:
        dom_groups[manifest[q].get("domain", "unknown")].append(q)
    dom_split = {}
    for e in EXPERIMENTS:
        dom_split[e] = {}
        for d, qs in dom_groups.items():
            bs = [brier_by_exp[e][q] for q in qs if q in brier_by_exp[e]]
            dom_split[e][d] = {"n": len(bs), "mean_brier": mean(bs) if bs else None}

    # Paired deltas (per question, brier(exp_a) - brier(exp_b)), mean + win rate
    def paired(a: str, b: str):
        rows = []
        for q in common:
            rows.append(brier_by_exp[a][q] - brier_by_exp[b][q])
        return {
            "mean_delta": mean(rows),
            "median_delta": sorted(rows)[len(rows) // 2],
            "a_better_count": sum(1 for d in rows if d < 0),
            "b_better_count": sum(1 for d in rows if d > 0),
            "tie_count": sum(1 for d in rows if d == 0),
            "n": len(rows),
        }

    deltas = {
        "persona (e1 vs e4, same briefing)": paired("e1", "e4"),
        "briefing_vs_none (e1 vs e3)": paired("e1", "e3"),
        "refinement (e1 vs e2)": paired("e1", "e2"),
        "compression (e1 vs e1p)": paired("e1", "e1p"),
        "web_vs_curated (e4 vs e5)": paired("e4", "e5"),
        "trump_persona_on_web (e5 vs e3)": paired("e5", "e3"),
    }

    # Sample-level variance (intra-question stdev of p-on-correct)
    variance = {}
    for e in EXPERIMENTS:
        sds = list(var_by_exp[e].values())
        variance[e] = {
            "n_questions_with_variance": len(sds),
            "mean_intra_question_stdev": mean(sds) if sds else None,
            "max_intra_question_stdev": max(sds) if sds else None,
        }

    report = {
        "config": {
            "experiments": {e: EXP_LABELS[e] for e in EXPERIMENTS},
            "n_scorable_qids": len(all_qids),
            "n_common_qids_for_fair_comparison": len(common),
        },
        "counts": counts,
        "tokens": tokens,
        "complete_case_analysis": complete_case,
        "available_case_analysis_common_qids_only": available_case,
        "by_question_format_common": fmt_split,
        "by_difficulty_common": diff_split,
        "by_domain_common": dom_split,
        "paired_deltas_common": deltas,
        "sample_variance": variance,
        "e5_attrition": {
            "missing_qids_entirely": sorted(set(all_qids) - set(brier_by_exp["e5"].keys())),
        },
    }
    return report


# ── Markdown rendering ───────────────────────────────────────────────────────


def to_md(report: dict) -> str:
    lines = []
    lines.append("# Experiment Results Summary\n")
    lines.append(f"Scorable questions: {report['config']['n_scorable_qids']}")
    lines.append(f"Common qids across all 6 experiments (for fair comparison): "
                 f"{report['config']['n_common_qids_for_fair_comparison']}\n")

    lines.append("## Record counts and errors")
    lines.append("| exp | label | total | ok | errors | tok_in | tok_out |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for e in EXPERIMENTS:
        c, t = report["counts"][e], report["tokens"][e]
        lines.append(f"| {e} | {EXP_LABELS[e]} | {c['total']} | {c['ok']} | {c['errors']} | "
                     f"{t['in']:,} | {t['out']:,} |")

    lines.append("\n## Complete-case analysis (per-exp, each experiment's own surviving qids)")
    lines.append("| exp | n_qids | mean Brier | mean p(correct) |")
    lines.append("|---|---:|---:|---:|")
    for e in EXPERIMENTS:
        c = report["complete_case_analysis"][e]
        mb = f"{c['mean_brier']:.4f}" if c["mean_brier"] is not None else "—"
        mp = f"{c['mean_p_correct']:.4f}" if c["mean_p_correct"] is not None else "—"
        lines.append(f"| {e} | {c['n_questions']} | {mb} | {mp} |")

    lines.append("\n## Fair comparison (restricted to common qids across all 6 experiments)")
    lines.append("| exp | n_qids | mean Brier | mean p(correct) |")
    lines.append("|---|---:|---:|---:|")
    for e in EXPERIMENTS:
        c = report["available_case_analysis_common_qids_only"][e]
        mb = f"{c['mean_brier']:.4f}" if c["mean_brier"] is not None else "—"
        mp = f"{c['mean_p_correct']:.4f}" if c["mean_p_correct"] is not None else "—"
        lines.append(f"| {e} | {c['n_questions']} | {mb} | {mp} |")

    lines.append("\n## By question format (common qids only)")
    lines.append("| exp | binary n | binary Brier | action n | action Brier |")
    lines.append("|---|---:|---:|---:|---:|")
    for e in EXPERIMENTS:
        f = report["by_question_format_common"][e]
        bb = f"{f['binary']['mean_brier']:.4f}" if f["binary"]["mean_brier"] is not None else "—"
        ab = f"{f['action']['mean_brier']:.4f}" if f["action"]["mean_brier"] is not None else "—"
        lines.append(f"| {e} | {f['binary']['n']} | {bb} | {f['action']['n']} | {ab} |")

    lines.append("\n## By difficulty (common qids only)")
    # find diffs
    diffs = sorted({d for e in EXPERIMENTS for d in report["by_difficulty_common"][e].keys()})
    header = "| exp | " + " | ".join(f"{d} (n, Brier)" for d in diffs) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(diffs))
    for e in EXPERIMENTS:
        cells = []
        for d in diffs:
            entry = report["by_difficulty_common"][e].get(d, {})
            n = entry.get("n", 0)
            mb = entry.get("mean_brier")
            cells.append(f"{n}, {mb:.4f}" if mb is not None else f"{n}, —")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")

    lines.append("\n## By domain (common qids only)")
    doms = sorted({d for e in EXPERIMENTS for d in report["by_domain_common"][e].keys()})
    header = "| exp | " + " | ".join(f"{d}" for d in doms) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(doms))
    for e in EXPERIMENTS:
        cells = []
        for d in doms:
            entry = report["by_domain_common"][e].get(d, {})
            n = entry.get("n", 0)
            mb = entry.get("mean_brier")
            cells.append(f"n={n}, {mb:.4f}" if mb is not None else f"n={n}, —")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")

    lines.append("\n## Paired deltas (per-question Brier difference, common qids)")
    lines.append("Positive mean_delta means `a` is WORSE than `b` on that axis.")
    lines.append("| contrast | mean Δ | median Δ | a-better | b-better | ties | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, d in report["paired_deltas_common"].items():
        lines.append(f"| {name} | {d['mean_delta']:+.4f} | {d['median_delta']:+.4f} | "
                     f"{d['a_better_count']} | {d['b_better_count']} | {d['tie_count']} | {d['n']} |")

    lines.append("\n## Sample-level variance (mean intra-question stdev of p(correct))")
    lines.append("| exp | mean stdev | max stdev |")
    lines.append("|---|---:|---:|")
    for e in EXPERIMENTS:
        v = report["sample_variance"][e]
        ms = f"{v['mean_intra_question_stdev']:.4f}" if v["mean_intra_question_stdev"] is not None else "—"
        mx = f"{v['max_intra_question_stdev']:.4f}" if v["max_intra_question_stdev"] is not None else "—"
        lines.append(f"| {e} | {ms} | {mx} |")

    lines.append("\n## E5 attrition")
    missing = report["e5_attrition"]["missing_qids_entirely"]
    lines.append(f"Questions with zero valid E5 samples: {len(missing)}")
    if missing:
        lines.append("  " + ", ".join(missing))

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    report = build_report()
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    (OUT_DIR / "summary.md").write_text(to_md(report))
    print(f"Wrote {OUT_DIR/'summary.json'}")
    print(f"Wrote {OUT_DIR/'summary.md'}")
    print()
    # Quick top-line to stdout
    print("## Fair comparison (common qids only)")
    for e in EXPERIMENTS:
        c = report["available_case_analysis_common_qids_only"][e]
        print(f"  {e}: n={c['n_questions']:3d}  "
              f"Brier={c['mean_brier']:.4f}  "
              f"p(correct)={c['mean_p_correct']:.4f}  "
              f"[{EXP_LABELS[e]}]")
    print()
    print("## Paired deltas (common qids)")
    for name, d in report["paired_deltas_common"].items():
        sign = "favors a" if d["mean_delta"] < 0 else "favors b"
        print(f"  {name}: mean_delta={d['mean_delta']:+.4f}  ({sign})  "
              f"a-wins={d['a_better_count']}/b-wins={d['b_better_count']}/tie={d['tie_count']}/n={d['n']}")
