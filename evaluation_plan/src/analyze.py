"""
Step 8 analysis — cross-experiment comparison.

Inputs: per-experiment per-question Brier rows from score.per_question_scores().
Outputs:
  * Paired Wilcoxon signed-rank p-values for the key contrasts
    (E1 vs E3 — CHRONOS value; E1 vs E4 — persona value;
     E1 vs E2 — refinement value; E1 vs E1' — compression control).
  * Per-experiment headline table (Brier, log loss, ECE, top-1).
  * Reliability diagram data (confidence / accuracy per bin) as JSON.
  * `summary.md` with the headline table + a read-out paragraph.

Pure-stdlib — Wilcoxon implementation below. Plots (if wanted later) can
consume `reliability.json` with matplotlib; we don't generate PNGs here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from evaluation_plan.src.io_utils import repo_path
from evaluation_plan.src.score import (
    per_question_scores,
    score_experiment,
)


# ── Paired Wilcoxon signed-rank test (two-sided) ──────────────────────────────


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> dict[str, float]:
    """
    Two-sided paired Wilcoxon signed-rank test.

    Implements the standard recipe:
      1. d_i = a_i - b_i, drop zeros
      2. rank |d_i| with midranks on ties
      3. W+ = sum ranks where d_i > 0; W- = sum ranks where d_i < 0
      4. Statistic W = min(W+, W-)
      5. Normal approximation (with continuity correction for tie-adjusted SD)

    Returns {n, w_statistic, z, p_two_sided, median_diff}. P-value is
    approximate — use for > ~15 pairs.
    """
    assert len(a) == len(b), "paired inputs must be same length"
    diffs = [x - y for x, y in zip(a, b)]
    non_zero = [d for d in diffs if d != 0.0]
    n = len(non_zero)
    if n == 0:
        return {"n": 0, "w_statistic": 0.0, "z": 0.0, "p_two_sided": 1.0, "median_diff": 0.0}

    # Rank |d| with midranks
    abs_d = [abs(d) for d in non_zero]
    order = sorted(range(n), key=lambda i: abs_d[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    w_plus = sum(ranks[i] for i in range(n) if non_zero[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if non_zero[i] < 0)
    W = min(w_plus, w_minus)

    # Normal approx with tie correction
    mean_W = n * (n + 1) / 4.0
    var_W = n * (n + 1) * (2 * n + 1) / 24.0
    # Tie-correction: subtract sum(t^3 - t)/48 for each tie group of size t
    tie_counts: dict[float, int] = {}
    for r in ranks:
        tie_counts[r] = tie_counts.get(r, 0) + 1
    tie_adj = sum((t**3 - t) for t in tie_counts.values() if t > 1) / 48.0
    var_W -= tie_adj
    var_W = max(var_W, 1e-12)
    sd = math.sqrt(var_W)
    # Continuity correction: W shifted by 0.5 toward mean
    if W < mean_W:
        z = (W + 0.5 - mean_W) / sd
    else:
        z = (W - 0.5 - mean_W) / sd
    p = 2.0 * (1.0 - _std_normal_cdf(abs(z)))

    d_sorted = sorted(non_zero)
    median = d_sorted[n // 2] if n % 2 == 1 else 0.5 * (d_sorted[n // 2 - 1] + d_sorted[n // 2])

    return {
        "n": n,
        "w_statistic": W,
        "z": z,
        "p_two_sided": p,
        "median_diff": median,
    }


def _std_normal_cdf(x: float) -> float:
    """Standard-normal CDF via erf — no scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── Cross-experiment assembly ─────────────────────────────────────────────────


def paired_brier_rows(rows_by_exp: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """
    Return {qid: {exp: brier}} keyed on (qid × experiment). A qid is kept only
    if every requested experiment has a scored row for it.
    """
    all_qids = set.intersection(*[{r["question_id"] for r in rows} for rows in rows_by_exp.values()])
    out: dict[str, dict[str, float]] = {}
    for qid in all_qids:
        row = {}
        for exp, rows in rows_by_exp.items():
            match = next((r for r in rows if r["question_id"] == qid), None)
            if match is None:
                break
            row[exp] = match["brier"]
        else:
            out[qid] = row
    return out


def contrast(
    paired: dict[str, dict[str, float]],
    a: str,
    b: str,
) -> dict[str, float]:
    qids = sorted(paired)
    a_vals = [paired[q][a] for q in qids if a in paired[q] and b in paired[q]]
    b_vals = [paired[q][b] for q in qids if a in paired[q] and b in paired[q]]
    stats = wilcoxon_signed_rank(a_vals, b_vals)
    stats["mean_brier_a"] = mean(a_vals) if a_vals else 0.0
    stats["mean_brier_b"] = mean(b_vals) if b_vals else 0.0
    stats["delta_mean"] = stats["mean_brier_a"] - stats["mean_brier_b"]
    return stats


# ── Reliability-diagram data ──────────────────────────────────────────────────


def reliability_bins(per_q_rows: list[dict], n_bins: int = 10) -> list[dict]:
    """Per-bin mean confidence / accuracy / count, binary rows only."""
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for r in per_q_rows:
        if r["format"] != "binary":
            continue
        p = r["prediction"]
        y = r["truth"]
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    out = []
    for i, bucket in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not bucket:
            out.append({"bin": i, "range": [lo, hi], "count": 0, "mean_conf": None, "accuracy": None})
            continue
        out.append({
            "bin": i, "range": [lo, hi],
            "count": len(bucket),
            "mean_conf": mean(p for p, _ in bucket),
            "accuracy": mean(y for _, y in bucket),
        })
    return out


# ── Summary generator ─────────────────────────────────────────────────────────


HEADLINE_COLS = [
    ("Brier (raw)", "brier_raw"),
    ("Brier (cal)", "brier_calibrated"),
    ("Log loss", "log_loss_raw"),
    ("ECE", "ece_raw"),
    ("Top-1", "top1_accuracy"),
]


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def render_summary(
    summaries: dict[str, dict],
    contrasts: dict[str, dict[str, float]],
) -> str:
    lines = []
    lines.append("# Experiment results\n")
    lines.append("## Headline metrics\n")
    header = "| Experiment | " + " | ".join(name for name, _ in HEADLINE_COLS) + " |"
    divider = "| " + " | ".join(["---"] * (len(HEADLINE_COLS) + 1)) + " |"
    lines.append(header)
    lines.append(divider)
    for exp in sorted(summaries):
        s = summaries[exp]
        row = [exp] + [_fmt(s.get(key)) for _, key in HEADLINE_COLS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Baselines (mean Brier)\n")
    lines.append("| Experiment | coin_flip (binary) | uniform (action) | base_rate (binary) |")
    lines.append("| --- | --- | --- | --- |")
    for exp in sorted(summaries):
        s = summaries[exp]
        lines.append(
            f"| {exp} | {_fmt(s.get('coin_flip_brier'))} | "
            f"{_fmt(s.get('uniform_brier'))} | {_fmt(s.get('base_rate_brier'))} |"
        )
    lines.append("")

    lines.append("## Pairwise contrasts (paired Wilcoxon signed-rank)\n")
    lines.append("| Contrast | n | mean Brier A | mean Brier B | Δ mean | z | p (two-sided) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for label, stats in contrasts.items():
        lines.append(
            f"| {label} | {stats.get('n', 0)} | {_fmt(stats.get('mean_brier_a'))} | "
            f"{_fmt(stats.get('mean_brier_b'))} | {_fmt(stats.get('delta_mean'))} | "
            f"{_fmt(stats.get('z'))} | {_fmt(stats.get('p_two_sided'))} |"
        )
    lines.append("")

    lines.append("## Read-out\n")
    lines.append("Fill in after scoring is complete.")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


DEFAULT_CONTRASTS = [
    ("E1 − E3 (CHRONOS value)", "e1", "e3"),
    ("E1 − E4 (persona value)", "e1", "e4"),
    ("E1 − E2 (refinement value)", "e1", "e2"),
    ("E1 − E1p (compression control)", "e1", "e1p"),
]


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions-root",
        default="pipeline/output/predictions",
        help="dir holding e1/, e2/, ... subdirs of JSONL predictions",
    )
    ap.add_argument("--resolutions", default="pipeline/output/resolutions/resolutions.json")
    ap.add_argument("--manifest", default="evaluation_plan/output/final_manifest.json")
    ap.add_argument("--out-dir", default="pipeline/output/scores")
    ap.add_argument(
        "--experiments",
        nargs="+",
        default=["e1", "e1p", "e2", "e3", "e4", "e5"],
    )
    args = ap.parse_args()

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_exp: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}

    for exp in args.experiments:
        pred_dir = repo_path(args.predictions_root) / exp
        if not pred_dir.exists() or not any(pred_dir.glob("*.jsonl")):
            print(f"[{exp}] no predictions found at {pred_dir}; skipping")
            continue
        summary = score_experiment(
            predictions_path=pred_dir,
            resolutions_path=repo_path(args.resolutions),
            manifest_path=repo_path(args.manifest),
            experiment_id=exp,
        )
        (out_dir / f"{exp}.json").write_text(json.dumps(summary, indent=2, default=str))
        rows = per_question_scores(pred_dir, repo_path(args.resolutions), exp)
        (out_dir / f"{exp}_per_question.json").write_text(json.dumps(rows, indent=2, default=str))
        (out_dir / f"{exp}_reliability.json").write_text(json.dumps(reliability_bins(rows), indent=2))
        rows_by_exp[exp] = rows
        summaries[exp] = summary
        print(f"[{exp}] scored {summary['n_questions_scored']} questions → {out_dir}")

    if not rows_by_exp:
        print("No experiments scored. Nothing to contrast.")
        return 0

    paired = paired_brier_rows(rows_by_exp)
    print(f"paired-question set: {len(paired)}")
    contrasts: dict[str, dict] = {}
    for label, a, b in DEFAULT_CONTRASTS:
        if a in rows_by_exp and b in rows_by_exp:
            contrasts[label] = contrast(paired, a, b)

    (out_dir / "contrasts.json").write_text(json.dumps(contrasts, indent=2, default=str))
    (out_dir / "summary.md").write_text(render_summary(summaries, contrasts))
    print(f"wrote {out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
