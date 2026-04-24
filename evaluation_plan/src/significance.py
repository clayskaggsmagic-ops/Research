"""Paired sign-test and bootstrap CI for per-question Brier deltas across conditions."""
from __future__ import annotations

import json
import random
from math import comb
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "pipeline/output/predictions"
RES_PATH = ROOT / "pipeline/output/resolutions/resolutions.json"
MANIFEST_PATH = ROOT / "evaluation_plan/output/final_manifest.json"
OUT_DIR = ROOT / "evaluation_plan/output/analysis"

resolutions = {r["question_id"]: r for r in json.loads(RES_PATH.read_text())["resolutions"]}
manifest = {q["question_id"]: q for q in json.loads(MANIFEST_PATH.read_text())["questions"]}


def brier(rec):
    qid = rec["question_id"]
    if qid not in resolutions or rec.get("error"):
        return None
    r = resolutions[qid]
    if r["question_type"] == "binary":
        p = (rec.get("binary") or {}).get("probability")
        if p is None:
            return None
        y = 1 if r["correct_answer"] == "YES" else 0
        return (p - y) ** 2
    probs = (rec.get("action") or {}).get("probabilities") or {}
    if not probs:
        return None
    letters = sorted(set(list(probs.keys()) + [r["correct_answer"]]))
    return sum((probs.get(L, 0) - (1.0 if L == r["correct_answer"] else 0)) ** 2 for L in letters)


def per_qid(exp: str) -> dict:
    out = {}
    from collections import defaultdict
    buckets = defaultdict(list)
    for line in (PRED_DIR / exp / "predictions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        b = brier(r)
        if b is not None:
            buckets[r["question_id"]].append(b)
    return {q: mean(v) for q, v in buckets.items() if v}


def sign_test_pvalue(delta_list: list[float]) -> float:
    """Exact two-sided sign test. Ignores ties."""
    pos = sum(1 for d in delta_list if d > 0)
    neg = sum(1 for d in delta_list if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    # two-sided: P(X >= k) + P(X <= n-k) under Binom(n, 0.5), = 2 * P(X>=k) when n-k < k
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_ci_mean(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(42)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


EXPERIMENTS = ["e1", "e1p", "e2", "e3", "e4", "e5"]
brier_by_exp = {e: per_qid(e) for e in EXPERIMENTS}

# Common qids across all experiments
common = set(resolutions.keys())
for e in EXPERIMENTS:
    common &= set(brier_by_exp[e].keys())
common = sorted(common)

out = {"n_common": len(common)}

# Bootstrap CI for per-condition mean Brier on common qids
out["ci"] = {}
for e in EXPERIMENTS:
    vals = [brier_by_exp[e][q] for q in common]
    m = mean(vals)
    lo, hi = bootstrap_ci_mean(vals)
    out["ci"][e] = {"mean": m, "ci_lo": lo, "ci_hi": hi}

# Paired sign tests
contrasts = [
    ("persona_effect_e1_vs_e4", "e1", "e4"),      # Trump vs Analyst (same briefing)
    ("briefing_effect_e1_vs_e3", "e1", "e3"),     # Trump: broad-15 vs nothing
    ("refinement_effect_e1_vs_e2", "e1", "e2"),   # Trump: broad-15 vs refined
    ("compression_effect_e1_vs_e1p", "e1", "e1p"),# Trump: broad-15 vs broad-8
    ("web_vs_curated_e4_vs_e5", "e4", "e5"),      # Analyst: CHRONOS vs web
    ("e1_vs_e3_no_briefing", "e1", "e3"),
]

out["sign_tests"] = {}
for name, a, b in contrasts:
    deltas = [brier_by_exp[a][q] - brier_by_exp[b][q] for q in common]
    p = sign_test_pvalue(deltas)
    a_better = sum(1 for d in deltas if d < 0)
    b_better = sum(1 for d in deltas if d > 0)
    tied = sum(1 for d in deltas if d == 0)
    md = mean(deltas)
    # Bootstrap CI on mean delta
    lo, hi = bootstrap_ci_mean(deltas)
    out["sign_tests"][name] = {
        "a": a, "b": b,
        "n": len(deltas),
        "mean_delta": md,
        "delta_ci_lo": lo, "delta_ci_hi": hi,
        "a_better": a_better, "b_better": b_better, "ties": tied,
        "sign_test_p_two_sided": p,
    }

(OUT_DIR / "significance.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
