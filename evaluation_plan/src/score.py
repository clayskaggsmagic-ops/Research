"""
Scoring module for the leader-prediction experiments.

Loads predictions (JSONL, one row per sample) and resolutions (JSON),
aggregates across samples per question, computes standard forecasting
metrics, and writes an ExperimentScore.

Pure-stdlib implementation (no numpy/scipy). All metrics match the
formulas cited in evaluation_plan/experiment_design.md.

Usage:
    python -m evaluation_plan.src.score <experiment_id> \\
        --predictions pipeline/output/predictions/e1/ \\
        --resolutions pipeline/output/resolutions/resolutions.json \\
        --manifest   evaluation_plan/output/final_manifest.json \\
        --out        pipeline/output/scores/e1.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ── clamp helper ──────────────────────────────────────────────────────────────

_EPS = 1e-6


def _clamp(p: float, lo: float = _EPS, hi: float = 1.0 - _EPS) -> float:
    return max(lo, min(hi, p))


# ── Per-prediction metrics ────────────────────────────────────────────────────


def brier_binary(p_yes: float, y: int) -> float:
    """Binary Brier score. y must be 0 or 1."""
    assert y in (0, 1)
    return (p_yes - y) ** 2


def brier_multiclass(p_vec: dict[str, float], y_letter: str) -> float:
    """
    Multi-class Brier — sum of squared errors across all option letters.
    y_vec is one-hot at y_letter. p_vec keys are option letters.
    """
    assert y_letter in p_vec, f"truth {y_letter!r} missing from prediction keys {list(p_vec)}"
    total = 0.0
    for letter, p in p_vec.items():
        y = 1.0 if letter == y_letter else 0.0
        total += (p - y) ** 2
    return total


def log_loss_binary(p_yes: float, y: int) -> float:
    assert y in (0, 1)
    p = _clamp(p_yes)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def log_loss_multiclass(p_vec: dict[str, float], y_letter: str) -> float:
    p_true = _clamp(p_vec[y_letter])
    return -math.log(p_true)


def top_k_accuracy(p_vec: dict[str, float], y_letter: str, k: int) -> int:
    """Returns 1 if y_letter is in the top-k highest-probability options, else 0."""
    ranked = sorted(p_vec.items(), key=lambda kv: kv[1], reverse=True)
    top = {letter for letter, _ in ranked[:k]}
    return int(y_letter in top)


# ── Aggregation across samples ────────────────────────────────────────────────


def aggregate_binary_samples(samples: list[float]) -> tuple[float, float]:
    """Return (mean_probability, std_deviation) of the samples."""
    m = mean(samples)
    if len(samples) <= 1:
        return m, 0.0
    var = sum((s - m) ** 2 for s in samples) / (len(samples) - 1)
    return m, math.sqrt(var)


def aggregate_action_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    """Elementwise mean across a list of option→prob dicts. Renormalize to sum to 1."""
    keys = samples[0].keys()
    agg = {k: mean(s[k] for s in samples) for k in keys}
    total = sum(agg.values())
    if total == 0:
        # Degenerate — return uniform
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    return {k: v / total for k, v in agg.items()}


# ── Calibration ───────────────────────────────────────────────────────────────


def ece_binary(
    predictions: list[tuple[float, int]],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for binary predictions (Naeini et al. 2015).
    predictions: list of (p_yes, y_true) pairs where y_true ∈ {0, 1}.
    """
    if not predictions:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in predictions:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    N = len(predictions)
    total_gap = 0.0
    for bucket in bins:
        if not bucket:
            continue
        conf = mean(p for p, _ in bucket)
        acc = mean(y for _, y in bucket)
        total_gap += (len(bucket) / N) * abs(conf - acc)
    return total_gap


# ── Murphy decomposition ──────────────────────────────────────────────────────


def murphy_decomposition(
    predictions: list[tuple[float, int]],
    n_bins: int = 10,
) -> tuple[float, float, float]:
    """
    Decompose Brier into (reliability, resolution, uncertainty).
    Brier = reliability - resolution + uncertainty.

    reliability  (lower is better) — calibration error
    resolution   (higher is better) — ability to discriminate between classes
    uncertainty  — inherent variance of the outcome (fixed by data)
    """
    if not predictions:
        return 0.0, 0.0, 0.0
    N = len(predictions)
    bar_y = mean(y for _, y in predictions)
    uncertainty = bar_y * (1.0 - bar_y)

    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in predictions:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    reliability = 0.0
    resolution = 0.0
    for bucket in bins:
        if not bucket:
            continue
        n_k = len(bucket)
        conf = mean(p for p, _ in bucket)
        acc = mean(y for _, y in bucket)
        reliability += (n_k / N) * (conf - acc) ** 2
        resolution += (n_k / N) * (acc - bar_y) ** 2
    return reliability, resolution, uncertainty


# ── Temperature scaling ───────────────────────────────────────────────────────


def _apply_temperature_binary(p: float, T: float) -> float:
    p = _clamp(p)
    logit = math.log(p / (1 - p))
    scaled = logit / T
    return 1.0 / (1.0 + math.exp(-scaled))


def _apply_temperature_multi(p_vec: dict[str, float], T: float) -> dict[str, float]:
    logits = {k: math.log(_clamp(v)) for k, v in p_vec.items()}
    # Softmax with temperature
    m = max(logits.values())
    exps = {k: math.exp((v - m) / T) for k, v in logits.items()}
    Z = sum(exps.values())
    return {k: e / Z for k, e in exps.items()}


def fit_temperature_binary(
    predictions: list[tuple[float, int]],
    T_range: tuple[float, float] = (0.5, 4.0),
    steps: int = 71,
) -> float:
    """Grid search T to minimize mean log loss. Returns best T."""
    if not predictions:
        return 1.0
    step = (T_range[1] - T_range[0]) / (steps - 1)
    best_T, best_nll = 1.0, float("inf")
    for i in range(steps):
        T = T_range[0] + i * step
        nll = mean(log_loss_binary(_apply_temperature_binary(p, T), y) for p, y in predictions)
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


def fit_temperature_multi(
    predictions: list[tuple[dict[str, float], str]],
    T_range: tuple[float, float] = (0.5, 4.0),
    steps: int = 71,
) -> float:
    if not predictions:
        return 1.0
    step = (T_range[1] - T_range[0]) / (steps - 1)
    best_T, best_nll = 1.0, float("inf")
    for i in range(steps):
        T = T_range[0] + i * step
        nll = mean(log_loss_multiclass(_apply_temperature_multi(p, T), y) for p, y in predictions)
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


# ── Baselines ─────────────────────────────────────────────────────────────────


def coin_flip_brier_binary(y: int) -> float:
    return brier_binary(0.5, y)


def uniform_brier_multiclass(option_letters: list[str], y_letter: str) -> float:
    n = len(option_letters)
    p_vec = {letter: 1.0 / n for letter in option_letters}
    return brier_multiclass(p_vec, y_letter)


def base_rate_brier_binary(base_rate: float, y: int) -> float:
    return brier_binary(_clamp(base_rate, 0.0, 1.0), y)


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_resolutions(path: str | Path) -> dict[str, dict]:
    """Returns {question_id: resolution_record} for all scorable questions."""
    data = json.loads(Path(path).read_text())
    return {r["question_id"]: r for r in data["resolutions"]}


def load_manifest(path: str | Path) -> dict[str, dict]:
    """Returns {question_id: manifest_entry} keyed by question_id."""
    data = json.loads(Path(path).read_text())
    return {q["question_id"]: q for q in data["questions"]}


def load_predictions(path: str | Path) -> list[dict]:
    """
    Load prediction records from a directory of JSONL files or a single JSONL file.
    Each line must match evaluation_plan/src/schemas.py::PredictionRecord fields.
    """
    p = Path(path)
    records = []
    files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Ground-truth normalization ────────────────────────────────────────────────


def resolution_to_truth(res: dict) -> tuple[str, int | str]:
    """
    Map a resolution record to a normalized (format, truth) tuple.
    - binary: ("binary", 0 or 1)
    - action: ("action", "A"|"B"|...)
    """
    status = res["resolution_status"]
    ans = res["correct_answer"]
    if status == "resolved_yes":
        return "binary", 1
    if status == "resolved_no":
        return "binary", 0
    if status == "resolved_option":
        return "action", ans
    raise ValueError(f"Cannot map resolution_status={status!r} to truth")


# ── End-to-end experiment scoring ─────────────────────────────────────────────


def score_experiment(
    predictions_path: str | Path,
    resolutions_path: str | Path,
    manifest_path: str | Path,
    experiment_id: str,
    n_bins: int = 10,
    calibration_holdout_fraction: float = 0.30,
    calibration_seed: int = 42,
) -> dict:
    """
    Score one experiment end-to-end. Returns a dict matching ExperimentScore.
    """
    resolutions = load_resolutions(resolutions_path)
    manifest = load_manifest(manifest_path)
    raw = load_predictions(predictions_path)

    # Group samples by question_id
    by_q: dict[str, list[dict]] = defaultdict(list)
    for rec in raw:
        if rec["experiment"] != experiment_id:
            continue
        by_q[rec["question_id"]].append(rec)

    # Aggregate per-question predictions
    binary_preds: list[tuple[str, float, int]] = []      # (qid, p_yes, y)
    action_preds: list[tuple[str, dict[str, float], str]] = []
    missing = []
    for qid, res in resolutions.items():
        samples = by_q.get(qid)
        if not samples:
            missing.append(qid)
            continue
        fmt, truth = resolution_to_truth(res)
        if fmt == "binary":
            ps = [s["binary"]["probability"] for s in samples if s.get("binary")]
            if not ps:
                missing.append(qid)
                continue
            p_mean, _ = aggregate_binary_samples(ps)
            binary_preds.append((qid, p_mean, int(truth)))
        else:
            dists = [s["action"]["probabilities"] for s in samples if s.get("action")]
            if not dists:
                missing.append(qid)
                continue
            dist = aggregate_action_samples(dists)
            action_preds.append((qid, dist, str(truth)))

    # ── Raw metrics ──
    brier_b = [brier_binary(p, y) for _, p, y in binary_preds]
    brier_m = [brier_multiclass(d, y) for _, d, y in action_preds]
    brier_all = brier_b + brier_m
    brier_raw = mean(brier_all) if brier_all else 0.0

    nll_b = [log_loss_binary(p, y) for _, p, y in binary_preds]
    nll_m = [log_loss_multiclass(d, y) for _, d, y in action_preds]
    nll_all = nll_b + nll_m
    ll_raw = mean(nll_all) if nll_all else 0.0

    # ECE and Murphy on binary only (standard reporting)
    binary_pairs = [(p, y) for _, p, y in binary_preds]
    ece_raw = ece_binary(binary_pairs, n_bins=n_bins)
    reliability, resolution, uncertainty = murphy_decomposition(binary_pairs, n_bins=n_bins)

    # Top-k accuracy on action_selection only
    top1 = mean(top_k_accuracy(d, y, 1) for _, d, y in action_preds) if action_preds else None
    top2 = mean(top_k_accuracy(d, y, 2) for _, d, y in action_preds) if action_preds else None

    # ── Temperature scaling (fit on holdout split) ──
    rng = random.Random(calibration_seed)
    binary_shuf = list(binary_preds)
    action_shuf = list(action_preds)
    rng.shuffle(binary_shuf)
    rng.shuffle(action_shuf)
    cut_b = int(len(binary_shuf) * calibration_holdout_fraction)
    cut_a = int(len(action_shuf) * calibration_holdout_fraction)
    fit_b = [(p, y) for _, p, y in binary_shuf[:cut_b]]
    fit_a = [(d, y) for _, d, y in action_shuf[:cut_a]]
    T_b = fit_temperature_binary(fit_b) if fit_b else 1.0
    T_a = fit_temperature_multi(fit_a) if fit_a else 1.0
    # Use the mean of the two as the reported temperature when both exist;
    # otherwise use whichever is relevant.
    T_report = mean(t for t in (T_b if binary_preds else None, T_a if action_preds else None) if t is not None)

    # Apply temperature to the *evaluation* split (the 70% not used to fit)
    eval_b = binary_shuf[cut_b:]
    eval_a = action_shuf[cut_a:]
    brier_b_cal = [brier_binary(_apply_temperature_binary(p, T_b), y) for _, p, y in eval_b]
    brier_m_cal = [brier_multiclass(_apply_temperature_multi(d, T_a), y) for _, d, y in eval_a]
    brier_cal_all = brier_b_cal + brier_m_cal
    brier_calibrated = mean(brier_cal_all) if brier_cal_all else 0.0

    nll_b_cal = [log_loss_binary(_apply_temperature_binary(p, T_b), y) for _, p, y in eval_b]
    nll_m_cal = [log_loss_multiclass(_apply_temperature_multi(d, T_a), y) for _, d, y in eval_a]
    nll_cal_all = nll_b_cal + nll_m_cal
    ll_calibrated = mean(nll_cal_all) if nll_cal_all else 0.0
    ece_calibrated = ece_binary(
        [(_apply_temperature_binary(p, T_b), y) for _, p, y in eval_b], n_bins=n_bins
    )

    # ── Baselines (computed on the same question set for a fair delta) ──
    coin_flip_brier = mean(coin_flip_brier_binary(y) for _, _, y in binary_preds) if binary_preds else None
    uniform_brier = (
        mean(uniform_brier_multiclass(list(d.keys()), y) for _, d, y in action_preds)
        if action_preds else None
    )
    # base-rate baseline uses manifest's base_rate_estimate
    base_rate_scores = []
    for qid, _, y in binary_preds:
        br = manifest.get(qid, {}).get("base_rate_estimate")
        if br is not None:
            base_rate_scores.append(base_rate_brier_binary(br, y))
    base_rate_brier = mean(base_rate_scores) if base_rate_scores else None

    return {
        "experiment": experiment_id,
        "n_questions_scored": len(binary_preds) + len(action_preds),
        "n_binary": len(binary_preds),
        "n_action": len(action_preds),
        "n_missing_predictions": len(missing),
        "missing_question_ids": missing,
        # headline metrics
        "brier_raw": brier_raw,
        "brier_calibrated": brier_calibrated,
        "log_loss_raw": ll_raw,
        "log_loss_calibrated": ll_calibrated,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_calibrated,
        "top1_accuracy": top1,
        "top2_accuracy": top2,
        # Murphy (binary only)
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        # calibration fit
        "temperature_binary": T_b,
        "temperature_multi": T_a,
        "temperature_reported": T_report,
        # baselines
        "coin_flip_brier": coin_flip_brier,
        "uniform_brier": uniform_brier,
        "base_rate_brier": base_rate_brier,
    }


# ── Per-question dump (for cross-experiment stats in Step 8) ──────────────────


def per_question_scores(
    predictions_path: str | Path,
    resolutions_path: str | Path,
    experiment_id: str,
) -> list[dict]:
    """Returns one row per (question, experiment) with raw Brier for stats in Step 8."""
    resolutions = load_resolutions(resolutions_path)
    raw = load_predictions(predictions_path)
    by_q: dict[str, list[dict]] = defaultdict(list)
    for rec in raw:
        if rec["experiment"] == experiment_id:
            by_q[rec["question_id"]].append(rec)

    rows = []
    for qid, res in resolutions.items():
        samples = by_q.get(qid)
        if not samples:
            continue
        fmt, truth = resolution_to_truth(res)
        if fmt == "binary":
            ps = [s["binary"]["probability"] for s in samples if s.get("binary")]
            if not ps:
                continue
            p_mean, p_std = aggregate_binary_samples(ps)
            rows.append({
                "question_id": qid,
                "experiment": experiment_id,
                "format": "binary",
                "prediction": p_mean,
                "prediction_std": p_std,
                "truth": int(truth),
                "brier": brier_binary(p_mean, int(truth)),
                "log_loss": log_loss_binary(p_mean, int(truth)),
            })
        else:
            dists = [s["action"]["probabilities"] for s in samples if s.get("action")]
            if not dists:
                continue
            dist = aggregate_action_samples(dists)
            rows.append({
                "question_id": qid,
                "experiment": experiment_id,
                "format": "action",
                "prediction": dist,
                "truth": truth,
                "brier": brier_multiclass(dist, str(truth)),
                "log_loss": log_loss_multiclass(dist, str(truth)),
                "top1_hit": top_k_accuracy(dist, str(truth), 1),
                "top2_hit": top_k_accuracy(dist, str(truth), 2),
            })
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_id")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--resolutions", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-question-out", default=None)
    args = ap.parse_args()

    summary = score_experiment(
        predictions_path=args.predictions,
        resolutions_path=args.resolutions,
        manifest_path=args.manifest,
        experiment_id=args.experiment_id,
    )
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")

    if args.per_question_out:
        rows = per_question_scores(args.predictions, args.resolutions, args.experiment_id)
        Path(args.per_question_out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.per_question_out}")


if __name__ == "__main__":
    _main()
