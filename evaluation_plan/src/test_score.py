"""
Unit tests for score.py — pure-stdlib, runnable as `python -m evaluation_plan.src.test_score`.

Tests are hand-built synthetic cases with known-correct answers. No real data.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from evaluation_plan.src.score import (
    _apply_temperature_binary,
    _apply_temperature_multi,
    aggregate_action_samples,
    aggregate_binary_samples,
    base_rate_brier_binary,
    brier_binary,
    brier_multiclass,
    coin_flip_brier_binary,
    ece_binary,
    fit_temperature_binary,
    fit_temperature_multi,
    log_loss_binary,
    log_loss_multiclass,
    murphy_decomposition,
    score_experiment,
    top_k_accuracy,
    uniform_brier_multiclass,
)


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ── Brier ─────────────────────────────────────────────────────────────────────


def test_brier_binary_endpoints() -> None:
    _assert(brier_binary(1.0, 1) == 0.0, "perfect YES prediction must be 0")
    _assert(brier_binary(0.0, 0) == 0.0, "perfect NO prediction must be 0")
    _assert(brier_binary(0.0, 1) == 1.0, "worst YES prediction must be 1")
    _assert(brier_binary(1.0, 0) == 1.0, "worst NO prediction must be 1")
    _assert(brier_binary(0.5, 1) == 0.25, "coin flip on YES must be 0.25")
    _assert(brier_binary(0.5, 0) == 0.25, "coin flip on NO must be 0.25")


def test_brier_multiclass() -> None:
    # {A:0.1, B:0.9}, truth=B → (0.1-0)^2 + (0.9-1)^2 = 0.01 + 0.01 = 0.02
    b = brier_multiclass({"A": 0.1, "B": 0.9}, "B")
    _assert(_approx(b, 0.02), f"expected 0.02, got {b}")
    # Uniform over 4: each is 0.25, truth=A → (0.75)^2 + 3*(0.25)^2 = 0.5625 + 0.1875 = 0.75
    b = brier_multiclass({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, "A")
    _assert(_approx(b, 0.75), f"expected 0.75, got {b}")
    # Perfect prediction
    b = brier_multiclass({"A": 0.0, "B": 1.0, "C": 0.0}, "B")
    _assert(_approx(b, 0.0), f"expected 0.0, got {b}")


# ── Log loss ──────────────────────────────────────────────────────────────────


def test_log_loss_binary() -> None:
    # Perfect → near 0 (clamped, not literally 0)
    _assert(log_loss_binary(1.0, 1) < 1e-4, "perfect prediction should be ~0 log loss")
    # 50/50 → log(2) ≈ 0.693
    _assert(_approx(log_loss_binary(0.5, 1), math.log(2), tol=1e-4), "0.5 on true should be log(2)")
    # Worst → clamped — should be a large positive number, not inf
    worst = log_loss_binary(0.0, 1)
    _assert(worst > 10 and worst < 20, f"worst prediction clamped should be ~14, got {worst}")


def test_log_loss_multiclass() -> None:
    ll = log_loss_multiclass({"A": 0.5, "B": 0.5}, "A")
    _assert(_approx(ll, math.log(2), tol=1e-4), "uniform 2-class log loss should be log(2)")


# ── Top-k accuracy ────────────────────────────────────────────────────────────


def test_top_k_accuracy() -> None:
    p = {"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.1}
    _assert(top_k_accuracy(p, "B", 1) == 1, "top-1 should hit B")
    _assert(top_k_accuracy(p, "C", 1) == 0, "top-1 should miss C")
    _assert(top_k_accuracy(p, "C", 2) == 1, "top-2 should hit C (ranked 2nd)")
    _assert(top_k_accuracy(p, "A", 2) == 0, "top-2 should miss A")


# ── Aggregation ───────────────────────────────────────────────────────────────


def test_aggregate_binary_samples() -> None:
    m, s = aggregate_binary_samples([0.1, 0.2, 0.3])
    _assert(_approx(m, 0.2), f"mean should be 0.2, got {m}")
    _assert(s > 0, "std should be > 0 for varying samples")
    m, s = aggregate_binary_samples([0.5])
    _assert(m == 0.5 and s == 0.0, "single sample: std should be 0")


def test_aggregate_action_samples() -> None:
    s1 = {"A": 0.1, "B": 0.9}
    s2 = {"A": 0.3, "B": 0.7}
    agg = aggregate_action_samples([s1, s2])
    _assert(_approx(agg["A"], 0.2), f"A should be 0.2, got {agg['A']}")
    _assert(_approx(agg["B"], 0.8), f"B should be 0.8, got {agg['B']}")
    _assert(_approx(sum(agg.values()), 1.0), "aggregate should sum to 1")


# ── ECE ───────────────────────────────────────────────────────────────────────


def test_ece_perfectly_calibrated() -> None:
    # 100 predictions at p=0.7, 70 are true. Single bucket at bin 7. Confidence=0.7, accuracy=0.7 → ECE=0.
    preds = [(0.7, 1)] * 70 + [(0.7, 0)] * 30
    ece = ece_binary(preds, n_bins=10)
    _assert(_approx(ece, 0.0, tol=1e-6), f"perfectly calibrated should have ECE=0, got {ece}")


def test_ece_miscalibrated() -> None:
    # Confidence 0.9 but accuracy 0.5 → ECE = 0.4
    preds = [(0.9, 1)] * 5 + [(0.9, 0)] * 5
    ece = ece_binary(preds, n_bins=10)
    _assert(_approx(ece, 0.4), f"expected ECE=0.4, got {ece}")


# ── Murphy decomposition ──────────────────────────────────────────────────────


def test_murphy_identity() -> None:
    """Brier = reliability - resolution + uncertainty (when binned consistently)."""
    preds = [(0.1, 0), (0.2, 0), (0.3, 1), (0.4, 0), (0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1), (0.9, 1)]
    rel, res, unc = murphy_decomposition(preds, n_bins=10)
    # Brier computed by the same binning
    # Since each bin has 1 item, reliability = (1/9) * sum((conf_i - acc_i)^2)
    # Here acc_i is literally y_i, so this equals mean Brier EXACTLY when bins don't group.
    brier = sum(brier_binary(p, y) for p, y in preds) / len(preds)
    identity = rel - res + unc
    _assert(
        _approx(brier, identity, tol=1e-6),
        f"Brier={brier}, rel-res+unc={identity}, diff={brier-identity}",
    )


def test_murphy_ranges() -> None:
    preds = [(0.7, 1)] * 70 + [(0.7, 0)] * 30
    rel, res, unc = murphy_decomposition(preds, n_bins=10)
    # Perfectly calibrated, no resolution (single bucket, no discrimination)
    _assert(_approx(rel, 0.0, tol=1e-6), f"reliability should be 0, got {rel}")
    _assert(_approx(res, 0.0, tol=1e-6), f"resolution should be 0 (single bucket), got {res}")
    _assert(_approx(unc, 0.21), f"uncertainty should be 0.21 (0.7*0.3), got {unc}")


# ── Temperature scaling ───────────────────────────────────────────────────────


def test_temperature_binary_no_change() -> None:
    # Well-calibrated input: T=1.0 should win
    preds = [(0.7, 1)] * 70 + [(0.7, 0)] * 30
    T = fit_temperature_binary(preds)
    _assert(0.9 < T < 1.1, f"well-calibrated should fit T≈1, got {T}")


def test_temperature_binary_overconfident() -> None:
    # Overconfident: model says 0.99 but true rate is 0.5 → T should be > 1 (cool it down)
    preds = [(0.99, 1)] * 50 + [(0.99, 0)] * 50
    T = fit_temperature_binary(preds)
    _assert(T > 1.5, f"overconfident preds should fit T>1.5 (cool down), got {T}")


def test_apply_temperature_binary_identity() -> None:
    _assert(_approx(_apply_temperature_binary(0.7, 1.0), 0.7, tol=1e-4), "T=1 must be identity")
    # T→∞ pushes toward 0.5
    _assert(_apply_temperature_binary(0.9, 100.0) < 0.6, "high T should pull toward 0.5")


def test_apply_temperature_multi_identity() -> None:
    p = {"A": 0.5, "B": 0.3, "C": 0.2}
    out = _apply_temperature_multi(p, 1.0)
    for k in p:
        _assert(_approx(out[k], p[k], tol=1e-4), f"T=1 should be identity on {k}")


def test_fit_temperature_multi() -> None:
    # Calibrated predictions → T≈1
    preds = [({"A": 0.6, "B": 0.4}, "A")] * 60 + [({"A": 0.6, "B": 0.4}, "B")] * 40
    T = fit_temperature_multi(preds)
    _assert(0.7 < T < 1.4, f"calibrated multiclass should fit T≈1, got {T}")


# ── Baselines ─────────────────────────────────────────────────────────────────


def test_baselines() -> None:
    _assert(coin_flip_brier_binary(1) == 0.25, "coin flip Brier = 0.25")
    _assert(coin_flip_brier_binary(0) == 0.25, "coin flip Brier = 0.25")
    _assert(_approx(uniform_brier_multiclass(["A", "B", "C", "D"], "A"), 0.75), "uniform-4 on truth A → 0.75")
    _assert(_approx(base_rate_brier_binary(0.8, 1), 0.04), "0.8 base rate on YES → 0.04")


# ── End-to-end: score_experiment on synthetic JSONL ───────────────────────────


def test_score_experiment_end_to_end() -> None:
    """Write synthetic predictions + resolutions + manifest, score, verify outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 3 binary questions, 2 action_selection questions
        resolutions = {
            "resolutions": [
                {"question_id": "B1", "question_type": "binary", "correct_answer": "YES", "resolution_status": "resolved_yes"},
                {"question_id": "B2", "question_type": "binary", "correct_answer": "NO",  "resolution_status": "resolved_no"},
                {"question_id": "B3", "question_type": "binary", "correct_answer": "YES", "resolution_status": "resolved_yes"},
                {"question_id": "A1", "question_type": "action_selection", "correct_answer": "A", "resolution_status": "resolved_option"},
                {"question_id": "A2", "question_type": "action_selection", "correct_answer": "B", "resolution_status": "resolved_option"},
            ]
        }
        manifest = {
            "questions": [
                {"question_id": "B1", "base_rate_estimate": 0.5},
                {"question_id": "B2", "base_rate_estimate": 0.5},
                {"question_id": "B3", "base_rate_estimate": 0.5},
                {"question_id": "A1", "base_rate_estimate": None},
                {"question_id": "A2", "base_rate_estimate": None},
            ]
        }
        (tmp / "resolutions.json").write_text(json.dumps(resolutions))
        (tmp / "manifest.json").write_text(json.dumps(manifest))

        # Predictions: 2 samples per question, all perfect to verify plumbing
        pred_dir = tmp / "preds"
        pred_dir.mkdir()
        rows = []
        for qid in ["B1", "B3"]:
            for i in range(2):
                rows.append({
                    "question_id": qid, "experiment": "e3", "sample_idx": i,
                    "question_format": "binary",
                    "model_id": "test", "temperature": 1.0,
                    "prompt_hash": "x", "briefing_hash": None,
                    "binary": {"probability": 0.9, "reasoning": "r"}, "action": None,
                    "raw_response": "r", "tokens_in": 1, "tokens_out": 1, "latency_ms": 1,
                    "created_at": "2026-04-19T00:00:00", "error": None,
                })
        for i in range(2):
            rows.append({
                "question_id": "B2", "experiment": "e3", "sample_idx": i,
                "question_format": "binary",
                "model_id": "test", "temperature": 1.0,
                "prompt_hash": "x", "briefing_hash": None,
                "binary": {"probability": 0.1, "reasoning": "r"}, "action": None,
                "raw_response": "r", "tokens_in": 1, "tokens_out": 1, "latency_ms": 1,
                "created_at": "2026-04-19T00:00:00", "error": None,
            })
        for qid, truth_letter in [("A1", "A"), ("A2", "B")]:
            for i in range(2):
                dist = {"A": 0.0, "B": 0.0, "C": 0.0}
                dist[truth_letter] = 1.0
                rows.append({
                    "question_id": qid, "experiment": "e3", "sample_idx": i,
                    "question_format": "action_selection",
                    "model_id": "test", "temperature": 1.0,
                    "prompt_hash": "x", "briefing_hash": None,
                    "binary": None, "action": {"probabilities": dist, "reasoning": "r"},
                    "raw_response": "r", "tokens_in": 1, "tokens_out": 1, "latency_ms": 1,
                    "created_at": "2026-04-19T00:00:00", "error": None,
                })
        (pred_dir / "preds.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

        summary = score_experiment(
            predictions_path=pred_dir,
            resolutions_path=tmp / "resolutions.json",
            manifest_path=tmp / "manifest.json",
            experiment_id="e3",
        )

        _assert(summary["n_questions_scored"] == 5, f"expected 5, got {summary['n_questions_scored']}")
        _assert(summary["n_binary"] == 3, f"expected 3 binary, got {summary['n_binary']}")
        _assert(summary["n_action"] == 2, f"expected 2 action, got {summary['n_action']}")
        _assert(summary["n_missing_predictions"] == 0, "no predictions should be missing")
        # Binary at 0.9 (true) and 0.1 (true NO): brier = mean(0.01, 0.01, 0.01) = 0.01
        # Action at 1.0 (true): brier = 0.0
        # Overall mean = (3*0.01 + 2*0.0) / 5 = 0.006
        _assert(_approx(summary["brier_raw"], 0.006, tol=1e-6),
                f"expected brier_raw=0.006, got {summary['brier_raw']}")
        _assert(summary["top1_accuracy"] == 1.0, "all action predictions are perfect → top1=1.0")
        _assert(summary["coin_flip_brier"] == 0.25, "binary coin flip baseline should be 0.25")
        _assert(_approx(summary["uniform_brier"], 0.6666666, tol=1e-4),
                f"uniform over 3 options → truth=A → (2/3)^2 + 2*(1/3)^2 = 0.6666, got {summary['uniform_brier']}")


# ── Runner ────────────────────────────────────────────────────────────────────


TESTS = [
    test_brier_binary_endpoints,
    test_brier_multiclass,
    test_log_loss_binary,
    test_log_loss_multiclass,
    test_top_k_accuracy,
    test_aggregate_binary_samples,
    test_aggregate_action_samples,
    test_ece_perfectly_calibrated,
    test_ece_miscalibrated,
    test_murphy_identity,
    test_murphy_ranges,
    test_temperature_binary_no_change,
    test_temperature_binary_overconfident,
    test_apply_temperature_binary_identity,
    test_apply_temperature_multi_identity,
    test_fit_temperature_multi,
    test_baselines,
    test_score_experiment_end_to_end,
]


def main() -> None:
    failed = []
    for t in TESTS:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"ERR   {t.__name__}: {type(e).__name__}: {e}")
            failed.append(t.__name__)
    print()
    print(f"{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
