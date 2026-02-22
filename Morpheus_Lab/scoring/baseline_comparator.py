"""
Morpheus Lab — Baseline Comparator
=====================================
Compares candidate results against baseline.

Absolute profit comparisons are forbidden.
Relative improvements only.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    with open(path) as f:
        return json.load(f)


def compute_improvement_deltas(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute improvement deltas between candidate and baseline.

    All comparisons are relative (percentage improvement).
    Absolute profit comparison is forbidden by framework.

    Args:
        baseline: Baseline metrics dict (from MetricsResult.to_dict()).
        candidate: Candidate metrics dict.

    Returns:
        Dict with normalized improvement scores for each metric.
    """
    deltas = {}

    # --- Expectancy improvement ---
    base_exp = baseline.get("expectancy", 0)
    cand_exp = candidate.get("expectancy", 0)
    deltas["expectancy_delta"] = cand_exp - base_exp
    deltas["expectancy_pct_change"] = _safe_pct_change(base_exp, cand_exp)

    # --- Win rate improvement ---
    base_wr = baseline.get("win_rate", 0)
    cand_wr = candidate.get("win_rate", 0)
    deltas["win_rate_delta"] = cand_wr - base_wr
    deltas["win_rate_pct_change"] = _safe_pct_change(base_wr, cand_wr)

    # --- Profit factor improvement ---
    base_pf = baseline.get("profit_factor", 0)
    cand_pf = candidate.get("profit_factor", 0)
    deltas["profit_factor_delta"] = cand_pf - base_pf
    deltas["profit_factor_pct_change"] = _safe_pct_change(base_pf, cand_pf)

    # --- Drawdown reduction (lower is better) ---
    base_dd = baseline.get("max_drawdown", 0)
    cand_dd = candidate.get("max_drawdown", 0)
    deltas["drawdown_delta"] = base_dd - cand_dd  # Positive = improvement
    deltas["drawdown_pct_change"] = _safe_pct_change(base_dd, cand_dd, invert=True)

    # --- Variance reduction (lower is better) ---
    base_var = baseline.get("daily_variance", 0)
    cand_var = candidate.get("daily_variance", 0)
    deltas["variance_delta"] = base_var - cand_var  # Positive = improvement
    deltas["variance_pct_change"] = _safe_pct_change(base_var, cand_var, invert=True)

    # --- Sharpe improvement ---
    base_sharpe = baseline.get("sharpe_like_ratio", 0)
    cand_sharpe = candidate.get("sharpe_like_ratio", 0)
    deltas["sharpe_delta"] = cand_sharpe - base_sharpe
    deltas["sharpe_pct_change"] = _safe_pct_change(base_sharpe, cand_sharpe)

    # --- Worst day improvement (less negative is better) ---
    base_worst = baseline.get("worst_day_95pct", 0)
    cand_worst = candidate.get("worst_day_95pct", 0)
    deltas["worst_day_delta"] = cand_worst - base_worst  # Less negative = positive delta
    deltas["worst_day_improved"] = cand_worst > base_worst

    return deltas


def normalize_improvements(deltas: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize improvement deltas to 0-1 scale for promotion scoring.

    Mapping:
        0.0 = no improvement or degradation
        0.5 = moderate improvement
        1.0 = exceptional improvement

    Returns:
        Dict with normalized scores for each dimension.
    """
    normalized = {}

    # Expectancy: cap at ±100% change
    exp_pct = deltas.get("expectancy_pct_change", 0)
    normalized["expectancy_improvement"] = _clamp_normalize(exp_pct, -1.0, 1.0)

    # Stability: combination of variance and sharpe improvement
    var_pct = deltas.get("variance_pct_change", 0)
    sharpe_pct = deltas.get("sharpe_pct_change", 0)
    stability = (var_pct + sharpe_pct) / 2
    normalized["stability_improvement"] = _clamp_normalize(stability, -1.0, 1.0)

    # Drawdown control
    dd_pct = deltas.get("drawdown_pct_change", 0)
    normalized["drawdown_improvement"] = _clamp_normalize(dd_pct, -1.0, 1.0)

    return normalized


def compare_regime_consistency(
    baseline_regimes: Dict[str, Dict[str, Any]],
    candidate_regimes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare regime-level performance between baseline and candidate.

    A candidate that improves in one regime but degrades in others
    receives a low consistency score.

    Args:
        baseline_regimes: Dict of regime -> metrics for baseline.
        candidate_regimes: Dict of regime -> metrics for candidate.

    Returns:
        Dict with regime consistency analysis.
    """
    regime_deltas = {}
    improvements = 0
    degradations = 0
    total = 0

    for regime in baseline_regimes:
        if regime in candidate_regimes:
            base_exp = baseline_regimes[regime].get("expectancy", 0)
            cand_exp = candidate_regimes[regime].get("expectancy", 0)

            delta = cand_exp - base_exp
            regime_deltas[regime] = {
                "baseline_expectancy": base_exp,
                "candidate_expectancy": cand_exp,
                "delta": round(delta, 4),
                "improved": delta > 0,
            }

            if delta > 0:
                improvements += 1
            elif delta < 0:
                degradations += 1
            total += 1

    # Consistency score: proportion of regimes that improved (or held)
    consistency = (total - degradations) / total if total > 0 else 0.0

    return {
        "regime_deltas": regime_deltas,
        "regimes_improved": improvements,
        "regimes_degraded": degradations,
        "consistency_score": round(consistency, 4),
    }


def _safe_pct_change(base: float, candidate: float, invert: bool = False) -> float:
    """
    Compute percentage change. For inverted metrics (where lower is better),
    flips the sign so positive = improvement.
    """
    if base == 0:
        if candidate == 0:
            return 0.0
        return 1.0 if (not invert and candidate > 0) or (invert and candidate < 0) else -1.0

    pct = (candidate - base) / abs(base)

    if invert:
        pct = -pct  # Lower is better, so reduction = positive improvement

    return round(pct, 4)


def _clamp_normalize(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range and normalize to 0-1."""
    clamped = max(min_val, min(max_val, value))
    return round((clamped - min_val) / (max_val - min_val), 4)
