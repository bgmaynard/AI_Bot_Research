"""
Morpheus Lab — Promotion Scoring Engine
==========================================
Weighted composite scoring for promotion decisions.

Weights (default):
  - Expectancy improvement: 35%
  - Stability improvement:  25%
  - Drawdown reduction:     20%
  - Regime consistency:     10%
  - Execution robustness:   10%

Thresholds:
  ≥ 0.75 → validated
  0.65–0.75 → candidate
  < 0.65 → rejected
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Default weights — configurable
DEFAULT_WEIGHTS = {
    "expectancy_improvement": 0.35,
    "stability_improvement": 0.25,
    "drawdown_improvement": 0.20,
    "regime_consistency": 0.10,
    "execution_robustness": 0.10,
}

# Default thresholds — configurable
DEFAULT_THRESHOLDS = {
    "validated": 0.75,
    "candidate": 0.65,
}


@dataclass
class PromotionResult:
    """Result of promotion scoring."""
    hypothesis_id: str
    promotion_score: float
    status: str  # "validated", "candidate", or "rejected"
    component_scores: Dict[str, float]
    weights_used: Dict[str, float]
    thresholds_used: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_promotion_score(
    hypothesis_id: str,
    expectancy_improvement: float,
    stability_improvement: float,
    drawdown_improvement: float,
    regime_consistency: float,
    execution_robustness: float,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> PromotionResult:
    """
    Compute weighted promotion score and status.

    All input scores should be normalized to 0-1 range.

    Args:
        hypothesis_id: ID of the hypothesis being scored.
        expectancy_improvement: 0-1 normalized expectancy improvement.
        stability_improvement: 0-1 normalized stability improvement.
        drawdown_improvement: 0-1 normalized drawdown reduction.
        regime_consistency: 0-1 regime consistency score.
        execution_robustness: 0-1 execution robustness score.
        weights: Optional custom weights (must sum to 1.0).
        thresholds: Optional custom thresholds.

    Returns:
        PromotionResult with score and status.
    """
    w = weights or DEFAULT_WEIGHTS.copy()
    t = thresholds or DEFAULT_THRESHOLDS.copy()

    # Validate weights sum to ~1.0
    weight_sum = sum(w.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(f"Weights sum to {weight_sum}, not 1.0. Normalizing.")
        for key in w:
            w[key] /= weight_sum

    component_scores = {
        "expectancy_improvement": round(expectancy_improvement, 4),
        "stability_improvement": round(stability_improvement, 4),
        "drawdown_improvement": round(drawdown_improvement, 4),
        "regime_consistency": round(regime_consistency, 4),
        "execution_robustness": round(execution_robustness, 4),
    }

    # Weighted composite
    score = (
        w["expectancy_improvement"] * expectancy_improvement
        + w["stability_improvement"] * stability_improvement
        + w["drawdown_improvement"] * drawdown_improvement
        + w["regime_consistency"] * regime_consistency
        + w["execution_robustness"] * execution_robustness
    )

    score = round(score, 4)

    # Determine status
    if score >= t["validated"]:
        status = "validated"
    elif score >= t["candidate"]:
        status = "candidate"
    else:
        status = "rejected"

    result = PromotionResult(
        hypothesis_id=hypothesis_id,
        promotion_score=score,
        status=status,
        component_scores=component_scores,
        weights_used=w,
        thresholds_used=t,
    )

    logger.info(
        f"Promotion score for {hypothesis_id}: "
        f"{score:.4f} → {status.upper()}"
    )

    return result


def save_scoring_result(result: PromotionResult, output_dir: str) -> str:
    """
    Save promotion scoring result to the appropriate directory.

    Args:
        result: PromotionResult to save.
        output_dir: Base results directory.

    Returns:
        Path to saved scoring file.
    """
    # Route to appropriate subdirectory
    base = Path(output_dir)
    if result.status == "validated":
        dest = base / "validated"
    elif result.status == "candidate":
        dest = base / "candidates"
    else:
        dest = base / "rejected"

    dest.mkdir(parents=True, exist_ok=True)

    filepath = dest / f"{result.hypothesis_id}_scoring.json"
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Scoring result saved: {filepath}")
    return str(filepath)


def load_scoring_result(filepath: str) -> PromotionResult:
    """Load a previously saved scoring result."""
    with open(filepath) as f:
        data = json.load(f)
    return PromotionResult(**data)
