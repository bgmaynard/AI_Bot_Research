"""
Morpheus Lab — Regime Segmenter
=================================
Detects and segments trades by market regime.

Regimes:
  - dead_tape
  - momentum
  - catalyst_heavy
  - ssr_heavy
  - mixed
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Valid regime labels
VALID_REGIMES = ["dead_tape", "momentum", "catalyst_heavy", "ssr_heavy", "mixed"]


def classify_regime(trade: Dict[str, Any]) -> str:
    """
    Classify a single trade's regime.

    This function checks for an existing 'regime' field on the trade.
    If not present, it uses heuristics based on available fields.

    Override this function to integrate your existing regime classifier.

    Args:
        trade: Trade dict with optional fields:
               - regime: str (pre-classified)
               - relative_volume: float
               - has_catalyst: bool
               - is_ssr: bool
               - tape_speed: float (trades per minute or similar)

    Returns:
        Regime string label.
    """
    # If pre-classified, validate and return
    if "regime" in trade and trade["regime"] in VALID_REGIMES:
        return trade["regime"]

    # --- Heuristic classification ---
    # These thresholds should be calibrated to your data.
    # Replace with your actual regime classifier integration.

    is_ssr = trade.get("is_ssr", False)
    has_catalyst = trade.get("has_catalyst", False)
    rvol = trade.get("relative_volume", 1.0)
    tape_speed = trade.get("tape_speed", 0)

    if is_ssr:
        return "ssr_heavy"

    if has_catalyst:
        return "catalyst_heavy"

    if rvol >= 3.0 and tape_speed >= 100:
        return "momentum"

    if rvol < 1.5 and tape_speed < 30:
        return "dead_tape"

    return "mixed"


def segment_trades_by_regime(trades: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Segment a list of trades by regime.

    Args:
        trades: List of trade dicts.

    Returns:
        Dict mapping regime name to list of trades in that regime.
        All valid regimes are present as keys (may have empty lists).
    """
    segmented = {regime: [] for regime in VALID_REGIMES}

    for trade in trades:
        regime = classify_regime(trade)
        segmented[regime].append(trade)

    # Log distribution
    for regime, regime_trades in segmented.items():
        if regime_trades:
            logger.info(f"Regime '{regime}': {len(regime_trades)} trades")

    return segmented


def attach_regime_tags(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach regime tags to each trade (in-place mutation + return).
    Useful for pre-processing before segmentation.

    Args:
        trades: List of trade dicts.

    Returns:
        Same list with 'regime' field set on each trade.
    """
    for trade in trades:
        if "regime" not in trade or trade["regime"] not in VALID_REGIMES:
            trade["regime"] = classify_regime(trade)

    return trades
