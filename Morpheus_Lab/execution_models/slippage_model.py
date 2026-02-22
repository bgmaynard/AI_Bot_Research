"""
Morpheus Lab — Execution Sensitivity Model
=============================================
Applies slippage, latency delay, and order type simulation to trade results.

This module is injectable — configured via hypothesis file, not hardcoded.
"""

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def apply_execution_model(
    trades: List[Dict[str, Any]],
    slippage: float = 0.0,
    latency_ms: int = 0,
    order_type: str = "market",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Apply execution modeling to a list of trades.

    Simulates real-world execution friction by adjusting entry/exit
    prices based on slippage, latency, and order type.

    Args:
        trades: List of trade dicts. Each must have:
                - entry_price: float
                - exit_price: float
                - pnl: float
                - direction: str ("long" or "short")
                - shares: int (optional, default 100)
        slippage: Per-share slippage in dollars (e.g., 0.03).
        latency_ms: Simulated latency in milliseconds.
                    Adds additional adverse price movement.
        order_type: "market" or "limit".
                    Limit orders may fail to fill (trade removed).
        seed: Random seed for deterministic latency modeling.

    Returns:
        New list of trades with adjusted prices and P&L.
    """
    if seed is not None:
        random.seed(seed)

    adjusted_trades = []
    skipped = 0

    for trade in trades:
        adjusted = trade.copy()
        direction = trade.get("direction", "long")
        shares = trade.get("shares", 100)
        entry = trade["entry_price"]
        exit_price = trade["exit_price"]

        # --- Slippage ---
        entry_slip, exit_slip = _compute_slippage(slippage, direction)

        # --- Latency impact ---
        latency_impact = _compute_latency_impact(latency_ms, direction)

        # --- Order type modeling ---
        if order_type == "limit":
            # Limit orders may not fill if price moves adversely
            fill_probability = _limit_fill_probability(slippage, latency_ms)
            if random.random() > fill_probability:
                skipped += 1
                continue

        # Apply adjustments
        adjusted_entry = entry + entry_slip + latency_impact
        adjusted_exit = exit_price - exit_slip - latency_impact

        # Recalculate P&L
        if direction == "long":
            new_pnl = (adjusted_exit - adjusted_entry) * shares
        else:
            new_pnl = (adjusted_entry - adjusted_exit) * shares

        adjusted["entry_price"] = round(adjusted_entry, 4)
        adjusted["exit_price"] = round(adjusted_exit, 4)
        adjusted["pnl"] = round(new_pnl, 4)
        adjusted["original_pnl"] = trade["pnl"]
        adjusted["slippage_applied"] = slippage
        adjusted["latency_applied"] = latency_ms
        adjusted["execution_drag"] = round(trade["pnl"] - new_pnl, 4)

        adjusted_trades.append(adjusted)

    if skipped > 0:
        logger.info(f"Limit order modeling: {skipped}/{len(trades)} fills skipped")

    logger.info(
        f"Execution model applied: slippage=${slippage}, "
        f"latency={latency_ms}ms, type={order_type}, "
        f"trades={len(adjusted_trades)}/{len(trades)}"
    )

    return adjusted_trades


def _compute_slippage(slippage: float, direction: str) -> tuple:
    """
    Compute entry and exit slippage.

    For longs: entry slips UP, exit slips DOWN (both adverse).
    For shorts: entry slips DOWN, exit slips UP (both adverse).

    Returns:
        (entry_slippage, exit_slippage) as dollar amounts.
    """
    if direction == "long":
        return (slippage, slippage)  # Pay more entry, receive less exit
    else:
        return (-slippage, -slippage)  # Receive less entry, pay more exit


def _compute_latency_impact(latency_ms: int, direction: str) -> float:
    """
    Model adverse price movement during latency window.

    Assumes average adverse movement of $0.005 per 100ms of latency
    for low-float momentum stocks. This is conservative.

    Calibrate this based on your actual tape speed data.
    """
    if latency_ms <= 0:
        return 0.0

    # Adverse movement rate: dollars per millisecond
    adverse_rate = 0.00005  # $0.005 per 100ms

    impact = latency_ms * adverse_rate

    if direction == "long":
        return impact  # Price moves up before you fill
    else:
        return -impact  # Price moves down before you fill


def _limit_fill_probability(slippage: float, latency_ms: int) -> float:
    """
    Estimate limit order fill probability.

    Higher slippage environment → less likely limit fills.
    Higher latency → less likely limit fills.

    Returns probability between 0 and 1.
    """
    base_fill = 0.85
    slippage_penalty = slippage * 5  # Each $0.01 slippage = 5% less fill
    latency_penalty = latency_ms * 0.001  # Each ms = 0.1% less fill

    probability = max(0.1, base_fill - slippage_penalty - latency_penalty)
    return probability


def compute_execution_robustness(
    base_metrics: Dict[str, Any],
    stressed_metrics_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute execution robustness score.

    Measures how much strategy degrades under execution stress.

    Args:
        base_metrics: Metrics at zero slippage/latency.
        stressed_metrics_list: List of metrics under various stress levels.

    Returns:
        Dict with robustness scores.
    """
    if not stressed_metrics_list:
        return {"robustness_score": 1.0, "degradation": {}}

    base_exp = base_metrics.get("expectancy", 0)

    degradations = []
    for stressed in stressed_metrics_list:
        stressed_exp = stressed.get("expectancy", 0)
        if base_exp != 0:
            deg = (base_exp - stressed_exp) / abs(base_exp)
        else:
            deg = 0.0
        degradations.append(deg)

    avg_degradation = sum(degradations) / len(degradations) if degradations else 0.0
    max_degradation = max(degradations) if degradations else 0.0

    # Robustness = 1.0 (no degradation) to 0.0 (complete collapse)
    robustness = max(0.0, 1.0 - avg_degradation)

    # Strategy collapses if any stressed variant goes negative
    collapsed = any(m.get("expectancy", 0) < 0 for m in stressed_metrics_list)

    return {
        "robustness_score": round(robustness, 4),
        "avg_degradation": round(avg_degradation, 4),
        "max_degradation": round(max_degradation, 4),
        "collapsed_under_stress": collapsed,
        "variants_tested": len(stressed_metrics_list),
    }
