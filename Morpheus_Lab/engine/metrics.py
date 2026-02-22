"""
Morpheus Lab — Metrics Calculator
===================================
Computes all required performance metrics from a list of trade results.

No incomplete metric sets allowed.
"""

import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Complete metric set for a collection of trades."""
    total_pnl: float
    expectancy: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    worst_day_95pct: float
    daily_variance: float
    trade_count: int
    avg_trade_pnl: float
    sharpe_like_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics(trades: List[Dict[str, Any]]) -> MetricsResult:
    """
    Compute complete metrics from a list of trade dicts.

    Each trade dict must contain at minimum:
        - pnl: float (profit/loss of the trade)
        - date: str (YYYY-MM-DD date of the trade)

    Args:
        trades: List of trade dicts.

    Returns:
        MetricsResult with all required metrics.
    """
    if not trades:
        return MetricsResult(
            total_pnl=0.0,
            expectancy=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            worst_day_95pct=0.0,
            daily_variance=0.0,
            trade_count=0,
            avg_trade_pnl=0.0,
            sharpe_like_ratio=0.0,
        )

    trade_count = len(trades)
    pnls = [t["pnl"] for t in trades]

    # --- Total P&L ---
    total_pnl = sum(pnls)

    # --- Win Rate ---
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / trade_count if trade_count > 0 else 0.0

    # --- Expectancy ---
    expectancy = total_pnl / trade_count if trade_count > 0 else 0.0

    # --- Profit Factor ---
    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # --- Average Trade P&L ---
    avg_trade_pnl = total_pnl / trade_count if trade_count > 0 else 0.0

    # --- Max Drawdown ---
    max_drawdown = _compute_max_drawdown(pnls)

    # --- Daily P&L aggregation ---
    daily_pnl = _aggregate_daily_pnl(trades)
    daily_values = list(daily_pnl.values())

    # --- 95th percentile worst day ---
    worst_day_95pct = _percentile_worst_day(daily_values, 0.95)

    # --- Daily Variance ---
    daily_variance = _variance(daily_values)

    # --- Sharpe-like ratio ---
    sharpe_like_ratio = _sharpe_like(daily_values)

    result = MetricsResult(
        total_pnl=round(total_pnl, 4),
        expectancy=round(expectancy, 4),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        max_drawdown=round(max_drawdown, 4),
        worst_day_95pct=round(worst_day_95pct, 4),
        daily_variance=round(daily_variance, 4),
        trade_count=trade_count,
        avg_trade_pnl=round(avg_trade_pnl, 4),
        sharpe_like_ratio=round(sharpe_like_ratio, 4),
    )

    logger.debug(
        f"Metrics: {trade_count} trades, "
        f"WR={result.win_rate:.1%}, "
        f"Exp={result.expectancy:.4f}, "
        f"PF={result.profit_factor:.2f}"
    )

    return result


def _compute_max_drawdown(pnls: List[float]) -> float:
    """
    Compute maximum drawdown from a sequence of trade P&Ls.
    Returns the maximum peak-to-trough decline (as a positive number).
    """
    if not pnls:
        return 0.0

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for pnl in pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    return max_dd


def _aggregate_daily_pnl(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate trade P&Ls by date.

    Returns:
        Dict mapping date string to total daily P&L.
    """
    daily = {}
    for t in trades:
        date = t.get("date", "unknown")
        daily[date] = daily.get(date, 0.0) + t["pnl"]
    return daily


def _percentile_worst_day(daily_values: List[float], percentile: float) -> float:
    """
    Compute the worst-day P&L at the given percentile.
    95th percentile worst day = 5th percentile of daily P&L distribution.
    """
    if not daily_values:
        return 0.0

    sorted_vals = sorted(daily_values)
    idx = max(0, int(math.floor((1 - percentile) * len(sorted_vals))))
    return sorted_vals[idx]


def _variance(values: List[float]) -> float:
    """Compute population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _sharpe_like(daily_values: List[float]) -> float:
    """
    Compute a Sharpe-like ratio: mean(daily P&L) / std(daily P&L).
    Not annualized — just a consistency measure.
    """
    if len(daily_values) < 2:
        return 0.0

    mean = sum(daily_values) / len(daily_values)
    var = _variance(daily_values)
    std = math.sqrt(var) if var > 0 else 0.0

    if std == 0:
        return 0.0

    return mean / std
