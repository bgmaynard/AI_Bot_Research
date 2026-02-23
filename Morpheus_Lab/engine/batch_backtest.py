"""
Morpheus Lab — Batch Backtest Engine
=======================================
Connects the replay engine to batch strategies via numpy array callbacks.

Architecture:
    DatabentoTradeLoader → batch callback → Strategy.on_batch() → BatchResult

Supports:
  - Regime classification (tag each trade with market regime at entry)
  - Regime filtering (--only-regime: skip trades outside target regime)
  - Regime breakdown (metrics per regime)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.dbn_loader import DatabentoTradeLoader, _date_to_ns
from strategies.batch_strategy import BatchStrategy, BatchResult, BatchTrade

logger = logging.getLogger(__name__)


class BatchBacktestEngine:
    """
    High-speed vectorized backtest engine.

    Uses batch callbacks to pass numpy arrays directly to strategies.
    """

    def __init__(self, loader: DatabentoTradeLoader):
        self.loader = loader

    def run(
        self,
        strategy: BatchStrategy,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        regime_breakdown: bool = False,
        only_regime: Optional[str] = None,
        regime_window: int = 200,
    ) -> BatchResult:
        """
        Run a batch backtest over symbols and date range.

        Args:
            strategy: Strategy instance
            symbols: List of symbols
            start_date, end_date: Date range
            regime_breakdown: If True, classify regimes and tag trades
            only_regime: If set, only keep trades entering in this regime
            regime_window: Lookback for regime classifier
        """
        result = BatchResult(
            strategy_name=strategy.name,
            params=dict(strategy.params),
        )

        ts_start = _date_to_ns(start_date) if start_date else 0
        ts_end = _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1

        do_regime = regime_breakdown or (only_regime is not None)

        wall_start = time.perf_counter()
        symbols_with_data = set()

        for sym in symbols:
            strategy.reset()

            def _on_batch(ts_arr, price_arr, size_arr, symbol):
                nonlocal result
                trades = strategy.on_batch(ts_arr, price_arr, size_arr, symbol)

                if trades and do_regime:
                    from engine.regime_classifier import classify_tick_regime
                    regimes = classify_tick_regime(
                        price_arr, size_arr, window=regime_window
                    )
                    trades = _tag_trades_with_regime(
                        trades, ts_arr, regimes, only_regime
                    )

                if trades:
                    result.trades.extend(trades)
                result.batches_processed += 1
                result.total_events += len(ts_arr)
                symbols_with_data.add(symbol)

            self.loader.replay_symbol_batch_callback(
                sym, _on_batch,
                start_ts=ts_start, end_ts=ts_end,
                start_date=start_date, end_date=end_date,
            )

        result.symbols_processed = len(symbols_with_data)
        result.elapsed_seconds = round(time.perf_counter() - wall_start, 6)

        return result

    def run_grid(
        self,
        strategy_class,
        param_grid: List[dict],
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[BatchResult]:
        """Run multiple parameter combinations."""
        results = []

        for i, params in enumerate(param_grid):
            strategy = strategy_class(**params)
            logger.info(f"Grid [{i+1}/{len(param_grid)}] {params}")
            result = self.run(strategy, symbols, start_date, end_date)
            results.append(result)

        results.sort(key=lambda r: r.total_pnl, reverse=True)
        return results


def _tag_trades_with_regime(
    trades: List[BatchTrade],
    ts_arr: np.ndarray,
    regimes: np.ndarray,
    only_regime: Optional[str] = None,
) -> List[BatchTrade]:
    """
    Tag each trade with the regime at its entry timestamp.
    Optionally filter to only keep trades in a specific regime.

    Uses binary search to find the tick index closest to each trade's entry_ts.
    """
    if not trades:
        return trades

    tagged = []
    for trade in trades:
        # Find closest tick index to entry timestamp
        idx = np.searchsorted(ts_arr, trade.entry_ts, side='left')
        idx = min(idx, len(regimes) - 1)

        regime = str(regimes[idx])
        trade.entry_regime = regime

        if only_regime is not None:
            if regime == only_regime:
                tagged.append(trade)
        else:
            tagged.append(trade)

    return tagged


def compute_regime_breakdown(
    trades: List[BatchTrade],
) -> Dict:
    """
    Compute metrics broken down by regime.

    Returns dict: {regime_name: {trades, win_rate, pf, avg_pnl, ...}}
    """
    from engine.edge_metrics import compute_edge_metrics

    # Group trades by regime
    regime_groups: Dict[str, List[BatchTrade]] = {}
    for t in trades:
        regime = t.entry_regime or "UNCLASSIFIED"
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(t)

    breakdown = {}
    for regime, group in sorted(regime_groups.items()):
        m = compute_edge_metrics(group, strategy_name="", params={})

        breakdown[regime] = {
            "trades": m.total_trades,
            "winners": m.winners,
            "losers": m.losers,
            "win_rate": round(m.win_rate, 1),
            "total_pnl": round(m.total_pnl, 2),
            "avg_pnl": round(m.avg_pnl, 2),
            "avg_winner": round(m.avg_winner, 2),
            "avg_loser": round(m.avg_loser, 2),
            "rr": round(m.reward_risk, 2),
            "profit_factor": round(m.profit_factor, 2) if m.profit_factor != float('inf') else "inf",
            "max_drawdown": round(m.max_drawdown, 2),
            "avg_hold_seconds": round(m.avg_hold_seconds, 1),
            "expectancy": round(m.expectancy, 4),
            "sharpe": round(m.sharpe, 2),
        }

    return breakdown


def save_regime_breakdown(
    breakdown: Dict,
    filepath: str,
    params: Optional[Dict] = None,
    symbols: Optional[List[str]] = None,
) -> None:
    """Save regime breakdown to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "params": params or {},
        "symbols": symbols or [],
        "regimes": breakdown,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
