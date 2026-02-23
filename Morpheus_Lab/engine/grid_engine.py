"""
Morpheus Lab — Parameter Grid Engine
========================================
Systematic parameter sweep for edge discovery.

Parses parameter combinations from CLI, runs batch backtest
for each combo, collects EdgeMetrics, outputs ranked CSV.

Usage:
    engine = GridEngine(loader)
    results = engine.run_grid(
        strategy_class=MomentumBreakout,
        param_grid=param_grid,
        symbols=["CISS"],
        start_date="2026-01-30",
        end_date="2026-02-06",
    )
"""

import csv
import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Type

from core.dbn_loader import DatabentoTradeLoader, _date_to_ns
from engine.edge_metrics import EdgeMetrics, compute_edge_metrics
from strategies.batch_strategy import BatchStrategy, BatchResult, BatchTrade

logger = logging.getLogger(__name__)


def parse_grid_string(grid_str: str) -> Dict[str, List]:
    """
    Parse CLI grid string into parameter dict.

    Format: 'lookback=50,100,200 target_pct=2,3,4 stop_pct=0.5,1.0'
    Returns: {'lookback': [50, 100, 200], 'target_pct': [2.0, 3.0, 4.0], ...}
    """
    params = {}
    for token in grid_str.split():
        if "=" not in token:
            continue
        key, values_str = token.split("=", 1)
        values = []
        for v in values_str.split(","):
            v = v.strip()
            if not v:
                continue
            # Try int first, then float
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)
        params[key] = values
    return params


def expand_grid(param_dict: Dict[str, List]) -> List[Dict]:
    """
    Expand parameter dict into list of all combinations.

    Input:  {'lookback': [50, 100], 'target_pct': [2, 3]}
    Output: [{'lookback': 50, 'target_pct': 2},
             {'lookback': 50, 'target_pct': 3},
             {'lookback': 100, 'target_pct': 2},
             {'lookback': 100, 'target_pct': 3}]
    """
    if not param_dict:
        return [{}]

    keys = sorted(param_dict.keys())
    value_lists = [param_dict[k] for k in keys]

    combos = []
    for values in product(*value_lists):
        combo = dict(zip(keys, values))
        combos.append(combo)

    return combos


class GridEngine:
    """
    Systematic parameter sweep engine.

    Runs batch backtest for each parameter combination,
    collects EdgeMetrics, ranks results.
    """

    def __init__(self, loader: DatabentoTradeLoader):
        self.loader = loader

    def run_grid(
        self,
        strategy_class: Type[BatchStrategy],
        param_grid: List[Dict],
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_trades: int = 10,
        progress_callback=None,
    ) -> List[EdgeMetrics]:
        """
        Run all parameter combinations and collect metrics.

        Args:
            strategy_class: BatchStrategy subclass (not instance)
            param_grid: List of param dicts (from expand_grid)
            symbols: Symbols to test
            start_date, end_date: Date range
            min_trades: Minimum trades to include in results
            progress_callback: Optional fn(combo_idx, total, params, metrics)

        Returns:
            List of EdgeMetrics, sorted by profit_factor descending.
        """
        ts_start = _date_to_ns(start_date) if start_date else 0
        ts_end = _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1

        results = []
        total = len(param_grid)
        wall_start = time.perf_counter()

        for idx, params in enumerate(param_grid):
            combo_start = time.perf_counter()

            strategy = strategy_class(**params)
            all_trades: List[BatchTrade] = []
            total_events = 0

            for sym in symbols:
                strategy.reset()

                def _on_batch(ts_arr, price_arr, size_arr, symbol):
                    nonlocal total_events
                    trades = strategy.on_batch(ts_arr, price_arr, size_arr, symbol)
                    if trades:
                        all_trades.extend(trades)
                    total_events += len(ts_arr)

                self.loader.replay_symbol_batch_callback(
                    sym, _on_batch,
                    start_ts=ts_start, end_ts=ts_end,
                    start_date=start_date, end_date=end_date,
                )

            combo_elapsed = time.perf_counter() - combo_start

            metrics = compute_edge_metrics(
                trades=all_trades,
                strategy_name=strategy.name,
                params=params,
                symbols=[s.upper() for s in symbols],
                start_date=start_date or "",
                end_date=end_date or "",
                total_events=total_events,
                elapsed_seconds=combo_elapsed,
            )

            results.append(metrics)

            if progress_callback:
                progress_callback(idx + 1, total, params, metrics)

        # Sort: profit_factor desc, then max_drawdown asc
        results.sort(key=lambda m: (-m.profit_factor if m.profit_factor != float('inf') else -999, m.max_drawdown))

        total_elapsed = time.perf_counter() - wall_start
        logger.info(f"Grid complete: {total} combos in {total_elapsed:.1f}s")

        return results

    def save_grid_csv(
        self,
        results: List[EdgeMetrics],
        filepath: str,
        min_trades: int = 10,
    ) -> int:
        """
        Save grid results to CSV.

        Returns number of rows written.
        """
        # Filter by min trades
        filtered = [m for m in results if m.total_trades >= min_trades]

        if not filtered:
            logger.warning("No results met minimum trade threshold")
            return 0

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get column headers from first result
        first_row = filtered[0].to_row()
        fieldnames = list(first_row.keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in filtered:
                writer.writerow(m.to_row())

        return len(filtered)

    def save_grid_json(
        self,
        results: List[EdgeMetrics],
        filepath: str,
        min_trades: int = 10,
    ) -> int:
        """Save grid results to JSON."""
        filtered = [m for m in results if m.total_trades >= min_trades]

        if not filtered:
            return 0

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [m.to_row() for m in filtered]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return len(filtered)
