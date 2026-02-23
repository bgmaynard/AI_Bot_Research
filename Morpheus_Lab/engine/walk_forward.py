"""
Morpheus Lab — Walk-Forward Analyzer
========================================
Validates edge stability by splitting data into train/validate periods.

Process:
  1. Split date range: train (70%) / validate (30%)
  2. Run grid on train period → find top N candidates
  3. Test top candidates on validation period
  4. Compute stability ratio: OOS_PF / IS_PF

Stability Ratio Definition:
  ratio = out_of_sample_profit_factor / in_sample_profit_factor
  
  Interpretation:
    ≥ 0.8 → Stable edge (parameter set is robust)
    0.5-0.8 → Degraded (some overfitting, use with caution)
    < 0.5 → Unstable (likely overfit, do not promote)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Type

from core.dbn_loader import DatabentoTradeLoader
from engine.grid_engine import GridEngine, expand_grid
from engine.edge_metrics import EdgeMetrics, compute_edge_metrics
from strategies.batch_strategy import BatchStrategy, BatchTrade

logger = logging.getLogger(__name__)


def split_date_range(
    start_date: str,
    end_date: str,
    train_pct: float = 0.7,
) -> tuple:
    """
    Split date range into train/validate periods.

    Returns: (train_start, train_end, val_start, val_end)
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end - start).days
    train_days = max(1, int(total_days * train_pct))

    train_end = start + timedelta(days=train_days)
    val_start = train_end + timedelta(days=1)

    return (
        start_date,
        train_end.strftime("%Y-%m-%d"),
        val_start.strftime("%Y-%m-%d"),
        end_date,
    )


class WalkForwardResult:
    """Results from a walk-forward analysis."""

    def __init__(self):
        self.train_start: str = ""
        self.train_end: str = ""
        self.val_start: str = ""
        self.val_end: str = ""
        self.candidates: List[Dict] = []  # top N from train
        self.train_metrics: List[EdgeMetrics] = []
        self.val_metrics: List[EdgeMetrics] = []

    def to_dict(self) -> Dict:
        """Serializable dict for JSON output."""
        results = {
            "train_period": f"{self.train_start} to {self.train_end}",
            "validation_period": f"{self.val_start} to {self.val_end}",
            "candidates": [],
        }

        for i, (train_m, val_m) in enumerate(zip(self.train_metrics, self.val_metrics)):
            is_pf = train_m.profit_factor
            oos_pf = val_m.profit_factor

            # Stability ratio
            if is_pf > 0 and is_pf != float('inf'):
                stability = oos_pf / is_pf if oos_pf != float('inf') else 0.0
            else:
                stability = 0.0

            candidate = {
                "rank": i + 1,
                "params": train_m.params,
                "in_sample": {
                    "trades": train_m.total_trades,
                    "win_rate": round(train_m.win_rate, 1),
                    "profit_factor": round(is_pf, 2) if is_pf != float('inf') else "inf",
                    "total_pnl": round(train_m.total_pnl, 2),
                    "max_drawdown": round(train_m.max_drawdown, 2),
                    "sharpe": round(train_m.sharpe, 2),
                    "rr": round(train_m.reward_risk, 2),
                },
                "out_of_sample": {
                    "trades": val_m.total_trades,
                    "win_rate": round(val_m.win_rate, 1),
                    "profit_factor": round(oos_pf, 2) if oos_pf != float('inf') else "inf",
                    "total_pnl": round(val_m.total_pnl, 2),
                    "max_drawdown": round(val_m.max_drawdown, 2),
                    "sharpe": round(val_m.sharpe, 2),
                    "rr": round(val_m.reward_risk, 2),
                },
                "stability_ratio": round(stability, 3),
                "parameter_drift": self._assess_drift(train_m, val_m),
            }
            results["candidates"].append(candidate)

        return results

    def _assess_drift(self, train: EdgeMetrics, val: EdgeMetrics) -> str:
        """Assess parameter drift between IS and OOS."""
        notes = []

        # Win rate drift
        wr_diff = abs(train.win_rate - val.win_rate)
        if wr_diff > 15:
            notes.append(f"Win rate shifted {wr_diff:.0f}pp")
        elif wr_diff > 8:
            notes.append(f"Win rate drifted {wr_diff:.0f}pp")

        # R:R drift
        if train.reward_risk > 0:
            rr_ratio = val.reward_risk / train.reward_risk if train.reward_risk > 0 else 0
            if rr_ratio < 0.5:
                notes.append(f"R:R collapsed ({train.reward_risk:.1f}→{val.reward_risk:.1f})")
            elif rr_ratio < 0.75:
                notes.append(f"R:R degraded ({train.reward_risk:.1f}→{val.reward_risk:.1f})")

        # Trade count
        if train.total_trades > 0:
            trade_ratio = val.total_trades / train.total_trades
            # Normalize by period length (val is ~30% of total)
            expected_ratio = 0.43  # 30/70
            if trade_ratio < expected_ratio * 0.5:
                notes.append("Far fewer trades OOS (regime change?)")
            elif trade_ratio > expected_ratio * 2:
                notes.append("Far more trades OOS (overfitting filter?)")

        return "; ".join(notes) if notes else "Stable"

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Walk-Forward Analysis",
            f"  Train:    {self.train_start} → {self.train_end}",
            f"  Validate: {self.val_start} → {self.val_end}",
            f"  Candidates tested: {len(self.train_metrics)}",
            f"{'─' * 60}",
        ]

        for i, (tm, vm) in enumerate(zip(self.train_metrics, self.val_metrics)):
            is_pf = tm.profit_factor
            oos_pf = vm.profit_factor
            stability = (oos_pf / is_pf) if (is_pf > 0 and is_pf != float('inf') and oos_pf != float('inf')) else 0.0

            is_pf_str = f"{is_pf:.2f}" if is_pf != float('inf') else "inf"
            oos_pf_str = f"{oos_pf:.2f}" if oos_pf != float('inf') else "inf"

            lines.append(
                f"  #{i+1}  IS_PF={is_pf_str:>6}  OOS_PF={oos_pf_str:>6}  "
                f"Stability={stability:.3f}  "
                f"Trades={tm.total_trades}/{vm.total_trades}  "
                f"Params={tm.params}"
            )

        return "\n".join(lines)


class WalkForwardAnalyzer:
    """
    Walk-forward validation engine.

    Splits data into train/validate, runs grid on train,
    tests top candidates on validate, computes stability.
    """

    def __init__(self, loader: DatabentoTradeLoader):
        self.loader = loader
        self.grid_engine = GridEngine(loader)

    def analyze(
        self,
        strategy_class: Type[BatchStrategy],
        param_grid: List[Dict],
        symbols: List[str],
        start_date: str,
        end_date: str,
        train_pct: float = 0.7,
        top_n: int = 5,
        min_trades: int = 10,
        progress_callback=None,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        1. Split dates into train/validate
        2. Grid search on train period
        3. Test top N candidates on validation period
        4. Compute stability ratios
        """
        result = WalkForwardResult()

        # Split dates
        train_start, train_end, val_start, val_end = split_date_range(
            start_date, end_date, train_pct
        )
        result.train_start = train_start
        result.train_end = train_end
        result.val_start = val_start
        result.val_end = val_end

        logger.info(f"Train: {train_start} → {train_end}")
        logger.info(f"Validate: {val_start} → {val_end}")

        # Phase 1: Grid on train period
        logger.info(f"Running grid: {len(param_grid)} combos on train period")
        train_results = self.grid_engine.run_grid(
            strategy_class=strategy_class,
            param_grid=param_grid,
            symbols=symbols,
            start_date=train_start,
            end_date=train_end,
            min_trades=min_trades,
            progress_callback=progress_callback,
        )

        # Filter by min trades and take top N
        qualified = [m for m in train_results if m.total_trades >= min_trades]
        candidates = qualified[:top_n]

        if not candidates:
            logger.warning("No candidates met minimum trade threshold in train period")
            return result

        # Phase 2: Test candidates on validation period
        logger.info(f"Testing {len(candidates)} candidates on validation period")

        from core.dbn_loader import _date_to_ns

        ts_start = _date_to_ns(val_start)
        ts_end = _date_to_ns(val_end, end_of_day=True)

        for train_m in candidates:
            strategy = strategy_class(**train_m.params)
            all_trades: List[BatchTrade] = []
            total_events = 0

            import time
            combo_start = time.perf_counter()

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
                    start_date=val_start, end_date=val_end,
                )

            combo_elapsed = time.perf_counter() - combo_start

            val_m = compute_edge_metrics(
                trades=all_trades,
                strategy_name=strategy.name,
                params=train_m.params,
                symbols=[s.upper() for s in symbols],
                start_date=val_start,
                end_date=val_end,
                total_events=total_events,
                elapsed_seconds=combo_elapsed,
            )

            result.train_metrics.append(train_m)
            result.val_metrics.append(val_m)

        return result

    def save_results(self, result: WalkForwardResult, filepath: str) -> None:
        """Save walk-forward results to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Walk-forward results saved: {filepath}")
