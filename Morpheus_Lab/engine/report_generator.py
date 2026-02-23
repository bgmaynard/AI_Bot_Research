"""
Morpheus Lab — Report Generator
===================================
Creates structured reports for every backtest, grid, and walk-forward run.

Outputs:
  reports/
    strategy_signature.json   — identity + hash
    metrics.json              — full EdgeMetrics
    trades.csv                — trade log
    equity_curve.csv          — cumulative PnL series
"""

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from strategies.batch_strategy import BatchTrade
from engine.edge_metrics import EdgeMetrics

logger = logging.getLogger(__name__)


def compute_signature(
    strategy_name: str,
    params: Dict,
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    Compute deterministic hash signature for a strategy run.
    Same inputs always produce the same hash.
    """
    payload = json.dumps({
        "strategy": strategy_name,
        "params": params,
        "symbols": sorted(symbols),
        "start": start_date,
        "end": end_date,
    }, sort_keys=True)

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ReportGenerator:
    """Generates structured report artifacts for a backtest run."""

    def __init__(self, base_dir: str = "reports"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        metrics: EdgeMetrics,
        trades: List[BatchTrade],
        run_type: str = "backtest",  # 'backtest', 'grid', 'walkforward'
    ) -> str:
        """
        Generate full report package.

        Returns: report directory path.
        """
        # Create unique directory
        sig = compute_signature(
            metrics.strategy_name,
            metrics.params,
            metrics.symbols,
            metrics.start_date,
            metrics.end_date,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirname = f"{run_type}_{metrics.strategy_name}_{sig}_{timestamp}"
        report_dir = self.base_dir / dirname
        report_dir.mkdir(parents=True, exist_ok=True)

        # 1. Strategy signature
        self._write_signature(report_dir, metrics, sig, run_type)

        # 2. Metrics
        self._write_metrics(report_dir, metrics)

        # 3. Trades CSV
        self._write_trades(report_dir, trades)

        # 4. Equity curve
        self._write_equity_curve(report_dir, metrics)

        logger.info(f"Report generated: {report_dir}")
        return str(report_dir)

    def _write_signature(
        self,
        report_dir: Path,
        metrics: EdgeMetrics,
        sig: str,
        run_type: str,
    ) -> None:
        """Write strategy_signature.json."""
        data = {
            "strategy_name": metrics.strategy_name,
            "parameter_set": metrics.params,
            "symbols": metrics.symbols,
            "date_range": {
                "start": metrics.start_date,
                "end": metrics.end_date,
            },
            "hash_signature": sig,
            "timestamp": datetime.now().isoformat(),
            "run_type": run_type,
            "engine": "morpheus_lab",
            "source": "research_pc",
        }

        with open(report_dir / "strategy_signature.json", "w") as f:
            json.dump(data, f, indent=2)

    def _write_metrics(self, report_dir: Path, metrics: EdgeMetrics) -> None:
        """Write metrics.json."""
        data = {
            "core": {
                "total_trades": metrics.total_trades,
                "winners": metrics.winners,
                "losers": metrics.losers,
                "scratches": metrics.scratches,
                "win_rate": round(metrics.win_rate, 2),
            },
            "pnl": {
                "total_pnl": round(metrics.total_pnl, 2),
                "avg_pnl": round(metrics.avg_pnl, 2),
                "avg_winner": round(metrics.avg_winner, 2),
                "avg_loser": round(metrics.avg_loser, 2),
                "median_pnl": round(metrics.median_pnl, 2),
                "pnl_stddev": round(metrics.pnl_stddev, 2),
            },
            "ratios": {
                "reward_risk": round(metrics.reward_risk, 2),
                "profit_factor": round(metrics.profit_factor, 2) if metrics.profit_factor != float('inf') else "inf",
                "expectancy": round(metrics.expectancy, 4),
            },
            "drawdown": {
                "max_drawdown": round(metrics.max_drawdown, 2),
                "max_drawdown_pct": round(metrics.max_drawdown_pct, 1),
                "max_drawdown_trades": metrics.max_drawdown_trades,
            },
            "time": {
                "avg_hold_seconds": round(metrics.avg_hold_seconds, 1),
                "median_hold_seconds": round(metrics.median_hold_seconds, 1),
                "avg_winner_hold": round(metrics.avg_winner_hold, 1),
                "avg_loser_hold": round(metrics.avg_loser_hold, 1),
            },
            "risk_adjusted": {
                "sharpe": round(metrics.sharpe, 2),
                "calmar": round(metrics.calmar, 2) if metrics.calmar != float('inf') else "inf",
            },
            "exit_reasons": metrics.exit_reasons,
            "engine": {
                "total_events": metrics.total_events,
                "elapsed_seconds": round(metrics.elapsed_seconds, 3),
                "throughput": int(metrics.throughput),
            },
        }

        with open(report_dir / "metrics.json", "w") as f:
            json.dump(data, f, indent=2)

    def _write_trades(self, report_dir: Path, trades: List[BatchTrade]) -> None:
        """Write trades.csv."""
        if not trades:
            return

        fieldnames = [
            "trade_num", "symbol", "direction", "entry_price", "exit_price",
            "size", "pnl", "pnl_pct", "exit_reason", "hold_seconds",
            "entry_ts", "exit_ts", "won", "entry_regime", "entry_type",
        ]

        with open(report_dir / "trades.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, t in enumerate(trades):
                writer.writerow({
                    "trade_num": i + 1,
                    "symbol": t.symbol,
                    "direction": "LONG" if t.direction == 1 else "SHORT",
                    "entry_price": round(t.entry_price, 4),
                    "exit_price": round(t.exit_price, 4),
                    "size": t.size,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 2),
                    "exit_reason": t.exit_reason,
                    "hold_seconds": round(t.hold_seconds, 1),
                    "entry_ts": t.entry_ts,
                    "exit_ts": t.exit_ts,
                    "won": t.won,
                    "entry_regime": t.entry_regime,
                    "entry_type": getattr(t, 'entry_type', ''),
                })

    def _write_equity_curve(self, report_dir: Path, metrics: EdgeMetrics) -> None:
        """Write equity_curve.csv."""
        if not metrics.equity_curve:
            return

        with open(report_dir / "equity_curve.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trade_num", "cumulative_pnl"])
            for i, eq in enumerate(metrics.equity_curve):
                writer.writerow([i + 1, round(eq, 2)])
