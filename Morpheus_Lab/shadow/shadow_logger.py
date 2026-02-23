"""
Morpheus Lab — Shadow Trade Logger
======================================
Logs shadow trades to JSONL and produces daily summary comparisons
against production Morpheus trades.

Output files:
  logs/shadow_flush_reclaim.jsonl  — individual shadow trades
  logs/shadow_daily_summary.jsonl  — daily aggregated comparison

NO PRODUCTION MODIFICATION. Read-only access to production logs.
"""

import json
import os
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ShadowTradeLog:
    """Single shadow trade record."""
    timestamp: str           # ISO format entry time
    symbol: str
    regime: str
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str         # target, stop, eod
    pnl: float
    pnl_pct: float
    hold_seconds: float
    risk_pct: float          # (entry - stop) / entry
    reward_risk: float       # target distance / stop distance
    share_size: int
    strategy: str = "flush_reclaim_v1"
    mode: str = "SHADOW"
    exit_timestamp: str = ""


@dataclass
class DailySummary:
    """Daily performance summary for comparison."""
    date: str
    source: str              # "shadow" or "production"
    trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    avg_hold_seconds: float
    best_trade: float
    worst_trade: float


class ShadowLogger:
    """
    Append-only JSONL logger for shadow trades.

    Thread-safe via file append mode. No in-memory accumulation
    beyond current-day buffer for daily summary.
    """

    def __init__(
        self,
        trade_log_path: str = "logs/shadow_flush_reclaim.jsonl",
        summary_log_path: str = "logs/shadow_daily_summary.jsonl",
        production_log_path: str = "logs/production_trades.jsonl",
    ):
        self.trade_log_path = Path(trade_log_path)
        self.summary_log_path = Path(summary_log_path)
        self.production_log_path = Path(production_log_path)

        # Ensure directories exist
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Current day buffer for daily summary
        self._today: Optional[date] = None
        self._day_trades: List[ShadowTradeLog] = []

        logger.info(f"Shadow logger initialized: {self.trade_log_path}")

    def log_trade(self, trade: ShadowTradeLog) -> None:
        """Append a single shadow trade to JSONL log."""
        # Write to file (append mode)
        with open(self.trade_log_path, "a") as f:
            f.write(json.dumps(asdict(trade)) + "\n")

        # Buffer for daily summary
        trade_date = datetime.fromisoformat(trade.timestamp).date()

        if self._today is None:
            self._today = trade_date

        if trade_date != self._today:
            # New day — flush previous day's summary
            self._flush_daily_summary()
            self._today = trade_date
            self._day_trades = []

        self._day_trades.append(trade)

        logger.info(
            f"SHADOW {trade.symbol} "
            f"{'WIN' if trade.pnl > 0 else 'LOSS'} "
            f"${trade.entry_price:.2f}->${trade.exit_price:.2f} "
            f"PnL=${trade.pnl:+.2f} ({trade.exit_reason}) "
            f"[{trade.regime}]"
        )

    def flush(self) -> None:
        """Force flush current day's summary (call at EOD or shutdown)."""
        if self._day_trades:
            self._flush_daily_summary()
            self._day_trades = []

    def _flush_daily_summary(self) -> None:
        """Compute and write daily summary for shadow trades."""
        if not self._day_trades:
            return

        shadow_summary = self._compute_summary(
            self._day_trades, self._today, "shadow"
        )

        # Try to load production trades for comparison
        prod_summary = self._load_production_summary(self._today)

        # Write shadow summary
        with open(self.summary_log_path, "a") as f:
            f.write(json.dumps(asdict(shadow_summary)) + "\n")
            if prod_summary:
                f.write(json.dumps(asdict(prod_summary)) + "\n")

        # Print comparison
        self._print_comparison(shadow_summary, prod_summary)

    def _compute_summary(
        self, trades: List[ShadowTradeLog], day: date, source: str
    ) -> DailySummary:
        """Compute daily summary metrics from trade list."""
        pnls = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        gross_wins = sum(winners) if winners else 0
        gross_losses = abs(sum(losers)) if losers else 0

        pf = (gross_wins / gross_losses) if gross_losses > 0 else float('inf')

        return DailySummary(
            date=day.isoformat(),
            source=source,
            trades=len(trades),
            winners=len(winners),
            losers=len(losers),
            win_rate=(round(len(winners) / len(trades) * 100, 1)) if trades else 0,
            total_pnl=round(total_pnl, 2),
            avg_pnl=round(total_pnl / len(trades), 2) if trades else 0,
            profit_factor=round(pf, 2) if pf != float('inf') else 999.0,
            avg_hold_seconds=round(
                sum(t.hold_seconds for t in trades) / len(trades), 1
            ) if trades else 0,
            best_trade=round(max(pnls), 2) if pnls else 0,
            worst_trade=round(min(pnls), 2) if pnls else 0,
        )

    def _load_production_summary(self, day: date) -> Optional[DailySummary]:
        """
        Load production Morpheus trades for the given day.

        Reads from production_log_path (JSONL) and filters to matching date.
        Returns None if no production log or no trades for that day.
        """
        if not self.production_log_path.exists():
            logger.debug(f"No production log at {self.production_log_path}")
            return None

        day_str = day.isoformat()
        prod_pnls = []
        prod_holds = []

        try:
            with open(self.production_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Match on date (check timestamp or date field)
                    rec_date = record.get("date", "")
                    if not rec_date:
                        ts = record.get("timestamp", "")
                        if ts:
                            try:
                                rec_date = datetime.fromisoformat(ts).date().isoformat()
                            except (ValueError, TypeError):
                                continue

                    if rec_date == day_str:
                        pnl = record.get("pnl", record.get("realized_pnl", 0))
                        prod_pnls.append(float(pnl))
                        hold = record.get("hold_seconds", record.get("duration_s", 0))
                        prod_holds.append(float(hold))

        except Exception as e:
            logger.warning(f"Error reading production log: {e}")
            return None

        if not prod_pnls:
            return None

        winners = [p for p in prod_pnls if p > 0]
        losers = [p for p in prod_pnls if p <= 0]
        total = sum(prod_pnls)
        gross_w = sum(winners) if winners else 0
        gross_l = abs(sum(losers)) if losers else 0
        pf = (gross_w / gross_l) if gross_l > 0 else 999.0

        return DailySummary(
            date=day_str,
            source="production",
            trades=len(prod_pnls),
            winners=len(winners),
            losers=len(losers),
            win_rate=round(len(winners) / len(prod_pnls) * 100, 1),
            total_pnl=round(total, 2),
            avg_pnl=round(total / len(prod_pnls), 2),
            profit_factor=round(pf, 2),
            avg_hold_seconds=round(
                sum(prod_holds) / len(prod_holds), 1
            ) if prod_holds else 0,
            best_trade=round(max(prod_pnls), 2),
            worst_trade=round(min(prod_pnls), 2),
        )

    def _print_comparison(
        self, shadow: DailySummary, production: Optional[DailySummary]
    ) -> None:
        """Print side-by-side daily comparison."""
        print(f"\n{'='*60}")
        print(f"  SHADOW vs PRODUCTION — {shadow.date}")
        print(f"{'='*60}")

        def _row(label, s_val, p_val):
            p_str = f"{p_val}" if production else "N/A"
            print(f"  {label:<20} {'Shadow':>10}  {'Production':>10}")
            print(f"  {'':<20} {str(s_val):>10}  {p_str:>10}")

        print(f"  {'Metric':<20} {'Shadow':>10}  {'Production':>10}")
        print(f"  {'─'*50}")

        p = production
        print(f"  {'Trades':<20} {shadow.trades:>10}  {p.trades if p else 'N/A':>10}")
        print(f"  {'Win Rate':<20} {shadow.win_rate:>9.1f}%  {f'{p.win_rate:.1f}%' if p else 'N/A':>10}")
        print(f"  {'Total PnL':<20} ${shadow.total_pnl:>9.2f}  {f'${p.total_pnl:.2f}' if p else 'N/A':>10}")
        print(f"  {'Profit Factor':<20} {shadow.profit_factor:>10.2f}  {f'{p.profit_factor:.2f}' if p else 'N/A':>10}")
        print(f"  {'Avg Hold (s)':<20} {shadow.avg_hold_seconds:>10.1f}  {f'{p.avg_hold_seconds:.1f}' if p else 'N/A':>10}")
        print(f"  {'Best Trade':<20} ${shadow.best_trade:>9.2f}  {f'${p.best_trade:.2f}' if p else 'N/A':>10}")
        print(f"  {'Worst Trade':<20} ${shadow.worst_trade:>9.2f}  {f'${p.worst_trade:.2f}' if p else 'N/A':>10}")

        print(f"  {'─'*50}")

        if production:
            delta = shadow.total_pnl - production.total_pnl
            print(f"  {'Shadow Edge':<20} ${delta:>+9.2f}")

        print(f"{'='*60}\n")


def load_shadow_history(
    log_path: str = "logs/shadow_flush_reclaim.jsonl",
) -> List[Dict]:
    """Load all shadow trades from JSONL for analysis."""
    path = Path(log_path)
    if not path.exists():
        return []

    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return trades


def load_daily_summaries(
    log_path: str = "logs/shadow_daily_summary.jsonl",
) -> List[Dict]:
    """Load all daily summaries for trend analysis."""
    path = Path(log_path)
    if not path.exists():
        return []

    summaries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    summaries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return summaries
