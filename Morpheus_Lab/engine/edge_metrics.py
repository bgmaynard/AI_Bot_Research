"""
Morpheus Lab — Edge Metrics
===============================
Comprehensive metrics for edge discovery and validation.

Computes all metrics needed for:
  - Grid result ranking
  - Walk-forward stability analysis
  - Promotion gate decisions

All metrics are computed from a list of BatchTrade objects.
No market data needed — pure trade-level analysis.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from strategies.batch_strategy import BatchTrade


@dataclass
class EdgeMetrics:
    """Complete metrics suite for a parameter combination."""

    # Identity
    strategy_name: str = ""
    params: Dict = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""

    # Core counts
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    scratches: int = 0  # PnL == 0

    # Win rate
    win_rate: float = 0.0

    # PnL
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    median_pnl: float = 0.0
    pnl_stddev: float = 0.0

    # Ratios
    reward_risk: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # avg_pnl per trade (same as avg_pnl but explicit)

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0  # DD as % of peak equity
    max_drawdown_trades: int = 0  # trades from peak to trough

    # Time
    avg_hold_seconds: float = 0.0
    median_hold_seconds: float = 0.0
    avg_winner_hold: float = 0.0
    avg_loser_hold: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0  # annualized (assuming 252 trading days)
    calmar: float = 0.0  # total_pnl / max_drawdown

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)

    # Exit analysis
    exit_reasons: Dict[str, int] = field(default_factory=dict)

    # Engine stats
    total_events: int = 0
    elapsed_seconds: float = 0.0
    throughput: float = 0.0

    def to_dict(self) -> Dict:
        """Flat dict for CSV/JSON export. Excludes equity_curve."""
        d = {}
        for k, v in self.__dict__.items():
            if k == 'equity_curve':
                continue
            if isinstance(v, (dict, list)):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    def to_row(self) -> Dict:
        """Flat dict with params expanded for CSV grid output."""
        row = {}
        # Expand params into columns
        for k, v in self.params.items():
            row[f"p_{k}"] = v
        # Core metrics
        row["trades"] = self.total_trades
        row["win_rate"] = round(self.win_rate, 1)
        row["total_pnl"] = round(self.total_pnl, 2)
        row["avg_pnl"] = round(self.avg_pnl, 2)
        row["avg_winner"] = round(self.avg_winner, 2)
        row["avg_loser"] = round(self.avg_loser, 2)
        row["rr"] = round(self.reward_risk, 2)
        row["profit_factor"] = round(self.profit_factor, 2)
        row["expectancy"] = round(self.expectancy, 4)
        row["max_drawdown"] = round(self.max_drawdown, 2)
        row["max_dd_pct"] = round(self.max_drawdown_pct, 1)
        row["avg_hold_s"] = round(self.avg_hold_seconds, 1)
        row["pnl_stddev"] = round(self.pnl_stddev, 2)
        row["sharpe"] = round(self.sharpe, 2)
        row["calmar"] = round(self.calmar, 2)
        row["symbols"] = ",".join(self.symbols)
        row["throughput"] = int(self.throughput)
        return row


def compute_edge_metrics(
    trades: List[BatchTrade],
    strategy_name: str = "",
    params: Optional[Dict] = None,
    symbols: Optional[List[str]] = None,
    start_date: str = "",
    end_date: str = "",
    total_events: int = 0,
    elapsed_seconds: float = 0.0,
) -> EdgeMetrics:
    """
    Compute full metrics suite from a list of trades.

    This is the single source of truth for all metrics in Morpheus Lab.
    """
    m = EdgeMetrics(
        strategy_name=strategy_name,
        params=params or {},
        symbols=symbols or [],
        start_date=start_date,
        end_date=end_date,
        total_events=total_events,
        elapsed_seconds=elapsed_seconds,
    )

    if not trades:
        return m

    # ── CORE COUNTS ──────────────────────────────────────

    pnls = [t.pnl for t in trades]
    pnl_arr = np.array(pnls)

    m.total_trades = len(trades)
    m.winners = sum(1 for p in pnls if p > 0)
    m.losers = sum(1 for p in pnls if p < 0)
    m.scratches = sum(1 for p in pnls if p == 0)

    decided = m.winners + m.losers
    m.win_rate = (m.winners / decided * 100) if decided > 0 else 0.0

    # ── PNL ──────────────────────────────────────────────

    m.total_pnl = float(pnl_arr.sum())
    m.avg_pnl = float(pnl_arr.mean())
    m.median_pnl = float(np.median(pnl_arr))
    m.pnl_stddev = float(pnl_arr.std()) if len(pnl_arr) > 1 else 0.0

    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]

    m.avg_winner = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    m.avg_loser = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0

    # ── RATIOS ───────────────────────────────────────────

    m.reward_risk = abs(m.avg_winner / m.avg_loser) if m.avg_loser != 0 else 0.0

    gross_win = sum(win_pnls) if win_pnls else 0.0
    gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0.0
    m.profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

    m.expectancy = m.avg_pnl  # per-trade expectancy in dollars

    # ── EQUITY CURVE & DRAWDOWN ──────────────────────────

    equity = np.cumsum(pnl_arr)
    m.equity_curve = equity.tolist()

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity  # always >= 0

    if len(drawdown) > 0:
        max_dd_idx = int(np.argmax(drawdown))
        m.max_drawdown = float(drawdown[max_dd_idx])

        # DD as percentage of peak at that point
        peak_at_dd = float(peak[max_dd_idx])
        m.max_drawdown_pct = (m.max_drawdown / peak_at_dd * 100) if peak_at_dd > 0 else 0.0

        # Trades from peak to trough
        if m.max_drawdown > 0:
            peak_idx = int(np.argmax(equity[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
            m.max_drawdown_trades = max_dd_idx - peak_idx

    # ── HOLD TIMES ───────────────────────────────────────

    holds = [t.hold_seconds for t in trades]
    m.avg_hold_seconds = sum(holds) / len(holds) if holds else 0.0
    m.median_hold_seconds = float(np.median(holds)) if holds else 0.0

    winner_holds = [t.hold_seconds for t in trades if t.won]
    loser_holds = [t.hold_seconds for t in trades if not t.won and t.pnl != 0]
    m.avg_winner_hold = sum(winner_holds) / len(winner_holds) if winner_holds else 0.0
    m.avg_loser_hold = sum(loser_holds) / len(loser_holds) if loser_holds else 0.0

    # ── RISK-ADJUSTED ────────────────────────────────────

    # Sharpe: annualized assuming ~252 trading days, ~20 trades/day as proxy
    if m.pnl_stddev > 0 and len(pnl_arr) > 1:
        daily_trades = max(len(pnl_arr), 1)
        m.sharpe = (m.avg_pnl / m.pnl_stddev) * math.sqrt(daily_trades)
    else:
        m.sharpe = 0.0

    # Calmar: total PnL / max drawdown
    m.calmar = m.total_pnl / m.max_drawdown if m.max_drawdown > 0 else float('inf')

    # ── EXIT ANALYSIS ────────────────────────────────────

    m.exit_reasons = {}
    for t in trades:
        m.exit_reasons[t.exit_reason] = m.exit_reasons.get(t.exit_reason, 0) + 1

    # ── THROUGHPUT ───────────────────────────────────────

    m.throughput = total_events / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return m
