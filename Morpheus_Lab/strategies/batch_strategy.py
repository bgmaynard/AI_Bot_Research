"""
Morpheus Lab — Batch Strategy Framework
==========================================
Base class for vectorized research strategies.

Strategies receive numpy arrays (ts, price, size) per symbol-day
and return trade signals + PnL using vectorized numpy operations.

No per-tick Python loops. Pure numpy in the hot path.

Usage:
    class MyStrategy(BatchStrategy):
        def on_batch(self, ts, price, size, symbol):
            # vectorized signal + PnL logic
            return trades  # list of BatchTrade

Architecture:
    Replay → on_batch(numpy arrays) → BatchTrade results → aggregate
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class BatchTrade:
    """Single trade result from a batch strategy."""
    symbol: str
    entry_ts: int          # nanosecond
    exit_ts: int           # nanosecond
    entry_price: float
    exit_price: float
    size: int
    direction: int         # 1 = long, -1 = short
    exit_reason: str       # 'target', 'stop', 'timeout', 'eod'
    entry_regime: str = "" # regime at entry (empty = not classified)
    entry_type: str = ""   # 'true_breakout', 'pullback', 'flush_reclaim', '' = unclassified

    @property
    def pnl(self) -> float:
        return self.direction * (self.exit_price - self.entry_price) * self.size

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.direction * (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def hold_ns(self) -> int:
        return self.exit_ts - self.entry_ts

    @property
    def hold_seconds(self) -> float:
        return self.hold_ns / 1_000_000_000

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class BatchResult:
    """Aggregated results from a batch backtest run."""
    strategy_name: str = ""
    params: dict = field(default_factory=dict)
    trades: List[BatchTrade] = field(default_factory=list)
    symbols_processed: int = 0
    batches_processed: int = 0
    total_events: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> int:
        return sum(1 for t in self.trades if t.won)

    @property
    def losers(self) -> int:
        return sum(1 for t in self.trades if not t.won and t.pnl != 0)

    @property
    def win_rate(self) -> float:
        total = self.winners + self.losers
        return self.winners / total * 100 if total > 0 else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / len(self.trades) if self.trades else 0.0

    @property
    def avg_winner(self) -> float:
        wins = [t.pnl for t in self.trades if t.won]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loser(self) -> float:
        losses = [t.pnl for t in self.trades if not t.won and t.pnl != 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def reward_risk(self) -> float:
        al = self.avg_loser
        return abs(self.avg_winner / al) if al != 0 else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.won)
        gross_loss = abs(sum(t.pnl for t in self.trades if not t.won and t.pnl != 0))
        return gross_win / gross_loss if gross_loss > 0 else float('inf')

    def summary(self) -> str:
        lines = [
            f"Strategy: {self.strategy_name}",
            f"Params:   {self.params}",
            f"─" * 50,
            f"Trades:       {self.total_trades:>8}",
            f"Winners:      {self.winners:>8}  ({self.win_rate:.1f}%)",
            f"Losers:       {self.losers:>8}",
            f"Total PnL:   ${self.total_pnl:>10,.2f}",
            f"Avg PnL:     ${self.avg_pnl:>10,.2f}",
            f"Avg Winner:  ${self.avg_winner:>10,.2f}",
            f"Avg Loser:   ${self.avg_loser:>10,.2f}",
            f"R:R:          {self.reward_risk:>8.2f}",
            f"Profit Fac:   {self.profit_factor:>8.2f}",
            f"─" * 50,
            f"Events:   {self.total_events:>12,}",
            f"Batches:  {self.batches_processed:>12}",
            f"Symbols:  {self.symbols_processed:>12}",
            f"Time:     {self.elapsed_seconds:>12.3f}s",
            f"Throughput:{self.total_events / self.elapsed_seconds if self.elapsed_seconds > 0 else 0:>11,.0f} evt/s",
        ]
        return "\n".join(lines)


class BatchStrategy:
    """
    Base class for vectorized batch strategies.

    Subclass and implement on_batch() with pure numpy operations.
    """

    name: str = "base"

    def __init__(self, **params):
        self.params = params

    def on_batch(
        self,
        ts: np.ndarray,
        price: np.ndarray,
        size: np.ndarray,
        symbol: str,
    ) -> List[BatchTrade]:
        """
        Process a batch of trades. Override in subclass.

        Args:
            ts: nanosecond timestamps (int64 array)
            price: float64 prices
            size: int64 trade sizes
            symbol: ticker string

        Returns:
            List of BatchTrade objects (generated trades)
        """
        raise NotImplementedError

    def on_day_end(self, symbol: str) -> List[BatchTrade]:
        """
        Called at end of each symbol-day. Override to close open positions.
        Default: no action.
        """
        return []

    def reset(self):
        """Reset strategy state between runs. Override if stateful."""
        pass
