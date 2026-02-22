"""
Morpheus Lab — Unified Event Types
=====================================
Standard event objects consumed by market replay and backtest engine.
All feeds must output these types regardless of source schema.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """Single trade tick."""
    ts: int              # Unix nanoseconds
    price: float
    size: int
    symbol: str
    side: Optional[str] = None   # "B", "A", or None
    conditions: Optional[str] = None

    @property
    def ts_seconds(self) -> float:
        return self.ts / 1e9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BarEvent:
    """OHLCV bar at a given timeframe."""
    ts: int              # Unix nanoseconds (bar close time)
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    timeframe: str       # "1s", "1m", "1h", "1d"

    @property
    def ts_seconds(self) -> float:
        return self.ts / 1e9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuoteEvent:
    """Top-of-book quote snapshot."""
    ts: int
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    symbol: str

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def ts_seconds(self) -> float:
        return self.ts / 1e9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
