"""
Morpheus Lab — Event Types
============================
Lightweight immutable event structures.
No business logic here.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """Single trade tick at nanosecond resolution."""
    ts: int         # nanosecond timestamp
    symbol: str
    price: float
    size: int
