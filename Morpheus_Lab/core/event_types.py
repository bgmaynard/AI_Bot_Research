"""
Morpheus Lab — Event Types (Optimized)
=========================================
TradeEvent is a named tuple for minimal allocation overhead.
Nanosecond timestamps stay as raw int — no conversions in hot path.
"""

from collections import namedtuple

# TradeEvent: (ts, symbol, price, size)
# ts = nanosecond int, price = float, size = int
TradeEvent = namedtuple("TradeEvent", ["ts", "symbol", "price", "size"])
