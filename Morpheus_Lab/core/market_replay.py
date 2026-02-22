"""
Morpheus Lab — Market Replay Engine
======================================
Consumes events from DatabentoFeed and replays deterministically.

Features:
  - Multi-symbol interleaving by timestamp
  - Single-symbol fast loop mode for speed benchmarking
  - Deterministic replay (same data → same output every time)
  - Callback-based architecture for plugging in strategies
  - Event counting and progress reporting

Usage:
    feed = DatabentoFeed(cache_path)
    replay = MarketReplay(feed)

    # Register callbacks
    replay.on_bar(my_strategy.on_bar)
    replay.on_trade(my_strategy.on_trade)

    # Run
    stats = replay.run(symbols=["AAPL"], start="2026-01-01", end="2026-01-31")
"""

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from core.events import BarEvent, TradeEvent, QuoteEvent
from datafeeds.databento_feed import DatabentoFeed

logger = logging.getLogger(__name__)


@dataclass
class ReplayStats:
    """Statistics from a replay run."""
    total_events: int = 0
    bar_events: int = 0
    trade_events: int = 0
    symbols_seen: int = 0
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    elapsed_seconds: float = 0.0
    events_per_second: float = 0.0
    mode: str = ""

    def summary(self) -> str:
        return (
            f"Replay: {self.total_events:,} events "
            f"({self.bar_events:,} bars, {self.trade_events:,} trades) | "
            f"{self.symbols_seen} symbols | "
            f"{self.elapsed_seconds:.2f}s ({self.events_per_second:,.0f} evt/s) | "
            f"mode={self.mode}"
        )


class MarketReplay:
    """
    Deterministic market replay engine.

    Consumes events from a DatabentoFeed and dispatches
    to registered callbacks in timestamp order.
    """

    def __init__(self, feed: DatabentoFeed):
        self.feed = feed
        self._bar_callbacks: List[Callable[[BarEvent], None]] = []
        self._trade_callbacks: List[Callable[[TradeEvent], None]] = []
        self._any_callbacks: List[Callable[[Union[BarEvent, TradeEvent]], None]] = []

    def on_bar(self, callback: Callable[[BarEvent], None]) -> None:
        """Register a callback for BarEvents."""
        self._bar_callbacks.append(callback)

    def on_trade(self, callback: Callable[[TradeEvent], None]) -> None:
        """Register a callback for TradeEvents."""
        self._trade_callbacks.append(callback)

    def on_any(self, callback: Callable[[Union[BarEvent, TradeEvent]], None]) -> None:
        """Register a callback for all events."""
        self._any_callbacks.append(callback)

    def _dispatch(self, event: Union[BarEvent, TradeEvent]) -> None:
        """Dispatch an event to registered callbacks."""
        for cb in self._any_callbacks:
            cb(event)

        if isinstance(event, BarEvent):
            for cb in self._bar_callbacks:
                cb(event)
        elif isinstance(event, TradeEvent):
            for cb in self._trade_callbacks:
                cb(event)

    def run(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        mode: str = "auto",
        progress_interval: int = 100_000,
    ) -> ReplayStats:
        """
        Run multi-symbol interleaved replay.

        Events are dispatched in strict timestamp order across all symbols.
        This is the default mode for backtesting.

        Args:
            symbols: Symbols to replay. None = all.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            mode: Feed mode ("auto", "bars_1s", "trades", "bars_1m").
            progress_interval: Log progress every N events.

        Returns:
            ReplayStats with run summary.
        """
        resolved_mode = self.feed._resolve_mode(mode)
        logger.info(
            f"Starting replay: symbols={symbols}, {start} → {end}, "
            f"mode={resolved_mode}"
        )

        stats = ReplayStats(mode=resolved_mode)
        symbols_seen = set()
        wall_start = time.time()

        event_stream = self.feed.iter_events(
            symbols=symbols, start=start, end=end, mode=mode
        )

        for event in event_stream:
            self._dispatch(event)

            # Track stats
            stats.total_events += 1
            symbols_seen.add(event.symbol)

            if isinstance(event, BarEvent):
                stats.bar_events += 1
            elif isinstance(event, TradeEvent):
                stats.trade_events += 1

            if stats.start_ts is None:
                stats.start_ts = event.ts
            stats.end_ts = event.ts

            # Progress
            if stats.total_events % progress_interval == 0:
                elapsed = time.time() - wall_start
                rate = stats.total_events / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Progress: {stats.total_events:,} events, "
                    f"{len(symbols_seen)} symbols, "
                    f"{rate:,.0f} evt/s"
                )

        stats.symbols_seen = len(symbols_seen)
        stats.elapsed_seconds = round(time.time() - wall_start, 3)
        stats.events_per_second = round(
            stats.total_events / stats.elapsed_seconds
            if stats.elapsed_seconds > 0 else 0, 1
        )

        logger.info(stats.summary())
        return stats

    def run_single_symbol(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        mode: str = "auto",
    ) -> ReplayStats:
        """
        Fast single-symbol replay.

        Skips multi-symbol interleaving overhead.
        Use for speed benchmarking or isolated symbol analysis.

        Args:
            symbol: Single symbol to replay.
            start: Start date.
            end: End date.
            mode: Feed mode.

        Returns:
            ReplayStats.
        """
        logger.info(f"Single-symbol fast replay: {symbol}")
        return self.run(
            symbols=[symbol],
            start=start,
            end=end,
            mode=mode,
            progress_interval=500_000,
        )

    def run_multi_symbol_interleaved(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        mode: str = "auto",
    ) -> ReplayStats:
        """
        Multi-symbol replay with strict timestamp interleaving.

        Buffers events from each symbol's stream and dispatches
        in globally sorted timestamp order using a heap.

        Use when precise cross-symbol timing matters.

        Args:
            symbols: List of symbols.
            start: Start date.
            end: End date.
            mode: Feed mode.

        Returns:
            ReplayStats.
        """
        resolved_mode = self.feed._resolve_mode(mode)
        logger.info(
            f"Interleaved replay: {len(symbols)} symbols, mode={resolved_mode}"
        )

        stats = ReplayStats(mode=resolved_mode)
        symbols_seen = set()
        wall_start = time.time()

        # Collect all events per symbol, then merge via heap
        # For streaming, we'd use per-symbol generators + heap merge
        # But for cached data, this approach is clean and deterministic

        all_events = []
        for sym in symbols:
            sym_events = list(self.feed.iter_events(
                symbols=[sym], start=start, end=end, mode=mode
            ))
            for evt in sym_events:
                # Heap key: (timestamp, symbol) for deterministic ordering
                heapq.heappush(all_events, (evt.ts, evt.symbol, evt))

        while all_events:
            ts, sym, event = heapq.heappop(all_events)
            self._dispatch(event)

            stats.total_events += 1
            symbols_seen.add(event.symbol)

            if isinstance(event, BarEvent):
                stats.bar_events += 1
            elif isinstance(event, TradeEvent):
                stats.trade_events += 1

            if stats.start_ts is None:
                stats.start_ts = ts
            stats.end_ts = ts

        stats.symbols_seen = len(symbols_seen)
        stats.elapsed_seconds = round(time.time() - wall_start, 3)
        stats.events_per_second = round(
            stats.total_events / stats.elapsed_seconds
            if stats.elapsed_seconds > 0 else 0, 1
        )

        logger.info(stats.summary())
        return stats


class TradeCollector:
    """
    Simple callback that collects trade results.
    Plug this into a strategy to accumulate trade dicts
    compatible with the metrics engine.
    """

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []

    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        shares: int,
        direction: str,
        entry_ts: int,
        exit_ts: int,
        regime: str = "mixed",
        **extra,
    ) -> None:
        """Record a completed trade."""
        if direction == "long":
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        from datetime import datetime, timezone
        date = datetime.fromtimestamp(entry_ts / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")

        trade = {
            "symbol": symbol,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "shares": shares,
            "direction": direction,
            "pnl": round(pnl, 4),
            "date": date,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "regime": regime,
            **extra,
        }
        self.trades.append(trade)

    def reset(self) -> None:
        self.trades.clear()
