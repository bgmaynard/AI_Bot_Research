"""
Morpheus Lab — Market Replay Engine
======================================
Deterministic, high-performance trade-level replay.

Design:
  - Heap-merges multiple per-symbol iterators by timestamp
  - Fully streaming — no full memory load
  - Strategy-agnostic — yields TradeEvent, consumer decides what to do
  - Deterministic — same inputs always produce identical event order

Future compatibility:
  for event in replay_engine.replay(...):
      strategy.on_trade(event)
      slippage_model.apply(event)
      order_simulator.check(event)
"""

import heapq
import logging
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

from core.event_types import TradeEvent
from core.dbn_loader import DatabentoTradeLoader, _date_to_ns

logger = logging.getLogger(__name__)


@dataclass
class ReplayStats:
    """Statistics from a replay run."""
    total_events: int = 0
    symbols_requested: int = 0
    symbols_with_data: int = 0
    elapsed_seconds: float = 0.0
    events_per_second: float = 0.0

    def summary(self) -> str:
        return (
            f"Replay: {self.total_events:,} events | "
            f"{self.symbols_with_data}/{self.symbols_requested} symbols | "
            f"{self.elapsed_seconds:.3f}s | "
            f"{self.events_per_second:,.0f} evt/s"
        )


class MarketReplayEngine:
    """
    Deterministic market replay engine.

    Merges per-symbol trade streams into a single chronologically
    ordered event stream using a heap.

    All iteration is streaming. No full memory loads.
    """

    def __init__(self, loader: DatabentoTradeLoader):
        """
        Args:
            loader: Configured DatabentoTradeLoader instance.
        """
        self.loader = loader

    def replay(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Iterator[TradeEvent]:
        """
        Multi-symbol interleaved replay.

        Events are yielded in strict chronological order across all symbols.
        Ties broken by symbol name for determinism.

        Args:
            symbols: List of ticker symbols to replay.
            start_date: Start date "YYYY-MM-DD" (inclusive).
            end_date: End date "YYYY-MM-DD" (inclusive).
            start_ts: Start nanosecond timestamp (overrides start_date).
            end_ts: End nanosecond timestamp (overrides end_date).

        Yields:
            TradeEvent objects in chronological order.
        """
        # Resolve timestamps
        ts_start = start_ts if start_ts is not None else (
            _date_to_ns(start_date) if start_date else 0
        )
        ts_end = end_ts if end_ts is not None else (
            _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1
        )

        # Create per-symbol iterators
        # Heap entries: (timestamp, symbol_name, event, iterator)
        # symbol_name as tiebreaker for determinism
        heap = []
        symbols_upper = [s.upper() for s in symbols]

        for sym in symbols_upper:
            it = self.loader.iter_symbol(
                sym,
                start_ts=ts_start,
                end_ts=ts_end,
                start_date=start_date,
                end_date=end_date,
            )
            # Prime the iterator — get first event
            event = next(it, None)
            if event is not None:
                heapq.heappush(heap, (event.ts, event.symbol, event, it))

        # Merge via heap
        while heap:
            ts, sym, event, it = heapq.heappop(heap)
            yield event

            # Advance this symbol's iterator
            next_event = next(it, None)
            if next_event is not None:
                heapq.heappush(heap, (next_event.ts, next_event.symbol, next_event, it))

    def replay_single(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Iterator[TradeEvent]:
        """
        Single-symbol fast replay.

        Skips heap overhead — direct iterator passthrough.
        Use for speed benchmarking or isolated symbol analysis.

        Args:
            symbol: Single ticker symbol.
            start_date: Start date "YYYY-MM-DD".
            end_date: End date "YYYY-MM-DD".
            start_ts: Start nanosecond timestamp.
            end_ts: End nanosecond timestamp.

        Yields:
            TradeEvent objects in chronological order.
        """
        ts_start = start_ts if start_ts is not None else (
            _date_to_ns(start_date) if start_date else 0
        )
        ts_end = end_ts if end_ts is not None else (
            _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1
        )

        yield from self.loader.iter_symbol(
            symbol,
            start_ts=ts_start,
            end_ts=ts_end,
            start_date=start_date,
            end_date=end_date,
        )

    def benchmark(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        single_mode: bool = False,
    ) -> ReplayStats:
        """
        Run replay and collect performance statistics.

        Args:
            symbols: Symbols to replay.
            start_date: Start date.
            end_date: End date.
            single_mode: If True and single symbol, skip heap.

        Returns:
            ReplayStats with performance data.
        """
        stats = ReplayStats(symbols_requested=len(symbols))
        symbols_seen = set()
        wall_start = time.perf_counter()

        if single_mode and len(symbols) == 1:
            iterator = self.replay_single(
                symbols[0],
                start_date=start_date,
                end_date=end_date,
            )
        else:
            iterator = self.replay(
                symbols,
                start_date=start_date,
                end_date=end_date,
            )

        for event in iterator:
            stats.total_events += 1
            symbols_seen.add(event.symbol)

        stats.symbols_with_data = len(symbols_seen)
        stats.elapsed_seconds = round(time.perf_counter() - wall_start, 6)
        stats.events_per_second = round(
            stats.total_events / stats.elapsed_seconds
            if stats.elapsed_seconds > 0 else 0, 1
        )

        return stats
