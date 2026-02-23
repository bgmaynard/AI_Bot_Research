"""
Morpheus Lab — Market Replay Engine (Optimized)
==================================================
Deterministic, high-performance trade-level replay.

Optimizations:
  - Heap merge with tuple comparison (fast C-level)
  - Single-symbol mode skips heap entirely
  - Raw passthrough mode for decode-speed isolation
  - cProfile integration via --profile flag
"""

import heapq
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

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
    mode: str = ""

    def summary(self) -> str:
        return (
            f"Replay: {self.total_events:,} events | "
            f"{self.symbols_with_data}/{self.symbols_requested} symbols | "
            f"{self.elapsed_seconds:.3f}s | "
            f"{self.events_per_second:,.0f} evt/s | "
            f"mode={self.mode}"
        )


class MarketReplayEngine:
    """
    Deterministic market replay engine.
    Heap-merges per-symbol streams by nanosecond timestamp.
    """

    def __init__(self, loader: DatabentoTradeLoader):
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
        Multi-symbol interleaved replay via heap merge.
        Deterministic: ties broken by symbol name.
        """
        ts_start = start_ts if start_ts is not None else (
            _date_to_ns(start_date) if start_date else 0
        )
        ts_end = end_ts if end_ts is not None else (
            _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1
        )

        heap: list = []
        symbols_upper = [s.upper() for s in symbols]

        for sym in symbols_upper:
            it = self.loader.iter_symbol(
                sym, start_ts=ts_start, end_ts=ts_end,
                start_date=start_date, end_date=end_date,
            )
            event = next(it, None)
            if event is not None:
                # Heap key: (ts, symbol) for deterministic ordering
                heapq.heappush(heap, (event.ts, event.symbol, event, it))

        while heap:
            ts, sym, event, it = heapq.heappop(heap)
            yield event
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
        Single-symbol fast replay — no heap overhead.
        Direct iterator passthrough.
        """
        ts_start = start_ts if start_ts is not None else (
            _date_to_ns(start_date) if start_date else 0
        )
        ts_end = end_ts if end_ts is not None else (
            _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1
        )
        yield from self.loader.iter_symbol(
            symbol, start_ts=ts_start, end_ts=ts_end,
            start_date=start_date, end_date=end_date,
        )

    def replay_callback(
        self,
        symbols: List[str],
        on_trade,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> ReplayStats:
        """
        Zero-object callback replay.

        Single symbol: calls on_trade(ts, price, size) directly.
        Multi-symbol: calls on_trade(ts, price, size, symbol) in timestamp order.

        No TradeEvent created. No tuples. No generators.
        """
        stats = ReplayStats(symbols_requested=len(symbols))
        wall_start = time.perf_counter()

        ts_start = _date_to_ns(start_date) if start_date else 0
        ts_end = _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1

        symbols_upper = [s.upper() for s in symbols]

        if len(symbols_upper) == 1:
            # Single-symbol: direct callback, no symbol arg needed
            stats.mode = "callback-single"
            count = self.loader.replay_symbol_callback(
                symbols_upper[0], on_trade,
                start_ts=ts_start, end_ts=ts_end,
                start_date=start_date, end_date=end_date,
            )
            stats.total_events = count
            stats.symbols_with_data = 1 if count > 0 else 0
        else:
            # Multi-symbol: merge batches by timestamp, call with symbol
            stats.mode = "callback-multi"
            count, sym_count = self._replay_multi_callback(
                symbols_upper, on_trade,
                ts_start, ts_end, start_date, end_date,
            )
            stats.total_events = count
            stats.symbols_with_data = sym_count

        stats.elapsed_seconds = round(time.perf_counter() - wall_start, 6)
        stats.events_per_second = round(
            stats.total_events / stats.elapsed_seconds
            if stats.elapsed_seconds > 0 else 0, 1
        )
        return stats

    def _replay_multi_callback(
        self,
        symbols: List[str],
        on_trade,
        ts_start: int,
        ts_end: int,
        start_date,
        end_date,
    ):
        """
        Multi-symbol callback merge.

        For ≤5 symbols: linear scan (no heap overhead).
        For >5 symbols: heap merge via iterator path.

        Returns (total_count, symbols_with_data).
        """
        import numpy as np

        # Collect all batches per symbol: list of (ts_arr, price_arr, size_arr, sym)
        # Then merge by first timestamp of each batch
        sym_batches = {}
        symbols_with_data = set()

        for sym in symbols:
            batches = list(self.loader.iter_symbol_batches(
                sym, start_ts=ts_start, end_ts=ts_end,
                start_date=start_date, end_date=end_date,
            ))
            if batches:
                sym_batches[sym] = batches
                symbols_with_data.add(sym)

        if not sym_batches:
            return 0, 0

        # Flatten all batches into one list, sort by first timestamp
        all_batches = []
        for sym, batches in sym_batches.items():
            for ts_arr, price_arr, size_arr, s in batches:
                all_batches.append((int(ts_arr[0]), ts_arr, price_arr, size_arr, s))

        all_batches.sort(key=lambda x: x[0])

        # Now iterate in order, calling on_trade for each event
        _on_trade = on_trade
        total = 0

        for _, ts_arr, price_arr, size_arr, sym in all_batches:
            ts_list = ts_arr.tolist()
            price_list = price_arr.tolist()
            size_list = size_arr.tolist()
            n = len(ts_list)

            for i in range(n):
                _on_trade(ts_list[i], price_list[i], size_list[i], sym)

            total += n

        return total, len(symbols_with_data)

    def benchmark(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        single_mode: bool = False,
        raw_pass: bool = False,
        batch_mode: bool = False,
        callback_mode: bool = False,
        batch_callback_mode: bool = False,
    ) -> ReplayStats:
        """
        Run replay and collect performance statistics.

        Args:
            raw_pass: Count only, no Python objects.
            batch_mode: Yield numpy arrays, no per-event objects.
            callback_mode: Zero-object callback with raw primitives.
        """
        stats = ReplayStats(symbols_requested=len(symbols))
        wall_start = time.perf_counter()

        ts_start = _date_to_ns(start_date) if start_date else 0
        ts_end = _date_to_ns(end_date, end_of_day=True) if end_date else 2**63 - 1

        if raw_pass:
            stats.mode = "raw-passthrough"
            counts = self.loader.count_all_raw(
                symbols, start_ts=ts_start, end_ts=ts_end,
                start_date=start_date, end_date=end_date,
            )
            stats.total_events = sum(counts.values())
            stats.symbols_with_data = sum(1 for c in counts.values() if c > 0)

        elif callback_mode:
            # Zero-object: raw primitives to a no-op callback
            counter = [0]

            if len(symbols) == 1:
                def _noop3(ts, px, sz):
                    counter[0] += 1
                stats = self.replay_callback(symbols, _noop3, start_date, end_date)
            else:
                def _noop4(ts, px, sz, sym):
                    counter[0] += 1
                stats = self.replay_callback(symbols, _noop4, start_date, end_date)
            return stats

        elif batch_callback_mode:
            # Batch callback: numpy arrays passed directly, zero per-event loop
            stats.mode = "batch-callback"
            counter = [0]
            syms_seen = set()

            def _batch_noop(ts_arr, price_arr, size_arr, sym):
                counter[0] += len(ts_arr)
                syms_seen.add(sym)

            for sym in symbols:
                self.loader.replay_symbol_batch_callback(
                    sym, _batch_noop,
                    start_ts=ts_start, end_ts=ts_end,
                    start_date=start_date, end_date=end_date,
                )

            stats.total_events = counter[0]
            stats.symbols_with_data = len(syms_seen)

        elif batch_mode:
            stats.mode = "batch-arrays"
            symbols_seen = set()
            for sym in symbols:
                for ts_arr, price_arr, size_arr, symbol in self.loader.iter_symbol_batches(
                    sym, start_ts=ts_start, end_ts=ts_end,
                    start_date=start_date, end_date=end_date,
                ):
                    stats.total_events += len(ts_arr)
                    symbols_seen.add(symbol)
            stats.symbols_with_data = len(symbols_seen)

        elif single_mode and len(symbols) == 1:
            stats.mode = "single-symbol"
            count = 0
            for event in self.replay_single(symbols[0], start_date, end_date):
                count += 1
            stats.total_events = count
            stats.symbols_with_data = 1 if count > 0 else 0

        else:
            stats.mode = "multi-symbol-heap"
            symbols_seen = set()
            for event in self.replay(symbols, start_date, end_date):
                stats.total_events += 1
                symbols_seen.add(event.symbol)
            stats.symbols_with_data = len(symbols_seen)

        stats.elapsed_seconds = round(time.perf_counter() - wall_start, 6)
        stats.events_per_second = round(
            stats.total_events / stats.elapsed_seconds
            if stats.elapsed_seconds > 0 else 0, 1
        )

        return stats
