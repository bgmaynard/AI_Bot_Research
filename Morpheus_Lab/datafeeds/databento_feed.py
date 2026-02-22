"""
Morpheus Lab — Databento Feed
================================
Provides unified event iteration over Databento cache files.

Auto-detects best available timeframe:
  1. bars_1s  (ohlcv-1s) if present
  2. trades + bar_aggregator(1s) if trades present
  3. bars_1m  (ohlcv-1m) fallback

All output is unified TradeEvent / BarEvent objects.
Supports chunked streaming iteration to avoid memory blowups.

Usage:
    feed = DatabentoFeed(cache_path="Z:\\AI_BOT_DATA\\databento_cache")
    for event in feed.iter_events(symbols=["AAPL"], start="2026-01-01", end="2026-01-31"):
        print(event)
"""

import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Union

import databento as db

from core.events import BarEvent, QuoteEvent, TradeEvent

logger = logging.getLogger(__name__)

FIXED_PRICE_SCALE = db.FIXED_PRICE_SCALE  # 1e9


def _to_price(fixed: int) -> float:
    """Convert Databento fixed-point price to float."""
    if fixed == db.UNDEF_PRICE:
        return 0.0
    return fixed / FIXED_PRICE_SCALE


def _ts_to_date(ts_ns: int) -> str:
    """Convert nanosecond timestamp to YYYY-MM-DD."""
    return datetime.utcfromtimestamp(ts_ns / 1e9).strftime("%Y-%m-%d")


class BarAggregator:
    """
    Aggregates TradeEvents into 1-second BarEvents.
    Used when only raw trades are available.
    """

    def __init__(self):
        self._buckets: Dict[str, Dict[int, dict]] = defaultdict(dict)

    def _bucket_key(self, ts_ns: int) -> int:
        """Floor timestamp to 1-second boundary (nanoseconds)."""
        return (ts_ns // 1_000_000_000) * 1_000_000_000

    def add_trade(self, trade: TradeEvent) -> Optional[BarEvent]:
        """
        Add a trade. Returns a completed BarEvent if the bucket
        has moved past this second (previous second is complete).
        """
        bucket_ts = self._bucket_key(trade.ts)
        sym = trade.symbol
        completed = None

        if sym in self._buckets and len(self._buckets[sym]) > 0:
            # Check if we've moved to a new second
            existing_keys = list(self._buckets[sym].keys())
            for k in existing_keys:
                if k < bucket_ts:
                    # Previous bucket is complete
                    completed = self._finalize_bucket(sym, k)
                    break

        # Add to current bucket
        if bucket_ts not in self._buckets[sym]:
            self._buckets[sym][bucket_ts] = {
                "open": trade.price,
                "high": trade.price,
                "low": trade.price,
                "close": trade.price,
                "volume": trade.size,
            }
        else:
            b = self._buckets[sym][bucket_ts]
            b["high"] = max(b["high"], trade.price)
            b["low"] = min(b["low"], trade.price)
            b["close"] = trade.price
            b["volume"] += trade.size

        return completed

    def _finalize_bucket(self, symbol: str, bucket_ts: int) -> BarEvent:
        """Convert a completed bucket into a BarEvent and remove it."""
        b = self._buckets[symbol].pop(bucket_ts)
        return BarEvent(
            ts=bucket_ts,
            open=b["open"],
            high=b["high"],
            low=b["low"],
            close=b["close"],
            volume=b["volume"],
            symbol=symbol,
            timeframe="1s",
        )

    def flush_all(self) -> List[BarEvent]:
        """Flush all remaining buckets (call at end of stream)."""
        bars = []
        for sym in list(self._buckets.keys()):
            for ts in sorted(self._buckets[sym].keys()):
                bars.append(self._finalize_bucket(sym, ts))
        return bars


class DatabentoFeed:
    """
    Unified feed over Databento cache with auto-detection.

    Modes:
        "auto"    — choose best available (1s bars > trades > 1m bars)
        "bars_1s" — force ohlcv-1s schema
        "trades"  — force trades schema (raw ticks)
        "bars_1m" — force ohlcv-1m schema
    """

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

        self._files_by_schema: Dict[str, List[Path]] = defaultdict(list)
        self._scan_cache()

    def _scan_cache(self) -> None:
        """Scan cache and index files by schema."""
        for f in self.cache_path.rglob("*.dbn.zst"):
            try:
                store = db.DBNStore.from_file(str(f))
                schema = str(store.schema)
                self._files_by_schema[schema].append(f)
            except Exception as e:
                logger.debug(f"Skipping {f}: {e}")

        for f in self.cache_path.rglob("*.dbn"):
            if not str(f).endswith(".dbn.zst"):
                try:
                    store = db.DBNStore.from_file(str(f))
                    schema = str(store.schema)
                    self._files_by_schema[schema].append(f)
                except Exception as e:
                    logger.debug(f"Skipping {f}: {e}")

        total = sum(len(v) for v in self._files_by_schema.values())
        logger.info(
            f"Cache scan complete: {total} files across "
            f"{list(self._files_by_schema.keys())} schemas"
        )

    @property
    def available_schemas(self) -> List[str]:
        return list(self._files_by_schema.keys())

    def detect_mode(self) -> str:
        """
        Auto-detect best mode based on available schemas.
        Priority: bars_1s > trades+aggregator > bars_1m
        """
        if "ohlcv-1s" in self._files_by_schema:
            logger.info("Auto mode → bars_1s (ohlcv-1s available)")
            return "bars_1s"
        elif "trades" in self._files_by_schema:
            logger.info("Auto mode → trades (will aggregate to 1s bars)")
            return "trades"
        elif "ohlcv-1m" in self._files_by_schema:
            logger.info("Auto mode → bars_1m (ohlcv-1m available)")
            return "bars_1m"
        else:
            available = list(self._files_by_schema.keys())
            raise ValueError(
                f"No usable schema found. Available: {available}. "
                f"Need one of: ohlcv-1s, trades, ohlcv-1m"
            )

    def _resolve_mode(self, mode: str) -> str:
        """Resolve 'auto' to a concrete mode."""
        if mode == "auto":
            return self.detect_mode()
        return mode

    def _schema_for_mode(self, mode: str) -> str:
        """Map mode to Databento schema string."""
        return {
            "bars_1s": "ohlcv-1s",
            "trades": "trades",
            "bars_1m": "ohlcv-1m",
        }.get(mode, mode)

    def _filter_files(
        self,
        schema: str,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Path]:
        """
        Filter cache files by schema, symbol, and date range.
        """
        files = self._files_by_schema.get(schema, [])
        if not files:
            return []

        filtered = []
        for f in files:
            try:
                store = db.DBNStore.from_file(str(f))

                # Date filter
                if start and store.end is not None:
                    file_end = str(store.end)[:10]
                    if file_end < start:
                        continue
                if end and store.start is not None:
                    file_start = str(store.start)[:10]
                    if file_start > end:
                        continue

                # Symbol filter (check if any requested symbols are in the file)
                if symbols:
                    try:
                        file_syms = set(store.symbols) if store.symbols else set()
                        if file_syms and not file_syms.intersection(set(symbols)):
                            continue
                    except Exception:
                        pass  # If we can't check symbols, include the file

                filtered.append(f)

            except Exception as e:
                logger.debug(f"Filter skip {f}: {e}")

        logger.info(f"Filtered to {len(filtered)}/{len(files)} files for {schema}")
        return filtered

    def iter_events(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        mode: str = "auto",
    ) -> Generator[Union[TradeEvent, BarEvent], None, None]:
        """
        Iterate over events from the cache.

        Args:
            symbols: Filter to these symbols. None = all.
            start: Start date "YYYY-MM-DD" (inclusive).
            end: End date "YYYY-MM-DD" (inclusive).
            mode: "auto", "bars_1s", "trades", "bars_1m"

        Yields:
            TradeEvent or BarEvent objects, ordered by timestamp.
        """
        resolved_mode = self._resolve_mode(mode)
        schema = self._schema_for_mode(resolved_mode)
        symbol_set = set(symbols) if symbols else None

        logger.info(
            f"Iterating: mode={resolved_mode}, schema={schema}, "
            f"symbols={symbols}, {start} → {end}"
        )

        files = self._filter_files(schema, symbols, start, end)
        if not files:
            logger.warning(f"No files found for schema={schema}")
            return

        # Sort files by start date for chronological order
        files = sorted(files, key=lambda f: str(f))

        if resolved_mode == "trades":
            # Trades mode: yield raw trades + optional bar aggregation
            yield from self._iter_trades(files, symbol_set, start, end)
        elif resolved_mode in ("bars_1s", "bars_1m"):
            yield from self._iter_bars(files, symbol_set, start, end, resolved_mode)
        else:
            raise ValueError(f"Unknown mode: {resolved_mode}")

    def iter_bars_from_trades(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Generator[BarEvent, None, None]:
        """
        Iterate over 1-second bars aggregated from raw trades.
        Use when ohlcv-1s is not available but trades are.
        """
        schema = "trades"
        symbol_set = set(symbols) if symbols else None
        files = self._filter_files(schema, symbols, start, end)

        if not files:
            logger.warning("No trade files found for bar aggregation")
            return

        aggregator = BarAggregator()

        for filepath in sorted(files):
            logger.info(f"Aggregating trades from: {filepath.name}")
            try:
                store = db.DBNStore.from_file(str(filepath))
                for record in store.replay():
                    if not isinstance(record, db.TradeMsg):
                        continue

                    # Symbol mapping
                    try:
                        sym = record.pretty_symbol if hasattr(record, 'pretty_symbol') else str(record.hd.instrument_id)
                    except Exception:
                        sym = str(record.hd.instrument_id)

                    if symbol_set and sym not in symbol_set:
                        continue

                    ts = record.hd.ts_event if hasattr(record.hd, 'ts_event') else record.ts_recv
                    price = _to_price(record.price)

                    if price <= 0:
                        continue

                    trade = TradeEvent(
                        ts=ts,
                        price=price,
                        size=record.size,
                        symbol=sym,
                    )

                    bar = aggregator.add_trade(trade)
                    if bar is not None:
                        yield bar

            except Exception as e:
                logger.error(f"Error aggregating {filepath}: {e}")

        # Flush remaining
        for bar in aggregator.flush_all():
            yield bar

    def _iter_trades(
        self,
        files: List[Path],
        symbol_set: Optional[Set[str]],
        start: Optional[str],
        end: Optional[str],
    ) -> Generator[TradeEvent, None, None]:
        """Stream TradeEvents from trade files."""
        for filepath in files:
            logger.info(f"Reading trades: {filepath.name}")
            try:
                store = db.DBNStore.from_file(str(filepath))

                # Use symbology mapping if available
                sym_map = {}
                try:
                    mappings = store.mappings
                    if mappings:
                        for m in mappings:
                            sym_map[m.instrument_id] = m.raw_symbol if hasattr(m, 'raw_symbol') else str(m.instrument_id)
                except Exception:
                    pass

                for record in store.replay():
                    if not isinstance(record, db.TradeMsg):
                        continue

                    # Resolve symbol
                    inst_id = record.hd.instrument_id
                    try:
                        sym = record.pretty_symbol if hasattr(record, 'pretty_symbol') else sym_map.get(inst_id, str(inst_id))
                    except Exception:
                        sym = sym_map.get(inst_id, str(inst_id))

                    if symbol_set and sym not in symbol_set:
                        continue

                    ts = record.hd.ts_event if hasattr(record.hd, 'ts_event') else record.ts_recv
                    price = _to_price(record.price)

                    if price <= 0:
                        continue

                    # Date filter
                    if start or end:
                        date_str = _ts_to_date(ts)
                        if start and date_str < start:
                            continue
                        if end and date_str > end:
                            continue

                    side = None
                    if hasattr(record, 'side'):
                        if record.side == db.Side.BID if hasattr(db, 'Side') else False:
                            side = "B"
                        elif record.side == db.Side.ASK if hasattr(db, 'Side') else False:
                            side = "A"

                    yield TradeEvent(
                        ts=ts,
                        price=price,
                        size=record.size,
                        symbol=sym,
                        side=side,
                    )

            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")

    def _iter_bars(
        self,
        files: List[Path],
        symbol_set: Optional[Set[str]],
        start: Optional[str],
        end: Optional[str],
        mode: str,
    ) -> Generator[BarEvent, None, None]:
        """Stream BarEvents from OHLCV files."""
        timeframe = "1s" if mode == "bars_1s" else "1m"

        for filepath in files:
            logger.info(f"Reading bars ({timeframe}): {filepath.name}")
            try:
                store = db.DBNStore.from_file(str(filepath))

                sym_map = {}
                try:
                    mappings = store.mappings
                    if mappings:
                        for m in mappings:
                            sym_map[m.instrument_id] = m.raw_symbol if hasattr(m, 'raw_symbol') else str(m.instrument_id)
                except Exception:
                    pass

                for record in store.replay():
                    if not isinstance(record, db.OHLCVMsg):
                        continue

                    inst_id = record.hd.instrument_id
                    try:
                        sym = record.pretty_symbol if hasattr(record, 'pretty_symbol') else sym_map.get(inst_id, str(inst_id))
                    except Exception:
                        sym = sym_map.get(inst_id, str(inst_id))

                    if symbol_set and sym not in symbol_set:
                        continue

                    ts = record.hd.ts_event if hasattr(record.hd, 'ts_event') else record.ts_recv

                    # Date filter
                    if start or end:
                        date_str = _ts_to_date(ts)
                        if start and date_str < start:
                            continue
                        if end and date_str > end:
                            continue

                    o = _to_price(record.open)
                    h = _to_price(record.high)
                    l = _to_price(record.low)
                    c = _to_price(record.close)
                    v = record.volume if hasattr(record, 'volume') else 0

                    if c <= 0:
                        continue

                    yield BarEvent(
                        ts=ts,
                        open=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=v,
                        symbol=sym,
                        timeframe=timeframe,
                    )

            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python databento_feed.py <cache_path> [--mode auto|trades|bars_1s|bars_1m]")
        sys.exit(1)

    feed = DatabentoFeed(sys.argv[1])
    print(f"Available schemas: {feed.available_schemas}")
    print(f"Auto mode: {feed.detect_mode()}")

    count = 0
    for event in feed.iter_events(mode="auto"):
        if count < 5:
            print(f"  {event}")
        count += 1
        if count >= 100:
            break

    print(f"... ({count} events sampled)")
