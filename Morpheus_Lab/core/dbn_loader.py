"""
Morpheus Lab — Databento Trade Loader
========================================
Streams TradeEvent objects from compressed .dbn.zst files.

Design:
  - Discovers DBN files for symbol + date range
  - Streams events via store.replay() — never loads full file
  - Yields TradeEvent objects
  - Filters by nanosecond timestamp range
  - No pandas. No DataFrames. Pure streaming iterator.

Performance:
  - Sequential streaming through sorted trade records
  - Minimal object allocation (frozen dataclass)
  - Symbol resolved from filename (zero overhead)
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import databento as db

from core.event_types import TradeEvent

logger = logging.getLogger(__name__)

FIXED_PRICE_SCALE = 1_000_000_000  # db.FIXED_PRICE_SCALE


def _parse_symbol_from_filename(filename: str) -> str:
    """
    Extract symbol from DBN filename.

    Handles patterns:
      CISS_20260130T0800_20260130T2100.dbn.zst
      BBAI_2026-02-06T120000_2026-02-06T143000.dbn.zst
      NVDA_20260209T1200_20260209T2000.dbn.zst

    Returns symbol string (e.g., "CISS", "BBAI", "NVDA").
    """
    # Strip extensions
    name = filename.replace(".dbn.zst", "").replace(".dbn", "")
    # Symbol is everything before first underscore followed by a digit
    match = re.match(r"^([A-Za-z]+)_", name)
    if match:
        return match.group(1).upper()
    return name.upper()


def _parse_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract start date from filename as YYYY-MM-DD.

    Handles:
      CISS_20260130T0800... → 2026-01-30
      BBAI_2026-02-06T1200... → 2026-02-06
    """
    name = filename.replace(".dbn.zst", "").replace(".dbn", "")

    # Pattern 1: SYMBOL_YYYYMMDDTHHMMSS
    m = re.search(r"_(\d{4})(\d{2})(\d{2})T", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Pattern 2: SYMBOL_YYYY-MM-DDTHHMMSS
    m = re.search(r"_(\d{4}-\d{2}-\d{2})T", name)
    if m:
        return m.group(1)

    return None


def _date_to_ns(date_str: str, end_of_day: bool = False) -> int:
    """
    Convert YYYY-MM-DD to nanosecond timestamp (UTC).
    If end_of_day, returns 23:59:59.999999999 of that day.
    """
    from datetime import datetime, timezone

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return int(dt.timestamp() * 1_000_000_000)


class DatabentoTradeLoader:
    """
    Streaming trade loader over local Databento .dbn.zst cache.

    All iteration is streaming — no full file loads, no pandas.
    """

    def __init__(self, cache_path: str):
        """
        Args:
            cache_path: Root directory containing .dbn.zst files.
        """
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

        # Index: symbol → list of (filepath, file_date)
        self._index: Dict[str, List[tuple]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Scan cache directory and index files by symbol."""
        count = 0
        for f in self.cache_path.rglob("*.dbn.zst"):
            sym = _parse_symbol_from_filename(f.name)
            file_date = _parse_date_from_filename(f.name)
            if sym not in self._index:
                self._index[sym] = []
            self._index[sym].append((f, file_date))
            count += 1

        # Sort each symbol's files by date for deterministic ordering
        for sym in self._index:
            self._index[sym].sort(key=lambda x: x[1] or "")

        logger.info(
            f"Index built: {count} files, {len(self._index)} symbols"
        )

    @property
    def symbols(self) -> List[str]:
        """All available symbols in cache."""
        return sorted(self._index.keys())

    def files_for_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Path]:
        """
        Get matching files for a symbol within a date range.

        Args:
            symbol: Ticker symbol (case-insensitive).
            start_date: "YYYY-MM-DD" inclusive. None = no lower bound.
            end_date: "YYYY-MM-DD" inclusive. None = no upper bound.

        Returns:
            List of file paths, sorted by date.
        """
        sym = symbol.upper()
        if sym not in self._index:
            return []

        files = []
        for filepath, file_date in self._index[sym]:
            if file_date:
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
            files.append(filepath)

        return files

    def iter_symbol(
        self,
        symbol: str,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Iterator[TradeEvent]:
        """
        Stream TradeEvents for a single symbol.

        Events are yielded in chronological order.
        Files are pre-filtered by date, then records filtered by nanosecond timestamp.

        Args:
            symbol: Ticker symbol.
            start_ts: Start nanosecond timestamp (inclusive).
            end_ts: End nanosecond timestamp (inclusive).
            start_date: Start date for file filtering "YYYY-MM-DD".
            end_date: End date for file filtering "YYYY-MM-DD".

        Yields:
            TradeEvent objects in chronological order.
        """
        sym = symbol.upper()
        files = self.files_for_symbol(sym, start_date, end_date)

        if not files:
            logger.debug(f"No files found for {sym}")
            return

        for filepath in files:
            yield from self._stream_file(filepath, sym, start_ts, end_ts)

    def _stream_file(
        self,
        filepath: Path,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> Iterator[TradeEvent]:
        """
        Stream TradeEvents from a single .dbn.zst file.

        Uses store.to_ndarray(count=CHUNK_SIZE) for chunked streaming.
        No pandas. No full memory load.
        """
        CHUNK_SIZE = 50_000  # records per chunk

        try:
            store = db.DBNStore.from_file(str(filepath))
        except Exception as e:
            logger.error(f"Failed to open {filepath}: {e}")
            return

        try:
            chunks = store.to_ndarray(count=CHUNK_SIZE)
        except Exception as e:
            logger.error(f"Failed to read ndarray from {filepath}: {e}")
            return

        for chunk in chunks:
            # Structured numpy array with fields: ts_event, price, size, etc.
            ts_col = chunk["ts_event"]
            price_col = chunk["price"]
            size_col = chunk["size"]

            for i in range(len(chunk)):
                ts = int(ts_col[i])

                if ts < start_ts:
                    continue
                if ts > end_ts:
                    return  # Records are chronological — safe to stop

                price_raw = int(price_col[i])
                if price_raw == db.UNDEF_PRICE:
                    continue

                price = price_raw / FIXED_PRICE_SCALE
                if price <= 0:
                    continue

                yield TradeEvent(
                    ts=ts,
                    symbol=symbol,
                    price=price,
                    size=int(size_col[i]),
                )

    def iter_all_files(
        self,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
    ) -> Iterator[TradeEvent]:
        """
        Stream all trades from all files (for full cache replay).
        Files are processed in alphabetical order per symbol.
        """
        for sym in sorted(self._index.keys()):
            yield from self.iter_symbol(sym, start_ts, end_ts)
