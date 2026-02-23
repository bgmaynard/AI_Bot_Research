"""
Morpheus Lab — Databento Trade Loader (Optimized)
=====================================================
Streams trades from .dbn.zst using vectorized numpy filtering.

Optimizations:
  - Vectorized timestamp/price filtering (numpy boolean masks)
  - TradeEvent as namedtuple (minimal allocation)
  - Raw passthrough mode: count-only, zero Python object creation
  - Single-symbol mode: symbol string passed once, not per-event
  - Chunk size tuned for cache-line efficiency
  - No pandas. No DataFrames. Streaming only.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import databento as db

from core.event_types import TradeEvent

logger = logging.getLogger(__name__)

FIXED_PRICE_SCALE = 1_000_000_000
UNDEF_PRICE = db.UNDEF_PRICE
CHUNK_SIZE = 100_000  # larger chunks = fewer Python-level iterations


def _parse_symbol_from_filename(filename: str) -> str:
    """Extract symbol from DBN filename."""
    name = filename.replace(".dbn.zst", "").replace(".dbn", "")
    match = re.match(r"^([A-Za-z]+)_", name)
    return match.group(1).upper() if match else name.upper()


def _parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract start date from filename as YYYY-MM-DD."""
    name = filename.replace(".dbn.zst", "").replace(".dbn", "")
    m = re.search(r"_(\d{4})(\d{2})(\d{2})T", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"_(\d{4}-\d{2}-\d{2})T", name)
    return m.group(1) if m else None


def _date_to_ns(date_str: str, end_of_day: bool = False) -> int:
    """Convert YYYY-MM-DD to nanosecond timestamp (UTC)."""
    from datetime import datetime, timezone
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return int(dt.timestamp() * 1_000_000_000)


class DatabentoTradeLoader:
    """
    High-performance streaming trade loader over local Databento cache.
    """

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

        self._index: Dict[str, List[Tuple[Path, Optional[str]]]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Scan cache and index files by symbol."""
        count = 0
        for f in self.cache_path.rglob("*.dbn.zst"):
            sym = _parse_symbol_from_filename(f.name)
            file_date = _parse_date_from_filename(f.name)
            if sym not in self._index:
                self._index[sym] = []
            self._index[sym].append((f, file_date))
            count += 1

        for sym in self._index:
            self._index[sym].sort(key=lambda x: x[1] or "")

        logger.info(f"Index built: {count} files, {len(self._index)} symbols")

    @property
    def symbols(self) -> List[str]:
        return sorted(self._index.keys())

    def files_for_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Path]:
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

    # ──────────────────────────────────────────────────────────
    #  NORMAL MODE — yields TradeEvent namedtuples
    # ──────────────────────────────────────────────────────────

    def iter_symbol(
        self,
        symbol: str,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Iterator[TradeEvent]:
        """Stream TradeEvents for a single symbol (vectorized)."""
        sym = symbol.upper()
        files = self.files_for_symbol(sym, start_date, end_date)
        if not files:
            return
        for filepath in files:
            yield from self._stream_file_vectorized(filepath, sym, start_ts, end_ts)

    def iter_symbol_batches(
        self,
        symbol: str,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """
        Batch mode: yield (ts_array, price_array, size_array, symbol) tuples.

        No per-event Python objects — pure numpy arrays.
        ~5-10x faster than per-event mode.
        Use for strategies that can process arrays directly.
        """
        sym = symbol.upper()
        files = self.files_for_symbol(sym, start_date, end_date)
        if not files:
            return
        for filepath in files:
            yield from self._stream_file_batches(filepath, sym, start_ts, end_ts)

    def _stream_file_batches(
        self,
        filepath: Path,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """Yield filtered numpy array batches from a single file."""
        try:
            store = db.DBNStore.from_file(str(filepath))
            chunks = store.to_ndarray(count=CHUNK_SIZE)
        except Exception as e:
            logger.error(f"Failed to open {filepath}: {e}")
            return

        for chunk in chunks:
            ts_arr = chunk["ts_event"]
            price_arr = chunk["price"]
            size_arr = chunk["size"]

            mask = (ts_arr >= start_ts) & (ts_arr <= end_ts) & (price_arr != UNDEF_PRICE) & (price_arr > 0)

            if len(ts_arr) > 0 and ts_arr[-1] < start_ts:
                continue
            if len(ts_arr) > 0 and ts_arr[0] > end_ts:
                return

            ts_filtered = ts_arr[mask]
            if len(ts_filtered) == 0:
                continue

            price_filtered = price_arr[mask].astype(np.float64) / FIXED_PRICE_SCALE
            size_filtered = size_arr[mask]

            yield (ts_filtered, price_filtered, size_filtered, symbol)

    def _stream_file_vectorized(
        self,
        filepath: Path,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> Iterator[TradeEvent]:
        """
        Vectorized streaming: numpy boolean masks filter entire chunks,
        then iterate only surviving rows.
        """
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

        _TradeEvent = TradeEvent  # local ref avoids global lookup per event
        _sym = symbol             # single symbol — attach once

        for chunk in chunks:
            ts_arr = chunk["ts_event"]
            price_arr = chunk["price"]
            size_arr = chunk["size"]

            # Vectorized filtering — entire chunk at once
            mask = (ts_arr >= start_ts) & (ts_arr <= end_ts) & (price_arr != UNDEF_PRICE) & (price_arr > 0)

            # Check if we've passed end_ts entirely
            if len(ts_arr) > 0 and ts_arr[-1] < start_ts:
                continue
            if len(ts_arr) > 0 and ts_arr[0] > end_ts:
                return

            # Extract surviving rows
            ts_filtered = ts_arr[mask]
            price_filtered = price_arr[mask]
            size_filtered = size_arr[mask]

            n = len(ts_filtered)
            if n == 0:
                continue

            # tolist() bulk converts in C — faster than per-element numpy scalar access
            price_float = (price_filtered / FIXED_PRICE_SCALE).tolist()
            ts_list = ts_filtered.tolist()
            size_list = size_filtered.tolist()

            # Yield events from Python lists (fast indexing)
            for i in range(n):
                yield _TradeEvent(ts_list[i], _sym, price_float[i], size_list[i])

    # ──────────────────────────────────────────────────────────
    #  RAW PASSTHROUGH — count only, zero Python object creation
    # ──────────────────────────────────────────────────────────

    def count_symbol_raw(
        self,
        symbol: str,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        Raw passthrough: counts valid trades without creating Python objects.
        Isolates Databento decode speed from Python object allocation.
        """
        sym = symbol.upper()
        files = self.files_for_symbol(sym, start_date, end_date)
        if not files:
            return 0

        total = 0
        for filepath in files:
            total += self._count_file_raw(filepath, start_ts, end_ts)
        return total

    def _count_file_raw(self, filepath: Path, start_ts: int, end_ts: int) -> int:
        """Count valid trades in a file using pure numpy — no Python objects."""
        try:
            store = db.DBNStore.from_file(str(filepath))
        except Exception:
            return 0

        try:
            chunks = store.to_ndarray(count=CHUNK_SIZE)
        except Exception:
            return 0

        count = 0
        for chunk in chunks:
            ts_arr = chunk["ts_event"]
            price_arr = chunk["price"]

            mask = (ts_arr >= start_ts) & (ts_arr <= end_ts) & (price_arr != UNDEF_PRICE) & (price_arr > 0)
            count += int(np.count_nonzero(mask))

        return count

    def count_all_raw(
        self,
        symbols: List[str],
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, int]:
        """Raw passthrough count for multiple symbols."""
        results = {}
        for sym in symbols:
            results[sym] = self.count_symbol_raw(
                sym, start_ts, end_ts, start_date, end_date
            )
        return results

    def iter_all_files(
        self,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
    ) -> Iterator[TradeEvent]:
        """Stream all trades from all files."""
        for sym in sorted(self._index.keys()):
            yield from self.iter_symbol(sym, start_ts, end_ts)

    # ──────────────────────────────────────────────────────────
    #  CALLBACK MODE — zero object creation, raw primitives
    # ──────────────────────────────────────────────────────────

    def replay_symbol_callback(
        self,
        symbol: str,
        on_trade,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        Zero-object replay: calls on_trade(ts, price, size) with raw
        Python primitives. No tuples, no namedtuples, no containers.

        Returns event count.
        """
        sym = symbol.upper()
        files = self.files_for_symbol(sym, start_date, end_date)
        if not files:
            return 0

        total = 0
        for filepath in files:
            total += self._stream_file_callback(filepath, sym, on_trade, start_ts, end_ts)
        return total

    def _stream_file_callback(
        self,
        filepath: Path,
        symbol: str,
        on_trade,
        start_ts: int,
        end_ts: int,
    ) -> int:
        """
        Inner loop: vectorized filter then iterate Python lists
        calling on_trade(ts, price, size) with zero allocations.
        """
        try:
            store = db.DBNStore.from_file(str(filepath))
            chunks = store.to_ndarray(count=CHUNK_SIZE)
        except Exception:
            return 0

        _scale = FIXED_PRICE_SCALE
        _undef = UNDEF_PRICE
        _on_trade = on_trade  # local binding
        count = 0

        for chunk in chunks:
            ts_arr = chunk["ts_event"]
            price_arr = chunk["price"]
            size_arr = chunk["size"]

            # Early exit checks
            n = len(ts_arr)
            if n == 0:
                continue
            if ts_arr[-1] < start_ts:
                continue
            if ts_arr[0] > end_ts:
                break

            # Vectorized filter
            mask = (ts_arr >= start_ts) & (ts_arr <= end_ts) & (price_arr != _undef) & (price_arr > 0)

            ts_f = ts_arr[mask]
            nf = len(ts_f)
            if nf == 0:
                continue

            # tolist() is faster than per-element numpy indexing
            # (bulk C conversion vs numpy scalar wrapper per access)
            price_list = (price_arr[mask] / _scale).tolist()
            ts_list = ts_f.tolist()
            size_list = size_arr[mask].tolist()

            # Hot loop
            for i in range(nf):
                _on_trade(ts_list[i], price_list[i], size_list[i])

            count += nf

        return count

    def replay_symbol_batch_callback(
        self,
        symbol: str,
        on_batch,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        Batch callback: calls on_batch(ts_arr, price_arr, size_arr, symbol)
        with numpy arrays. No per-event Python loop. Approaches raw speed.

        Strategy must be able to process numpy arrays directly.
        This is the fastest possible Python path.
        """
        sym = symbol.upper()
        files = self.files_for_symbol(sym, start_date, end_date)
        if not files:
            return 0

        total = 0
        for filepath in files:
            total += self._stream_file_batch_callback(filepath, sym, on_batch, start_ts, end_ts)
        return total

    def _stream_file_batch_callback(
        self,
        filepath: Path,
        symbol: str,
        on_batch,
        start_ts: int,
        end_ts: int,
    ) -> int:
        """Batch callback: numpy arrays passed directly, zero per-event overhead."""
        try:
            store = db.DBNStore.from_file(str(filepath))
            chunks = store.to_ndarray(count=CHUNK_SIZE)
        except Exception:
            return 0

        _scale = FIXED_PRICE_SCALE
        _undef = UNDEF_PRICE
        _on_batch = on_batch
        _sym = symbol
        count = 0

        for chunk in chunks:
            ts_arr = chunk["ts_event"]
            price_arr = chunk["price"]
            size_arr = chunk["size"]

            n = len(ts_arr)
            if n == 0:
                continue
            if ts_arr[-1] < start_ts:
                continue
            if ts_arr[0] > end_ts:
                break

            mask = (ts_arr >= start_ts) & (ts_arr <= end_ts) & (price_arr != _undef) & (price_arr > 0)

            ts_f = ts_arr[mask]
            nf = len(ts_f)
            if nf == 0:
                continue

            price_f = price_arr[mask] / _scale
            size_f = size_arr[mask]

            _on_batch(ts_f, price_f, size_f, _sym)
            count += nf

        return count

    def replay_symbol_callback_with_sym(
        self,
        symbol: str,
        on_trade,
        start_ts: int = 0,
        end_ts: int = 2**63 - 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        Like replay_symbol_callback but calls on_trade(ts, price, size, symbol).
        Fourth arg is the symbol string (same object ref, not recreated).
        """
        sym = symbol.upper()
        files = self.files_for_symbol(sym, start_date, end_date)
        if not files:
            return 0

        total = 0
        for filepath in files:
            total += self._stream_file_callback_sym(filepath, sym, on_trade, start_ts, end_ts)
        return total

    def _stream_file_callback_sym(
        self,
        filepath: Path,
        symbol: str,
        on_trade,
        start_ts: int,
        end_ts: int,
    ) -> int:
        """Callback with symbol arg for multi-symbol replay."""
        try:
            store = db.DBNStore.from_file(str(filepath))
            chunks = store.to_ndarray(count=CHUNK_SIZE)
        except Exception:
            return 0

        _scale = FIXED_PRICE_SCALE
        _undef = UNDEF_PRICE
        _on_trade = on_trade
        _sym = symbol
        count = 0

        for chunk in chunks:
            ts_arr = chunk["ts_event"]
            price_arr = chunk["price"]
            size_arr = chunk["size"]

            n = len(ts_arr)
            if n == 0:
                continue
            if ts_arr[-1] < start_ts:
                continue
            if ts_arr[0] > end_ts:
                break

            mask = (ts_arr >= start_ts) & (ts_arr <= end_ts) & (price_arr != _undef) & (price_arr > 0)

            ts_f = ts_arr[mask]
            nf = len(ts_f)
            if nf == 0:
                continue

            price_f = (price_arr[mask] / _scale).tolist()
            ts_list = ts_f.tolist()
            size_list = size_arr[mask].tolist()

            for i in range(nf):
                _on_trade(ts_list[i], price_f[i], size_list[i], _sym)

            count += nf

        return count
