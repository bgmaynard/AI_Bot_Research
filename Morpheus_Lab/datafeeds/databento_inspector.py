"""
Morpheus Lab — Databento Cache Inspector
===========================================
Scans a Databento cache directory to detect:
  - Dataset + schema types present (trades/quotes/bars)
  - Timestamp resolution
  - Symbol coverage + date range
  - Estimated event volume

Writes machine-readable summary to reports/dataset_profile.json

Usage:
    python -m Morpheus_Lab.cli inspect-databento --cache Z:\\AI_BOT_DATA\\databento_cache
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import databento as db

logger = logging.getLogger(__name__)

# Databento fixed-point scale
FIXED_PRICE_SCALE = db.FIXED_PRICE_SCALE  # 1e9


@dataclass
class FileProfile:
    """Profile of a single .dbn.zst file."""
    path: str
    dataset: str
    schema: str
    size_bytes: int
    start: Optional[str] = None
    end: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    record_count: int = 0
    ts_resolution: str = "nanosecond"


@dataclass
class DatasetProfile:
    """Complete profile of a Databento cache."""
    cache_path: str
    scan_timestamp: str
    total_files: int
    total_size_bytes: int
    total_size_mb: float
    datasets: List[str]
    schemas_available: Dict[str, int]  # schema -> file count
    schema_priority: str               # best available for backtesting
    symbol_coverage: Dict[str, int]    # symbol -> event estimate
    date_range: Dict[str, str]         # earliest/latest
    estimated_total_events: int
    ts_resolution: str
    files: List[Dict[str, Any]]
    recommended_mode: str              # "bars_1s", "trades", "bars_1m"
    recommendation_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Schema preference order for backtesting
SCHEMA_PRIORITY = [
    "ohlcv-1s",   # Best: 1-second bars — fast + sufficient
    "trades",      # Good: tick-level — most granular but slower
    "ohlcv-1m",   # Acceptable: 1-minute bars
    "mbp-1",      # Usable: L1 quotes
    "tbbo",       # Usable: top BBO
    "ohlcv-1h",   # Coarse
    "ohlcv-1d",   # Very coarse
]

SCHEMA_TO_MODE = {
    "ohlcv-1s": "bars_1s",
    "trades": "trades",
    "ohlcv-1m": "bars_1m",
    "mbp-1": "quotes",
    "tbbo": "quotes",
    "ohlcv-1h": "bars_1h",
    "ohlcv-1d": "bars_1d",
}


def find_dbn_files(cache_path: str) -> List[Path]:
    """
    Recursively find all .dbn.zst files in a cache directory.
    Also finds .dbn files (uncompressed).
    """
    root = Path(cache_path)
    if not root.exists():
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

    files = []
    for ext in ["*.dbn.zst", "*.dbn"]:
        files.extend(root.rglob(ext))

    logger.info(f"Found {len(files)} DBN files in {cache_path}")
    return sorted(files)


def profile_file(filepath: Path, quick: bool = True) -> Optional[FileProfile]:
    """
    Profile a single DBN file.

    Args:
        filepath: Path to .dbn.zst file.
        quick: If True, only read metadata (fast). If False, scan records.
    """
    try:
        store = db.DBNStore.from_file(str(filepath))
        meta = store.metadata

        profile = FileProfile(
            path=str(filepath),
            dataset=meta.dataset if hasattr(meta, 'dataset') else str(getattr(meta, 'dataset', 'unknown')),
            schema=str(store.schema) if store.schema else "unknown",
            size_bytes=filepath.stat().st_size,
        )

        # Date range from metadata
        if store.start is not None:
            profile.start = str(store.start)
        if store.end is not None:
            profile.end = str(store.end)

        # Symbols from metadata
        try:
            syms = store.symbols
            if syms:
                profile.symbols = list(syms)
        except Exception:
            pass

        # Record count estimate
        if not quick:
            try:
                df = store.to_df()
                profile.record_count = len(df)
                if len(df) > 0 and 'symbol' in df.columns:
                    profile.symbols = list(df['symbol'].unique())
            except Exception as e:
                logger.debug(f"Could not read records from {filepath}: {e}")
        else:
            # Rough estimate based on file size and schema
            profile.record_count = _estimate_record_count(
                profile.size_bytes, profile.schema
            )

        return profile

    except Exception as e:
        logger.warning(f"Could not profile {filepath}: {e}")
        return None


def _estimate_record_count(size_bytes: int, schema: str) -> int:
    """
    Rough record count estimate based on file size.
    Compressed .dbn.zst typically has ~5-10x compression ratio.
    Record sizes vary by schema.
    """
    # Approximate uncompressed record sizes (bytes)
    RECORD_SIZES = {
        "trades": 64,
        "ohlcv-1s": 56,
        "ohlcv-1m": 56,
        "ohlcv-1h": 56,
        "ohlcv-1d": 56,
        "mbp-1": 72,
        "mbp-10": 360,
        "tbbo": 72,
        "mbo": 80,
    }

    compression_ratio = 7  # typical for market data
    record_size = RECORD_SIZES.get(schema, 64)
    uncompressed = size_bytes * compression_ratio

    return max(1, int(uncompressed / record_size))


def inspect_cache(
    cache_path: str,
    quick: bool = True,
    report_path: Optional[str] = None,
) -> DatasetProfile:
    """
    Full inspection of a Databento cache directory.

    Args:
        cache_path: Root path of the Databento cache.
        quick: If True, use metadata only (fast). If False, scan all records.
        report_path: Path to save JSON report. Defaults to reports/dataset_profile.json

    Returns:
        DatasetProfile with complete cache analysis.
    """
    start_time = time.time()
    logger.info(f"Inspecting Databento cache: {cache_path}")

    dbn_files = find_dbn_files(cache_path)
    if not dbn_files:
        raise FileNotFoundError(f"No .dbn.zst or .dbn files found in {cache_path}")

    # Profile each file
    profiles: List[FileProfile] = []
    for i, f in enumerate(dbn_files):
        logger.info(f"Profiling [{i+1}/{len(dbn_files)}]: {f.name}")
        profile = profile_file(f, quick=quick)
        if profile:
            profiles.append(profile)

    # Aggregate
    datasets: Set[str] = set()
    schemas: Dict[str, int] = defaultdict(int)
    symbols: Dict[str, int] = defaultdict(int)
    all_starts = []
    all_ends = []
    total_size = 0
    total_events = 0

    for p in profiles:
        datasets.add(p.dataset)
        schemas[p.schema] += 1
        total_size += p.size_bytes
        total_events += p.record_count

        if p.start:
            all_starts.append(p.start)
        if p.end:
            all_ends.append(p.end)

        for sym in p.symbols:
            symbols[sym] += p.record_count // max(len(p.symbols), 1)

    # Determine best available schema
    best_schema = "unknown"
    recommended_mode = "trades"
    recommendation_reason = "No recognized schemas found"

    for schema in SCHEMA_PRIORITY:
        if schema in schemas:
            best_schema = schema
            recommended_mode = SCHEMA_TO_MODE.get(schema, "trades")
            recommendation_reason = (
                f"'{schema}' is available ({schemas[schema]} files). "
                f"This is the highest-priority schema for backtesting."
            )
            break

    # Date range
    date_range = {}
    if all_starts:
        date_range["earliest"] = min(all_starts)
    if all_ends:
        date_range["latest"] = max(all_ends)

    elapsed = time.time() - start_time

    result = DatasetProfile(
        cache_path=cache_path,
        scan_timestamp=datetime.now(timezone.utc).isoformat(),
        total_files=len(profiles),
        total_size_bytes=total_size,
        total_size_mb=round(total_size / (1024 * 1024), 2),
        datasets=sorted(datasets),
        schemas_available=dict(schemas),
        schema_priority=best_schema,
        symbol_coverage=dict(sorted(symbols.items(), key=lambda x: -x[1])[:50]),
        date_range=date_range,
        estimated_total_events=total_events,
        ts_resolution="nanosecond",
        files=[asdict(p) for p in profiles],
        recommended_mode=recommended_mode,
        recommendation_reason=recommendation_reason,
    )

    logger.info(
        f"Inspection complete in {elapsed:.1f}s: "
        f"{len(profiles)} files, {len(datasets)} datasets, "
        f"{len(schemas)} schemas, ~{total_events:,} events"
    )

    # Save report
    if report_path is None:
        report_path = os.path.join("reports", "dataset_profile.json")

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    logger.info(f"Report saved: {report_path}")

    return result


def print_inspection_report(profile: DatasetProfile) -> None:
    """Pretty-print inspection results to console."""
    print()
    print("=" * 64)
    print("  DATABENTO CACHE INSPECTION REPORT")
    print("=" * 64)
    print(f"  Cache path:    {profile.cache_path}")
    print(f"  Scan time:     {profile.scan_timestamp}")
    print(f"  Total files:   {profile.total_files}")
    print(f"  Total size:    {profile.total_size_mb:.1f} MB")
    print(f"  Est. events:   {profile.estimated_total_events:,}")
    print(f"  Resolution:    {profile.ts_resolution}")
    print()

    print("  Datasets:")
    for ds in profile.datasets:
        print(f"    • {ds}")
    print()

    print("  Schemas Available:")
    for schema, count in sorted(profile.schemas_available.items()):
        marker = " ★" if schema == profile.schema_priority else ""
        print(f"    {schema}: {count} files{marker}")
    print()

    if profile.date_range:
        print(f"  Date Range:")
        print(f"    Earliest: {profile.date_range.get('earliest', 'unknown')}")
        print(f"    Latest:   {profile.date_range.get('latest', 'unknown')}")
        print()

    print(f"  Symbol Coverage ({len(profile.symbol_coverage)} symbols):")
    for sym, count in list(profile.symbol_coverage.items())[:15]:
        print(f"    {sym}: ~{count:,} events")
    if len(profile.symbol_coverage) > 15:
        print(f"    ... and {len(profile.symbol_coverage) - 15} more")
    print()

    print("  " + "─" * 60)
    print(f"  ★ RECOMMENDED MODE: {profile.recommended_mode}")
    print(f"    {profile.recommendation_reason}")
    print("  " + "─" * 60)
    print()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python databento_inspector.py <cache_path> [--deep]")
        sys.exit(1)

    cache = sys.argv[1]
    deep = "--deep" in sys.argv

    profile = inspect_cache(cache, quick=not deep)
    print_inspection_report(profile)
