"""
Smart Databento Downloader — Morpheus Trade-Aligned
=====================================================

Reads Morpheus trade_ledger.jsonl files to identify exactly which
symbol-date pairs need raw trade data from Databento XNAS.ITCH.

Only downloads data for symbols/dates where Morpheus had actual trades,
ensuring the replay engine has matching microstructure data.

Usage:
    python download_data.py                          # Download all
    python download_data.py --top 30                 # Top 30 symbols by trade count
    python download_data.py --symbols CISS ANL OCG   # Specific symbols
    python download_data.py --dry-run                # Show what would be downloaded
"""

import json
import databento as db
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse
import sys

# ============================================================
# CONFIGURATION
# ============================================================

# Where Morpheus trade reports live (via Z:\ mount from trading PC)
REPORTS_ROOT = Path(r"\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports")

# Where to store downloaded Databento trade data
OUTPUT_DIR = Path(r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades")

# Databento settings
DATASET = "XNAS.ITCH"
SCHEMA = "trades"

# Download window: full trading day (pre-market through close)
# UTC times: 4:00 AM ET = 09:00 UTC, 8:00 PM ET = 01:00 UTC next day
# Using 08:00-21:00 UTC to cover pre-market through after-hours
START_HOUR = "08:00"
END_HOUR = "21:00"


# ============================================================
# STEP 1: SCAN TRADE LEDGER FOR SYMBOL-DATE PAIRS
# ============================================================

def scan_trade_ledger(reports_root: Path, symbols_filter=None, top_n=None):
    """
    Read all trade_ledger.jsonl files and build a map of
    symbol -> set of dates where trades occurred.
    """
    sym_dates = defaultdict(set)
    sym_count = defaultdict(int)
    sym_pnl = defaultdict(float)

    ledger_files = sorted(reports_root.rglob("trade_ledger.jsonl"))

    if not ledger_files:
        print(f"[ERROR] No trade_ledger.jsonl files found in {reports_root}")
        sys.exit(1)

    for ledger_path in ledger_files:
        try:
            with open(ledger_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sym = rec.get("symbol", "")
                    entry_time = rec.get("entry_time", "")
                    if not sym or not entry_time:
                        continue

                    # Filter symbols if specified
                    if symbols_filter and sym not in symbols_filter:
                        continue

                    date_str = entry_time[:10]  # YYYY-MM-DD
                    sym_dates[sym].add(date_str)
                    sym_count[sym] += 1
                    sym_pnl[sym] += float(rec.get("pnl", 0))

        except Exception as e:
            print(f"  [WARN] Error reading {ledger_path}: {e}")

    # Sort by trade count
    ranked = sorted(sym_count.keys(), key=lambda s: -sym_count[s])

    # Apply top_n filter
    if top_n:
        ranked = ranked[:top_n]
        sym_dates = {s: sym_dates[s] for s in ranked}

    return sym_dates, sym_count, sym_pnl, ranked


# ============================================================
# STEP 2: BUILD DOWNLOAD MANIFEST
# ============================================================

def build_manifest(sym_dates, output_dir):
    """
    Build list of (symbol, date, filepath) to download,
    skipping files that already exist.
    """
    to_download = []
    already_exist = []

    for symbol in sorted(sym_dates.keys()):
        for date_str in sorted(sym_dates[symbol]):
            start = f"{date_str}T{START_HOUR}"
            end = f"{date_str}T{END_HOUR}"
            fname = f"{symbol}_{start.replace('-','').replace(':','')}_{end.replace('-','').replace(':','')}.dbn.zst"
            fpath = output_dir / fname

            if fpath.exists():
                already_exist.append((symbol, date_str, fpath))
            else:
                to_download.append((symbol, date_str, start, end, fpath))

    return to_download, already_exist


# ============================================================
# STEP 3: DOWNLOAD
# ============================================================

def download_all(to_download, dry_run=False):
    """Download all symbol-date pairs from Databento."""
    if not to_download:
        print("\n  Nothing to download — all files exist!")
        return

    if dry_run:
        print(f"\n  [DRY RUN] Would download {len(to_download)} files:")
        for symbol, date_str, start, end, fpath in to_download:
            print(f"    {symbol} {date_str} -> {fpath.name}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = db.Historical()

    success = 0
    failed = 0
    failed_list = []

    for i, (symbol, date_str, start, end, fpath) in enumerate(to_download):
        print(f"  [{i+1}/{len(to_download)}] {symbol} {date_str} ... ", end="", flush=True)
        try:
            data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=[symbol],
                schema=SCHEMA,
                start=start,
                end=end,
            )
            data.to_file(str(fpath))
            print(f"OK -> {fpath.name}")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            failed_list.append((symbol, date_str, str(e)))

    print(f"\n  Download complete: {success} succeeded, {failed} failed")

    if failed_list:
        print(f"\n  Failed downloads:")
        for sym, date, err in failed_list:
            print(f"    {sym} {date}: {err}")

    return success, failed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Databento Downloader")
    parser.add_argument("--reports", type=str, default=str(REPORTS_ROOT),
                        help="Path to Morpheus reports directory")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Path to save downloaded data")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Download only these symbols")
    parser.add_argument("--top", type=int, default=None,
                        help="Download only top N symbols by trade count")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")

    args = parser.parse_args()

    reports_root = Path(args.reports)
    output_dir = Path(args.output)

    print("=" * 80)
    print("SMART DATABENTO DOWNLOADER — Morpheus Trade-Aligned")
    print("=" * 80)

    # Step 1: Scan
    print(f"\n[1/3] Scanning trade ledger at {reports_root}...")
    sym_dates, sym_count, sym_pnl, ranked = scan_trade_ledger(
        reports_root,
        symbols_filter=args.symbols,
        top_n=args.top,
    )

    total_pairs = sum(len(d) for d in sym_dates.values())
    total_trades = sum(sym_count[s] for s in sym_dates.keys())

    print(f"  Symbols: {len(sym_dates)}")
    print(f"  Symbol-date pairs: {total_pairs}")
    print(f"  Total trades covered: {total_trades}")

    print(f"\n  {'SYMBOL':<10} {'TRADES':>6} {'DATES':>5} {'PNL':>10}")
    print(f"  {'-'*35}")
    for sym in ranked[:30]:
        print(f"  {sym:<10} {sym_count[sym]:>6} {len(sym_dates[sym]):>5} "
              f"${sym_pnl[sym]:>+9.2f}")

    if len(ranked) > 30:
        print(f"  ... and {len(ranked) - 30} more symbols")

    # Step 2: Build manifest
    print(f"\n[2/3] Building download manifest...")
    to_download, already_exist = build_manifest(sym_dates, output_dir)

    print(f"  Already cached: {len(already_exist)} files")
    print(f"  Need to download: {len(to_download)} files")

    # Step 3: Download
    print(f"\n[3/3] Downloading from Databento ({DATASET} / {SCHEMA})...")
    download_all(to_download, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"  Output: {output_dir}")
    print(f"  Total files: {len(already_exist) + len(to_download)}")
    print("=" * 80)
