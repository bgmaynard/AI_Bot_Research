#!/usr/bin/env python3
"""
Edge Preservation Study for Containment v2 (2026-03-03)
SuperBot Research Engine - READ ONLY, no production changes.

Takes all v2-pass signals (~74) and evaluates forward price action
using QUOTE_UPDATE events from the production events log.

Windows: 1m, 3m, 5m, 10m after signal timestamp
Metrics: MFE%, MAE%, close_return% at each window
Exit model: stop -1.0%, trail activate +0.8% trail 0.4%, exit at +5m close
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path

# === PATHS ===
SUPERBOT_ROOT = Path("C:/AI_Bot_Research/MORPHEUS_SUPERBOT")
REPLAY_JSON = SUPERBOT_ROOT / "engine" / "output" / "containment_v2_replay_2026-03-03.json"
EVENTS_LOG = Path("//bob1/c/morpheus/morpheus_ai/logs/events/events_2026-03-03.jsonl")
QUOTES_CACHE_DIR = SUPERBOT_ROOT / "engine" / "cache" / "quotes"
OUTPUT_JSON = SUPERBOT_ROOT / "engine" / "output" / "edge_preservation_v2_2026-03-03.json"
OUTPUT_MD = SUPERBOT_ROOT / "engine" / "output" / "edge_preservation_v2_2026-03-03.md"

# === EXIT MODEL PARAMETERS ===
STOP_PCT = -1.0       # hard stop at -1.0%
TRAIL_ACTIVATE = 0.8  # trailing stop activates at +0.8%
TRAIL_DISTANCE = 0.4  # trail distance 0.4%
EXIT_WINDOW_SEC = 300  # 5 minutes = 300 seconds

# === FORWARD WINDOWS (seconds) ===
WINDOWS = [60, 180, 300, 600]  # 1m, 3m, 5m, 10m
WINDOW_LABELS = {60: "1m", 180: "3m", 300: "5m", 600: "10m"}

# === DATE CONSTANT ===
STUDY_DATE = "2026-03-03"


def parse_timestamp(ts_str):
    """Parse timestamp from signal (HH:MM:SS format) to datetime."""
    # Signal timestamps are in UTC, format HH:MM:SS
    h, m, s = ts_str.split(":")
    return datetime(2026, 3, 3, int(h), int(m), int(s), tzinfo=timezone.utc)


def parse_event_timestamp(ts_str):
    """Parse ISO timestamp from events log."""
    # Format: 2026-03-03T10:26:53.435828+00:00
    # Handle various formats
    ts_str = ts_str.strip()
    if "+" in ts_str:
        # Has timezone offset
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            # Fallback: strip microseconds if too long
            base, tz = ts_str.rsplit("+", 1)
            if "." in base:
                base_dt, micro = base.rsplit(".", 1)
                micro = micro[:6]
                base = f"{base_dt}.{micro}"
            return datetime.fromisoformat(f"{base}+{tz}")
    elif ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    else:
        return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


def step1_extract_quotes(target_symbols):
    """Extract QUOTE_UPDATE events for target symbols from events log, cache per-symbol."""
    print(f"\n=== STEP 1: Extract QUOTE_UPDATE events for {len(target_symbols)} symbols ===")

    # Check if cache already exists
    cached_count = 0
    for sym in target_symbols:
        cache_file = QUOTES_CACHE_DIR / f"{sym}_quotes.json"
        if cache_file.exists():
            cached_count += 1

    if cached_count == len(target_symbols):
        print(f"  All {cached_count} symbol caches exist. Skipping extraction.")
        return True

    print(f"  Cached: {cached_count}/{len(target_symbols)}. Extracting from events log...")
    print(f"  Source: {EVENTS_LOG}")

    if not EVENTS_LOG.exists():
        print(f"  ERROR: Events log not found at {EVENTS_LOG}")
        return False

    # Read events log and extract QUOTE_UPDATE for target symbols
    symbol_quotes = defaultdict(list)
    line_count = 0
    quote_count = 0
    error_count = 0

    with open(EVENTS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  ... processed {line_count} lines, found {quote_count} quotes")

            if '"QUOTE_UPDATE"' not in line:
                continue

            try:
                event = json.loads(line.strip())
            except json.JSONDecodeError:
                error_count += 1
                continue

            if event.get("event_type") != "QUOTE_UPDATE":
                continue

            payload = event.get("payload", {})
            symbol = payload.get("symbol", "")

            if symbol not in target_symbols:
                continue

            timestamp = event.get("timestamp", "")
            try:
                dt = parse_event_timestamp(timestamp)
            except Exception:
                error_count += 1
                continue

            quote_count += 1
            symbol_quotes[symbol].append({
                "ts": dt.isoformat(),
                "epoch": dt.timestamp(),
                "bid": payload.get("bid"),
                "ask": payload.get("ask"),
                "last": payload.get("last"),
                "volume": payload.get("volume"),
            })

    print(f"  Total lines: {line_count}, quotes extracted: {quote_count}, errors: {error_count}")

    # Sort by timestamp and cache per symbol
    QUOTES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in target_symbols:
        quotes = symbol_quotes.get(sym, [])
        quotes.sort(key=lambda q: q["epoch"])
        cache_file = QUOTES_CACHE_DIR / f"{sym}_quotes.json"
        with open(cache_file, "w") as f:
            json.dump({"symbol": sym, "date": STUDY_DATE, "count": len(quotes), "quotes": quotes}, f)
        print(f"  {sym}: {len(quotes)} quotes cached -> {cache_file.name}")

    # Check coverage
    missing = [s for s in target_symbols if len(symbol_quotes.get(s, [])) == 0]
    if missing:
        print(f"  WARNING: No quotes found for: {missing}")

    coverage_pct = (len(target_symbols) - len(missing)) / len(target_symbols) * 100
    print(f"  Coverage: {coverage_pct:.0f}% ({len(target_symbols) - len(missing)}/{len(target_symbols)} symbols have data)")

    if coverage_pct < 50:
        print(f"  INSUFFICIENT DATA: <50% symbol coverage. Flagging as insufficient.")

    return True


def load_quotes(symbol):
    """Load cached quotes for a symbol."""
    cache_file = QUOTES_CACHE_DIR / f"{symbol}_quotes.json"
    if not cache_file.exists():
        return []
    with open(cache_file, "r") as f:
        data = json.load(f)
    return data.get("quotes", [])


def compute_forward_metrics(quotes, signal_epoch, entry_price, window_sec):
    """
    Compute MFE%, MAE%, close_return% for a given forward window.

    Uses 'last' price from QUOTE_UPDATE as the market price.
    Falls back to midpoint of bid/ask if 'last' is null.
    """
    window_end = signal_epoch + window_sec

    # Get all quotes in the forward window
    forward_quotes = []
    for q in quotes:
        if q["epoch"] > signal_epoch and q["epoch"] <= window_end:
            price = q.get("last")
            if price is None or price <= 0:
                bid = q.get("bid", 0) or 0
                ask = q.get("ask", 0) or 0
                if bid > 0 and ask > 0:
                    price = (bid + ask) / 2
                else:
                    continue
            forward_quotes.append({
                "epoch": q["epoch"],
                "price": price,
                "bid": q.get("bid"),
                "ask": q.get("ask"),
            })

    if not forward_quotes:
        return None

    # Compute returns relative to entry
    returns = [(q["price"] - entry_price) / entry_price * 100 for q in forward_quotes]

    mfe = max(returns)  # max favorable excursion
    mae = min(returns)  # max adverse excursion (most negative)
    close_return = returns[-1]  # return at end of window
    close_price = forward_quotes[-1]["price"]
    n_ticks = len(forward_quotes)

    return {
        "mfe_pct": round(mfe, 4),
        "mae_pct": round(mae, 4),
        "close_return_pct": round(close_return, 4),
        "close_price": close_price,
        "n_ticks": n_ticks,
    }


def simulate_exit_model(quotes, signal_epoch, entry_price):
    """
    Simulate exit model over 5-minute window:
    - Stop: -1.0%
    - Trail activate at +0.8%, trail distance 0.4%
    - Otherwise exit at +5m close

    Returns dict with exit details.
    """
    window_end = signal_epoch + EXIT_WINDOW_SEC

    # Get forward quotes
    forward_quotes = []
    for q in quotes:
        if q["epoch"] > signal_epoch and q["epoch"] <= window_end:
            price = q.get("last")
            if price is None or price <= 0:
                bid = q.get("bid", 0) or 0
                ask = q.get("ask", 0) or 0
                if bid > 0 and ask > 0:
                    price = (bid + ask) / 2
                else:
                    continue
            forward_quotes.append({
                "epoch": q["epoch"],
                "price": price,
            })

    if not forward_quotes:
        return {
            "exit_type": "NO_DATA",
            "exit_return_pct": None,
            "exit_price": None,
            "exit_time_sec": None,
            "trail_activated": False,
            "peak_return_pct": None,
        }

    trail_activated = False
    trail_high = entry_price
    peak_return = 0.0

    for q in forward_quotes:
        price = q["price"]
        ret_pct = (price - entry_price) / entry_price * 100
        elapsed_sec = q["epoch"] - signal_epoch

        # Track peak
        if ret_pct > peak_return:
            peak_return = ret_pct

        # Check hard stop
        if ret_pct <= STOP_PCT:
            return {
                "exit_type": "STOP",
                "exit_return_pct": round(ret_pct, 4),
                "exit_price": price,
                "exit_time_sec": round(elapsed_sec, 1),
                "trail_activated": trail_activated,
                "peak_return_pct": round(peak_return, 4),
            }

        # Check trailing stop
        if not trail_activated and ret_pct >= TRAIL_ACTIVATE:
            trail_activated = True
            trail_high = price

        if trail_activated:
            if price > trail_high:
                trail_high = price

            trail_stop_price = trail_high * (1 - TRAIL_DISTANCE / 100)
            if price <= trail_stop_price:
                return {
                    "exit_type": "TRAIL_STOP",
                    "exit_return_pct": round(ret_pct, 4),
                    "exit_price": price,
                    "exit_time_sec": round(elapsed_sec, 1),
                    "trail_activated": True,
                    "peak_return_pct": round(peak_return, 4),
                }

    # No stop triggered -> exit at 5m close (last quote in window)
    last = forward_quotes[-1]
    final_ret = (last["price"] - entry_price) / entry_price * 100
    return {
        "exit_type": "TIME_EXIT",
        "exit_return_pct": round(final_ret, 4),
        "exit_price": last["price"],
        "exit_time_sec": round(last["epoch"] - signal_epoch, 1),
        "trail_activated": trail_activated,
        "peak_return_pct": round(peak_return, 4),
    }


def step2_analyze_signals(pass_signals):
    """Run forward-window analysis and exit model for each v2-pass signal."""
    print(f"\n=== STEP 2: Forward-window analysis for {len(pass_signals)} signals ===")

    results = []
    symbols_with_data = set()
    symbols_no_data = set()

    # Pre-load all symbol quotes
    symbol_cache = {}
    symbols = set(s["symbol"] for s in pass_signals)
    for sym in symbols:
        quotes = load_quotes(sym)
        symbol_cache[sym] = quotes
        if quotes:
            symbols_with_data.add(sym)
            print(f"  Loaded {len(quotes)} quotes for {sym}")
        else:
            symbols_no_data.add(sym)
            print(f"  WARNING: No quotes for {sym}")

    coverage_pct = len(symbols_with_data) / len(symbols) * 100 if symbols else 0
    print(f"\n  Symbol coverage: {len(symbols_with_data)}/{len(symbols)} ({coverage_pct:.0f}%)")

    if coverage_pct < 50:
        print(f"  INSUFFICIENT DATA FLAG: <50% symbols have price data")

    # Process each signal
    for sig in pass_signals:
        sym = sig["symbol"]
        ts = sig["timestamp"]
        entry = sig["entry_price"]
        signal_epoch = parse_timestamp(ts).timestamp()
        quotes = symbol_cache.get(sym, [])

        sig_result = {
            "num": sig["num"],
            "symbol": sym,
            "timestamp": ts,
            "strategy": sig["strategy"],
            "phase": sig["phase"],
            "confidence": sig["confidence"],
            "momentum_score": sig["momentum_score"],
            "entry_price": entry,
            "spread_pct": sig["spread_pct"],
            "has_price_data": len(quotes) > 0,
        }

        if not quotes:
            sig_result["windows"] = {}
            sig_result["exit_model"] = {
                "exit_type": "NO_DATA",
                "exit_return_pct": None,
            }
            results.append(sig_result)
            continue

        # Forward window metrics
        windows = {}
        for w_sec in WINDOWS:
            label = WINDOW_LABELS[w_sec]
            metrics = compute_forward_metrics(quotes, signal_epoch, entry, w_sec)
            if metrics:
                windows[label] = metrics
            else:
                windows[label] = {"mfe_pct": None, "mae_pct": None, "close_return_pct": None, "n_ticks": 0}

        sig_result["windows"] = windows

        # Exit model simulation
        sig_result["exit_model"] = simulate_exit_model(quotes, signal_epoch, entry)

        results.append(sig_result)

    return results, symbols_with_data, symbols_no_data


def step3_generate_summary(results, symbols_with_data, symbols_no_data):
    """Generate summary statistics."""
    print(f"\n=== STEP 3: Generate summary statistics ===")

    # Filter to signals with data
    with_data = [r for r in results if r["has_price_data"]]
    without_data = [r for r in results if not r["has_price_data"]]

    print(f"  Signals with price data: {len(with_data)}")
    print(f"  Signals without price data: {len(without_data)}")

    summary = {
        "date": STUDY_DATE,
        "total_v2_pass": len(results),
        "signals_with_data": len(with_data),
        "signals_no_data": len(without_data),
        "symbols_with_data": sorted(symbols_with_data),
        "symbols_no_data": sorted(symbols_no_data),
        "coverage_pct": round(len(symbols_with_data) / (len(symbols_with_data) + len(symbols_no_data)) * 100, 1) if (symbols_with_data or symbols_no_data) else 0,
        "insufficient_data": len(symbols_with_data) / (len(symbols_with_data) + len(symbols_no_data)) < 0.50 if (symbols_with_data or symbols_no_data) else True,
    }

    # Window summary statistics
    window_stats = {}
    for label in ["1m", "3m", "5m", "10m"]:
        returns = []
        mfes = []
        maes = []
        for r in with_data:
            w = r.get("windows", {}).get(label, {})
            if w.get("close_return_pct") is not None:
                returns.append(w["close_return_pct"])
            if w.get("mfe_pct") is not None:
                mfes.append(w["mfe_pct"])
            if w.get("mae_pct") is not None:
                maes.append(w["mae_pct"])

        if returns:
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r <= 0]
            window_stats[label] = {
                "n": len(returns),
                "win_rate": round(len(wins) / len(returns) * 100, 1),
                "avg_return": round(sum(returns) / len(returns), 4),
                "median_return": round(sorted(returns)[len(returns) // 2], 4),
                "avg_mfe": round(sum(mfes) / len(mfes), 4) if mfes else None,
                "avg_mae": round(sum(maes) / len(maes), 4) if maes else None,
                "max_mfe": round(max(mfes), 4) if mfes else None,
                "min_mae": round(min(maes), 4) if maes else None,
                "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
                "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
            }
        else:
            window_stats[label] = {"n": 0, "win_rate": None, "avg_return": None}

    summary["window_stats"] = window_stats

    # Exit model summary
    exit_types = defaultdict(int)
    exit_returns = []
    for r in with_data:
        em = r.get("exit_model", {})
        exit_types[em.get("exit_type", "UNKNOWN")] += 1
        if em.get("exit_return_pct") is not None:
            exit_returns.append(em["exit_return_pct"])

    exit_wins = [r for r in exit_returns if r > 0]
    exit_losses = [r for r in exit_returns if r <= 0]

    summary["exit_model"] = {
        "params": {
            "stop_pct": STOP_PCT,
            "trail_activate_pct": TRAIL_ACTIVATE,
            "trail_distance_pct": TRAIL_DISTANCE,
            "exit_window_sec": EXIT_WINDOW_SEC,
        },
        "n": len(exit_returns),
        "exit_type_distribution": dict(exit_types),
        "win_rate": round(len(exit_wins) / len(exit_returns) * 100, 1) if exit_returns else None,
        "avg_return": round(sum(exit_returns) / len(exit_returns), 4) if exit_returns else None,
        "median_return": round(sorted(exit_returns)[len(exit_returns) // 2], 4) if exit_returns else None,
        "total_return": round(sum(exit_returns), 4) if exit_returns else None,
        "avg_win": round(sum(exit_wins) / len(exit_wins), 4) if exit_wins else None,
        "avg_loss": round(sum(exit_losses) / len(exit_losses), 4) if exit_losses else None,
        "best_trade": round(max(exit_returns), 4) if exit_returns else None,
        "worst_trade": round(min(exit_returns), 4) if exit_returns else None,
    }

    # Strategy breakdown
    strategy_stats = {}
    for strat in set(r["strategy"] for r in with_data):
        strat_signals = [r for r in with_data if r["strategy"] == strat]
        strat_exits = [r["exit_model"]["exit_return_pct"] for r in strat_signals
                       if r["exit_model"].get("exit_return_pct") is not None]
        strat_wins = [r for r in strat_exits if r > 0]
        strat_losses = [r for r in strat_exits if r <= 0]

        # 5m window stats for strategy
        strat_5m = []
        for r in strat_signals:
            w = r.get("windows", {}).get("5m", {})
            if w.get("close_return_pct") is not None:
                strat_5m.append(w["close_return_pct"])

        strategy_stats[strat] = {
            "n": len(strat_signals),
            "exit_model_win_rate": round(len(strat_wins) / len(strat_exits) * 100, 1) if strat_exits else None,
            "exit_model_avg_return": round(sum(strat_exits) / len(strat_exits), 4) if strat_exits else None,
            "window_5m_win_rate": round(len([r for r in strat_5m if r > 0]) / len(strat_5m) * 100, 1) if strat_5m else None,
            "window_5m_avg_return": round(sum(strat_5m) / len(strat_5m), 4) if strat_5m else None,
        }

    summary["strategy_breakdown"] = strategy_stats

    # Per-symbol breakdown
    symbol_stats = {}
    for sym in sorted(set(r["symbol"] for r in with_data)):
        sym_signals = [r for r in with_data if r["symbol"] == sym]
        sym_exits = [r["exit_model"]["exit_return_pct"] for r in sym_signals
                     if r["exit_model"].get("exit_return_pct") is not None]
        sym_wins = [r for r in sym_exits if r > 0]

        symbol_stats[sym] = {
            "n": len(sym_signals),
            "exit_win_rate": round(len(sym_wins) / len(sym_exits) * 100, 1) if sym_exits else None,
            "exit_avg_return": round(sum(sym_exits) / len(sym_exits), 4) if sym_exits else None,
            "exit_total_return": round(sum(sym_exits), 4) if sym_exits else None,
        }

    summary["symbol_breakdown"] = symbol_stats

    # v2 block reason analysis: compare pass vs block outcomes
    # (For this we'd need block signals too - skip for now, just note it)

    return summary


def step4_generate_report(summary, results):
    """Generate markdown report."""
    print(f"\n=== STEP 4: Generate report ===")

    ws = summary["window_stats"]
    em = summary["exit_model"]

    lines = []
    lines.append(f"# Edge Preservation Study - Containment v2 (2026-03-03)")
    lines.append(f"")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append(f"**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append(f"**Basis:** {summary['total_v2_pass']} v2-pass signals from 2026-03-03")
    lines.append(f"**Price data:** QUOTE_UPDATE events from production events log")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Data coverage
    lines.append(f"## Data Coverage")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total v2-pass signals | {summary['total_v2_pass']} |")
    lines.append(f"| Signals with price data | {summary['signals_with_data']} |")
    lines.append(f"| Signals without data | {summary['signals_no_data']} |")
    lines.append(f"| Symbol coverage | {summary['coverage_pct']}% ({len(summary['symbols_with_data'])}/{len(summary['symbols_with_data']) + len(summary['symbols_no_data'])}) |")
    if summary["insufficient_data"]:
        lines.append(f"| **DATA FLAG** | **INSUFFICIENT: <50% coverage** |")
    if summary["symbols_no_data"]:
        lines.append(f"| Missing symbols | {', '.join(summary['symbols_no_data'])} |")
    lines.append(f"")

    # Forward window summary
    lines.append(f"## Forward Window Summary")
    lines.append(f"")
    lines.append(f"| Window | N | Win Rate | Avg Return | Median | Avg MFE | Avg MAE | Avg Win | Avg Loss |")
    lines.append(f"|--------|---|----------|------------|--------|---------|---------|---------|----------|")
    for label in ["1m", "3m", "5m", "10m"]:
        s = ws.get(label, {})
        if s.get("n", 0) == 0:
            lines.append(f"| {label} | 0 | - | - | - | - | - | - | - |")
        else:
            avg_win_str = "-" if s.get("avg_win") is None else f"{s['avg_win']:.3f}%"
            avg_loss_str = "-" if s.get("avg_loss") is None else f"{s['avg_loss']:.3f}%"
            lines.append(f"| {label} | {s['n']} | {s['win_rate']:.1f}% | {s['avg_return']:.3f}% | {s['median_return']:.3f}% | {s['avg_mfe']:.3f}% | {s['avg_mae']:.3f}% | {avg_win_str} | {avg_loss_str} |")
    lines.append(f"")

    # Exit model summary
    lines.append(f"## Exit Model Results")
    lines.append(f"")
    lines.append(f"**Parameters:** Stop={STOP_PCT}%, Trail activate={TRAIL_ACTIVATE}%, Trail distance={TRAIL_DISTANCE}%, Exit window={EXIT_WINDOW_SEC}s")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Trades evaluated | {em['n']} |")
    lines.append(f"| Win rate | {em['win_rate']:.1f}% |" if em['win_rate'] is not None else "| Win rate | - |")
    lines.append(f"| Avg return | {em['avg_return']:.3f}% |" if em['avg_return'] is not None else "| Avg return | - |")
    lines.append(f"| Median return | {em['median_return']:.3f}% |" if em['median_return'] is not None else "| Median return | - |")
    lines.append(f"| Total return (sum) | {em['total_return']:.3f}% |" if em['total_return'] is not None else "| Total return | - |")
    lines.append(f"| Avg win | {em['avg_win']:.3f}% |" if em['avg_win'] is not None else "| Avg win | - |")
    lines.append(f"| Avg loss | {em['avg_loss']:.3f}% |" if em['avg_loss'] is not None else "| Avg loss | - |")
    lines.append(f"| Best trade | {em['best_trade']:.3f}% |" if em['best_trade'] is not None else "| Best trade | - |")
    lines.append(f"| Worst trade | {em['worst_trade']:.3f}% |" if em['worst_trade'] is not None else "| Worst trade | - |")
    lines.append(f"")

    # Exit type distribution
    lines.append(f"### Exit Type Distribution")
    lines.append(f"")
    lines.append(f"| Exit Type | Count | % |")
    lines.append(f"|-----------|-------|---|")
    for etype, count in sorted(em["exit_type_distribution"].items(), key=lambda x: -x[1]):
        pct = count / em["n"] * 100 if em["n"] > 0 else 0
        lines.append(f"| {etype} | {count} | {pct:.1f}% |")
    lines.append(f"")

    # Strategy breakdown
    lines.append(f"## Strategy Breakdown")
    lines.append(f"")
    lines.append(f"| Strategy | N | Exit Win Rate | Exit Avg Return | 5m Win Rate | 5m Avg Return |")
    lines.append(f"|----------|---|--------------|-----------------|-------------|---------------|")
    for strat, stats in summary["strategy_breakdown"].items():
        wr = f"{stats['exit_model_win_rate']:.1f}%" if stats['exit_model_win_rate'] is not None else "-"
        ar = f"{stats['exit_model_avg_return']:.3f}%" if stats['exit_model_avg_return'] is not None else "-"
        wr5 = f"{stats['window_5m_win_rate']:.1f}%" if stats['window_5m_win_rate'] is not None else "-"
        ar5 = f"{stats['window_5m_avg_return']:.3f}%" if stats['window_5m_avg_return'] is not None else "-"
        lines.append(f"| {strat} | {stats['n']} | {wr} | {ar} | {wr5} | {ar5} |")
    lines.append(f"")

    # Symbol breakdown
    lines.append(f"## Per-Symbol Breakdown")
    lines.append(f"")
    lines.append(f"| Symbol | N | Exit Win Rate | Exit Avg Return | Total Return |")
    lines.append(f"|--------|---|--------------|-----------------|--------------|")
    for sym, stats in summary["symbol_breakdown"].items():
        wr = f"{stats['exit_win_rate']:.1f}%" if stats['exit_win_rate'] is not None else "-"
        ar = f"{stats['exit_avg_return']:.3f}%" if stats['exit_avg_return'] is not None else "-"
        tr = f"{stats['exit_total_return']:.3f}%" if stats['exit_total_return'] is not None else "-"
        lines.append(f"| {sym} | {stats['n']} | {wr} | {ar} | {tr} |")
    lines.append(f"")

    # Signal-level detail table (top/bottom performers)
    with_exits = [r for r in results if r.get("exit_model", {}).get("exit_return_pct") is not None]
    if with_exits:
        sorted_by_return = sorted(with_exits, key=lambda r: r["exit_model"]["exit_return_pct"], reverse=True)

        lines.append(f"## Top 10 Trades")
        lines.append(f"")
        lines.append(f"| # | Symbol | Time | Strategy | Entry | Exit Type | Return | MFE(5m) | MAE(5m) |")
        lines.append(f"|---|--------|------|----------|-------|-----------|--------|---------|---------|")
        for r in sorted_by_return[:10]:
            em_r = r["exit_model"]
            w5m = r.get("windows", {}).get("5m", {})
            mfe5 = f"{w5m['mfe_pct']:.3f}%" if w5m.get("mfe_pct") is not None else "-"
            mae5 = f"{w5m['mae_pct']:.3f}%" if w5m.get("mae_pct") is not None else "-"
            lines.append(f"| {r['num']} | {r['symbol']} | {r['timestamp']} | {r['strategy'][:15]} | ${r['entry_price']:.2f} | {em_r['exit_type']} | {em_r['exit_return_pct']:.3f}% | {mfe5} | {mae5} |")
        lines.append(f"")

        lines.append(f"## Bottom 10 Trades")
        lines.append(f"")
        lines.append(f"| # | Symbol | Time | Strategy | Entry | Exit Type | Return | MFE(5m) | MAE(5m) |")
        lines.append(f"|---|--------|------|----------|-------|-----------|--------|---------|---------|")
        for r in sorted_by_return[-10:]:
            em_r = r["exit_model"]
            w5m = r.get("windows", {}).get("5m", {})
            mfe5 = f"{w5m['mfe_pct']:.3f}%" if w5m.get("mfe_pct") is not None else "-"
            mae5 = f"{w5m['mae_pct']:.3f}%" if w5m.get("mae_pct") is not None else "-"
            lines.append(f"| {r['num']} | {r['symbol']} | {r['timestamp']} | {r['strategy'][:15]} | ${r['entry_price']:.2f} | {em_r['exit_type']} | {em_r['exit_return_pct']:.3f}% | {mfe5} | {mae5} |")
        lines.append(f"")

    # Full signal table
    lines.append(f"## Full Signal Detail")
    lines.append(f"")
    lines.append(f"| # | Sym | Time | Strat | Entry | Spread | Conf | Exit Type | Exit Ret | MFE(1m) | MAE(1m) | MFE(5m) | MAE(5m) | MFE(10m) | MAE(10m) |")
    lines.append(f"|---|-----|------|-------|-------|--------|------|-----------|----------|---------|---------|---------|---------|----------|----------|")
    for r in results:
        em_r = r.get("exit_model", {})
        exit_ret = f"{em_r['exit_return_pct']:.3f}%" if em_r.get("exit_return_pct") is not None else "NO_DATA"
        exit_type = em_r.get("exit_type", "?")

        def fmt_w(window_label, field):
            w = r.get("windows", {}).get(window_label, {})
            v = w.get(field)
            return f"{v:.3f}%" if v is not None else "-"

        lines.append(f"| {r['num']} | {r['symbol']} | {r['timestamp']} | {r['strategy'][:12]} | {r['entry_price']:.2f} | {r['spread_pct']:.2f}% | {r['confidence']:.2f} | {exit_type} | {exit_ret} | {fmt_w('1m','mfe_pct')} | {fmt_w('1m','mae_pct')} | {fmt_w('5m','mfe_pct')} | {fmt_w('5m','mae_pct')} | {fmt_w('10m','mfe_pct')} | {fmt_w('10m','mae_pct')} |")
    lines.append(f"")

    # Conclusions
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Key Findings")
    lines.append(f"")

    if em["win_rate"] is not None:
        if em["win_rate"] >= 50 and em["avg_return"] > 0:
            lines.append(f"- **POSITIVE EDGE**: {em['win_rate']:.1f}% win rate with {em['avg_return']:.3f}% avg return suggests the v2 filter preserves a tradeable edge.")
        elif em["win_rate"] >= 40:
            lines.append(f"- **MARGINAL EDGE**: {em['win_rate']:.1f}% win rate with {em['avg_return']:.3f}% avg return. Edge exists but may require refinement.")
        else:
            lines.append(f"- **WEAK/NO EDGE**: {em['win_rate']:.1f}% win rate with {em['avg_return']:.3f}% avg return. The v2 filter may be too permissive.")

    # MFE vs MAE analysis
    w5 = ws.get("5m", {})
    if w5.get("avg_mfe") is not None and w5.get("avg_mae") is not None:
        mfe_mae_ratio = abs(w5["avg_mfe"] / w5["avg_mae"]) if w5["avg_mae"] != 0 else float("inf")
        if mfe_mae_ratio > 1.5:
            lines.append(f"- **FAVORABLE MFE/MAE**: 5m avg MFE ({w5['avg_mfe']:.3f}%) vs avg MAE ({w5['avg_mae']:.3f}%) ratio = {mfe_mae_ratio:.2f}x, indicating favorable risk/reward.")
        elif mfe_mae_ratio > 1.0:
            lines.append(f"- **NEUTRAL MFE/MAE**: 5m avg MFE ({w5['avg_mfe']:.3f}%) vs avg MAE ({w5['avg_mae']:.3f}%) ratio = {mfe_mae_ratio:.2f}x.")
        else:
            lines.append(f"- **UNFAVORABLE MFE/MAE**: 5m avg MFE ({w5['avg_mfe']:.3f}%) vs avg MAE ({w5['avg_mae']:.3f}%) ratio = {mfe_mae_ratio:.2f}x, suggesting adverse price action dominates.")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*This study is research-only. No production changes applied.*")
    lines.append(f"*All recommendations require review and approval before implementation.*")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("EDGE PRESERVATION STUDY - Containment v2 (2026-03-03)")
    print("SuperBot Research Engine - READ ONLY")
    print("=" * 70)

    # Load v2 replay data
    print(f"\nLoading v2 replay data from {REPLAY_JSON}")
    with open(REPLAY_JSON, "r") as f:
        replay = json.load(f)

    pass_signals = [s for s in replay["signals"] if s["v2_result"] == "PASS"]
    print(f"  Total signals: {replay['total_signals']}")
    print(f"  v2-pass signals: {len(pass_signals)}")

    target_symbols = sorted(set(s["symbol"] for s in pass_signals))
    print(f"  Target symbols: {target_symbols}")

    # Step 1: Extract and cache QUOTE_UPDATE data
    success = step1_extract_quotes(set(target_symbols))
    if not success:
        print("\nERROR: Failed to extract quote data. Aborting.")
        sys.exit(1)

    # Step 2: Forward-window analysis
    results, symbols_with_data, symbols_no_data = step2_analyze_signals(pass_signals)

    # Step 3: Summary statistics
    summary = step3_generate_summary(results, symbols_with_data, symbols_no_data)

    # Step 4: Generate report
    report_md = step4_generate_report(summary, results)

    # Write outputs
    output_data = {
        "date": STUDY_DATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "type": "edge_preservation_v2",
        "summary": summary,
        "signals": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nJSON output -> {OUTPUT_JSON}")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"MD report  -> {OUTPUT_MD}")

    # Print key results
    em = summary["exit_model"]
    print(f"\n{'=' * 70}")
    print(f"KEY RESULTS")
    print(f"{'=' * 70}")
    print(f"  Signals analyzed: {summary['signals_with_data']}/{summary['total_v2_pass']}")
    print(f"  Symbol coverage: {summary['coverage_pct']}%")
    if em["win_rate"] is not None:
        print(f"  Exit model win rate: {em['win_rate']:.1f}%")
        print(f"  Exit model avg return: {em['avg_return']:.3f}%")
        print(f"  Exit model total return: {em['total_return']:.3f}%")
    for label in ["1m", "3m", "5m", "10m"]:
        ws = summary["window_stats"].get(label, {})
        if ws.get("n", 0) > 0:
            print(f"  {label} window: win_rate={ws['win_rate']:.1f}%, avg={ws['avg_return']:.3f}%, MFE={ws['avg_mfe']:.3f}%, MAE={ws['avg_mae']:.3f}%")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
