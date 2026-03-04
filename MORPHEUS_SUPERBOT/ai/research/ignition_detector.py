#!/usr/bin/env python3
"""
Momentum Ignition Detection Study (SuperBot Research)
READ ONLY - no production changes.

Detects ignition events in quote streams and measures how far after
ignition Morpheus enters. Correlates delay with profitability.

Usage:
    python -m ai.research.ignition_detector --date 2026-03-03
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean, stdev

# === PATHS ===
SUPERBOT_ROOT = Path("C:/AI_Bot_Research/MORPHEUS_SUPERBOT")
QUOTES_CACHE_DIR = SUPERBOT_ROOT / "engine" / "cache" / "quotes"
REPLAY_JSON = SUPERBOT_ROOT / "engine" / "output" / "containment_v2_replay_2026-03-03.json"
EDGE_JSON = SUPERBOT_ROOT / "engine" / "output" / "edge_preservation_v2_2026-03-03.json"
OUTPUT_DIR = SUPERBOT_ROOT / "engine" / "output"
COMMS_PATH = SUPERBOT_ROOT / "comms" / "outbox_chatgpt.json"

# === IGNITION DETECTION PARAMETERS ===
# Rolling windows for indicators
VELOCITY_WINDOWS = [3, 5]  # seconds (1s too fine for typical tick density)
QUOTE_RATE_WINDOW = 5      # seconds
SPREAD_ROLLING_WINDOW = 30  # seconds for rolling avg spread

# Ignition thresholds (per-symbol relative, calibrated below)
MIN_CRITERIA_FOR_IGNITION = 3  # out of 5
IGNITION_LOOKBACK = 5  # seconds for multi-criteria coincidence

# Bid stepping parameters
BID_STEP_WINDOW = 5   # seconds
BID_STEP_MIN_TICKS = 3  # minimum bid increases

# Micro breakout
BREAKOUT_LOOKBACK = 30  # seconds for recent high

# === EXIT MODEL ===
STOP_PCT = -1.0
TRAIL_ACTIVATE = 0.8
TRAIL_DISTANCE = 0.4
EXIT_WINDOW_SEC = 300

# === DELAY BUCKETS ===
DELAY_BUCKETS = [
    (0, 5, "0-5s"),
    (5, 15, "5-15s"),
    (15, 30, "15-30s"),
    (30, 60, "30-60s"),
    (60, 120, "60-120s"),
    (120, 300, "120-300s"),
    (300, float("inf"), "300+s"),
]


def parse_signal_epoch(ts_str):
    h, m, s = ts_str.split(":")
    return datetime(2026, 3, 3, int(h), int(m), int(s), tzinfo=timezone.utc).timestamp()


def epoch_to_str(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


def load_quotes_with_ffill(symbol):
    """
    Load cached quotes with forward-filled bid/ask.
    Returns list of {epoch, price, bid, ask, spread_pct}.
    """
    cache_file = QUOTES_CACHE_DIR / f"{symbol}_quotes.json"
    if not cache_file.exists():
        return []
    with open(cache_file) as f:
        data = json.load(f)

    result = []
    last_bid = None
    last_ask = None

    for q in data.get("quotes", []):
        price = q.get("last")
        bid = q.get("bid")
        ask = q.get("ask")

        # Forward-fill bid/ask
        if bid is not None and bid > 0:
            last_bid = bid
        if ask is not None and ask > 0:
            last_ask = ask

        if price is None or price <= 0:
            if last_bid and last_ask:
                price = (last_bid + last_ask) / 2
            else:
                continue

        cur_bid = last_bid if last_bid and last_bid > 0 else None
        cur_ask = last_ask if last_ask and last_ask > 0 else None

        spread_pct = None
        if cur_bid and cur_ask and cur_ask > cur_bid:
            mid = (cur_bid + cur_ask) / 2
            spread_pct = (cur_ask - cur_bid) / mid * 100

        result.append({
            "epoch": q["epoch"],
            "price": price,
            "bid": cur_bid,
            "ask": cur_ask,
            "spread_pct": spread_pct,
        })

    return result


# ========================================================================
# STEP 3: COMPUTE IGNITION INDICATORS
# ========================================================================
def compute_indicators(quotes):
    """
    Compute rolling ignition indicators at each quote tick.
    Returns list of dicts with indicators per tick.
    """
    n = len(quotes)
    if n < 5:
        return []

    ticks = []

    for i in range(n):
        t = quotes[i]["epoch"]
        p = quotes[i]["price"]

        tick = {
            "idx": i,
            "epoch": t,
            "price": p,
            "bid": quotes[i]["bid"],
            "ask": quotes[i]["ask"],
            "spread_pct": quotes[i]["spread_pct"],
        }

        # --- Price velocity (3s, 5s) ---
        for w in VELOCITY_WINDOWS:
            lookback_quotes = [q for q in quotes[max(0, i-50):i+1]
                               if t - w <= q["epoch"] <= t]
            if len(lookback_quotes) >= 2:
                p_start = lookback_quotes[0]["price"]
                p_end = lookback_quotes[-1]["price"]
                dt = lookback_quotes[-1]["epoch"] - lookback_quotes[0]["epoch"]
                if dt > 0 and p_start > 0:
                    vel = (p_end - p_start) / p_start * 100 / (dt / w)
                    tick[f"velocity_{w}s"] = round(vel, 4)
                else:
                    tick[f"velocity_{w}s"] = 0.0
            else:
                tick[f"velocity_{w}s"] = 0.0

        # --- Quote rate (5s window) ---
        rate_quotes = [q for q in quotes[max(0, i-100):i+1]
                       if t - QUOTE_RATE_WINDOW <= q["epoch"] <= t]
        tick["quote_rate_5s"] = len(rate_quotes) / QUOTE_RATE_WINDOW

        # --- Spread compression ---
        # Current spread vs rolling 30s avg
        rolling_spreads = [q["spread_pct"] for q in quotes[max(0, i-100):i+1]
                           if t - SPREAD_ROLLING_WINDOW <= q["epoch"] <= t
                           and q["spread_pct"] is not None]
        if rolling_spreads and tick["spread_pct"] is not None:
            avg_spread = mean(rolling_spreads)
            if avg_spread > 0:
                tick["spread_compression_ratio"] = round(tick["spread_pct"] / avg_spread, 4)
            else:
                tick["spread_compression_ratio"] = 1.0
            tick["spread_rolling_avg"] = round(avg_spread, 4)
        else:
            tick["spread_compression_ratio"] = 1.0
            tick["spread_rolling_avg"] = None

        # --- Bid stepping (count bid increases in last 5s) ---
        bid_window = [q for q in quotes[max(0, i-50):i+1]
                      if t - BID_STEP_WINDOW <= q["epoch"] <= t
                      and q["bid"] is not None]
        bid_steps = 0
        for j in range(1, len(bid_window)):
            if bid_window[j]["bid"] is not None and bid_window[j-1]["bid"] is not None:
                if bid_window[j]["bid"] > bid_window[j-1]["bid"]:
                    bid_steps += 1
        tick["bid_steps_5s"] = bid_steps

        # --- Micro breakout (price crosses recent high) ---
        high_window = [q for q in quotes[max(0, i-200):i]
                       if t - BREAKOUT_LOOKBACK <= q["epoch"] < t]
        if high_window:
            recent_high = max(q["price"] for q in high_window)
            tick["recent_high"] = recent_high
            tick["micro_breakout"] = p > recent_high
        else:
            tick["recent_high"] = p
            tick["micro_breakout"] = False

        ticks.append(tick)

    return ticks


def calibrate_thresholds(ticks):
    """
    Compute per-symbol adaptive thresholds based on distribution of indicators.
    Uses percentile-based thresholds (top 10% = extreme).
    """
    vel_3s = [t["velocity_3s"] for t in ticks if t["velocity_3s"] != 0]
    vel_5s = [t["velocity_5s"] for t in ticks if t["velocity_5s"] != 0]
    rates = [t["quote_rate_5s"] for t in ticks if t["quote_rate_5s"] > 0]

    thresholds = {}

    # Velocity threshold: 90th percentile of absolute values
    if vel_3s:
        abs_vel = sorted([abs(v) for v in vel_3s])
        thresholds["velocity_3s"] = abs_vel[int(len(abs_vel) * 0.85)] if len(abs_vel) > 10 else 0.1
    else:
        thresholds["velocity_3s"] = 0.1

    if vel_5s:
        abs_vel = sorted([abs(v) for v in vel_5s])
        thresholds["velocity_5s"] = abs_vel[int(len(abs_vel) * 0.85)] if len(abs_vel) > 10 else 0.1
    else:
        thresholds["velocity_5s"] = 0.1

    # Quote rate threshold: 80th percentile
    if rates:
        sorted_rates = sorted(rates)
        thresholds["quote_rate_spike"] = sorted_rates[int(len(sorted_rates) * 0.80)] if len(sorted_rates) > 10 else 0.5
    else:
        thresholds["quote_rate_spike"] = 0.5

    # Spread compression: ratio < 0.7 (30% compression)
    thresholds["spread_compression"] = 0.7

    # Bid stepping: 3+ steps in 5s
    thresholds["bid_steps"] = BID_STEP_MIN_TICKS

    return thresholds


# ========================================================================
# STEP 4: DEFINE IGNITION EVENTS
# ========================================================================
def detect_ignitions(ticks, thresholds, min_criteria=3):
    """
    Detect ignition events where >= min_criteria fire within 5 seconds.
    Returns list of ignition events with timestamps and criteria met.
    """
    ignitions = []
    cooldown_until = 0  # prevent duplicate detections within 10s

    for i, tick in enumerate(ticks):
        if tick["epoch"] < cooldown_until:
            continue

        # Check each criterion
        criteria = {}

        # 1. High price velocity (3s or 5s)
        if abs(tick.get("velocity_3s", 0)) > thresholds["velocity_3s"]:
            criteria["velocity"] = {
                "value": tick["velocity_3s"],
                "threshold": thresholds["velocity_3s"],
                "direction": "up" if tick["velocity_3s"] > 0 else "down",
            }

        # 2. Quote rate spike
        if tick.get("quote_rate_5s", 0) > thresholds["quote_rate_spike"]:
            criteria["quote_rate_spike"] = {
                "value": tick["quote_rate_5s"],
                "threshold": thresholds["quote_rate_spike"],
            }

        # 3. Spread compression
        if tick.get("spread_compression_ratio", 1) < thresholds["spread_compression"]:
            criteria["spread_compression"] = {
                "value": tick["spread_compression_ratio"],
                "threshold": thresholds["spread_compression"],
            }

        # 4. Bid stepping
        if tick.get("bid_steps_5s", 0) >= thresholds["bid_steps"]:
            criteria["bid_stepping"] = {
                "value": tick["bid_steps_5s"],
                "threshold": thresholds["bid_steps"],
            }

        # 5. Micro breakout
        if tick.get("micro_breakout", False):
            criteria["micro_breakout"] = {
                "value": True,
                "recent_high": tick.get("recent_high"),
                "price": tick["price"],
            }

        if len(criteria) >= min_criteria:
            ignition = {
                "epoch": tick["epoch"],
                "time": epoch_to_str(tick["epoch"]),
                "price": tick["price"],
                "criteria_met": len(criteria),
                "criteria": criteria,
                "velocity_3s": tick.get("velocity_3s", 0),
                "velocity_5s": tick.get("velocity_5s", 0),
                "quote_rate_5s": tick.get("quote_rate_5s", 0),
                "spread_pct": tick.get("spread_pct"),
            }
            ignitions.append(ignition)
            cooldown_until = tick["epoch"] + 10  # 10s cooldown

    return ignitions


# ========================================================================
# STEP 5: MEASURE ENTRY DELAY
# ========================================================================
def measure_entry_delays(signals, ignitions_by_symbol):
    """
    For each signal, find the nearest preceding ignition and compute delay.
    """
    results = []

    for sig in signals:
        sym = sig["symbol"]
        sig_epoch = parse_signal_epoch(sig["timestamp"])
        ignitions = ignitions_by_symbol.get(sym, [])

        # Find the most recent ignition before or near signal time
        # Look for ignitions within [-600s, +5s] of signal
        # Wide window because Morpheus pipeline has significant latency
        # (signal scoring -> ignition gate -> extension -> containment -> RA)
        candidates = [ig for ig in ignitions
                      if sig_epoch - 600 <= ig["epoch"] <= sig_epoch + 5]

        if candidates:
            # Find the closest preceding ignition
            preceding = [ig for ig in candidates if ig["epoch"] <= sig_epoch]
            if preceding:
                nearest = max(preceding, key=lambda ig: ig["epoch"])
            else:
                nearest = min(candidates, key=lambda ig: abs(ig["epoch"] - sig_epoch))

            delay = sig_epoch - nearest["epoch"]
            result = {
                "num": sig["num"],
                "symbol": sym,
                "signal_time": sig["timestamp"],
                "signal_epoch": sig_epoch,
                "ignition_time": nearest["time"],
                "ignition_epoch": nearest["epoch"],
                "ignition_price": nearest["price"],
                "entry_price": sig.get("entry_price"),
                "delay_seconds": round(delay, 1),
                "criteria_met": nearest["criteria_met"],
                "ignition_velocity": nearest.get("velocity_3s", 0),
                "matched": True,
            }
        else:
            result = {
                "num": sig["num"],
                "symbol": sym,
                "signal_time": sig["timestamp"],
                "signal_epoch": sig_epoch,
                "ignition_time": None,
                "ignition_epoch": None,
                "ignition_price": None,
                "entry_price": sig.get("entry_price"),
                "delay_seconds": None,
                "criteria_met": None,
                "ignition_velocity": None,
                "matched": False,
            }

        results.append(result)

    return results


# ========================================================================
# STEP 6: PROFIT CORRELATION
# ========================================================================
def correlate_with_outcomes(delay_results, edge_signals):
    """Correlate entry delay with forward outcomes."""

    # Build outcome lookup by signal num
    outcome_map = {}
    for es in edge_signals:
        outcome_map[es["num"]] = es

    # Merge delay + outcome
    merged = []
    for dr in delay_results:
        outcome = outcome_map.get(dr["num"])
        if outcome and outcome.get("has_price_data"):
            exit_model = outcome.get("exit_model", {})
            w1m = outcome.get("windows", {}).get("1m", {})
            w5m = outcome.get("windows", {}).get("5m", {})
            w10m = outcome.get("windows", {}).get("10m", {})

            merged.append({
                **dr,
                "exit_return": exit_model.get("exit_return_pct"),
                "exit_type": exit_model.get("exit_type"),
                "return_1m": w1m.get("close_return_pct"),
                "return_5m": w5m.get("close_return_pct"),
                "return_10m": w10m.get("close_return_pct"),
                "mfe_5m": w5m.get("mfe_pct"),
                "mae_5m": w5m.get("mae_pct"),
                "win": 1 if (exit_model.get("exit_return_pct") or 0) > 0 else 0,
            })

    return merged


def compute_bucket_stats(merged_data):
    """Compute stats per delay bucket."""
    bucket_results = []

    for lo, hi, label in DELAY_BUCKETS:
        bucket_signals = [m for m in merged_data
                          if m["matched"] and m["delay_seconds"] is not None
                          and lo <= m["delay_seconds"] < hi]

        if not bucket_signals:
            bucket_results.append({
                "bucket": label,
                "lo": lo, "hi": hi if hi != float("inf") else None,
                "n": 0,
                "win_rate": None, "avg_return": None, "profit_factor": None,
                "avg_return_1m": None,
            })
            continue

        returns = [s["exit_return"] for s in bucket_signals if s.get("exit_return") is not None]
        returns_1m = [s["return_1m"] for s in bucket_signals if s.get("return_1m") is not None]

        wins = [r for r in returns if r > 0]
        gross_wins = sum(r for r in returns if r > 0)
        gross_losses = abs(sum(r for r in returns if r <= 0))
        pf = round(gross_wins / gross_losses, 3) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0)

        bucket_results.append({
            "bucket": label,
            "lo": lo, "hi": hi if hi != float("inf") else None,
            "n": len(returns),
            "win_rate": round(len(wins) / len(returns) * 100, 1) if returns else None,
            "avg_return": round(mean(returns), 4) if returns else None,
            "profit_factor": pf,
            "avg_return_1m": round(mean(returns_1m), 4) if returns_1m else None,
            "median_delay": round(median([s["delay_seconds"] for s in bucket_signals if s["delay_seconds"] is not None]), 1) if bucket_signals else None,
        })

    return bucket_results


# ========================================================================
# STEP 7: DELAY CURVE DATA
# ========================================================================
def compute_delay_curve(merged_data):
    """Compute continuous delay vs return curve in 5-second increments."""
    curve = []
    for delay_sec in range(0, 605, 5):
        lo = delay_sec
        hi = delay_sec + 5
        signals = [m for m in merged_data
                    if m["matched"] and m["delay_seconds"] is not None
                    and lo <= m["delay_seconds"] < hi]

        if not signals:
            curve.append({"delay_seconds": delay_sec, "n": 0, "avg_return": None, "win_rate": None, "profit_factor": None})
            continue

        returns = [s["exit_return"] for s in signals if s.get("exit_return") is not None]
        if not returns:
            curve.append({"delay_seconds": delay_sec, "n": 0, "avg_return": None, "win_rate": None, "profit_factor": None})
            continue

        wins = [r for r in returns if r > 0]
        gross_wins = sum(r for r in returns if r > 0)
        gross_losses = abs(sum(r for r in returns if r <= 0))
        pf = round(gross_wins / gross_losses, 3) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0)

        curve.append({
            "delay_seconds": delay_sec,
            "n": len(returns),
            "avg_return": round(mean(returns), 4),
            "win_rate": round(len(wins) / len(returns) * 100, 1),
            "profit_factor": pf,
        })

    return curve


# ========================================================================
# REPORTS
# ========================================================================
def generate_reports(date_str, ignitions_by_symbol, delay_results, merged_data,
                     bucket_stats, delay_curve, all_thresholds):
    """Generate all output files."""

    # --- ignition_events JSON ---
    ignition_out = OUTPUT_DIR / f"ignition_events_{date_str}.json"
    ignition_data = {
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "type": "ignition_events",
        "thresholds_per_symbol": {sym: {k: round(v, 4) for k, v in th.items()}
                                  for sym, th in all_thresholds.items()},
        "events_per_symbol": {},
    }
    total_ignitions = 0
    for sym, events in sorted(ignitions_by_symbol.items()):
        ignition_data["events_per_symbol"][sym] = {
            "count": len(events),
            "events": events,
        }
        total_ignitions += len(events)
    ignition_data["total_events"] = total_ignitions
    with open(ignition_out, "w") as f:
        json.dump(ignition_data, f, indent=2, default=str)

    # --- ignition_edge_curve JSON ---
    curve_out = OUTPUT_DIR / f"ignition_edge_curve_{date_str}.json"
    with open(curve_out, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "ignition_edge_curve",
            "bucket_stats": bucket_stats,
        }, f, indent=2)

    # --- ignition_delay_curve JSON ---
    delay_curve_out = OUTPUT_DIR / f"ignition_delay_curve_{date_str}.json"
    with open(delay_curve_out, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "ignition_delay_curve",
            "curve": delay_curve,
        }, f, indent=2)

    # --- Entry delay MD report ---
    matched = [d for d in delay_results if d["matched"]]
    unmatched = [d for d in delay_results if not d["matched"]]
    delays = [d["delay_seconds"] for d in matched if d["delay_seconds"] is not None]

    report_path = OUTPUT_DIR / f"ignition_entry_delay_{date_str}.md"
    lines = []
    lines.append(f"# Momentum Ignition Detection Study - {date_str}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append("**Method:** Multi-criteria ignition detection from QUOTE_UPDATE streams")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview
    lines.append("## Ignition Detection Overview")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total ignition events detected | {total_ignitions} |")
    lines.append(f"| Symbols with ignitions | {sum(1 for v in ignitions_by_symbol.values() if v)} |")
    lines.append(f"| Signals matched to ignition | {len(matched)}/{len(delay_results)} |")
    lines.append(f"| Signals with no ignition | {len(unmatched)} |")
    if delays:
        lines.append(f"| Mean entry delay | {mean(delays):.1f}s |")
        lines.append(f"| Median entry delay | {median(delays):.1f}s |")
        lines.append(f"| P90 entry delay | {sorted(delays)[int(len(delays)*0.9)]:.1f}s |")
        lines.append(f"| Min delay | {min(delays):.1f}s |")
        lines.append(f"| Max delay | {max(delays):.1f}s |")
    lines.append("")

    # Per-symbol ignition counts
    lines.append("## Ignitions Per Symbol")
    lines.append("")
    lines.append("| Symbol | Ignitions | Signals | Matched | Avg Delay |")
    lines.append("|--------|-----------|---------|---------|-----------|")
    for sym in sorted(ignitions_by_symbol.keys()):
        n_ig = len(ignitions_by_symbol[sym])
        sym_matched = [d for d in matched if d["symbol"] == sym]
        sym_delays = [d["delay_seconds"] for d in sym_matched if d["delay_seconds"] is not None]
        sym_sigs = sum(1 for d in delay_results if d["symbol"] == sym)
        avg_d = f"{mean(sym_delays):.1f}s" if sym_delays else "-"
        lines.append(f"| {sym} | {n_ig} | {sym_sigs} | {len(sym_matched)} | {avg_d} |")
    lines.append("")

    # Delay distribution
    lines.append("## Entry Delay Distribution")
    lines.append("")
    lines.append("| Bucket | Count | % | Win Rate | Avg Return | PF | 1m Avg |")
    lines.append("|--------|-------|---|----------|------------|-----|--------|")
    for bs in bucket_stats:
        n = bs["n"]
        pct = f"{n/len(matched)*100:.1f}%" if matched else "-"
        wr = f"{bs['win_rate']:.1f}%" if bs["win_rate"] is not None else "-"
        ar = f"{bs['avg_return']:+.4f}%" if bs["avg_return"] is not None else "-"
        pf = f"{bs['profit_factor']}" if bs["profit_factor"] is not None else "-"
        r1m = f"{bs['avg_return_1m']:+.4f}%" if bs["avg_return_1m"] is not None else "-"
        lines.append(f"| {bs['bucket']} | {n} | {pct} | {wr} | {ar} | {pf} | {r1m} |")
    lines.append("")

    # Delay curve visualization
    lines.append("## Ignition Delay -> Return Curve")
    lines.append("")
    lines.append("```")
    max_abs = 0
    for c in delay_curve:
        if c["avg_return"] is not None:
            max_abs = max(max_abs, abs(c["avg_return"]))
    if max_abs == 0:
        max_abs = 1
    bar_width = 30
    for c in delay_curve:
        val = c["avg_return"] if c["avg_return"] is not None else 0
        n = c["n"]
        if n == 0:
            lines.append(f"  {c['delay_seconds']:>4}s  (n=0)")
            continue
        bar_len = int(abs(val) / max_abs * bar_width)
        if val >= 0:
            bar = " " * bar_width + "|" + "#" * bar_len
        else:
            bar = " " * (bar_width - bar_len) + "#" * bar_len + "|"
        lines.append(f"  {c['delay_seconds']:>4}s  {val:+.3f}% (n={n:>2}) {bar}")
    lines.append("```")
    lines.append("")

    # Ignition examples
    lines.append("## Ignition Event Examples")
    lines.append("")
    # Show top 5 signals by delay (shortest)
    shortest = sorted([d for d in merged_data if d["matched"]], key=lambda d: d["delay_seconds"])
    lines.append("### Fastest Entries (shortest ignition delay)")
    lines.append("")
    lines.append("| # | Sym | Signal | Ignition | Delay | Entry$ | Ign$ | Exit Ret |")
    lines.append("|---|-----|--------|----------|-------|--------|------|----------|")
    for d in shortest[:8]:
        er = f"{d['exit_return']:+.3f}%" if d.get("exit_return") is not None else "-"
        lines.append(f"| {d['num']} | {d['symbol']} | {d['signal_time']} | {d['ignition_time']} | {d['delay_seconds']:.1f}s | {d.get('entry_price','?')} | {d.get('ignition_price','?')} | {er} |")
    lines.append("")

    # Longest delays
    longest = sorted([d for d in merged_data if d["matched"]], key=lambda d: d["delay_seconds"], reverse=True)
    lines.append("### Slowest Entries (longest ignition delay)")
    lines.append("")
    lines.append("| # | Sym | Signal | Ignition | Delay | Entry$ | Ign$ | Exit Ret |")
    lines.append("|---|-----|--------|----------|-------|--------|------|----------|")
    for d in longest[:8]:
        er = f"{d['exit_return']:+.3f}%" if d.get("exit_return") is not None else "-"
        lines.append(f"| {d['num']} | {d['symbol']} | {d['signal_time']} | {d['ignition_time']} | {d['delay_seconds']:.1f}s | {d.get('entry_price','?')} | {d.get('ignition_price','?')} | {er} |")
    lines.append("")

    # Key findings
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    if delays:
        lines.append(f"- **Median ignition delay: {median(delays):.1f}s** - Morpheus enters {median(delays):.0f} seconds after the ignition event")

        # Find optimal bucket
        valid_buckets = [b for b in bucket_stats if b["n"] >= 3 and b["avg_return"] is not None]
        if valid_buckets:
            best_bucket = max(valid_buckets, key=lambda b: b["avg_return"])
            lines.append(f"- **Optimal entry window: {best_bucket['bucket']}** (avg return: {best_bucket['avg_return']:+.4f}%, WR: {best_bucket['win_rate']:.1f}%)")

            worst_bucket = min(valid_buckets, key=lambda b: b["avg_return"])
            lines.append(f"- **Worst entry window: {worst_bucket['bucket']}** (avg return: {worst_bucket['avg_return']:+.4f}%, WR: {worst_bucket['win_rate']:.1f}%)")

        # Correlation
        delay_return_pairs = [(d["delay_seconds"], d["exit_return"])
                              for d in merged_data if d["matched"]
                              and d["delay_seconds"] is not None
                              and d.get("exit_return") is not None]
        if len(delay_return_pairs) >= 5:
            import numpy as np
            delays_arr = np.array([p[0] for p in delay_return_pairs])
            returns_arr = np.array([p[1] for p in delay_return_pairs])
            corr = np.corrcoef(delays_arr, returns_arr)[0, 1]
            lines.append(f"- **Delay-return correlation: {corr:+.3f}** ({'longer delay = worse returns' if corr < 0 else 'no clear delay penalty'})")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This study is research-only. No production changes applied.*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "ignition_events": ignition_out,
        "edge_curve": curve_out,
        "delay_curve": delay_curve_out,
        "report": report_path,
    }


def update_comms(date_str, delay_results, bucket_stats, delay_curve, total_ignitions, all_thresholds):
    """Append results to comms bridge."""
    if not COMMS_PATH.exists():
        return

    with open(COMMS_PATH) as f:
        comms = json.load(f)

    matched = [d for d in delay_results if d["matched"]]
    delays = [d["delay_seconds"] for d in matched if d["delay_seconds"] is not None]

    valid_buckets = [b for b in bucket_stats if b["n"] >= 3 and b["avg_return"] is not None]
    best_bucket = max(valid_buckets, key=lambda b: b["avg_return"]) if valid_buckets else None

    body_parts = [
        f"== MOMENTUM IGNITION DETECTION STUDY ({date_str}) ==",
        "",
        f"Total ignition events detected: {total_ignitions}",
        f"Signals matched to ignition: {len(matched)}/{len(delay_results)}",
    ]
    if delays:
        body_parts.append(f"Median entry delay: {median(delays):.1f}s")
        body_parts.append(f"Mean entry delay: {mean(delays):.1f}s")
        body_parts.append(f"P90 entry delay: {sorted(delays)[int(len(delays)*0.9)]:.1f}s")

    body_parts.append("")
    body_parts.append("== DELAY BUCKET PERFORMANCE ==")
    for bs in bucket_stats:
        if bs["n"] > 0:
            wr = f"{bs['win_rate']:.1f}%" if bs["win_rate"] is not None else "-"
            ar = f"{bs['avg_return']:+.4f}%" if bs["avg_return"] is not None else "-"
            body_parts.append(f"  {bs['bucket']:>8}: n={bs['n']:>2}, WR={wr}, avg={ar}, PF={bs['profit_factor']}")

    if best_bucket:
        body_parts.append(f"\nOptimal entry window: {best_bucket['bucket']} ({best_bucket['avg_return']:+.4f}%, WR={best_bucket['win_rate']:.1f}%)")

    body_parts.append("")
    body_parts.append("== DELIVERABLES ==")
    body_parts.append(f"ignition_events_{date_str}.json")
    body_parts.append(f"ignition_entry_delay_{date_str}.md")
    body_parts.append(f"ignition_edge_curve_{date_str}.json")
    body_parts.append(f"ignition_delay_curve_{date_str}.json")
    body_parts.append("")
    body_parts.append("Production remains frozen. All analysis is read-only research.")

    msg = {
        "id": f"msg_chatgpt_{date_str.replace('-','')}_008",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": "claude",
        "to": "chatgpt",
        "type": "action_result",
        "subject": f"Ignition Detection ({date_str}): median delay={median(delays):.0f}s, optimal window={best_bucket['bucket'] if best_bucket else '?'}",
        "body": "\n".join(body_parts),
        "references": [
            f"engine/output/ignition_events_{date_str}.json",
            f"engine/output/ignition_entry_delay_{date_str}.md",
            f"engine/output/ignition_edge_curve_{date_str}.json",
            f"engine/output/ignition_delay_curve_{date_str}.json",
        ],
    }

    comms["messages"].append(msg)
    with open(COMMS_PATH, "w") as f:
        json.dump(comms, f, indent=2)
    print(f"  Comms updated -> {COMMS_PATH.name}")


# ========================================================================
# MAIN
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description="Momentum Ignition Detection Study")
    parser.add_argument("--date", default="2026-03-03", help="Date to analyze")
    args = parser.parse_args()
    date_str = args.date

    print("=" * 70)
    print(f"MOMENTUM IGNITION DETECTION STUDY ({date_str})")
    print("SuperBot Research Engine - READ ONLY")
    print("=" * 70)

    # Load signals
    print("\nLoading signals...")
    with open(REPLAY_JSON) as f:
        replay = json.load(f)
    pass_signals = [s for s in replay["signals"] if s["v2_result"] == "PASS"]
    print(f"  v2-pass signals: {len(pass_signals)}")

    # Load edge outcomes
    print("Loading edge preservation outcomes...")
    with open(EDGE_JSON) as f:
        edge_data = json.load(f)
    edge_signals = edge_data["signals"]
    print(f"  Edge signals: {len(edge_signals)}")

    # Process each symbol
    symbols = sorted(set(s["symbol"] for s in pass_signals))
    print(f"\nProcessing {len(symbols)} symbols...")

    ignitions_by_symbol = {}
    all_thresholds = {}
    total_ignitions = 0

    for sym in symbols:
        print(f"\n  {sym}:")
        quotes = load_quotes_with_ffill(sym)
        if len(quotes) < 20:
            print(f"    Insufficient quotes ({len(quotes)}), skipping")
            ignitions_by_symbol[sym] = []
            continue

        print(f"    {len(quotes)} quotes loaded (ffilled)")

        # Compute indicators
        ticks = compute_indicators(quotes)
        print(f"    {len(ticks)} ticks with indicators")

        # Calibrate thresholds
        thresholds = calibrate_thresholds(ticks)
        all_thresholds[sym] = thresholds
        print(f"    Thresholds: vel3={thresholds['velocity_3s']:.3f}%, rate={thresholds['quote_rate_spike']:.2f}/s")

        # Detect ignitions
        ignitions = detect_ignitions(ticks, thresholds, MIN_CRITERIA_FOR_IGNITION)
        ignitions_by_symbol[sym] = ignitions
        total_ignitions += len(ignitions)
        print(f"    Ignitions detected: {len(ignitions)}")

        # Show a few examples
        for ig in ignitions[:3]:
            criteria_list = ", ".join(ig["criteria"].keys())
            print(f"      {ig['time']} ${ig['price']:.2f} vel={ig['velocity_3s']:+.3f}% [{criteria_list}]")

    print(f"\n  Total ignitions: {total_ignitions}")

    # Step 5: Measure entry delays
    print(f"\n{'='*70}")
    print("STEP 5: Measure Entry Delays")
    delay_results = measure_entry_delays(pass_signals, ignitions_by_symbol)
    matched = [d for d in delay_results if d["matched"]]
    unmatched = [d for d in delay_results if not d["matched"]]
    print(f"  Matched: {len(matched)}/{len(delay_results)}")
    print(f"  Unmatched: {len(unmatched)}")

    delays = [d["delay_seconds"] for d in matched if d["delay_seconds"] is not None]
    if delays:
        print(f"  Delay stats: mean={mean(delays):.1f}s, median={median(delays):.1f}s, p90={sorted(delays)[int(len(delays)*0.9)]:.1f}s")

    # Step 6: Profit correlation
    print(f"\n{'='*70}")
    print("STEP 6: Profit Correlation")
    merged_data = correlate_with_outcomes(delay_results, edge_signals)
    bucket_stats = compute_bucket_stats(merged_data)

    print("\n  Delay bucket performance:")
    for bs in bucket_stats:
        if bs["n"] > 0:
            wr = f"{bs['win_rate']:.1f}%" if bs["win_rate"] is not None else "-"
            ar = f"{bs['avg_return']:+.4f}%" if bs["avg_return"] is not None else "-"
            print(f"    {bs['bucket']:>8}: n={bs['n']:>2}, WR={wr}, avg={ar}, PF={bs['profit_factor']}")

    # Step 7: Delay curve
    print(f"\n{'='*70}")
    print("STEP 7: Delay Curve")
    delay_curve = compute_delay_curve(merged_data)

    # Generate reports
    print(f"\n{'='*70}")
    print("Generating reports...")
    outputs = generate_reports(date_str, ignitions_by_symbol, delay_results,
                               merged_data, bucket_stats, delay_curve, all_thresholds)
    for name, path in outputs.items():
        print(f"  -> {path.name}")

    # Update comms
    print(f"\n{'='*70}")
    print("Updating comms...")
    update_comms(date_str, delay_results, bucket_stats, delay_curve,
                 total_ignitions, all_thresholds)

    # Final summary
    print(f"\n{'='*70}")
    print("IGNITION DETECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Signals analyzed: {len(pass_signals)}")
    print(f"Ignition events: {total_ignitions}")
    print(f"Signals matched: {len(matched)}/{len(delay_results)}")
    if delays:
        print(f"Median ignition delay: {median(delays):.1f}s")
        valid_buckets = [b for b in bucket_stats if b["n"] >= 3 and b["avg_return"] is not None]
        if valid_buckets:
            best = max(valid_buckets, key=lambda b: b["avg_return"])
            print(f"Optimal entry window: {best['bucket']} (avg={best['avg_return']:+.4f}%, WR={best['win_rate']:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
