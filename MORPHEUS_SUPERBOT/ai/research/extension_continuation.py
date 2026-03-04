"""
Extension Continuation Study — 2026-03-03
==========================================
Measures whether momentum continues after ignition long enough
to support earlier entry than Morpheus's current ~180s delay.

Workflow:
  1. Load ignition events (520 across 11 symbols)
  2. Load quote caches with forward-fill
  3. For each ignition event, compute forward returns at 7 windows
  4. Calculate continuation probability per window
  5. Build momentum decay curve (5-second granularity)
  6. Identify optimal entry window
  7. Update comms

Usage:
  python -m ai.research.extension_continuation --date 2026-03-03
"""

import json
import os
import sys
import argparse
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE / "engine" / "output"
CACHE_DIR = BASE / "engine" / "cache" / "quotes"
COMMS_FILE = BASE / "comms" / "outbox_chatgpt.json"

# ── Windows ────────────────────────────────────────────────────────
WINDOWS = [15, 30, 60, 90, 120, 180, 300]
CURVE_STEP = 5  # seconds
CURVE_MAX = 305  # up to 300s inclusive


def load_ignition_events(date_str: str) -> dict:
    """Load ignition events JSON."""
    path = OUTPUT_DIR / f"ignition_events_{date_str}.json"
    with open(path) as f:
        return json.load(f)


def load_quotes_ffill(symbol: str) -> list[dict]:
    """Load quote cache with forward-filled bid/ask."""
    path = CACHE_DIR / f"{symbol}_quotes.json"
    with open(path) as f:
        data = json.load(f)

    quotes = data["quotes"]
    last_bid = None
    last_ask = None
    filled = []
    for q in quotes:
        bid = q.get("bid")
        ask = q.get("ask")
        if bid is not None:
            last_bid = bid
        if ask is not None:
            last_ask = ask
        filled.append({
            "epoch": q["epoch"],
            "bid": last_bid,
            "ask": last_ask,
            "last": q.get("last"),
            "mid": ((last_bid + last_ask) / 2) if last_bid and last_ask else q.get("last"),
        })
    return filled


def find_price_at_offset(quotes: list[dict], start_epoch: float, offset_s: float) -> float | None:
    """Find the mid price closest to start_epoch + offset_s.
    Uses the last quote at or before the target time.
    """
    target = start_epoch + offset_s
    best = None
    for q in quotes:
        if q["epoch"] > target + 1.0:  # small tolerance
            break
        if q["mid"] is not None:
            best = q
    return best["mid"] if best else None


def compute_mfe_mae(quotes: list[dict], start_idx: int, start_price: float,
                    direction: int, duration_s: float) -> tuple[float, float]:
    """Compute MFE and MAE within a time window.

    direction: +1 for upward ignition, -1 for downward
    MFE = max move in ignition direction (always positive %)
    MAE = max move against ignition direction (always positive %)
    """
    start_epoch = quotes[start_idx]["epoch"]
    end_epoch = start_epoch + duration_s
    mfe = 0.0
    mae = 0.0

    for i in range(start_idx + 1, len(quotes)):
        q = quotes[i]
        if q["epoch"] > end_epoch:
            break
        if q["mid"] is None:
            continue
        ret_pct = ((q["mid"] - start_price) / start_price) * 100.0
        directional_ret = ret_pct * direction  # positive = favorable

        if directional_ret > mfe:
            mfe = directional_ret
        if directional_ret < -mae:
            mae = -directional_ret  # MAE stored as positive

    return mfe, mae


def find_start_index(quotes: list[dict], epoch: float) -> int | None:
    """Binary search for the quote index closest to epoch."""
    lo, hi = 0, len(quotes) - 1
    best = None
    best_diff = float("inf")

    while lo <= hi:
        mid_idx = (lo + hi) // 2
        diff = abs(quotes[mid_idx]["epoch"] - epoch)
        if diff < best_diff:
            best_diff = diff
            best = mid_idx
        if quotes[mid_idx]["epoch"] < epoch:
            lo = mid_idx + 1
        else:
            hi = mid_idx - 1

    # Accept if within 5 seconds
    if best is not None and best_diff <= 5.0:
        return best
    return None


def step1_forward_windows(ignition_data: dict) -> list[dict]:
    """For each ignition event, compute forward returns at all windows."""
    print("\n=== STEP 1: Forward Window Analysis ===")

    all_results = []
    symbols_processed = 0

    for symbol, sym_data in ignition_data["events_per_symbol"].items():
        events = sym_data["events"]
        if not events:
            continue

        # Load quotes
        try:
            quotes = load_quotes_ffill(symbol)
        except FileNotFoundError:
            print(f"  {symbol}: no quote cache, skipping")
            continue

        symbols_processed += 1
        matched = 0

        for evt in events:
            ign_epoch = evt["epoch"]
            ign_price = evt["price"]

            # Determine direction from velocity
            vel_3s = evt.get("velocity_3s", 0)
            direction = 1 if vel_3s >= 0 else -1

            # Find starting quote index
            start_idx = find_start_index(quotes, ign_epoch)
            if start_idx is None:
                continue

            # Use mid price at ignition as reference
            start_price = quotes[start_idx]["mid"]
            if start_price is None or start_price <= 0:
                continue

            matched += 1
            result = {
                "symbol": symbol,
                "ignition_epoch": ign_epoch,
                "ignition_time": evt["time"],
                "ignition_price": ign_price,
                "start_mid": start_price,
                "direction": "up" if direction == 1 else "down",
                "velocity_3s": vel_3s,
                "criteria_met": evt["criteria_met"],
                "windows": {},
            }

            for w in WINDOWS:
                # Forward return
                future_price = find_price_at_offset(quotes, ign_epoch, w)
                if future_price is None:
                    continue

                raw_ret = ((future_price - start_price) / start_price) * 100.0
                dir_ret = raw_ret * direction  # positive = continuation

                # MFE/MAE
                mfe, mae = compute_mfe_mae(quotes, start_idx, start_price, direction, w)

                result["windows"][str(w)] = {
                    "return_pct": round(raw_ret, 4),
                    "directional_return_pct": round(dir_ret, 4),
                    "max_favorable_excursion": round(mfe, 4),
                    "max_adverse_excursion": round(mae, 4),
                }

            all_results.append(result)

        print(f"  {symbol}: {matched}/{len(events)} events matched to quotes")

    print(f"\n  Total: {len(all_results)} events with forward windows across {symbols_processed} symbols")
    return all_results


def step2_continuation_stats(results: list[dict], date_str: str) -> dict:
    """Calculate continuation probability for each window."""
    print("\n=== STEP 2: Continuation Probability ===")

    stats = {}
    for w in WINDOWS:
        ws = str(w)
        returns = []
        dir_returns = []
        mfes = []
        maes = []

        for r in results:
            if ws not in r["windows"]:
                continue
            wdata = r["windows"][ws]
            returns.append(wdata["return_pct"])
            dir_returns.append(wdata["directional_return_pct"])
            mfes.append(wdata["max_favorable_excursion"])
            maes.append(wdata["max_adverse_excursion"])

        if not dir_returns:
            continue

        n = len(dir_returns)
        up_count = sum(1 for d in dir_returns if d > 0)
        down_count = sum(1 for d in dir_returns if d < 0)
        flat_count = sum(1 for d in dir_returns if d == 0)

        avg_ret = statistics.mean(dir_returns)
        med_ret = statistics.median(dir_returns)
        avg_mfe = statistics.mean(mfes)
        avg_mae = statistics.mean(maes)

        # Win rate and profit factor
        wins = [d for d in dir_returns if d > 0]
        losses = [d for d in dir_returns if d < 0]
        wr = (len(wins) / n * 100) if n > 0 else 0
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (999.0 if wins else 0.0)

        stats[ws] = {
            "window_seconds": w,
            "n": n,
            "prob_continuation": round(up_count / n * 100, 1),
            "prob_reversal": round(down_count / n * 100, 1),
            "prob_flat": round(flat_count / n * 100, 1),
            "avg_directional_return": round(avg_ret, 4),
            "median_directional_return": round(med_ret, 4),
            "avg_mfe": round(avg_mfe, 4),
            "avg_mae": round(avg_mae, 4),
            "mfe_mae_ratio": round(avg_mfe / avg_mae, 3) if avg_mae > 0 else 999.0,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 3),
        }

        print(f"  {w:>4}s: n={n:>3}, cont={stats[ws]['prob_continuation']:>5.1f}%, "
              f"avg={avg_ret:>+.4f}%, MFE/MAE={stats[ws]['mfe_mae_ratio']:.2f}x, "
              f"WR={wr:.1f}%, PF={pf:.3f}")

    return stats


def step3_decay_curve(results: list[dict]) -> list[dict]:
    """Build momentum decay curve at 5-second granularity."""
    print("\n=== STEP 3: Momentum Decay Curve ===")

    # We need to recompute returns at fine granularity
    # Reuse the quote data
    # For each ignition, we need the quote stream — reload per symbol
    sym_quotes = {}
    for r in results:
        sym = r["symbol"]
        if sym not in sym_quotes:
            try:
                sym_quotes[sym] = load_quotes_ffill(sym)
            except FileNotFoundError:
                pass

    curve = []
    for t in range(CURVE_STEP, CURVE_MAX + 1, CURVE_STEP):
        dir_returns = []

        for r in results:
            sym = r["symbol"]
            if sym not in sym_quotes:
                continue

            quotes = sym_quotes[sym]
            ign_epoch = r["ignition_epoch"]
            start_price = r["start_mid"]
            direction = 1 if r["direction"] == "up" else -1

            if start_price is None or start_price <= 0:
                continue

            future_price = find_price_at_offset(quotes, ign_epoch, t)
            if future_price is None:
                continue

            raw_ret = ((future_price - start_price) / start_price) * 100.0
            dir_ret = raw_ret * direction
            dir_returns.append(dir_ret)

        if not dir_returns:
            curve.append({
                "time_since_ignition": t,
                "n": 0,
                "avg_return": None,
                "win_rate": None,
                "profit_factor": None,
            })
            continue

        n = len(dir_returns)
        avg_ret = statistics.mean(dir_returns)
        wins = [d for d in dir_returns if d > 0]
        losses = [d for d in dir_returns if d < 0]
        wr = len(wins) / n * 100 if n > 0 else 0
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (999.0 if wins else 0.0)

        curve.append({
            "time_since_ignition": t,
            "n": n,
            "avg_return": round(avg_ret, 4),
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 3),
        })

    # Print highlights
    print(f"\n  Decay curve ({CURVE_STEP}s steps, {len(curve)} points):")
    for pt in curve:
        t = pt["time_since_ignition"]
        if t in [5, 10, 15, 30, 60, 90, 120, 180, 300]:
            if pt["n"] > 0:
                print(f"    {t:>4}s: n={pt['n']:>3}, avg={pt['avg_return']:>+.4f}%, "
                      f"WR={pt['win_rate']:.1f}%, PF={pt['profit_factor']:.3f}")
            else:
                print(f"    {t:>4}s: n=0")

    return curve


def step4_optimal_window(stats: dict, curve: list[dict]) -> dict:
    """Identify the optimal entry window."""
    print("\n=== STEP 4: Optimal Entry Window ===")

    # Score each window: combine PF, avg_return, win_rate
    best_score = -999
    best_window = None

    for ws, s in stats.items():
        if s["n"] < 10:  # need reasonable sample
            continue
        # Composite score: normalize each metric
        # PF weight=0.4, avg_return weight=0.3, WR weight=0.3
        pf_norm = min(s["profit_factor"], 3.0) / 3.0  # cap at 3
        ret_norm = (s["avg_directional_return"] + 1.0) / 2.0  # shift to 0-1 range
        wr_norm = s["win_rate"] / 100.0
        score = 0.4 * pf_norm + 0.3 * ret_norm + 0.3 * wr_norm
        print(f"  {s['window_seconds']:>4}s: PF={s['profit_factor']:.3f}, "
              f"avg={s['avg_directional_return']:>+.4f}%, WR={s['win_rate']:.1f}%, "
              f"score={score:.4f}")
        if score > best_score:
            best_score = score
            best_window = s

    # Also find peak from the fine-grained curve
    peak_curve_point = None
    peak_pf = 0
    for pt in curve:
        if pt["n"] and pt["n"] >= 20 and pt["profit_factor"] is not None:
            if pt["profit_factor"] > peak_pf:
                peak_pf = pt["profit_factor"]
                peak_curve_point = pt

    # Find the range where PF stays above 1.0
    above_1_start = None
    above_1_end = None
    for pt in curve:
        if pt["n"] and pt["n"] >= 20 and pt["profit_factor"] is not None:
            if pt["profit_factor"] >= 1.0:
                if above_1_start is None:
                    above_1_start = pt["time_since_ignition"]
                above_1_end = pt["time_since_ignition"]

    # Find the range where WR stays above 50%
    above_50wr_start = None
    above_50wr_end = None
    for pt in curve:
        if pt["n"] and pt["n"] >= 20 and pt["win_rate"] is not None:
            if pt["win_rate"] >= 50.0:
                if above_50wr_start is None:
                    above_50wr_start = pt["time_since_ignition"]
                above_50wr_end = pt["time_since_ignition"]

    result = {
        "best_window_seconds": best_window["window_seconds"] if best_window else None,
        "best_window_stats": best_window,
        "best_composite_score": round(best_score, 4),
        "peak_curve_time": peak_curve_point["time_since_ignition"] if peak_curve_point else None,
        "peak_curve_pf": round(peak_pf, 3) if peak_curve_point else None,
        "pf_above_1_range": {
            "start": above_1_start,
            "end": above_1_end,
        } if above_1_start else None,
        "wr_above_50_range": {
            "start": above_50wr_start,
            "end": above_50wr_end,
        } if above_50wr_start else None,
        "recommended_entry_start": None,
        "recommended_entry_end": None,
        "recommended_offset_vs_current": None,
    }

    # Determine recommended window
    if above_1_start is not None:
        result["recommended_entry_start"] = above_1_start
        result["recommended_entry_end"] = above_1_end
        offset_vs_180 = 180 - above_1_start
        result["recommended_offset_vs_current"] = f"-{offset_vs_180}s vs current 180s median"
    elif best_window:
        result["recommended_entry_start"] = max(15, best_window["window_seconds"] - 30)
        result["recommended_entry_end"] = best_window["window_seconds"]

    print(f"\n  Best window: {result['best_window_seconds']}s (score={best_score:.4f})")
    if peak_curve_point:
        print(f"  Peak curve PF: {peak_pf:.3f} at {peak_curve_point['time_since_ignition']}s")
    if above_1_start:
        print(f"  PF >= 1.0 range: {above_1_start}s - {above_1_end}s")
    if above_50wr_start:
        print(f"  WR >= 50% range: {above_50wr_start}s - {above_50wr_end}s")

    return result


def generate_report(date_str: str, stats: dict, curve: list[dict],
                    optimal: dict, results: list[dict]) -> str:
    """Generate the continuation stats markdown report."""

    # Per-symbol aggregation
    sym_stats = defaultdict(lambda: {"n": 0, "dir_returns": [], "mfes": [], "maes": []})
    for r in results:
        sym = r["symbol"]
        # Use 60s window for symbol comparison
        if "60" in r["windows"]:
            w60 = r["windows"]["60"]
            sym_stats[sym]["n"] += 1
            sym_stats[sym]["dir_returns"].append(w60["directional_return_pct"])
            sym_stats[sym]["mfes"].append(w60["max_favorable_excursion"])
            sym_stats[sym]["maes"].append(w60["max_adverse_excursion"])

    lines = []
    lines.append(f"# Extension Continuation Study — {date_str}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append("**Method:** Forward-window analysis from ignition events")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview
    total_events = len(results)
    up_events = sum(1 for r in results if r["direction"] == "up")
    down_events = total_events - up_events
    n_symbols = len(set(r["symbol"] for r in results))

    lines.append("## Study Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Ignition events analyzed | {total_events} |")
    lines.append(f"| Symbols | {n_symbols} |")
    lines.append(f"| Upward ignitions | {up_events} ({up_events/total_events*100:.1f}%) |")
    lines.append(f"| Downward ignitions | {down_events} ({down_events/total_events*100:.1f}%) |")
    lines.append(f"| Current Morpheus median entry | 180s |")
    if optimal.get("recommended_entry_start"):
        lines.append(f"| Recommended entry start | {optimal['recommended_entry_start']}s |")
        lines.append(f"| Recommended entry end | {optimal['recommended_entry_end']}s |")
    lines.append("")

    # Continuation probability table
    lines.append("## Continuation Probability by Window")
    lines.append("")
    lines.append("| Window | N | P(Continue) | P(Reverse) | Avg Dir Ret | Med Dir Ret | Avg MFE | Avg MAE | MFE/MAE | WR | PF |")
    lines.append("|--------|---|-------------|------------|-------------|-------------|---------|---------|---------|-----|-----|")

    for w in WINDOWS:
        ws = str(w)
        if ws not in stats:
            continue
        s = stats[ws]
        lines.append(
            f"| {w}s | {s['n']} | {s['prob_continuation']:.1f}% | {s['prob_reversal']:.1f}% | "
            f"{s['avg_directional_return']:+.4f}% | {s['median_directional_return']:+.4f}% | "
            f"{s['avg_mfe']:.4f}% | {s['avg_mae']:.4f}% | {s['mfe_mae_ratio']:.2f}x | "
            f"{s['win_rate']:.1f}% | {s['profit_factor']:.3f} |"
        )

    lines.append("")

    # Momentum decay curve (ASCII)
    lines.append("## Momentum Decay Curve")
    lines.append("")
    lines.append("```")
    lines.append("Time   Avg Dir Ret  WR      PF     N    Visual")

    for pt in curve:
        t = pt["time_since_ignition"]
        if pt["n"] == 0 or pt["avg_return"] is None:
            continue
        # Only show key intervals to keep readable
        if t <= 30 or t % 15 == 0 or t in [5, 10, 45, 75, 105]:
            ret = pt["avg_return"]
            wr = pt["win_rate"]
            pf = pt["profit_factor"]
            n = pt["n"]

            # ASCII bar
            bar_width = int(abs(ret) * 80)
            bar_width = min(bar_width, 40)
            if ret >= 0:
                bar = " " * 20 + "|" + "#" * bar_width
            else:
                padding = 20 - bar_width
                bar = " " * max(0, padding) + "#" * bar_width + "|"

            lines.append(f"  {t:>4}s  {ret:>+.4f}%  {wr:>5.1f}%  {pf:>6.3f}  {n:>3}  {bar}")

    lines.append("```")
    lines.append("")

    # Per-symbol continuation at 60s
    lines.append("## Per-Symbol Continuation (60s window)")
    lines.append("")
    lines.append("| Symbol | N | Avg Dir Ret | WR | Avg MFE | Avg MAE | MFE/MAE |")
    lines.append("|--------|---|-------------|-----|---------|---------|---------|")

    for sym in sorted(sym_stats.keys()):
        ss = sym_stats[sym]
        if ss["n"] == 0:
            continue
        avg_ret = statistics.mean(ss["dir_returns"])
        wr = sum(1 for d in ss["dir_returns"] if d > 0) / ss["n"] * 100
        avg_mfe = statistics.mean(ss["mfes"])
        avg_mae = statistics.mean(ss["maes"])
        ratio = avg_mfe / avg_mae if avg_mae > 0 else 999.0
        lines.append(f"| {sym} | {ss['n']} | {avg_ret:+.4f}% | {wr:.1f}% | "
                     f"{avg_mfe:.4f}% | {avg_mae:.4f}% | {ratio:.2f}x |")

    lines.append("")

    # Optimal entry window
    lines.append("## Optimal Entry Window")
    lines.append("")
    if optimal.get("recommended_entry_start"):
        lines.append(f"- **Recommended entry start:** {optimal['recommended_entry_start']}s after ignition")
        lines.append(f"- **Recommended entry end:** {optimal['recommended_entry_end']}s after ignition")
    if optimal.get("recommended_offset_vs_current"):
        lines.append(f"- **Offset vs current:** {optimal['recommended_offset_vs_current']}")
    if optimal.get("peak_curve_time"):
        lines.append(f"- **Peak PF on decay curve:** {optimal['peak_curve_pf']:.3f} at {optimal['peak_curve_time']}s")
    if optimal.get("pf_above_1_range"):
        r = optimal["pf_above_1_range"]
        lines.append(f"- **PF >= 1.0 range:** {r['start']}s to {r['end']}s")
    if optimal.get("wr_above_50_range"):
        r = optimal["wr_above_50_range"]
        lines.append(f"- **WR >= 50% range:** {r['start']}s to {r['end']}s")
    lines.append(f"- **Best window composite score:** {optimal['best_composite_score']:.4f}")
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Find the window with highest continuation prob
    best_cont = max(stats.values(), key=lambda s: s["prob_continuation"]) if stats else None
    if best_cont:
        lines.append(f"- **Highest continuation probability:** {best_cont['prob_continuation']:.1f}% "
                     f"at {best_cont['window_seconds']}s (n={best_cont['n']})")

    # Check if 60-120s is indeed the sweet spot
    if "60" in stats and "120" in stats:
        s60 = stats["60"]
        s120 = stats["120"]
        lines.append(f"- **60s window:** {s60['prob_continuation']:.1f}% continuation, "
                     f"PF={s60['profit_factor']:.3f}, WR={s60['win_rate']:.1f}%")
        lines.append(f"- **120s window:** {s120['prob_continuation']:.1f}% continuation, "
                     f"PF={s120['profit_factor']:.3f}, WR={s120['win_rate']:.1f}%")

    if "180" in stats:
        s180 = stats["180"]
        lines.append(f"- **180s window (current Morpheus entry):** {s180['prob_continuation']:.1f}% continuation, "
                     f"PF={s180['profit_factor']:.3f}, WR={s180['win_rate']:.1f}%")

    if "300" in stats:
        s300 = stats["300"]
        lines.append(f"- **300s window:** {s300['prob_continuation']:.1f}% continuation, "
                     f"PF={s300['profit_factor']:.3f}, WR={s300['win_rate']:.1f}%")

    # Summarize edge decay
    if "15" in stats and "300" in stats:
        early = stats["15"]["avg_directional_return"]
        late = stats["300"]["avg_directional_return"]
        decay = early - late
        lines.append(f"- **Edge decay (15s vs 300s):** {early:+.4f}% -> {late:+.4f}% "
                     f"(decay = {decay:+.4f}%)")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This study is research-only. No production changes applied.*")

    return "\n".join(lines)


def step5_comms(date_str: str, stats: dict, optimal: dict):
    """Append supervisor message to comms outbox."""
    print("\n=== STEP 5: Comms Update ===")

    with open(COMMS_FILE) as f:
        comms = json.load(f)

    # Build summary
    summary_parts = []
    for w in WINDOWS:
        ws = str(w)
        if ws in stats:
            s = stats[ws]
            summary_parts.append(f"{w}s: cont={s['prob_continuation']:.1f}%, "
                                 f"WR={s['win_rate']:.1f}%, PF={s['profit_factor']:.3f}")

    cont_summary = "\n".join(summary_parts)

    rec_start = optimal.get("recommended_entry_start", "N/A")
    rec_end = optimal.get("recommended_entry_end", "N/A")
    rec_offset = optimal.get("recommended_offset_vs_current", "N/A")

    body = (
        f"EXTENSION CONTINUATION STUDY COMPLETE\n\n"
        f"== CONTINUATION PROBABILITY BY WINDOW ==\n"
        f"{cont_summary}\n\n"
        f"== OPTIMAL ENTRY WINDOW ==\n"
        f"Recommended entry: {rec_start}s to {rec_end}s after ignition\n"
        f"Offset vs current: {rec_offset}\n"
        f"Best composite score: {optimal.get('best_composite_score', 'N/A')}\n"
    )

    if optimal.get("peak_curve_time"):
        body += f"Peak PF on decay curve: {optimal['peak_curve_pf']:.3f} at {optimal['peak_curve_time']}s\n"
    if optimal.get("pf_above_1_range"):
        r = optimal["pf_above_1_range"]
        body += f"PF >= 1.0 range: {r['start']}s to {r['end']}s\n"
    if optimal.get("wr_above_50_range"):
        r = optimal["wr_above_50_range"]
        body += f"WR >= 50% range: {r['start']}s to {r['end']}s\n"

    body += (
        f"\n== FILES ==\n"
        f"Forward windows: engine/output/ignition_forward_windows_{date_str}.json\n"
        f"Continuation stats: engine/output/ignition_continuation_stats_{date_str}.md\n"
        f"Decay curve: engine/output/momentum_decay_curve_{date_str}.json\n"
        f"Optimal window: engine/output/optimal_entry_window_{date_str}.md\n\n"
        f"Production remains frozen. All analysis is read-only research."
    )

    msg = {
        "id": f"msg_chatgpt_{date_str.replace('-', '')}_009",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": "claude",
        "to": "chatgpt",
        "type": "research_result",
        "subject": f"Extension Continuation Study ({date_str}): momentum continuation after ignition",
        "body": body,
        "references": [
            f"engine/output/ignition_forward_windows_{date_str}.json",
            f"engine/output/ignition_continuation_stats_{date_str}.md",
            f"engine/output/momentum_decay_curve_{date_str}.json",
            f"engine/output/optimal_entry_window_{date_str}.md",
        ],
    }

    comms["messages"].append(msg)
    with open(COMMS_FILE, "w", encoding="utf-8") as f:
        json.dump(comms, f, indent=2)

    print(f"  Appended msg_009 to comms outbox")


def main():
    parser = argparse.ArgumentParser(description="Extension Continuation Study")
    parser.add_argument("--date", required=True, help="Date string YYYY-MM-DD")
    args = parser.parse_args()
    date_str = args.date

    print(f"Extension Continuation Study — {date_str}")
    print("=" * 60)

    # Load ignition events
    ignition_data = load_ignition_events(date_str)
    total = ignition_data.get("total_events", 0)
    print(f"\nLoaded {total} ignition events across "
          f"{len(ignition_data['events_per_symbol'])} symbols")

    # Step 1: Forward windows
    results = step1_forward_windows(ignition_data)

    if not results:
        print("\nERROR: No ignition events matched to quote data.")
        sys.exit(1)

    # Save forward windows
    fw_path = OUTPUT_DIR / f"ignition_forward_windows_{date_str}.json"
    with open(fw_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "ignition_forward_windows",
            "total_events": len(results),
            "windows": [str(w) for w in WINDOWS],
            "events": results,
        }, f, indent=2)
    print(f"\n  Saved: {fw_path}")

    # Step 2: Continuation stats
    stats = step2_continuation_stats(results, date_str)

    # Step 3: Decay curve
    curve = step3_decay_curve(results)

    # Save decay curve
    dc_path = OUTPUT_DIR / f"momentum_decay_curve_{date_str}.json"
    with open(dc_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "momentum_decay_curve",
            "step_seconds": CURVE_STEP,
            "curve": curve,
        }, f, indent=2)
    print(f"\n  Saved: {dc_path}")

    # Step 4: Optimal entry window
    optimal = step4_optimal_window(stats, curve)

    # Save optimal window report
    ow_path = OUTPUT_DIR / f"optimal_entry_window_{date_str}.md"
    # Generate the optimal window MD
    ow_lines = []
    ow_lines.append(f"# Optimal Entry Window — {date_str}")
    ow_lines.append("")
    ow_lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    ow_lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    ow_lines.append("")
    ow_lines.append("---")
    ow_lines.append("")
    ow_lines.append("## Recommendation")
    ow_lines.append("")
    if optimal.get("recommended_entry_start"):
        ow_lines.append(f"| Parameter | Value |")
        ow_lines.append(f"|-----------|-------|")
        ow_lines.append(f"| Optimal entry start | {optimal['recommended_entry_start']}s after ignition |")
        ow_lines.append(f"| Optimal entry end | {optimal['recommended_entry_end']}s after ignition |")
        ow_lines.append(f"| Current Morpheus median entry | 180s after ignition |")
        ow_lines.append(f"| Recommended offset | {optimal.get('recommended_offset_vs_current', 'N/A')} |")
        ow_lines.append(f"| Composite score | {optimal['best_composite_score']:.4f} |")
    ow_lines.append("")

    ow_lines.append("## Window Comparison")
    ow_lines.append("")
    ow_lines.append("| Window | P(Continue) | Avg Dir Ret | WR | PF | Score |")
    ow_lines.append("|--------|-------------|-------------|-----|-----|-------|")
    for w in WINDOWS:
        ws = str(w)
        if ws not in stats:
            continue
        s = stats[ws]
        # Recalculate score
        if s["n"] >= 10:
            pf_n = min(s["profit_factor"], 3.0) / 3.0
            ret_n = (s["avg_directional_return"] + 1.0) / 2.0
            wr_n = s["win_rate"] / 100.0
            score = 0.4 * pf_n + 0.3 * ret_n + 0.3 * wr_n
        else:
            score = 0
        marker = " **<-- OPTIMAL**" if optimal.get("best_window_seconds") == w else ""
        ow_lines.append(f"| {w}s | {s['prob_continuation']:.1f}% | "
                        f"{s['avg_directional_return']:+.4f}% | {s['win_rate']:.1f}% | "
                        f"{s['profit_factor']:.3f} | {score:.4f}{marker} |")

    ow_lines.append("")

    if optimal.get("pf_above_1_range"):
        r = optimal["pf_above_1_range"]
        ow_lines.append(f"## Profitable Zone")
        ow_lines.append("")
        ow_lines.append(f"Momentum continuation produces positive edge (PF >= 1.0) between "
                        f"**{r['start']}s** and **{r['end']}s** after ignition.")
        ow_lines.append("")
        ow_lines.append(f"Morpheus currently enters at ~180s median, which is "
                        f"{'within' if r['start'] <= 180 <= r['end'] else 'outside'} this zone.")
        ow_lines.append("")

    ow_lines.append("## Conclusion")
    ow_lines.append("")
    if optimal.get("recommended_entry_start") and optimal["recommended_entry_start"] < 180:
        gap = 180 - optimal["recommended_entry_start"]
        ow_lines.append(f"Moving entry {gap}s earlier (from 180s to {optimal['recommended_entry_start']}s "
                        f"after ignition) would capture more momentum continuation edge.")
    else:
        ow_lines.append("Current entry timing may already be near optimal based on this data.")
    ow_lines.append("")
    ow_lines.append("---")
    ow_lines.append("")
    ow_lines.append("*This study is research-only. No production changes applied.*")

    with open(ow_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ow_lines))
    print(f"\n  Saved: {ow_path}")

    # Generate continuation stats report
    report = generate_report(date_str, stats, curve, optimal, results)
    cs_path = OUTPUT_DIR / f"ignition_continuation_stats_{date_str}.md"
    with open(cs_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {cs_path}")

    # Step 5: Comms
    step5_comms(date_str, stats, optimal)

    # Final summary
    print("\n" + "=" * 60)
    print("EXTENSION CONTINUATION STUDY COMPLETE")
    print("=" * 60)
    print(f"\nFiles generated:")
    print(f"  1. {fw_path}")
    print(f"  2. {cs_path}")
    print(f"  3. {dc_path}")
    print(f"  4. {ow_path}")
    print(f"  5. comms/outbox_chatgpt.json (msg_009)")


if __name__ == "__main__":
    main()
