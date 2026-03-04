#!/usr/bin/env python3
"""
Momentum Edge Diagnostics Study (2026-03-03)
SuperBot Research Engine - READ ONLY, no production changes.

Three studies on 74 v2-pass signals:
  1. Entry Timing Offset - are entries too late?
  2. Hold Time Sweep - optimal holding period
  3. Symbol Expectancy Ranking - which symbols carry edge?
"""

import json
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

# === PATHS ===
SUPERBOT_ROOT = Path("C:/AI_Bot_Research/MORPHEUS_SUPERBOT")
REPLAY_JSON = SUPERBOT_ROOT / "engine" / "output" / "containment_v2_replay_2026-03-03.json"
QUOTES_CACHE_DIR = SUPERBOT_ROOT / "engine" / "cache" / "quotes"
OUTPUT_DIR = SUPERBOT_ROOT / "engine" / "output"

# === EXIT MODEL (baseline) ===
STOP_PCT = -1.0
TRAIL_ACTIVATE = 0.8
TRAIL_DISTANCE = 0.4
BASELINE_EXIT_SEC = 300

# === STUDY 1: ENTRY OFFSET LEVELS ===
ENTRY_OFFSETS = [-0.50, -0.40, -0.30, -0.20, -0.10, 0.00, +0.10]

# === STUDY 2: HOLD TIME LEVELS ===
HOLD_TIMES = [30, 45, 60, 90, 120, 180, 300]

# === EDGE CLASSIFICATION THRESHOLDS ===
STRONG_EDGE_THRESHOLD = 0.10   # avg_return > +0.10%
NEGATIVE_EDGE_THRESHOLD = -0.10  # avg_return < -0.10%


def parse_signal_epoch(ts_str):
    """HH:MM:SS -> epoch for 2026-03-03 UTC."""
    h, m, s = ts_str.split(":")
    dt = datetime(2026, 3, 3, int(h), int(m), int(s), tzinfo=timezone.utc)
    return dt.timestamp()


def load_quotes(symbol):
    """Load cached quotes for a symbol. Returns list of {epoch, price, bid, ask}."""
    cache_file = QUOTES_CACHE_DIR / f"{symbol}_quotes.json"
    if not cache_file.exists():
        return []
    with open(cache_file, "r") as f:
        data = json.load(f)
    result = []
    for q in data.get("quotes", []):
        price = q.get("last")
        if price is None or price <= 0:
            bid = q.get("bid", 0) or 0
            ask = q.get("ask", 0) or 0
            if bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue
        result.append({
            "epoch": q["epoch"],
            "price": price,
            "bid": q.get("bid"),
            "ask": q.get("ask"),
        })
    return result


def get_forward_prices(quotes, start_epoch, window_sec):
    """Get all price ticks in [start_epoch, start_epoch + window_sec]."""
    end_epoch = start_epoch + window_sec
    return [q for q in quotes if q["epoch"] > start_epoch and q["epoch"] <= end_epoch]


def simulate_exit(forward_quotes, entry_price, stop_pct, trail_act, trail_dist, max_sec, start_epoch):
    """
    Simulate exit model on forward quotes.
    Returns dict with exit details.
    """
    if not forward_quotes:
        return {"exit_type": "NO_DATA", "exit_return_pct": None}

    trail_activated = False
    trail_high = entry_price
    peak_return = 0.0

    for q in forward_quotes:
        price = q["price"]
        ret_pct = (price - entry_price) / entry_price * 100
        elapsed = q["epoch"] - start_epoch

        if ret_pct > peak_return:
            peak_return = ret_pct

        # Hard stop
        if ret_pct <= stop_pct:
            return {
                "exit_type": "STOP",
                "exit_return_pct": round(ret_pct, 4),
                "exit_time_sec": round(elapsed, 1),
                "peak_return_pct": round(peak_return, 4),
            }

        # Trail activation
        if not trail_activated and ret_pct >= trail_act:
            trail_activated = True
            trail_high = price

        if trail_activated:
            if price > trail_high:
                trail_high = price
            trail_stop_price = trail_high * (1 - trail_dist / 100)
            if price <= trail_stop_price:
                return {
                    "exit_type": "TRAIL_STOP",
                    "exit_return_pct": round(ret_pct, 4),
                    "exit_time_sec": round(elapsed, 1),
                    "peak_return_pct": round(peak_return, 4),
                }

    # Time exit
    last = forward_quotes[-1]
    final_ret = (last["price"] - entry_price) / entry_price * 100
    return {
        "exit_type": "TIME_EXIT",
        "exit_return_pct": round(final_ret, 4),
        "exit_time_sec": round(last["epoch"] - start_epoch, 1),
        "peak_return_pct": round(peak_return, 4),
    }


def compute_mfe_mae(forward_quotes, entry_price):
    """Compute MFE and MAE from forward quotes."""
    if not forward_quotes:
        return None, None
    returns = [(q["price"] - entry_price) / entry_price * 100 for q in forward_quotes]
    return round(max(returns), 4), round(min(returns), 4)


def compute_expectancy(returns):
    """Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)."""
    if not returns:
        return None
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    return round(win_rate * avg_win + loss_rate * avg_loss, 4)


def compute_profit_factor(returns):
    """Profit factor = gross_wins / abs(gross_losses)."""
    gross_wins = sum(r for r in returns if r > 0)
    gross_losses = abs(sum(r for r in returns if r <= 0))
    if gross_losses == 0:
        return float("inf") if gross_wins > 0 else 0
    return round(gross_wins / gross_losses, 3)


# ========================================================================
# STUDY 1: Entry Timing Offset
# ========================================================================
def study1_entry_offset(pass_signals, symbol_quotes):
    """
    For each offset, shift entry price by offset%, re-run exit model.
    An offset of -0.30% means entering 0.30% cheaper (earlier/better fill).
    """
    print("\n=== STUDY 1: Entry Timing Offset ===")

    results = []

    for offset in ENTRY_OFFSETS:
        returns = []
        mfes = []
        maes = []

        for sig in pass_signals:
            quotes = symbol_quotes.get(sig["symbol"], [])
            if not quotes:
                continue

            base_entry = sig["entry_price"]
            # Offset entry: negative = better (lower) entry for long
            adjusted_entry = base_entry * (1 + offset / 100)
            signal_epoch = parse_signal_epoch(sig["timestamp"])

            fwd = get_forward_prices(quotes, signal_epoch, BASELINE_EXIT_SEC)
            if not fwd:
                continue

            exit_result = simulate_exit(fwd, adjusted_entry, STOP_PCT, TRAIL_ACTIVATE, TRAIL_DISTANCE, BASELINE_EXIT_SEC, signal_epoch)
            if exit_result["exit_return_pct"] is not None:
                returns.append(exit_result["exit_return_pct"])

            mfe, mae = compute_mfe_mae(fwd, adjusted_entry)
            if mfe is not None:
                mfes.append(mfe)
                maes.append(mae)

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        row = {
            "offset_pct": offset,
            "n": len(returns),
            "win_rate": round(len(wins) / len(returns) * 100, 1) if returns else None,
            "avg_return": round(sum(returns) / len(returns), 4) if returns else None,
            "median_return": round(sorted(returns)[len(returns) // 2], 4) if returns else None,
            "avg_mfe": round(sum(mfes) / len(mfes), 4) if mfes else None,
            "avg_mae": round(sum(maes) / len(maes), 4) if maes else None,
            "expectancy": compute_expectancy(returns),
            "profit_factor": compute_profit_factor(returns),
            "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
        }
        results.append(row)
        wr = row["win_rate"] if row["win_rate"] is not None else 0
        ar = row["avg_return"] if row["avg_return"] is not None else 0
        print(f"  offset={offset:+.2f}%: WR={wr:.1f}%, avg={ar:+.4f}%, expectancy={row['expectancy']}, PF={row['profit_factor']}")

    # Find best offset
    valid = [r for r in results if r["expectancy"] is not None]
    best = max(valid, key=lambda r: r["expectancy"]) if valid else None
    current = next((r for r in results if r["offset_pct"] == 0.0), None)

    improvement = None
    if best and current and best["avg_return"] is not None and current["avg_return"] is not None:
        improvement = round(best["avg_return"] - current["avg_return"], 4)

    return {
        "offsets": results,
        "best_offset": best["offset_pct"] if best else None,
        "best_expectancy": best["expectancy"] if best else None,
        "best_avg_return": best["avg_return"] if best else None,
        "current_avg_return": current["avg_return"] if current else None,
        "improvement_vs_current": improvement,
    }


# ========================================================================
# STUDY 2: Hold Time Sweep
# ========================================================================
def study2_hold_time_sweep(pass_signals, symbol_quotes):
    """
    For each hold time, simulate fixed-time exit (no trail, just stop + time).
    Also run with the trail model for comparison.
    """
    print("\n=== STUDY 2: Hold Time Sweep ===")

    results = []

    for hold_sec in HOLD_TIMES:
        # Fixed exit (stop + time only, no trailing)
        fixed_returns = []
        fixed_mfes = []
        fixed_maes = []

        # With trail model
        trail_returns = []

        for sig in pass_signals:
            quotes = symbol_quotes.get(sig["symbol"], [])
            if not quotes:
                continue

            entry = sig["entry_price"]
            signal_epoch = parse_signal_epoch(sig["timestamp"])
            fwd = get_forward_prices(quotes, signal_epoch, hold_sec)
            if not fwd:
                continue

            # Fixed exit: stop at -1.0%, otherwise exit at hold_sec close
            fixed_exit = simulate_exit(fwd, entry, STOP_PCT, 999.0, 0.0, hold_sec, signal_epoch)
            if fixed_exit["exit_return_pct"] is not None:
                fixed_returns.append(fixed_exit["exit_return_pct"])

            mfe, mae = compute_mfe_mae(fwd, entry)
            if mfe is not None:
                fixed_mfes.append(mfe)
                fixed_maes.append(mae)

            # Trail model exit
            trail_exit = simulate_exit(fwd, entry, STOP_PCT, TRAIL_ACTIVATE, TRAIL_DISTANCE, hold_sec, signal_epoch)
            if trail_exit["exit_return_pct"] is not None:
                trail_returns.append(trail_exit["exit_return_pct"])

        fixed_wins = [r for r in fixed_returns if r > 0]

        row = {
            "hold_sec": hold_sec,
            "n": len(fixed_returns),
            "win_rate": round(len(fixed_wins) / len(fixed_returns) * 100, 1) if fixed_returns else None,
            "avg_return": round(sum(fixed_returns) / len(fixed_returns), 4) if fixed_returns else None,
            "median_return": round(sorted(fixed_returns)[len(fixed_returns) // 2], 4) if fixed_returns else None,
            "profit_factor": compute_profit_factor(fixed_returns),
            "avg_mfe": round(sum(fixed_mfes) / len(fixed_mfes), 4) if fixed_mfes else None,
            "avg_mae": round(sum(fixed_maes) / len(fixed_maes), 4) if fixed_maes else None,
            "expectancy": compute_expectancy(fixed_returns),
            "trail_model_avg_return": round(sum(trail_returns) / len(trail_returns), 4) if trail_returns else None,
            "trail_model_win_rate": round(len([r for r in trail_returns if r > 0]) / len(trail_returns) * 100, 1) if trail_returns else None,
        }
        results.append(row)
        wr = row["win_rate"] if row["win_rate"] is not None else 0
        ar = row["avg_return"] if row["avg_return"] is not None else 0
        print(f"  hold={hold_sec:>3}s: WR={wr:.1f}%, avg={ar:+.4f}%, PF={row['profit_factor']}, MFE={row['avg_mfe']}, MAE={row['avg_mae']}")

    # Find optimal hold time
    valid = [r for r in results if r["expectancy"] is not None]
    best = max(valid, key=lambda r: r["expectancy"]) if valid else None

    # Build return decay curve
    decay_curve = []
    for r in results:
        decay_curve.append({
            "hold_sec": r["hold_sec"],
            "avg_return": r["avg_return"],
            "win_rate": r["win_rate"],
        })

    return {
        "hold_times": results,
        "optimal_hold_sec": best["hold_sec"] if best else None,
        "optimal_expectancy": best["expectancy"] if best else None,
        "optimal_avg_return": best["avg_return"] if best else None,
        "return_decay_curve": decay_curve,
    }


# ========================================================================
# STUDY 3: Symbol Expectancy Ranking
# ========================================================================
def study3_symbol_expectancy(pass_signals, symbol_quotes):
    """
    Per-symbol metrics with edge classification.
    """
    print("\n=== STUDY 3: Symbol Expectancy Ranking ===")

    # Group signals by symbol
    by_symbol = defaultdict(list)
    for sig in pass_signals:
        by_symbol[sig["symbol"]].append(sig)

    results = []

    for sym in sorted(by_symbol.keys()):
        sigs = by_symbol[sym]
        quotes = symbol_quotes.get(sym, [])
        if not quotes:
            continue

        returns = []
        mfes = []
        maes = []

        for sig in sigs:
            entry = sig["entry_price"]
            signal_epoch = parse_signal_epoch(sig["timestamp"])
            fwd = get_forward_prices(quotes, signal_epoch, BASELINE_EXIT_SEC)
            if not fwd:
                continue

            exit_result = simulate_exit(fwd, entry, STOP_PCT, TRAIL_ACTIVATE, TRAIL_DISTANCE, BASELINE_EXIT_SEC, signal_epoch)
            if exit_result["exit_return_pct"] is not None:
                returns.append(exit_result["exit_return_pct"])

            mfe, mae = compute_mfe_mae(fwd, entry)
            if mfe is not None:
                mfes.append(mfe)
                maes.append(mae)

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        avg_ret = sum(returns) / len(returns) if returns else 0

        # Classification
        if avg_ret > STRONG_EDGE_THRESHOLD:
            classification = "STRONG_EDGE"
        elif avg_ret < NEGATIVE_EDGE_THRESHOLD:
            classification = "NEGATIVE_EDGE"
        else:
            classification = "NEUTRAL"

        row = {
            "symbol": sym,
            "signals": len(sigs),
            "trades": len(returns),
            "win_rate": round(len(wins) / len(returns) * 100, 1) if returns else None,
            "avg_return": round(avg_ret, 4),
            "total_return": round(sum(returns), 4) if returns else 0,
            "avg_mfe": round(sum(mfes) / len(mfes), 4) if mfes else None,
            "avg_mae": round(sum(maes) / len(maes), 4) if maes else None,
            "profit_factor": compute_profit_factor(returns),
            "expectancy": compute_expectancy(returns),
            "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
            "classification": classification,
        }
        results.append(row)

        wr = row["win_rate"] if row["win_rate"] is not None else 0
        print(f"  {sym:>5}: {row['signals']:>2} sigs, WR={wr:.1f}%, avg={row['avg_return']:+.4f}%, PF={row['profit_factor']}, [{classification}]")

    # Sort by expectancy descending
    results.sort(key=lambda r: r["expectancy"] if r["expectancy"] is not None else -999, reverse=True)

    # Edge concentration
    strong = [r for r in results if r["classification"] == "STRONG_EDGE"]
    negative = [r for r in results if r["classification"] == "NEGATIVE_EDGE"]
    neutral = [r for r in results if r["classification"] == "NEUTRAL"]

    strong_total = sum(r["total_return"] for r in strong)
    negative_total = sum(r["total_return"] for r in negative)
    all_total = sum(r["total_return"] for r in results)

    # Allow list / blacklist candidates
    allow_list = [r["symbol"] for r in results if r["classification"] == "STRONG_EDGE"]
    blacklist = [r["symbol"] for r in results if r["classification"] == "NEGATIVE_EDGE"]

    return {
        "symbols": results,
        "edge_concentration": {
            "strong_edge_symbols": len(strong),
            "neutral_symbols": len(neutral),
            "negative_edge_symbols": len(negative),
            "strong_edge_total_return": round(strong_total, 4),
            "negative_edge_total_return": round(negative_total, 4),
            "all_total_return": round(all_total, 4),
        },
        "allow_list": allow_list,
        "blacklist": blacklist,
    }


# ========================================================================
# REPORT GENERATION
# ========================================================================
def generate_report(study1, study2, study3, total_signals):
    """Generate the markdown diagnostics report."""

    lines = []
    lines.append("# Momentum Edge Diagnostics - 2026-03-03")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append(f"**Basis:** {total_signals} v2-pass signals from 2026-03-03")
    lines.append("**Exit model:** stop=-1.0%, trail=+0.8%/0.4%, fallback=5m")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- STUDY 1 ----
    lines.append("## Study 1: Entry Timing Offset")
    lines.append("")
    lines.append("Tests whether entering earlier (lower price) or later (higher price) improves results.")
    lines.append("Negative offset = earlier/better entry. Positive offset = later/worse entry.")
    lines.append("")
    lines.append("| Offset | N | Win Rate | Avg Return | Expectancy | PF | Avg MFE | Avg MAE | Avg Win | Avg Loss |")
    lines.append("|--------|---|----------|------------|------------|-----|---------|---------|---------|----------|")

    for r in study1["offsets"]:
        off = f"{r['offset_pct']:+.2f}%"
        n = r["n"]
        wr = f"{r['win_rate']:.1f}%" if r["win_rate"] is not None else "-"
        ar = f"{r['avg_return']:+.4f}%" if r["avg_return"] is not None else "-"
        exp = f"{r['expectancy']:+.4f}" if r["expectancy"] is not None else "-"
        pf = f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "-"
        mfe = f"{r['avg_mfe']:.3f}%" if r["avg_mfe"] is not None else "-"
        mae = f"{r['avg_mae']:.3f}%" if r["avg_mae"] is not None else "-"
        aw = f"{r['avg_win']:+.3f}%" if r["avg_win"] is not None else "-"
        al = f"{r['avg_loss']:+.3f}%" if r["avg_loss"] is not None else "-"
        marker = " **<-- current**" if r["offset_pct"] == 0.0 else ""
        marker2 = " **<-- best**" if r["offset_pct"] == study1["best_offset"] else ""
        lines.append(f"| {off}{marker}{marker2} | {n} | {wr} | {ar} | {exp} | {pf} | {mfe} | {mae} | {aw} | {al} |")

    lines.append("")
    lines.append(f"**Best offset:** {study1['best_offset']:+.2f}%")
    lines.append(f"**Best expectancy:** {study1['best_expectancy']:+.4f}")
    lines.append(f"**Best avg return:** {study1['best_avg_return']:+.4f}%")
    lines.append(f"**Current avg return:** {study1['current_avg_return']:+.4f}%")
    imp = study1["improvement_vs_current"]
    lines.append(f"**Improvement vs current:** {imp:+.4f}%" if imp is not None else "**Improvement vs current:** N/A")
    lines.append("")

    # Interpretation
    if study1["best_offset"] is not None and study1["best_offset"] < 0:
        lines.append(f"**Finding:** Earlier entry by {abs(study1['best_offset']):.2f}% improves returns by {imp:+.4f}%. Signals are entering LATE.")
    elif study1["best_offset"] is not None and study1["best_offset"] == 0:
        lines.append("**Finding:** Current entry timing is optimal. Entry timing is NOT the problem.")
    else:
        lines.append("**Finding:** Later entry improves results, suggesting entries are slightly early.")
    lines.append("")

    # ---- STUDY 2 ----
    lines.append("---")
    lines.append("")
    lines.append("## Study 2: Hold Time Sweep")
    lines.append("")
    lines.append("Tests fixed-time exits (stop + time only, no trailing stop) at various hold periods.")
    lines.append("")
    lines.append("| Hold Time | N | Win Rate | Avg Return | Expectancy | PF | Avg MFE | Avg MAE | Trail WR | Trail Avg |")
    lines.append("|-----------|---|----------|------------|------------|-----|---------|---------|----------|-----------|")

    for r in study2["hold_times"]:
        ht = f"{r['hold_sec']}s"
        n = r["n"]
        wr = f"{r['win_rate']:.1f}%" if r["win_rate"] is not None else "-"
        ar = f"{r['avg_return']:+.4f}%" if r["avg_return"] is not None else "-"
        exp = f"{r['expectancy']:+.4f}" if r["expectancy"] is not None else "-"
        pf = f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "-"
        mfe = f"{r['avg_mfe']:.3f}%" if r["avg_mfe"] is not None else "-"
        mae = f"{r['avg_mae']:.3f}%" if r["avg_mae"] is not None else "-"
        twr = f"{r['trail_model_win_rate']:.1f}%" if r["trail_model_win_rate"] is not None else "-"
        tar = f"{r['trail_model_avg_return']:+.4f}%" if r["trail_model_avg_return"] is not None else "-"
        marker = " **<-- optimal**" if r["hold_sec"] == study2["optimal_hold_sec"] else ""
        lines.append(f"| {ht}{marker} | {n} | {wr} | {ar} | {exp} | {pf} | {mfe} | {mae} | {twr} | {tar} |")

    lines.append("")
    lines.append(f"**Optimal hold time:** {study2['optimal_hold_sec']}s")
    lines.append(f"**Optimal expectancy:** {study2['optimal_expectancy']:+.4f}")
    lines.append(f"**Optimal avg return:** {study2['optimal_avg_return']:+.4f}%")
    lines.append("")

    # Return decay curve
    lines.append("### Return Decay Curve")
    lines.append("")
    lines.append("```")
    max_bar_width = 40
    curve = study2["return_decay_curve"]
    max_abs = max(abs(r["avg_return"]) for r in curve if r["avg_return"] is not None) or 1
    for r in curve:
        val = r["avg_return"] if r["avg_return"] is not None else 0
        bar_len = int(abs(val) / max_abs * max_bar_width)
        if val >= 0:
            bar = " " * max_bar_width + "|" + "#" * bar_len
        else:
            bar = " " * (max_bar_width - bar_len) + "#" * bar_len + "|"
        lines.append(f"  {r['hold_sec']:>4}s {val:+.4f}%  {bar}")
    lines.append("```")
    lines.append("")

    # ---- STUDY 3 ----
    lines.append("---")
    lines.append("")
    lines.append("## Study 3: Symbol Expectancy Ranking")
    lines.append("")
    lines.append("| Rank | Symbol | Signals | Win Rate | Avg Return | Total Return | PF | Avg MFE | Avg MAE | Classification |")
    lines.append("|------|--------|---------|----------|------------|-------------|-----|---------|---------|----------------|")

    for i, r in enumerate(study3["symbols"], 1):
        wr = f"{r['win_rate']:.1f}%" if r["win_rate"] is not None else "-"
        ar = f"{r['avg_return']:+.4f}%" if r["avg_return"] is not None else "-"
        tr = f"{r['total_return']:+.3f}%" if r["total_return"] is not None else "-"
        pf = f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "-"
        mfe = f"{r['avg_mfe']:.3f}%" if r["avg_mfe"] is not None else "-"
        mae = f"{r['avg_mae']:.3f}%" if r["avg_mae"] is not None else "-"
        cls_marker = r["classification"]
        lines.append(f"| {i} | {r['symbol']} | {r['signals']} | {wr} | {ar} | {tr} | {pf} | {mfe} | {mae} | {cls_marker} |")

    lines.append("")

    # Edge concentration
    ec = study3["edge_concentration"]
    lines.append("### Edge Concentration")
    lines.append("")
    lines.append(f"| Category | Symbols | Total Return |")
    lines.append(f"|----------|---------|-------------|")
    lines.append(f"| STRONG_EDGE | {ec['strong_edge_symbols']} | {ec['strong_edge_total_return']:+.3f}% |")
    lines.append(f"| NEUTRAL | {ec['neutral_symbols']} | - |")
    lines.append(f"| NEGATIVE_EDGE | {ec['negative_edge_symbols']} | {ec['negative_edge_total_return']:+.3f}% |")
    lines.append(f"| **ALL** | **{ec['strong_edge_symbols'] + ec['neutral_symbols'] + ec['negative_edge_symbols']}** | **{ec['all_total_return']:+.3f}%** |")
    lines.append("")

    if study3["allow_list"]:
        lines.append(f"**Allow-list candidates:** {', '.join(study3['allow_list'])}")
    if study3["blacklist"]:
        lines.append(f"**Blacklist candidates:** {', '.join(study3['blacklist'])}")
    lines.append("")

    # Symbol edge visualization
    lines.append("### Symbol Edge Map")
    lines.append("")
    lines.append("```")
    for r in study3["symbols"]:
        val = r["avg_return"]
        bar_len = min(int(abs(val) * 20), 40)
        if val >= 0:
            bar = "+" * bar_len
            lines.append(f"  {r['symbol']:>5}  {val:+.4f}%  |{bar}")
        else:
            bar = "-" * bar_len
            lines.append(f"  {r['symbol']:>5}  {val:+.4f}%  {bar}|")
    lines.append("```")
    lines.append("")

    # ---- FINAL DIAGNOSIS ----
    lines.append("---")
    lines.append("")
    lines.append("## Final Diagnosis")
    lines.append("")

    # Determine primary cause
    causes = []

    # Entry timing
    if study1["best_offset"] is not None and study1["best_offset"] < -0.1:
        entry_severity = abs(study1["improvement_vs_current"]) if study1["improvement_vs_current"] else 0
        causes.append(("LATE ENTRY", entry_severity, f"Best offset is {study1['best_offset']:+.2f}%, improvement of {study1['improvement_vs_current']:+.4f}%"))
    elif study1["best_offset"] is not None and study1["best_offset"] < 0:
        entry_severity = abs(study1["improvement_vs_current"]) if study1["improvement_vs_current"] else 0
        causes.append(("SLIGHTLY LATE ENTRY", entry_severity, f"Marginal improvement at {study1['best_offset']:+.2f}% offset"))

    # Hold time
    if study2["optimal_hold_sec"] is not None and study2["optimal_hold_sec"] < 120:
        hold_severity = abs((study2["optimal_avg_return"] or 0) - (study1["current_avg_return"] or 0))
        causes.append(("HOLDING TOO LONG", hold_severity, f"Optimal hold is {study2['optimal_hold_sec']}s vs current 300s"))

    # Symbol selection
    neg_symbols = len([r for r in study3["symbols"] if r["classification"] == "NEGATIVE_EDGE"])
    if neg_symbols >= 3:
        neg_return = abs(ec["negative_edge_total_return"])
        causes.append(("BAD SYMBOL SELECTION", neg_return, f"{neg_symbols} symbols with negative edge, costing {ec['negative_edge_total_return']:+.3f}%"))

    causes.sort(key=lambda c: c[1], reverse=True)

    if causes:
        lines.append("**Performance issues ranked by severity:**")
        lines.append("")
        for i, (cause, severity, detail) in enumerate(causes, 1):
            lines.append(f"{i}. **{cause}** (impact: {severity:.3f}%) - {detail}")
        lines.append("")
        lines.append(f"**Primary cause:** {causes[0][0]}")
    else:
        lines.append("No clear single cause identified. Issues may be structural or sample-size related.")

    lines.append("")

    # Recommendations
    lines.append("### Recommendations")
    lines.append("")
    if study1["best_offset"] is not None and study1["best_offset"] < 0:
        lines.append(f"1. **Entry timing:** Consider limit-order entries {abs(study1['best_offset']):.2f}% below signal price")
    if study2["optimal_hold_sec"] is not None and study2["optimal_hold_sec"] < 180:
        lines.append(f"2. **Hold time:** Reduce max hold from 300s to {study2['optimal_hold_sec']}s")
    if study3["blacklist"]:
        lines.append(f"3. **Symbol filter:** Consider blacklisting: {', '.join(study3['blacklist'])}")
    if study3["allow_list"]:
        lines.append(f"4. **Symbol focus:** Prioritize: {', '.join(study3['allow_list'])}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This study is research-only. No production changes applied.*")
    lines.append("*All recommendations require review and approval before implementation.*")

    return "\n".join(lines)


# ========================================================================
# MAIN
# ========================================================================
def main():
    print("=" * 70)
    print("MOMENTUM EDGE DIAGNOSTICS STUDY (2026-03-03)")
    print("SuperBot Research Engine - READ ONLY")
    print("=" * 70)

    # Load data
    print("\nLoading v2 replay data...")
    with open(REPLAY_JSON, "r") as f:
        replay = json.load(f)

    pass_signals = [s for s in replay["signals"] if s["v2_result"] == "PASS"]
    print(f"  v2-pass signals: {len(pass_signals)}")

    # Load all symbol quotes
    print("\nLoading quote caches...")
    symbol_quotes = {}
    for sym in sorted(set(s["symbol"] for s in pass_signals)):
        quotes = load_quotes(sym)
        symbol_quotes[sym] = quotes
        print(f"  {sym}: {len(quotes)} quotes")

    # Run studies
    study1 = study1_entry_offset(pass_signals, symbol_quotes)
    study2 = study2_hold_time_sweep(pass_signals, symbol_quotes)
    study3 = study3_symbol_expectancy(pass_signals, symbol_quotes)

    # Generate report
    print("\n=== Generating report ===")
    report_md = generate_report(study1, study2, study3, len(pass_signals))

    # Write outputs
    out1 = OUTPUT_DIR / "entry_offset_study_2026-03-03.json"
    out2 = OUTPUT_DIR / "hold_time_sweep_2026-03-03.json"
    out3 = OUTPUT_DIR / "symbol_expectancy_table_2026-03-03.json"
    out_md = OUTPUT_DIR / "momentum_edge_diagnostics_2026-03-03.md"

    with open(out1, "w") as f:
        json.dump(study1, f, indent=2)
    with open(out2, "w") as f:
        json.dump(study2, f, indent=2)
    with open(out3, "w") as f:
        json.dump(study3, f, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n  -> {out1.name}")
    print(f"  -> {out2.name}")
    print(f"  -> {out3.name}")
    print(f"  -> {out_md.name}")

    # Final console output
    print(f"\n{'=' * 70}")
    print("Momentum edge diagnostics complete")
    print(f"Signals analyzed: {len(pass_signals)}")
    print(f"Best entry offset: {study1['best_offset']:+.2f}%" if study1['best_offset'] is not None else "Best entry offset: N/A")
    print(f"Optimal hold time: {study2['optimal_hold_sec']} seconds" if study2['optimal_hold_sec'] is not None else "Optimal hold time: N/A")

    strong = [r["symbol"] for r in study3["symbols"] if r["classification"] == "STRONG_EDGE"]
    weak = [r["symbol"] for r in study3["symbols"] if r["classification"] == "NEGATIVE_EDGE"]
    print(f"Top symbols: {', '.join(strong) if strong else 'None'}")
    print(f"Worst symbols: {', '.join(weak) if weak else 'None'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
