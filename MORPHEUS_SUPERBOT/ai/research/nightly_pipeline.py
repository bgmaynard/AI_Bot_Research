"""
Nightly Research Pipeline
=========================
Automated research validation that runs nightly after market close.

Modules:
  1. Shadow Replay Optimization — Grid search across exit parameters
  2. Alpha Heatmap — Volatility/spread/OFI performance matrices
  3. Regime Filter Validation — Baseline vs filtered side-by-side
  4. Research Dashboard — Unified daily summary

Auto-detects latest trading day from quote cache.
All outputs date-stamped under reports/research/.

Usage:
    python -m ai.research.nightly_pipeline
    python -m ai.research.nightly_pipeline --date 2026-03-03
    python -m ai.research.nightly_pipeline --date auto

READ ONLY - no production changes.
"""

import argparse
import json
import math
import csv
import time as _time
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from bisect import bisect_left, bisect_right
from itertools import product


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
SUPERBOT = Path(__file__).resolve().parent.parent.parent
ROOT = SUPERBOT.parent
QUOTE_DIR = SUPERBOT / "engine" / "cache" / "quotes"
SIGNALS_FILE = SUPERBOT / "engine" / "output" / "live_signals.json"
PAPER_TRADES_FILE = SUPERBOT / "engine" / "output" / "paper_trades.json"
REPORT_ROOT = ROOT / "reports" / "research"


# ═══════════════════════════════════════════════════════════════════════════════
# Date Detection
# ═══════════════════════════════════════════════════════════════════════════════
def get_quote_cache_dates():
    """Scan quote cache files, return {date_str: [symbols]} for dates with data."""
    dates = defaultdict(list)
    for qf in QUOTE_DIR.glob("*_quotes.json"):
        try:
            with open(qf, encoding="utf-8") as f:
                data = json.load(f)
            d = data.get("date")
            count = data.get("count", 0)
            sym = qf.stem.replace("_quotes", "")
            if d and count > 0:
                dates[d].append(sym)
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return dict(dates)


def detect_latest_trading_day():
    """Find the most recent date with quote cache data."""
    dates = get_quote_cache_dates()
    if not dates:
        return None, []
    latest = max(dates.keys())
    return latest, dates[latest]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_quote_cache(symbol):
    """Load quote cache for a symbol. Returns (quotes, epochs)."""
    path = QUOTE_DIR / ("%s_quotes.json" % symbol)
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    quotes = sorted(data.get("quotes", []), key=lambda q: q["epoch"])
    epochs = [q["epoch"] for q in quotes]
    return quotes, epochs


def load_signals(date_str):
    """Load ignition-passed signals, filtered to the target date."""
    if not SIGNALS_FILE.exists():
        return []
    with open(SIGNALS_FILE, encoding="utf-8") as f:
        signals = json.load(f)

    # Filter to ignition-passed only
    ign = [s for s in signals if s.get("ignition") is True]

    # Filter to target date by epoch
    target = datetime.strptime(date_str, "%Y-%m-%d")
    day_start = target.replace(hour=0, minute=0, second=0).timestamp()
    day_end = (target + timedelta(days=1)).timestamp()

    filtered = [s for s in ign if day_start <= s.get("epoch", 0) < day_end]
    filtered.sort(key=lambda s: s["epoch"])
    return filtered


def load_paper_trades():
    """Load paper_trades.json if it exists."""
    if not PAPER_TRADES_FILE.exists():
        return {"trades": [], "rejected": [], "metrics": {}}
    with open(PAPER_TRADES_FILE, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Feature Computation
# ═══════════════════════════════════════════════════════════════════════════════
def vol_1m(quotes, epochs, epoch):
    """Compute 1-minute volatility (std of log returns) at epoch."""
    hi = bisect_right(epochs, epoch)
    lo = bisect_left(epochs, epoch - 60)
    window = quotes[lo:hi]
    if len(window) < 5:
        return None
    prices = [q["last"] for q in window if q.get("last") and q["last"] > 0]
    if len(prices) < 5:
        return None
    lr = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))
          if prices[i] > 0 and prices[i - 1] > 0]
    if len(lr) < 3:
        return None
    m = sum(lr) / len(lr)
    return math.sqrt(sum((r - m) ** 2 for r in lr) / len(lr)) * 100


def spread_at(quotes, epochs, epoch):
    """Get bid-ask spread % at the nearest quote to epoch."""
    idx = bisect_right(epochs, epoch)
    for off in range(20):
        for d in [0, -1, 1, -2, 2]:
            j = idx + d + (off if d >= 0 else -off)
            if 0 <= j < len(quotes):
                q = quotes[j]
                if q.get("bid") and q.get("ask") and q["bid"] > 0 and q["ask"] > q["bid"]:
                    mid = (q["bid"] + q["ask"]) / 2
                    return (q["ask"] - q["bid"]) / mid * 100
    return None


def ofi_30s(quotes, epochs, epoch):
    """Compute order flow imbalance in 30s window before epoch."""
    hi = bisect_right(epochs, epoch)
    lo = bisect_left(epochs, epoch - 30)
    window = quotes[lo:hi]
    if len(window) < 5:
        return None
    ups = downs = 0
    for i in range(1, len(window)):
        p1 = window[i].get("last", 0)
        p0 = window[i - 1].get("last", 0)
        if p1 > p0:
            ups += 1
        elif p1 < p0:
            downs += 1
    total = ups + downs
    return (ups - downs) / total if total > 0 else 0.0


def classify_tod(epoch):
    """Classify time of day (EST, pre-DST)."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    # Determine EST vs EDT offset based on date
    # DST starts second Sunday of March, ends first Sunday of November
    year = dt.year
    # Simple approximation: March 8-14 range for DST start
    dst_start = datetime(year, 3, 8, tzinfo=timezone.utc)  # approximate
    dst_end = datetime(year, 11, 1, tzinfo=timezone.utc)
    if dst_start <= dt.replace(tzinfo=timezone.utc) < dst_end:
        offset = 4  # EDT
    else:
        offset = 5  # EST
    h_et = dt.hour - offset
    t = h_et * 60 + dt.minute
    if t < 570:
        return "premarket"
    if t < 630:
        return "open"
    if t < 810:
        return "midday"
    return "power_hour"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Trade Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def sim_trade(quotes, epochs, entry_epoch, entry_price,
              trail_pct=1.0, max_hold_s=300, hard_stop_pct=2.0):
    """Simulate a trade with trailing stop and hard stop."""
    start = bisect_left(epochs, entry_epoch)
    peak = entry_price
    exit_price = entry_price
    mfe = mae = hold_s = 0.0
    reason = "TIME_CAP"

    for i in range(start, len(quotes)):
        q = quotes[i]
        dt = q["epoch"] - entry_epoch
        if dt > max_hold_s:
            exit_price = q.get("last", entry_price)
            hold_s = dt
            break
        price = q.get("last", entry_price)
        if price <= 0:
            continue
        chg = (price - entry_price) / entry_price * 100
        if chg > mfe:
            mfe = chg
            peak = price
        if -chg > mae:
            mae = -chg
        if peak > entry_price:
            dd = (peak - price) / peak * 100
            if dd >= trail_pct:
                exit_price = price
                reason = "TRAIL_EXIT"
                hold_s = dt
                break
        if chg <= -hard_stop_pct:
            exit_price = price
            reason = "STOP_EXIT"
            hold_s = dt
            break
        hold_s = dt
        exit_price = price
    else:
        if len(quotes) > start:
            exit_price = quotes[-1].get("last", entry_price)
            hold_s = quotes[-1]["epoch"] - entry_epoch

    ret = (exit_price - entry_price) / entry_price * 100
    return {
        "entry_price": entry_price, "exit_price": exit_price,
        "return_pct": ret, "mfe_pct": mfe, "mae_pct": mae,
        "hold_s": hold_s, "exit_reason": reason,
    }


def calc_metrics(trades):
    """Calculate trading metrics from a list of trade dicts."""
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "avg_pnl": 0, "total_pnl": 0,
                "avg_winner": 0, "avg_loser": 0, "max_dd": 0,
                "avg_mfe": 0, "avg_mae": 0, "avg_hold": 0}
    wins = [t for t in trades if t.get("pnl", t.get("return_pct", 0)) > 0]
    losses = [t for t in trades if t.get("pnl", t.get("return_pct", 0)) <= 0]

    def pnl_of(t):
        return t.get("pnl", t.get("return_pct", 0))

    gw = sum(pnl_of(t) for t in wins)
    gl = sum(abs(pnl_of(t)) for t in losses)
    pf = round(gw / gl, 2) if gl > 0 else ("INF" if gw > 0 else 0)
    total = sum(pnl_of(t) for t in trades)

    equity = 0
    peak_eq = 0
    max_dd = 0
    for t in trades:
        equity += pnl_of(t)
        if equity > peak_eq:
            peak_eq = equity
        dd = peak_eq - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "n": len(trades),
        "wr": round(len(wins) / len(trades) * 100, 1),
        "pf": pf,
        "avg_pnl": round(total / len(trades), 2),
        "total_pnl": round(total, 2),
        "avg_winner": round(gw / len(wins), 2) if wins else 0,
        "avg_loser": round(-gl / len(losses), 2) if losses else 0,
        "max_dd": round(max_dd, 2),
        "avg_mfe": round(sum(t.get("mfe_pct", 0) for t in trades) / len(trades), 3),
        "avg_mae": round(sum(t.get("mae_pct", 0) for t in trades) / len(trades), 3),
        "avg_hold": round(sum(t.get("hold_s", 0) for t in trades) / len(trades), 1),
    }


def fmtd(v):
    """Format dollar value with sign and commas."""
    if v >= 0:
        return "$+{:,.0f}".format(v)
    return "$-{:,.0f}".format(abs(v))


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Shadow Replay Grid Optimization
# ═══════════════════════════════════════════════════════════════════════════════
HOLD_TIMES = [120, 180, 240, 300, 420]
TRAIL_STARTS = [0.10, 0.15, 0.20, 0.25]
TRAIL_OFFSETS = [0.05, 0.08, 0.10, 0.15]
SPREAD_THRESHOLDS = [0.4, 0.6, 0.8, 1.0]
CONTAINMENT_PULLBACKS = [0.15, 0.25, 0.35, 0.50]
SESSION_CAPS = [20, 30, 40]
POSITION_SIZE_DOLLARS = 100_000


def run_grid_replay(date_str, symbols, qcaches, out_dir):
    """Run grid replay optimization across all symbols for the date."""
    print("\n" + "=" * 70)
    print("MODULE 1: SHADOW REPLAY GRID OPTIMIZATION")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather all signals for the date
    signals = load_signals(date_str)
    if not signals:
        # Fall back to paper_trades.json
        paper = load_paper_trades()
        signals_raw = []
        for t in paper.get("trades", []):
            signals_raw.append({"epoch": t["entry_epoch"], "price": t["entry_price"],
                                "symbol": t.get("symbol", "BATL"), "source": "executed"})
        for r in paper.get("rejected", []):
            signals_raw.append({"epoch": r["epoch"], "price": r["price"],
                                "symbol": r.get("symbol", "BATL"), "source": "rejected"})
        signals = sorted(signals_raw, key=lambda s: s.get("epoch", 0))

    if not signals:
        print("  No signals found — skipping grid replay")
        return {"status": "NO_DATA", "configs": 0}

    print("  Signals: %d" % len(signals))

    # Pre-compute trade outcomes for each (hold_time, trail_start, trail_offset) combo
    exit_combos = list(product(HOLD_TIMES, TRAIL_STARTS, TRAIL_OFFSETS))
    print("  Exit combos: %d" % len(exit_combos))

    # For each signal, simulate with each exit combo
    sig_results = {}  # (sig_idx, hold, trail_s, trail_o) -> trade result
    for si, sig in enumerate(signals):
        sym = sig.get("symbol", "BATL")
        if sym not in qcaches:
            continue
        quotes, epochs = qcaches[sym]
        entry_epoch = sig.get("epoch", sig.get("entry_epoch", 0))
        entry_price = sig.get("price", sig.get("entry_price", 0))
        if entry_price <= 0:
            continue

        for hold, ts, to in exit_combos:
            # Simulate with specific trail parameters
            start = bisect_left(epochs, entry_epoch)
            if start >= len(quotes):
                continue
            peak_price = entry_price
            exit_p = entry_price
            trail_active = False
            trail_stop = 0.0
            exit_reason = "TIME_CAP"
            mfe = mae = 0.0

            for i in range(start, len(quotes)):
                q = quotes[i]
                dt = q["epoch"] - entry_epoch
                if dt > hold:
                    exit_p = q.get("last", entry_price)
                    break
                price = q.get("last", entry_price)
                if price <= 0:
                    continue

                chg_pct = (price - entry_price) / entry_price * 100
                if chg_pct > mfe:
                    mfe = chg_pct
                    peak_price = price
                if -chg_pct > mae:
                    mae = -chg_pct

                # Trailing stop logic
                if not trail_active and chg_pct >= ts:
                    trail_active = True
                    trail_stop = price * (1 - to / 100)

                if trail_active:
                    new_stop = price * (1 - to / 100)
                    if new_stop > trail_stop:
                        trail_stop = new_stop
                    if price <= trail_stop:
                        exit_p = price
                        exit_reason = "TRAIL_EXIT"
                        break

                # Hard stop at -2%
                if chg_pct <= -2.0:
                    exit_p = price
                    exit_reason = "STOP_EXIT"
                    break

                exit_p = price

            pnl_pct = (exit_p - entry_price) / entry_price * 100
            pnl_dollar = pnl_pct / 100 * POSITION_SIZE_DOLLARS

            # Get spread at entry
            sp = spread_at(quotes, epochs, entry_epoch)

            # Get recent high for containment
            end_idx = bisect_left(epochs, entry_epoch)
            start_ep = entry_epoch - 120
            s_idx = bisect_left(epochs, start_ep)
            recent_prices = [q["last"] for q in quotes[s_idx:end_idx]
                             if q.get("last") and q["last"] > 0]
            recent_high = max(recent_prices) if recent_prices else None

            sig_results[(si, hold, ts, to)] = {
                "pnl_pct": pnl_pct, "pnl": pnl_dollar,
                "mfe_pct": mfe, "mae_pct": mae,
                "exit_reason": exit_reason, "spread": sp,
                "entry_price": entry_price, "recent_high": recent_high,
                "symbol": sym,
            }

    # Evaluate all grid configurations
    configs = list(product(HOLD_TIMES, TRAIL_STARTS, TRAIL_OFFSETS,
                           SPREAD_THRESHOLDS, CONTAINMENT_PULLBACKS, SESSION_CAPS))
    print("  Grid configs: %d" % len(configs))

    results = []
    for hold, ts, to, sp_thresh, cpb, cap in configs:
        trades = []
        cap_count = 0
        for si, sig in enumerate(signals):
            key = (si, hold, ts, to)
            if key not in sig_results:
                continue
            r = sig_results[key]

            # Spread filter
            if r["spread"] is not None and r["spread"] > sp_thresh:
                continue
            # Containment filter
            if r["recent_high"] is not None and r["entry_price"] > 0:
                pullback = (r["recent_high"] - r["entry_price"]) / r["recent_high"] * 100
                if pullback < 0 and abs(pullback) > cpb * 100:
                    continue

            # Cap
            if cap_count >= cap:
                continue
            trades.append(r)
            cap_count += 1

        if not trades:
            continue

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins)
        gl = sum(abs(t["pnl"]) for t in losses)
        pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0)
        total_pnl = sum(t["pnl"] for t in trades)

        # Skip configs with too few trades for statistical significance
        if len(trades) < 5:
            continue

        results.append({
            "hold": hold, "trail_start": ts, "trail_offset": to,
            "spread_thresh": sp_thresh, "pullback": cpb, "cap": cap,
            "n": len(trades), "wins": len(wins), "losses": len(losses),
            "wr": len(wins) / len(trades) * 100 if trades else 0,
            "pf": round(pf, 3), "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        })

    # Sort: prefer configs with real losses for meaningful PF, then by total_pnl
    # Configs with <3 losses have unreliable PF, rank them lower
    results.sort(key=lambda r: (
        1 if r["losses"] >= 3 else 0,  # meaningful loss sample first
        r["pf"] if r["losses"] >= 3 else 0,  # PF only matters with losses
        r["total_pnl"],  # tiebreaker: most profitable
    ), reverse=True)
    print("  Evaluated: %d configs with trades" % len(results))

    # Write results
    # Top configurations report
    top = results[:20] if results else []
    lines = [
        "# Shadow Replay Grid Optimization",
        "## Date: %s | %d Configs | %d Signals" % (date_str, len(configs), len(signals)),
        "## Generated: %s" % datetime.now().strftime("%Y-%m-%d"),
        "", "---", "",
        "## TOP 20 CONFIGURATIONS BY PROFIT FACTOR",
        "",
        "| # | Hold | Trail Start | Trail Offset | Spread | Pullback | Cap | N | WR | PF | Total PnL |",
        "|---|------|-------------|-------------|--------|----------|-----|---|----|----|-----------|",
    ]
    for i, r in enumerate(top, 1):
        lines.append("| %d | %ds | %.2f%% | %.2f%% | %.1f%% | %.2f | %d | %d | %.1f%% | %.3f | %s |" % (
            i, r["hold"], r["trail_start"], r["trail_offset"],
            r["spread_thresh"], r["pullback"], r["cap"],
            r["n"], r["wr"], r["pf"], fmtd(r["total_pnl"])))

    if results:
        best = results[0]
        lines += ["", "---", "",
                   "## BEST CONFIGURATION",
                   "",
                   "| Parameter | Value |",
                   "|-----------|-------|",
                   "| Hold Time | %ds |" % best["hold"],
                   "| Trail Start | %.2f%% |" % best["trail_start"],
                   "| Trail Offset | %.2f%% |" % best["trail_offset"],
                   "| Spread Threshold | %.1f%% |" % best["spread_thresh"],
                   "| Containment Pullback | %.2f |" % best["pullback"],
                   "| Session Cap | %d |" % best["cap"],
                   "| Trades | %d |" % best["n"],
                   "| Win Rate | %.1f%% |" % best["wr"],
                   "| Profit Factor | %.3f |" % best["pf"],
                   "| Total PnL | %s |" % fmtd(best["total_pnl"]),
                   ]

    # Parameter sensitivity
    lines += ["", "---", "",
              "## PARAMETER SENSITIVITY", ""]
    for param_name, param_values, param_key in [
        ("Hold Time", HOLD_TIMES, "hold"),
        ("Trail Start", TRAIL_STARTS, "trail_start"),
        ("Trail Offset", TRAIL_OFFSETS, "trail_offset"),
        ("Spread", SPREAD_THRESHOLDS, "spread_thresh"),
        ("Cap", SESSION_CAPS, "cap"),
    ]:
        lines.append("### %s" % param_name)
        lines.append("")
        lines.append("| Value | Avg PF | Avg WR | Avg PnL |")
        lines.append("|-------|--------|--------|---------|")
        for val in param_values:
            subset = [r for r in results if r[param_key] == val]
            if subset:
                avg_pf = sum(r["pf"] for r in subset) / len(subset)
                avg_wr = sum(r["wr"] for r in subset) / len(subset)
                avg_pnl = sum(r["avg_pnl"] for r in subset) / len(subset)
                lines.append("| %s | %.3f | %.1f%% | %s |" % (
                    val, avg_pf, avg_wr, fmtd(avg_pnl)))
        lines.append("")

    lines += ["", "---",
              "*Data source: quote cache + signals (READ-ONLY)*",
              "*NO production changes were made.*"]

    with open(out_dir / "top_configurations.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # CSV output
    if results:
        with open(out_dir / "replay_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    best_config = results[0] if results else None
    print("  Best config: PF=%.3f" % best_config["pf"] if best_config else "  No results")
    print("  Reports written to: %s" % out_dir)

    return {
        "status": "OK",
        "configs": len(results),
        "best": best_config,
        "signals": len(signals),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Alpha Heatmap Validation
# ═══════════════════════════════════════════════════════════════════════════════
def run_alpha_heatmap(date_str, symbols, qcaches, out_dir):
    """Compute alpha heatmaps: vol/spread/OFI performance matrices."""
    print("\n" + "=" * 70)
    print("MODULE 2: ALPHA HEATMAP VALIDATION")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    signals = load_signals(date_str)
    if not signals:
        print("  No signals found — skipping")
        return {"status": "NO_DATA", "trades": 0}

    print("  Signals: %d across %d symbols" % (len(signals), len(set(s.get("symbol", "") for s in signals))))

    # Simulate all trades and extract features
    POSITION_SIZE = 5000
    trades = []
    for sig in signals:
        sym = sig.get("symbol", "")
        if sym not in qcaches:
            continue
        quotes, epochs = qcaches[sym]
        epoch = sig["epoch"]
        price = sig["price"]
        if price <= 0:
            continue

        result = sim_trade(quotes, epochs, epoch, price)
        if result["hold_s"] < 1:
            continue

        v = vol_1m(quotes, epochs, epoch)
        sp = spread_at(quotes, epochs, epoch)
        ofi = ofi_30s(quotes, epochs, epoch)
        tod = classify_tod(epoch)
        regime = sig.get("regime", "unknown")

        pnl = (result["exit_price"] - price) * POSITION_SIZE
        result["pnl"] = pnl
        result["symbol"] = sym
        result["epoch"] = epoch
        result["vol_1m"] = v
        result["spread"] = sp
        result["ofi"] = ofi
        result["tod"] = tod
        result["regime"] = regime
        trades.append(result)

    if not trades:
        print("  No trades simulated")
        return {"status": "NO_TRADES", "trades": 0}

    print("  Trades simulated: %d" % len(trades))

    # Bin functions
    def vol_bin(v):
        if v is None:
            return "N/A"
        if v < 0.3:
            return "low"
        if v < 0.8:
            return "medium"
        return "high"

    def spread_bin(s):
        if s is None:
            return "N/A"
        if s < 0.3:
            return "<0.3%"
        if s < 0.6:
            return "0.3-0.6%"
        if s < 1.0:
            return "0.6-1.0%"
        return ">1.0%"

    def ofi_bin(o):
        if o is None:
            return "N/A"
        if o < -0.2:
            return "weak"
        if o < 0.3:
            return "moderate"
        return "strong"

    # Build heatmaps
    def heatmap_2d(trades, key_a, key_b, bin_a, bin_b):
        grid = defaultdict(list)
        for t in trades:
            a = bin_a(t.get(key_a))
            b = bin_b(t.get(key_b))
            grid[(a, b)].append(t)
        return grid

    # 1. Volatility × Spread
    vs_grid = heatmap_2d(trades, "vol_1m", "spread", vol_bin, spread_bin)

    lines = [
        "# Volatility × Spread Heatmap",
        "## Date: %s | %d Trades" % (date_str, len(trades)),
        "## Generated: %s" % datetime.now().strftime("%Y-%m-%d"),
        "", "---", "",
        "| Vol \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% | N/A |",
        "|-------------|-------|----------|----------|-------|-----|",
    ]
    for vb in ["low", "medium", "high", "N/A"]:
        cells = []
        for sb in ["<0.3%", "0.3-0.6%", "0.6-1.0%", ">1.0%", "N/A"]:
            ts = vs_grid.get((vb, sb), [])
            if ts:
                m = calc_metrics(ts)
                cells.append("n=%d PF=%s" % (m["n"], m["pf"]))
            else:
                cells.append("-")
        lines.append("| %s | %s |" % (vb, " | ".join(cells)))

    lines += ["", "---",
              "*All data READ-ONLY. NO production changes.*"]

    with open(out_dir / "volatility_spread_heatmap.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 2. OFI × Spread
    os_grid = heatmap_2d(trades, "ofi", "spread", ofi_bin, spread_bin)

    lines = [
        "# Order Flow × Spread Heatmap",
        "## Date: %s | %d Trades" % (date_str, len(trades)),
        "## Generated: %s" % datetime.now().strftime("%Y-%m-%d"),
        "", "---", "",
        "| OFI \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% | N/A |",
        "|-------------|-------|----------|----------|-------|-----|",
    ]
    for ob in ["weak", "moderate", "strong", "N/A"]:
        cells = []
        for sb in ["<0.3%", "0.3-0.6%", "0.6-1.0%", ">1.0%", "N/A"]:
            ts = os_grid.get((ob, sb), [])
            if ts:
                m = calc_metrics(ts)
                cells.append("n=%d PF=%s" % (m["n"], m["pf"]))
            else:
                cells.append("-")
        lines.append("| %s | %s |" % (ob, " | ".join(cells)))

    lines += ["", "---",
              "*All data READ-ONLY. NO production changes.*"]

    with open(out_dir / "orderflow_spread_heatmap.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 3. Time of Day Performance
    tod_groups = defaultdict(list)
    for t in trades:
        tod_groups[t["tod"]].append(t)

    lines = [
        "# Time of Day Performance",
        "## Date: %s | %d Trades" % (date_str, len(trades)),
        "## Generated: %s" % datetime.now().strftime("%Y-%m-%d"),
        "", "---", "",
        "| Session | Trades | WR | PF | Avg PnL | Total PnL |",
        "|---------|--------|-----|-----|---------|-----------|",
    ]
    for session in ["premarket", "open", "midday", "power_hour"]:
        ts = tod_groups.get(session, [])
        if ts:
            m = calc_metrics(ts)
            lines.append("| %s | %d | %.1f%% | %s | %s | %s |" % (
                session, m["n"], m["wr"], m["pf"],
                fmtd(m["avg_pnl"]), fmtd(m["total_pnl"])))
        else:
            lines.append("| %s | 0 | - | - | - | - |" % session)

    lines += ["", "---",
              "*All data READ-ONLY. NO production changes.*"]

    with open(out_dir / "time_of_day_performance.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 4. Per-symbol summary
    sym_groups = defaultdict(list)
    for t in trades:
        sym_groups[t["symbol"]].append(t)

    lines = [
        "# Per-Symbol Performance",
        "## Date: %s | %d Trades | %d Symbols" % (date_str, len(trades), len(sym_groups)),
        "## Generated: %s" % datetime.now().strftime("%Y-%m-%d"),
        "", "---", "",
        "| Symbol | Trades | WR | PF | Avg PnL | Total PnL |",
        "|--------|--------|-----|-----|---------|-----------|",
    ]
    for sym in sorted(sym_groups.keys()):
        ts = sym_groups[sym]
        m = calc_metrics(ts)
        lines.append("| %s | %d | %.1f%% | %s | %s | %s |" % (
            sym, m["n"], m["wr"], m["pf"],
            fmtd(m["avg_pnl"]), fmtd(m["total_pnl"])))

    lines += ["", "---",
              "*All data READ-ONLY. NO production changes.*"]

    with open(out_dir / "per_symbol_performance.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Combined filter validation
    filter_pass = [t for t in trades
                   if (t["vol_1m"] is None or t["vol_1m"] >= 0.3)
                   and (t["spread"] is None or t["spread"] <= 0.6)
                   and (t["ofi"] is None or t["ofi"] >= -0.2)]
    filter_fail = [t for t in trades if t not in filter_pass]

    m_all = calc_metrics(trades)
    m_pass = calc_metrics(filter_pass)
    m_fail = calc_metrics(filter_fail)

    print("  All: n=%d PF=%s WR=%.1f%%" % (m_all["n"], m_all["pf"], m_all["wr"]))
    print("  Filter PASS: n=%d PF=%s WR=%.1f%%" % (m_pass["n"], m_pass["pf"], m_pass["wr"]))
    print("  Filter FAIL: n=%d PF=%s WR=%.1f%%" % (m_fail["n"], m_fail["pf"], m_fail["wr"]))
    print("  Reports written to: %s" % out_dir)

    return {
        "status": "OK",
        "trades": len(trades),
        "symbols": len(sym_groups),
        "all": m_all,
        "filter_pass": m_pass,
        "filter_fail": m_fail,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Regime Filter Validation
# ═══════════════════════════════════════════════════════════════════════════════
VOL_THRESHOLD = 0.3
SPREAD_THRESHOLD = 0.6
OFI_THRESHOLD = -0.2
SUPPRESS_REGIMES = {"LOW_VOLATILITY"}
SUPPRESS_SESSIONS = {"power_hour"}
POSITION_SIZE = 5000


def run_regime_validation(date_str, symbols, qcaches, out_dir, scorecard_path):
    """Run baseline vs filtered regime validation."""
    print("\n" + "=" * 70)
    print("MODULE 3: REGIME FILTER VALIDATION")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    signals = load_signals(date_str)
    if not signals:
        print("  No signals found — skipping")
        return {"status": "NO_DATA"}

    print("  Signals: %d" % len(signals))

    all_baseline = []
    all_filtered = []
    all_blocked = []

    for sig in signals:
        sym = sig.get("symbol", "")
        if sym not in qcaches:
            continue
        quotes, epochs = qcaches[sym]
        epoch = sig["epoch"]
        price = sig["price"]
        if price <= 0:
            continue

        result = sim_trade(quotes, epochs, epoch, price)
        if result["hold_s"] < 1:
            continue

        pnl = (result["exit_price"] - price) * POSITION_SIZE
        result["pnl"] = pnl

        v = vol_1m(quotes, epochs, epoch)
        sp = spread_at(quotes, epochs, epoch)
        ofi = ofi_30s(quotes, epochs, epoch)
        tod = classify_tod(epoch)
        regime = sig.get("regime", "unknown")

        trade = {**result, "symbol": sym, "epoch": epoch,
                 "vol_1m": v, "spread": sp, "ofi": ofi,
                 "tod": tod, "regime": regime}

        all_baseline.append(trade)

        passes = True
        reasons = []
        if v is not None and v < VOL_THRESHOLD:
            passes = False
            reasons.append("LOW_VOL")
        if sp is not None and sp > SPREAD_THRESHOLD:
            passes = False
            reasons.append("HIGH_SPREAD")
        if ofi is not None and ofi < OFI_THRESHOLD:
            passes = False
            reasons.append("WEAK_OFI")
        if regime in SUPPRESS_REGIMES:
            passes = False
            reasons.append("SUPPRESS_REGIME")
        if tod in SUPPRESS_SESSIONS:
            passes = False
            reasons.append("SUPPRESS_SESSION")

        if passes:
            all_filtered.append(trade)
        else:
            trade["block_reasons"] = reasons
            all_blocked.append(trade)

    m_base = calc_metrics(all_baseline)
    m_filt = calc_metrics(all_filtered)
    m_blocked = calc_metrics(all_blocked)

    print("  Baseline: %d trades, PF=%s, WR=%.1f%%" % (m_base["n"], m_base["pf"], m_base["wr"]))
    print("  Filtered: %d trades, PF=%s, WR=%.1f%%" % (m_filt["n"], m_filt["pf"], m_filt["wr"]))
    print("  Blocked: %d trades, PF=%s, WR=%.1f%%" % (m_blocked["n"], m_blocked["pf"], m_blocked["wr"]))

    # Block reason distribution
    reason_counts = Counter()
    for t in all_blocked:
        for r in t.get("block_reasons", []):
            reason_counts[r] += 1

    # Per-symbol
    def by_group(trades, key):
        groups = defaultdict(list)
        for t in trades:
            groups[t.get(key, "unknown")].append(t)
        return {k: calc_metrics(v) for k, v in groups.items()}

    base_sym = by_group(all_baseline, "symbol")
    filt_sym = by_group(all_filtered, "symbol")
    blocked_sym = by_group(all_blocked, "symbol")

    base_sess = by_group(all_baseline, "tod")
    filt_sess = by_group(all_filtered, "tod")

    # Write daily report
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = out_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Regime Filter Validation — Daily Report",
        "## Date: %s" % date_str,
        "## Generated: %s" % today,
        "", "---", "",
        "## FILTER CONFIGURATION",
        "",
        "```",
        "volatility_1m >= %.1f%% | spread <= %.1f%% | OFI >= %.1f" % (
            VOL_THRESHOLD, SPREAD_THRESHOLD, OFI_THRESHOLD),
        "suppress: %s | %s" % (
            ", ".join(SUPPRESS_REGIMES), ", ".join(SUPPRESS_SESSIONS)),
        "simulation: trail=1.0%%, max_hold=300s",
        "```",
        "", "---", "",
        "## SIDE-BY-SIDE (Uncapped)",
        "",
        "| Metric | Baseline | Filtered | Blocked | Filter Edge |",
        "|--------|----------|----------|---------|-------------|",
        "| Trades | %d | %d | %d | - |" % (m_base["n"], m_filt["n"], m_blocked["n"]),
        "| Win Rate | %.1f%% | %.1f%% | %.1f%% | %+.1f pp |" % (
            m_base["wr"], m_filt["wr"], m_blocked["wr"],
            m_filt["wr"] - m_base["wr"]),
        "| Profit Factor | %s | %s | %s | %s |" % (
            m_base["pf"], m_filt["pf"], m_blocked["pf"],
            "+%.2f" % (float(m_filt["pf"]) - float(m_base["pf"]))
            if isinstance(m_filt["pf"], (int, float)) and isinstance(m_base["pf"], (int, float))
            else "N/A"),
        "| Total PnL | %s | %s | %s | %s |" % (
            fmtd(m_base["total_pnl"]), fmtd(m_filt["total_pnl"]),
            fmtd(m_blocked["total_pnl"]),
            fmtd(m_filt["total_pnl"] - m_base["total_pnl"])),
        "| Avg PnL | %s | %s | %s | %s |" % (
            fmtd(m_base["avg_pnl"]), fmtd(m_filt["avg_pnl"]),
            fmtd(m_blocked["avg_pnl"]),
            fmtd(m_filt["avg_pnl"] - m_base["avg_pnl"])),
        "| Max DD | %s | %s | - | %s |" % (
            fmtd(m_base["max_dd"]), fmtd(m_filt["max_dd"]),
            fmtd(m_base["max_dd"] - m_filt["max_dd"])),
    ]

    # Block reason table
    lines += ["", "---", "",
              "## BLOCK REASON DISTRIBUTION", "",
              "| Reason | Count | %% of Blocked |",
              "|--------|-------|--------------:|"]
    total_reasons = sum(reason_counts.values())
    for reason, count in reason_counts.most_common():
        pct = count / total_reasons * 100 if total_reasons > 0 else 0
        lines.append("| %s | %d | %.1f%% |" % (reason, count, pct))

    # Per-symbol table
    lines += ["", "---", "",
              "## PER-SYMBOL", "",
              "| Symbol | Base n | Base PF | Filt n | Filt PF | Blocked n | Blocked PF | Edge |",
              "|--------|--------|---------|--------|---------|-----------|------------|------|"]
    for sym in sorted(set(list(base_sym.keys()) + list(filt_sym.keys()))):
        bm = base_sym.get(sym, {"n": 0, "pf": 0})
        fm = filt_sym.get(sym, {"n": 0, "pf": 0})
        blm = blocked_sym.get(sym, {"n": 0, "pf": 0})
        edge = "YES" if (isinstance(fm["pf"], (int, float)) and isinstance(bm["pf"], (int, float))
                         and fm["pf"] > bm["pf"] and fm["n"] >= 5) else "NO"
        if fm["n"] == 0:
            edge = "N/A"
        lines.append("| %s | %d | %s | %d | %s | %d | %s | %s |" % (
            sym, bm["n"], bm["pf"], fm["n"], fm["pf"],
            blm["n"], blm["pf"], edge))

    # Per-session table
    lines += ["", "---", "",
              "## PER-SESSION", "",
              "| Session | Base n | Base PF | Filt n | Filt PF | Edge |",
              "|---------|--------|---------|--------|---------|------|"]
    for sess in ["premarket", "open", "midday", "power_hour"]:
        bm = base_sess.get(sess, {"n": 0, "pf": 0})
        fm = filt_sess.get(sess, {"n": 0, "pf": 0})
        if fm["n"] == 0:
            edge = "SUPPRESSED" if sess in SUPPRESS_SESSIONS else "N/A"
        elif isinstance(fm["pf"], (int, float)) and isinstance(bm["pf"], (int, float)):
            edge = "YES" if fm["pf"] > bm["pf"] else "NO"
        else:
            edge = "N/A"
        lines.append("| %s | %d | %s | %d | %s | %s |" % (
            sess, bm["n"], bm["pf"], fm["n"], fm["pf"], edge))

    lines += ["", "---",
              "*All data READ-ONLY. NO production changes.*"]

    with open(day_dir / "daily_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Missed trades analysis
    blocked_winners = [t for t in all_blocked if t.get("pnl", 0) > 0]
    blocked_losers = [t for t in all_blocked if t.get("pnl", 0) <= 0]
    missed_profit = sum(t["pnl"] for t in blocked_winners)
    avoided_losses = sum(abs(t["pnl"]) for t in blocked_losers)

    lines = [
        "# Missed Trades Analysis",
        "## Date: %s" % date_str,
        "## Generated: %s" % today,
        "", "---", "",
        "## MISSED vs AVOIDED", "",
        "| Metric | Value |",
        "|--------|-------|",
        "| Total blocked trades | %d |" % len(all_blocked),
        "| Blocked winners | %d |" % len(blocked_winners),
        "| Blocked losers | %d |" % len(blocked_losers),
        "| Missed profit (winners blocked) | %s |" % fmtd(missed_profit),
        "| Avoided losses (losers blocked) | %s |" % fmtd(avoided_losses),
        "| Net filter value | %s |" % fmtd(avoided_losses - missed_profit),
        "",
        "**The filter %s %s more in losses than it misses in profits.**" % (
            "avoids" if avoided_losses > missed_profit else "misses",
            fmtd(abs(avoided_losses - missed_profit))),
        "", "---",
        "*All data READ-ONLY. NO production changes.*",
    ]

    with open(day_dir / "missed_trades.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Update rolling scorecard
    _update_scorecard(date_str, m_base, m_filt, m_blocked, reason_counts,
                      base_sym, filt_sym, blocked_sym,
                      base_sess, filt_sess, scorecard_path)

    print("  Reports written to: %s" % day_dir)

    return {
        "status": "OK",
        "baseline": m_base,
        "filtered": m_filt,
        "blocked": m_blocked,
        "net_filter_value": avoided_losses - missed_profit,
    }


def _update_scorecard(date_str, m_base, m_filt, m_blocked, reason_counts,
                      base_sym, filt_sym, blocked_sym,
                      base_sess, filt_sess, scorecard_path):
    """Update the rolling regime filter scorecard."""
    # Load existing scorecard data
    data_file = scorecard_path.parent / "scorecard_data.json"
    if data_file.exists():
        with open(data_file, encoding="utf-8") as f:
            scorecard_data = json.load(f)
    else:
        scorecard_data = {"days": []}

    # Add or update this day's data
    day_entry = {
        "date": date_str,
        "baseline": m_base,
        "filtered": m_filt,
        "blocked": m_blocked,
        "reason_counts": dict(reason_counts),
    }

    # Replace if date already exists
    scorecard_data["days"] = [d for d in scorecard_data["days"] if d["date"] != date_str]
    scorecard_data["days"].append(day_entry)
    scorecard_data["days"].sort(key=lambda d: d["date"])

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(scorecard_data, f, indent=2)

    # Regenerate scorecard markdown
    days = scorecard_data["days"]
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# Regime Filter Daily Scorecard",
        "## Rolling Summary Across %d Day(s)" % len(days),
        "## Generated: %s" % today,
        "", "---", "",
        "## DAILY METRICS", "",
        "| Date | Base n | Base PF | Filt n | Filt PF | Blocked n | Blocked PF | Filter Edge |",
        "|------|--------|---------|--------|---------|-----------|------------|-------------|",
    ]

    total_base_n = 0
    total_filt_n = 0
    days_filter_helps = 0

    for d in days:
        b = d["baseline"]
        f_ = d["filtered"]
        bl = d["blocked"]
        edge = "YES"
        if isinstance(f_["pf"], (int, float)) and isinstance(b["pf"], (int, float)):
            edge = "YES" if f_["pf"] > b["pf"] else "NO"
            if f_["pf"] > b["pf"]:
                days_filter_helps += 1
        total_base_n += b["n"]
        total_filt_n += f_["n"]
        lines.append("| %s | %d | %s | %d | %s | %d | %s | %s |" % (
            d["date"], b["n"], b["pf"], f_["n"], f_["pf"],
            bl["n"], bl["pf"], edge))

    # Stability metrics
    filter_reduction = (1 - total_filt_n / total_base_n) * 100 if total_base_n > 0 else 0

    lines += [
        "", "---", "",
        "## STABILITY METRICS", "",
        "| Metric | Value | Assessment |",
        "|--------|-------|------------|",
        "| Days with data | %d | %s |" % (
            len(days), "SUFFICIENT" if len(days) >= 5 else "INSUFFICIENT for production"),
        "| Days filter improves PF | %d/%d | %s |" % (
            days_filter_helps, len(days),
            "Positive" if days_filter_helps > len(days) / 2 else "Mixed"),
        "| Avg trade reduction | %.0f%% | - |" % filter_reduction,
    ]

    # Deployment readiness
    if len(days) >= 5 and days_filter_helps >= len(days) * 0.7:
        readiness = "READY (pending manual review)"
    elif len(days) >= 5:
        readiness = "MARGINAL — filter doesn't consistently help"
    else:
        readiness = "NOT READY — need %d more day(s)" % (5 - len(days))

    lines += [
        "", "---", "",
        "## DEPLOYMENT READINESS: %s" % readiness.split(" — ")[0].split(" (")[0],
        "",
        "**%s**" % readiness,
        "", "---",
        "*All data READ-ONLY. NO production changes.*",
    ]

    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Research Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def generate_dashboard(date_str, replay_result, heatmap_result, regime_result, out_path):
    """Generate unified daily research summary."""
    print("\n" + "=" * 70)
    print("MODULE 4: RESEARCH DASHBOARD")
    print("=" * 70)

    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# Daily Research Summary",
        "## Trading Day: %s" % date_str,
        "## Generated: %s" % today,
        "", "---", "",
        "## PIPELINE STATUS", "",
        "| Module | Status | Key Metric |",
        "|--------|--------|------------|",
    ]

    # Replay summary
    if replay_result and replay_result.get("status") == "OK":
        best = replay_result.get("best", {})
        lines.append("| Shadow Replay | OK | Best PF=%.3f (%d configs) |" % (
            best.get("pf", 0), replay_result.get("configs", 0)))
    else:
        lines.append("| Shadow Replay | %s | - |" % (
            replay_result.get("status", "SKIPPED") if replay_result else "SKIPPED"))

    # Heatmap summary
    if heatmap_result and heatmap_result.get("status") == "OK":
        m = heatmap_result.get("all", {})
        fp = heatmap_result.get("filter_pass", {})
        lines.append("| Alpha Heatmap | OK | %d trades, Filter PASS PF=%s |" % (
            heatmap_result.get("trades", 0), fp.get("pf", "N/A")))
    else:
        lines.append("| Alpha Heatmap | %s | - |" % (
            heatmap_result.get("status", "SKIPPED") if heatmap_result else "SKIPPED"))

    # Regime summary
    if regime_result and regime_result.get("status") == "OK":
        b = regime_result.get("baseline", {})
        f_ = regime_result.get("filtered", {})
        nfv = regime_result.get("net_filter_value", 0)
        lines.append("| Regime Filter | OK | PF %s->%s, NFV=%s |" % (
            b.get("pf", "?"), f_.get("pf", "?"), fmtd(nfv)))
    else:
        lines.append("| Regime Filter | %s | - |" % (
            regime_result.get("status", "SKIPPED") if regime_result else "SKIPPED"))

    # Overall assessment
    lines += ["", "---", "",
              "## HIGHLIGHTS", ""]

    if replay_result and replay_result.get("status") == "OK":
        best = replay_result.get("best", {})
        lines += [
            "### Shadow Replay",
            "- **Best config**: hold=%ds, trail_start=%.2f%%, trail_offset=%.2f%%, "
            "spread=%.1f%%, cap=%d" % (
                best.get("hold", 0), best.get("trail_start", 0),
                best.get("trail_offset", 0), best.get("spread_thresh", 0),
                best.get("cap", 0)),
            "- **PF**: %.3f | **WR**: %.1f%% | **Trades**: %d" % (
                best.get("pf", 0), best.get("wr", 0), best.get("n", 0)),
            "- **Signals evaluated**: %d" % replay_result.get("signals", 0),
            "",
        ]

    if regime_result and regime_result.get("status") == "OK":
        b = regime_result.get("baseline", {})
        f_ = regime_result.get("filtered", {})
        lines += [
            "### Regime Filter",
            "- **Baseline**: %d trades, PF=%s, WR=%.1f%%" % (
                b.get("n", 0), b.get("pf", "?"), b.get("wr", 0)),
            "- **Filtered**: %d trades, PF=%s, WR=%.1f%%" % (
                f_.get("n", 0), f_.get("pf", "?"), f_.get("wr", 0)),
            "- **Trade reduction**: %.0f%%" % (
                (1 - f_.get("n", 0) / b.get("n", 1)) * 100 if b.get("n", 0) > 0 else 0),
            "- **Net filter value**: %s" % fmtd(regime_result.get("net_filter_value", 0)),
            "",
        ]

    if heatmap_result and heatmap_result.get("status") == "OK":
        lines += [
            "### Alpha Heatmap",
            "- **Trades analyzed**: %d across %d symbols" % (
                heatmap_result.get("trades", 0), heatmap_result.get("symbols", 0)),
            "- **Filter PASS**: n=%d, PF=%s" % (
                heatmap_result.get("filter_pass", {}).get("n", 0),
                heatmap_result.get("filter_pass", {}).get("pf", "N/A")),
            "- **Filter FAIL**: n=%d, PF=%s" % (
                heatmap_result.get("filter_fail", {}).get("n", 0),
                heatmap_result.get("filter_fail", {}).get("pf", "N/A")),
            "",
        ]

    lines += [
        "---", "",
        "## OUTPUT FILES", "",
        "| Module | Directory |",
        "|--------|-----------|",
        "| Shadow Replay | `reports/research/replay/%s/` |" % date_str,
        "| Alpha Heatmap | `reports/research/alpha_heatmap/%s/` |" % date_str,
        "| Regime Filter | `reports/research/regime_paper_validation/%s/` |" % date_str,
        "| Scorecard | `reports/research/regime_paper_validation/regime_filter_daily_scorecard.md` |",
        "| Dashboard | `reports/research/daily_summary.md` |",
        "", "---",
        "*Pipeline completed at %s*" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "*All data READ-ONLY. NO production changes.*",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("  Dashboard written to: %s" % out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(date_str=None):
    """Run the full nightly research pipeline.

    Args:
        date_str: Target date (YYYY-MM-DD) or None for auto-detect.

    Returns:
        dict with results from each module.
    """
    start_time = _time.time()

    print("=" * 70)
    print("  NIGHTLY RESEARCH PIPELINE")
    print("  SuperBot Research Lab — Automated Validation")
    print("=" * 70)

    # Date detection
    if date_str is None or date_str == "auto":
        date_str, symbols = detect_latest_trading_day()
        if not date_str:
            print("\nERROR: No trading day data found in quote cache")
            return {"status": "NO_DATA"}
        print("\n  Auto-detected latest trading day: %s" % date_str)
        print("  Symbols with data: %s" % ", ".join(sorted(symbols)))
    else:
        dates = get_quote_cache_dates()
        symbols = dates.get(date_str, [])
        print("\n  Target date: %s" % date_str)
        if symbols:
            print("  Symbols with data: %s" % ", ".join(sorted(symbols)))
        else:
            print("  WARNING: No quote data found for %s" % date_str)

    # Load all quote caches
    print("\nLoading quote caches...")
    qcaches = {}
    for sym in symbols:
        q, e = load_quote_cache(sym)
        if q:
            qcaches[sym] = (q, e)
            print("  %s: %d quotes" % (sym, len(q)))

    if not qcaches:
        print("\nERROR: No quote data loaded — cannot proceed")
        return {"status": "NO_QUOTES", "date": date_str}

    # Output directories
    replay_dir = REPORT_ROOT / "replay" / date_str
    heatmap_dir = REPORT_ROOT / "alpha_heatmap" / date_str
    regime_dir = REPORT_ROOT / "regime_paper_validation"
    scorecard_path = regime_dir / "regime_filter_daily_scorecard.md"
    dashboard_path = REPORT_ROOT / "daily_summary.md"

    results = {"date": date_str, "symbols": symbols}

    # Module 1: Shadow Replay
    try:
        results["replay"] = run_grid_replay(date_str, symbols, qcaches, replay_dir)
    except Exception as e:
        print("  ERROR in grid replay: %s" % e)
        traceback.print_exc()
        results["replay"] = {"status": "ERROR", "error": str(e)}

    # Module 2: Alpha Heatmap
    try:
        results["heatmap"] = run_alpha_heatmap(date_str, symbols, qcaches, heatmap_dir)
    except Exception as e:
        print("  ERROR in alpha heatmap: %s" % e)
        traceback.print_exc()
        results["heatmap"] = {"status": "ERROR", "error": str(e)}

    # Module 3: Regime Filter
    try:
        results["regime"] = run_regime_validation(
            date_str, symbols, qcaches, regime_dir, scorecard_path)
    except Exception as e:
        print("  ERROR in regime validation: %s" % e)
        traceback.print_exc()
        results["regime"] = {"status": "ERROR", "error": str(e)}

    # Module 4: Dashboard
    try:
        generate_dashboard(
            date_str,
            results.get("replay"),
            results.get("heatmap"),
            results.get("regime"),
            dashboard_path,
        )
        results["dashboard"] = {"status": "OK", "path": str(dashboard_path)}
    except Exception as e:
        print("  ERROR generating dashboard: %s" % e)
        traceback.print_exc()
        results["dashboard"] = {"status": "ERROR", "error": str(e)}

    elapsed = _time.time() - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print("  Date: %s" % date_str)
    print("  Time: %.1fs" % elapsed)
    for mod in ["replay", "heatmap", "regime", "dashboard"]:
        r = results.get(mod, {})
        status = r.get("status", "SKIPPED")
        print("  %s: %s" % (mod.capitalize(), status))
    print("=" * 70)

    results["elapsed_s"] = round(elapsed, 1)
    results["status"] = "OK"

    # Save results JSON
    results_file = REPORT_ROOT / "pipeline_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Nightly Research Pipeline - automated validation")
    parser.add_argument("--date", default="auto",
                        help="Target date (YYYY-MM-DD) or 'auto' for latest")
    args = parser.parse_args()

    date_str = args.date if args.date != "auto" else None
    results = run_pipeline(date_str)

    if results.get("status") != "OK":
        print("\nPipeline failed: %s" % results.get("status"))
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
