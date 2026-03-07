"""
Alpha Heatmap Study — Morpheus_AI Market Regime Analysis
========================================================
Analyzes executed trades against market conditions at entry time to determine
which regimes Morpheus performs best in.

Features extracted per trade:
  - volatility_1m, volatility_5m (from quote tick price changes)
  - bid_ask_spread_pct (from quote bid/ask at entry)
  - relative_volume (quote rate at entry vs session average)
  - order_flow_imbalance (bid-ask tick direction proxy)
  - L2_pressure_ratio (from nearby pressure events)
  - time_of_day (market session segment)
  - symbol_price (entry price level)

Data: 2026-03-03 BATL paper trades + quote cache + pressure events
Output: reports/research/alpha_heatmap/
"""

import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # AI_Bot_Research
SUPERBOT = ROOT / "MORPHEUS_SUPERBOT"
OUT_DIR = ROOT / "reports" / "research" / "alpha_heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_TRADES = SUPERBOT / "engine" / "output" / "paper_trades.json"
QUOTE_CACHE = SUPERBOT / "engine" / "cache" / "quotes" / "BATL_quotes.json"
PRESSURE_EVENTS = SUPERBOT / "engine" / "output" / "pressure_events_2026-03-03.json"
SIGNAL_LEDGER = SUPERBOT / "engine" / "cache" / "morpheus_reports" / "2026-03-03" / "signal_ledger.jsonl"
MICRO_FEATURES = SUPERBOT / "engine" / "output" / "microstructure_features_2026-03-03.json"


# ── load data ──────────────────────────────────────────────────────────
def load_trades():
    with open(PAPER_TRADES, encoding="utf-8") as f:
        data = json.load(f)
    return data["trades"]


def load_quotes():
    with open(QUOTE_CACHE, encoding="utf-8") as f:
        data = json.load(f)
    return data["quotes"]


def load_pressure_events():
    with open(PRESSURE_EVENTS, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events_per_symbol", {}).get("BATL", {}).get("events", [])


def load_ignition_momentum():
    """Load momentum snapshots from signal ledger IGNITION_PASS/FAIL entries."""
    entries = []
    with open(SIGNAL_LEDGER, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("symbol") == "BATL" and e.get("momentum_snapshot"):
                entries.append(e)
    return entries


# ── feature extraction ─────────────────────────────────────────────────
def find_quotes_window(quotes, epoch, window_s, side="before"):
    """Find quotes within window_s seconds before/after epoch."""
    result = []
    for q in quotes:
        dt = q["epoch"] - epoch
        if side == "before" and -window_s <= dt <= 0:
            result.append(q)
        elif side == "after" and 0 <= dt <= window_s:
            result.append(q)
        elif side == "around" and -window_s <= dt <= window_s:
            result.append(q)
    return result


def compute_volatility(quotes, epoch, window_s):
    """Compute price volatility (std of log returns) over window before entry."""
    window = find_quotes_window(quotes, epoch, window_s, "before")
    if len(window) < 3:
        return None
    prices = [q["last"] for q in window if q.get("last")]
    if len(prices) < 3:
        return None
    log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))
                   if prices[i] > 0 and prices[i - 1] > 0]
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    var = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    return math.sqrt(var) * 100  # percentage


def compute_spread_at_entry(quotes, epoch):
    """Compute bid-ask spread at entry time (closest quote)."""
    best = None
    best_dt = float("inf")
    for q in quotes:
        dt = abs(q["epoch"] - epoch)
        if dt < best_dt:
            best_dt = dt
            best = q
        if q["epoch"] > epoch + 5:
            break
    if best and best.get("bid") and best.get("ask") and best["bid"] > 0:
        mid = (best["bid"] + best["ask"]) / 2
        return (best["ask"] - best["bid"]) / mid * 100
    return None


def compute_relative_volume(quotes, epoch, window_s=60):
    """Compute quote rate in window vs session average as relative volume proxy."""
    window = find_quotes_window(quotes, epoch, window_s, "before")
    if not window or not quotes:
        return None
    entry_rate = len(window) / window_s
    # session average: total quotes / total time span
    total_span = quotes[-1]["epoch"] - quotes[0]["epoch"]
    if total_span <= 0:
        return None
    avg_rate = len(quotes) / total_span
    if avg_rate <= 0:
        return None
    return entry_rate / avg_rate


def compute_order_flow_proxy(quotes, epoch, window_s=30):
    """
    Proxy for order flow imbalance using tick direction.
    Count upticks vs downticks in the window before entry.
    """
    window = find_quotes_window(quotes, epoch, window_s, "before")
    if len(window) < 5:
        return None
    ups = 0
    downs = 0
    for i in range(1, len(window)):
        p1 = window[i].get("last", 0)
        p0 = window[i - 1].get("last", 0)
        if p1 > p0:
            ups += 1
        elif p1 < p0:
            downs += 1
    total = ups + downs
    if total == 0:
        return 0.0
    return (ups - downs) / total  # -1.0 to +1.0


def find_nearest_pressure(pressure_events, epoch, max_dt=30):
    """Find pressure event closest to trade entry within max_dt seconds."""
    best = None
    best_dt = float("inf")
    for pe in pressure_events:
        dt = abs(pe["epoch"] - epoch)
        if dt < best_dt and dt <= max_dt:
            best_dt = dt
            best = pe
    return best


def classify_time_of_day(epoch):
    """Classify entry time into market session segment."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    # Convert to ET (UTC-4 for EDT in March 2026)
    hour_et = dt.hour - 4
    minute = dt.minute
    time_mins = hour_et * 60 + minute

    if time_mins < 570:  # before 9:30
        return "premarket"
    elif time_mins < 630:  # 9:30-10:30
        return "open"
    elif time_mins < 810:  # 10:30-13:30
        return "midday"
    else:  # 13:30-16:00
        return "power_hour"


def classify_price(price):
    """Classify price into category."""
    if price < 2:
        return "sub-$2"
    elif price < 5:
        return "$2-$5"
    elif price < 20:
        return "$5-$20"
    else:
        return "$20+"


def bin_volatility(vol):
    if vol is None:
        return "unknown"
    if vol < 0.3:
        return "low"
    elif vol < 0.8:
        return "medium"
    else:
        return "high"


def bin_spread(spread):
    if spread is None:
        return "unknown"
    if spread < 0.3:
        return "<0.3%"
    elif spread < 0.6:
        return "0.3-0.6%"
    elif spread < 1.0:
        return "0.6-1.0%"
    else:
        return ">1.0%"


def bin_order_flow(ofi):
    if ofi is None:
        return "unknown"
    if ofi < -0.2:
        return "weak"
    elif ofi < 0.3:
        return "moderate"
    else:
        return "strong"


def bin_relative_volume(rv):
    if rv is None:
        return "unknown"
    if rv < 0.5:
        return "low"
    elif rv < 1.5:
        return "average"
    else:
        return "high"


# ── metrics computation ────────────────────────────────────────────────
def compute_metrics(trades_in_bin):
    """Compute PF, avg PnL, win rate for a group of trades."""
    if not trades_in_bin:
        return {"n": 0, "pf": None, "avg_pnl": None, "wr": None, "total_pnl": 0}
    wins = sum(1 for t in trades_in_bin if t["pnl"] > 0)
    gross_wins = sum(t["pnl"] for t in trades_in_bin if t["pnl"] > 0)
    gross_losses = sum(abs(t["pnl"]) for t in trades_in_bin if t["pnl"] <= 0)
    pf = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0)
    total_pnl = sum(t["pnl"] for t in trades_in_bin)
    return {
        "n": len(trades_in_bin),
        "pf": round(pf, 2) if pf != float("inf") else "INF",
        "avg_pnl": round(total_pnl / len(trades_in_bin), 2),
        "wr": round(wins / len(trades_in_bin) * 100, 1),
        "total_pnl": round(total_pnl, 2),
    }


# ── heatmap builder ───────────────────────────────────────────────────
def build_2d_heatmap(enriched_trades, row_key, col_key, row_bins, col_bins):
    """Build a 2D heatmap matrix for given row/col feature keys."""
    matrix = {}
    for rb in row_bins:
        matrix[rb] = {}
        for cb in col_bins:
            matrix[rb][cb] = []

    for t in enriched_trades:
        rv = t["features"].get(row_key, "unknown")
        cv = t["features"].get(col_key, "unknown")
        if rv in matrix and cv in matrix.get(rv, {}):
            matrix[rv][cv].append(t)

    result = {}
    for rb in row_bins:
        result[rb] = {}
        for cb in col_bins:
            result[rb][cb] = compute_metrics(matrix[rb][cb])
    return result


def format_heatmap_md(title, row_label, col_label, matrix, row_bins, col_bins, metric="pf"):
    """Format a heatmap matrix as a markdown table."""
    lines = [f"### {title}\n"]

    # Header
    header = f"| {row_label} \\\\ {col_label} |"
    for cb in col_bins:
        header += f" {cb} |"
    lines.append(header)
    lines.append("|" + "---|" * (len(col_bins) + 1))

    for rb in row_bins:
        row = f"| **{rb}** |"
        for cb in col_bins:
            m = matrix[rb][cb]
            if m["n"] == 0:
                row += " - |"
            elif metric == "pf":
                row += f" PF={m['pf']} (n={m['n']}) |"
            elif metric == "avg_pnl":
                row += f" ${m['avg_pnl']:+,.0f} (n={m['n']}) |"
            elif metric == "wr":
                row += f" {m['wr']}% (n={m['n']}) |"
            else:
                row += f" {m[metric]} |"
        lines.append(row)

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ALPHA HEATMAP STUDY — Morpheus_AI Market Regime Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    trades = load_trades()
    quotes = load_quotes()
    pressure_events = load_pressure_events()
    print(f"  {len(trades)} trades, {len(quotes)} quotes, {len(pressure_events)} pressure events")

    # Pre-sort quotes by epoch for faster lookups
    quotes.sort(key=lambda q: q["epoch"])
    quote_epochs = [q["epoch"] for q in quotes]

    # Binary search helper
    import bisect

    def get_quotes_window_fast(epoch, window_s):
        """Fast window lookup using bisect."""
        lo = bisect.bisect_left(quote_epochs, epoch - window_s)
        hi = bisect.bisect_right(quote_epochs, epoch)
        return quotes[lo:hi]

    def get_quotes_around_fast(epoch, window_s):
        lo = bisect.bisect_left(quote_epochs, epoch - window_s)
        hi = bisect.bisect_right(quote_epochs, epoch + window_s)
        return quotes[lo:hi]

    # ── Extract features for each trade ────────────────────────────────
    print("\nExtracting features...")
    enriched_trades = []

    for i, trade in enumerate(trades):
        epoch = trade["entry_epoch"]
        price = trade["entry_price"]

        # Get quote windows
        window_1m = get_quotes_window_fast(epoch, 60)
        window_5m = get_quotes_window_fast(epoch, 300)

        # Volatility
        vol_1m = compute_volatility(quotes, epoch, 60)
        vol_5m = compute_volatility(quotes, epoch, 300)

        # Spread
        spread = compute_spread_at_entry(quotes, epoch)

        # Relative volume
        rel_vol = compute_relative_volume(quotes, epoch, 60)

        # Order flow
        ofi = compute_order_flow_proxy(quotes, epoch, 30)

        # Pressure (L2 proxy)
        pe = find_nearest_pressure(pressure_events, epoch, 30)
        l2_pressure = None
        pressure_conditions = 0
        if pe:
            pressure_conditions = pe.get("conditions_met", 0)
            conds = pe.get("conditions", {})
            # Composite L2 pressure from available conditions
            scores = []
            if "bid_stepping" in conds:
                bs = conds["bid_stepping"]
                scores.append(min(bs["steps"] / max(bs["threshold"], 1), 2.0))
            if "spread_compression" in conds:
                sc = conds["spread_compression"]
                scores.append(1.0 - sc.get("ratio", 1.0))  # lower ratio = more compression = stronger
            if "trade_imbalance" in conds:
                ti = conds["trade_imbalance"]
                scores.append(min(ti.get("ratio", 1.0) / max(ti.get("threshold", 1.8), 1), 2.0))
            l2_pressure = sum(scores) / len(scores) if scores else None

        # Time of day
        tod = classify_time_of_day(epoch)

        # Price class
        price_class = classify_price(price)

        # Binned features
        features = {
            "volatility_1m": vol_1m,
            "volatility_5m": vol_5m,
            "vol_bin": bin_volatility(vol_1m),
            "spread_pct": spread,
            "spread_bin": bin_spread(spread),
            "relative_volume": rel_vol,
            "rvol_bin": bin_relative_volume(rel_vol),
            "order_flow": ofi,
            "ofi_bin": bin_order_flow(ofi),
            "l2_pressure": l2_pressure,
            "pressure_conditions": pressure_conditions,
            "time_of_day": tod,
            "price_class": price_class,
            "regime": trade["regime"],
            "trap_probability": trade["trap_probability"],
            "pressure_score": trade["pressure_score"],
        }

        enriched = {**trade, "features": features, "trade_idx": i + 1}
        enriched_trades.append(enriched)

        print(f"  Trade {i+1:2d}: entry=${price:.2f} | vol_1m={vol_1m:.4f}" if vol_1m else
              f"  Trade {i+1:2d}: entry=${price:.2f} | vol_1m=N/A", end="")
        print(f" | spread={spread:.3f}%" if spread else " | spread=N/A", end="")
        print(f" | ofi={ofi:+.2f}" if ofi is not None else " | ofi=N/A", end="")
        print(f" | rvol={rel_vol:.2f}" if rel_vol else " | rvol=N/A", end="")
        print(f" | {tod} | pnl=${trade['pnl']:+,.0f}")

    # ── Build heatmaps ─────────────────────────────────────────────────
    print("\nBuilding heatmaps...")

    vol_bins = ["low", "medium", "high"]
    spread_bins = ["<0.3%", "0.3-0.6%", "0.6-1.0%", ">1.0%"]
    ofi_bins = ["weak", "moderate", "strong"]
    rvol_bins = ["low", "average", "high"]
    tod_bins = ["premarket", "open", "midday", "power_hour"]
    price_bins = ["sub-$2", "$2-$5", "$5-$20", "$20+"]
    regime_bins = ["RANGE_BOUND", "LOW_VOLATILITY"]

    # Heatmap 1: Volatility vs Spread
    vol_spread = build_2d_heatmap(enriched_trades, "vol_bin", "spread_bin", vol_bins, spread_bins)

    # Heatmap 2: Order Flow vs Spread
    ofi_spread = build_2d_heatmap(enriched_trades, "ofi_bin", "spread_bin", ofi_bins, spread_bins)

    # Heatmap 3: Volatility vs Order Flow
    vol_ofi = build_2d_heatmap(enriched_trades, "vol_bin", "ofi_bin", vol_bins, ofi_bins)

    # Heatmap 4: Regime vs Volatility
    regime_vol = build_2d_heatmap(enriched_trades, "regime", "vol_bin", regime_bins, vol_bins)

    # Heatmap 5: Relative Volume vs Spread
    rvol_spread = build_2d_heatmap(enriched_trades, "rvol_bin", "spread_bin", rvol_bins, spread_bins)

    # ── Write Report 1: volatility_spread_heatmap.md ───────────────────
    print("  Writing volatility_spread_heatmap.md...")
    lines = [
        "# Volatility vs Spread Alpha Heatmap",
        "## Data: 2026-03-03 BATL | 20 Executed Trades",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## Feature Definitions",
        "",
        "| Feature | Source | Bins |",
        "|---------|--------|------|",
        "| Volatility (1m) | Std of log returns, 60s window before entry | low (<0.3%), medium (0.3-0.8%), high (>0.8%) |",
        "| Spread | Bid-ask spread at closest quote to entry | <0.3%, 0.3-0.6%, 0.6-1.0%, >1.0% |",
        "",
        "---",
        "",
        "## Profit Factor Heatmap",
        "",
    ]
    lines.append(format_heatmap_md("Volatility vs Spread — Profit Factor",
                                    "Volatility", "Spread", vol_spread, vol_bins, spread_bins, "pf"))
    lines.extend(["", "## Average PnL Heatmap", ""])
    lines.append(format_heatmap_md("Volatility vs Spread — Avg PnL",
                                    "Volatility", "Spread", vol_spread, vol_bins, spread_bins, "avg_pnl"))
    lines.extend(["", "## Win Rate Heatmap", ""])
    lines.append(format_heatmap_md("Volatility vs Spread — Win Rate",
                                    "Volatility", "Spread", vol_spread, vol_bins, spread_bins, "wr"))

    # Add regime vs volatility
    lines.extend([
        "", "---", "",
        "## Regime vs Volatility",
        "",
    ])
    lines.append(format_heatmap_md("Regime vs Volatility — Profit Factor",
                                    "Regime", "Volatility", regime_vol, regime_bins, vol_bins, "pf"))
    lines.extend(["", ""])
    lines.append(format_heatmap_md("Regime vs Volatility — Avg PnL",
                                    "Regime", "Volatility", regime_vol, regime_bins, vol_bins, "avg_pnl"))

    # Add relative volume vs spread
    lines.extend([
        "", "---", "",
        "## Relative Volume vs Spread",
        "",
    ])
    lines.append(format_heatmap_md("Relative Volume vs Spread — Profit Factor",
                                    "Rel Volume", "Spread", rvol_spread, rvol_bins, spread_bins, "pf"))

    # Interpretation
    lines.extend([
        "", "---", "",
        "## Key Observations",
        "",
    ])

    # Find best and worst cells
    best_cell = None
    worst_cell = None
    for rb in vol_bins:
        for cb in spread_bins:
            m = vol_spread[rb][cb]
            if m["n"] >= 2:
                if best_cell is None or (m["avg_pnl"] or 0) > (best_cell[2]["avg_pnl"] or 0):
                    best_cell = (rb, cb, m)
                if worst_cell is None or (m["avg_pnl"] or 0) < (worst_cell[2]["avg_pnl"] or 0):
                    worst_cell = (rb, cb, m)

    if best_cell:
        lines.append(f"- **Best regime**: {best_cell[0]} volatility + {best_cell[1]} spread "
                      f"(n={best_cell[2]['n']}, PF={best_cell[2]['pf']}, avg=${best_cell[2]['avg_pnl']:+,.0f})")
    if worst_cell:
        lines.append(f"- **Worst regime**: {worst_cell[0]} volatility + {worst_cell[1]} spread "
                      f"(n={worst_cell[2]['n']}, PF={worst_cell[2]['pf']}, avg=${worst_cell[2]['avg_pnl']:+,.0f})")

    lines.extend([
        "", "---", "",
        "*Data source: paper_trades.json, BATL_quotes.json (READ-ONLY)*",
        "*NO production changes were made.*",
    ])

    with open(OUT_DIR / "volatility_spread_heatmap.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Write Report 2: orderflow_spread_heatmap.md ────────────────────
    print("  Writing orderflow_spread_heatmap.md...")
    lines = [
        "# Order Flow vs Spread Alpha Heatmap",
        "## Data: 2026-03-03 BATL | 20 Executed Trades",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## Feature Definitions",
        "",
        "| Feature | Source | Bins |",
        "|---------|--------|------|",
        "| Order Flow Imbalance | Tick direction proxy: (upticks-downticks)/total, 30s window | weak (<-0.2), moderate (-0.2 to +0.3), strong (>+0.3) |",
        "| Spread | Bid-ask spread at closest quote to entry | <0.3%, 0.3-0.6%, 0.6-1.0%, >1.0% |",
        "",
        "---",
        "",
        "## Profit Factor Heatmap",
        "",
    ]
    lines.append(format_heatmap_md("Order Flow vs Spread — Profit Factor",
                                    "Order Flow", "Spread", ofi_spread, ofi_bins, spread_bins, "pf"))
    lines.extend(["", "## Average PnL Heatmap", ""])
    lines.append(format_heatmap_md("Order Flow vs Spread — Avg PnL",
                                    "Order Flow", "Spread", ofi_spread, ofi_bins, spread_bins, "avg_pnl"))
    lines.extend(["", "## Win Rate Heatmap", ""])
    lines.append(format_heatmap_md("Order Flow vs Spread — Win Rate",
                                    "Order Flow", "Spread", ofi_spread, ofi_bins, spread_bins, "wr"))

    # Volatility vs Order Flow
    lines.extend([
        "", "---", "",
        "## Volatility vs Order Flow",
        "",
    ])
    lines.append(format_heatmap_md("Volatility vs Order Flow — Profit Factor",
                                    "Volatility", "Order Flow", vol_ofi, vol_bins, ofi_bins, "pf"))
    lines.extend(["", ""])
    lines.append(format_heatmap_md("Volatility vs Order Flow — Avg PnL",
                                    "Volatility", "Order Flow", vol_ofi, vol_bins, ofi_bins, "avg_pnl"))

    # Find best order flow regimes
    lines.extend(["", "---", "", "## Key Observations", ""])
    best_ofi = None
    for rb in ofi_bins:
        for cb in spread_bins:
            m = ofi_spread[rb][cb]
            if m["n"] >= 2:
                if best_ofi is None or (m["avg_pnl"] or 0) > (best_ofi[2]["avg_pnl"] or 0):
                    best_ofi = (rb, cb, m)
    if best_ofi:
        lines.append(f"- **Best regime**: {best_ofi[0]} order flow + {best_ofi[1]} spread "
                      f"(n={best_ofi[2]['n']}, PF={best_ofi[2]['pf']}, avg=${best_ofi[2]['avg_pnl']:+,.0f})")

    lines.extend([
        "", "---", "",
        "*Data source: paper_trades.json, BATL_quotes.json (READ-ONLY)*",
        "*NO production changes were made.*",
    ])

    with open(OUT_DIR / "orderflow_spread_heatmap.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Write Report 3: time_of_day_performance.md ─────────────────────
    print("  Writing time_of_day_performance.md...")

    # Group trades by time of day
    tod_groups = defaultdict(list)
    for t in enriched_trades:
        tod_groups[t["features"]["time_of_day"]].append(t)

    tod_metrics = {}
    for segment in tod_bins:
        tod_metrics[segment] = compute_metrics(tod_groups[segment])

    # Also compute per 15-min window
    minute_groups = defaultdict(list)
    for t in enriched_trades:
        dt = datetime.fromtimestamp(t["entry_epoch"], tz=timezone.utc)
        hour_et = dt.hour - 4
        minute = dt.minute
        bucket = f"{hour_et:02d}:{(minute // 15) * 15:02d}"
        minute_groups[bucket].append(t)

    lines = [
        "# Time-of-Day Performance Analysis",
        "## Data: 2026-03-03 BATL | 20 Executed Trades",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## Session Segment Performance",
        "",
        "| Segment | Time Window | Trades | Win Rate | Profit Factor | Avg PnL | Total PnL |",
        "|---------|-------------|--------|----------|--------------|---------|-----------|",
    ]

    segment_times = {
        "premarket": "Pre-9:30 ET",
        "open": "9:30-10:30 ET",
        "midday": "10:30-13:30 ET",
        "power_hour": "13:30-16:00 ET",
    }

    for seg in tod_bins:
        m = tod_metrics[seg]
        if m["n"] > 0:
            lines.append(f"| **{seg}** | {segment_times[seg]} | {m['n']} | "
                          f"{m['wr']}% | {m['pf']} | ${m['avg_pnl']:+,.0f} | ${m['total_pnl']:+,.0f} |")
        else:
            lines.append(f"| **{seg}** | {segment_times[seg]} | 0 | - | - | - | - |")

    lines.extend([
        "",
        "---",
        "",
        "## 15-Minute Window Breakdown",
        "",
        "| Window (ET) | Trades | Win Rate | Avg PnL | Total PnL |",
        "|-------------|--------|----------|---------|-----------|",
    ])

    for bucket in sorted(minute_groups.keys()):
        m = compute_metrics(minute_groups[bucket])
        lines.append(f"| {bucket} | {m['n']} | {m['wr']}% | ${m['avg_pnl']:+,.0f} | ${m['total_pnl']:+,.0f} |")

    # Trade timeline
    lines.extend([
        "",
        "---",
        "",
        "## Trade Timeline",
        "",
        "| # | Time (ET) | Entry Price | PnL | Regime | Result |",
        "|---|-----------|-------------|-----|--------|--------|",
    ])

    for t in enriched_trades:
        dt = datetime.fromtimestamp(t["entry_epoch"], tz=timezone.utc)
        time_et = f"{dt.hour - 4:02d}:{dt.minute:02d}:{dt.second:02d}"
        result = "WIN" if t["pnl"] > 0 else "LOSS"
        lines.append(f"| {t['trade_idx']} | {time_et} | ${t['entry_price']:.2f} | "
                      f"${t['pnl']:+,.0f} | {t['regime']} | {result} |")

    # Find best and worst segments
    lines.extend(["", "---", "", "## Key Observations", ""])

    best_seg = max((s for s in tod_bins if tod_metrics[s]["n"] > 0),
                   key=lambda s: tod_metrics[s]["avg_pnl"] or -999, default=None)
    worst_seg = min((s for s in tod_bins if tod_metrics[s]["n"] > 0),
                    key=lambda s: tod_metrics[s]["avg_pnl"] or 999, default=None)

    if best_seg:
        m = tod_metrics[best_seg]
        lines.append(f"- **Best session**: {best_seg} ({segment_times[best_seg]}) — "
                      f"PF={m['pf']}, WR={m['wr']}%, avg=${m['avg_pnl']:+,.0f}")
    if worst_seg and worst_seg != best_seg:
        m = tod_metrics[worst_seg]
        lines.append(f"- **Worst session**: {worst_seg} ({segment_times[worst_seg]}) — "
                      f"PF={m['pf']}, WR={m['wr']}%, avg=${m['avg_pnl']:+,.0f}")

    lines.extend([
        "", "---", "",
        "*Data source: paper_trades.json (READ-ONLY)*",
        "*NO production changes were made.*",
    ])

    with open(OUT_DIR / "time_of_day_performance.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Write Report 4: price_class_analysis.md ────────────────────────
    print("  Writing price_class_analysis.md...")

    price_groups = defaultdict(list)
    for t in enriched_trades:
        price_groups[t["features"]["price_class"]].append(t)

    price_metrics = {}
    for pc in price_bins:
        price_metrics[pc] = compute_metrics(price_groups[pc])

    # Also compute dollar-range buckets for BATL's intraday range
    dollar_groups = defaultdict(list)
    for t in enriched_trades:
        p = t["entry_price"]
        if p < 18:
            dollar_groups["$17-18"].append(t)
        elif p < 19:
            dollar_groups["$18-19"].append(t)
        elif p < 20:
            dollar_groups["$19-20"].append(t)
        elif p < 21:
            dollar_groups["$20-21"].append(t)
        else:
            dollar_groups["$21+"].append(t)

    lines = [
        "# Price Class Alpha Analysis",
        "## Data: 2026-03-03 BATL | 20 Executed Trades",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## Standard Price Categories",
        "",
        "| Price Class | Trades | Win Rate | Profit Factor | Avg PnL | Total PnL |",
        "|-------------|--------|----------|--------------|---------|-----------|",
    ]

    for pc in price_bins:
        m = price_metrics[pc]
        if m["n"] > 0:
            lines.append(f"| **{pc}** | {m['n']} | {m['wr']}% | {m['pf']} | "
                          f"${m['avg_pnl']:+,.0f} | ${m['total_pnl']:+,.0f} |")
        else:
            lines.append(f"| **{pc}** | 0 | - | - | - | - |")

    lines.extend([
        "",
        "**Note**: All 20 trades are BATL ($17.86-$20.65 range). Standard price class",
        "analysis is limited to the price categories BATL occupies.",
        "",
        "---",
        "",
        "## BATL Intraday Price Level Performance",
        "",
        "| Price Range | Trades | Win Rate | Profit Factor | Avg PnL | Total PnL | Avg MFE | Avg MAE |",
        "|-------------|--------|----------|--------------|---------|-----------|---------|---------|",
    ])

    for dr in sorted(dollar_groups.keys()):
        group = dollar_groups[dr]
        m = compute_metrics(group)
        avg_mfe = sum(t["mfe_pct"] for t in group) / len(group)
        avg_mae = sum(t["mae_pct"] for t in group) / len(group)
        lines.append(f"| **{dr}** | {m['n']} | {m['wr']}% | {m['pf']} | "
                      f"${m['avg_pnl']:+,.0f} | ${m['total_pnl']:+,.0f} | "
                      f"{avg_mfe:.2f}% | {avg_mae:.2f}% |")

    # Per-trade detail
    lines.extend([
        "",
        "---",
        "",
        "## Per-Trade Entry Price vs Outcome",
        "",
        "| # | Entry Price | PnL | Return% | MFE% | MAE% | Hold(s) | Regime |",
        "|---|-------------|-----|---------|------|------|---------|--------|",
    ])

    for t in enriched_trades:
        lines.append(f"| {t['trade_idx']} | ${t['entry_price']:.2f} | ${t['pnl']:+,.0f} | "
                      f"{t['return_pct']:+.2f}% | {t['mfe_pct']:.2f}% | {t['mae_pct']:.2f}% | "
                      f"{t['hold_s']:.0f} | {t['regime']} |")

    # Price momentum analysis
    lines.extend(["", "---", "", "## Price vs Alpha Relationship", ""])

    # Early trades (lower price) vs late trades (higher price)
    early = [t for t in enriched_trades if t["entry_price"] < 19]
    late = [t for t in enriched_trades if t["entry_price"] >= 20]
    m_early = compute_metrics(early)
    m_late = compute_metrics(late)

    lines.extend([
        "| Phase | Price Range | Trades | WR | PF | Avg PnL |",
        "|-------|------------|--------|-----|-----|---------|",
        f"| Early (low price) | <$19 | {m_early['n']} | {m_early['wr']}% | {m_early['pf']} | ${m_early['avg_pnl']:+,.0f} |",
        f"| Late (high price) | >=$20 | {m_late['n']} | {m_late['wr']}% | {m_late['pf']} | ${m_late['avg_pnl']:+,.0f} |",
    ])

    lines.extend([
        "", "---", "",
        "## Key Observations", "",
    ])

    best_dr = max(dollar_groups.keys(), key=lambda k: compute_metrics(dollar_groups[k])["avg_pnl"] or -999)
    m = compute_metrics(dollar_groups[best_dr])
    lines.append(f"- **Best price level**: {best_dr} — PF={m['pf']}, avg=${m['avg_pnl']:+,.0f}")

    worst_dr = min(dollar_groups.keys(), key=lambda k: compute_metrics(dollar_groups[k])["avg_pnl"] or 999)
    m = compute_metrics(dollar_groups[worst_dr])
    lines.append(f"- **Worst price level**: {worst_dr} — PF={m['pf']}, avg=${m['avg_pnl']:+,.0f}")

    lines.extend([
        "", "---", "",
        "*Data source: paper_trades.json (READ-ONLY)*",
        "*NO production changes were made.*",
    ])

    with open(OUT_DIR / "price_class_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Write Report 5: alpha_heatmap_summary.md ───────────────────────
    print("  Writing alpha_heatmap_summary.md...")

    # Collect all regime combos and rank them
    all_combos = []

    # Vol x Spread combos
    for rb in vol_bins:
        for cb in spread_bins:
            m = vol_spread[rb][cb]
            if m["n"] >= 1:
                all_combos.append({
                    "regime": f"{rb} vol + {cb} spread",
                    "dims": "vol_x_spread",
                    **m,
                })

    # OFI x Spread combos
    for rb in ofi_bins:
        for cb in spread_bins:
            m = ofi_spread[rb][cb]
            if m["n"] >= 1:
                all_combos.append({
                    "regime": f"{rb} OFI + {cb} spread",
                    "dims": "ofi_x_spread",
                    **m,
                })

    # Vol x OFI combos
    for rb in vol_bins:
        for cb in ofi_bins:
            m = vol_ofi[rb][cb]
            if m["n"] >= 1:
                all_combos.append({
                    "regime": f"{rb} vol + {cb} OFI",
                    "dims": "vol_x_ofi",
                    **m,
                })

    # Sort by avg_pnl
    profitable = sorted([c for c in all_combos if c["n"] >= 2],
                         key=lambda c: c["avg_pnl"] or 0, reverse=True)
    losing = sorted([c for c in all_combos if c["n"] >= 2],
                     key=lambda c: c["avg_pnl"] or 0)

    lines = [
        "# Alpha Heatmap Summary — Morpheus_AI Regime Analysis",
        "## Data: 2026-03-03 BATL | 20 Executed Trades",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## IMPORTANT CAVEAT",
        "",
        "This analysis is based on **20 trades on a single symbol (BATL) on a single day**",
        "(+87% intraday move — extreme outlier). Statistical significance is limited.",
        "Results should be validated across multiple days and symbols before being used",
        "as production filter rules.",
        "",
        "---",
        "",
        "## Top 3 Profitable Regimes",
        "",
        "| Rank | Regime | Trades | Win Rate | Profit Factor | Avg PnL |",
        "|------|--------|--------|----------|--------------|---------|",
    ]

    for i, c in enumerate(profitable[:3]):
        lines.append(f"| {i+1} | **{c['regime']}** | {c['n']} | {c['wr']}% | "
                      f"{c['pf']} | ${c['avg_pnl']:+,.0f} |")

    lines.extend([
        "",
        "## Worst Regimes (Where Morpheus Loses Money)",
        "",
        "| Rank | Regime | Trades | Win Rate | Profit Factor | Avg PnL |",
        "|------|--------|--------|----------|--------------|---------|",
    ])

    for i, c in enumerate(losing[:3]):
        lines.append(f"| {i+1} | **{c['regime']}** | {c['n']} | {c['wr']}% | "
                      f"{c['pf']} | ${c['avg_pnl']:+,.0f} |")

    # Time of day summary
    lines.extend([
        "",
        "---",
        "",
        "## Time-of-Day Summary",
        "",
        "| Segment | Trades | WR | PF | Avg PnL | Total PnL |",
        "|---------|--------|-----|-----|---------|-----------|",
    ])

    for seg in tod_bins:
        m = tod_metrics[seg]
        if m["n"] > 0:
            lines.append(f"| {seg} | {m['n']} | {m['wr']}% | {m['pf']} | "
                          f"${m['avg_pnl']:+,.0f} | ${m['total_pnl']:+,.0f} |")

    # Price class summary
    lines.extend([
        "",
        "## Price Level Summary",
        "",
        "| Phase | Trades | WR | PF | Avg PnL |",
        "|-------|--------|-----|-----|---------|",
        f"| Early (<$19) | {m_early['n']} | {m_early['wr']}% | {m_early['pf']} | ${m_early['avg_pnl']:+,.0f} |",
        f"| Late (>=$20) | {m_late['n']} | {m_late['wr']}% | {m_late['pf']} | ${m_late['avg_pnl']:+,.0f} |",
    ])

    # Regime from paper trades
    regime_groups = defaultdict(list)
    for t in enriched_trades:
        regime_groups[t["regime"]].append(t)

    lines.extend([
        "",
        "## Market Regime Summary (from Trade Labels)",
        "",
        "| Regime | Trades | WR | PF | Avg PnL | Total PnL |",
        "|--------|--------|-----|-----|---------|-----------|",
    ])

    for regime in sorted(regime_groups.keys()):
        m = compute_metrics(regime_groups[regime])
        lines.append(f"| {regime} | {m['n']} | {m['wr']}% | {m['pf']} | "
                      f"${m['avg_pnl']:+,.0f} | ${m['total_pnl']:+,.0f} |")

    # Feature summary table
    lines.extend([
        "",
        "---",
        "",
        "## Per-Trade Feature Matrix",
        "",
        "| # | Price | Vol_1m | Spread% | OFI | RVol | Time | Regime | PnL |",
        "|---|-------|--------|---------|-----|------|------|--------|-----|",
    ])

    for t in enriched_trades:
        f = t["features"]
        v1m = f"{f['volatility_1m']:.3f}" if f["volatility_1m"] else "N/A"
        sp = f"{f['spread_pct']:.3f}" if f["spread_pct"] else "N/A"
        ofi_v = f"{f['order_flow']:+.2f}" if f["order_flow"] is not None else "N/A"
        rv = f"{f['relative_volume']:.2f}" if f["relative_volume"] else "N/A"
        lines.append(f"| {t['trade_idx']} | ${t['entry_price']:.2f} | {v1m} | {sp} | "
                      f"{ofi_v} | {rv} | {f['time_of_day']} | {t['regime']} | ${t['pnl']:+,.0f} |")

    # Recommended regime filter rules
    lines.extend([
        "",
        "---",
        "",
        "## Recommended Regime Filter Rules",
        "",
        "Based on this analysis, the following filters would improve expected PnL:",
        "",
        "```",
        "ONLY trade when:",
    ])

    # Derive rules from data
    # Volatility
    vol_perf = {}
    for vb in vol_bins:
        trades_in = [t for t in enriched_trades if t["features"]["vol_bin"] == vb]
        if trades_in:
            vol_perf[vb] = compute_metrics(trades_in)

    best_vol = max(vol_perf.keys(), key=lambda k: vol_perf[k]["avg_pnl"] or -999) if vol_perf else None

    # Spread
    spread_perf = {}
    for sb in spread_bins:
        trades_in = [t for t in enriched_trades if t["features"]["spread_bin"] == sb]
        if trades_in:
            spread_perf[sb] = compute_metrics(trades_in)

    # OFI
    ofi_perf = {}
    for ob in ofi_bins:
        trades_in = [t for t in enriched_trades if t["features"]["ofi_bin"] == ob]
        if trades_in:
            ofi_perf[ob] = compute_metrics(trades_in)

    best_ofi_bin = max(ofi_perf.keys(), key=lambda k: ofi_perf[k]["avg_pnl"] or -999) if ofi_perf else None

    if best_vol:
        lines.append(f"  volatility >= {best_vol}")
    lines.append("  spread <= 0.6%")
    if best_ofi_bin:
        lines.append(f"  order_flow_intensity >= {best_ofi_bin}")

    # Time filter
    if best_seg:
        lines.append(f"  session = {best_seg} (preferred)")

    lines.extend([
        "```",
        "",
        "### Confidence Assessment",
        "",
        "| Rule | Confidence | Basis |",
        "|------|-----------|-------|",
    ])

    if best_vol:
        lines.append(f"| volatility >= {best_vol} | LOW | Only {vol_perf[best_vol]['n']} trades in bin |")

    lines.extend([
        "| spread <= 0.6% | MEDIUM | Aligns with containment study (0.4-0.6% sweet spot) |",
    ])

    if best_ofi_bin:
        lines.append(f"| OFI >= {best_ofi_bin} | LOW | Tick direction proxy, not true L2 data |")

    if best_seg:
        lines.append(f"| session = {best_seg} | MEDIUM | {tod_metrics[best_seg]['n']} trades, "
                      f"but single-day data |")

    lines.extend([
        "",
        "**Overall confidence: LOW** — 20 trades on 1 symbol on 1 extreme day is insufficient",
        "for production filter rules. These are hypotheses to validate, not rules to deploy.",
        "",
        "---",
        "",
        "## Research Reports Generated",
        "",
        "| # | Report | Key Finding |",
        "|---|--------|-------------|",
        "| 1 | `volatility_spread_heatmap.md` | Volatility vs spread regime matrix |",
        "| 2 | `orderflow_spread_heatmap.md` | Order flow vs spread regime matrix |",
        "| 3 | `time_of_day_performance.md` | Session segment PnL breakdown |",
        "| 4 | `price_class_analysis.md` | Price level alpha analysis |",
        "| 5 | `alpha_heatmap_summary.md` | This file |",
        "",
        "---",
        "",
        "*All data sources accessed READ-ONLY. NO production changes were made.*",
        "*Script: ai/research/alpha_heatmap_study.py*",
    ])

    with open(OUT_DIR / "alpha_heatmap_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 70)
    print("ALPHA HEATMAP STUDY COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Files generated:")
    for fn in sorted(OUT_DIR.iterdir()):
        print(f"  {fn.name}")


if __name__ == "__main__":
    main()
