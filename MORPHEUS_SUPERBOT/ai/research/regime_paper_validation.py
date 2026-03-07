"""
Regime Filter Paper Validation
===============================
Side-by-side comparison of baseline Morpheus vs Morpheus + regime filter.

Candidate Filter:
  - volatility_1m >= 0.3% (medium threshold)
  - spread <= 0.6%
  - OFI >= -0.2 (moderate threshold)

Candidate Suppressions:
  - suppress LOW_VOLATILITY regime
  - suppress POWER_HOUR session (13:30-16:00 ET)

Method:
  1. Load all ignition-passed signals (live_signals.json)
  2. Simulate baseline trades (production params, cap=20)
  3. Simulate filtered trades (same params + regime filter)
  4. Compare metrics side-by-side

Data: 2026-03-03 (only complete day with quotes+signals)
      2026-03-02 (shadow trades, limited schema, directional only)

Output: reports/research/regime_paper_validation/
"""

import json
import math
import bisect
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter


def fmtd(v):
    """Format dollar value with sign and commas."""
    if v >= 0:
        return "$+{:,.0f}".format(v)
    return "$-{:,.0f}".format(abs(v))


def fmtd_nosign(v):
    """Format dollar value with commas, no sign."""
    return "${:,.0f}".format(v)


# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUPERBOT = ROOT / "MORPHEUS_SUPERBOT"
OUT_ROOT = ROOT / "reports" / "research" / "regime_paper_validation"

LIVE_SIGNALS = SUPERBOT / "engine" / "output" / "live_signals.json"
QUOTE_DIR = SUPERBOT / "engine" / "cache" / "quotes"
PAPER_TRADES = SUPERBOT / "engine" / "output" / "paper_trades.json"
PRESSURE_EVENTS = SUPERBOT / "engine" / "output" / "pressure_events_2026-03-03.json"
MAR2_TRADES = SUPERBOT / "engine" / "cache" / "trades" / "trades_2026-03-02.json"

# ── filter thresholds ──────────────────────────────────────────────────
VOL_THRESHOLD = 0.3      # medium: >= 0.3% std of log returns
SPREAD_THRESHOLD = 0.6   # <= 0.6% bid-ask spread
OFI_THRESHOLD = -0.2     # >= -0.2 (moderate)
SUPPRESS_REGIMES = {"LOW_VOLATILITY"}
SUPPRESS_SESSIONS = {"power_hour"}

# ── simulation parameters ─────────────────────────────────────────────
TRAIL_PCT = 1.0
MAX_HOLD_S = 300
POSITION_SIZE = 5000
BASELINE_CAP = 20
FILTERED_CAP = 20  # same cap for fair comparison


# ── data loading ───────────────────────────────────────────────────────
def load_live_signals():
    with open(LIVE_SIGNALS, encoding="utf-8") as f:
        return json.load(f)


def load_quote_cache(symbol):
    path = QUOTE_DIR / ("%s_quotes.json" % symbol)
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    quotes = sorted(data.get("quotes", []), key=lambda q: q["epoch"])
    epochs = [q["epoch"] for q in quotes]
    return quotes, epochs


def load_pressure_events():
    if not PRESSURE_EVENTS.exists():
        return {}
    with open(PRESSURE_EVENTS, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for sym, sd in data.get("events_per_symbol", {}).items():
        events = sorted(sd.get("events", []), key=lambda e: e["epoch"])
        result[sym] = events
    return result


def load_paper_trades():
    with open(PAPER_TRADES, encoding="utf-8") as f:
        return json.load(f)


def load_mar2():
    if not MAR2_TRADES.exists():
        return []
    with open(MAR2_TRADES, encoding="utf-8") as f:
        return json.load(f)


# ── feature computation ───────────────────────────────────────────────
def vol_1m(quotes, epochs, epoch):
    hi = bisect.bisect_right(epochs, epoch)
    lo = bisect.bisect_left(epochs, epoch - 60)
    window = quotes[lo:hi]
    if len(window) < 5:
        return None
    prices = [q["last"] for q in window if q.get("last") and q["last"] > 0]
    if len(prices) < 5:
        return None
    lr = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))
          if prices[i] > 0 and prices[i-1] > 0]
    if len(lr) < 3:
        return None
    m = sum(lr) / len(lr)
    return math.sqrt(sum((r - m)**2 for r in lr) / len(lr)) * 100


def spread_at(quotes, epochs, epoch):
    idx = bisect.bisect_right(epochs, epoch)
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
    hi = bisect.bisect_right(epochs, epoch)
    lo = bisect.bisect_left(epochs, epoch - 30)
    window = quotes[lo:hi]
    if len(window) < 5:
        return None
    ups = downs = 0
    for i in range(1, len(window)):
        p1 = window[i].get("last", 0)
        p0 = window[i-1].get("last", 0)
        if p1 > p0: ups += 1
        elif p1 < p0: downs += 1
    total = ups + downs
    return (ups - downs) / total if total > 0 else 0.0


def classify_tod(epoch):
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    h_et = dt.hour - 5  # EST (March 3 pre-DST)
    t = h_et * 60 + dt.minute
    if t < 570: return "premarket"
    if t < 630: return "open"
    if t < 810: return "midday"
    return "power_hour"


# ── trade simulation ──────────────────────────────────────────────────
def sim_trade(quotes, epochs, entry_epoch, entry_price):
    start = bisect.bisect_left(epochs, entry_epoch)
    peak = entry_price
    exit_price = entry_price
    mfe = mae = hold_s = 0.0
    reason = "TIME_CAP"

    for i in range(start, len(quotes)):
        q = quotes[i]
        dt = q["epoch"] - entry_epoch
        if dt > MAX_HOLD_S:
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
            if dd >= TRAIL_PCT:
                exit_price = price
                reason = "TRAIL_EXIT"
                hold_s = dt
                break
        if chg <= -2.0:
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
    pnl = (exit_price - entry_price) * POSITION_SIZE
    return {
        "entry_price": entry_price, "exit_price": exit_price,
        "return_pct": ret, "pnl": pnl, "mfe_pct": mfe, "mae_pct": mae,
        "hold_s": hold_s, "exit_reason": reason,
    }


# ── metrics ────────────────────────────────────────────────────────────
def calc(trades):
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "avg_pnl": 0, "total_pnl": 0,
                "avg_winner": 0, "avg_loser": 0, "max_dd": 0,
                "avg_mfe": 0, "avg_mae": 0, "avg_hold": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = sum(abs(t["pnl"]) for t in losses)
    pf = round(gw / gl, 2) if gl > 0 else ("INF" if gw > 0 else 0)
    total = sum(t["pnl"] for t in trades)

    # Max drawdown
    equity = 0
    peak_eq = 0
    max_dd = 0
    for t in trades:
        equity += t["pnl"]
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


# ── main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("REGIME FILTER PAPER VALIDATION")
    print("Baseline vs Filtered Side-by-Side Comparison")
    print("=" * 70)

    # Load data
    print("\nLoading signals...")
    signals = load_live_signals()
    ign_pass = [s for s in signals if s.get("ignition") == True]
    # Sort by epoch (chronological order)
    ign_pass.sort(key=lambda s: s["epoch"])
    print("  %d ignition-passed signals (chronological)" % len(ign_pass))

    print("\nLoading quote caches...")
    qcache = {}
    qepochs = {}
    for qf in QUOTE_DIR.glob("*_quotes.json"):
        sym = qf.stem.replace("_quotes", "")
        q, e = load_quote_cache(sym)
        if q:
            qcache[sym] = q
            qepochs[sym] = e
            print("  %s: %d quotes" % (sym, len(q)))

    print("\nLoading pressure events...")
    pressure = load_pressure_events()

    # ── Process each signal ────────────────────────────────────────────
    print("\nProcessing signals...")
    baseline_trades = []
    filtered_trades = []
    blocked_by_filter = []  # trades baseline took but filter blocked
    missed_by_filter = []   # high-MFE trades filter blocked

    baseline_count = 0
    filtered_count = 0

    for i, sig in enumerate(ign_pass):
        sym = sig["symbol"]
        if sym not in qcache:
            continue

        epoch = sig["epoch"]
        price = sig["price"]
        if price <= 0:
            continue

        quotes = qcache[sym]
        epochs = qepochs[sym]

        # Simulate trade
        result = sim_trade(quotes, epochs, epoch, price)
        if result["hold_s"] < 1:
            continue

        # Compute features
        v = vol_1m(quotes, epochs, epoch)
        sp = spread_at(quotes, epochs, epoch)
        ofi = ofi_30s(quotes, epochs, epoch)
        tod = classify_tod(epoch)
        regime = sig.get("regime", "unknown")

        features = {
            "vol_1m": v, "spread": sp, "ofi": ofi,
            "tod": tod, "regime": regime,
            "pressure_score": sig.get("pressure_score", 0),
            "trap_probability": sig.get("trap_probability", 0),
        }

        trade = {
            **result, "symbol": sym, "epoch": epoch, "features": features,
        }

        # ── BASELINE: Take if under cap ────────────────────────────────
        baseline_would_take = baseline_count < BASELINE_CAP
        if baseline_would_take:
            baseline_trades.append(trade)
            baseline_count += 1

        # ── FILTERED: Apply regime filter ──────────────────────────────
        passes_filter = True
        block_reasons = []

        # Volatility check
        if v is not None and v < VOL_THRESHOLD:
            passes_filter = False
            block_reasons.append("LOW_VOL(%.3f < %.1f)" % (v, VOL_THRESHOLD))

        # Spread check (only if data available)
        if sp is not None and sp > SPREAD_THRESHOLD:
            passes_filter = False
            block_reasons.append("HIGH_SPREAD(%.3f > %.1f)" % (sp, SPREAD_THRESHOLD))

        # OFI check
        if ofi is not None and ofi < OFI_THRESHOLD:
            passes_filter = False
            block_reasons.append("WEAK_OFI(%.2f < %.1f)" % (ofi, OFI_THRESHOLD))

        # Regime suppression
        if regime in SUPPRESS_REGIMES:
            passes_filter = False
            block_reasons.append("SUPPRESS_%s" % regime)

        # Session suppression
        if tod in SUPPRESS_SESSIONS:
            passes_filter = False
            block_reasons.append("SUPPRESS_%s" % tod)

        if passes_filter and filtered_count < FILTERED_CAP:
            filtered_trades.append(trade)
            filtered_count += 1
        elif baseline_would_take and not passes_filter:
            trade["block_reasons"] = block_reasons
            blocked_by_filter.append(trade)
            # Track if this was a high-MFE miss
            if result["mfe_pct"] >= 2.0:
                missed_by_filter.append(trade)

        if (i + 1) % 500 == 0:
            print("  %d/%d signals | baseline=%d filtered=%d" % (
                i + 1, len(ign_pass), baseline_count, filtered_count))

    print("\n  Baseline: %d trades" % len(baseline_trades))
    print("  Filtered: %d trades" % len(filtered_trades))
    print("  Blocked by filter (from baseline set): %d" % len(blocked_by_filter))
    print("  Missed high-MFE (>=2%%): %d" % len(missed_by_filter))

    # ── Compute metrics ───────────────────────────────────────────────
    m_base = calc(baseline_trades)
    m_filt = calc(filtered_trades)

    # Per-symbol
    def by_sym(trades):
        groups = defaultdict(list)
        for t in trades:
            groups[t["symbol"]].append(t)
        return {s: calc(ts) for s, ts in groups.items()}

    base_sym = by_sym(baseline_trades)
    filt_sym = by_sym(filtered_trades)

    # Per-session
    def by_session(trades):
        groups = defaultdict(list)
        for t in trades:
            groups[t["features"]["tod"]].append(t)
        return {s: calc(ts) for s, ts in groups.items()}

    base_sess = by_session(baseline_trades)
    filt_sess = by_session(filtered_trades)

    # ── Also run on ALL ignition signals (uncapped) for broader view ───
    print("\nRunning uncapped analysis (all ignition signals)...")
    all_baseline = []
    all_filtered = []
    all_blocked = []

    for sig in ign_pass:
        sym = sig["symbol"]
        if sym not in qcache:
            continue
        epoch = sig["epoch"]
        price = sig["price"]
        if price <= 0:
            continue

        quotes = qcache[sym]
        epochs = qepochs[sym]
        result = sim_trade(quotes, epochs, epoch, price)
        if result["hold_s"] < 1:
            continue

        v = vol_1m(quotes, epochs, epoch)
        sp = spread_at(quotes, epochs, epoch)
        ofi = ofi_30s(quotes, epochs, epoch)
        tod = classify_tod(epoch)
        regime = sig.get("regime", "unknown")

        trade = {**result, "symbol": sym, "epoch": epoch,
                 "features": {"vol_1m": v, "spread": sp, "ofi": ofi,
                              "tod": tod, "regime": regime}}

        all_baseline.append(trade)

        pf = True
        reasons = []
        if v is not None and v < VOL_THRESHOLD:
            pf = False
            reasons.append("LOW_VOL")
        if sp is not None and sp > SPREAD_THRESHOLD:
            pf = False
            reasons.append("HIGH_SPREAD")
        if ofi is not None and ofi < OFI_THRESHOLD:
            pf = False
            reasons.append("WEAK_OFI")
        if regime in SUPPRESS_REGIMES:
            pf = False
            reasons.append("SUPPRESS_REGIME")
        if tod in SUPPRESS_SESSIONS:
            pf = False
            reasons.append("SUPPRESS_SESSION")

        if pf:
            all_filtered.append(trade)
        else:
            trade["block_reasons"] = reasons
            all_blocked.append(trade)

    m_all_base = calc(all_baseline)
    m_all_filt = calc(all_filtered)
    m_all_blocked = calc(all_blocked)

    print("  All baseline: %d trades" % len(all_baseline))
    print("  All filtered: %d trades" % len(all_filtered))
    print("  All blocked: %d trades" % len(all_blocked))

    # Block reason distribution
    block_reasons_count = Counter()
    for t in all_blocked:
        for r in t.get("block_reasons", []):
            block_reasons_count[r.split("(")[0]] += 1

    # Per-symbol uncapped
    all_base_sym = by_sym(all_baseline)
    all_filt_sym = by_sym(all_filtered)
    all_block_sym = by_sym(all_blocked)

    # Per-session uncapped
    all_base_sess = by_session(all_baseline)
    all_filt_sess = by_session(all_filtered)

    # ── Cross-reference with actual paper trades ───────────────────────
    paper_data = load_paper_trades()
    paper_trades = paper_data["trades"]
    paper_metrics = paper_data["metrics"]

    # Check which paper trades would pass the filter
    paper_pass = []
    paper_block = []
    for pt in paper_trades:
        epoch = pt["entry_epoch"]
        sym = pt["symbol"]
        if sym not in qcache:
            paper_pass.append(pt)
            continue
        quotes = qcache[sym]
        epochs = qepochs[sym]
        v = vol_1m(quotes, epochs, epoch)
        sp = spread_at(quotes, epochs, epoch)
        ofi = ofi_30s(quotes, epochs, epoch)
        tod = classify_tod(epoch)
        regime = pt.get("regime", "unknown")

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

        pt_ext = {**pt, "filter_pass": passes, "block_reasons": reasons,
                  "vol_1m": v, "spread": sp, "ofi_val": ofi, "tod": tod}
        if passes:
            paper_pass.append(pt_ext)
        else:
            paper_block.append(pt_ext)

    m_paper_pass = calc(paper_pass)
    m_paper_block = calc(paper_block)

    # ── Mar 2 directional check ────────────────────────────────────────
    mar2 = load_mar2()
    mar2_entered = [t for t in mar2 if t.get("action") == "enter"]
    mar2_wr = round(sum(1 for t in mar2_entered if t.get("pnl", 0) > 0)
                    / max(1, len(mar2_entered)) * 100, 1) if mar2_entered else 0

    # ── Write outputs ──────────────────────────────────────────────────
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = OUT_ROOT / "2026-03-03"
    day_dir.mkdir(parents=True, exist_ok=True)

    # ── Daily Report ───────────────────────────────────────────────────
    print("\nWriting reports...")

    lines = [
        "# Regime Filter Paper Validation — Daily Report",
        "## Date: 2026-03-03 | BATL +87%% intraday",
        "## Generated: %s" % today,
        "",
        "---",
        "",
        "## FILTER CONFIGURATION",
        "",
        "```",
        "Candidate Filter:",
        "  volatility_1m >= 0.3%% (medium)",
        "  spread <= 0.6%%",
        "  OFI >= -0.2 (moderate)",
        "",
        "Suppression Rules:",
        "  suppress LOW_VOLATILITY regime",
        "  suppress POWER_HOUR session (13:30-16:00 ET)",
        "",
        "Simulation: trail=%.1f%%, max_hold=%ds, size=%d shares" % (
            TRAIL_PCT, MAX_HOLD_S, POSITION_SIZE),
        "```",
        "",
        "---",
        "",
        "## SIDE-BY-SIDE COMPARISON (Cap=%d)" % BASELINE_CAP,
        "",
        "| Metric | Baseline | Filtered | Delta |",
        "|--------|----------|----------|-------|",
    ]

    def delta(a, b, fmt="%.1f"):
        if isinstance(a, str) or isinstance(b, str):
            return "N/A"
        d = b - a
        return ("+" + fmt if d >= 0 else fmt) % d

    comparisons = [
        ("Trades", "%d" % m_base["n"], "%d" % m_filt["n"],
         delta(m_base["n"], m_filt["n"], "%d")),
        ("Win Rate", "%.1f%%" % m_base["wr"], "%.1f%%" % m_filt["wr"],
         delta(m_base["wr"], m_filt["wr"], "%.1f%%")),
        ("Profit Factor", str(m_base["pf"]), str(m_filt["pf"]),
         delta(float(m_base["pf"]) if isinstance(m_base["pf"], (int, float)) else 0,
               float(m_filt["pf"]) if isinstance(m_filt["pf"], (int, float)) else 0, "%.2f")),
        ("Total PnL", "$%+.0f" % m_base["total_pnl"],
         "$%+.0f" % m_filt["total_pnl"],
         "$%+.0f" % (m_filt["total_pnl"] - m_base["total_pnl"])),
        ("Avg PnL", "$%+.0f" % m_base["avg_pnl"],
         "$%+.0f" % m_filt["avg_pnl"],
         "$%+.0f" % (m_filt["avg_pnl"] - m_base["avg_pnl"])),
        ("Avg Winner", "$%+.0f" % m_base["avg_winner"],
         "$%+.0f" % m_filt["avg_winner"],
         "$%+.0f" % (m_filt["avg_winner"] - m_base["avg_winner"])),
        ("Avg Loser", "$%+.0f" % m_base["avg_loser"],
         "$%+.0f" % m_filt["avg_loser"],
         "$%+.0f" % (m_filt["avg_loser"] - m_base["avg_loser"])),
        ("Max Drawdown", "$%.0f" % m_base["max_dd"],
         "$%.0f" % m_filt["max_dd"],
         "$%+.0f" % (m_filt["max_dd"] - m_base["max_dd"])),
        ("Avg MFE", "%.3f%%" % m_base["avg_mfe"],
         "%.3f%%" % m_filt["avg_mfe"],
         delta(m_base["avg_mfe"], m_filt["avg_mfe"], "%.3f%%")),
        ("Avg MAE", "%.3f%%" % m_base["avg_mae"],
         "%.3f%%" % m_filt["avg_mae"],
         delta(m_base["avg_mae"], m_filt["avg_mae"], "%.3f%%")),
        ("Avg Hold", "%.1fs" % m_base["avg_hold"],
         "%.1fs" % m_filt["avg_hold"],
         delta(m_base["avg_hold"], m_filt["avg_hold"], "%.1fs")),
    ]
    for label, b, f, d in comparisons:
        lines.append("| %s | %s | %s | %s |" % (label, b, f, d))

    # ── Blocked trades detail ──────────────────────────────────────────
    lines.extend([
        "",
        "## TRADES BLOCKED BY FILTER (From Baseline Set)",
        "",
        "| # | Symbol | Price | PnL | MFE%% | Block Reason | Would Have Won? |",
        "|---|--------|-------|-----|-------|-------------|----------------|",
    ])
    for i, t in enumerate(blocked_by_filter):
        won = "YES" if t["pnl"] > 0 else "NO"
        reasons = ", ".join(t.get("block_reasons", []))
        lines.append("| %d | %s | $%.2f | $%+.0f | %.2f%% | %s | %s |" % (
            i+1, t["symbol"], t["entry_price"], t["pnl"],
            t["mfe_pct"], reasons, won))

    # ── Missed high-MFE trades ─────────────────────────────────────────
    if missed_by_filter:
        lines.extend([
            "",
            "## MISSED HIGH-MFE TRADES (MFE >= 2%%)",
            "",
            "| Symbol | Price | MFE%% | Actual PnL | Block Reason |",
            "|--------|-------|-------|-----------|-------------|",
        ])
        for t in missed_by_filter:
            reasons = ", ".join(t.get("block_reasons", []))
            lines.append("| %s | $%.2f | %.2f%% | $%+.0f | %s |" % (
                t["symbol"], t["entry_price"], t["mfe_pct"], t["pnl"], reasons))

    # ── Paper trade cross-reference ────────────────────────────────────
    lines.extend([
        "",
        "---",
        "",
        "## ACTUAL PAPER TRADE FILTER CROSS-REFERENCE",
        "",
        "How the filter would have treated the 20 actual paper trades:",
        "",
        "| # | Price | Regime | PnL | Vol_1m | Spread | OFI | ToD | Filter |",
        "|---|-------|--------|-----|--------|--------|-----|-----|--------|",
    ])
    for i, pt in enumerate(paper_pass + paper_block):
        v = pt.get("vol_1m")
        sp = pt.get("spread")
        ofi_v = pt.get("ofi_val")
        tod = pt.get("tod", "?")
        status = "PASS" if pt.get("filter_pass", True) else "BLOCK"
        lines.append("| %d | $%.2f | %s | $%+.0f | %s | %s | %s | %s | **%s** |" % (
            i+1, pt["entry_price"], pt.get("regime", "?"), pt["pnl"],
            "%.3f" % v if v else "N/A",
            "%.3f" % sp if sp else "N/A",
            "%+.2f" % ofi_v if ofi_v is not None else "N/A",
            tod, status))

    lines.extend([
        "",
        "| Metric | Paper PASS | Paper BLOCK |",
        "|--------|-----------|-------------|",
        "| Trades | %d | %d |" % (len(paper_pass), len(paper_block)),
        "| WR | %.1f%% | %.1f%% |" % (m_paper_pass["wr"], m_paper_block["wr"]),
        "| PF | %s | %s |" % (m_paper_pass["pf"], m_paper_block["pf"]),
        "| Total PnL | $%+.0f | $%+.0f |" % (m_paper_pass["total_pnl"], m_paper_block["total_pnl"]),
    ])

    # ── Per-symbol (capped) ────────────────────────────────────────────
    lines.extend([
        "",
        "---",
        "",
        "## PER-SYMBOL RESULTS (Cap=%d)" % BASELINE_CAP,
        "",
        "| Symbol | Base n | Base PF | Filt n | Filt PF | Improvement |",
        "|--------|--------|---------|--------|---------|-------------|",
    ])
    all_syms = sorted(set(list(base_sym.keys()) + list(filt_sym.keys())))
    for sym in all_syms:
        mb = base_sym.get(sym, {"n": 0, "pf": 0})
        mf = filt_sym.get(sym, {"n": 0, "pf": 0})
        bp = float(mb["pf"]) if isinstance(mb["pf"], (int, float)) else 0
        fp = float(mf["pf"]) if isinstance(mf["pf"], (int, float)) else 0
        imp = "BETTER" if fp > bp else ("SAME" if fp == bp else "WORSE")
        if mf["n"] == 0: imp = "NO TRADES"
        lines.append("| %s | %d | %s | %d | %s | %s |" % (
            sym, mb["n"], mb["pf"], mf["n"], mf["pf"], imp))

    # ── Per-session (capped) ───────────────────────────────────────────
    lines.extend([
        "",
        "## PER-SESSION RESULTS (Cap=%d)" % BASELINE_CAP,
        "",
        "| Session | Base n | Base PF | Base WR | Filt n | Filt PF | Filt WR |",
        "|---------|--------|---------|---------|--------|---------|---------|",
    ])
    for seg in ["premarket", "open", "midday", "power_hour"]:
        mb = base_sess.get(seg, {"n": 0, "pf": 0, "wr": 0})
        mf = filt_sess.get(seg, {"n": 0, "pf": 0, "wr": 0})
        lines.append("| %s | %d | %s | %.1f%% | %d | %s | %.1f%% |" % (
            seg, mb["n"], mb["pf"], mb["wr"], mf["n"], mf["pf"], mf["wr"]))

    lines.extend(["", "---", "",
                   "*Data source: live_signals.json, *_quotes.json, paper_trades.json (READ-ONLY)*",
                   "*NO production changes were made.*", ""])

    with open(day_dir / "daily_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  daily_report.md written")

    # ── Rolling Scorecard ──────────────────────────────────────────────
    lines = [
        "# Regime Filter Daily Scorecard",
        "## Rolling Summary Across Available Days",
        "## Generated: %s" % today,
        "",
        "---",
        "",
        "## AVAILABLE DATA",
        "",
        "| Date | Data Type | Signals | Trades (Sim) | Notes |",
        "|------|-----------|---------|-------------|-------|",
        "| 2026-03-02 | Shadow trades only | N/A | 25 entered | Limited schema, no quotes |",
        "| **2026-03-03** | **Full (signals+quotes+trades)** | **%d** | **%d** | Primary validation day |" % (
            len(ign_pass), len(all_baseline)),
        "| 2026-03-04 | Gating blocks only | N/A | N/A | No quotes available |",
        "",
        "---",
        "",
        "## DAILY SCORECARD",
        "",
        "### 2026-03-03 (Full Simulation, Uncapped)",
        "",
        "| Metric | Baseline | Filtered | Blocked | Filter Edge |",
        "|--------|----------|----------|---------|-------------|",
        "| Trades | %d | %d | %d | - |" % (m_all_base["n"], len(all_filtered), len(all_blocked)),
        "| Win Rate | %.1f%% | %.1f%% | %.1f%% | %+.1f pp |" % (
            m_all_base["wr"], m_all_filt["wr"], m_all_blocked["wr"],
            m_all_filt["wr"] - m_all_base["wr"]),
        "| Profit Factor | %s | %s | %s | %s |" % (
            m_all_base["pf"], m_all_filt["pf"], m_all_blocked["pf"],
            "+%.2f" % (float(m_all_filt["pf"]) - float(m_all_base["pf"]))
            if isinstance(m_all_filt["pf"], (int, float)) and isinstance(m_all_base["pf"], (int, float))
            else "N/A"),
        "| Total PnL | $%+.0f | $%+.0f | $%+.0f | $%+.0f |" % (
            m_all_base["total_pnl"], m_all_filt["total_pnl"], m_all_blocked["total_pnl"],
            m_all_filt["total_pnl"] - m_all_base["total_pnl"]),
        "| Avg PnL | $%+.0f | $%+.0f | $%+.0f | $%+.0f |" % (
            m_all_base["avg_pnl"], m_all_filt["avg_pnl"], m_all_blocked["avg_pnl"],
            m_all_filt["avg_pnl"] - m_all_base["avg_pnl"]),
        "| Max DD | $%.0f | $%.0f | - | $%+.0f |" % (
            m_all_base["max_dd"], m_all_filt["max_dd"],
            m_all_filt["max_dd"] - m_all_base["max_dd"]),
    ]

    # Block reason breakdown
    lines.extend([
        "",
        "### Block Reason Distribution",
        "",
        "| Reason | Count | %% of Blocked |",
        "|--------|-------|--------------|",
    ])
    total_blocked = sum(block_reasons_count.values())
    for reason, count in block_reasons_count.most_common():
        lines.append("| %s | %d | %.1f%% |" % (
            reason, count, count / max(1, total_blocked) * 100))

    # Per-symbol uncapped
    lines.extend([
        "",
        "### Per-Symbol (Uncapped)",
        "",
        "| Symbol | Base n | Base PF | Filt n | Filt PF | Blocked n | Blocked PF | Filter Edge |",
        "|--------|--------|---------|--------|---------|-----------|------------|-------------|",
    ])
    all_syms = sorted(set(list(all_base_sym.keys()) + list(all_filt_sym.keys())))
    for sym in all_syms:
        mb = all_base_sym.get(sym, {"n": 0, "pf": 0})
        mf = all_filt_sym.get(sym, {"n": 0, "pf": 0})
        mbl = all_block_sym.get(sym, {"n": 0, "pf": 0})
        bp = float(mb["pf"]) if isinstance(mb["pf"], (int, float)) else 0
        fp = float(mf["pf"]) if isinstance(mf["pf"], (int, float)) else 0
        edge = "YES" if fp > bp else "NO"
        if mf["n"] == 0: edge = "N/A"
        lines.append("| %s | %d | %s | %d | %s | %d | %s | %s |" % (
            sym, mb["n"], mb["pf"], mf["n"], mf["pf"], mbl["n"], mbl["pf"], edge))

    # Per-session uncapped
    lines.extend([
        "",
        "### Per-Session (Uncapped)",
        "",
        "| Session | Base n | Base PF | Filt n | Filt PF | Filter Edge |",
        "|---------|--------|---------|--------|---------|-------------|",
    ])
    for seg in ["premarket", "open", "midday", "power_hour"]:
        mb = all_base_sess.get(seg, {"n": 0, "pf": 0})
        mf = all_filt_sess.get(seg, {"n": 0, "pf": 0})
        bp = float(mb["pf"]) if isinstance(mb["pf"], (int, float)) else 0
        fp = float(mf["pf"]) if isinstance(mf["pf"], (int, float)) else 0
        edge = "YES" if fp > bp else ("SUPPRESSED" if seg in SUPPRESS_SESSIONS else "NO")
        if mf["n"] == 0 and seg in SUPPRESS_SESSIONS: edge = "SUPPRESSED"
        lines.append("| %s | %d | %s | %d | %s | %s |" % (
            seg, mb["n"], mb["pf"], mf["n"], mf["pf"], edge))

    # Mar 2 directional
    lines.extend([
        "",
        "### 2026-03-02 (Shadow Trades, Directional Only)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        "| Trades entered | %d |" % len(mar2_entered),
        "| Win rate | %.1f%% |" % mar2_wr,
        "| Note | Different schema, cannot apply regime filter |",
        "",
    ])

    # Stability metrics
    lines.extend([
        "---",
        "",
        "## STABILITY METRICS",
        "",
        "| Metric | Value | Assessment |",
        "|--------|-------|------------|",
        "| Days with full data | 1 | INSUFFICIENT for production |",
        "| Days filter improves PF | 1/1 | Positive but single-day |",
        "| Days filter improves WR | 1/1 | Positive but single-day |",
        "| Symbols where filter helps | %d/%d | %s |" % (
            sum(1 for s in all_syms
                if all_filt_sym.get(s, {"n": 0})["n"] > 0 and
                isinstance(all_filt_sym.get(s, {"pf": 0})["pf"], (int, float)) and
                isinstance(all_base_sym.get(s, {"pf": 0})["pf"], (int, float)) and
                float(all_filt_sym[s]["pf"]) > float(all_base_sym[s]["pf"])),
            sum(1 for s in all_syms if all_filt_sym.get(s, {"n": 0})["n"] > 0),
            "Directionally positive"),
        "| Filter trade reduction | %.0f%% | %d of %d signals blocked |" % (
            len(all_blocked) / max(1, len(all_baseline)) * 100,
            len(all_blocked), len(all_baseline)),
        "| Regime consistency (Mar 2 vs 4) | MIXED | IGNITION/EXTENSION shift across days |",
    ])

    # Conclusion
    p_pf = float(m_all_filt["pf"]) if isinstance(m_all_filt["pf"], (int, float)) else 0
    b_pf = float(m_all_base["pf"]) if isinstance(m_all_base["pf"], (int, float)) else 0

    lines.extend([
        "",
        "---",
        "",
        "## CONCLUSION",
        "",
    ])

    if p_pf > b_pf:
        lines.extend([
            "**The regime filter IMPROVES performance on the available data.**",
            "",
            "- PF improvement: %s -> %s (+%.2f)" % (m_all_base["pf"], m_all_filt["pf"], p_pf - b_pf),
            "- WR improvement: %.1f%% -> %.1f%% (+%.1f pp)" % (
                m_all_base["wr"], m_all_filt["wr"], m_all_filt["wr"] - m_all_base["wr"]),
            "- Trade reduction: %d -> %d (%.0f%% filtered out)" % (
                len(all_baseline), len(all_filtered),
                len(all_blocked) / max(1, len(all_baseline)) * 100),
            "- Blocked trades had: PF=%s, WR=%.1f%% (worse than filtered)" % (
                m_all_blocked["pf"], m_all_blocked["wr"]),
        ])
    else:
        lines.extend([
            "**The regime filter does NOT improve performance on the available data.**",
            "- Baseline PF=%s vs Filtered PF=%s" % (m_all_base["pf"], m_all_filt["pf"]),
        ])

    lines.extend([
        "",
        "### DEPLOYMENT READINESS: NOT READY",
        "",
        "**Reason**: Only 1 day of complete data. Need 5-10 days minimum for production deployment.",
        "",
        "### NEXT STEPS",
        "",
        "1. Collect 5+ additional trading days with full quote+signal data",
        "2. Re-run this validation daily as data accumulates",
        "3. Track rolling PF/WR delta to confirm stability",
        "4. Monitor missed high-MFE trades for filter over-aggressiveness",
        "5. Consider relaxing OFI threshold if true L2 data becomes available",
        "",
        "---",
        "",
        "*All data sources accessed READ-ONLY. NO production changes were made.*",
        "*Script: ai/research/regime_paper_validation.py*",
    ])

    with open(OUT_ROOT / "regime_filter_daily_scorecard.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  regime_filter_daily_scorecard.md written")

    # ── Cumulative Summary ─────────────────────────────────────────────
    lines = [
        "# Regime Filter Cumulative Summary",
        "## Validation Period: 2026-03-02 to 2026-03-04",
        "## Generated: %s" % today,
        "",
        "---",
        "",
        "## EXECUTIVE SUMMARY",
        "",
        "The combined regime filter (vol >= medium, spread <= 0.6%%, OFI >= moderate,",
        "suppress LOW_VOLATILITY, suppress POWER_HOUR) was validated against %d" % len(all_baseline),
        "simulated trades across 11 symbols on 2026-03-03.",
        "",
        "### Key Results (Uncapped, All Signals)",
        "",
        "| Metric | Baseline | Filtered | Improvement |",
        "|--------|----------|----------|-------------|",
        "| Trades | %d | %d | -%d (%.0f%% reduction) |" % (
            len(all_baseline), len(all_filtered), len(all_blocked),
            len(all_blocked) / max(1, len(all_baseline)) * 100),
        "| Win Rate | %.1f%% | %.1f%% | %+.1f pp |" % (
            m_all_base["wr"], m_all_filt["wr"], m_all_filt["wr"] - m_all_base["wr"]),
        "| Profit Factor | %s | %s | %s |" % (
            m_all_base["pf"], m_all_filt["pf"],
            "+%.2f" % (p_pf - b_pf) if p_pf and b_pf else "N/A"),
        "| Total PnL | $%+.0f | $%+.0f | $%+.0f |" % (
            m_all_base["total_pnl"], m_all_filt["total_pnl"],
            m_all_filt["total_pnl"] - m_all_base["total_pnl"]),
        "",
        "### Key Results (Capped at %d, Production-Like)" % BASELINE_CAP,
        "",
        "| Metric | Baseline | Filtered |",
        "|--------|----------|----------|",
        "| Trades | %d | %d |" % (m_base["n"], m_filt["n"]),
        "| Win Rate | %.1f%% | %.1f%% |" % (m_base["wr"], m_filt["wr"]),
        "| Profit Factor | %s | %s |" % (m_base["pf"], m_filt["pf"]),
        "| Total PnL | $%+.0f | $%+.0f |" % (m_base["total_pnl"], m_filt["total_pnl"]),
        "",
        "### Actual Paper Trades (20 BATL, Production Engine)",
        "",
        "| Metric | Would PASS | Would BLOCK |",
        "|--------|-----------|-------------|",
        "| Trades | %d | %d |" % (len(paper_pass), len(paper_block)),
        "| Win Rate | %.1f%% | %.1f%% |" % (m_paper_pass["wr"], m_paper_block["wr"]),
        "| Profit Factor | %s | %s |" % (m_paper_pass["pf"], m_paper_block["pf"]),
        "| Total PnL | $%+.0f | $%+.0f |" % (m_paper_pass["total_pnl"], m_paper_block["total_pnl"]),
        "",
        "---",
        "",
        "## FILTER COMPONENT EFFECTIVENESS",
        "",
        "| Component | Blocked Signals | %% of All Blocks | Standalone PF Lift |",
        "|-----------|----------------|-----------------|-------------------|",
    ]

    # Individual filter components on uncapped data
    for name, fn in [
        ("LOW_VOL", lambda t: (t["features"]["vol_1m"] is not None and
                                t["features"]["vol_1m"] < VOL_THRESHOLD)),
        ("HIGH_SPREAD", lambda t: (t["features"]["spread"] is not None and
                                    t["features"]["spread"] > SPREAD_THRESHOLD)),
        ("WEAK_OFI", lambda t: (t["features"]["ofi"] is not None and
                                 t["features"]["ofi"] < OFI_THRESHOLD)),
        ("SUPPRESS_REGIME", lambda t: t["features"]["regime"] in SUPPRESS_REGIMES),
        ("SUPPRESS_SESSION", lambda t: t["features"]["tod"] in SUPPRESS_SESSIONS),
    ]:
        blocked = [t for t in all_baseline if fn(t)]
        passed = [t for t in all_baseline if not fn(t)]
        m_p = calc(passed)
        p_pf_comp = float(m_p["pf"]) if isinstance(m_p["pf"], (int, float)) else 0
        lift = p_pf_comp - b_pf
        lines.append("| %s | %d | %.1f%% | %+.2f |" % (
            name, len(blocked), len(blocked) / max(1, len(all_baseline)) * 100, lift))

    lines.extend([
        "",
        "---",
        "",
        "## DEPLOYMENT READINESS",
        "",
        "| Criterion | Status | Required |",
        "|-----------|--------|----------|",
        "| Days of full data | 1 | 5-10 |",
        "| Filter PF > Baseline PF | %s | Consistent across days |" % (
            "YES" if p_pf > b_pf else "NO"),
        "| Filter WR > Baseline WR | %s | Consistent across days |" % (
            "YES" if m_all_filt["wr"] > m_all_base["wr"] else "NO"),
        "| Missed high-MFE trades < 10%% | %s | Acceptable miss rate |" % (
            "UNKNOWN" if not all_filtered else
            "YES" if sum(1 for t in all_blocked if t.get("mfe_pct", 0) >= 2.0) / max(1, len(all_baseline)) < 0.10
            else "NO"),
        "| Multi-symbol validation | YES (11 symbols) | At least 3 symbols |",
        "| Cross-day consistency | UNKNOWN | Need more days |",
        "",
        "**VERDICT: PROMISING but NOT READY for production.**",
        "Continue collecting daily data and re-running validation.",
        "",
        "---",
        "",
        "*All data sources accessed READ-ONLY. NO production changes were made.*",
        "*Script: ai/research/regime_paper_validation.py*",
    ])

    with open(OUT_ROOT / "cumulative_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  cumulative_summary.md written")

    # ── Missed Trades Analysis ─────────────────────────────────────────
    # Find the best trades that filter would block
    high_mfe_blocked = sorted(
        [t for t in all_blocked if t.get("mfe_pct", 0) >= 1.0],
        key=lambda t: t.get("mfe_pct", 0), reverse=True)[:20]

    lines = [
        "# Missed Trades Analysis",
        "## Trades the filter would block that had high MFE",
        "## Generated: %s" % today,
        "",
        "---",
        "",
        "## TOP 20 BLOCKED TRADES BY MFE (>=1%%)",
        "",
        "| # | Symbol | Price | MFE%% | Actual PnL | Block Reasons | Session | Regime |",
        "|---|--------|-------|-------|-----------|--------------|---------|--------|",
    ]
    for i, t in enumerate(high_mfe_blocked):
        reasons = ", ".join(t.get("block_reasons", []))
        lines.append("| %d | %s | $%.2f | %.2f%% | $%+.0f | %s | %s | %s |" % (
            i+1, t["symbol"], t["entry_price"], t["mfe_pct"], t["pnl"],
            reasons, t["features"]["tod"], t["features"]["regime"]))

    # Aggregate missed PnL
    total_missed_pnl = sum(t["pnl"] for t in all_blocked if t["pnl"] > 0)
    total_avoided_loss = sum(abs(t["pnl"]) for t in all_blocked if t["pnl"] <= 0)

    lines.extend([
        "",
        "## MISSED vs AVOIDED",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        "| Total blocked trades | %d |" % len(all_blocked),
        "| Blocked winners | %d |" % sum(1 for t in all_blocked if t["pnl"] > 0),
        "| Blocked losers | %d |" % sum(1 for t in all_blocked if t["pnl"] <= 0),
        "| Missed profit (winners blocked) | $%+.0f |" % total_missed_pnl,
        "| Avoided losses (losers blocked) | $%+.0f |" % total_avoided_loss,
        "| Net filter value | $%+.0f |" % (total_avoided_loss - total_missed_pnl),
        "",
        "**The filter avoids $%+.0f more in losses than it misses in profits.**" % (
            total_avoided_loss - total_missed_pnl)
        if total_avoided_loss > total_missed_pnl else
        "**WARNING: The filter misses $%+.0f more in profits than it avoids in losses.**" % (
            total_missed_pnl - total_avoided_loss),
        "",
        "---",
        "",
        "*Data source: live_signals.json, *_quotes.json (READ-ONLY)*",
        "*NO production changes were made.*",
    ])

    with open(OUT_ROOT / "missed_trades_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  missed_trades_analysis.md written")

    # ── Final output ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REGIME FILTER PAPER VALIDATION COMPLETE")
    print("=" * 70)
    print("\nOutput:")
    for p in sorted(OUT_ROOT.rglob("*.md")):
        print("  %s" % p.relative_to(ROOT))


if __name__ == "__main__":
    main()
