"""
MAX_AI Scanner Discovery Audit
===============================
Determine whether MAX_AI is discovering high-quality momentum stocks
or feeding Morpheus_AI weak candidates.

Data: signal_ledger.jsonl, gating_blocks.jsonl, quote cache (Mar 3, 2026)

NO production changes. Research-only analysis.
"""

import json
import re
import statistics
from pathlib import Path
from datetime import datetime, timezone
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
SUPERBOT = ROOT / "MORPHEUS_SUPERBOT"
REPORTS = ROOT / "reports"

SIGNAL_LEDGER = SUPERBOT / "engine" / "cache" / "morpheus_reports" / "2026-03-03" / "signal_ledger.jsonl"
GATING_BLOCKS = SUPERBOT / "engine" / "cache" / "morpheus_reports" / "2026-03-03" / "gating_blocks.jsonl"
DOWNSTREAM = SUPERBOT / "engine" / "cache" / "morpheus_reports" / "2026-03-03" / "downstream_blocks_summary.json"
QUOTE_DIR = SUPERBOT / "engine" / "cache" / "quotes"

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
OUTPUT_MD = REPORTS / f"MAX_AI_SCANNER_DISCOVERY_AUDIT_{TODAY}.md"

# RTH open = 14:30 UTC (9:30 ET)
RTH_OPEN_HOUR_UTC = 14
RTH_OPEN_MIN_UTC = 30


# ---------------------------------------------------------------------------
# Part 1: Load Data
# ---------------------------------------------------------------------------
def load_signals():
    """Load signal_ledger into per-symbol structure."""
    symbols = defaultdict(lambda: {
        "first_ts": None, "first_epoch": None,
        "signal_count": 0, "scored_count": 0,
        "ignition_pass": 0, "ignition_fail": 0,
        "meta_approved": 0, "risk_approved": 0,
        "extension_reduced": 0,
        "strategies": Counter(), "tags": Counter(),
        "scanner_scores": [], "gap_pcts": [],
        "entry_prices": [], "regimes": Counter(),
        "market_modes": Counter(),
        "has_catalyst_tag": False,
        "confidence_scores": [],
        "momentum_scores": [],
    })

    with open(SIGNAL_LEDGER) as f:
        for line in f:
            rec = json.loads(line)
            sym = rec.get("symbol", "")
            if not sym:
                continue
            s = symbols[sym]
            decision = rec.get("decision", "")

            if decision == "":
                # Initial signal
                s["signal_count"] += 1
                ts = rec.get("timestamp", "")
                if s["first_ts"] is None or ts < s["first_ts"]:
                    s["first_ts"] = ts
                strat = rec.get("strategy", "")
                if strat:
                    s["strategies"][strat] += 1
                tags = rec.get("tags", [])
                for t in tags:
                    s["tags"][t] += 1
                    if t.startswith("scanner:"):
                        try:
                            s["scanner_scores"].append(int(t.split(":")[1]))
                        except ValueError:
                            pass
                    if t.startswith("gap:"):
                        m = re.match(r"gap:(\d+(?:\.\d+)?)\s*(?:pct)?$", t)
                        if m:
                            s["gap_pcts"].append(float(m.group(1)))
                    if t in ("catalyst", "news", "earnings"):
                        s["has_catalyst_tag"] = True
                ep = rec.get("entry_price")
                if ep:
                    s["entry_prices"].append(float(ep))

            elif decision == "SIGNAL_SCORED":
                s["scored_count"] += 1
                conf = rec.get("confidence")
                if conf is not None:
                    s["confidence_scores"].append(float(conf))
            elif decision == "IGNITION_PASS":
                s["ignition_pass"] += 1
                ms = rec.get("momentum_snapshot", {})
                if ms and ms.get("momentum_score") is not None:
                    s["momentum_scores"].append(float(ms["momentum_score"]))
            elif decision == "IGNITION_FAIL":
                s["ignition_fail"] += 1
            elif decision == "META_APPROVED":
                s["meta_approved"] += 1
            elif decision == "RISK_APPROVED":
                s["risk_approved"] += 1
                # Extract regime
                details = rec.get("details", {})
                gr = details.get("gate_result", {})
                ss = gr.get("scored_signal", {})
                sig = ss.get("signal", {})
                regime = sig.get("regime", "unknown")
                mm = sig.get("market_mode", "unknown")
                s["regimes"][regime] += 1
                s["market_modes"][mm] += 1
            elif decision == "EXTENSION_REDUCED":
                s["extension_reduced"] += 1

    return dict(symbols)


def load_quote_cache():
    """Load quote cache with precomputed epochs."""
    cache = {}
    for qf in sorted(QUOTE_DIR.glob("*_quotes.json")):
        with open(qf) as f:
            data = json.load(f)
        sym = data["symbol"]
        quotes = data["quotes"]
        quotes.sort(key=lambda q: q["epoch"])
        epochs = [q["epoch"] for q in quotes]
        cache[sym] = {"quotes": quotes, "epochs": epochs, "count": len(quotes)}
    return cache


def load_downstream():
    if DOWNSTREAM.exists():
        with open(DOWNSTREAM) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Part 2: Discovery Timeline
# ---------------------------------------------------------------------------
def compute_discovery_timeline(symbols, cache):
    """For each symbol, compute discovery timing relative to price action."""
    timeline = {}
    for sym, s in symbols.items():
        if sym not in cache or not s["first_ts"]:
            timeline[sym] = {
                "has_quotes": False,
                "discovery_class": "NO_QUOTE_DATA",
            }
            continue

        from dateutil import parser as dp
        disc_ts = dp.parse(s["first_ts"])
        disc_epoch = disc_ts.timestamp()

        quotes = cache[sym]["quotes"]
        epochs = cache[sym]["epochs"]
        prices = [q["last"] for q in quotes if q.get("last") is not None]
        if not prices:
            timeline[sym] = {"has_quotes": True, "discovery_class": "NO_PRICE_DATA"}
            continue

        # Find the intraday high and its time
        best_price = max(prices)
        best_idx = next(i for i, q in enumerate(quotes) if q.get("last") == best_price)
        best_epoch = epochs[best_idx]

        # Find first significant move (>2% from open)
        open_price = prices[0]
        breakout_epoch = None
        for i, q in enumerate(quotes):
            p = q.get("last")
            if p and (p - open_price) / open_price * 100 >= 2.0:
                breakout_epoch = epochs[i]
                break

        # If no breakout, use the high time
        if breakout_epoch is None:
            breakout_epoch = best_epoch

        lead_time = breakout_epoch - disc_epoch

        if lead_time > 60:
            disc_class = "EARLY_DISCOVERY"
        elif lead_time > 0:
            disc_class = "LATE_DISCOVERY"
        else:
            disc_class = "POST_BREAKOUT"

        timeline[sym] = {
            "has_quotes": True,
            "disc_epoch": disc_epoch,
            "breakout_epoch": breakout_epoch,
            "high_epoch": best_epoch,
            "lead_time_s": lead_time,
            "discovery_class": disc_class,
            "open_price": open_price,
            "high_price": best_price,
            "intraday_range_pct": (best_price - open_price) / open_price * 100,
        }
    return timeline


# ---------------------------------------------------------------------------
# Part 3: Momentum Validation (MFE/MAE from discovery time)
# ---------------------------------------------------------------------------
def compute_momentum(symbols, cache):
    """Compute MFE/MAE from each symbol's first signal time."""
    from dateutil import parser as dp
    momentum = {}
    for sym, s in symbols.items():
        if sym not in cache or not s["first_ts"]:
            momentum[sym] = None
            continue

        disc_epoch = dp.parse(s["first_ts"]).timestamp()
        quotes = cache[sym]["quotes"]
        epochs = cache[sym]["epochs"]

        # Get ref price at discovery
        idx = bisect_left(epochs, disc_epoch)
        if idx >= len(quotes):
            idx = len(quotes) - 1
        ref_price = None
        for i in [idx, idx - 1, idx + 1]:
            if 0 <= i < len(quotes) and quotes[i].get("last") is not None:
                ref_price = quotes[i]["last"]
                break
        if ref_price is None or ref_price <= 0:
            momentum[sym] = None
            continue

        result = {"ref_price": ref_price}
        for window in [30, 60, 120]:
            i_start = bisect_left(epochs, disc_epoch)
            i_end = bisect_right(epochs, disc_epoch + window)
            window_prices = [q["last"] for q in quotes[i_start:i_end]
                           if q.get("last") is not None]
            if window_prices:
                changes = [(p - ref_price) / ref_price * 100 for p in window_prices]
                result[f"mfe_{window}s"] = max(changes)
                result[f"mae_{window}s"] = min(changes)
            else:
                result[f"mfe_{window}s"] = None
                result[f"mae_{window}s"] = None

        # Classify outcome based on 120s MFE
        mfe = result.get("mfe_120s")
        mae = result.get("mae_120s")
        if mfe is not None:
            if mfe >= 1.0:
                result["outcome"] = "STRONG_BREAKOUT"
            elif mfe >= 0.5:
                result["outcome"] = "MODERATE_BREAKOUT"
            elif mae is not None and mae < -0.5:
                result["outcome"] = "FALSE_BREAKOUT"
            else:
                result["outcome"] = "WEAK_MOVE"
        else:
            result["outcome"] = "NO_DATA"

        momentum[sym] = result
    return momentum


# ---------------------------------------------------------------------------
# Part 5: Liquidity & Microstructure
# ---------------------------------------------------------------------------
def compute_liquidity(cache):
    """Compute liquidity metrics per symbol from quote cache."""
    liquidity = {}
    for sym, data in cache.items():
        quotes = data["quotes"]
        spreads = []
        trade_sizes = []
        prev_vol = None

        for q in quotes:
            bid, ask, last = q.get("bid"), q.get("ask"), q.get("last")
            vol = q.get("volume")
            if bid and ask and bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid * 100
                spreads.append(spread_pct)
            if prev_vol is not None and vol is not None and vol > prev_vol:
                trade_sizes.append(vol - prev_vol)
            if vol is not None:
                prev_vol = vol

        # Volume acceleration: compare first 30min vs next 30min
        rth_start = None
        for q in quotes:
            epoch = q["epoch"]
            # Find RTH open (14:30 UTC)
            ts_str = q.get("ts", "")
            if "14:30" in ts_str or "14:31" in ts_str:
                rth_start = epoch
                break

        vol_accel = None
        if rth_start:
            # Volume in first 30 min vs second 30 min of RTH
            vol_30_1, vol_30_2 = 0, 0
            for q in quotes:
                e = q["epoch"]
                v = q.get("volume")
                if v is None:
                    continue
                if rth_start <= e < rth_start + 1800:
                    vol_30_1 = max(vol_30_1, v)
                elif rth_start + 1800 <= e < rth_start + 3600:
                    vol_30_2 = max(vol_30_2, v)
            if vol_30_1 > 0 and vol_30_2 > vol_30_1:
                vol_accel = (vol_30_2 - vol_30_1) / vol_30_1

        liquidity[sym] = {
            "avg_spread_pct": statistics.mean(spreads) if spreads else None,
            "median_spread_pct": statistics.median(spreads) if spreads else None,
            "avg_trade_size": statistics.mean(trade_sizes) if trade_sizes else None,
            "total_volume": (quotes[-1].get("volume") or 0) if quotes else 0,
            "quote_count": len(quotes),
            "vol_acceleration": vol_accel,
        }
    return liquidity


# ---------------------------------------------------------------------------
# Part 9: Missed Opportunity Detection
# ---------------------------------------------------------------------------
def detect_missed_opportunities(symbols, cache):
    """Find symbols in quote cache with big moves that weren't heavily signaled."""
    missed = []
    for sym, data in cache.items():
        quotes = data["quotes"]
        prices = [q["last"] for q in quotes if q.get("last") is not None]
        if len(prices) < 10:
            continue

        open_p = prices[0]
        hi = max(prices)
        lo = min(prices)
        max_move_up = (hi - open_p) / open_p * 100
        max_move_down = (lo - open_p) / open_p * 100

        sig_count = symbols.get(sym, {}).get("signal_count", 0) if sym in symbols else 0
        risk_approved = symbols.get(sym, {}).get("risk_approved", 0) if sym in symbols else 0

        missed.append({
            "symbol": sym,
            "max_move_up_pct": max_move_up,
            "max_move_down_pct": max_move_down,
            "signal_count": sig_count,
            "risk_approved": risk_approved,
            "open_price": open_p,
            "high_price": hi,
            "was_discovered": sig_count > 0,
        })
    return sorted(missed, key=lambda x: -x["max_move_up_pct"])


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def generate_report(symbols, timeline, momentum, liquidity,
                    missed, downstream, cache):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []

    total_syms = len(symbols)
    quoted_syms = [s for s in symbols if s in cache]
    total_signals = sum(s["signal_count"] for s in symbols.values())
    total_risk = sum(s["risk_approved"] for s in symbols.values())
    total_ignition_pass = sum(s["ignition_pass"] for s in symbols.values())

    # ---- Header ----
    lines.append("# MAX_AI Scanner Discovery Audit")
    lines.append(f"## Date: 2026-03-03 | Generated: {now}")
    lines.append("")

    # ---- Part 1: Scanner Coverage ----
    lines.append("## 1. Scanner Coverage")
    lines.append("")
    lines.append(f"- **Symbols discovered by MAX_AI:** {total_syms}")
    lines.append(f"- **Symbols with quote data:** {len(quoted_syms)}")
    lines.append(f"- **Symbols without quotes:** {total_syms - len(quoted_syms)}")
    lines.append(f"- **Total signals generated:** {total_signals:,}")
    lines.append(f"- **Ignition passes:** {total_ignition_pass}")
    lines.append(f"- **Risk approved:** {total_risk}")
    lines.append(f"- **Executed trades:** 0")
    lines.append("")

    # Signal funnel
    pipe = downstream.get("pipeline_blocks", {})
    lines.append("### Signal Funnel")
    lines.append("")
    lines.append("```")
    lines.append(f"  Signals Detected:     {total_signals:>7,}")
    lines.append(f"  Extension Blocked:    {pipe.get('EXTENSION_GATE', 0):>7,}  ({pipe.get('EXTENSION_GATE', 0)/total_signals*100:.1f}%)")
    lines.append(f"  Ignition Blocked:     {pipe.get('IGNITION_GATE', 0):>7,}  ({pipe.get('IGNITION_GATE', 0)/total_signals*100:.1f}%)")
    lines.append(f"  Ignition Passed:      {total_ignition_pass:>7}")
    lines.append(f"  Containment Blocked:  {pipe.get('CONTAINMENT', 0):>7}")
    lines.append(f"  Risk Approved:        {total_risk:>7}")
    lines.append(f"  Executed:             {0:>7}")
    lines.append("```")
    lines.append("")

    # Per-symbol table
    lines.append("### Per-Symbol Pipeline")
    lines.append("")
    lines.append("| Symbol | Signals | Scored | Ign.Pass | Risk.Appr | Gap% | Scanner | Quoted |")
    lines.append("|--------|---------|--------|----------|-----------|------|---------|--------|")
    for sym in sorted(symbols.keys(), key=lambda s: -symbols[s]["signal_count"]):
        s = symbols[sym]
        gap = f"{s['gap_pcts'][0]:.0f}%" if s["gap_pcts"] else "N/A"
        sc_max = max(s["scanner_scores"]) if s["scanner_scores"] else "N/A"
        quoted = "Yes" if sym in cache else "No"
        lines.append(f"| {sym} | {s['signal_count']:,} | {s['scored_count']:,} | "
                    f"{s['ignition_pass']} | {s['risk_approved']} | "
                    f"{gap} | {sc_max} | {quoted} |")
    lines.append("")

    # ---- Part 2: Discovery Timeline ----
    lines.append("## 2. Discovery Timing")
    lines.append("")

    disc_classes = Counter(t["discovery_class"] for t in timeline.values())
    lines.append("| Classification | Count | Description |")
    lines.append("|---------------|-------|-------------|")
    lines.append(f"| EARLY_DISCOVERY | {disc_classes.get('EARLY_DISCOVERY', 0)} | >60s before breakout |")
    lines.append(f"| LATE_DISCOVERY | {disc_classes.get('LATE_DISCOVERY', 0)} | 0-60s before breakout |")
    lines.append(f"| POST_BREAKOUT | {disc_classes.get('POST_BREAKOUT', 0)} | After breakout started |")
    lines.append(f"| NO_QUOTE_DATA | {disc_classes.get('NO_QUOTE_DATA', 0)} | No price data to assess |")
    lines.append("")

    # Detail table for quoted symbols
    lines.append("| Symbol | Lead Time | Class | Intraday Range | Open | High |")
    lines.append("|--------|-----------|-------|---------------|------|------|")
    for sym in sorted(timeline.keys(), key=lambda s: timeline[s].get("lead_time_s", 999999)):
        t = timeline[sym]
        if not t.get("has_quotes") or t.get("intraday_range_pct") is None:
            continue
        lt = t.get("lead_time_s", 0)
        lt_str = f"{lt:.0f}s" if abs(lt) < 3600 else f"{lt/60:.0f}m"
        lines.append(f"| {sym} | {lt_str} | {t['discovery_class']} | "
                    f"{t['intraday_range_pct']:.1f}% | "
                    f"${t['open_price']:.2f} | ${t['high_price']:.2f} |")
    lines.append("")

    # ---- Part 3: Momentum Validation ----
    lines.append("## 3. Momentum Validation")
    lines.append("")

    outcomes = Counter(m["outcome"] for m in momentum.values() if m)
    lines.append("### Breakout Classification (120s window from discovery)")
    lines.append("")
    lines.append("| Outcome | Count | % |")
    lines.append("|---------|-------|---|")
    total_m = sum(outcomes.values())
    for outcome in ["STRONG_BREAKOUT", "MODERATE_BREAKOUT", "WEAK_MOVE", "FALSE_BREAKOUT", "NO_DATA"]:
        c = outcomes.get(outcome, 0)
        pct = c / total_m * 100 if total_m else 0
        lines.append(f"| {outcome} | {c} | {pct:.1f}% |")
    lines.append("")

    # MFE/MAE by symbol
    lines.append("### MFE/MAE by Symbol")
    lines.append("")
    lines.append("| Symbol | Outcome | MFE_30s | MAE_30s | MFE_60s | MAE_60s | MFE_120s | MAE_120s |")
    lines.append("|--------|---------|---------|---------|---------|---------|----------|----------|")
    for sym in sorted(momentum.keys()):
        m = momentum[sym]
        if m is None:
            continue
        def fp(v):
            return f"{v:.3f}%" if v is not None else "N/A"
        lines.append(f"| {sym} | {m['outcome']} | "
                    f"{fp(m.get('mfe_30s'))} | {fp(m.get('mae_30s'))} | "
                    f"{fp(m.get('mfe_60s'))} | {fp(m.get('mae_60s'))} | "
                    f"{fp(m.get('mfe_120s'))} | {fp(m.get('mae_120s'))} |")
    lines.append("")

    # ---- Part 4: Scanner Quality Metrics ----
    lines.append("## 4. Scanner Quality Metrics")
    lines.append("")

    # Precision: symbols with positive edge / total discovered
    profitable_syms = [sym for sym, m in momentum.items()
                       if m and m.get("outcome") in ("STRONG_BREAKOUT", "MODERATE_BREAKOUT")]
    precision = len(profitable_syms) / total_syms * 100 if total_syms else 0

    # Discovery hit rate: symbols reaching risk_approved / total discovered
    risk_approved_syms = [sym for sym, s in symbols.items() if s["risk_approved"] > 0]
    hit_rate = len(risk_approved_syms) / total_syms * 100 if total_syms else 0

    # Continuation rate: of those with MFE data, how many continue >0.5%
    with_mfe = {sym: m for sym, m in momentum.items()
                if m and m.get("mfe_120s") is not None}
    cont_rate = sum(1 for m in with_mfe.values() if m["mfe_120s"] >= 0.5) / len(with_mfe) * 100 if with_mfe else 0

    # False breakout rate
    fb_rate = sum(1 for m in with_mfe.values()
                  if m.get("mae_120s") is not None and m["mae_120s"] < -0.5
                  and m["mfe_120s"] < 0.5) / len(with_mfe) * 100 if with_mfe else 0

    lines.append(f"- **Scanner precision** (profitable/total): {len(profitable_syms)}/{total_syms} ({precision:.1f}%)")
    lines.append(f"- **Discovery hit rate** (risk_approved/total): {len(risk_approved_syms)}/{total_syms} ({hit_rate:.1f}%)")
    lines.append(f"- **Breakout continuation rate** (MFE>0.5% in 120s): {cont_rate:.1f}%")
    lines.append(f"- **False breakout rate** (MAE<-0.5%, MFE<0.5%): {fb_rate:.1f}%")
    lines.append("")

    # ---- Part 5: Liquidity & Microstructure ----
    lines.append("## 5. Liquidity & Microstructure")
    lines.append("")
    lines.append("| Symbol | Avg Spread% | Med Spread% | Avg Trade Size | Total Vol | Quotes |")
    lines.append("|--------|------------|------------|---------------|-----------|--------|")
    for sym in sorted(liquidity.keys()):
        lq = liquidity[sym]
        def fn(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None else "N/A"
        lines.append(f"| {sym} | {fn(lq['avg_spread_pct'])}% | {fn(lq['median_spread_pct'])}% | "
                    f"{fn(lq['avg_trade_size'], '.0f')} | {lq['total_volume']:,} | {lq['quote_count']:,} |")
    lines.append("")

    # Compare successful vs unsuccessful by spread
    lines.append("### Spread: Successful vs Unsuccessful Symbols")
    lines.append("")
    succ_spreads = [liquidity[s]["avg_spread_pct"] for s in profitable_syms
                    if s in liquidity and liquidity[s]["avg_spread_pct"] is not None]
    fail_syms = [s for s in liquidity if s not in profitable_syms]
    fail_spreads = [liquidity[s]["avg_spread_pct"] for s in fail_syms
                    if liquidity[s]["avg_spread_pct"] is not None]

    if succ_spreads:
        lines.append(f"- Successful (breakout >0.5%): avg spread = {statistics.mean(succ_spreads):.3f}%")
    if fail_spreads:
        lines.append(f"- Unsuccessful: avg spread = {statistics.mean(fail_spreads):.3f}%")
    lines.append("")

    # ---- Part 6: Catalyst Correlation ----
    lines.append("## 6. Catalyst Correlation")
    lines.append("")

    catalyst_syms = {sym for sym, s in symbols.items() if s["has_catalyst_tag"]}
    no_catalyst = {sym for sym in symbols if sym not in catalyst_syms}

    # Gap-based catalyst proxy (gap > 20% likely has catalyst)
    high_gap = {sym for sym, s in symbols.items()
                if s["gap_pcts"] and max(s["gap_pcts"]) >= 20}
    low_gap = {sym for sym in symbols if sym not in high_gap}

    lines.append("Using gap% as catalyst proxy (>20% gap = likely catalyst):")
    lines.append("")
    lines.append("| Category | Symbols | Breakout >0.5% | Rate |")
    lines.append("|----------|---------|---------------|------|")

    for label, sym_set in [("High gap (>=20%)", high_gap), ("Low gap (<20%)", low_gap)]:
        n = len(sym_set)
        breakouts = sum(1 for s in sym_set
                       if s in with_mfe and with_mfe[s].get("mfe_120s", 0) >= 0.5)
        rate = breakouts / n * 100 if n else 0
        lines.append(f"| {label} | {n} | {breakouts} | {rate:.1f}% |")
    lines.append("")

    # Gap distribution
    lines.append("### Gap % Distribution")
    lines.append("")
    lines.append("| Symbol | Gap% | MFE_120s | Outcome |")
    lines.append("|--------|------|----------|---------|")
    for sym in sorted(symbols.keys(), key=lambda s: -(symbols[s]["gap_pcts"][0] if symbols[s]["gap_pcts"] else 0)):
        s = symbols[sym]
        gap = f"{s['gap_pcts'][0]:.0f}%" if s["gap_pcts"] else "N/A"
        m = momentum.get(sym)
        mfe = f"{m['mfe_120s']:.3f}%" if m and m.get("mfe_120s") is not None else "N/A"
        outcome = m["outcome"] if m else "NO_DATA"
        lines.append(f"| {sym} | {gap} | {mfe} | {outcome} |")
    lines.append("")

    # ---- Part 7: Float & Liquidity Study ----
    lines.append("## 7. Price Tier Analysis (Float Proxy)")
    lines.append("")
    lines.append("No float data available. Using entry price as liquidity proxy.")
    lines.append("")

    # Price tiers
    tiers = [
        ("$0-$2", 0, 2), ("$2-$5", 2, 5), ("$5-$10", 5, 10),
        ("$10-$20", 10, 20), ("$20+", 20, 9999),
    ]
    lines.append("| Price Tier | Symbols | Avg MFE_120s | Breakout >0.5% | Avg Spread |")
    lines.append("|-----------|---------|-------------|---------------|------------|")
    for label, lo, hi in tiers:
        tier_syms = [sym for sym, s in symbols.items()
                     if s["entry_prices"] and lo <= statistics.mean(s["entry_prices"]) < hi]
        if not tier_syms:
            lines.append(f"| {label} | 0 | N/A | N/A | N/A |")
            continue
        mfes = [momentum[s]["mfe_120s"] for s in tier_syms
                if s in momentum and momentum[s] and momentum[s].get("mfe_120s") is not None]
        breakouts = sum(1 for m in mfes if m >= 0.5)
        avg_mfe = statistics.mean(mfes) if mfes else None
        spreads = [liquidity[s]["avg_spread_pct"] for s in tier_syms
                   if s in liquidity and liquidity[s]["avg_spread_pct"] is not None]
        avg_spr = statistics.mean(spreads) if spreads else None
        lines.append(f"| {label} | {len(tier_syms)} | "
                    f"{'%.3f%%' % avg_mfe if avg_mfe is not None else 'N/A'} | "
                    f"{breakouts}/{len(tier_syms)} | "
                    f"{'%.3f%%' % avg_spr if avg_spr is not None else 'N/A'} |")
    lines.append("")

    # ---- Part 8: Regime Interaction ----
    lines.append("## 8. Regime Interaction")
    lines.append("")

    # Aggregate regime stats across all risk_approved signals
    regime_perf = defaultdict(lambda: {"count": 0, "symbols": set()})
    for sym, s in symbols.items():
        for regime, cnt in s["regimes"].items():
            key = regime
            regime_perf[key]["count"] += cnt
            regime_perf[key]["symbols"].add(sym)

    lines.append("| Regime | Risk Approved | Symbols | Breakout >0.5% |")
    lines.append("|--------|-------------|---------|---------------|")
    for regime in sorted(regime_perf.keys()):
        rp = regime_perf[regime]
        syms_in_regime = rp["symbols"]
        breakouts = sum(1 for s in syms_in_regime
                       if s in with_mfe and with_mfe[s].get("mfe_120s", 0) >= 0.5)
        lines.append(f"| {regime} | {rp['count']} | {len(syms_in_regime)} | "
                    f"{breakouts}/{len(syms_in_regime)} |")
    lines.append("")

    # Market mode breakdown
    lines.append("### By Market Mode")
    lines.append("")
    mode_perf = defaultdict(lambda: {"count": 0, "symbols": set()})
    for sym, s in symbols.items():
        for mode, cnt in s["market_modes"].items():
            mode_perf[mode]["count"] += cnt
            mode_perf[mode]["symbols"].add(sym)

    lines.append("| Mode | Risk Approved | Symbols |")
    lines.append("|------|-------------|---------|")
    for mode in sorted(mode_perf.keys()):
        mp = mode_perf[mode]
        lines.append(f"| {mode} | {mp['count']} | {len(mp['symbols'])} |")
    lines.append("")

    # ---- Part 9: Missed Opportunities ----
    lines.append("## 9. Missed Opportunity Detection")
    lines.append("")
    lines.append("Symbols in quote cache with significant moves:")
    lines.append("")
    lines.append("| Symbol | Max Move Up | Signals | Risk Approved | Discovered |")
    lines.append("|--------|-----------|---------|---------------|------------|")
    for m in missed:
        disc = "Yes" if m["was_discovered"] else "**NO**"
        lines.append(f"| {m['symbol']} | {m['max_move_up_pct']:.1f}% | "
                    f"{m['signal_count']:,} | {m['risk_approved']} | {disc} |")
    lines.append("")

    big_movers = [m for m in missed if m["max_move_up_pct"] >= 5]
    undiscovered = [m for m in missed if not m["was_discovered"]]
    lines.append(f"- Symbols with >5% intraday move: {len(big_movers)}")
    lines.append(f"- Symbols NOT discovered by MAX_AI: {len(undiscovered)}")
    if big_movers:
        coverage = sum(1 for m in big_movers if m["was_discovered"]) / len(big_movers) * 100
        lines.append(f"- Scanner coverage of big movers: {coverage:.0f}%")
    lines.append("")

    # ---- Part 10: Summary Statistics ----
    lines.append("## 10. Summary Statistics")
    lines.append("")

    # Key question: is the market bad or is MAX_AI finding wrong stocks?
    lines.append("### The Central Question")
    lines.append("")
    lines.append("*Is the market bad, or is MAX_AI finding the wrong stocks?*")
    lines.append("")

    # Evidence table
    all_mfes_120 = [m["mfe_120s"] for m in momentum.values()
                    if m and m.get("mfe_120s") is not None]
    all_maes_120 = [m["mae_120s"] for m in momentum.values()
                    if m and m.get("mae_120s") is not None]

    if all_mfes_120:
        avg_mfe = statistics.mean(all_mfes_120)
        avg_mae = statistics.mean(all_maes_120) if all_maes_120 else None
        pct_breakout = sum(1 for m in all_mfes_120 if m >= 0.5) / len(all_mfes_120) * 100

        lines.append("| Metric | Value | Interpretation |")
        lines.append("|--------|-------|---------------|")
        lines.append(f"| Avg MFE_120s | {avg_mfe:.3f}% | {'Positive momentum exists' if avg_mfe > 0.3 else 'Weak momentum'} |")
        lines.append(f"| Avg MAE_120s | {avg_mae:.3f}% | {'High drawdown risk' if avg_mae and avg_mae < -0.5 else 'Moderate risk'} |")
        lines.append(f"| Breakout >0.5% | {pct_breakout:.0f}% | {'Decent hit rate' if pct_breakout > 30 else 'Low hit rate'} |")
        lines.append(f"| Scanner precision | {precision:.1f}% | {'Adequate' if precision > 25 else 'Needs improvement'} |")
        lines.append(f"| Pipeline pass-through | {hit_rate:.1f}% | {'Healthy funnel' if hit_rate > 30 else 'Over-filtering or weak candidates'} |")
        lines.append(f"| Execution rate | 0% | Total pipeline blockage |")
    lines.append("")

    # ---- Part 11: Recommendations ----
    lines.append("## 11. Recommendations")
    lines.append("")

    # Build data-driven recommendations
    recs = []

    # Float/price filtering
    low_price_syms = [sym for sym, s in symbols.items()
                      if s["entry_prices"] and statistics.mean(s["entry_prices"]) < 2]
    if len(low_price_syms) > total_syms * 0.3:
        recs.append(("PRICE FILTER", "HIGH",
                     f"{len(low_price_syms)}/{total_syms} symbols are sub-$2. "
                     "These have wide spreads and poor fill quality. "
                     "Recommend minimum price filter of $2.00."))

    # Relative volume threshold
    if total_ignition_pass < total_signals * 0.01:
        recs.append(("IGNITION RATE", "HIGH",
                     f"Only {total_ignition_pass}/{total_signals:,} signals pass ignition "
                     f"({total_ignition_pass/total_signals*100:.2f}%). "
                     "Most fail on momentum quality. Consider pre-filtering "
                     "symbols with stronger volume profiles."))

    # Gap filtering
    extreme_gap_syms = [sym for sym, s in symbols.items()
                        if s["gap_pcts"] and max(s["gap_pcts"]) >= 40]
    if extreme_gap_syms:
        ext_blocked = sum(symbols[s]["extension_reduced"] + (1 if symbols[s]["ignition_fail"] > symbols[s]["signal_count"] * 0.9 else 0)
                         for s in extreme_gap_syms)
        recs.append(("GAP FILTER", "MEDIUM",
                     f"{len(extreme_gap_syms)} symbols have gap >40% (e.g. {', '.join(extreme_gap_syms[:3])}). "
                     f"These are overwhelmingly blocked by extension gate. "
                     "Consider MAX_AI pre-filtering gaps >40% without catalyst."))

    # Spread
    wide_spread_syms = [sym for sym in liquidity
                        if liquidity[sym]["avg_spread_pct"] is not None
                        and liquidity[sym]["avg_spread_pct"] > 0.5]
    if wide_spread_syms:
        recs.append(("SPREAD FILTER", "MEDIUM",
                     f"{len(wide_spread_syms)} symbols have avg spread >0.5% "
                     f"({', '.join(wide_spread_syms)}). "
                     "Wide spreads erode edge. Consider spread pre-screen at scanner level."))

    # Discovery timing
    post_breakout = [sym for sym, t in timeline.items()
                     if t.get("discovery_class") == "POST_BREAKOUT"]
    if post_breakout:
        recs.append(("DISCOVERY TIMING", "MEDIUM",
                     f"{len(post_breakout)} symbols discovered AFTER breakout "
                     f"({', '.join(post_breakout)}). "
                     "Late discovery = chasing. Improve pre-market scanning."))

    # Catalyst weighting
    recs.append(("CATALYST WEIGHTING", "LOW",
                 "No explicit catalyst/news data available in logs. "
                 "Recommend MAX_AI tag news catalysts to enable weighted filtering."))

    # Symbol concentration
    top3_signals = sorted(symbols.values(), key=lambda s: -s["signal_count"])[:3]
    top3_pct = sum(s["signal_count"] for s in top3_signals) / total_signals * 100
    if top3_pct > 50:
        recs.append(("SIGNAL CONCENTRATION", "LOW",
                     f"Top 3 symbols generate {top3_pct:.0f}% of all signals. "
                     "Consider throttling repeat signals for same symbol."))

    lines.append("| # | Area | Priority | Recommendation |")
    lines.append("|---|------|----------|----------------|")
    for i, (area, prio, desc) in enumerate(recs, 1):
        lines.append(f"| {i} | {area} | {prio} | {desc} |")
    lines.append("")

    # ---- Final Verdict ----
    lines.append("## 12. Verdict")
    lines.append("")

    # Determine: market bad or scanner bad?
    if all_mfes_120:
        avg_mfe = statistics.mean(all_mfes_120)
        if avg_mfe >= 0.5 and precision >= 30:
            verdict = "SCANNER OK, PIPELINE OVER-FILTERING"
            detail = ("MAX_AI is finding stocks that move. The problem lies downstream "
                      "in ignition/containment gates blocking valid signals. "
                      "Focus optimization on gate parameters, not scanner quality.")
        elif avg_mfe >= 0.3:
            verdict = "MIXED — SCANNER ADEQUATE, CANDIDATES MARGINAL"
            detail = ("MAX_AI finds moderate movers but not consistently strong breakouts. "
                      "The pipeline then filters out the marginal ones, leaving nothing. "
                      "Both scanner quality AND pipeline parameters need tuning.")
        else:
            verdict = "SCANNER FINDING WEAK CANDIDATES"
            detail = ("The discovered symbols show weak momentum after detection. "
                      "MAX_AI is either finding the wrong stocks or discovering them "
                      "too late. Priority: improve scanner candidate selection.")
    else:
        verdict = "INSUFFICIENT DATA"
        detail = "Not enough quote data to make a determination."

    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append(detail)
    lines.append("")
    lines.append("**NO production changes were made. This is research-only analysis.**")
    lines.append("")
    lines.append("---")
    lines.append("*Production remains frozen. All analysis is read-only research.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("MAX_AI SCANNER DISCOVERY AUDIT")
    print("=" * 65)
    print()

    print("Part 1: Loading data...")
    symbols = load_signals()
    print(f"  {len(symbols)} symbols discovered")
    print(f"  {sum(s['signal_count'] for s in symbols.values()):,} total signals")

    cache = load_quote_cache()
    print(f"  {len(cache)} symbols in quote cache")

    downstream = load_downstream()

    print()
    print("Part 2: Building discovery timeline...")
    timeline = compute_discovery_timeline(symbols, cache)
    from collections import Counter
    classes = Counter(t["discovery_class"] for t in timeline.values())
    for cls, cnt in classes.most_common():
        print(f"  {cls}: {cnt}")

    print()
    print("Part 3: Computing momentum validation...")
    momentum = compute_momentum(symbols, cache)
    outcomes = Counter(m["outcome"] for m in momentum.values() if m)
    for o, c in outcomes.most_common():
        print(f"  {o}: {c}")

    print()
    print("Part 5: Analyzing liquidity...")
    liquidity = compute_liquidity(cache)
    print(f"  {len(liquidity)} symbols analyzed")

    print()
    print("Part 9: Detecting missed opportunities...")
    missed = detect_missed_opportunities(symbols, cache)
    big = [m for m in missed if m["max_move_up_pct"] >= 5]
    print(f"  {len(big)} symbols with >5% intraday move")
    undiscovered = [m for m in missed if not m["was_discovered"]]
    print(f"  {len(undiscovered)} NOT discovered by MAX_AI")

    print()
    print("Generating report...")
    report = generate_report(symbols, timeline, momentum, liquidity,
                            missed, downstream, cache)

    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"  Report: {OUTPUT_MD}")

    print()
    print("=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
