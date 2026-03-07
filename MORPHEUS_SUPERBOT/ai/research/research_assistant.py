"""
Research Assistant
==================
Plain-English interpreter for all research outputs.

Reads pipeline results, scorecard data, and heatmaps.
Produces a briefing that tells you:
  1. What happened (in plain English, no jargon)
  2. What's working and what isn't
  3. What you should do next
  4. Red flags and warnings

Usage:
    python -m ai.research.research_assistant
    python -m ai.research.research_assistant --mode briefing
    python -m ai.research.research_assistant --mode diagnose
    python -m ai.research.research_assistant --mode explain

Modes:
    briefing  — Full daily briefing (default)
    diagnose  — "Something's wrong" diagnostic
    explain   — Metric glossary and interpretation guide

READ ONLY — no production changes.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
SUPERBOT = Path(__file__).resolve().parent.parent.parent
ROOT = SUPERBOT.parent
REPORT_ROOT = ROOT / "reports" / "research"
PIPELINE_RESULTS = REPORT_ROOT / "pipeline_results.json"
SCORECARD_DATA = REPORT_ROOT / "regime_paper_validation" / "scorecard_data.json"
QUOTE_DIR = SUPERBOT / "engine" / "cache" / "quotes"
SIGNALS_FILE = SUPERBOT / "engine" / "output" / "live_signals.json"
PAPER_TRADES = SUPERBOT / "engine" / "output" / "paper_trades.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_available_data():
    """Scan what data and reports exist."""
    info = {
        "pipeline_results": PIPELINE_RESULTS.exists(),
        "scorecard_data": SCORECARD_DATA.exists(),
        "quote_symbols": [],
        "has_signals": SIGNALS_FILE.exists(),
        "has_paper_trades": PAPER_TRADES.exists(),
        "report_dates": [],
    }

    # Quote cache symbols
    for qf in QUOTE_DIR.glob("*_quotes.json"):
        try:
            with open(qf, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("count", 0) > 0:
                info["quote_symbols"].append(qf.stem.replace("_quotes", ""))
        except Exception:
            pass

    # Report dates
    for d in ["replay", "alpha_heatmap", "regime_paper_validation"]:
        rd = REPORT_ROOT / d
        if rd.exists():
            for sub in rd.iterdir():
                if sub.is_dir() and len(sub.name) == 10 and sub.name[4] == "-":
                    if sub.name not in info["report_dates"]:
                        info["report_dates"].append(sub.name)
    info["report_dates"] = sorted(set(info["report_dates"]))

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Interpretation
# ═══════════════════════════════════════════════════════════════════════════════
def interpret_pf(pf, context=""):
    """Interpret Profit Factor in plain English."""
    if isinstance(pf, str):
        if pf == "INF":
            return "infinite (all winners, no losses — too good, likely small sample)"
        return "unknown"
    if pf >= 99:
        return "extremely high (likely an outlier day or tiny sample — don't trust this number alone)"
    if pf >= 3.0:
        return "excellent — for every $1 lost, you're making $%.1f back" % pf
    if pf >= 2.0:
        return "very good — making $%.1f for every $1 lost" % pf
    if pf >= 1.5:
        return "good — making $%.1f for every $1 lost" % pf
    if pf >= 1.1:
        return "marginally profitable — making $%.2f for every $1 lost (thin edge)" % pf
    if pf >= 1.0:
        return "break-even — barely covering losses"
    if pf >= 0.8:
        return "losing money — only making $%.2f for every $1 lost" % pf
    if pf >= 0.5:
        return "losing significantly — making only $%.2f for every $1 lost" % pf
    return "severe losses — making only $%.2f for every $1 lost" % pf


def interpret_wr(wr):
    """Interpret Win Rate in plain English."""
    if wr >= 60:
        return "high (%.0f%% of trades are winners)" % wr
    if wr >= 50:
        return "above average (%.0f%% winners)" % wr
    if wr >= 40:
        return "moderate (%.0f%% winners — needs good risk/reward to be profitable)" % wr
    if wr >= 30:
        return "low (only %.0f%% winners — needs large winners to compensate)" % wr
    return "very low (%.0f%% winners — most trades lose)" % wr


def interpret_filter_value(nfv):
    """Interpret Net Filter Value."""
    if nfv > 100000:
        return "strong positive — the filter saves ${:,.0f} more than it misses".format(nfv)
    if nfv > 10000:
        return "positive — the filter saves ${:,.0f} net".format(nfv)
    if nfv > 0:
        return "slightly positive — the filter saves ${:,.0f} but it's a thin margin".format(nfv)
    if nfv == 0:
        return "neutral — filter neither helps nor hurts"
    return "NEGATIVE — the filter is costing ${:,.0f} by blocking good trades".format(abs(nfv))


def interpret_trade_reduction(base_n, filt_n):
    """Interpret how many trades the filter blocks."""
    if base_n == 0:
        return "no trades to compare"
    pct = (1 - filt_n / base_n) * 100
    if pct >= 90:
        return "blocks %.0f%% of trades — very aggressive, only %d of %d pass" % (pct, filt_n, base_n)
    if pct >= 70:
        return "blocks %.0f%% of trades — selective, %d of %d pass" % (pct, filt_n, base_n)
    if pct >= 50:
        return "blocks %.0f%% of trades — moderate filtering" % pct
    if pct >= 20:
        return "blocks %.0f%% of trades — light filtering" % pct
    return "barely filtering (%.0f%% blocked)" % pct


def interpret_drawdown(max_dd, total_pnl):
    """Interpret max drawdown."""
    if max_dd <= 0:
        return "no drawdown recorded"
    if total_pnl > 0:
        ratio = max_dd / total_pnl
        if ratio > 5:
            return "WARNING: Max drawdown (${:,.0f}) is {}x the total profit — very risky path to get here".format(
                max_dd, int(ratio))
        if ratio > 2:
            return "Max drawdown (${:,.0f}) is {}x the total profit — bumpy ride".format(max_dd, round(ratio, 1))
        return "Max drawdown (${:,.0f}) is manageable relative to ${:,.0f} total profit".format(max_dd, total_pnl)
    else:
        return "Max drawdown ${:,.0f} on a losing strategy — the hole keeps getting deeper".format(max_dd)


def interpret_block_reasons(reason_counts):
    """Interpret why trades are being blocked."""
    if not reason_counts:
        return []
    total = sum(reason_counts.values())
    lines = []
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100

        if reason == "LOW_VOL":
            lines.append("  LOW VOLATILITY (%.0f%%): %d signals fired when the stock wasn't moving much. "
                         "These tend to be false signals — the price jiggles but doesn't trend." % (pct, count))
        elif reason == "SUPPRESS_REGIME":
            lines.append("  REGIME SUPPRESSION (%.0f%%): %d signals fired during LOW_VOLATILITY market regime. "
                         "The overall market was quiet, so momentum signals are unreliable." % (pct, count))
        elif reason == "HIGH_SPREAD":
            lines.append("  HIGH SPREAD (%.0f%%): %d signals had wide bid-ask spreads (>0.6%%). "
                         "Wide spreads eat into profit — you're paying too much to get in/out." % (pct, count))
        elif reason == "WEAK_OFI":
            lines.append("  WEAK ORDER FLOW (%.0f%%): %d signals had no clear buying pressure. "
                         "Price was going up on ticks but not sustained buying." % (pct, count))
        elif reason == "SUPPRESS_SESSION":
            lines.append("  SESSION SUPPRESSION (%.0f%%): %d signals fired during power hour (1:30-4pm). "
                         "Late-day trading is historically the worst session." % (pct, count))
        else:
            lines.append("  %s (%.0f%%): %d signals blocked for this reason." % (reason, pct, count))
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Briefing Generator
# ═══════════════════════════════════════════════════════════════════════════════
def generate_briefing():
    """Generate a plain-English daily briefing."""
    info = get_available_data()
    pipeline = load_json(PIPELINE_RESULTS)
    scorecard = load_json(SCORECARD_DATA)

    lines = []
    lines.append("=" * 70)
    lines.append("  RESEARCH BRIEFING")
    lines.append("  Generated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("=" * 70)

    if not pipeline:
        lines.append("")
        lines.append("  No pipeline results found.")
        lines.append("  Run the nightly pipeline first:")
        lines.append("    python -m ai.research.nightly_pipeline")
        lines.append("")
        lines.append("  Available data:")
        lines.append("    Quote symbols: %s" % ", ".join(sorted(info["quote_symbols"])))
        lines.append("    Has signals: %s" % ("Yes" if info["has_signals"] else "No"))
        lines.append("    Has paper trades: %s" % ("Yes" if info["has_paper_trades"] else "No"))
        return "\n".join(lines)

    date = pipeline.get("date", "unknown")
    lines.append("")
    lines.append("  Last analyzed trading day: %s" % date)
    lines.append("  Symbols tracked: %s" % ", ".join(sorted(pipeline.get("symbols", []))))
    lines.append("")

    # ── Section 1: The Bottom Line ──────────────────────────────────────
    lines.append("-" * 70)
    lines.append("  1. THE BOTTOM LINE")
    lines.append("-" * 70)
    lines.append("")

    regime = pipeline.get("regime", {})
    heatmap = pipeline.get("heatmap", {})
    replay = pipeline.get("replay", {})

    if regime.get("status") == "OK":
        base = regime["baseline"]
        filt = regime["filtered"]
        blocked = regime["blocked"]

        # Is the system profitable without filtering?
        if base["pf"] >= 1.0:
            lines.append("  The trading system WAS PROFITABLE on %s without any filtering." % date)
            lines.append("  Baseline profit factor: %s (%s)" % (base["pf"], interpret_pf(base["pf"])))
        else:
            lines.append("  The trading system LOST MONEY on %s without filtering." % date)
            lines.append("  Baseline profit factor: %s (%s)" % (base["pf"], interpret_pf(base["pf"])))
            lines.append("  Total loss: ${:,.0f} across {:,} trades.".format(abs(base["total_pnl"]), base["n"]))

        lines.append("")

        # Does the filter help?
        if isinstance(filt["pf"], (int, float)) and isinstance(base["pf"], (int, float)):
            if filt["pf"] > base["pf"]:
                lines.append("  THE REGIME FILTER IMPROVES PERFORMANCE:")
                lines.append("    Without filter: PF=%s, Win Rate=%.1f%% (%d trades)" % (
                    base["pf"], base["wr"], base["n"]))
                lines.append("    With filter:    PF=%s, Win Rate=%.1f%% (%d trades)" % (
                    filt["pf"], filt["wr"], filt["n"]))
                lines.append("")
                lines.append("    In plain English: The filter turned a %s strategy" % (
                    "losing" if base["pf"] < 1.0 else "marginal"))
                lines.append("    into a %s one." % (
                    "profitable" if filt["pf"] >= 1.1 else "break-even"))
                lines.append("")
                lines.append("    Trade reduction: %s" % interpret_trade_reduction(base["n"], filt["n"]))
            else:
                lines.append("  WARNING: The regime filter is NOT helping on this day.")
                lines.append("    Filtered PF (%s) is worse than baseline (%s)." % (filt["pf"], base["pf"]))

        # Net filter value
        nfv = regime.get("net_filter_value", 0)
        lines.append("")
        lines.append("  Net filter value: %s" % interpret_filter_value(nfv))

    lines.append("")

    # ── Section 2: What the Filter is Blocking ──────────────────────────
    lines.append("-" * 70)
    lines.append("  2. WHY TRADES ARE BEING BLOCKED")
    lines.append("-" * 70)
    lines.append("")

    if scorecard:
        days = scorecard.get("days", [])
        if days:
            latest = days[-1]
            reasons = latest.get("reason_counts", {})
            if reasons:
                lines.append("  The filter blocked %d of %d signals (%.0f%%). Here's why:" % (
                    latest["blocked"]["n"], latest["baseline"]["n"],
                    latest["blocked"]["n"] / max(1, latest["baseline"]["n"]) * 100))
                lines.append("")
                for line in interpret_block_reasons(reasons):
                    lines.append(line)
                lines.append("")
                lines.append("  WHAT THIS MEANS:")
                lines.append("    The blocked trades had PF=%s and WR=%.1f%%." % (
                    latest["blocked"]["pf"], latest["blocked"]["wr"]))
                lines.append("    The trades that PASSED had PF=%s and WR=%.1f%%." % (
                    latest["filtered"]["pf"], latest["filtered"]["wr"]))
                lines.append("")
                if isinstance(latest["blocked"]["pf"], (int, float)) and latest["blocked"]["pf"] < 1.0:
                    lines.append("    The filter is correctly blocking losing trades.")
                    lines.append("    Blocked trades would have lost money (PF < 1.0).")
                else:
                    lines.append("    WARNING: Blocked trades were also profitable.")
                    lines.append("    The filter may be too aggressive.")

    lines.append("")

    # ── Section 2.5: Sector Performance ───────────────────────────────
    sector_cls = pipeline.get("sector_classification")
    sector_heat = pipeline.get("sector_heat")
    if sector_cls or sector_heat:
        lines.append("-" * 70)
        lines.append("  2.5  SECTOR PERFORMANCE")
        lines.append("-" * 70)
        lines.append("")

        if sector_cls:
            by_type = sector_cls.get("by_asset_type", {})
            if by_type:
                lines.append("  SYMBOL CLASSIFICATION:")
                for atype, syms in sorted(by_type.items()):
                    lines.append("    %-20s: %s" % (atype, ", ".join(sorted(syms))))
                lines.append("")

        if sector_heat:
            lines.append("  SECTOR HEAT SCORES:")
            for sec, heat in sorted(sector_heat.items(),
                                     key=lambda x: -x[1].get("heat_score", 0)):
                zone = heat.get("heat_zone", "?")
                score = heat.get("heat_score", 0)
                zone_label = zone.upper()
                if zone == "hot":
                    lines.append("    %-20s: %3.0f  [%s] — trade aggressively" % (sec, score, zone_label))
                elif zone == "cold":
                    lines.append("    %-20s: %3.0f  [%s] — pull back" % (sec, score, zone_label))
                elif zone == "frozen":
                    lines.append("    %-20s: %3.0f  [%s] — skip entirely" % (sec, score, zone_label))
                else:
                    lines.append("    %-20s: %3.0f  [%s]" % (sec, score, zone_label))
            lines.append("")

        # Check for sector mismatches — profitable symbols being filtered
        if regime.get("status") == "OK" and sector_cls:
            classifications = sector_cls.get("classifications", {})
            regime_data = pipeline.get("regime", {})
            blocked_data = regime_data.get("blocked", {})
            # Look for inverse ETFs that might be incorrectly filtered
            inverse_syms = [s for s, c in classifications.items()
                           if c.get("asset_type") in ("inverse_etf", "leveraged_etf")]
            if inverse_syms:
                lines.append("  SECTOR INSIGHTS:")
                lines.append("    Leveraged/inverse ETFs in universe: %s" % ", ".join(sorted(inverse_syms)))
                lines.append("    These symbols need WIDER spread gates and may NOT need")
                lines.append("    regime suppression — verify they aren't being over-filtered.")
                lines.append("")

        lines.append("")

    # ── Section 3: Alpha Heatmap Insights ───────────────────────────────
    lines.append("-" * 70)
    lines.append("  3. WHEN THE SYSTEM WORKS BEST")
    lines.append("-" * 70)
    lines.append("")

    if heatmap.get("status") == "OK":
        all_m = heatmap["all"]
        fp = heatmap["filter_pass"]
        ff = heatmap["filter_fail"]

        lines.append("  Across %d simulated trades on %d symbols:" % (
            heatmap["trades"], heatmap["symbols"]))
        lines.append("")
        lines.append("  GOOD CONDITIONS (filter passes):")
        lines.append("    - %d trades, Win Rate %.1f%%, PF %s" % (fp["n"], fp["wr"], fp["pf"]))
        lines.append("    - Average winner: ${:,.0f} | Average loser: ${:,.0f}".format(
            fp["avg_winner"], abs(fp["avg_loser"])))
        lines.append("    - These trades have HIGHER volatility, TIGHTER spreads, and")
        lines.append("      POSITIVE order flow — the stock is actively moving with buyers.")
        lines.append("")
        lines.append("  BAD CONDITIONS (filter fails):")
        lines.append("    - %d trades, Win Rate %.1f%%, PF %s" % (ff["n"], ff["wr"], ff["pf"]))
        lines.append("    - Average winner: ${:,.0f} | Average loser: ${:,.0f}".format(
            ff["avg_winner"], abs(ff["avg_loser"])))
        lines.append("    - These trades fire when the stock is QUIET, WIDE-SPREAD, or")
        lines.append("      has no clear buying pressure — false breakout territory.")
        lines.append("")

        # MFE vs MAE insight
        if fp.get("avg_mfe") and fp.get("avg_mae"):
            mfe_mae = fp["avg_mfe"] / fp["avg_mae"] if fp["avg_mae"] > 0 else 0
            lines.append("  RISK/REWARD INSIGHT:")
            lines.append("    In good conditions, the average trade moves %.2f%% in your favor" % fp["avg_mfe"])
            lines.append("    before pulling back %.2f%% against you." % fp["avg_mae"])
            if mfe_mae > 2:
                lines.append("    That's a %.1fx reward-to-risk ratio — favorable." % mfe_mae)
            elif mfe_mae > 1:
                lines.append("    That's a %.1fx reward-to-risk ratio — acceptable but thin." % mfe_mae)
            else:
                lines.append("    That's a %.1fx reward-to-risk ratio — concerning." % mfe_mae)

    lines.append("")

    # ── Section 4: Grid Replay Results ──────────────────────────────────
    lines.append("-" * 70)
    lines.append("  4. EXIT PARAMETER OPTIMIZATION")
    lines.append("-" * 70)
    lines.append("")

    if replay.get("status") == "OK":
        best = replay.get("best", {})
        if best:
            lines.append("  Tested %d different exit parameter combinations." % replay["configs"])
            lines.append("")
            lines.append("  Best configuration found:")
            lines.append("    Hold time:    %ds (%s)" % (
                best["hold"],
                "very short" if best["hold"] <= 120 else
                "short" if best["hold"] <= 180 else
                "moderate" if best["hold"] <= 300 else "long"))
            lines.append("    Trail start:  %.2f%% (trail activates after this gain)" % best["trail_start"])
            lines.append("    Trail offset: %.2f%% (exit when price drops this far from peak)" % best["trail_offset"])
            lines.append("    Spread limit: %.1f%%" % best["spread_thresh"])
            lines.append("    Trade cap:    %d per session" % best["cap"])
            lines.append("")
            lines.append("    Result: %d trades, %.0f%% winners, PF=%s" % (
                best["n"], best["wr"], best["pf"]))
            lines.append("")

            if best["pf"] >= 99:
                lines.append("    NOTE: PF=99 means all losers had near-zero losses.")
                lines.append("    This is likely a statistical artifact from an extreme day")
                lines.append("    (BATL +87%). Don't optimize for this — it won't repeat.")
            elif best["pf"] >= 3:
                lines.append("    This is a strong result, but verify it holds across multiple days")
                lines.append("    before changing production parameters.")
            elif best["pf"] >= 1.5:
                lines.append("    Solid result. Compare with current production settings to see")
                lines.append("    if the improvement is worth the parameter change.")

    lines.append("")

    # ── Section 5: Deployment Readiness ─────────────────────────────────
    lines.append("-" * 70)
    lines.append("  5. DEPLOYMENT READINESS")
    lines.append("-" * 70)
    lines.append("")

    if scorecard:
        days = scorecard.get("days", [])
        n_days = len(days)

        if n_days >= 5:
            days_helped = sum(1 for d in days
                              if isinstance(d["filtered"]["pf"], (int, float))
                              and isinstance(d["baseline"]["pf"], (int, float))
                              and d["filtered"]["pf"] > d["baseline"]["pf"])
            if days_helped >= n_days * 0.7:
                lines.append("  STATUS: READY FOR DEPLOYMENT REVIEW")
                lines.append("")
                lines.append("  The regime filter has improved performance on %d of %d days (%.0f%%)." % (
                    days_helped, n_days, days_helped / n_days * 100))
                lines.append("  This meets the minimum threshold for production consideration.")
                lines.append("")
                lines.append("  NEXT STEP: Run walk-forward validation to check for overfitting:")
                lines.append("    python -m ai.research.walk_forward_validation")
            else:
                lines.append("  STATUS: NOT READY — INCONSISTENT RESULTS")
                lines.append("")
                lines.append("  The filter only helped on %d of %d days (%.0f%%)." % (
                    days_helped, n_days, days_helped / n_days * 100))
                lines.append("  Need 70%% or better consistency before considering deployment.")
        else:
            lines.append("  STATUS: NOT READY — INSUFFICIENT DATA")
            lines.append("")
            lines.append("  Only %d day(s) of data. Need at least 5 days to assess stability." % n_days)
            lines.append("  The nightly pipeline will accumulate this automatically.")
            lines.append("  Check back after %d more trading days." % (5 - n_days))
    else:
        lines.append("  No scorecard data. Run the nightly pipeline to start tracking.")

    lines.append("")

    # ── Section 6: Action Items ─────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("  6. RECOMMENDED ACTIONS")
    lines.append("-" * 70)
    lines.append("")

    actions = _generate_actions(pipeline, scorecard, info)
    for i, action in enumerate(actions, 1):
        lines.append("  %d. %s" % (i, action["action"]))
        lines.append("     WHY: %s" % action["why"])
        if action.get("how"):
            lines.append("     HOW: %s" % action["how"])
        lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF BRIEFING")
    lines.append("=" * 70)

    return "\n".join(lines)


def _generate_actions(pipeline, scorecard, info):
    """Generate prioritized action items based on current state."""
    actions = []

    if not pipeline:
        actions.append({
            "action": "Run the nightly pipeline",
            "why": "No analysis has been run yet. You need baseline data.",
            "how": "python -m ai.research.nightly_pipeline",
        })
        return actions

    regime = pipeline.get("regime", {})
    heatmap = pipeline.get("heatmap", {})

    # Check days of data
    n_days = 0
    if scorecard:
        n_days = len(scorecard.get("days", []))

    if n_days < 5:
        actions.append({
            "action": "Collect more data (%d more trading days needed)" % (5 - n_days),
            "why": "Only %d day(s) of data — minimum 5 needed for reliable conclusions." % n_days,
            "how": "Run nightly pipeline each evening: python -m ai.research.nightly_pipeline",
        })

    # Check if baseline is losing
    if regime.get("status") == "OK":
        base = regime["baseline"]
        filt = regime["filtered"]

        if base["pf"] < 1.0 and filt["pf"] >= 1.0:
            actions.append({
                "action": "The regime filter is critical — keep it active",
                "why": "Without the filter, the system loses money (PF=%.2f). "
                       "With it, it's profitable (PF=%.2f)." % (base["pf"], filt["pf"]),
            })

        if base["wr"] < 35:
            actions.append({
                "action": "Investigate low win rate (%.1f%%)" % base["wr"],
                "why": "Fewer than 1 in 3 trades wins. This could mean signals are "
                       "firing on noise, not real momentum.",
                "how": "python -m ai.research.pressure_quality_study  (check signal quality)\n"
                       "         python -m ai.research.ignition_detector_v2     (check ignition accuracy)",
            })

        # Check if filter is too aggressive
        if base["n"] > 0 and filt["n"] / base["n"] < 0.1:
            actions.append({
                "action": "Review filter aggressiveness (only %.0f%% of trades pass)" % (
                    filt["n"] / base["n"] * 100),
                "why": "The filter blocks over 90%% of signals. This might be too selective — "
                       "you could be missing real opportunities.",
                "how": "Check missed_trades.md in the regime validation output.\n"
                       "         Look at the 'Missed vs Avoided' section to see if blocked "
                       "winners outweigh avoided losses.",
            })

    # Check drawdown
    if heatmap.get("status") == "OK":
        all_m = heatmap["all"]
        if all_m["max_dd"] > abs(all_m["total_pnl"]) * 3 and all_m["total_pnl"] < 0:
            actions.append({
                "action": "Address severe drawdown risk",
                "why": "Max drawdown (${:,.0f}) is massive. Even if the filter "
                       "helps, the unfiltered path is dangerous.".format(all_m["max_dd"]),
                "how": "python -m ai.research.adaptive_exit_study   (optimize exits)\n"
                       "         python -m ai.research.time_of_day_gate          (block toxic time windows)",
            })

    # Always recommend review
    actions.append({
        "action": "Review the daily dashboard",
        "why": "Quick visual check of all metrics in one place.",
        "how": "Open: reports/research/daily_summary.md",
    })

    return actions


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic Mode
# ═══════════════════════════════════════════════════════════════════════════════
def generate_diagnostic():
    """Diagnose what might be wrong based on available data."""
    pipeline = load_json(PIPELINE_RESULTS)
    scorecard = load_json(SCORECARD_DATA)
    info = get_available_data()

    lines = []
    lines.append("=" * 70)
    lines.append("  RESEARCH DIAGNOSTIC")
    lines.append("  Generated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("=" * 70)
    lines.append("")

    problems = []
    ok_items = []

    # Check data availability
    if not info["quote_symbols"]:
        problems.append({
            "issue": "NO QUOTE DATA",
            "detail": "The quote cache directory is empty. No analysis can run.",
            "fix": "Ensure Morpheus_AI is exporting quote data to engine/cache/quotes/",
        })
    else:
        ok_items.append("Quote data available for %d symbols: %s" % (
            len(info["quote_symbols"]), ", ".join(sorted(info["quote_symbols"]))))

    if not info["has_signals"]:
        problems.append({
            "issue": "NO SIGNAL DATA",
            "detail": "live_signals.json not found. The nightly pipeline needs this.",
            "fix": "Ensure Morpheus_AI is exporting signals to engine/output/live_signals.json",
        })
    else:
        ok_items.append("Signal data exists (live_signals.json)")

    if not info["has_paper_trades"]:
        problems.append({
            "issue": "NO PAPER TRADE DATA",
            "detail": "paper_trades.json not found. Grid replay needs this for signal events.",
            "fix": "Ensure paper trading is active and exporting to engine/output/paper_trades.json",
        })
    else:
        ok_items.append("Paper trade data exists")

    # Check pipeline
    if not pipeline:
        problems.append({
            "issue": "NIGHTLY PIPELINE NEVER RUN",
            "detail": "No pipeline_results.json found. The automated analysis hasn't run yet.",
            "fix": "Run: python -m ai.research.nightly_pipeline",
        })
    else:
        elapsed = pipeline.get("elapsed_s", 0)
        ok_items.append("Pipeline last ran on %s (took %.1fs)" % (pipeline["date"], elapsed))

        for module in ["replay", "heatmap", "regime", "dashboard"]:
            status = pipeline.get(module, {}).get("status", "MISSING")
            if status == "OK":
                ok_items.append("  %s: OK" % module.capitalize())
            elif status == "ERROR":
                err = pipeline.get(module, {}).get("error", "unknown")
                problems.append({
                    "issue": "%s MODULE FAILED" % module.upper(),
                    "detail": "Error: %s" % err,
                    "fix": "Check the error message above. Common causes: missing data files, "
                           "corrupt JSON, or date mismatch.",
                })
            else:
                problems.append({
                    "issue": "%s MODULE STATUS: %s" % (module.upper(), status),
                    "detail": "Module didn't complete successfully.",
                    "fix": "Re-run the pipeline and check console output for errors.",
                })

    # Check regime filter effectiveness
    if pipeline and pipeline.get("regime", {}).get("status") == "OK":
        base = pipeline["regime"]["baseline"]
        filt = pipeline["regime"]["filtered"]

        if base["pf"] < 0.5:
            problems.append({
                "issue": "BASELINE STRATEGY SEVERELY UNPROFITABLE",
                "detail": "Without filtering, PF=%.2f means you lose $%.2f for every $1 you make. "
                          "This day had %d trades that collectively lost ${:,.0f}.".format(
                    abs(base["total_pnl"])) % (base["pf"], 1 / base["pf"] if base["pf"] > 0 else 999, base["n"]),
                "fix": "The filter helps, but the underlying signal quality may need work.\n"
                       "         Run: python -m ai.research.pressure_quality_study\n"
                       "         Run: python -m ai.research.ignition_detector_v2",
            })

        if isinstance(filt["pf"], (int, float)) and filt["pf"] < 1.0:
            problems.append({
                "issue": "REGIME FILTER NOT ENOUGH",
                "detail": "Even after filtering, PF=%.2f (still losing money)." % filt["pf"],
                "fix": "The filter criteria may need tightening, or the signal quality is too poor.\n"
                       "         Consider: tighter vol threshold, stricter spread, or "
                       "adding symbol-level gates.",
            })

    # Check scorecard
    if scorecard:
        days = scorecard.get("days", [])
        if len(days) == 1:
            problems.append({
                "issue": "SINGLE-DAY DATA",
                "detail": "All conclusions are based on ONE trading day. Any day can be an outlier.",
                "fix": "Keep running the nightly pipeline. Need 5+ days for reliable results.",
            })
    elif pipeline:
        problems.append({
            "issue": "NO SCORECARD TRACKING",
            "detail": "Scorecard data not accumulating across days.",
            "fix": "The nightly pipeline should create this automatically. "
                   "Re-run: python -m ai.research.nightly_pipeline",
        })

    # Output
    if problems:
        lines.append("  PROBLEMS FOUND: %d" % len(problems))
        lines.append("")
        for i, p in enumerate(problems, 1):
            lines.append("  [%d] %s" % (i, p["issue"]))
            lines.append("      %s" % p["detail"])
            lines.append("      FIX: %s" % p["fix"])
            lines.append("")
    else:
        lines.append("  NO PROBLEMS DETECTED")
        lines.append("")

    lines.append("  HEALTHY ITEMS:")
    for item in ok_items:
        lines.append("    [OK] %s" % item)

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Explain Mode — Metric Glossary
# ═══════════════════════════════════════════════════════════════════════════════
def generate_glossary():
    """Generate plain-English metric explanations."""
    return """======================================================================
  METRIC GLOSSARY — What Every Number Means
  Generated: %s
======================================================================

  PROFIT FACTOR (PF)
  ──────────────────
  What it is:  Total money won / Total money lost
  How to read:
    PF = 2.0  →  You make $2 for every $1 you lose
    PF = 1.0  →  Break-even (winning and losing same amount)
    PF = 0.5  →  You lose $2 for every $1 you make
    PF > 1.5  →  Good. Consistent edge.
    PF > 2.0  →  Very good. Strong strategy.
    PF > 3.0  →  Suspicious. Might be overfitting or small sample.
    PF = 99   →  Artifact. All losers had near-zero losses. Ignore.
    PF = INF  →  No losing trades at all (tiny sample, unreliable).

  WIN RATE (WR)
  ─────────────
  What it is:  %% of trades that made money
  How to read:
    WR = 60%%  →  6 of 10 trades win. Good if winners ≈ losers in size.
    WR = 40%%  →  4 of 10 win. OK if winners are 2-3x larger than losers.
    WR = 30%%  →  3 of 10 win. Needs very large winners to compensate.
  Why it matters:
    High WR + low PF = many small wins wiped out by a few big losses
    Low WR + high PF = few big wins that more than cover many small losses
    You want BOTH above their thresholds (WR>40%% and PF>1.2 is healthy)

  NET FILTER VALUE (NFV)
  ──────────────────────
  What it is:  (Losses avoided by blocking) - (Profits missed by blocking)
  How to read:
    NFV > 0   →  The filter helps. It blocks more bad trades than good ones.
    NFV < 0   →  The filter hurts. It's blocking too many good trades.
    NFV >> 0  →  Strong filter. Keep it.
  Example:
    Filter blocks 100 trades. 70 would have lost $500 each = $35,000 avoided.
    30 would have won $300 each = $9,000 missed.
    NFV = $35,000 - $9,000 = $26,000 (filter saves $26K net)

  MAX DRAWDOWN (Max DD)
  ─────────────────────
  What it is:  The deepest hole your account fell into (peak-to-trough)
  How to read:
    Even profitable strategies have drawdowns.
    If Max DD > 2x Total Profit → the ride was very bumpy
    If Max DD > 5x Total Profit → dangerously volatile
  Why it matters:
    A strategy that makes $10K but drops $50K first is psychologically
    and financially dangerous — you might quit before recovery.

  MFE (Maximum Favorable Excursion)
  ─────────────────────────────────
  What it is:  The BEST the trade got before it closed
  How to read:
    High MFE + negative PnL = "it went up but you didn't exit in time"
    This means your EXIT timing is wrong, not your entry.
    Fix: Tighten trailing stop or reduce hold time.

  MAE (Maximum Adverse Excursion)
  ────────────────────────────────
  What it is:  The WORST the trade got before it closed
  How to read:
    High MAE = "the trade went against you before recovering (or not)"
    High MAE + positive PnL = you survived the dip (scary but worked)
    High MAE + negative PnL = the dip didn't recover (stop was too wide)
    Fix: Tighten stops or improve entry timing to reduce initial drawdown.

  MFE/MAE RATIO
  ──────────────
  What it is:  How far the trade goes FOR you vs AGAINST you
  How to read:
    MFE/MAE > 2.0  →  Favorable. Trades run 2x further up than down.
    MFE/MAE = 1.0  →  Neutral. As much upside as downside.
    MFE/MAE < 1.0  →  Unfavorable. More downside than upside.

  TRADE REDUCTION
  ────────────────
  What it is:  %% of signals blocked by the filter
  How to read:
    <20%%  →  Light filtering. Most signals pass.
    50-70%% →  Moderate. Selective.
    >80%%  →  Aggressive. Only the best signals pass.
  Warning:
    If you block 90%% of trades, you better be SURE the 10%% that
    pass are genuinely better. Check the missed trades analysis.

  VOLATILITY_1M
  ──────────────
  What it is:  Standard deviation of 1-minute log returns (in %%)
  How to read:
    < 0.3%%  →  Low. Stock is barely moving. Signals are likely noise.
    0.3-0.8%% →  Medium. Normal trading range. Signals are plausible.
    > 0.8%%  →  High. Stock is moving fast. Best conditions for momentum.

  ORDER FLOW IMBALANCE (OFI)
  ──────────────────────────
  What it is:  (upticks - downticks) / total ticks in 30-second window
  How to read:
    OFI > 0.3   →  Strong buying pressure (more upticks than downticks)
    OFI = 0     →  Balanced (no clear direction)
    OFI < -0.2  →  Selling pressure (more downticks)
  Limitation:
    This is a TICK-DIRECTION PROXY, not true L2 order flow.
    It's the best approximation available without Level 2 data.

  BID-ASK SPREAD
  ───────────────
  What it is:  (Ask - Bid) / Midpoint, expressed as %%
  How to read:
    < 0.1%%  →  Extremely tight. Institutional-grade liquidity.
    < 0.3%%  →  Tight. Good for momentum trading.
    0.3-0.6%% →  Normal. Acceptable but eats into small gains.
    > 0.6%%  →  Wide. Trading costs are high. Avoid unless strong signal.
    > 1.0%%  →  Very wide. Illiquid stock. High risk of slippage.

  SESSION LABELS
  ──────────────
  premarket   →  Before 9:30am ET. Lower volume, wider spreads, but early movers.
  open        →  9:30-10:30am ET. Highest volume. Most volatile. Best liquidity.
  midday      →  10:30am-1:30pm ET. Volume drops. Chop zone. Many false signals.
  power_hour  →  1:30-4:00pm ET. Volume picks up but historically worst for momentum.

  REGIME LABELS
  ─────────────
  LOW_VOLATILITY  →  Market is quiet. Momentum signals are unreliable.
  RANGE_BOUND     →  Market is choppy. Some signals work if carefully filtered.
  TRENDING        →  Market has direction. Best conditions for momentum.
  HIGH_VOLATILITY →  Market is chaotic. Big moves but also big whipsaws.

======================================================================
  HOW TO USE THIS GLOSSARY
======================================================================

  When you see a report, ask yourself:

  1. Is PF > 1.0?
     YES → The strategy makes money. Keep going.
     NO  → The strategy loses money. Something needs fixing.

  2. Is WR > 35%%?
     YES → Signal quality is acceptable.
     NO  → Too many false signals. Check ignition/pressure studies.

  3. Is NFV > 0?
     YES → The filter is helping. Keep it.
     NO  → The filter is hurting. Consider relaxing or removing.

  4. Is Max DD < 2x Total Profit?
     YES → Risk is manageable.
     NO  → Too risky. Tighten stops or reduce position sizes.

  5. Is MFE >> MAE?
     YES → Good entries. The trade moves in your favor more than against.
     NO  → Bad entries or timing. Review entry_offset_optimizer.

  6. Do results hold across 5+ days?
     YES → Ready for deployment review.
     NO  → Keep collecting data. Single-day results are unreliable.

======================================================================
""" % datetime.now().strftime("%Y-%m-%d %H:%M")


# ═══════════════════════════════════════════════════════════════════════════════
# API-callable function
# ═══════════════════════════════════════════════════════════════════════════════
def get_briefing_json():
    """Return briefing data as structured JSON for the API."""
    pipeline = load_json(PIPELINE_RESULTS)
    scorecard = load_json(SCORECARD_DATA)
    info = get_available_data()

    result = {
        "generated": datetime.now().isoformat(),
        "data_available": info,
        "briefing_text": generate_briefing(),
        "diagnostic_text": generate_diagnostic(),
    }

    if pipeline and pipeline.get("regime", {}).get("status") == "OK":
        base = pipeline["regime"]["baseline"]
        filt = pipeline["regime"]["filtered"]
        result["summary"] = {
            "date": pipeline["date"],
            "baseline_profitable": base["pf"] >= 1.0,
            "filter_helps": (isinstance(filt["pf"], (int, float))
                             and isinstance(base["pf"], (int, float))
                             and filt["pf"] > base["pf"]),
            "baseline_pf": base["pf"],
            "filtered_pf": filt["pf"],
            "net_filter_value": pipeline["regime"].get("net_filter_value", 0),
            "deployment_ready": (scorecard and len(scorecard.get("days", [])) >= 5),
            "days_of_data": len(scorecard.get("days", [])) if scorecard else 0,
        }
        result["actions"] = _generate_actions(pipeline, scorecard, info)

    # Sector data
    if pipeline:
        sector_cls = pipeline.get("sector_classification")
        sector_heat = pipeline.get("sector_heat")
        if sector_cls:
            result["sector_classification"] = sector_cls
        if sector_heat:
            result["sector_heat"] = sector_heat

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Research Assistant — plain-English research interpretation")
    parser.add_argument("--mode", default="briefing",
                        choices=["briefing", "diagnose", "explain"],
                        help="briefing=daily summary, diagnose=find problems, explain=metric glossary")
    args = parser.parse_args()

    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if args.mode == "briefing":
        print(generate_briefing())
    elif args.mode == "diagnose":
        print(generate_diagnostic())
    elif args.mode == "explain":
        print(generate_glossary())


if __name__ == "__main__":
    main()
