#!/usr/bin/env python3
"""
Daily Microstructure Discovery Module (SuperBot Research)
READ ONLY - no production changes.

Discovers which microstructure conditions predict positive expectancy,
then ranks today's symbols by microstructure quality.

Usage:
    python -m ai.research.microstructure_discovery --date 2026-03-03 --session PM,RTH1

Studies:
  A) Feature extractor - per-symbol microstructure stats
  B) Scoring model - rank by microstructure quality
  C) Daily preferred set - top/bottom with explanations
  D) Validation loop - compare top vs bottom forward returns
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median, stdev, mean

# === PATHS ===
SUPERBOT_ROOT = Path("C:/AI_Bot_Research/MORPHEUS_SUPERBOT")
QUOTES_CACHE_DIR = SUPERBOT_ROOT / "engine" / "cache" / "quotes"
REPLAY_JSON = SUPERBOT_ROOT / "engine" / "output" / "containment_v2_replay_2026-03-03.json"
EDGE_JSON = SUPERBOT_ROOT / "engine" / "output" / "edge_preservation_v2_2026-03-03.json"
CONFIG_PATH = SUPERBOT_ROOT / "configs" / "microstructure_profiles.json"
OUTPUT_DIR = SUPERBOT_ROOT / "engine" / "output"
COMMS_PATH = SUPERBOT_ROOT / "comms" / "outbox_chatgpt.json"

# === EXIT MODEL (same as edge preservation) ===
STOP_PCT = -1.0
TRAIL_ACTIVATE = 0.8
TRAIL_DISTANCE = 0.4
EXIT_WINDOW_SEC = 300

# === SESSION BOUNDARIES (UTC) for 2026-03-03 ===
def get_session_epochs(date_str):
    """Return session epoch boundaries for a given date."""
    y, m, d = map(int, date_str.split("-"))
    base = datetime(y, m, d, tzinfo=timezone.utc)
    return {
        "PM": (
            base.replace(hour=9, minute=0).timestamp(),
            base.replace(hour=14, minute=30).timestamp(),
        ),
        "RTH1": (
            base.replace(hour=14, minute=30).timestamp(),
            base.replace(hour=15, minute=30).timestamp(),
        ),
        "RTH_FULL": (
            base.replace(hour=14, minute=30).timestamp(),
            base.replace(hour=21, minute=0).timestamp(),
        ),
        "ALL": (
            base.replace(hour=9, minute=0).timestamp(),
            base.replace(hour=21, minute=0).timestamp(),
        ),
    }


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_quotes(symbol):
    """Load cached quotes, return list of dicts with epoch/price/bid/ask."""
    cache_file = QUOTES_CACHE_DIR / f"{symbol}_quotes.json"
    if not cache_file.exists():
        return []
    with open(cache_file) as f:
        data = json.load(f)
    result = []
    for q in data.get("quotes", []):
        price = q.get("last")
        bid = q.get("bid", 0) or 0
        ask = q.get("ask", 0) or 0
        if price is None or price <= 0:
            if bid > 0 and ask > 0:
                price = (bid + ask) / 2
            else:
                continue
        result.append({
            "epoch": q["epoch"],
            "price": price,
            "bid": bid,
            "ask": ask,
        })
    return result


def filter_quotes_session(quotes, session_start, session_end):
    """Filter quotes to a session window."""
    return [q for q in quotes if session_start <= q["epoch"] < session_end]


def discover_symbols(date_str):
    """Discover candidate symbols from available quote caches."""
    symbols = []
    for f in QUOTES_CACHE_DIR.glob("*_quotes.json"):
        sym = f.stem.replace("_quotes", "")
        symbols.append(sym)
    return sorted(symbols)


# ========================================================================
# PART A: FEATURE EXTRACTOR
# ========================================================================
def extract_features(symbol, quotes, signals_for_symbol=None):
    """
    Compute microstructure features for a symbol from its quote stream.
    Returns a dict of feature values.
    """
    if len(quotes) < 10:
        return None

    features = {}

    # --- SPREAD METRICS ---
    spreads = []
    for q in quotes:
        if q["bid"] > 0 and q["ask"] > 0 and q["ask"] > q["bid"]:
            mid = (q["bid"] + q["ask"]) / 2
            spread_pct = (q["ask"] - q["bid"]) / mid * 100
            spreads.append(spread_pct)

    if spreads:
        spreads_sorted = sorted(spreads)
        features["spread_median"] = round(median(spreads), 4)
        features["spread_mean"] = round(mean(spreads), 4)
        features["spread_p90"] = round(spreads_sorted[int(len(spreads_sorted) * 0.9)], 4)
        features["spread_std"] = round(stdev(spreads), 4) if len(spreads) > 1 else 0.0
        # Spread stability: coefficient of variation
        features["spread_stability"] = round(features["spread_std"] / features["spread_mean"], 4) if features["spread_mean"] > 0 else 0.0
    else:
        features["spread_median"] = None
        features["spread_mean"] = None
        features["spread_p90"] = None
        features["spread_std"] = None
        features["spread_stability"] = None

    # --- QUOTE FRESHNESS / INTER-QUOTE GAPS ---
    gaps_ms = []
    for i in range(1, len(quotes)):
        gap = (quotes[i]["epoch"] - quotes[i-1]["epoch"]) * 1000
        gaps_ms.append(gap)

    if gaps_ms:
        gaps_sorted = sorted(gaps_ms)
        features["quote_gap_median_ms"] = round(median(gaps_ms), 1)
        features["quote_gap_mean_ms"] = round(mean(gaps_ms), 1)
        features["quote_gap_p90_ms"] = round(gaps_sorted[int(len(gaps_sorted) * 0.9)], 1)

        stale_threshold = 2000  # ms
        stale_bursts = [g for g in gaps_ms if g > stale_threshold]
        features["stale_burst_count"] = len(stale_bursts)
        features["stale_burst_rate"] = round(len(stale_bursts) / len(gaps_ms) * 100, 2)

        gap_warning = 500  # ms
        big_gaps = [g for g in gaps_ms if g > gap_warning]
        features["gap_frequency"] = round(len(big_gaps) / len(gaps_ms) * 100, 2)

        # Burstiness: std/mean of inter-quote times
        if mean(gaps_ms) > 0 and len(gaps_ms) > 1:
            features["burstiness"] = round(stdev(gaps_ms) / mean(gaps_ms), 4)
        else:
            features["burstiness"] = 0.0
    else:
        for k in ["quote_gap_median_ms", "quote_gap_mean_ms", "quote_gap_p90_ms",
                   "stale_burst_count", "stale_burst_rate", "gap_frequency", "burstiness"]:
            features[k] = None

    # --- TICK RATE ---
    if len(quotes) >= 2:
        total_sec = quotes[-1]["epoch"] - quotes[0]["epoch"]
        features["tick_rate"] = round(len(quotes) / total_sec, 4) if total_sec > 0 else 0.0
        features["total_ticks"] = len(quotes)
        features["total_seconds"] = round(total_sec, 1)
    else:
        features["tick_rate"] = 0.0
        features["total_ticks"] = len(quotes)
        features["total_seconds"] = 0.0

    # --- VOLATILITY (1s, 5s, 30s return std) ---
    for window_sec in [1, 5, 30]:
        returns = _compute_interval_returns(quotes, window_sec)
        if returns and len(returns) > 1:
            features[f"volatility_{window_sec}s"] = round(stdev(returns) * 100, 4)  # as pct
        else:
            features[f"volatility_{window_sec}s"] = None

    # --- TREND PERSISTENCE ---
    trend_window = 10  # seconds
    features["trend_persistence"] = _compute_trend_persistence(quotes, trend_window)

    # --- ENTRY LATENESS PROXY ---
    if signals_for_symbol:
        lateness_values = []
        for sig in signals_for_symbol:
            sig_epoch = _parse_ts(sig["timestamp"])
            lookback = 60  # seconds
            window_quotes = [q for q in quotes
                             if sig_epoch - lookback <= q["epoch"] <= sig_epoch]
            if window_quotes:
                local_low = min(q["price"] for q in window_quotes)
                entry_price = sig["entry_price"]
                if local_low > 0:
                    lateness_pct = (entry_price - local_low) / local_low * 100
                    lateness_values.append(lateness_pct)
        features["entry_lateness_mean"] = round(mean(lateness_values), 4) if lateness_values else None
        features["entry_lateness_median"] = round(median(lateness_values), 4) if lateness_values else None
    else:
        features["entry_lateness_mean"] = None
        features["entry_lateness_median"] = None

    # --- SLIPPAGE / PROFIT MARGIN PROXY ---
    if features["spread_median"] is not None:
        effective_cost = features["spread_median"] / 2 + 0.15  # spread/2 + slippage
        features["effective_cost_pct"] = round(effective_cost, 4)
        # Need expected move - use volatility as proxy
        vol_5s = features.get("volatility_5s")
        if vol_5s and vol_5s > 0:
            # Expected move ~ 5-minute vol projection (sqrt scaling from 5s)
            expected_move = vol_5s * math.sqrt(60)  # 5s -> 5min = 60 intervals
            features["profit_margin_ratio"] = round(expected_move / effective_cost, 4) if effective_cost > 0 else None
        else:
            features["profit_margin_ratio"] = None
    else:
        features["effective_cost_pct"] = None
        features["profit_margin_ratio"] = None

    return features


def _compute_interval_returns(quotes, interval_sec):
    """Compute returns at fixed intervals by sampling nearest quotes."""
    if len(quotes) < 2:
        return []

    start = quotes[0]["epoch"]
    end = quotes[-1]["epoch"]
    returns = []
    t = start
    prev_price = None

    # Build epoch->price index for fast lookup
    idx = 0
    while t <= end:
        # Find nearest quote at or after t
        while idx < len(quotes) - 1 and quotes[idx]["epoch"] < t:
            idx += 1
        price = quotes[idx]["price"]
        if prev_price is not None and prev_price > 0:
            ret = (price - prev_price) / prev_price
            returns.append(ret)
        prev_price = price
        t += interval_sec

    return returns


def _compute_trend_persistence(quotes, window_sec):
    """Fraction of consecutive windows showing higher highs or higher lows."""
    if len(quotes) < 5:
        return None

    start = quotes[0]["epoch"]
    end = quotes[-1]["epoch"]
    windows = []
    t = start

    while t + window_sec <= end:
        window_quotes = [q for q in quotes if t <= q["epoch"] < t + window_sec]
        if window_quotes:
            high = max(q["price"] for q in window_quotes)
            low = min(q["price"] for q in window_quotes)
            windows.append({"high": high, "low": low})
        t += window_sec

    if len(windows) < 2:
        return None

    higher_highs = 0
    higher_lows = 0
    for i in range(1, len(windows)):
        if windows[i]["high"] > windows[i-1]["high"]:
            higher_highs += 1
        if windows[i]["low"] > windows[i-1]["low"]:
            higher_lows += 1

    total_comparisons = (len(windows) - 1) * 2
    if total_comparisons == 0:
        return None

    return round((higher_highs + higher_lows) / total_comparisons, 4)


def _parse_ts(ts_str):
    """HH:MM:SS -> epoch for 2026-03-03."""
    h, m, s = ts_str.split(":")
    return datetime(2026, 3, 3, int(h), int(m), int(s), tzinfo=timezone.utc).timestamp()


# ========================================================================
# PART B: SCORING MODEL
# ========================================================================
def score_symbols(features_by_symbol, config):
    """
    Compute z-scored weighted rank for each symbol.
    Returns list of {symbol, score, feature_contributions, classification}.
    """
    weights = config["scoring_weights"]

    # Scoring features (subset that maps to weights)
    score_features = [
        "spread_median", "spread_p90", "spread_stability",
        "tick_rate", "stale_burst_rate", "gap_frequency",
        "volatility_5s", "trend_persistence", "entry_lateness_mean",
        "profit_margin_feasibility", "quote_gap_p90_ms",
    ]

    # Map internal feature names to weight keys
    feature_to_weight = {
        "spread_median": "spread_median",
        "spread_p90": "spread_p90",
        "spread_stability": "spread_stability",
        "tick_rate": "tick_rate",
        "stale_burst_rate": "stale_burst_rate",
        "gap_frequency": "gap_frequency",
        "volatility_5s": "volatility_5s",
        "trend_persistence": "trend_persistence",
        "entry_lateness_mean": "entry_lateness",
        "profit_margin_ratio": "profit_margin_feasibility",
        "quote_gap_p90_ms": "quote_freshness_p90",
    }

    # Collect raw values per feature across all symbols
    symbols = sorted(features_by_symbol.keys())
    raw_matrix = {}  # feature -> {symbol: value}
    for feat in feature_to_weight:
        raw_matrix[feat] = {}
        for sym in symbols:
            val = features_by_symbol[sym].get(feat)
            if val is not None:
                raw_matrix[feat][sym] = val

    # Z-score normalization per feature
    z_matrix = {}  # feature -> {symbol: z_score}
    for feat in feature_to_weight:
        vals = list(raw_matrix[feat].values())
        if len(vals) < 2:
            z_matrix[feat] = {sym: 0.0 for sym in symbols}
            continue
        mu = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 1.0
        if sd == 0:
            sd = 1.0
        z_matrix[feat] = {}
        for sym in symbols:
            if sym in raw_matrix[feat]:
                z_matrix[feat][sym] = (raw_matrix[feat][sym] - mu) / sd
            else:
                z_matrix[feat][sym] = 0.0  # missing = neutral

    # Compute weighted score per symbol
    results = []
    for sym in symbols:
        total_score = 0.0
        contributions = {}
        for feat, weight_key in feature_to_weight.items():
            w = weights.get(weight_key, 0)
            z = z_matrix[feat].get(sym, 0.0)
            contrib = w * z
            total_score += contrib
            contributions[feat] = {
                "raw": raw_matrix[feat].get(sym),
                "z_score": round(z, 3),
                "weight": w,
                "contribution": round(contrib, 3),
            }

        # Classification
        strong_thresh = config["classification"]["strong_edge_threshold"]
        neg_thresh = config["classification"]["negative_edge_threshold"]
        if total_score >= strong_thresh:
            classification = "PREFERRED"
        elif total_score <= neg_thresh:
            classification = "AVOID"
        else:
            classification = "NEUTRAL"

        results.append({
            "symbol": sym,
            "score": round(total_score, 3),
            "classification": classification,
            "contributions": contributions,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ========================================================================
# PART C: PREFERRED SET DISCOVERY
# ========================================================================
def discover_preferred_set(scored_symbols, features_by_symbol):
    """Generate top/bottom sets with explanations."""

    top_n = min(10, len(scored_symbols))
    bottom_n = min(10, len(scored_symbols))

    top_set = scored_symbols[:top_n]
    bottom_set = scored_symbols[-bottom_n:]

    def get_top_reasons(sym_result, n=3):
        """Get top N contributing features for a symbol's score."""
        contribs = sym_result["contributions"]
        sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]["contribution"]), reverse=True)
        reasons = []
        for feat, data in sorted_contribs[:n]:
            direction = "+" if data["contribution"] > 0 else "-"
            raw_str = f"{data['raw']:.3f}" if data["raw"] is not None else "N/A"
            reasons.append({
                "feature": feat,
                "raw_value": data["raw"],
                "z_score": data["z_score"],
                "contribution": data["contribution"],
                "direction": direction,
                "explanation": _explain_feature(feat, data["raw"], data["contribution"]),
            })
        return reasons

    preferred = []
    for s in top_set:
        preferred.append({
            "symbol": s["symbol"],
            "score": s["score"],
            "classification": s["classification"],
            "top_reasons": get_top_reasons(s),
        })

    avoid = []
    for s in bottom_set:
        avoid.append({
            "symbol": s["symbol"],
            "score": s["score"],
            "classification": s["classification"],
            "top_reasons": get_top_reasons(s),
        })

    return {"preferred": preferred, "avoid": avoid}


def _explain_feature(feature, raw_value, contribution):
    """Human-readable explanation for a feature contribution."""
    if raw_value is None:
        return "No data available"

    explanations = {
        "spread_median": f"Median spread is {raw_value:.3f}% ({'tight' if raw_value < 0.4 else 'wide'})",
        "spread_p90": f"90th percentile spread is {raw_value:.3f}% ({'stable' if raw_value < 0.6 else 'spiky'})",
        "spread_stability": f"Spread coefficient of variation is {raw_value:.3f} ({'consistent' if raw_value < 0.5 else 'erratic'})",
        "tick_rate": f"Quote rate is {raw_value:.2f} ticks/sec ({'active' if raw_value > 0.3 else 'sparse'})",
        "stale_burst_rate": f"Stale burst rate is {raw_value:.1f}% ({'clean' if raw_value < 1 else 'problematic'})",
        "gap_frequency": f"Gap frequency (>500ms) is {raw_value:.1f}% ({'smooth' if raw_value < 20 else 'choppy'})",
        "volatility_5s": f"5s volatility is {raw_value:.4f}% ({'calm' if raw_value < 0.1 else 'volatile'})",
        "trend_persistence": f"Trend persistence is {raw_value:.3f} ({'trending' if raw_value > 0.55 else 'choppy'})",
        "entry_lateness_mean": f"Mean entry lateness is {raw_value:.3f}% above local low ({'timely' if raw_value < 0.5 else 'late'})",
        "profit_margin_ratio": f"Profit margin ratio is {raw_value:.2f}x ({'feasible' if raw_value > 2 else 'tight'})",
        "quote_gap_p90_ms": f"P90 quote gap is {raw_value:.0f}ms ({'fresh' if raw_value < 1000 else 'stale'})",
    }
    return explanations.get(feature, f"Value: {raw_value}")


# ========================================================================
# PART D: VALIDATION LOOP
# ========================================================================
def validate_sets(preferred_set, avoid_set, all_signals, symbol_quotes):
    """
    Run forward-window replay on preferred vs avoid sets.
    Compare WR/avg_return for 1m/5m/10m.
    """
    preferred_syms = set(s["symbol"] for s in preferred_set)
    avoid_syms = set(s["symbol"] for s in avoid_set)

    preferred_signals = [s for s in all_signals if s["symbol"] in preferred_syms]
    avoid_signals = [s for s in all_signals if s["symbol"] in avoid_syms]

    def evaluate_set(signals, label):
        """Run exit model on a set of signals."""
        windows = [60, 300, 600]
        window_labels = {60: "1m", 300: "5m", 600: "10m"}

        # Exit model results
        exit_returns = []
        window_returns = {w: [] for w in windows}

        for sig in signals:
            quotes = symbol_quotes.get(sig["symbol"], [])
            if not quotes:
                continue

            entry = sig["entry_price"]
            sig_epoch = _parse_ts(sig["timestamp"])

            # Exit model (5m)
            fwd_5m = [q for q in quotes if q["epoch"] > sig_epoch and q["epoch"] <= sig_epoch + 300]
            if fwd_5m:
                exit_r = _simulate_exit(fwd_5m, entry, sig_epoch)
                if exit_r["exit_return_pct"] is not None:
                    exit_returns.append(exit_r["exit_return_pct"])

            # Window returns
            for w in windows:
                fwd = [q for q in quotes if q["epoch"] > sig_epoch and q["epoch"] <= sig_epoch + w]
                if fwd:
                    close_ret = (fwd[-1]["price"] - entry) / entry * 100
                    window_returns[w].append(close_ret)

        result = {
            "label": label,
            "n_signals": len(signals),
            "n_evaluated": len(exit_returns),
            "exit_model": {},
            "windows": {},
        }

        if exit_returns:
            wins = [r for r in exit_returns if r > 0]
            result["exit_model"] = {
                "win_rate": round(len(wins) / len(exit_returns) * 100, 1),
                "avg_return": round(mean(exit_returns), 4),
                "total_return": round(sum(exit_returns), 4),
            }

        for w in windows:
            rets = window_returns[w]
            if rets:
                wins = [r for r in rets if r > 0]
                result["windows"][window_labels[w]] = {
                    "n": len(rets),
                    "win_rate": round(len(wins) / len(rets) * 100, 1),
                    "avg_return": round(mean(rets), 4),
                }
            else:
                result["windows"][window_labels[w]] = {"n": 0, "win_rate": None, "avg_return": None}

        return result

    preferred_result = evaluate_set(preferred_signals, "PREFERRED")
    avoid_result = evaluate_set(avoid_signals, "AVOID")

    return {"preferred": preferred_result, "avoid": avoid_result}


def _simulate_exit(forward_quotes, entry_price, start_epoch):
    """Simplified exit model."""
    trail_activated = False
    trail_high = entry_price
    for q in forward_quotes:
        price = q["price"]
        ret = (price - entry_price) / entry_price * 100
        if ret <= STOP_PCT:
            return {"exit_return_pct": round(ret, 4), "exit_type": "STOP"}
        if not trail_activated and ret >= TRAIL_ACTIVATE:
            trail_activated = True
            trail_high = price
        if trail_activated:
            if price > trail_high:
                trail_high = price
            if price <= trail_high * (1 - TRAIL_DISTANCE / 100):
                return {"exit_return_pct": round(ret, 4), "exit_type": "TRAIL_STOP"}
    last = forward_quotes[-1]
    final_ret = (last["price"] - entry_price) / entry_price * 100
    return {"exit_return_pct": round(final_ret, 4), "exit_type": "TIME_EXIT"}


# ========================================================================
# REPORT GENERATION
# ========================================================================
def generate_features_report(features_by_symbol, date_str):
    """Save features JSON."""
    out_path = OUTPUT_DIR / f"microstructure_features_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "microstructure_features",
            "symbols": features_by_symbol,
        }, f, indent=2)
    return out_path


def generate_rank_report(scored_symbols, date_str):
    """Save rank JSON."""
    out_path = OUTPUT_DIR / f"microstructure_rank_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "microstructure_rank",
            "rankings": scored_symbols,
        }, f, indent=2)
    return out_path


def generate_preferred_report(preferred_data, date_str):
    """Generate preferred symbols markdown."""
    out_path = OUTPUT_DIR / f"daily_preferred_symbols_{date_str}.md"
    lines = []
    lines.append(f"# Daily Preferred Symbols - {date_str}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append("**Method:** Microstructure feature scoring (z-score weighted)")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Preferred Symbols (Trade Candidates)")
    lines.append("")
    lines.append("| Rank | Symbol | Score | Classification | Top Reason |")
    lines.append("|------|--------|-------|----------------|------------|")
    for i, s in enumerate(preferred_data["preferred"], 1):
        top_r = s["top_reasons"][0] if s["top_reasons"] else {"explanation": "N/A"}
        lines.append(f"| {i} | {s['symbol']} | {s['score']:+.3f} | {s['classification']} | {top_r['explanation']} |")

    lines.append("")
    lines.append("### Detailed Reasons")
    lines.append("")
    for s in preferred_data["preferred"]:
        lines.append(f"**{s['symbol']}** (score: {s['score']:+.3f})")
        for r in s["top_reasons"]:
            lines.append(f"  - [{r['direction']}] {r['explanation']} (contrib: {r['contribution']:+.3f})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Avoid Symbols (Negative Microstructure)")
    lines.append("")
    lines.append("| Rank | Symbol | Score | Classification | Top Reason |")
    lines.append("|------|--------|-------|----------------|------------|")
    for i, s in enumerate(preferred_data["avoid"], 1):
        top_r = s["top_reasons"][0] if s["top_reasons"] else {"explanation": "N/A"}
        lines.append(f"| {i} | {s['symbol']} | {s['score']:+.3f} | {s['classification']} | {top_r['explanation']} |")

    lines.append("")
    lines.append("### Detailed Reasons")
    lines.append("")
    for s in preferred_data["avoid"]:
        lines.append(f"**{s['symbol']}** (score: {s['score']:+.3f})")
        for r in s["top_reasons"]:
            lines.append(f"  - [{r['direction']}] {r['explanation']} (contrib: {r['contribution']:+.3f})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*This analysis is research-only. No production changes applied.*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def generate_validation_report(validation, preferred_data, scored_symbols, date_str):
    """Generate validation markdown comparing preferred vs avoid."""
    out_path = OUTPUT_DIR / f"microstructure_validation_{date_str}.md"

    pref = validation["preferred"]
    avoid = validation["avoid"]

    lines = []
    lines.append(f"# Microstructure Validation - {date_str}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append("**Method:** Forward-window replay, preferred vs avoid sets")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary comparison
    lines.append("## Preferred vs Avoid: Exit Model")
    lines.append("")
    lines.append("| Metric | Preferred | Avoid | Delta |")
    lines.append("|--------|-----------|-------|-------|")

    p_em = pref.get("exit_model", {})
    a_em = avoid.get("exit_model", {})

    for metric in ["win_rate", "avg_return", "total_return"]:
        p_val = p_em.get(metric)
        a_val = a_em.get(metric)
        p_str = f"{p_val:.1f}%" if metric == "win_rate" and p_val is not None else (f"{p_val:+.4f}%" if p_val is not None else "-")
        a_str = f"{a_val:.1f}%" if metric == "win_rate" and a_val is not None else (f"{a_val:+.4f}%" if a_val is not None else "-")
        if p_val is not None and a_val is not None:
            delta = p_val - a_val
            d_str = f"{delta:+.4f}%"
        else:
            d_str = "-"
        lines.append(f"| {metric} | {p_str} | {a_str} | {d_str} |")

    lines.append(f"| signals | {pref['n_signals']} | {avoid['n_signals']} | - |")
    lines.append("")

    # Window comparison
    lines.append("## Preferred vs Avoid: Forward Windows")
    lines.append("")
    lines.append("| Window | Pref WR | Pref Avg | Avoid WR | Avoid Avg | WR Delta | Return Delta |")
    lines.append("|--------|---------|----------|----------|-----------|----------|--------------|")

    for w_label in ["1m", "5m", "10m"]:
        pw = pref.get("windows", {}).get(w_label, {})
        aw = avoid.get("windows", {}).get(w_label, {})
        p_wr = f"{pw.get('win_rate', 0):.1f}%" if pw.get("win_rate") is not None else "-"
        p_ar = f"{pw.get('avg_return', 0):+.4f}%" if pw.get("avg_return") is not None else "-"
        a_wr = f"{aw.get('win_rate', 0):.1f}%" if aw.get("win_rate") is not None else "-"
        a_ar = f"{aw.get('avg_return', 0):+.4f}%" if aw.get("avg_return") is not None else "-"

        wr_delta = "-"
        ret_delta = "-"
        if pw.get("win_rate") is not None and aw.get("win_rate") is not None:
            wr_delta = f"{pw['win_rate'] - aw['win_rate']:+.1f}pp"
        if pw.get("avg_return") is not None and aw.get("avg_return") is not None:
            ret_delta = f"{pw['avg_return'] - aw['avg_return']:+.4f}%"

        lines.append(f"| {w_label} | {p_wr} | {p_ar} | {a_wr} | {a_ar} | {wr_delta} | {ret_delta} |")

    lines.append("")

    # Validation verdict
    p_1m_wr = pref.get("windows", {}).get("1m", {}).get("win_rate")
    a_1m_wr = avoid.get("windows", {}).get("1m", {}).get("win_rate")
    p_1m_ret = pref.get("windows", {}).get("1m", {}).get("avg_return")
    a_1m_ret = avoid.get("windows", {}).get("1m", {}).get("avg_return")

    lines.append("## Validation Verdict")
    lines.append("")

    if p_1m_ret is not None and a_1m_ret is not None:
        if p_1m_ret > a_1m_ret:
            lines.append(f"**VALIDATED**: Preferred set outperforms avoid set at 1m window ({p_1m_ret:+.4f}% vs {a_1m_ret:+.4f}%)")
        else:
            lines.append(f"**NOT VALIDATED**: Avoid set matches or beats preferred set at 1m ({a_1m_ret:+.4f}% vs {p_1m_ret:+.4f}%)")
    else:
        lines.append("**INSUFFICIENT DATA** for validation.")

    p_em_wr = p_em.get("win_rate")
    a_em_wr = a_em.get("win_rate")
    p_em_ret = p_em.get("avg_return")
    a_em_ret = a_em.get("avg_return")

    if p_em_ret is not None and a_em_ret is not None:
        if p_em_ret > a_em_ret:
            lines.append(f"**EXIT MODEL VALIDATED**: Preferred {p_em_ret:+.4f}% vs Avoid {a_em_ret:+.4f}% (delta: {p_em_ret - a_em_ret:+.4f}%)")
        else:
            lines.append(f"**EXIT MODEL NOT VALIDATED**: Preferred {p_em_ret:+.4f}% vs Avoid {a_em_ret:+.4f}%")

    lines.append("")

    # Score vs outcome correlation
    lines.append("## Score vs Outcome Correlation")
    lines.append("")
    lines.append("| Symbol | Micro Score | Exit Avg Return | Correlated? |")
    lines.append("|--------|------------|-----------------|-------------|")

    # Load edge preservation outcomes
    try:
        edge_data = json.load(open(EDGE_JSON))
        sym_outcomes = edge_data["summary"]["symbol_breakdown"]
    except Exception:
        sym_outcomes = {}

    for s in scored_symbols:
        sym = s["symbol"]
        outcome = sym_outcomes.get(sym, {})
        avg_ret = outcome.get("exit_avg_return")
        score = s["score"]
        if avg_ret is not None:
            correlated = "YES" if (score > 0 and avg_ret > 0) or (score < 0 and avg_ret < 0) else "NO"
            lines.append(f"| {sym} | {score:+.3f} | {avg_ret:+.4f}% | {correlated} |")
        else:
            lines.append(f"| {sym} | {score:+.3f} | N/A | - |")

    # Count correlation
    correct = 0
    total = 0
    for s in scored_symbols:
        outcome = sym_outcomes.get(s["symbol"], {})
        avg_ret = outcome.get("exit_avg_return")
        if avg_ret is not None:
            total += 1
            if (s["score"] > 0 and avg_ret > 0) or (s["score"] < 0 and avg_ret < 0):
                correct += 1

    if total > 0:
        lines.append("")
        lines.append(f"**Correlation accuracy:** {correct}/{total} ({correct/total*100:.0f}%)")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This validation is research-only. No production changes applied.*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def update_comms(preferred_data, validation, scored_symbols, date_str):
    """Append summary to comms/outbox_chatgpt.json."""
    if not COMMS_PATH.exists():
        return

    with open(COMMS_PATH) as f:
        comms = json.load(f)

    pref_syms = [s["symbol"] for s in preferred_data["preferred"] if s["classification"] == "PREFERRED"]
    avoid_syms = [s["symbol"] for s in preferred_data["avoid"] if s["classification"] == "AVOID"]

    p = validation["preferred"]
    a = validation["avoid"]
    p_em = p.get("exit_model", {})
    a_em = a.get("exit_model", {})
    p_1m = p.get("windows", {}).get("1m", {})
    a_1m = a.get("windows", {}).get("1m", {})

    body_parts = [
        f"== DAILY MICROSTRUCTURE DISCOVERY ({date_str}) ==",
        f"Analyzed {len(scored_symbols)} symbols from quote caches.",
        "",
        f"== PREFERRED SET ==",
    ]
    for s in preferred_data["preferred"]:
        reasons = "; ".join(r["explanation"] for r in s["top_reasons"][:2])
        body_parts.append(f"  {s['symbol']:>5}: score={s['score']:+.3f} [{s['classification']}] - {reasons}")

    body_parts.append("")
    body_parts.append(f"== AVOID SET ==")
    for s in preferred_data["avoid"]:
        reasons = "; ".join(r["explanation"] for r in s["top_reasons"][:2])
        body_parts.append(f"  {s['symbol']:>5}: score={s['score']:+.3f} [{s['classification']}] - {reasons}")

    body_parts.append("")
    body_parts.append(f"== VALIDATION ==")
    if p_em.get("avg_return") is not None and a_em.get("avg_return") is not None:
        body_parts.append(f"Exit model: Preferred={p_em['avg_return']:+.4f}% vs Avoid={a_em['avg_return']:+.4f}%")
        body_parts.append(f"Exit model WR: Preferred={p_em.get('win_rate',0):.1f}% vs Avoid={a_em.get('win_rate',0):.1f}%")
    if p_1m.get("avg_return") is not None and a_1m.get("avg_return") is not None:
        body_parts.append(f"1m window: Preferred={p_1m['avg_return']:+.4f}% vs Avoid={a_1m['avg_return']:+.4f}%")
    validated = "YES" if (p_em.get("avg_return", 0) or 0) > (a_em.get("avg_return", 0) or 0) else "NO"
    body_parts.append(f"Validated: {validated}")

    body_parts.append("")
    body_parts.append("== DELIVERABLES ==")
    body_parts.append(f"microstructure_features_{date_str}.json")
    body_parts.append(f"microstructure_rank_{date_str}.json")
    body_parts.append(f"daily_preferred_symbols_{date_str}.md")
    body_parts.append(f"microstructure_validation_{date_str}.md")
    body_parts.append("")
    body_parts.append("Production remains frozen. All analysis is read-only research.")

    msg = {
        "id": f"msg_chatgpt_{date_str.replace('-','')}_006",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": "claude",
        "to": "chatgpt",
        "type": "action_result",
        "subject": f"Microstructure Discovery ({date_str}): Preferred={','.join(pref_syms[:4])}, Avoid={','.join(avoid_syms[:4])}",
        "body": "\n".join(body_parts),
        "references": [
            f"engine/output/microstructure_features_{date_str}.json",
            f"engine/output/microstructure_rank_{date_str}.json",
            f"engine/output/daily_preferred_symbols_{date_str}.md",
            f"engine/output/microstructure_validation_{date_str}.md",
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
    parser = argparse.ArgumentParser(description="Daily Microstructure Discovery")
    parser.add_argument("--date", default="2026-03-03", help="Date to analyze (YYYY-MM-DD)")
    parser.add_argument("--session", default="PM,RTH1", help="Session windows (comma-separated)")
    args = parser.parse_args()

    date_str = args.date
    sessions = args.session.split(",")

    print("=" * 70)
    print(f"DAILY MICROSTRUCTURE DISCOVERY ({date_str})")
    print(f"Sessions: {sessions}")
    print("SuperBot Research Engine - READ ONLY")
    print("=" * 70)

    config = load_config()
    session_epochs = get_session_epochs(date_str)

    # Discover symbols
    symbols = discover_symbols(date_str)
    print(f"\nDiscovered {len(symbols)} symbols: {symbols}")

    # Load v2-pass signals for entry lateness computation
    print("\nLoading v2 replay signals...")
    try:
        with open(REPLAY_JSON) as f:
            replay = json.load(f)
        pass_signals = [s for s in replay["signals"] if s["v2_result"] == "PASS"]
        signals_by_symbol = defaultdict(list)
        for s in pass_signals:
            signals_by_symbol[s["symbol"]].append(s)
    except Exception as e:
        print(f"  Warning: Could not load replay data: {e}")
        pass_signals = []
        signals_by_symbol = {}

    # ================================================================
    # PART A: Feature Extraction
    # ================================================================
    print(f"\n{'='*70}")
    print("PART A: Feature Extraction")
    print(f"{'='*70}")

    features_by_symbol = {}
    for sym in symbols:
        all_quotes = load_quotes(sym)
        if not all_quotes:
            print(f"  {sym}: no quotes, skipping")
            continue

        # Filter to requested sessions
        session_quotes = []
        for sess in sessions:
            if sess in session_epochs:
                start, end = session_epochs[sess]
                sq = filter_quotes_session(all_quotes, start, end)
                session_quotes.extend(sq)

        # Deduplicate and sort
        seen = set()
        unique_quotes = []
        for q in session_quotes:
            if q["epoch"] not in seen:
                seen.add(q["epoch"])
                unique_quotes.append(q)
        unique_quotes.sort(key=lambda q: q["epoch"])

        if not unique_quotes:
            # Fallback: use all quotes
            unique_quotes = all_quotes

        sigs = signals_by_symbol.get(sym, None)
        features = extract_features(sym, unique_quotes, sigs)
        if features:
            features_by_symbol[sym] = features
            print(f"  {sym}: {features['total_ticks']} ticks, spread_med={features['spread_median']}, tick_rate={features['tick_rate']}, trend={features['trend_persistence']}")
        else:
            print(f"  {sym}: insufficient data")

    features_path = generate_features_report(features_by_symbol, date_str)
    print(f"\n  -> {features_path.name}")

    # ================================================================
    # PART B: Scoring Model
    # ================================================================
    print(f"\n{'='*70}")
    print("PART B: Microstructure Scoring")
    print(f"{'='*70}")

    scored_symbols = score_symbols(features_by_symbol, config)

    for s in scored_symbols:
        print(f"  {s['symbol']:>5}: score={s['score']:+.3f} [{s['classification']}]")

    rank_path = generate_rank_report(scored_symbols, date_str)
    print(f"\n  -> {rank_path.name}")

    # ================================================================
    # PART C: Preferred Set Discovery
    # ================================================================
    print(f"\n{'='*70}")
    print("PART C: Preferred Set Discovery")
    print(f"{'='*70}")

    preferred_data = discover_preferred_set(scored_symbols, features_by_symbol)

    print("\n  PREFERRED:")
    for s in preferred_data["preferred"]:
        if s["classification"] == "PREFERRED":
            reasons = "; ".join(r["explanation"] for r in s["top_reasons"][:2])
            print(f"    {s['symbol']:>5} ({s['score']:+.3f}): {reasons}")

    print("\n  AVOID:")
    for s in preferred_data["avoid"]:
        if s["classification"] == "AVOID":
            reasons = "; ".join(r["explanation"] for r in s["top_reasons"][:2])
            print(f"    {s['symbol']:>5} ({s['score']:+.3f}): {reasons}")

    preferred_path = generate_preferred_report(preferred_data, date_str)
    print(f"\n  -> {preferred_path.name}")

    # ================================================================
    # PART D: Validation Loop
    # ================================================================
    print(f"\n{'='*70}")
    print("PART D: Validation Loop")
    print(f"{'='*70}")

    # Load symbol quotes for validation
    symbol_quotes = {}
    for sym in symbols:
        symbol_quotes[sym] = load_quotes(sym)

    validation = validate_sets(
        preferred_data["preferred"],
        preferred_data["avoid"],
        pass_signals,
        symbol_quotes,
    )

    pref_v = validation["preferred"]
    avoid_v = validation["avoid"]

    print(f"\n  PREFERRED ({pref_v['n_signals']} signals):")
    em = pref_v.get("exit_model", {})
    if em:
        print(f"    Exit model: WR={em.get('win_rate','-')}%, avg={em.get('avg_return','-')}%, total={em.get('total_return','-')}%")
    for w in ["1m", "5m", "10m"]:
        wd = pref_v.get("windows", {}).get(w, {})
        if wd.get("win_rate") is not None:
            print(f"    {w}: WR={wd['win_rate']:.1f}%, avg={wd['avg_return']:+.4f}%")

    print(f"\n  AVOID ({avoid_v['n_signals']} signals):")
    em = avoid_v.get("exit_model", {})
    if em:
        print(f"    Exit model: WR={em.get('win_rate','-')}%, avg={em.get('avg_return','-')}%, total={em.get('total_return','-')}%")
    for w in ["1m", "5m", "10m"]:
        wd = avoid_v.get("windows", {}).get(w, {})
        if wd.get("win_rate") is not None:
            print(f"    {w}: WR={wd['win_rate']:.1f}%, avg={wd['avg_return']:+.4f}%")

    validation_path = generate_validation_report(validation, preferred_data, scored_symbols, date_str)
    print(f"\n  -> {validation_path.name}")

    # ================================================================
    # Comms Bridge
    # ================================================================
    print(f"\n{'='*70}")
    print("Updating comms bridge")
    update_comms(preferred_data, validation, scored_symbols, date_str)

    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("MICROSTRUCTURE DISCOVERY COMPLETE")
    print(f"{'='*70}")
    print(f"Date: {date_str}")
    print(f"Sessions: {sessions}")
    print(f"Symbols analyzed: {len(features_by_symbol)}")

    pref_syms = [s["symbol"] for s in preferred_data["preferred"] if s["classification"] == "PREFERRED"]
    avoid_syms = [s["symbol"] for s in preferred_data["avoid"] if s["classification"] == "AVOID"]

    print(f"Preferred: {', '.join(pref_syms) if pref_syms else 'None'}")
    print(f"Avoid: {', '.join(avoid_syms) if avoid_syms else 'None'}")

    p_em = validation["preferred"].get("exit_model", {})
    a_em = validation["avoid"].get("exit_model", {})
    if p_em.get("avg_return") is not None and a_em.get("avg_return") is not None:
        print(f"Validation: Preferred={p_em['avg_return']:+.4f}% vs Avoid={a_em['avg_return']:+.4f}%")
        if p_em["avg_return"] > a_em["avg_return"]:
            print("STATUS: VALIDATED - preferred set outperforms avoid set")
        else:
            print("STATUS: NOT VALIDATED - scoring needs tuning")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
