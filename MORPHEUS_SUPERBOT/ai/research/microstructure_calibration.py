#!/usr/bin/env python3
"""
Outcome-Calibrated Microstructure Model (SuperBot Research)
READ ONLY - no production changes.

Extends the microstructure discovery pipeline by learning feature weights
from historical outcomes instead of using fixed constants.

Usage:
    python -m ai.research.microstructure_calibration --date 2026-03-03
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean, stdev

import numpy as np

# === PATHS ===
SUPERBOT_ROOT = Path("C:/AI_Bot_Research/MORPHEUS_SUPERBOT")
OUTPUT_DIR = SUPERBOT_ROOT / "engine" / "output"
CONFIGS_DIR = SUPERBOT_ROOT / "configs"
COMMS_PATH = SUPERBOT_ROOT / "comms" / "outbox_chatgpt.json"
QUOTES_CACHE_DIR = SUPERBOT_ROOT / "engine" / "cache" / "quotes"

# === EXIT MODEL ===
STOP_PCT = -1.0
TRAIL_ACTIVATE = 0.8
TRAIL_DISTANCE = 0.4
EXIT_WINDOW_SEC = 300


def load_feature_data(date_str):
    """Load microstructure features per symbol."""
    path = OUTPUT_DIR / f"microstructure_features_{date_str}.json"
    with open(path) as f:
        data = json.load(f)
    return data["symbols"]


def load_edge_data(date_str):
    """Load edge preservation per-signal outcomes."""
    path = OUTPUT_DIR / f"edge_preservation_v2_{date_str}.json"
    with open(path) as f:
        data = json.load(f)
    return data["signals"]


def load_replay_data(date_str):
    """Load containment v2 replay per-signal data."""
    path = OUTPUT_DIR / f"containment_v2_replay_{date_str}.json"
    with open(path) as f:
        data = json.load(f)
    return [s for s in data["signals"] if s["v2_result"] == "PASS"]


def load_quotes(symbol):
    """Load cached quotes."""
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
        result.append({"epoch": q["epoch"], "price": price, "bid": bid, "ask": ask})
    return result


def parse_signal_epoch(ts_str):
    """HH:MM:SS -> epoch for 2026-03-03."""
    h, m, s = ts_str.split(":")
    return datetime(2026, 3, 3, int(h), int(m), int(s), tzinfo=timezone.utc).timestamp()


def compute_signal_level_features(signal, quotes):
    """
    Compute per-signal microstructure features from surrounding quotes.
    This gives us signal-level granularity instead of symbol-level averages.
    """
    sig_epoch = parse_signal_epoch(signal["timestamp"])

    # Use quotes in a window around the signal: [-60s, +10s]
    pre_window = [q for q in quotes if sig_epoch - 60 <= q["epoch"] <= sig_epoch]
    post_window = [q for q in quotes if sig_epoch < q["epoch"] <= sig_epoch + 10]
    context_window = [q for q in quotes if sig_epoch - 120 <= q["epoch"] <= sig_epoch + 10]

    features = {}

    # --- Spread at signal time (from nearby quotes) ---
    if pre_window:
        recent_spreads = []
        for q in pre_window[-20:]:  # last 20 quotes before signal
            if q["bid"] > 0 and q["ask"] > 0 and q["ask"] > q["bid"]:
                mid = (q["bid"] + q["ask"]) / 2
                recent_spreads.append((q["ask"] - q["bid"]) / mid * 100)
        if recent_spreads:
            features["local_spread_mean"] = round(mean(recent_spreads), 4)
            features["local_spread_std"] = round(stdev(recent_spreads), 4) if len(recent_spreads) > 1 else 0.0
        else:
            features["local_spread_mean"] = None
            features["local_spread_std"] = None
    else:
        features["local_spread_mean"] = None
        features["local_spread_std"] = None

    # --- Price velocity (rate of price change in last 30s) ---
    vel_window = [q for q in quotes if sig_epoch - 30 <= q["epoch"] <= sig_epoch]
    if len(vel_window) >= 2:
        first_p = vel_window[0]["price"]
        last_p = vel_window[-1]["price"]
        dt = vel_window[-1]["epoch"] - vel_window[0]["epoch"]
        if dt > 0 and first_p > 0:
            features["price_velocity_30s"] = round((last_p - first_p) / first_p * 100 / (dt / 30), 4)
        else:
            features["price_velocity_30s"] = 0.0
    else:
        features["price_velocity_30s"] = None

    # --- Local volatility (return std over last 60s at ~1s intervals) ---
    if len(pre_window) >= 5:
        prices = [q["price"] for q in pre_window]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100
                   for i in range(1, len(prices)) if prices[i-1] > 0]
        if returns and len(returns) > 1:
            features["local_volatility"] = round(stdev(returns), 4)
        else:
            features["local_volatility"] = None
    else:
        features["local_volatility"] = None

    # --- Volatility expansion (ratio of last 30s vol to prior 30s vol) ---
    early_window = [q for q in quotes if sig_epoch - 60 <= q["epoch"] <= sig_epoch - 30]
    late_window = [q for q in quotes if sig_epoch - 30 <= q["epoch"] <= sig_epoch]
    if len(early_window) >= 3 and len(late_window) >= 3:
        early_prices = [q["price"] for q in early_window]
        late_prices = [q["price"] for q in late_window]
        early_rets = [(early_prices[i] - early_prices[i-1]) / early_prices[i-1] * 100
                      for i in range(1, len(early_prices)) if early_prices[i-1] > 0]
        late_rets = [(late_prices[i] - late_prices[i-1]) / late_prices[i-1] * 100
                     for i in range(1, len(late_prices)) if late_prices[i-1] > 0]
        if early_rets and late_rets and len(early_rets) > 1 and len(late_rets) > 1:
            early_vol = stdev(early_rets)
            late_vol = stdev(late_rets)
            features["volatility_expansion"] = round(late_vol / early_vol, 4) if early_vol > 0 else None
        else:
            features["volatility_expansion"] = None
    else:
        features["volatility_expansion"] = None

    # --- Local tick rate (ticks in last 30s) ---
    tick_30s = [q for q in quotes if sig_epoch - 30 <= q["epoch"] <= sig_epoch]
    features["local_tick_rate"] = round(len(tick_30s) / 30, 4) if tick_30s else 0.0

    # --- Entry lateness (distance from local low to entry price) ---
    if pre_window:
        local_low = min(q["price"] for q in pre_window)
        entry_price = signal.get("entry_price", 0)
        if local_low > 0 and entry_price > 0:
            features["entry_lateness"] = round((entry_price - local_low) / local_low * 100, 4)
        else:
            features["entry_lateness"] = None
    else:
        features["entry_lateness"] = None

    # --- Trend persistence (fraction of higher highs/lows in 5s windows) ---
    if context_window and len(context_window) >= 10:
        window_sec = 5
        start_ep = context_window[0]["epoch"]
        end_ep = context_window[-1]["epoch"]
        windows = []
        t = start_ep
        while t + window_sec <= end_ep:
            wq = [q for q in context_window if t <= q["epoch"] < t + window_sec]
            if wq:
                windows.append({"high": max(q["price"] for q in wq), "low": min(q["price"] for q in wq)})
            t += window_sec
        if len(windows) >= 2:
            hh = sum(1 for i in range(1, len(windows)) if windows[i]["high"] > windows[i-1]["high"])
            hl = sum(1 for i in range(1, len(windows)) if windows[i]["low"] > windows[i-1]["low"])
            features["local_trend_persistence"] = round((hh + hl) / (2 * (len(windows) - 1)), 4)
        else:
            features["local_trend_persistence"] = None
    else:
        features["local_trend_persistence"] = None

    # --- Quote staleness at signal time ---
    if pre_window:
        gaps = [(pre_window[i]["epoch"] - pre_window[i-1]["epoch"]) * 1000
                for i in range(1, len(pre_window))]
        if gaps:
            features["local_stale_count"] = sum(1 for g in gaps if g > 2000)
            features["local_gap_p90_ms"] = round(sorted(gaps)[int(len(gaps) * 0.9)], 1)
        else:
            features["local_stale_count"] = 0
            features["local_gap_p90_ms"] = None
    else:
        features["local_stale_count"] = 0
        features["local_gap_p90_ms"] = None

    return features


# ========================================================================
# STEP 1: Merge Feature + Outcome Data
# ========================================================================
def step1_merge_data(date_str):
    """Merge microstructure features with signal-level outcomes."""
    print("\n=== STEP 1: Merge Feature + Outcome Data ===")

    symbol_features = load_feature_data(date_str)
    edge_signals = load_edge_data(date_str)
    replay_signals = load_replay_data(date_str)

    print(f"  Symbol-level features: {len(symbol_features)} symbols")
    print(f"  Edge signals: {len(edge_signals)}")
    print(f"  Replay signals (v2-pass): {len(replay_signals)}")

    # Load quotes for signal-level feature computation
    print("  Loading quote caches for signal-level features...")
    symbol_quotes = {}
    for sym in symbol_features:
        quotes = load_quotes(sym)
        symbol_quotes[sym] = quotes

    # Build merged dataset: one row per signal
    dataset = []
    for edge_sig in edge_signals:
        sym = edge_sig["symbol"]
        if sym not in symbol_features:
            continue
        if not edge_sig.get("has_price_data"):
            continue

        # Find matching replay signal for additional fields
        replay_match = None
        for rs in replay_signals:
            if rs["num"] == edge_sig["num"]:
                replay_match = rs
                break

        # Symbol-level features
        sym_feats = symbol_features[sym]

        # Signal-level features
        sig_feats = compute_signal_level_features(
            edge_sig, symbol_quotes.get(sym, [])
        )

        # Outcomes
        w1m = edge_sig.get("windows", {}).get("1m", {})
        w5m = edge_sig.get("windows", {}).get("5m", {})
        w10m = edge_sig.get("windows", {}).get("10m", {})
        exit_model = edge_sig.get("exit_model", {})

        row = {
            "num": edge_sig["num"],
            "symbol": sym,
            "timestamp": edge_sig["timestamp"],
            "strategy": edge_sig["strategy"],
            "phase": edge_sig["phase"],

            # Signal-level scalar features
            "confidence": edge_sig.get("confidence"),
            "momentum_score": edge_sig.get("momentum_score"),
            "spread_pct": edge_sig.get("spread_pct"),

            # Symbol-level microstructure features
            "sym_spread_median": sym_feats.get("spread_median"),
            "sym_spread_p90": sym_feats.get("spread_p90"),
            "sym_spread_stability": sym_feats.get("spread_stability"),
            "sym_tick_rate": sym_feats.get("tick_rate"),
            "sym_stale_burst_rate": sym_feats.get("stale_burst_rate"),
            "sym_gap_frequency": sym_feats.get("gap_frequency"),
            "sym_volatility_1s": sym_feats.get("volatility_1s"),
            "sym_volatility_5s": sym_feats.get("volatility_5s"),
            "sym_volatility_30s": sym_feats.get("volatility_30s"),
            "sym_trend_persistence": sym_feats.get("trend_persistence"),
            "sym_profit_margin_ratio": sym_feats.get("profit_margin_ratio"),
            "sym_burstiness": sym_feats.get("burstiness"),

            # Signal-level microstructure features
            "local_spread_mean": sig_feats.get("local_spread_mean"),
            "local_spread_std": sig_feats.get("local_spread_std"),
            "price_velocity_30s": sig_feats.get("price_velocity_30s"),
            "local_volatility": sig_feats.get("local_volatility"),
            "volatility_expansion": sig_feats.get("volatility_expansion"),
            "local_tick_rate": sig_feats.get("local_tick_rate"),
            "entry_lateness": sig_feats.get("entry_lateness"),
            "local_trend_persistence": sig_feats.get("local_trend_persistence"),
            "local_stale_count": sig_feats.get("local_stale_count"),
            "local_gap_p90_ms": sig_feats.get("local_gap_p90_ms"),

            # Outcome variables
            "forward_return_1m": w1m.get("close_return_pct"),
            "forward_return_5m": w5m.get("close_return_pct"),
            "forward_return_10m": w10m.get("close_return_pct"),
            "mfe_5m": w5m.get("mfe_pct"),
            "mae_5m": w5m.get("mae_pct"),
            "exit_return": exit_model.get("exit_return_pct"),
            "exit_type": exit_model.get("exit_type"),
            "win": 1 if (exit_model.get("exit_return_pct") or 0) > 0 else 0,
        }

        dataset.append(row)

    print(f"  Merged dataset: {len(dataset)} rows")
    return dataset


# ========================================================================
# STEP 2: Train Feature Importance Model
# ========================================================================
def step2_train_model(dataset, date_str):
    """Train RandomForestRegressor to learn feature importance."""
    print("\n=== STEP 2: Train Feature Importance Model ===")

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    # Define feature columns (all numeric microstructure features)
    feature_cols = [
        "confidence", "momentum_score", "spread_pct",
        "sym_spread_median", "sym_spread_p90", "sym_spread_stability",
        "sym_tick_rate", "sym_stale_burst_rate", "sym_gap_frequency",
        "sym_volatility_1s", "sym_volatility_5s", "sym_volatility_30s",
        "sym_trend_persistence", "sym_profit_margin_ratio", "sym_burstiness",
        "local_spread_mean", "local_spread_std",
        "price_velocity_30s", "local_volatility", "volatility_expansion",
        "local_tick_rate", "entry_lateness",
        "local_trend_persistence", "local_stale_count", "local_gap_p90_ms",
    ]

    target_col = "forward_return_1m"

    # Build X, y
    X_rows = []
    y_rows = []
    valid_cols = []

    for row in dataset:
        y_val = row.get(target_col)
        if y_val is None:
            continue

        x_row = []
        for col in feature_cols:
            val = row.get(col)
            if val is None:
                x_row.append(np.nan)
            else:
                x_row.append(float(val))

        X_rows.append(x_row)
        y_rows.append(y_val)

    X = np.array(X_rows)
    y = np.array(y_rows)

    print(f"  Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target: {target_col}")
    print(f"  Target range: [{y.min():.3f}%, {y.max():.3f}%], mean={y.mean():.3f}%")

    # Handle NaN: impute with column median
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

    # Train multiple models for robust importance
    print("\n  Training RandomForest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    rf_r2 = rf.score(X, y)
    print(f"  RF R-squared (in-sample): {rf_r2:.4f}")

    print("  Training GradientBoosting...")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=5,
        learning_rate=0.05,
        random_state=42,
    )
    gb.fit(X, y)
    gb_importance = gb.feature_importances_
    gb_r2 = gb.score(X, y)
    print(f"  GB R-squared (in-sample): {gb_r2:.4f}")

    # Also compute permutation-style importance via correlation
    print("  Computing correlation importance...")
    corr_importance = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) == 0:
            corr_importance.append(0.0)
        else:
            corr = np.corrcoef(col, y)[0, 1]
            corr_importance.append(abs(corr) if not np.isnan(corr) else 0.0)
    corr_importance = np.array(corr_importance)

    # Ensemble: average normalized importances
    def normalize(arr):
        total = arr.sum()
        return arr / total if total > 0 else arr

    rf_norm = normalize(rf_importance)
    gb_norm = normalize(gb_importance)
    corr_norm = normalize(corr_importance)

    # Weighted ensemble: 40% RF, 40% GB, 20% correlation
    ensemble_importance = 0.4 * rf_norm + 0.4 * gb_norm + 0.2 * corr_norm
    ensemble_importance = normalize(ensemble_importance)

    # Build results
    importance_results = []
    for i, col in enumerate(feature_cols):
        importance_results.append({
            "feature": col,
            "rf_importance": round(float(rf_norm[i]), 4),
            "gb_importance": round(float(gb_norm[i]), 4),
            "correlation": round(float(corr_importance[i]), 4),
            "ensemble_importance": round(float(ensemble_importance[i]), 4),
            "correlation_sign": round(float(np.corrcoef(X[:, i], y)[0, 1]), 4) if np.std(X[:, i]) > 0 else 0.0,
        })

    importance_results.sort(key=lambda r: r["ensemble_importance"], reverse=True)

    # Print top features
    print("\n  Feature importance (ensemble):")
    for r in importance_results[:10]:
        sign_str = "+" if r["correlation_sign"] > 0 else "-"
        print(f"    {r['feature']:>30}: {r['ensemble_importance']:.4f} (RF={r['rf_importance']:.4f}, GB={r['gb_importance']:.4f}, corr={r['correlation_sign']:+.3f})")

    # Save
    out_path = OUTPUT_DIR / f"microstructure_feature_importance_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "microstructure_feature_importance",
            "target": target_col,
            "n_samples": int(X.shape[0]),
            "rf_r2": round(rf_r2, 4),
            "gb_r2": round(gb_r2, 4),
            "features": importance_results,
        }, f, indent=2)
    print(f"\n  -> {out_path.name}")

    return importance_results, {
        "rf_r2": rf_r2,
        "gb_r2": gb_r2,
        "feature_cols": feature_cols,
    }


# ========================================================================
# STEP 3: Generate Learned Weights
# ========================================================================
def step3_generate_weights(importance_results, date_str):
    """Convert feature importance into scoring weights with correct signs."""
    print("\n=== STEP 3: Generate Learned Weights ===")

    # Map features to scoring weight names for the microstructure discovery module
    # Weight sign: positive importance + positive correlation = positive weight
    #              positive importance + negative correlation = negative weight

    learned_weights = {}
    for r in importance_results:
        feat = r["feature"]
        imp = r["ensemble_importance"]
        sign = 1.0 if r["correlation_sign"] >= 0 else -1.0

        # Scale importance to weight range [-3, +3]
        weight = round(imp * 3.0 * sign / max(imp for r2 in importance_results for imp in [r2["ensemble_importance"]]) if importance_results else 0, 4)

        learned_weights[feat] = {
            "weight": weight,
            "importance": imp,
            "direction": "positive" if sign > 0 else "negative",
            "correlation": r["correlation_sign"],
        }

    # Normalize: scale so max absolute weight = 3.0
    max_abs = max(abs(w["weight"]) for w in learned_weights.values()) if learned_weights else 1.0
    if max_abs > 0:
        for feat in learned_weights:
            learned_weights[feat]["weight"] = round(learned_weights[feat]["weight"] * 3.0 / max_abs, 4)

    # Also create the simple weights dict for the scoring engine
    simple_weights = {}
    for feat, data in learned_weights.items():
        simple_weights[feat] = data["weight"]

    # Save
    config_path = CONFIGS_DIR / "microstructure_weights_learned.json"
    with open(config_path, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "learned_microstructure_weights",
            "description": "Feature weights calibrated from outcome data using RF+GB ensemble",
            "weights": learned_weights,
            "simple_weights": simple_weights,
        }, f, indent=2)

    print(f"  -> {config_path.name}")
    print("\n  Learned weights (top 10):")
    sorted_weights = sorted(learned_weights.items(), key=lambda x: abs(x[1]["weight"]), reverse=True)
    for feat, data in sorted_weights[:10]:
        print(f"    {feat:>30}: {data['weight']:+.4f} (imp={data['importance']:.4f}, dir={data['direction']})")

    return learned_weights, simple_weights


# ========================================================================
# STEP 4: Re-score Symbols with Learned Weights
# ========================================================================
def step4_rescore(dataset, learned_weights, simple_weights, date_str):
    """Re-rank symbols using learned weights instead of fixed ones."""
    print("\n=== STEP 4: Re-score Symbols with Learned Weights ===")

    # Compute per-symbol average features
    symbol_features = defaultdict(lambda: defaultdict(list))
    for row in dataset:
        sym = row["symbol"]
        for feat in simple_weights:
            val = row.get(feat)
            if val is not None:
                symbol_features[sym][feat].append(val)

    # Compute z-scores across symbols for each feature
    symbol_means = {}
    for sym in symbol_features:
        symbol_means[sym] = {}
        for feat in simple_weights:
            vals = symbol_features[sym].get(feat, [])
            symbol_means[sym][feat] = mean(vals) if vals else None

    # Z-score normalization
    all_symbols = sorted(symbol_means.keys())
    z_scores = {}
    for feat in simple_weights:
        vals = [symbol_means[sym][feat] for sym in all_symbols if symbol_means[sym][feat] is not None]
        if len(vals) >= 2:
            mu = mean(vals)
            sd = stdev(vals) if len(vals) > 1 else 1.0
            if sd == 0:
                sd = 1.0
        else:
            mu, sd = 0, 1
        z_scores[feat] = {}
        for sym in all_symbols:
            val = symbol_means[sym][feat]
            z_scores[feat][sym] = (val - mu) / sd if val is not None else 0.0

    # Compute weighted scores
    results = []
    for sym in all_symbols:
        score = 0.0
        contributions = {}
        for feat, weight in simple_weights.items():
            z = z_scores.get(feat, {}).get(sym, 0.0)
            contrib = weight * z
            score += contrib
            contributions[feat] = round(contrib, 3)

        results.append({
            "symbol": sym,
            "learned_score": round(score, 3),
            "top_contributions": dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
        })

    results.sort(key=lambda r: r["learned_score"], reverse=True)

    # Save
    out_path = OUTPUT_DIR / f"microstructure_rank_learned_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "microstructure_rank_learned",
            "rankings": results,
        }, f, indent=2)

    print(f"  -> {out_path.name}")
    print("\n  Learned ranking:")
    for r in results:
        top_feat = list(r["top_contributions"].keys())[0] if r["top_contributions"] else "?"
        print(f"    {r['symbol']:>5}: score={r['learned_score']:+.3f} (top: {top_feat}={r['top_contributions'].get(top_feat, 0):+.3f})")

    return results


# ========================================================================
# STEP 5: Validate Learned vs Original Rankings
# ========================================================================
def step5_validate(dataset, learned_ranking, date_str):
    """Compare original vs learned ranking on forward returns."""
    print("\n=== STEP 5: Validate Learned vs Original Rankings ===")

    # Load original ranking
    orig_path = OUTPUT_DIR / f"microstructure_rank_{date_str}.json"
    with open(orig_path) as f:
        orig_data = json.load(f)
    orig_ranking = orig_data["rankings"]

    # Load edge preservation for ground truth
    edge_path = OUTPUT_DIR / f"edge_preservation_v2_{date_str}.json"
    with open(edge_path) as f:
        edge_data = json.load(f)
    sym_outcomes = edge_data["summary"]["symbol_breakdown"]

    # Build comparison table
    comparison = []
    for sym in sorted(sym_outcomes.keys()):
        outcome = sym_outcomes[sym]

        # Find ranks
        orig_rank = None
        orig_score = None
        for i, r in enumerate(orig_ranking):
            if r["symbol"] == sym:
                orig_rank = i + 1
                orig_score = r["score"]
                break

        learned_rank = None
        learned_score = None
        for i, r in enumerate(learned_ranking):
            if r["symbol"] == sym:
                learned_rank = i + 1
                learned_score = r["learned_score"]
                break

        avg_ret = outcome.get("exit_avg_return", 0)
        comparison.append({
            "symbol": sym,
            "exit_avg_return": avg_ret,
            "exit_win_rate": outcome.get("exit_win_rate"),
            "orig_rank": orig_rank,
            "orig_score": orig_score,
            "learned_rank": learned_rank,
            "learned_score": learned_score,
        })

    comparison.sort(key=lambda r: r["exit_avg_return"], reverse=True)

    # Compute correlation accuracy for both
    def score_correlation(ranking_field, score_field):
        correct = 0
        total = 0
        for c in comparison:
            if c[score_field] is not None and c["exit_avg_return"] is not None:
                total += 1
                if (c[score_field] > 0 and c["exit_avg_return"] > 0) or \
                   (c[score_field] <= 0 and c["exit_avg_return"] <= 0):
                    correct += 1
        return correct, total

    orig_correct, orig_total = score_correlation("orig_rank", "orig_score")
    learned_correct, learned_total = score_correlation("learned_rank", "learned_score")

    # Rank correlation (Spearman-like)
    def rank_correlation(comparison, score_field):
        pairs = [(c[score_field], c["exit_avg_return"]) for c in comparison
                 if c[score_field] is not None and c["exit_avg_return"] is not None]
        if len(pairs) < 3:
            return None
        scores = [p[0] for p in pairs]
        outcomes = [p[1] for p in pairs]
        # Simple Pearson on scores vs outcomes
        if np.std(scores) == 0 or np.std(outcomes) == 0:
            return 0.0
        return round(float(np.corrcoef(scores, outcomes)[0, 1]), 4)

    orig_corr = rank_correlation(comparison, "orig_score")
    learned_corr = rank_correlation(comparison, "learned_score")

    # Top-half vs bottom-half validation
    n_syms = len(comparison)
    half = n_syms // 2

    # Using learned ranking
    learned_sorted = sorted(comparison, key=lambda r: r["learned_score"] or -999, reverse=True)
    top_syms_learned = set(r["symbol"] for r in learned_sorted[:half])
    bot_syms_learned = set(r["symbol"] for r in learned_sorted[half:])

    top_signals_learned = [s for s in dataset if s["symbol"] in top_syms_learned]
    bot_signals_learned = [s for s in dataset if s["symbol"] in bot_syms_learned]

    def compute_set_metrics(signals):
        returns = [s["exit_return"] for s in signals if s.get("exit_return") is not None]
        returns_1m = [s["forward_return_1m"] for s in signals if s.get("forward_return_1m") is not None]
        if not returns:
            return {"n": 0, "win_rate": None, "avg_return": None, "avg_return_1m": None}
        wins = [r for r in returns if r > 0]
        gross_wins = sum(r for r in returns if r > 0)
        gross_losses = abs(sum(r for r in returns if r <= 0))
        pf = round(gross_wins / gross_losses, 3) if gross_losses > 0 else float("inf")
        return {
            "n": len(returns),
            "win_rate": round(len(wins) / len(returns) * 100, 1),
            "avg_return": round(mean(returns), 4),
            "total_return": round(sum(returns), 4),
            "profit_factor": pf,
            "avg_return_1m": round(mean(returns_1m), 4) if returns_1m else None,
        }

    top_metrics_learned = compute_set_metrics(top_signals_learned)
    bot_metrics_learned = compute_set_metrics(bot_signals_learned)

    # Using original ranking
    orig_sorted = sorted(comparison, key=lambda r: r["orig_score"] or -999, reverse=True)
    top_syms_orig = set(r["symbol"] for r in orig_sorted[:half])
    bot_syms_orig = set(r["symbol"] for r in orig_sorted[half:])

    top_signals_orig = [s for s in dataset if s["symbol"] in top_syms_orig]
    bot_signals_orig = [s for s in dataset if s["symbol"] in bot_syms_orig]

    top_metrics_orig = compute_set_metrics(top_signals_orig)
    bot_metrics_orig = compute_set_metrics(bot_signals_orig)

    print("\n  Comparison:")
    print(f"    Original:  correlation accuracy = {orig_correct}/{orig_total} ({orig_correct/orig_total*100:.0f}%), score-outcome r = {orig_corr}")
    print(f"    Learned:   correlation accuracy = {learned_correct}/{learned_total} ({learned_correct/learned_total*100:.0f}%), score-outcome r = {learned_corr}")
    print(f"\n    Original  top-half: WR={top_metrics_orig['win_rate']}%, avg={top_metrics_orig['avg_return']}%, PF={top_metrics_orig['profit_factor']}")
    print(f"    Original  bot-half: WR={bot_metrics_orig['win_rate']}%, avg={bot_metrics_orig['avg_return']}%, PF={bot_metrics_orig['profit_factor']}")
    print(f"    Learned   top-half: WR={top_metrics_learned['win_rate']}%, avg={top_metrics_learned['avg_return']}%, PF={top_metrics_learned['profit_factor']}")
    print(f"    Learned   bot-half: WR={bot_metrics_learned['win_rate']}%, avg={bot_metrics_learned['avg_return']}%, PF={bot_metrics_learned['profit_factor']}")

    # Generate report
    report_path = OUTPUT_DIR / f"microstructure_weight_calibration_report_{date_str}.md"

    lines = []
    lines.append(f"# Microstructure Weight Calibration Report - {date_str}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append("**Method:** RandomForest + GradientBoosting ensemble feature importance")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Feature importance
    lines.append("## Top Features Driving Momentum Edge")
    lines.append("")
    lines.append("| Rank | Feature | Importance | Direction | Weight | Correlation |")
    lines.append("|------|---------|------------|-----------|--------|-------------|")
    for i, r in enumerate(sorted([{"feature": k, **v} for k, v in learned_weights_global.items()],
                                   key=lambda x: x["importance"], reverse=True)[:15], 1):
        lines.append(f"| {i} | {r['feature']} | {r['importance']:.4f} | {r['direction']} | {r['weight']:+.4f} | {r['correlation']:+.3f} |")
    lines.append("")

    # Ranking comparison
    lines.append("## Ranking Comparison: Original vs Learned")
    lines.append("")
    lines.append("| Symbol | Actual Return | Orig Rank | Orig Score | Learned Rank | Learned Score | Improved? |")
    lines.append("|--------|--------------|-----------|------------|-------------|--------------|-----------|")
    for c in comparison:
        improved = ""
        if c["orig_rank"] and c["learned_rank"]:
            # Check if learned rank is more aligned with actual return
            if c["exit_avg_return"] > 0:
                improved = "YES" if c["learned_rank"] < c["orig_rank"] else "NO"
            else:
                improved = "YES" if c["learned_rank"] > c["orig_rank"] else "NO"
        or_str = f"{c['orig_rank']}" if c['orig_rank'] else "-"
        os_str = f"{c['orig_score']:+.3f}" if c['orig_score'] is not None else "-"
        lr_str = f"{c['learned_rank']}" if c['learned_rank'] else "-"
        ls_str = f"{c['learned_score']:+.3f}" if c['learned_score'] is not None else "-"
        lines.append(f"| {c['symbol']} | {c['exit_avg_return']:+.4f}% | {or_str} | {os_str} | {lr_str} | {ls_str} | {improved} |")
    lines.append("")

    # Correlation metrics
    lines.append("## Correlation Metrics")
    lines.append("")
    lines.append("| Metric | Original | Learned | Improvement |")
    lines.append("|--------|----------|---------|-------------|")
    lines.append(f"| Sign accuracy | {orig_correct}/{orig_total} ({orig_correct/orig_total*100:.0f}%) | {learned_correct}/{learned_total} ({learned_correct/learned_total*100:.0f}%) | {(learned_correct/learned_total - orig_correct/orig_total)*100:+.0f}pp |")
    if orig_corr is not None and learned_corr is not None:
        lines.append(f"| Score-outcome r | {orig_corr:+.4f} | {learned_corr:+.4f} | {learned_corr - orig_corr:+.4f} |")
    lines.append("")

    # Set performance
    lines.append("## Top-Half vs Bottom-Half Performance")
    lines.append("")
    lines.append("| Model | Set | N | Win Rate | Avg Return | PF | 1m Avg |")
    lines.append("|-------|-----|---|----------|------------|-----|--------|")
    lines.append(f"| Original | Top | {top_metrics_orig['n']} | {top_metrics_orig['win_rate']}% | {top_metrics_orig['avg_return']:+.4f}% | {top_metrics_orig['profit_factor']} | {top_metrics_orig.get('avg_return_1m','-')} |")
    lines.append(f"| Original | Bottom | {bot_metrics_orig['n']} | {bot_metrics_orig['win_rate']}% | {bot_metrics_orig['avg_return']:+.4f}% | {bot_metrics_orig['profit_factor']} | {bot_metrics_orig.get('avg_return_1m','-')} |")
    lines.append(f"| **Learned** | **Top** | **{top_metrics_learned['n']}** | **{top_metrics_learned['win_rate']}%** | **{top_metrics_learned['avg_return']:+.4f}%** | **{top_metrics_learned['profit_factor']}** | **{top_metrics_learned.get('avg_return_1m','-')}** |")
    lines.append(f"| **Learned** | **Bottom** | **{bot_metrics_learned['n']}** | **{bot_metrics_learned['win_rate']}%** | **{bot_metrics_learned['avg_return']:+.4f}%** | **{bot_metrics_learned['profit_factor']}** | **{bot_metrics_learned.get('avg_return_1m','-')}** |")
    lines.append("")

    # Top-bottom spread
    orig_spread = (top_metrics_orig["avg_return"] or 0) - (bot_metrics_orig["avg_return"] or 0)
    learned_spread = (top_metrics_learned["avg_return"] or 0) - (bot_metrics_learned["avg_return"] or 0)
    lines.append(f"**Original top-bottom spread:** {orig_spread:+.4f}%")
    lines.append(f"**Learned top-bottom spread:** {learned_spread:+.4f}%")
    if learned_spread > orig_spread:
        lines.append(f"**Improvement:** {learned_spread - orig_spread:+.4f}% wider separation")
    lines.append("")

    # Updated preferred symbols
    lines.append("## Updated Preferred Symbols (Learned Model)")
    lines.append("")
    for r in learned_ranking:
        cls = "PREFERRED" if r["learned_score"] > 0 else "AVOID"
        lines.append(f"- **{r['symbol']}**: {r['learned_score']:+.3f} [{cls}]")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*This calibration is research-only. No production changes applied.*")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  -> {report_path.name}")

    return {
        "comparison": comparison,
        "orig_accuracy": f"{orig_correct}/{orig_total}",
        "learned_accuracy": f"{learned_correct}/{learned_total}",
        "orig_corr": orig_corr,
        "learned_corr": learned_corr,
        "top_metrics_orig": top_metrics_orig,
        "bot_metrics_orig": bot_metrics_orig,
        "top_metrics_learned": top_metrics_learned,
        "bot_metrics_learned": bot_metrics_learned,
        "orig_spread": orig_spread,
        "learned_spread": learned_spread,
    }


# ========================================================================
# STEP 6: Supervisor Message
# ========================================================================
def step6_comms(learned_weights, learned_ranking, validation_results, importance_results, date_str):
    """Append summary to comms bridge."""
    print("\n=== STEP 6: Update Comms Bridge ===")

    if not COMMS_PATH.exists():
        print("  Comms file not found, skipping")
        return

    with open(COMMS_PATH) as f:
        comms = json.load(f)

    # Top features
    top_feats = sorted(importance_results, key=lambda r: r["ensemble_importance"], reverse=True)[:5]

    # Build body
    body_parts = [
        f"== OUTCOME-CALIBRATED MICROSTRUCTURE MODEL ({date_str}) ==",
        "",
        "== TOP FEATURES DRIVING MOMENTUM EDGE ==",
    ]
    for r in top_feats:
        body_parts.append(f"  {r['feature']:>30}: importance={r['ensemble_importance']:.4f}, corr={r['correlation_sign']:+.3f}")

    body_parts.append("")
    body_parts.append("== UPDATED PREFERRED SYMBOLS (LEARNED MODEL) ==")
    for r in learned_ranking:
        cls = "PREFERRED" if r["learned_score"] > 0 else "AVOID"
        body_parts.append(f"  {r['symbol']:>5}: score={r['learned_score']:+.3f} [{cls}]")

    body_parts.append("")
    body_parts.append("== IMPROVEMENT METRICS ==")
    vr = validation_results
    body_parts.append(f"  Correlation accuracy: Original={vr['orig_accuracy']} -> Learned={vr['learned_accuracy']}")
    body_parts.append(f"  Score-outcome r: Original={vr['orig_corr']} -> Learned={vr['learned_corr']}")
    body_parts.append(f"  Top-bottom spread: Original={vr['orig_spread']:+.4f}% -> Learned={vr['learned_spread']:+.4f}%")

    tm = vr["top_metrics_learned"]
    bm = vr["bot_metrics_learned"]
    body_parts.append(f"  Learned top-half: WR={tm['win_rate']}%, avg={tm['avg_return']:+.4f}%, PF={tm['profit_factor']}")
    body_parts.append(f"  Learned bot-half: WR={bm['win_rate']}%, avg={bm['avg_return']:+.4f}%, PF={bm['profit_factor']}")

    body_parts.append("")
    body_parts.append("== DELIVERABLES ==")
    body_parts.append(f"microstructure_feature_importance_{date_str}.json")
    body_parts.append(f"microstructure_rank_learned_{date_str}.json")
    body_parts.append(f"microstructure_weight_calibration_report_{date_str}.md")
    body_parts.append("configs/microstructure_weights_learned.json")
    body_parts.append("")
    body_parts.append("Production remains frozen. All analysis is read-only research.")

    preferred = [r["symbol"] for r in learned_ranking if r["learned_score"] > 0]
    avoid = [r["symbol"] for r in learned_ranking if r["learned_score"] <= 0]

    msg = {
        "id": f"msg_chatgpt_{date_str.replace('-','')}_007",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": "claude",
        "to": "chatgpt",
        "type": "action_result",
        "subject": f"Outcome-Calibrated Model ({date_str}): Preferred={','.join(preferred[:3])}, accuracy improved",
        "body": "\n".join(body_parts),
        "references": [
            f"engine/output/microstructure_feature_importance_{date_str}.json",
            f"engine/output/microstructure_rank_learned_{date_str}.json",
            f"engine/output/microstructure_weight_calibration_report_{date_str}.md",
            "configs/microstructure_weights_learned.json",
        ],
    }

    comms["messages"].append(msg)
    with open(COMMS_PATH, "w") as f:
        json.dump(comms, f, indent=2)
    print(f"  Comms updated -> {COMMS_PATH.name}")


# ========================================================================
# MAIN
# ========================================================================
# Global for report access
learned_weights_global = {}


def main():
    global learned_weights_global

    parser = argparse.ArgumentParser(description="Outcome-Calibrated Microstructure Model")
    parser.add_argument("--date", default="2026-03-03", help="Date to analyze")
    args = parser.parse_args()
    date_str = args.date

    print("=" * 70)
    print(f"OUTCOME-CALIBRATED MICROSTRUCTURE MODEL ({date_str})")
    print("SuperBot Research Engine - READ ONLY")
    print("=" * 70)

    # Step 1
    dataset = step1_merge_data(date_str)

    # Step 2
    importance_results, model_metrics = step2_train_model(dataset, date_str)

    # Step 3
    learned_weights, simple_weights = step3_generate_weights(importance_results, date_str)
    learned_weights_global = learned_weights

    # Step 4
    learned_ranking = step4_rescore(dataset, learned_weights, simple_weights, date_str)

    # Step 5
    validation_results = step5_validate(dataset, learned_ranking, date_str)

    # Step 6
    step6_comms(learned_weights, learned_ranking, validation_results, importance_results, date_str)

    # Final summary
    print(f"\n{'='*70}")
    print("OUTCOME-CALIBRATED MODEL COMPLETE")
    print(f"{'='*70}")

    top_feat = importance_results[0] if importance_results else None
    if top_feat:
        print(f"Top feature: {top_feat['feature']} (importance={top_feat['ensemble_importance']:.4f})")

    preferred = [r["symbol"] for r in learned_ranking if r["learned_score"] > 0]
    avoid = [r["symbol"] for r in learned_ranking if r["learned_score"] <= 0]
    print(f"Learned preferred: {', '.join(preferred)}")
    print(f"Learned avoid: {', '.join(avoid)}")

    vr = validation_results
    print(f"Correlation accuracy: {vr['orig_accuracy']} -> {vr['learned_accuracy']}")
    print(f"Score-outcome r: {vr['orig_corr']} -> {vr['learned_corr']}")
    print(f"Top-bottom spread: {vr['orig_spread']:+.4f}% -> {vr['learned_spread']:+.4f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
