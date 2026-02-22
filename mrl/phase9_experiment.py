"""
Phase 9 — Ignition Precursors + Quality Filters
=================================================

Tests HYP-023, HYP-024, HYP-025 on the same 18-day/1,944-trade dataset
from HYP-013. No new data collection needed.

Experiment 1 (HYP-023): Pressure slope filter
Experiment 2 (HYP-024): Volume acceleration threshold sweep
Experiment 3 (HYP-025): Combined gate

Evaluates: win rate, mean/median PnL, MFE/MAE, reward:risk
Stability: per-ticker + time-of-day buckets
Statistics: bootstrap 95% CIs on all key metrics

Usage:
    python mrl/phase9_experiment.py \
        --reports "\\\\Bob1\\c\\ai_project_hub\\store\\code\\IBKR_Algo_BOT_V2\\reports" \
        --databento "Z:\\AI_BOT_DATA\\databento_cache\\XNAS.ITCH\\trades"
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

# Import from replay engine
from replay.replay_engine import (
    ReplayConfig,
    IgnitionEvent,
    PressureProfile,
    load_ignition_events,
    load_raw_trades_around_event,
    compute_pressure_profile,
)

logger = logging.getLogger(__name__)

BOOTSTRAP_N = 2000
TOD_BUCKETS = {
    "pre_market": (4.0, 9.5),      # 4:00-9:30 ET
    "open":       (9.5, 10.5),      # 9:30-10:30 ET
    "mid":        (10.5, 14.0),     # 10:30-14:00 ET
    "close":      (14.0, 16.0),     # 14:00-16:00 ET
    "after_hours": (16.0, 21.0),    # 16:00-21:00 ET
}


# ============================================================
# STEP 1: BUILD PER-EVENT DATAFRAME
# ============================================================

def build_event_dataframe(
    config: ReplayConfig,
    symbols: List[str] = None,
    max_events: int = None,
) -> pd.DataFrame:
    """
    Load all ignition events, compute pressure profiles, and merge
    profile metrics with trade outcomes into a single DataFrame.
    """
    print("[PHASE 9] Step 1: Building per-event DataFrame...")

    events = load_ignition_events(config.reports_root, symbols)
    print(f"  Loaded {len(events)} ignition events")

    if max_events:
        events = events[:max_events]

    rows = []
    matched = 0
    skipped = 0

    for i, evt in enumerate(events):
        if i % 100 == 0:
            print(f"  Processing event {i+1}/{len(events)}: {evt.symbol} "
                  f"{evt.entry_time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"(matched={matched}, skipped={skipped})")

        trades_df = load_raw_trades_around_event(
            config.databento_root, evt.symbol, evt.entry_time,
            config.pre_window_sec, config.post_window_sec,
        )
        if trades_df is None:
            skipped += 1
            continue

        profile = compute_pressure_profile(trades_df, evt, config, is_control=False)
        if profile is None:
            skipped += 1
            continue

        matched += 1

        # Convert entry_time to ET for time-of-day classification
        entry_et = evt.entry_time.tz_convert("US/Eastern")
        hour_decimal = entry_et.hour + entry_et.minute / 60.0

        tod_bucket = "unknown"
        for bucket_name, (start_h, end_h) in TOD_BUCKETS.items():
            if start_h <= hour_decimal < end_h:
                tod_bucket = bucket_name
                break

        rows.append({
            # Trade identity
            "trade_id": evt.trade_id,
            "symbol": evt.symbol,
            "entry_time": evt.entry_time,
            "entry_time_et": entry_et,
            "hour_et": hour_decimal,
            "tod_bucket": tod_bucket,
            "date": entry_et.strftime("%Y-%m-%d"),

            # Trade outcomes
            "pnl": evt.pnl,
            "pnl_percent": evt.pnl_percent,
            "is_winner": 1 if evt.pnl > 0 else 0,
            "max_gain_pct": evt.max_gain_percent,       # MFE
            "max_drawdown_pct": abs(evt.max_drawdown_percent),  # MAE (positive)
            "hold_time_sec": evt.hold_time_seconds,
            "entry_price": evt.entry_price,
            "entry_signal": evt.entry_signal,
            "exit_category": evt.primary_exit_category,
            "volatility_regime": evt.volatility_regime,
            "rvol": evt.rvol,
            "change_pct": evt.change_pct,
            "spread_pct": evt.spread_pct,

            # Pressure profile metrics
            "peak_pressure_z": profile.peak_pressure_z_pre,
            "mean_pressure_z": profile.mean_pressure_z_pre,
            "pressure_direction": profile.pressure_direction_pre,
            "buildup_rate": profile.pressure_buildup_rate,
            "first_cross_sec": profile.first_threshold_cross_sec,
            "bars_above_threshold": profile.bars_above_threshold_pre,
            "pressure_consistency": profile.pressure_consistency,
            "volume_acceleration": profile.volume_acceleration,

            # Data quality
            "pre_bars": profile.pre_bars,
            "total_bars": profile.total_bars,
            "coverage_pct": profile.coverage_pct,
        })

    print(f"  Matched: {matched} / {len(events)} ({matched/len(events)*100:.1f}%)")
    print(f"  Skipped: {skipped}")

    df = pd.DataFrame(rows)

    # Compute reward:risk (MFE / MAE) — guard against zero MAE
    df["reward_risk"] = np.where(
        df["max_drawdown_pct"] > 0,
        df["max_gain_pct"] / df["max_drawdown_pct"],
        np.where(df["max_gain_pct"] > 0, 10.0, 0.0),  # Cap at 10 if no drawdown
    )

    return df


# ============================================================
# STEP 2: METRICS COMPUTATION
# ============================================================

def compute_group_metrics(df: pd.DataFrame, label: str = "") -> Dict:
    """Compute all evaluation metrics for a group of trades."""
    n = len(df)
    if n == 0:
        return {"n": 0, "label": label}

    winners = df[df["is_winner"] == 1]
    losers = df[df["is_winner"] == 0]

    return {
        "label": label,
        "n": n,
        "win_rate": float(len(winners) / n * 100),
        "mean_pnl": float(df["pnl"].mean()),
        "median_pnl": float(df["pnl"].median()),
        "total_pnl": float(df["pnl"].sum()),
        "mean_pnl_pct": float(df["pnl_percent"].mean()),
        "median_mfe": float(df["max_gain_pct"].median()),
        "mean_mfe": float(df["max_gain_pct"].mean()),
        "median_mae": float(df["max_drawdown_pct"].median()),
        "mean_mae": float(df["max_drawdown_pct"].mean()),
        "median_reward_risk": float(df["reward_risk"].median()),
        "mean_reward_risk": float(df["reward_risk"].clip(upper=20).mean()),
        "mean_hold_sec": float(df["hold_time_sec"].mean()),
        "unique_symbols": int(df["symbol"].nunique()),
        "unique_dates": int(df["date"].nunique()),
    }


def bootstrap_ci(values, stat_fn=np.mean, n_boot=BOOTSTRAP_N, ci=95):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return (float("nan"), float("nan"))

    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(stat_fn(sample))

    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return (float(lower), float(upper))


def permutation_test(group_a, group_b, stat_fn=np.mean, n_perm=2000):
    """Two-sample permutation test."""
    rng = np.random.default_rng(42)
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 5 or len(b) < 5:
        return float("nan")

    observed = stat_fn(a) - stat_fn(b)
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = stat_fn(combined[:na]) - stat_fn(combined[na:])
        if abs(perm_diff) >= abs(observed):
            count += 1

    return count / n_perm


# ============================================================
# STEP 3: EXPERIMENT RUNNERS
# ============================================================

def experiment_slope_filter(df: pd.DataFrame) -> Dict:
    """
    HYP-023: Does pressure_buildup_rate >= 0 filter improve trade quality?
    Sweep multiple slope thresholds.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — HYP-023: Pressure Slope Filter")
    print("=" * 70)

    thresholds = [
        ("slope >= -0.01 (loose)", -0.01),
        ("slope >= 0 (neutral)", 0.0),
        ("slope >= 0.005 (mild)", 0.005),
        ("slope >= 0.01 (moderate)", 0.01),
        ("slope >= 0.02 (strong)", 0.02),
    ]

    baseline = compute_group_metrics(df, "ALL TRADES (baseline)")
    results = {"baseline": baseline, "thresholds": []}

    print(f"\n  Baseline: n={baseline['n']}, win_rate={baseline['win_rate']:.1f}%, "
          f"mean_pnl=${baseline['mean_pnl']:.2f}, R:R={baseline['median_reward_risk']:.3f}")

    for label, thresh in thresholds:
        passed = df[df["buildup_rate"] >= thresh]
        rejected = df[df["buildup_rate"] < thresh]

        m_pass = compute_group_metrics(passed, f"PASS ({label})")
        m_reject = compute_group_metrics(rejected, f"REJECT ({label})")

        # Permutation test on win rate (as 0/1)
        if len(passed) >= 10 and len(rejected) >= 10:
            p_winrate = permutation_test(
                passed["is_winner"].values, rejected["is_winner"].values
            )
            p_pnl = permutation_test(
                passed["pnl"].values, rejected["pnl"].values
            )
            ci_wr = bootstrap_ci(passed["is_winner"].values * 100)
            ci_rr = bootstrap_ci(passed["reward_risk"].clip(upper=20).values, stat_fn=np.median)
        else:
            p_winrate = float("nan")
            p_pnl = float("nan")
            ci_wr = (float("nan"), float("nan"))
            ci_rr = (float("nan"), float("nan"))

        entry = {
            "threshold": thresh,
            "label": label,
            "pass": m_pass,
            "reject": m_reject,
            "p_winrate": p_winrate,
            "p_pnl": p_pnl,
            "ci_winrate_95": ci_wr,
            "ci_reward_risk_95": ci_rr,
        }
        results["thresholds"].append(entry)

        sig = "***" if p_winrate < 0.01 else ("*" if p_winrate < 0.05 else "ns")
        print(f"\n  {label}:")
        print(f"    PASS:   n={m_pass['n']:>4}, WR={m_pass['win_rate']:>5.1f}%, "
              f"PnL=${m_pass['mean_pnl']:>+7.2f}, R:R={m_pass['median_reward_risk']:.3f}")
        print(f"    REJECT: n={m_reject['n']:>4}, WR={m_reject['win_rate']:>5.1f}%, "
              f"PnL=${m_reject['mean_pnl']:>+7.2f}, R:R={m_reject['median_reward_risk']:.3f}")
        print(f"    p(WR): {p_winrate:.4f} {sig}   "
              f"CI95(WR): [{ci_wr[0]:.1f}%, {ci_wr[1]:.1f}%]")

    return results


def experiment_volume_accel(df: pd.DataFrame) -> Dict:
    """
    HYP-024: Volume acceleration threshold sweep.
    Use quantiles of volume_acceleration to find optimal threshold.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — HYP-024: Volume Acceleration Threshold")
    print("=" * 70)

    # Compute quantiles
    quantiles = [0.50, 0.60, 0.70, 0.80, 0.90]
    quantile_values = df["volume_acceleration"].quantile(quantiles)

    baseline = compute_group_metrics(df, "ALL TRADES (baseline)")
    results = {"baseline": baseline, "quantiles": []}

    print(f"\n  Volume acceleration distribution:")
    print(f"    mean={df['volume_acceleration'].mean():.2f}  "
          f"median={df['volume_acceleration'].median():.2f}  "
          f"std={df['volume_acceleration'].std():.2f}")
    for q, v in zip(quantiles, quantile_values):
        print(f"    P{int(q*100)}={v:.2f}")

    for q in quantiles:
        thresh = float(df["volume_acceleration"].quantile(q))
        label = f"vol_accel >= P{int(q*100)} ({thresh:.1f})"

        passed = df[df["volume_acceleration"] >= thresh]
        rejected = df[df["volume_acceleration"] < thresh]

        m_pass = compute_group_metrics(passed, f"PASS ({label})")
        m_reject = compute_group_metrics(rejected, f"REJECT ({label})")

        if len(passed) >= 10 and len(rejected) >= 10:
            p_winrate = permutation_test(
                passed["is_winner"].values, rejected["is_winner"].values
            )
            p_pnl = permutation_test(
                passed["pnl"].values, rejected["pnl"].values
            )
            ci_wr = bootstrap_ci(passed["is_winner"].values * 100)
            ci_rr = bootstrap_ci(passed["reward_risk"].clip(upper=20).values, stat_fn=np.median)
        else:
            p_winrate = float("nan")
            p_pnl = float("nan")
            ci_wr = (float("nan"), float("nan"))
            ci_rr = (float("nan"), float("nan"))

        entry = {
            "quantile": q,
            "threshold": thresh,
            "label": label,
            "pass": m_pass,
            "reject": m_reject,
            "p_winrate": p_winrate,
            "p_pnl": p_pnl,
            "ci_winrate_95": ci_wr,
            "ci_reward_risk_95": ci_rr,
        }
        results["quantiles"].append(entry)

        sig = "***" if p_winrate < 0.01 else ("*" if p_winrate < 0.05 else "ns")
        print(f"\n  {label}:")
        print(f"    PASS:   n={m_pass['n']:>4}, WR={m_pass['win_rate']:>5.1f}%, "
              f"PnL=${m_pass['mean_pnl']:>+7.2f}, R:R={m_pass['median_reward_risk']:.3f}")
        print(f"    REJECT: n={m_reject['n']:>4}, WR={m_reject['win_rate']:>5.1f}%, "
              f"PnL=${m_reject['mean_pnl']:>+7.2f}, R:R={m_reject['median_reward_risk']:.3f}")
        print(f"    p(WR): {p_winrate:.4f} {sig}   "
              f"CI95(WR): [{ci_wr[0]:.1f}%, {ci_wr[1]:.1f}%]")

    return results


def experiment_combined_gate(df: pd.DataFrame, slope_thresh: float, vol_quantile: float) -> Dict:
    """
    HYP-025: Combined gate — (volume_accel > X) AND (pressure_slope >= Y)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — HYP-025: Combined Gate")
    print("=" * 70)

    vol_thresh = float(df["volume_acceleration"].quantile(vol_quantile))
    print(f"\n  Parameters: slope >= {slope_thresh}, vol_accel >= P{int(vol_quantile*100)} ({vol_thresh:.1f})")

    # Four groups
    both_pass = df[(df["buildup_rate"] >= slope_thresh) & (df["volume_acceleration"] >= vol_thresh)]
    slope_only = df[(df["buildup_rate"] >= slope_thresh) & (df["volume_acceleration"] < vol_thresh)]
    vol_only = df[(df["buildup_rate"] < slope_thresh) & (df["volume_acceleration"] >= vol_thresh)]
    neither = df[(df["buildup_rate"] < slope_thresh) & (df["volume_acceleration"] < vol_thresh)]

    baseline = compute_group_metrics(df, "ALL")
    m_both = compute_group_metrics(both_pass, "BOTH PASS")
    m_slope = compute_group_metrics(slope_only, "SLOPE ONLY")
    m_vol = compute_group_metrics(vol_only, "VOL ONLY")
    m_neither = compute_group_metrics(neither, "NEITHER")

    # Test combined vs neither
    if len(both_pass) >= 10 and len(neither) >= 10:
        p_wr = permutation_test(both_pass["is_winner"].values, neither["is_winner"].values)
        p_pnl = permutation_test(both_pass["pnl"].values, neither["pnl"].values)
        ci_wr = bootstrap_ci(both_pass["is_winner"].values * 100)
        ci_rr = bootstrap_ci(both_pass["reward_risk"].clip(upper=20).values, stat_fn=np.median)
    else:
        p_wr = float("nan")
        p_pnl = float("nan")
        ci_wr = (float("nan"), float("nan"))
        ci_rr = (float("nan"), float("nan"))

    # Test combined vs baseline
    if len(both_pass) >= 10:
        p_vs_base_wr = permutation_test(both_pass["is_winner"].values, df["is_winner"].values)
        p_vs_base_pnl = permutation_test(both_pass["pnl"].values, df["pnl"].values)
    else:
        p_vs_base_wr = float("nan")
        p_vs_base_pnl = float("nan")

    results = {
        "slope_threshold": slope_thresh,
        "vol_quantile": vol_quantile,
        "vol_threshold": vol_thresh,
        "baseline": baseline,
        "both_pass": m_both,
        "slope_only": m_slope,
        "vol_only": m_vol,
        "neither": m_neither,
        "p_wr_both_vs_neither": p_wr,
        "p_pnl_both_vs_neither": p_pnl,
        "p_wr_both_vs_baseline": p_vs_base_wr,
        "p_pnl_both_vs_baseline": p_vs_base_pnl,
        "ci_winrate_95": ci_wr,
        "ci_reward_risk_95": ci_rr,
    }

    for label, m in [("ALL (baseline)", baseline), ("BOTH PASS", m_both),
                     ("SLOPE ONLY", m_slope), ("VOL ONLY", m_vol), ("NEITHER", m_neither)]:
        print(f"\n  {label}:")
        print(f"    n={m['n']:>4}, WR={m['win_rate']:>5.1f}%, "
              f"PnL=${m['mean_pnl']:>+7.2f}, R:R={m['median_reward_risk']:.3f}, "
              f"MFE={m['median_mfe']:.2f}%, MAE={m['median_mae']:.2f}%")

    sig = "***" if p_wr < 0.01 else ("*" if p_wr < 0.05 else "ns")
    print(f"\n  BOTH vs NEITHER: p(WR)={p_wr:.4f} {sig}, p(PnL)={p_pnl:.4f}")
    print(f"  BOTH vs BASELINE: p(WR)={p_vs_base_wr:.4f}, p(PnL)={p_vs_base_pnl:.4f}")
    print(f"  CI95(WR): [{ci_wr[0]:.1f}%, {ci_wr[1]:.1f}%]")
    print(f"  CI95(R:R): [{ci_rr[0]:.3f}, {ci_rr[1]:.3f}]")

    return results


# ============================================================
# STEP 4: STABILITY CHECKS
# ============================================================

def stability_by_ticker(df: pd.DataFrame, slope_thresh: float, vol_quantile: float) -> Dict:
    """Check if filter effect holds across tickers."""
    print("\n" + "=" * 70)
    print("STABILITY CHECK — Per-Ticker Breakdown")
    print("=" * 70)

    vol_thresh = float(df["volume_acceleration"].quantile(vol_quantile))

    results = {}
    symbols = df["symbol"].value_counts()
    symbols = symbols[symbols >= 10].index.tolist()  # Only tickers with n >= 10

    print(f"\n  Tickers with n >= 10: {len(symbols)}")
    print(f"\n  {'SYMBOL':<8} {'N':>4} {'N_PASS':>6} {'WR_ALL':>7} {'WR_PASS':>8} {'DELTA':>6}")
    print(f"  {'-'*45}")

    for sym in sorted(symbols):
        sym_df = df[df["symbol"] == sym]
        sym_pass = sym_df[
            (sym_df["buildup_rate"] >= slope_thresh) &
            (sym_df["volume_acceleration"] >= vol_thresh)
        ]

        n_all = len(sym_df)
        n_pass = len(sym_pass)
        wr_all = sym_df["is_winner"].mean() * 100
        wr_pass = sym_pass["is_winner"].mean() * 100 if n_pass > 0 else float("nan")
        delta = wr_pass - wr_all if not np.isnan(wr_pass) else float("nan")

        results[sym] = {
            "n_all": n_all, "n_pass": n_pass,
            "wr_all": wr_all, "wr_pass": wr_pass, "delta": delta,
        }

        delta_str = f"{delta:>+5.1f}" if not np.isnan(delta) else "  N/A"
        wr_pass_str = f"{wr_pass:>7.1f}%" if not np.isnan(wr_pass) else "    N/A"
        print(f"  {sym:<8} {n_all:>4} {n_pass:>6} {wr_all:>6.1f}% {wr_pass_str} {delta_str}")

    # Count how many tickers show improvement
    improved = sum(1 for v in results.values() if not np.isnan(v["delta"]) and v["delta"] > 0)
    total_with_data = sum(1 for v in results.values() if not np.isnan(v["delta"]))
    print(f"\n  Tickers with improvement: {improved} / {total_with_data}")

    return results


def stability_by_tod(df: pd.DataFrame, slope_thresh: float, vol_quantile: float) -> Dict:
    """Check if filter effect holds across time-of-day buckets."""
    print("\n" + "=" * 70)
    print("STABILITY CHECK — Time-of-Day Buckets")
    print("=" * 70)

    vol_thresh = float(df["volume_acceleration"].quantile(vol_quantile))

    results = {}
    print(f"\n  {'BUCKET':<14} {'N':>5} {'N_PASS':>6} {'WR_ALL':>7} {'WR_PASS':>8} "
          f"{'DELTA':>6} {'PNL_ALL':>9} {'PNL_PASS':>9}")
    print(f"  {'-'*75}")

    for bucket_name in ["pre_market", "open", "mid", "close", "after_hours"]:
        bdf = df[df["tod_bucket"] == bucket_name]
        bdf_pass = bdf[
            (bdf["buildup_rate"] >= slope_thresh) &
            (bdf["volume_acceleration"] >= vol_thresh)
        ]

        n_all = len(bdf)
        n_pass = len(bdf_pass)

        if n_all == 0:
            continue

        wr_all = bdf["is_winner"].mean() * 100
        wr_pass = bdf_pass["is_winner"].mean() * 100 if n_pass > 0 else float("nan")
        delta = wr_pass - wr_all if not np.isnan(wr_pass) else float("nan")
        pnl_all = bdf["pnl"].mean()
        pnl_pass = bdf_pass["pnl"].mean() if n_pass > 0 else float("nan")

        results[bucket_name] = {
            "n_all": n_all, "n_pass": n_pass,
            "wr_all": wr_all, "wr_pass": wr_pass, "delta": delta,
            "pnl_all": pnl_all, "pnl_pass": pnl_pass,
        }

        delta_str = f"{delta:>+5.1f}" if not np.isnan(delta) else "  N/A"
        wr_pass_str = f"{wr_pass:>7.1f}%" if not np.isnan(wr_pass) else "    N/A"
        pnl_pass_str = f"${pnl_pass:>+8.2f}" if not np.isnan(pnl_pass) else "     N/A"
        print(f"  {bucket_name:<14} {n_all:>5} {n_pass:>6} {wr_all:>6.1f}% {wr_pass_str} "
              f"{delta_str} ${pnl_all:>+8.2f} {pnl_pass_str}")

    improved = sum(1 for v in results.values() if not np.isnan(v.get("delta", float("nan"))) and v["delta"] > 0)
    total_with_data = sum(1 for v in results.values() if not np.isnan(v.get("delta", float("nan"))))
    print(f"\n  Buckets with improvement: {improved} / {total_with_data}")

    return results


# ============================================================
# STEP 5: GENERATE REPORT
# ============================================================

def generate_report(
    df: pd.DataFrame,
    slope_results: Dict,
    vol_results: Dict,
    combined_results: Dict,
    ticker_stability: Dict,
    tod_stability: Dict,
    output_path: Path,
):
    """Generate Phase 9 results markdown report."""
    print(f"\n  Writing report to {output_path}")

    b = slope_results["baseline"]
    n_total = b["n"]

    # Find best individual filters
    best_slope = None
    best_slope_delta = -999
    for t in slope_results["thresholds"]:
        if t["pass"]["n"] >= 50:
            delta = t["pass"]["win_rate"] - b["win_rate"]
            if delta > best_slope_delta:
                best_slope_delta = delta
                best_slope = t

    best_vol = None
    best_vol_delta = -999
    for q in vol_results["quantiles"]:
        if q["pass"]["n"] >= 50:
            delta = q["pass"]["win_rate"] - b["win_rate"]
            if delta > best_vol_delta:
                best_vol_delta = delta
                best_vol = q

    # Count stability
    ticker_improved = sum(1 for v in ticker_stability.values()
                         if not np.isnan(v.get("delta", float("nan"))) and v["delta"] > 0)
    ticker_total = sum(1 for v in ticker_stability.values()
                      if not np.isnan(v.get("delta", float("nan"))))
    tod_improved = sum(1 for v in tod_stability.values()
                      if not np.isnan(v.get("delta", float("nan"))) and v["delta"] > 0)
    tod_total = sum(1 for v in tod_stability.values()
                   if not np.isnan(v.get("delta", float("nan"))))

    cb = combined_results["both_pass"]

    lines = []
    lines.append("# Phase 9 — Ignition Precursors + Quality Filters: Results")
    lines.append("")
    lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Dataset:** {n_total} trades with valid pressure profiles / 1,944 total Morpheus trades")
    lines.append(f"**Period:** 18 trading days (2026-01-29 to 2026-02-20)")
    lines.append(f"**Symbols:** {b['unique_symbols']} unique tickers")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    # Determine go/no-go
    combined_p = combined_results.get("p_wr_both_vs_neither", float("nan"))
    combined_significant = not np.isnan(combined_p) and combined_p < 0.05

    if combined_significant and cb["n"] >= 50 and tod_improved >= 3:
        verdict = "**GO** for paper validation"
        reason = (f"Combined gate shows significant improvement "
                  f"(p={combined_p:.4f}), n={cb['n']}, "
                  f"stable across {tod_improved}/{tod_total} time-of-day buckets.")
    elif cb["n"] >= 50 and cb["win_rate"] > b["win_rate"]:
        verdict = "**CONDITIONAL GO** — promising but needs more data"
        reason = (f"Combined gate improves win rate "
                  f"({cb['win_rate']:.1f}% vs {b['win_rate']:.1f}% baseline) "
                  f"but p={combined_p:.4f}. "
                  f"Stable in {tod_improved}/{tod_total} ToD buckets.")
    else:
        verdict = "**NO-GO** — filter does not reliably improve outcomes"
        reason = f"Combined gate n={cb['n']}, insufficient evidence."

    lines.append(f"**Verdict:** {verdict}")
    lines.append(f"**Basis:** {reason}")
    lines.append("")

    if best_slope and best_slope["pass"]["n"] >= 50:
        lines.append(f"**Best slope filter:** {best_slope['label']}")
        lines.append(f"  — n={best_slope['pass']['n']}, "
                     f"WR={best_slope['pass']['win_rate']:.1f}% "
                     f"(baseline {b['win_rate']:.1f}%), "
                     f"p(WR)={best_slope['p_winrate']:.4f}, "
                     f"CI95=[{best_slope['ci_winrate_95'][0]:.1f}%, {best_slope['ci_winrate_95'][1]:.1f}%]")
        lines.append("")

    if best_vol and best_vol["pass"]["n"] >= 50:
        lines.append(f"**Best volume filter:** {best_vol['label']}")
        lines.append(f"  — n={best_vol['pass']['n']}, "
                     f"WR={best_vol['pass']['win_rate']:.1f}% "
                     f"(baseline {b['win_rate']:.1f}%), "
                     f"p(WR)={best_vol['p_winrate']:.4f}, "
                     f"CI95=[{best_vol['ci_winrate_95'][0]:.1f}%, {best_vol['ci_winrate_95'][1]:.1f}%]")
        lines.append("")

    lines.append(f"**Combined gate:** slope >= {combined_results['slope_threshold']}, "
                 f"vol_accel >= P{int(combined_results['vol_quantile']*100)} "
                 f"({combined_results['vol_threshold']:.1f})")
    lines.append(f"  — n={cb['n']}, WR={cb['win_rate']:.1f}% (baseline {b['win_rate']:.1f}%), "
                 f"R:R={cb['median_reward_risk']:.3f} (baseline {b['median_reward_risk']:.3f})")
    lines.append(f"  — CI95(WR): [{combined_results['ci_winrate_95'][0]:.1f}%, "
                 f"{combined_results['ci_winrate_95'][1]:.1f}%]")
    lines.append(f"  — CI95(R:R): [{combined_results['ci_reward_risk_95'][0]:.3f}, "
                 f"{combined_results['ci_reward_risk_95'][1]:.3f}]")
    lines.append(f"  — Ticker stability: {ticker_improved}/{ticker_total} improved")
    lines.append(f"  — ToD stability: {tod_improved}/{tod_total} buckets improved")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Baseline
    lines.append("## Baseline (All Trades With Profiles)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total trades | {b['n']} |")
    lines.append(f"| Win rate | {b['win_rate']:.1f}% |")
    lines.append(f"| Mean PnL | ${b['mean_pnl']:.2f} |")
    lines.append(f"| Total PnL | ${b['total_pnl']:.2f} |")
    lines.append(f"| Median MFE | {b['median_mfe']:.2f}% |")
    lines.append(f"| Median MAE | {b['median_mae']:.2f}% |")
    lines.append(f"| Median R:R | {b['median_reward_risk']:.3f} |")
    lines.append(f"| Unique symbols | {b['unique_symbols']} |")
    lines.append(f"| Unique dates | {b['unique_dates']} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Experiment 1
    lines.append("## Experiment 1 — HYP-023: Pressure Slope Filter")
    lines.append("")
    lines.append("| Threshold | N Pass | WR Pass | WR Reject | Δ WR | p(WR) | Mean PnL Pass | R:R Pass |")
    lines.append("|-----------|--------|---------|-----------|------|-------|---------------|----------|")
    for t in slope_results["thresholds"]:
        p = t["pass"]
        r = t["reject"]
        delta = p["win_rate"] - r["win_rate"]
        sig = " *" if t["p_winrate"] < 0.05 else ""
        lines.append(f"| {t['label']} | {p['n']} | {p['win_rate']:.1f}% | {r['win_rate']:.1f}% | "
                     f"{delta:+.1f}pp | {t['p_winrate']:.4f}{sig} | "
                     f"${p['mean_pnl']:+.2f} | {p['median_reward_risk']:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Experiment 2
    lines.append("## Experiment 2 — HYP-024: Volume Acceleration Threshold")
    lines.append("")
    lines.append("| Quantile | Threshold | N Pass | WR Pass | WR Reject | Δ WR | p(WR) | Mean PnL Pass | R:R Pass |")
    lines.append("|----------|-----------|--------|---------|-----------|------|-------|---------------|----------|")
    for q in vol_results["quantiles"]:
        p = q["pass"]
        r = q["reject"]
        delta = p["win_rate"] - r["win_rate"]
        sig = " *" if q["p_winrate"] < 0.05 else ""
        lines.append(f"| P{int(q['quantile']*100)} | {q['threshold']:.1f} | {p['n']} | {p['win_rate']:.1f}% | "
                     f"{r['win_rate']:.1f}% | {delta:+.1f}pp | {q['p_winrate']:.4f}{sig} | "
                     f"${p['mean_pnl']:+.2f} | {p['median_reward_risk']:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Experiment 3
    lines.append("## Experiment 3 — HYP-025: Combined Gate")
    lines.append("")
    lines.append(f"Gate: `buildup_rate >= {combined_results['slope_threshold']}` "
                 f"AND `volume_acceleration >= {combined_results['vol_threshold']:.1f}` "
                 f"(P{int(combined_results['vol_quantile']*100)})")
    lines.append("")
    lines.append("| Group | N | Win Rate | Mean PnL | Median MFE | Median MAE | Median R:R |")
    lines.append("|-------|---|----------|----------|------------|------------|------------|")
    for label, key in [("ALL (baseline)", "baseline"), ("BOTH PASS", "both_pass"),
                       ("SLOPE ONLY", "slope_only"), ("VOL ONLY", "vol_only"),
                       ("NEITHER", "neither")]:
        m = combined_results[key]
        lines.append(f"| {label} | {m['n']} | {m['win_rate']:.1f}% | ${m['mean_pnl']:+.2f} | "
                     f"{m['median_mfe']:.2f}% | {m['median_mae']:.2f}% | {m['median_reward_risk']:.3f} |")
    lines.append("")
    sig = " ***" if combined_p < 0.01 else (" *" if combined_p < 0.05 else " (ns)")
    lines.append(f"**BOTH vs NEITHER:** p(WR)={combined_p:.4f}{sig}, "
                 f"p(PnL)={combined_results['p_pnl_both_vs_neither']:.4f}")
    lines.append(f"**BOTH vs BASELINE:** p(WR)={combined_results['p_wr_both_vs_baseline']:.4f}, "
                 f"p(PnL)={combined_results['p_pnl_both_vs_baseline']:.4f}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Stability — ToD
    lines.append("## Stability: Time-of-Day")
    lines.append("")
    lines.append("| Bucket | N All | N Pass | WR All | WR Pass | Δ WR | PnL All | PnL Pass |")
    lines.append("|--------|-------|--------|--------|---------|------|---------|----------|")
    for bucket_name in ["pre_market", "open", "mid", "close", "after_hours"]:
        v = tod_stability.get(bucket_name)
        if v is None:
            continue
        wr_pass_str = f"{v['wr_pass']:.1f}%" if not np.isnan(v.get("wr_pass", float("nan"))) else "N/A"
        delta_str = f"{v['delta']:+.1f}pp" if not np.isnan(v.get("delta", float("nan"))) else "N/A"
        pnl_pass_str = f"${v['pnl_pass']:+.2f}" if not np.isnan(v.get("pnl_pass", float("nan"))) else "N/A"
        lines.append(f"| {bucket_name} | {v['n_all']} | {v['n_pass']} | "
                     f"{v['wr_all']:.1f}% | {wr_pass_str} | {delta_str} | "
                     f"${v['pnl_all']:+.2f} | {pnl_pass_str} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Stability — Tickers
    lines.append("## Stability: Per-Ticker (n >= 10)")
    lines.append("")
    lines.append("| Symbol | N All | N Pass | WR All | WR Pass | Δ WR |")
    lines.append("|--------|-------|--------|--------|---------|------|")
    for sym in sorted(ticker_stability.keys()):
        v = ticker_stability[sym]
        wr_pass_str = f"{v['wr_pass']:.1f}%" if not np.isnan(v.get("wr_pass", float("nan"))) else "N/A"
        delta_str = f"{v['delta']:+.1f}pp" if not np.isnan(v.get("delta", float("nan"))) else "N/A"
        lines.append(f"| {sym} | {v['n_all']} | {v['n_pass']} | "
                     f"{v['wr_all']:.1f}% | {wr_pass_str} | {delta_str} |")
    lines.append("")

    report_text = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ============================================================
# MAIN
# ============================================================

def run_phase9(
    config: ReplayConfig = None,
    symbols: List[str] = None,
    max_events: int = None,
) -> Dict:
    """Full Phase 9 pipeline."""
    if config is None:
        config = ReplayConfig()

    print("=" * 70)
    print("PHASE 9 — Ignition Precursors + Quality Filters")
    print("=" * 70)

    # Step 1: Build DataFrame
    df = build_event_dataframe(config, symbols, max_events)

    if len(df) < 50:
        print(f"\n[ERROR] Only {len(df)} events with profiles. Need >= 50.")
        return {"error": f"Insufficient events: {len(df)}"}

    # Save raw data for inspection
    raw_path = config.output_root / "phase9_event_data.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n  Raw event data saved: {raw_path}")

    # Step 2: Experiment 1 — Slope filter
    slope_results = experiment_slope_filter(df)

    # Step 3: Experiment 2 — Volume acceleration
    vol_results = experiment_volume_accel(df)

    # Step 4: Determine best thresholds for combined test
    # Use slope >= 0 (the natural threshold from HYP-013 findings)
    # Use vol P60 as starting point (conservative, keeps more trades)
    # But also try what looked best in individual sweeps
    best_slope_thresh = 0.0
    best_vol_q = 0.60

    # If a higher vol quantile had n >= 50 and better WR, use that
    for q in vol_results["quantiles"]:
        if q["pass"]["n"] >= 50 and q["pass"]["win_rate"] > vol_results["quantiles"][0]["pass"]["win_rate"]:
            best_vol_q = q["quantile"]

    # Step 5: Experiment 3 — Combined gate
    combined_results = experiment_combined_gate(df, best_slope_thresh, best_vol_q)

    # Step 6: Stability checks
    ticker_stability = stability_by_ticker(df, best_slope_thresh, best_vol_q)
    tod_stability = stability_by_tod(df, best_slope_thresh, best_vol_q)

    # Step 7: Generate report
    report_path = config.output_root.parent / "docs" / "Research" / "PHASE9_IgnitionPrecursors_Results.md"
    report_text = generate_report(
        df, slope_results, vol_results, combined_results,
        ticker_stability, tod_stability, report_path,
    )

    # Also save to results dir
    results_json_path = config.output_root / "phase9_results.json"
    all_results = {
        "slope_filter": slope_results,
        "volume_accel": vol_results,
        "combined_gate": combined_results,
        "ticker_stability": ticker_stability,
        "tod_stability": tod_stability,
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON results saved: {results_json_path}")

    print("\n" + "=" * 70)
    print("PHASE 9 COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9 — Ignition Precursors + Quality Filters")
    parser.add_argument("--reports", type=str,
                        default=r"\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports",
                        help="Path to Morpheus reports directory")
    parser.add_argument("--databento", type=str,
                        default=r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades",
                        help="Path to Databento trade data")
    parser.add_argument("--output", type=str, default=r"C:\AI_Bot_Research\results",
                        help="Path to output results")
    parser.add_argument("--symbols", type=str, nargs="+", default=None)
    parser.add_argument("--max-events", type=int, default=None)

    args = parser.parse_args()

    config = ReplayConfig(
        reports_root=Path(args.reports),
        databento_root=Path(args.databento),
        output_root=Path(args.output),
    )

    run_phase9(config=config, symbols=args.symbols, max_events=args.max_events)
