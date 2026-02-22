"""
Phase 9 v2 — Ignition Precursors + Quality Filters (Fixed MFE/MAE)
====================================================================

CRITICAL FIX from v1:
  - 41% of Morpheus trades are flat scratches (PnL=0, MFE=0, MAE=0)
  - v1 included these, drowning all MFE/MAE/R:R metrics to zero
  - v2 filters to ACTIVE trades (PnL != 0) as primary analysis set
  - R:R computed only on trades where MAE > 0 (avoids div-by-zero)

Tests HYP-023, HYP-024, HYP-025 on the 18-day/1,944-trade dataset.

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
from typing import List, Dict, Tuple

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
PERMUTATION_N = 2000

TOD_BUCKETS = {
    "pre_market": (4.0, 9.5),
    "open":       (9.5, 10.5),
    "mid":        (10.5, 14.0),
    "close":      (14.0, 16.0),
    "after_hours": (16.0, 21.0),
}


# ============================================================
# STEP 1: BUILD PER-EVENT DATAFRAME
# ============================================================

def build_event_dataframe(config: ReplayConfig, symbols=None, max_events=None):
    """Load ignition events, compute pressure profiles, merge into DataFrame."""
    print("[PHASE 9 v2] Building per-event DataFrame...")

    events = load_ignition_events(config.reports_root, symbols)
    print(f"  Loaded {len(events)} ignition events")
    if max_events:
        events = events[:max_events]

    rows = []
    matched = 0
    skipped = 0

    for i, evt in enumerate(events):
        if i % 100 == 0:
            print(f"  Event {i+1}/{len(events)}: {evt.symbol} "
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

        entry_et = evt.entry_time.tz_convert("US/Eastern")
        hour_decimal = entry_et.hour + entry_et.minute / 60.0
        tod_bucket = "unknown"
        for bname, (s, e) in TOD_BUCKETS.items():
            if s <= hour_decimal < e:
                tod_bucket = bname
                break

        mfe = evt.max_gain_percent
        mae = abs(evt.max_drawdown_percent)
        pnl = evt.pnl

        # Reward:risk — only meaningful when MAE > 0
        if mae > 0 and mfe > 0:
            rr = min(mfe / mae, 20.0)
        elif mae == 0 and mfe > 0:
            rr = 20.0  # capped — no drawdown
        else:
            rr = 0.0

        rows.append({
            "trade_id": evt.trade_id,
            "symbol": evt.symbol,
            "entry_time": evt.entry_time,
            "entry_time_et": entry_et,
            "hour_et": hour_decimal,
            "tod_bucket": tod_bucket,
            "date": entry_et.strftime("%Y-%m-%d"),

            "pnl": pnl,
            "pnl_percent": evt.pnl_percent,
            "is_winner": 1 if pnl > 0 else 0,
            "is_flat": 1 if pnl == 0 else 0,
            "is_active": 1 if pnl != 0 else 0,
            "mfe_pct": mfe,
            "mae_pct": mae,
            "reward_risk": rr,
            "hold_time_sec": evt.hold_time_seconds,
            "entry_price": evt.entry_price,
            "exit_category": evt.primary_exit_category,
            "volatility_regime": evt.volatility_regime,

            "peak_pressure_z": profile.peak_pressure_z_pre,
            "mean_pressure_z": profile.mean_pressure_z_pre,
            "pressure_direction": profile.pressure_direction_pre,
            "buildup_rate": profile.pressure_buildup_rate,
            "first_cross_sec": profile.first_threshold_cross_sec,
            "bars_above_threshold": profile.bars_above_threshold_pre,
            "pressure_consistency": profile.pressure_consistency,
            "volume_acceleration": profile.volume_acceleration,
            "pre_bars": profile.pre_bars,
        })

    print(f"  Matched: {matched} / {len(events)} ({matched/len(events)*100:.1f}%)")

    df = pd.DataFrame(rows)
    n_flat = df["is_flat"].sum()
    n_active = df["is_active"].sum()
    print(f"  Flat (scratch) trades: {n_flat} ({n_flat/len(df)*100:.1f}%)")
    print(f"  Active trades: {n_active} ({n_active/len(df)*100:.1f}%)")

    return df


# ============================================================
# STEP 2: METRICS
# ============================================================

def compute_metrics(df, label=""):
    """Compute metrics for a group. Expects ACTIVE trades (pnl != 0)."""
    n = len(df)
    if n == 0:
        return {"label": label, "n": 0, "win_rate": 0, "mean_pnl": 0,
                "median_pnl": 0, "total_pnl": 0, "mean_pnl_pct": 0,
                "median_mfe": 0, "mean_mfe": 0, "median_mae": 0, "mean_mae": 0,
                "median_rr": 0, "mean_rr": 0, "n_with_mae": 0, "median_rr_active": 0,
                "n_with_both": 0, "median_rr_both": 0, "mean_hold_sec": 0,
                "unique_symbols": 0, "unique_dates": 0, "n_winners": 0, "n_losers": 0}

    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] < 0]
    has_mae = df[df["mae_pct"] > 0]
    has_both = df[(df["mfe_pct"] > 0) & (df["mae_pct"] > 0)]

    return {
        "label": label,
        "n": n,
        "n_winners": len(winners),
        "n_losers": len(losers),
        "win_rate": round(len(winners) / n * 100, 1),
        "mean_pnl": round(float(df["pnl"].mean()), 2),
        "median_pnl": round(float(df["pnl"].median()), 2),
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "mean_pnl_pct": round(float(df["pnl_percent"].mean()), 3),
        "median_mfe": round(float(df["mfe_pct"].median()), 3),
        "mean_mfe": round(float(df["mfe_pct"].mean()), 3),
        "median_mae": round(float(df["mae_pct"].median()), 3),
        "mean_mae": round(float(df["mae_pct"].mean()), 3),
        "median_rr": round(float(df["reward_risk"].median()), 3),
        "mean_rr": round(float(df["reward_risk"].mean()), 3),
        "n_with_mae": len(has_mae),
        "median_rr_active": round(float(has_mae["reward_risk"].median()), 3) if len(has_mae) > 0 else 0,
        "n_with_both": len(has_both),
        "median_rr_both": round(float(has_both["reward_risk"].median()), 3) if len(has_both) > 0 else 0,
        "mean_hold_sec": round(float(df["hold_time_sec"].mean()), 1),
        "unique_symbols": int(df["symbol"].nunique()),
        "unique_dates": int(df["date"].nunique()),
    }


def bootstrap_ci(arr, stat_fn=np.mean, n_boot=BOOTSTRAP_N, ci=95):
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 5:
        return (float("nan"), float("nan"))
    stats = [stat_fn(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(stats, (100 - ci) / 2)
    hi = np.percentile(stats, 100 - (100 - ci) / 2)
    return (round(float(lo), 4), round(float(hi), 4))


def permutation_test(a, b, stat_fn=np.mean, n_perm=PERMUTATION_N):
    """Two-sample permutation test. Returns p-value."""
    rng = np.random.default_rng(42)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    obs = stat_fn(a) - stat_fn(b)
    combo = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combo)
        perm_diff = stat_fn(perm[:na]) - stat_fn(perm[na:])
        if abs(perm_diff) >= abs(obs):
            count += 1
    return round(count / n_perm, 4)


# ============================================================
# STEP 3: EXPERIMENTS
# ============================================================

def print_metrics_line(label, m, baseline_wr=None):
    """Print one-line summary of a group."""
    delta = ""
    if baseline_wr is not None and m["n"] > 0:
        d = m["win_rate"] - baseline_wr
        delta = f" (Δ{d:+.1f}pp)"
    print(f"    {label:<18} n={m['n']:>4}  WR={m['win_rate']:>5.1f}%{delta:<12} "
          f"PnL=${m['mean_pnl']:>+7.2f}  "
          f"MFE={m['median_mfe']:.3f}%  MAE={m['median_mae']:.3f}%  "
          f"R:R={m['median_rr']:.3f}  "
          f"R:R(active)={m['median_rr_active']:.3f}(n={m['n_with_mae']})")


def experiment_slope(df):
    """HYP-023: Pressure slope filter sweep."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 — HYP-023: Pressure Slope Filter")
    print("=" * 80)

    base = compute_metrics(df, "BASELINE")
    print(f"\n  BASELINE (active trades):")
    print_metrics_line("ALL", base)

    pcts = df["buildup_rate"].quantile([0.25, 0.40, 0.50, 0.60, 0.75])
    thresholds = [
        ("P25", float(pcts[0.25])),
        ("P40", float(pcts[0.40])),
        ("slope >= 0", 0.0),
        ("P60", float(pcts[0.60])),
        ("P75", float(pcts[0.75])),
    ]

    print(f"\n  Buildup rate distribution:")
    print(f"    mean={df['buildup_rate'].mean():.4f}  median={df['buildup_rate'].median():.4f}  "
          f"std={df['buildup_rate'].std():.4f}")
    for label, val in thresholds:
        print(f"    {label} = {val:.4f}")

    results = {"baseline": base, "thresholds": []}

    for label, thresh in thresholds:
        passed = df[df["buildup_rate"] >= thresh]
        rejected = df[df["buildup_rate"] < thresh]
        mp = compute_metrics(passed, f"PASS({label})")
        mr = compute_metrics(rejected, f"REJECT({label})")

        p_wr = permutation_test(passed["is_winner"].values, rejected["is_winner"].values)
        p_pnl = permutation_test(passed["pnl"].values, rejected["pnl"].values)
        p_mfe = permutation_test(passed["mfe_pct"].values, rejected["mfe_pct"].values)
        p_mae = permutation_test(passed["mae_pct"].values, rejected["mae_pct"].values)

        ci_wr = bootstrap_ci(passed["is_winner"].values * 100)
        ci_rr = bootstrap_ci(passed["reward_risk"].values, stat_fn=np.median)
        ci_mfe = bootstrap_ci(passed["mfe_pct"].values, stat_fn=np.median)
        ci_pnl = bootstrap_ci(passed["pnl"].values)

        sig_wr = "***" if p_wr < 0.01 else ("*" if p_wr < 0.05 else "ns")
        sig_pnl = "***" if p_pnl < 0.01 else ("*" if p_pnl < 0.05 else "ns")

        entry = {
            "label": label, "threshold": thresh,
            "pass": mp, "reject": mr,
            "p_wr": p_wr, "p_pnl": p_pnl, "p_mfe": p_mfe, "p_mae": p_mae,
            "ci_wr": ci_wr, "ci_rr": ci_rr, "ci_mfe": ci_mfe, "ci_pnl": ci_pnl,
        }
        results["thresholds"].append(entry)

        print(f"\n  {label} (threshold={thresh:.4f}):")
        print_metrics_line("PASS", mp, base["win_rate"])
        print_metrics_line("REJECT", mr, base["win_rate"])
        print(f"    p(WR)={p_wr:.4f} {sig_wr}  p(PnL)={p_pnl:.4f} {sig_pnl}  "
              f"p(MFE)={p_mfe:.4f}  p(MAE)={p_mae:.4f}")
        print(f"    CI95(WR)=[{ci_wr[0]:.1f}%, {ci_wr[1]:.1f}%]  "
              f"CI95(MFE)=[{ci_mfe[0]:.3f}%, {ci_mfe[1]:.3f}%]  "
              f"CI95(PnL)=[${ci_pnl[0]:.2f}, ${ci_pnl[1]:.2f}]")

    return results


def experiment_volume(df):
    """HYP-024: Volume acceleration threshold sweep."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 — HYP-024: Volume Acceleration Threshold")
    print("=" * 80)

    base = compute_metrics(df, "BASELINE")

    quantiles = [0.50, 0.60, 0.70, 0.80, 0.90]
    qvals = df["volume_acceleration"].quantile(quantiles)

    print(f"\n  Volume acceleration distribution:")
    print(f"    mean={df['volume_acceleration'].mean():.2f}  "
          f"median={df['volume_acceleration'].median():.2f}  "
          f"std={df['volume_acceleration'].std():.2f}")
    for q in quantiles:
        print(f"    P{int(q*100)} = {qvals[q]:.2f}")

    print(f"\n  BASELINE:")
    print_metrics_line("ALL", base)

    results = {"baseline": base, "quantiles": []}

    for q in quantiles:
        thresh = float(qvals[q])
        label = f"P{int(q*100)}"

        passed = df[df["volume_acceleration"] >= thresh]
        rejected = df[df["volume_acceleration"] < thresh]
        mp = compute_metrics(passed, f"PASS({label})")
        mr = compute_metrics(rejected, f"REJECT({label})")

        p_wr = permutation_test(passed["is_winner"].values, rejected["is_winner"].values)
        p_pnl = permutation_test(passed["pnl"].values, rejected["pnl"].values)
        p_mfe = permutation_test(passed["mfe_pct"].values, rejected["mfe_pct"].values)
        p_mae = permutation_test(passed["mae_pct"].values, rejected["mae_pct"].values)

        ci_wr = bootstrap_ci(passed["is_winner"].values * 100)
        ci_rr = bootstrap_ci(passed["reward_risk"].values, stat_fn=np.median)
        ci_mfe = bootstrap_ci(passed["mfe_pct"].values, stat_fn=np.median)
        ci_pnl = bootstrap_ci(passed["pnl"].values)

        sig_wr = "***" if p_wr < 0.01 else ("*" if p_wr < 0.05 else "ns")

        entry = {
            "label": label, "quantile": q, "threshold": thresh,
            "pass": mp, "reject": mr,
            "p_wr": p_wr, "p_pnl": p_pnl, "p_mfe": p_mfe, "p_mae": p_mae,
            "ci_wr": ci_wr, "ci_rr": ci_rr, "ci_mfe": ci_mfe, "ci_pnl": ci_pnl,
        }
        results["quantiles"].append(entry)

        print(f"\n  >= {label} (threshold={thresh:.2f}):")
        print_metrics_line("PASS", mp, base["win_rate"])
        print_metrics_line("REJECT", mr, base["win_rate"])
        print(f"    p(WR)={p_wr:.4f} {sig_wr}  p(PnL)={p_pnl:.4f}  "
              f"p(MFE)={p_mfe:.4f}  p(MAE)={p_mae:.4f}")
        print(f"    CI95(WR)=[{ci_wr[0]:.1f}%, {ci_wr[1]:.1f}%]  "
              f"CI95(MFE)=[{ci_mfe[0]:.3f}%, {ci_mfe[1]:.3f}%]  "
              f"CI95(PnL)=[${ci_pnl[0]:.2f}, ${ci_pnl[1]:.2f}]")

    return results


def experiment_combined(df, slope_thresh, vol_q):
    """HYP-025: Combined gate."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 — HYP-025: Combined Gate")
    print("=" * 80)

    vol_thresh = float(df["volume_acceleration"].quantile(vol_q))
    print(f"\n  Gate: buildup_rate >= {slope_thresh}  AND  vol_accel >= P{int(vol_q*100)} ({vol_thresh:.2f})")

    both = df[(df["buildup_rate"] >= slope_thresh) & (df["volume_acceleration"] >= vol_thresh)]
    slope_only = df[(df["buildup_rate"] >= slope_thresh) & (df["volume_acceleration"] < vol_thresh)]
    vol_only = df[(df["buildup_rate"] < slope_thresh) & (df["volume_acceleration"] >= vol_thresh)]
    neither = df[(df["buildup_rate"] < slope_thresh) & (df["volume_acceleration"] < vol_thresh)]

    base = compute_metrics(df, "ALL")
    m_both = compute_metrics(both, "BOTH PASS")
    m_slope = compute_metrics(slope_only, "SLOPE ONLY")
    m_vol = compute_metrics(vol_only, "VOL ONLY")
    m_neither = compute_metrics(neither, "NEITHER")

    print(f"\n  Group breakdown:")
    for label, m in [("ALL (baseline)", base), ("BOTH PASS", m_both),
                     ("SLOPE ONLY", m_slope), ("VOL ONLY", m_vol), ("NEITHER", m_neither)]:
        print_metrics_line(label, m, base["win_rate"])

    # Statistical tests
    tests = {}
    for comp_name, ga, gb in [("BOTH_vs_NEITHER", both, neither),
                               ("BOTH_vs_ALL", both, df),
                               ("SLOPE_vs_ALL", slope_only, df),
                               ("VOL_vs_ALL", vol_only, df)]:
        if len(ga) >= 5 and len(gb) >= 5:
            tests[comp_name] = {
                "p_wr": permutation_test(ga["is_winner"].values, gb["is_winner"].values),
                "p_pnl": permutation_test(ga["pnl"].values, gb["pnl"].values),
                "p_mfe": permutation_test(ga["mfe_pct"].values, gb["mfe_pct"].values),
            }
        else:
            tests[comp_name] = {"p_wr": float("nan"), "p_pnl": float("nan"), "p_mfe": float("nan")}

    ci_both_wr = bootstrap_ci(both["is_winner"].values * 100) if len(both) >= 5 else (float("nan"), float("nan"))
    ci_both_rr = bootstrap_ci(both["reward_risk"].values, stat_fn=np.median) if len(both) >= 5 else (float("nan"), float("nan"))
    ci_both_mfe = bootstrap_ci(both["mfe_pct"].values, stat_fn=np.median) if len(both) >= 5 else (float("nan"), float("nan"))
    ci_both_pnl = bootstrap_ci(both["pnl"].values) if len(both) >= 5 else (float("nan"), float("nan"))

    print(f"\n  Statistical tests:")
    for comp, t in tests.items():
        sig = "***" if t["p_wr"] < 0.01 else ("*" if t["p_wr"] < 0.05 else "ns")
        print(f"    {comp}: p(WR)={t['p_wr']:.4f} {sig}  p(PnL)={t['p_pnl']:.4f}  p(MFE)={t['p_mfe']:.4f}")

    print(f"\n  BOTH PASS CIs:")
    print(f"    CI95(WR)=[{ci_both_wr[0]:.1f}%, {ci_both_wr[1]:.1f}%]")
    print(f"    CI95(R:R)=[{ci_both_rr[0]:.3f}, {ci_both_rr[1]:.3f}]")
    print(f"    CI95(MFE)=[{ci_both_mfe[0]:.3f}%, {ci_both_mfe[1]:.3f}%]")
    print(f"    CI95(PnL)=[${ci_both_pnl[0]:.2f}, ${ci_both_pnl[1]:.2f}]")

    return {
        "slope_threshold": slope_thresh, "vol_quantile": vol_q, "vol_threshold": vol_thresh,
        "baseline": base, "both_pass": m_both, "slope_only": m_slope,
        "vol_only": m_vol, "neither": m_neither,
        "tests": tests,
        "ci_both_wr": ci_both_wr, "ci_both_rr": ci_both_rr,
        "ci_both_mfe": ci_both_mfe, "ci_both_pnl": ci_both_pnl,
    }


# ============================================================
# STEP 4: STABILITY CHECKS
# ============================================================

def stability_ticker(df, slope_thresh, vol_q):
    """Per-ticker breakdown."""
    print("\n" + "=" * 80)
    print("STABILITY — Per-Ticker (n >= 5 active trades)")
    print("=" * 80)

    vol_thresh = float(df["volume_acceleration"].quantile(vol_q))
    syms = df["symbol"].value_counts()
    syms = syms[syms >= 5].index.tolist()

    print(f"\n  Tickers with n >= 5 active: {len(syms)}")
    print(f"\n  {'SYM':<8} {'N':>4} {'N_P':>4} {'WR_ALL':>7} {'WR_PASS':>8} "
          f"{'Δ WR':>6} {'PNL_ALL':>9} {'PNL_PASS':>9} "
          f"{'MFE_ALL':>8} {'MFE_PASS':>9} {'MAE_ALL':>8} {'MAE_PASS':>9}")
    print(f"  {'-'*105}")

    results = {}
    for sym in sorted(syms):
        sdf = df[df["symbol"] == sym]
        sp = sdf[(sdf["buildup_rate"] >= slope_thresh) & (sdf["volume_acceleration"] >= vol_thresh)]

        n = len(sdf)
        np_ = len(sp)
        wr_all = sdf["is_winner"].mean() * 100
        pnl_all = sdf["pnl"].mean()
        mfe_all = sdf["mfe_pct"].median()
        mae_all = sdf["mae_pct"].median()

        if np_ > 0:
            wr_p = sp["is_winner"].mean() * 100
            pnl_p = sp["pnl"].mean()
            mfe_p = sp["mfe_pct"].median()
            mae_p = sp["mae_pct"].median()
            d = wr_p - wr_all
        else:
            wr_p = pnl_p = mfe_p = mae_p = d = float("nan")

        results[sym] = {"n": n, "n_pass": np_, "wr_all": wr_all, "wr_pass": wr_p,
                        "delta": d, "pnl_all": pnl_all, "pnl_pass": pnl_p,
                        "mfe_all": mfe_all, "mfe_pass": mfe_p,
                        "mae_all": mae_all, "mae_pass": mae_p}

        wr_ps = f"{wr_p:>7.1f}%" if not np.isnan(wr_p) else "     N/A"
        ds = f"{d:>+5.1f}" if not np.isnan(d) else "  N/A"
        pnl_ps = f"${pnl_p:>+8.2f}" if not np.isnan(pnl_p) else "      N/A"
        mfe_ps = f"{mfe_p:>8.3f}%" if not np.isnan(mfe_p) else "      N/A"
        mae_ps = f"{mae_p:>8.3f}%" if not np.isnan(mae_p) else "      N/A"

        print(f"  {sym:<8} {n:>4} {np_:>4} {wr_all:>6.1f}% {wr_ps} "
              f"{ds} ${pnl_all:>+8.2f} {pnl_ps} "
              f"{mfe_all:>7.3f}% {mfe_ps} {mae_all:>7.3f}% {mae_ps}")

    improved = sum(1 for v in results.values() if not np.isnan(v["delta"]) and v["delta"] > 0)
    total = sum(1 for v in results.values() if not np.isnan(v["delta"]))
    print(f"\n  Improved: {improved}/{total} tickers")
    return results


def stability_tod(df, slope_thresh, vol_q):
    """Time-of-day breakdown."""
    print("\n" + "=" * 80)
    print("STABILITY — Time-of-Day Buckets")
    print("=" * 80)

    vol_thresh = float(df["volume_acceleration"].quantile(vol_q))

    print(f"\n  {'BUCKET':<14} {'N':>5} {'N_P':>5} {'WR_ALL':>7} {'WR_PASS':>8} "
          f"{'Δ WR':>6} {'PNL_ALL':>9} {'PNL_PASS':>9} "
          f"{'MFE_ALL':>8} {'MFE_PASS':>9} {'MAE_ALL':>8} {'MAE_PASS':>9}")
    print(f"  {'-'*115}")

    results = {}
    for bname in ["pre_market", "open", "mid", "close", "after_hours"]:
        bdf = df[df["tod_bucket"] == bname]
        bp = bdf[(bdf["buildup_rate"] >= slope_thresh) & (bdf["volume_acceleration"] >= vol_thresh)]

        n = len(bdf)
        np_ = len(bp)
        if n == 0:
            continue

        wr_all = bdf["is_winner"].mean() * 100
        pnl_all = bdf["pnl"].mean()
        mfe_all = bdf["mfe_pct"].median()
        mae_all = bdf["mae_pct"].median()

        if np_ > 0:
            wr_p = bp["is_winner"].mean() * 100
            pnl_p = bp["pnl"].mean()
            mfe_p = bp["mfe_pct"].median()
            mae_p = bp["mae_pct"].median()
            d = wr_p - wr_all
        else:
            wr_p = pnl_p = mfe_p = mae_p = d = float("nan")

        results[bname] = {"n": n, "n_pass": np_, "wr_all": wr_all, "wr_pass": wr_p,
                          "delta": d, "pnl_all": pnl_all, "pnl_pass": pnl_p,
                          "mfe_all": mfe_all, "mfe_pass": mfe_p,
                          "mae_all": mae_all, "mae_pass": mae_p}

        wr_ps = f"{wr_p:>7.1f}%" if not np.isnan(wr_p) else "     N/A"
        ds = f"{d:>+5.1f}" if not np.isnan(d) else "  N/A"
        pnl_ps = f"${pnl_p:>+8.2f}" if not np.isnan(pnl_p) else "      N/A"
        mfe_ps = f"{mfe_p:>8.3f}%" if not np.isnan(mfe_p) else "      N/A"
        mae_ps = f"{mae_p:>8.3f}%" if not np.isnan(mae_p) else "      N/A"

        print(f"  {bname:<14} {n:>5} {np_:>5} {wr_all:>6.1f}% {wr_ps} "
              f"{ds} ${pnl_all:>+8.2f} {pnl_ps} "
              f"{mfe_all:>7.3f}% {mfe_ps} {mae_all:>7.3f}% {mae_ps}")

    improved = sum(1 for v in results.values() if not np.isnan(v["delta"]) and v["delta"] > 0)
    total = sum(1 for v in results.values() if not np.isnan(v["delta"]))
    print(f"\n  Improved: {improved}/{total} buckets")
    return results


# ============================================================
# STEP 5: REPORT
# ============================================================

def write_report(df, slope_res, vol_res, combined_res, tick_stab, tod_stab, path):
    """Write PHASE9_Results.md."""
    b = slope_res["baseline"]
    cb = combined_res["both_pass"]

    best_slope = max(slope_res["thresholds"], key=lambda t: t["pass"]["win_rate"] if t["pass"]["n"] >= 30 else -1)
    best_vol = max(vol_res["quantiles"], key=lambda q: q["pass"]["win_rate"] if q["pass"]["n"] >= 30 else -1)

    tick_improved = sum(1 for v in tick_stab.values() if not np.isnan(v["delta"]) and v["delta"] > 0)
    tick_total = sum(1 for v in tick_stab.values() if not np.isnan(v["delta"]))
    tod_improved = sum(1 for v in tod_stab.values() if not np.isnan(v["delta"]) and v["delta"] > 0)
    tod_total = sum(1 for v in tod_stab.values() if not np.isnan(v["delta"]))

    combined_p = combined_res["tests"].get("BOTH_vs_NEITHER", {}).get("p_wr", float("nan"))

    L = []
    L.append("# Phase 9 v2 — Ignition Precursors + Quality Filters")
    L.append("")
    L.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    L.append(f"**Dataset:** {b['n']} active trades (PnL ≠ 0) from {b['unique_symbols']} symbols, {b['unique_dates']} days")
    L.append(f"**Fix:** Excluded flat/scratch trades (PnL=0) — 41% of Morpheus trades were instant scratches diluting MFE/MAE/R:R to zero.")
    L.append("")
    L.append("---")
    L.append("")

    # Executive Summary
    L.append("## Executive Summary")
    L.append("")
    if not np.isnan(combined_p) and combined_p < 0.05 and cb["n"] >= 30:
        L.append(f"**Verdict: GO** — Combined gate significant at p={combined_p:.4f}, n={cb['n']}")
    elif cb["n"] >= 30 and cb["win_rate"] > b["win_rate"] + 3:
        L.append(f"**Verdict: CONDITIONAL GO** — Combined gate WR={cb['win_rate']:.1f}% vs baseline {b['win_rate']:.1f}%, p={combined_p:.4f}")
    else:
        L.append(f"**Verdict: Awaiting interpretation** — see results below")
    L.append("")
    L.append("---")
    L.append("")

    # Baseline
    L.append("## Baseline — Active Trades")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    for k, v in [("N", b['n']), ("Win Rate", f"{b['win_rate']:.1f}%"),
                 ("Mean PnL", f"${b['mean_pnl']:.2f}"), ("Total PnL", f"${b['total_pnl']:.2f}"),
                 ("Median MFE", f"{b['median_mfe']:.3f}%"), ("Mean MFE", f"{b['mean_mfe']:.3f}%"),
                 ("Median MAE", f"{b['median_mae']:.3f}%"), ("Mean MAE", f"{b['mean_mae']:.3f}%"),
                 ("Median R:R", f"{b['median_rr']:.3f}"),
                 ("R:R (MAE>0 trades)", f"{b['median_rr_active']:.3f} (n={b['n_with_mae']})"),
                 ("R:R (MFE>0 & MAE>0)", f"{b['median_rr_both']:.3f} (n={b['n_with_both']})"),
                 ("Mean Hold", f"{b['mean_hold_sec']:.0f}s"),
                 ("Symbols", b['unique_symbols']), ("Days", b['unique_dates'])]:
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("---")
    L.append("")

    # Exp 1
    L.append("## Experiment 1 — HYP-023: Pressure Slope Filter")
    L.append("")
    L.append("| Threshold | N Pass | WR Pass | WR Rej | Δ WR | p(WR) | p(PnL) | p(MFE) | p(MAE) | PnL Pass | Med MFE | Med MAE | Med R:R | CI95 WR | CI95 MFE |")
    L.append("|-----------|--------|---------|--------|------|-------|--------|--------|--------|----------|---------|---------|---------|---------|----------|")
    for t in slope_res["thresholds"]:
        p, r = t["pass"], t["reject"]
        dwr = p["win_rate"] - r["win_rate"]
        sig = " *" if t["p_wr"] < 0.05 else ""
        L.append(f"| {t['label']} ({t['threshold']:.4f}) | {p['n']} | {p['win_rate']:.1f}% | {r['win_rate']:.1f}% | "
                 f"{dwr:+.1f} | {t['p_wr']:.4f}{sig} | {t['p_pnl']:.4f} | {t['p_mfe']:.4f} | {t['p_mae']:.4f} | "
                 f"${p['mean_pnl']:+.2f} | {p['median_mfe']:.3f}% | {p['median_mae']:.3f}% | {p['median_rr']:.3f} | "
                 f"[{t['ci_wr'][0]:.1f}%,{t['ci_wr'][1]:.1f}%] | [{t['ci_mfe'][0]:.3f}%,{t['ci_mfe'][1]:.3f}%] |")
    L.append("")
    L.append("---")
    L.append("")

    # Exp 2
    L.append("## Experiment 2 — HYP-024: Volume Acceleration Threshold")
    L.append("")
    L.append("| Pctl | Thresh | N Pass | WR Pass | WR Rej | Δ WR | p(WR) | p(PnL) | p(MFE) | p(MAE) | PnL Pass | Med MFE | Med MAE | Med R:R | CI95 WR | CI95 MFE |")
    L.append("|------|--------|--------|---------|--------|------|-------|--------|--------|--------|----------|---------|---------|---------|---------|----------|")
    for q in vol_res["quantiles"]:
        p, r = q["pass"], q["reject"]
        dwr = p["win_rate"] - r["win_rate"]
        sig = " *" if q["p_wr"] < 0.05 else ""
        L.append(f"| P{int(q['quantile']*100)} | {q['threshold']:.2f} | {p['n']} | {p['win_rate']:.1f}% | {r['win_rate']:.1f}% | "
                 f"{dwr:+.1f} | {q['p_wr']:.4f}{sig} | {q['p_pnl']:.4f} | {q['p_mfe']:.4f} | {q['p_mae']:.4f} | "
                 f"${p['mean_pnl']:+.2f} | {p['median_mfe']:.3f}% | {p['median_mae']:.3f}% | {p['median_rr']:.3f} | "
                 f"[{q['ci_wr'][0]:.1f}%,{q['ci_wr'][1]:.1f}%] | [{q['ci_mfe'][0]:.3f}%,{q['ci_mfe'][1]:.3f}%] |")
    L.append("")
    L.append("---")
    L.append("")

    # Exp 3
    L.append("## Experiment 3 — HYP-025: Combined Gate")
    L.append("")
    L.append(f"Gate: `buildup_rate >= {combined_res['slope_threshold']}` AND `vol_accel >= {combined_res['vol_threshold']:.2f}` (P{int(combined_res['vol_quantile']*100)})")
    L.append("")
    L.append("| Group | N | WR | Mean PnL | Med MFE | Med MAE | Med R:R | R:R(active) | Syms |")
    L.append("|-------|---|----|----------|---------|---------|---------|-------------|------|")
    for lbl, key in [("ALL", "baseline"), ("BOTH PASS", "both_pass"),
                     ("SLOPE ONLY", "slope_only"), ("VOL ONLY", "vol_only"), ("NEITHER", "neither")]:
        m = combined_res[key]
        L.append(f"| {lbl} | {m['n']} | {m['win_rate']:.1f}% | ${m['mean_pnl']:+.2f} | "
                 f"{m['median_mfe']:.3f}% | {m['median_mae']:.3f}% | {m['median_rr']:.3f} | "
                 f"{m['median_rr_active']:.3f}(n={m['n_with_mae']}) | {m['unique_symbols']} |")
    L.append("")
    for comp, t in combined_res["tests"].items():
        sig = " *" if t["p_wr"] < 0.05 else ""
        L.append(f"- **{comp}:** p(WR)={t['p_wr']:.4f}{sig}, p(PnL)={t['p_pnl']:.4f}, p(MFE)={t['p_mfe']:.4f}")
    L.append("")
    L.append(f"BOTH PASS CIs: WR=[{combined_res['ci_both_wr'][0]:.1f}%,{combined_res['ci_both_wr'][1]:.1f}%], "
             f"R:R=[{combined_res['ci_both_rr'][0]:.3f},{combined_res['ci_both_rr'][1]:.3f}], "
             f"MFE=[{combined_res['ci_both_mfe'][0]:.3f}%,{combined_res['ci_both_mfe'][1]:.3f}%]")
    L.append("")
    L.append("---")
    L.append("")

    # ToD
    L.append("## Stability: Time-of-Day")
    L.append("")
    L.append("| Bucket | N | N Pass | WR All | WR Pass | Δ WR | PnL All | PnL Pass | MFE All | MFE Pass | MAE All | MAE Pass |")
    L.append("|--------|---|--------|--------|---------|------|---------|----------|---------|----------|---------|----------|")
    for bname in ["pre_market", "open", "mid", "close", "after_hours"]:
        v = tod_stab.get(bname)
        if not v: continue
        fmt = lambda x: f"{x:.3f}%" if not np.isnan(x) else "N/A"
        fmtd = lambda x: f"${x:+.2f}" if not np.isnan(x) else "N/A"
        fmtw = lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A"
        fmtdelta = lambda x: f"{x:+.1f}" if not np.isnan(x) else "N/A"
        L.append(f"| {bname} | {v['n']} | {v['n_pass']} | {v['wr_all']:.1f}% | {fmtw(v['wr_pass'])} | "
                 f"{fmtdelta(v['delta'])} | {fmtd(v['pnl_all'])} | {fmtd(v['pnl_pass'])} | "
                 f"{fmt(v['mfe_all'])} | {fmt(v['mfe_pass'])} | {fmt(v['mae_all'])} | {fmt(v['mae_pass'])} |")
    L.append("")
    L.append(f"Improved: {tod_improved}/{tod_total} buckets")
    L.append("")
    L.append("---")
    L.append("")

    # Ticker
    L.append("## Stability: Per-Ticker (n >= 5)")
    L.append("")
    L.append("| Sym | N | N_P | WR All | WR Pass | Δ WR | PnL All | PnL Pass | MFE All | MFE Pass | MAE All | MAE Pass |")
    L.append("|-----|---|-----|--------|---------|------|---------|----------|---------|----------|---------|----------|")
    for sym in sorted(tick_stab.keys()):
        v = tick_stab[sym]
        fmt = lambda x: f"{x:.3f}%" if not np.isnan(x) else "N/A"
        fmtd = lambda x: f"${x:+.2f}" if not np.isnan(x) else "N/A"
        fmtw = lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A"
        fmtdelta = lambda x: f"{x:+.1f}" if not np.isnan(x) else "N/A"
        L.append(f"| {sym} | {v['n']} | {v['n_pass']} | {v['wr_all']:.1f}% | {fmtw(v['wr_pass'])} | "
                 f"{fmtdelta(v['delta'])} | {fmtd(v['pnl_all'])} | {fmtd(v['pnl_pass'])} | "
                 f"{fmt(v['mfe_all'])} | {fmt(v['mfe_pass'])} | {fmt(v['mae_all'])} | {fmt(v['mae_pass'])} |")
    L.append("")
    L.append(f"Improved: {tick_improved}/{tick_total} tickers")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    return "\n".join(L)


# ============================================================
# MAIN
# ============================================================

def run_phase9(config=None, symbols=None, max_events=None):
    if config is None:
        config = ReplayConfig()

    print("=" * 80)
    print("PHASE 9 v2 — Ignition Precursors + Quality Filters (MFE/MAE Fixed)")
    print("=" * 80)

    df_all = build_event_dataframe(config, symbols, max_events)

    if len(df_all) < 30:
        print(f"[ERROR] Only {len(df_all)} events. Need >= 30.")
        return

    # Filter to active trades
    df = df_all[df_all["is_active"] == 1].copy()
    n_flat = len(df_all) - len(df)

    print(f"\n  === ACTIVE TRADES FILTER ===")
    print(f"  Total profiled: {len(df_all)}")
    print(f"  Flat/scratch (excluded): {n_flat} ({n_flat/len(df_all)*100:.1f}%)")
    print(f"  Active (analysis set): {len(df)} ({len(df)/len(df_all)*100:.1f}%)")

    raw_path = config.output_root / "phase9v2_event_data.csv"
    df.to_csv(raw_path, index=False)
    print(f"  Saved: {raw_path}")

    # Experiments
    slope_res = experiment_slope(df)
    vol_res = experiment_volume(df)

    best_vol_q = 0.70
    for q in vol_res["quantiles"]:
        if q["pass"]["n"] >= 30 and q["pass"]["win_rate"] >= vol_res["quantiles"][0]["pass"]["win_rate"]:
            best_vol_q = q["quantile"]

    combined_res = experiment_combined(df, slope_thresh=0.0, vol_q=best_vol_q)
    tick_stab = stability_ticker(df, 0.0, best_vol_q)
    tod_stab = stability_tod(df, 0.0, best_vol_q)

    report_path = config.output_root.parent / "docs" / "Research" / "PHASE9_Results.md"
    write_report(df, slope_res, vol_res, combined_res, tick_stab, tod_stab, report_path)
    print(f"\n  Report: {report_path}")

    json_path = config.output_root / "phase9v2_results.json"
    all_res = {
        "slope": slope_res, "volume": vol_res, "combined": combined_res,
        "ticker_stability": tick_stab, "tod_stability": tod_stab,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    print("\n" + "=" * 80)
    print("PHASE 9 v2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 9 v2")
    parser.add_argument("--reports", type=str,
                        default=r"\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports")
    parser.add_argument("--databento", type=str,
                        default=r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades")
    parser.add_argument("--output", type=str, default=r"C:\AI_Bot_Research\results")
    parser.add_argument("--symbols", type=str, nargs="+", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    config = ReplayConfig(
        reports_root=Path(args.reports),
        databento_root=Path(args.databento),
        output_root=Path(args.output),
    )
    run_phase9(config=config, symbols=args.symbols, max_events=args.max_events)
