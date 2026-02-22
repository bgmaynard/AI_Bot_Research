"""
Phase 9 v2 — Ignition Precursors + Quality Filters
====================================================

Rebuilt with:
- MFE/MAE reconstructed from Databento ticks during hold period
- Percentile-based thresholds for BOTH slope and volume acceleration
- PnL percent as primary outcome (always reliable)
- Bootstrap CIs on all key metrics

Usage:
    python mrl/phase9_experiment.py \
        --reports "\\\\Bob1\\c\\ai_project_hub\\store\\code\\IBKR_Algo_BOT_V2\\reports" \
        --databento "Z:\\AI_BOT_DATA\\databento_cache\\XNAS.ITCH\\trades"
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from replay.replay_engine import (
    ReplayConfig, IgnitionEvent, PressureProfile,
    load_ignition_events, load_raw_trades_around_event,
    compute_pressure_profile, find_databento_file,
)

BOOTSTRAP_N = 2000
RNG = np.random.default_rng(42)

TOD_BUCKETS = {
    "pre_market": (4.0, 9.5),
    "open":       (9.5, 10.5),
    "mid":        (10.5, 14.0),
    "close":      (14.0, 16.0),
    "after_hours": (16.0, 21.0),
}


def reconstruct_mfe_mae(databento_root, symbol, entry_time, exit_time, entry_price):
    """Reconstruct true MFE/MAE from Databento tick data during hold period."""
    try:
        import databento as db
    except ImportError:
        return {"mfe_pct": None, "mae_pct": None}

    file_path = find_databento_file(databento_root, symbol, entry_time)
    if file_path is None:
        return {"mfe_pct": None, "mae_pct": None}

    try:
        store = db.DBNStore.from_file(file_path)
        df = store.to_df()
    except Exception:
        return {"mfe_pct": None, "mae_pct": None}

    if len(df) == 0:
        return {"mfe_pct": None, "mae_pct": None}

    df = df.reset_index()
    if df["ts_recv"].dt.tz is None:
        df["ts_recv"] = df["ts_recv"].dt.tz_localize("UTC")
    else:
        df["ts_recv"] = df["ts_recv"].dt.tz_convert("UTC")

    hold_df = df[(df["ts_recv"] >= entry_time) & (df["ts_recv"] <= exit_time)]
    if len(hold_df) == 0 or entry_price <= 0:
        return {"mfe_pct": None, "mae_pct": None}

    if "price" not in hold_df.columns:
        return {"mfe_pct": None, "mae_pct": None}

    prices = hold_df["price"].values
    prices = prices[prices > 0]
    if len(prices) == 0:
        return {"mfe_pct": None, "mae_pct": None}

    max_p = float(np.max(prices))
    min_p = float(np.min(prices))
    mfe_pct = (max_p - entry_price) / entry_price * 100.0
    mae_pct = (entry_price - min_p) / entry_price * 100.0

    return {"mfe_pct": mfe_pct, "mae_pct": mae_pct, "tick_count": len(prices)}


def build_event_dataframe(config):
    """Load events, compute pressure profiles + reconstruct MFE/MAE."""
    print("[PHASE 9 v2] Building per-event DataFrame...")

    events = load_ignition_events(config.reports_root)
    print(f"  Loaded {len(events)} ignition events")

    rows = []
    matched = 0
    skipped = 0
    mfe_recon = 0

    for i, evt in enumerate(events):
        if i % 100 == 0:
            print(f"  [{i+1}/{len(events)}] {evt.symbol} "
                  f"{evt.entry_time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"(matched={matched}, mfe_recon={mfe_recon})")

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

        # Reconstruct MFE/MAE
        exit_time = evt.entry_time + pd.Timedelta(seconds=evt.hold_time_seconds)
        mfe_data = reconstruct_mfe_mae(
            config.databento_root, evt.symbol,
            evt.entry_time, exit_time, evt.entry_price,
        )

        if mfe_data["mfe_pct"] is not None:
            mfe_recon += 1
            mfe_pct = mfe_data["mfe_pct"]
            mae_pct = mfe_data["mae_pct"]
            mfe_src = "databento"
        else:
            mfe_pct = evt.max_gain_percent
            mae_pct = abs(evt.max_drawdown_percent)
            mfe_src = "ledger"

        if mae_pct > 0.001:
            rr = mfe_pct / mae_pct
        elif mfe_pct > 0:
            rr = 10.0
        else:
            rr = 0.0

        entry_et = evt.entry_time.tz_convert("US/Eastern")
        hour_dec = entry_et.hour + entry_et.minute / 60.0
        tod = "unknown"
        for bname, (s, e) in TOD_BUCKETS.items():
            if s <= hour_dec < e:
                tod = bname
                break

        rows.append({
            "trade_id": evt.trade_id, "symbol": evt.symbol,
            "entry_time": evt.entry_time, "hour_et": hour_dec,
            "tod_bucket": tod, "date": entry_et.strftime("%Y-%m-%d"),
            "pnl": evt.pnl, "pnl_pct": evt.pnl_percent,
            "is_winner": 1 if evt.pnl > 0 else 0,
            "mfe_pct": mfe_pct, "mae_pct": mae_pct,
            "reward_risk": min(rr, 20.0),
            "hold_sec": evt.hold_time_seconds, "mfe_source": mfe_src,
            "peak_pz": profile.peak_pressure_z_pre,
            "buildup_rate": profile.pressure_buildup_rate,
            "vol_accel": profile.volume_acceleration,
            "pressure_consistency": profile.pressure_consistency,
            "bars_above_thresh": profile.bars_above_threshold_pre,
        })

    print(f"\n  Matched: {matched}/{len(events)} ({matched/len(events)*100:.1f}%)")
    print(f"  MFE/MAE from Databento: {mfe_recon}")

    return pd.DataFrame(rows)


# ---- STATISTICS ----

def bootstrap_ci(arr, stat_fn=np.mean, n_boot=BOOTSTRAP_N, ci=95):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return (float("nan"), float("nan"))
    stats = [stat_fn(RNG.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(stats, (100 - ci) / 2)
    hi = np.percentile(stats, 100 - (100 - ci) / 2)
    return (round(float(lo), 4), round(float(hi), 4))


def perm_test(a, b, stat_fn=np.mean, n_perm=2000):
    a = np.array(a, dtype=float); a = a[~np.isnan(a)]
    b = np.array(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    obs = stat_fn(a) - stat_fn(b)
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        RNG.shuffle(combined)
        d = stat_fn(combined[:na]) - stat_fn(combined[na:])
        if abs(d) >= abs(obs):
            count += 1
    return count / n_perm


def gs(df, label=""):
    n = len(df)
    if n == 0:
        return {"label": label, "n": 0, "win_rate": 0, "mean_pnl": 0,
                "sum_pnl": 0, "mean_pnl_pct": 0, "median_mfe": 0,
                "mean_mfe": 0, "median_mae": 0, "mean_mae": 0,
                "median_rr": 0, "mean_rr": 0, "symbols": 0, "dates": 0}
    return {
        "label": label, "n": n,
        "win_rate": round(df["is_winner"].mean() * 100, 1),
        "mean_pnl": round(df["pnl"].mean(), 3),
        "sum_pnl": round(df["pnl"].sum(), 2),
        "mean_pnl_pct": round(df["pnl_pct"].mean(), 4),
        "median_mfe": round(df["mfe_pct"].median(), 3),
        "mean_mfe": round(df["mfe_pct"].mean(), 3),
        "median_mae": round(df["mae_pct"].median(), 3),
        "mean_mae": round(df["mae_pct"].mean(), 3),
        "median_rr": round(df["reward_risk"].median(), 3),
        "mean_rr": round(df["reward_risk"].mean(), 3),
        "symbols": df["symbol"].nunique(),
        "dates": df["date"].nunique(),
    }


# ---- SWEEPS ----

def run_sweep(df, col, quantiles=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]):
    baseline = gs(df, "baseline")
    sweeps = []
    for q in quantiles:
        thresh = float(df[col].quantile(q))
        passed = df[df[col] >= thresh]
        rejected = df[df[col] < thresh]
        g_p = gs(passed, f"P{int(q*100)}")
        g_r = gs(rejected, f"<P{int(q*100)}")
        p_wr = perm_test(passed["is_winner"].values, rejected["is_winner"].values)
        p_pnl = perm_test(passed["pnl_pct"].values, rejected["pnl_pct"].values)
        ci_wr = bootstrap_ci(passed["is_winner"].values * 100)
        ci_rr = bootstrap_ci(passed["reward_risk"].values, stat_fn=np.median)
        sweeps.append({
            "quantile": q, "threshold": round(thresh, 4),
            "pass": g_p, "reject": g_r,
            "p_wr": round(p_wr, 4) if not np.isnan(p_wr) else "N/A",
            "p_pnl": round(p_pnl, 4) if not np.isnan(p_pnl) else "N/A",
            "ci_wr": ci_wr, "ci_rr": ci_rr,
        })
    return {"baseline": baseline, "sweeps": sweeps}


def run_combined(df, slope_t, vol_t, s_lbl, v_lbl):
    both = df[(df["buildup_rate"] >= slope_t) & (df["vol_accel"] >= vol_t)]
    s_only = df[(df["buildup_rate"] >= slope_t) & (df["vol_accel"] < vol_t)]
    v_only = df[(df["buildup_rate"] < slope_t) & (df["vol_accel"] >= vol_t)]
    neither = df[(df["buildup_rate"] < slope_t) & (df["vol_accel"] < vol_t)]

    result = {
        "baseline": gs(df, "ALL"), "both": gs(both, "BOTH PASS"),
        "slope_only": gs(s_only, "SLOPE ONLY"), "vol_only": gs(v_only, "VOL ONLY"),
        "neither": gs(neither, "NEITHER"),
        "slope_thresh": slope_t, "vol_thresh": vol_t,
        "slope_label": s_lbl, "vol_label": v_lbl,
    }

    if len(both) >= 5 and len(neither) >= 5:
        result["p_wr"] = round(perm_test(both["is_winner"].values, neither["is_winner"].values), 4)
        result["p_pnl"] = round(perm_test(both["pnl_pct"].values, neither["pnl_pct"].values), 4)
        result["ci_wr"] = bootstrap_ci(both["is_winner"].values * 100)
        result["ci_rr"] = bootstrap_ci(both["reward_risk"].values, stat_fn=np.median)
    else:
        result["p_wr"] = "N/A"
        result["p_pnl"] = "N/A"
        result["ci_wr"] = ("N/A", "N/A")
        result["ci_rr"] = ("N/A", "N/A")

    return result


def stability_tod(df, st, vt):
    results = {}
    for name in ["pre_market", "open", "mid", "close", "after_hours"]:
        bdf = df[df["tod_bucket"] == name]
        if len(bdf) < 5:
            continue
        bdf_p = bdf[(bdf["buildup_rate"] >= st) & (bdf["vol_accel"] >= vt)]
        wr_all = round(bdf["is_winner"].mean() * 100, 1)
        wr_p = round(bdf_p["is_winner"].mean() * 100, 1) if len(bdf_p) > 0 else None
        d = round(wr_p - wr_all, 1) if wr_p is not None else None
        results[name] = {"n_all": len(bdf), "n_pass": len(bdf_p),
                        "wr_all": wr_all, "wr_pass": wr_p, "delta_wr": d}
    return results


def stability_ticker(df, st, vt, min_n=10):
    results = {}
    for sym in df["symbol"].value_counts().index:
        sdf = df[df["symbol"] == sym]
        if len(sdf) < min_n:
            continue
        sdf_p = sdf[(sdf["buildup_rate"] >= st) & (sdf["vol_accel"] >= vt)]
        wr_all = round(sdf["is_winner"].mean() * 100, 1)
        wr_p = round(sdf_p["is_winner"].mean() * 100, 1) if len(sdf_p) > 0 else None
        d = round(wr_p - wr_all, 1) if wr_p is not None else None
        results[sym] = {"n_all": len(sdf), "n_pass": len(sdf_p),
                       "wr_all": wr_all, "wr_pass": wr_p, "delta_wr": d}
    return results


# ---- REPORT ----

def write_report(df, slope_sw, vol_sw, combos, tod_st, tick_st, path):
    b = slope_sw["baseline"]
    n_db = len(df[df["mfe_source"] == "databento"])
    L = []
    L.append("# Phase 9 — Grid Sweep Results")
    L.append("")
    L.append(f"**N={b['n']}** trades with pressure profiles | "
             f"**{b['symbols']}** symbols | **{b['dates']}** dates")
    L.append(f"**MFE/MAE:** {n_db} reconstructed from Databento ticks, "
             f"{b['n']-n_db} from trade ledger")
    L.append("")

    # Executive summary
    L.append("## Executive Summary")
    L.append("")

    any_sig = False
    for sw in [slope_sw, vol_sw]:
        for s in sw["sweeps"]:
            if s["p_wr"] != "N/A" and s["p_wr"] < 0.05 and s["pass"]["n"] >= 50:
                any_sig = True
    for c in combos:
        if c["p_wr"] != "N/A" and c["p_wr"] < 0.05 and c["both"]["n"] >= 50:
            any_sig = True

    if any_sig:
        L.append("**GO/NO-GO: GO** — At least one filter shows p < 0.05 with n ≥ 50.")
    else:
        L.append("**GO/NO-GO: NO-GO** — No filter reached p < 0.05 with n ≥ 50.")
    L.append("")

    # Baseline
    L.append("## Baseline")
    L.append("")
    L.append(f"| Metric | Value |")
    L.append(f"|--------|-------|")
    L.append(f"| N | {b['n']} |")
    L.append(f"| Win Rate | {b['win_rate']}% |")
    L.append(f"| Mean PnL | ${b['mean_pnl']} |")
    L.append(f"| Sum PnL | ${b['sum_pnl']} |")
    L.append(f"| Median MFE | {b['median_mfe']}% |")
    L.append(f"| Median MAE | {b['median_mae']}% |")
    L.append(f"| Median R:R | {b['median_rr']} |")
    L.append(f"| Mean R:R | {b['mean_rr']} |")
    L.append("")

    # Slope sweep
    L.append("## Experiment 1 — Pressure Slope Sweep (HYP-023)")
    L.append("")
    L.append("| Pctile | Thresh | N | WR | WR Rej | Δ | p(WR) | Med MFE | Med MAE | Med R:R | CI95(WR) |")
    L.append("|--------|--------|---|----|----|---|-------|---------|---------|---------|----------|")
    for s in slope_sw["sweeps"]:
        p = s["pass"]; r = s["reject"]
        d = round(p["win_rate"] - r["win_rate"], 1)
        sig = " *" if s["p_wr"] != "N/A" and s["p_wr"] < 0.05 else ""
        L.append(f"| P{int(s['quantile']*100)} | {s['threshold']} | {p['n']} | "
                 f"{p['win_rate']}% | {r['win_rate']}% | {d:+.1f} | "
                 f"{s['p_wr']}{sig} | {p['median_mfe']}% | {p['median_mae']}% | "
                 f"{p['median_rr']} | [{s['ci_wr'][0]}%, {s['ci_wr'][1]}%] |")
    L.append("")

    # Volume sweep
    L.append("## Experiment 2 — Volume Acceleration Sweep (HYP-024)")
    L.append("")
    L.append("| Pctile | Thresh | N | WR | WR Rej | Δ | p(WR) | Med MFE | Med MAE | Med R:R | CI95(WR) |")
    L.append("|--------|--------|---|----|----|---|-------|---------|---------|---------|----------|")
    for s in vol_sw["sweeps"]:
        p = s["pass"]; r = s["reject"]
        d = round(p["win_rate"] - r["win_rate"], 1)
        sig = " *" if s["p_wr"] != "N/A" and s["p_wr"] < 0.05 else ""
        L.append(f"| P{int(s['quantile']*100)} | {s['threshold']} | {p['n']} | "
                 f"{p['win_rate']}% | {r['win_rate']}% | {d:+.1f} | "
                 f"{s['p_wr']}{sig} | {p['median_mfe']}% | {p['median_mae']}% | "
                 f"{p['median_rr']} | [{s['ci_wr'][0]}%, {s['ci_wr'][1]}%] |")
    L.append("")

    # Combined gates
    L.append("## Experiment 3 — Combined Gates (HYP-025)")
    L.append("")
    for c in combos:
        L.append(f"### slope ≥ {c['slope_label']}, vol ≥ {c['vol_label']}")
        L.append("")
        L.append("| Group | N | WR | Mean PnL | Med MFE | Med MAE | Med R:R |")
        L.append("|-------|---|----|----------|---------|---------|---------|")
        for key in ["baseline", "both", "slope_only", "vol_only", "neither"]:
            g = c[key]
            if g["n"] == 0:
                L.append(f"| {g['label']} | 0 | — | — | — | — | — |")
            else:
                L.append(f"| {g['label']} | {g['n']} | {g['win_rate']}% | "
                         f"${g['mean_pnl']} | {g['median_mfe']}% | "
                         f"{g['median_mae']}% | {g['median_rr']} |")
        L.append(f"\np(WR both vs neither)={c['p_wr']} | "
                 f"CI95(WR)=[{c['ci_wr'][0]}%, {c['ci_wr'][1]}%] | "
                 f"CI95(R:R)=[{c['ci_rr'][0]}, {c['ci_rr'][1]}]")
        L.append("")

    # ToD stability
    L.append("## Stability: Time-of-Day")
    L.append("")
    L.append("| Bucket | N All | N Pass | WR All | WR Pass | Δ WR |")
    L.append("|--------|-------|--------|--------|---------|------|")
    for name in ["pre_market", "open", "mid", "close", "after_hours"]:
        v = tod_st.get(name)
        if not v:
            continue
        wp = f"{v['wr_pass']}%" if v['wr_pass'] is not None else "—"
        d = f"{v['delta_wr']:+.1f}" if v['delta_wr'] is not None else "—"
        L.append(f"| {name} | {v['n_all']} | {v['n_pass']} | "
                 f"{v['wr_all']}% | {wp} | {d} |")
    imp_tod = sum(1 for v in tod_st.values() if v.get("delta_wr") and v["delta_wr"] > 0)
    tot_tod = sum(1 for v in tod_st.values() if v.get("delta_wr") is not None)
    L.append(f"\nBuckets improved: {imp_tod}/{tot_tod}")
    L.append("")

    # Ticker stability
    L.append("## Stability: Per-Ticker (n ≥ 10)")
    L.append("")
    L.append("| Symbol | N All | N Pass | WR All | WR Pass | Δ WR |")
    L.append("|--------|-------|--------|--------|---------|------|")
    for sym in sorted(tick_st.keys()):
        v = tick_st[sym]
        wp = f"{v['wr_pass']}%" if v['wr_pass'] is not None else "—"
        d = f"{v['delta_wr']:+.1f}" if v['delta_wr'] is not None else "—"
        L.append(f"| {sym} | {v['n_all']} | {v['n_pass']} | "
                 f"{v['wr_all']}% | {wp} | {d} |")
    imp_t = sum(1 for v in tick_st.values() if v.get("delta_wr") and v["delta_wr"] > 0)
    tot_t = sum(1 for v in tick_st.values() if v.get("delta_wr") is not None)
    L.append(f"\nTickers improved: {imp_t}/{tot_t}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---- MAIN ----

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9 v2")
    parser.add_argument("--reports", type=str,
                        default=r"\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports")
    parser.add_argument("--databento", type=str,
                        default=r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades")
    parser.add_argument("--output", type=str, default=r"C:\AI_Bot_Research\results")
    args = parser.parse_args()

    config = ReplayConfig(
        reports_root=Path(args.reports),
        databento_root=Path(args.databento),
        output_root=Path(args.output),
    )

    print("=" * 70)
    print("PHASE 9 v2 — Grid Sweep + Databento MFE/MAE Reconstruction")
    print("=" * 70)

    df = build_event_dataframe(config)

    if len(df) < 50:
        print(f"[ERROR] Only {len(df)} events. Need >= 50.")
        return

    csv_path = config.output_root / "phase9v2_event_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Raw data: {csv_path}")
    print(f"\n  N={len(df)}, WR={df['is_winner'].mean()*100:.1f}%, "
          f"Med MFE={df['mfe_pct'].median():.3f}%, "
          f"Med MAE={df['mae_pct'].median():.3f}%, "
          f"Med R:R={df['reward_risk'].median():.3f}")
    print(f"  MFE from Databento: {len(df[df['mfe_source']=='databento'])}/{len(df)}")
    print(f"  Slope: mean={df['buildup_rate'].mean():.4f}, "
          f"med={df['buildup_rate'].median():.4f}")
    print(f"  Vol accel: mean={df['vol_accel'].mean():.1f}, "
          f"med={df['vol_accel'].median():.1f}")

    # Experiment 1: Slope sweep
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — Pressure Slope Percentile Sweep")
    print("=" * 70)
    slope_sw = run_sweep(df, "buildup_rate")
    for s in slope_sw["sweeps"]:
        sig = " *" if s["p_wr"] != "N/A" and s["p_wr"] < 0.05 else ""
        print(f"  P{int(s['quantile']*100):>2} (>={s['threshold']:>8.4f}): "
              f"n={s['pass']['n']:>3}, WR={s['pass']['win_rate']:>5.1f}%, "
              f"MFE={s['pass']['median_mfe']:>6.3f}%, MAE={s['pass']['median_mae']:>6.3f}%, "
              f"R:R={s['pass']['median_rr']:>5.3f}, p={s['p_wr']}{sig}")

    # Experiment 2: Volume sweep
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — Volume Acceleration Percentile Sweep")
    print("=" * 70)
    vol_sw = run_sweep(df, "vol_accel")
    for s in vol_sw["sweeps"]:
        sig = " *" if s["p_wr"] != "N/A" and s["p_wr"] < 0.05 else ""
        print(f"  P{int(s['quantile']*100):>2} (>={s['threshold']:>8.1f}): "
              f"n={s['pass']['n']:>3}, WR={s['pass']['win_rate']:>5.1f}%, "
              f"MFE={s['pass']['median_mfe']:>6.3f}%, MAE={s['pass']['median_mae']:>6.3f}%, "
              f"R:R={s['pass']['median_rr']:>5.3f}, p={s['p_wr']}{sig}")

    # Experiment 3: Combined gates
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — Combined Gates")
    print("=" * 70)

    sp50 = float(df["buildup_rate"].quantile(0.50))
    sp60 = float(df["buildup_rate"].quantile(0.60))
    vp60 = float(df["vol_accel"].quantile(0.60))
    vp70 = float(df["vol_accel"].quantile(0.70))

    combos = [
        (sp50, vp60, f"P50({sp50:.4f})", f"P60({vp60:.1f})"),
        (sp50, vp70, f"P50({sp50:.4f})", f"P70({vp70:.1f})"),
        (sp60, vp60, f"P60({sp60:.4f})", f"P60({vp60:.1f})"),
        (sp60, vp70, f"P60({sp60:.4f})", f"P70({vp70:.1f})"),
        (0.0, vp70, "≥0", f"P70({vp70:.1f})"),
    ]

    combo_results = []
    for st, vt, sl, vl in combos:
        c = run_combined(df, st, vt, sl, vl)
        combo_results.append(c)
        print(f"\n  slope ≥ {sl}, vol ≥ {vl}")
        print(f"    BOTH:    n={c['both']['n']:>3}, WR={c['both']['win_rate']:>5.1f}%, "
              f"R:R={c['both']['median_rr']}, MFE={c['both']['median_mfe']}%")
        print(f"    NEITHER: n={c['neither']['n']:>3}, WR={c['neither']['win_rate']:>5.1f}%, "
              f"R:R={c['neither']['median_rr']}")
        print(f"    p(WR)={c['p_wr']}")

    # Stability (use best combo or first)
    best_idx = 0
    best_d = -999
    for i, c in enumerate(combo_results):
        if c["both"]["n"] >= 20:
            d = c["both"]["win_rate"] - c["baseline"]["win_rate"]
            if d > best_d:
                best_d = d
                best_idx = i

    bc = combo_results[best_idx]

    print("\n" + "=" * 70)
    print(f"STABILITY — gate: slope ≥ {bc['slope_label']}, vol ≥ {bc['vol_label']}")
    print("=" * 70)

    tod_st = stability_tod(df, bc["slope_thresh"], bc["vol_thresh"])
    tick_st = stability_ticker(df, bc["slope_thresh"], bc["vol_thresh"])

    print("\n  Time-of-Day:")
    for name in ["pre_market", "open", "mid", "close", "after_hours"]:
        v = tod_st.get(name)
        if not v:
            continue
        d = f"{v['delta_wr']:+.1f}pp" if v.get("delta_wr") is not None else "N/A"
        print(f"    {name:<14} n={v['n_all']:>4}/{v['n_pass']:>3}  "
              f"WR={v['wr_all']}%→{v.get('wr_pass','N/A')}%  Δ={d}")

    imp = sum(1 for v in tick_st.values() if v.get("delta_wr") and v["delta_wr"] > 0)
    tot = sum(1 for v in tick_st.values() if v.get("delta_wr") is not None)
    print(f"\n  Tickers (n≥10): {len(tick_st)}, improved: {imp}/{tot}")

    # Write report
    rpt_path = config.output_root.parent / "docs" / "Research" / "PHASE9_Results.md"
    write_report(df, slope_sw, vol_sw, combo_results, tod_st, tick_st, rpt_path)
    print(f"\n  Report: {rpt_path}")

    # Save JSON
    json_path = config.output_root / "phase9v2_results.json"
    all_res = {"slope": slope_sw, "vol": vol_sw, "combos": combo_results,
               "tod": tod_st, "tickers": tick_st}
    with open(json_path, "w") as f:
        json.dump(all_res, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("PHASE 9 v2 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
