"""
Phase 10B — Structural Liquidity Zone Conditioning
====================================================

Tests whether pressure and ignition behavior differ materially when events
occur near Previous Day High (PDH) or Previous Day Low (PDL).

Conditioning analysis only — no trading code modifications.

Steps:
  1. Compute daily OHLC from Databento tick data → PDH/PDL per symbol-date
  2. Label each ignition event: TOP_ZONE / BOTTOM_ZONE / MID
  3. Compare metrics (WR, PnL, MFE, MAE, R:R) by zone
  4. Pressure precursor analysis by zone
  5. Fade behavior by zone (Phase 8 logic)
  6. Stability checks (per-ticker, per-ToD)

Usage:
    python mrl/phase10b_structural_zones.py \
        --reports "\\\\Bob1\\c\\ai_project_hub\\store\\code\\IBKR_Algo_BOT_V2\\reports" \
        --databento "Z:\\AI_BOT_DATA\\databento_cache\\XNAS.ITCH\\trades"
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from replay.replay_engine import (
    ReplayConfig,
    IgnitionEvent,
    PressureProfile,
    load_raw_trades_around_event,
    compute_pressure_profile,
    find_databento_file,
)

logger = logging.getLogger(__name__)

BOOTSTRAP_N = 2000
PERMUTATION_N = 2000

ZONE_THRESHOLDS = [0.10, 0.25, 0.50]

TOD_BUCKETS = {
    "pre_market": (4.0, 9.5),
    "open":       (9.5, 10.5),
    "mid":        (10.5, 14.0),
    "close":      (14.0, 16.0),
    "after_hours": (16.0, 21.0),
}


# ============================================================
# STEP 1: COMPUTE DAILY OHLC FROM DATABENTO
# ============================================================

def compute_daily_ohlc(databento_root: Path) -> Dict[Tuple[str, str], dict]:
    """
    Scan all Databento .dbn.zst files, compute daily OHLC per (symbol, date).
    Returns: {(symbol, 'YYYY-MM-DD'): {'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}}
    """
    try:
        import databento as db
    except ImportError:
        print("[ERROR] databento not installed")
        return {}

    ohlc = {}
    files = sorted(databento_root.glob("*.dbn.zst"))
    print(f"  Scanning {len(files)} Databento files for daily OHLC...")

    for i, fpath in enumerate(files):
        # Extract symbol from filename: SYMBOL_STARTDATE_ENDDATE.dbn.zst
        parts = fpath.stem.replace(".dbn", "").split("_")
        if len(parts) < 2:
            continue
        symbol = parts[0]

        try:
            store = db.DBNStore.from_file(fpath)
            df = store.to_df()
        except Exception as e:
            logger.warning(f"Error reading {fpath}: {e}")
            continue

        if len(df) == 0:
            continue

        df = df.reset_index()

        # Ensure UTC timestamps
        if df["ts_recv"].dt.tz is None:
            df["ts_recv"] = df["ts_recv"].dt.tz_localize("UTC")
        else:
            df["ts_recv"] = df["ts_recv"].dt.tz_convert("UTC")

        # Convert to ET for date grouping
        df["ts_et"] = df["ts_recv"].dt.tz_convert("US/Eastern")
        df["date"] = df["ts_et"].dt.date.astype(str)

        # Filter to regular + extended hours (4:00 - 20:00 ET)
        hour = df["ts_et"].dt.hour
        mask = (hour >= 4) & (hour < 20)
        df = df[mask]

        if len(df) == 0:
            continue

        for date_str, group in df.groupby("date"):
            prices = group["price"]
            key = (symbol, date_str)
            ohlc[key] = {
                "open": float(prices.iloc[0]),
                "high": float(prices.max()),
                "low": float(prices.min()),
                "close": float(prices.iloc[-1]),
                "volume": int(group["size"].sum()),
                "n_trades": len(group),
            }

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(files)} files, {len(ohlc)} daily bars")

    print(f"  Computed {len(ohlc)} daily OHLC bars from {len(files)} files")
    return ohlc


def get_previous_trading_day(ohlc: dict, symbol: str, date_str: str) -> Optional[dict]:
    """Find the most recent previous trading day's OHLC for this symbol."""
    all_dates = sorted(d for (s, d) in ohlc.keys() if s == symbol)
    prev_dates = [d for d in all_dates if d < date_str]
    if not prev_dates:
        return None
    prev_date = prev_dates[-1]
    return ohlc.get((symbol, prev_date))


# ============================================================
# STEP 2: LOAD EVENTS WITH ATR AND ZONE LABELS
# ============================================================

def load_events_with_zones(reports_root: Path, ohlc: dict, zone_atr_threshold: float = 0.25):
    """
    Load ignition events from trade ledger, add PDH/PDL zone labels.
    Uses ATR from trade ledger. Uses PDH/PDL from Databento daily OHLC.
    """
    events_data = []
    no_atr = 0
    no_prev = 0
    labeled = 0

    report_dirs = sorted(reports_root.iterdir())
    for rdir in report_dirs:
        ledger = rdir / "trade_ledger.jsonl"
        if not ledger.exists():
            continue

        date_str = rdir.name  # e.g. "2026-01-30"

        with open(ledger) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") != "closed":
                    continue

                symbol = rec["symbol"]
                entry_price = float(rec.get("entry_price", 0))
                atr = float(rec.get("atr", 0))
                pnl = float(rec.get("pnl", 0))

                if atr <= 0 or entry_price <= 0:
                    no_atr += 1
                    continue

                # Get previous day's OHLC
                prev = get_previous_trading_day(ohlc, symbol, date_str)
                if prev is None:
                    no_prev += 1
                    # Still include event but mark as NO_ZONE
                    pdh = None
                    pdl = None
                else:
                    pdh = prev["high"]
                    pdl = prev["low"]

                # Compute zone
                if pdh is not None and pdl is not None:
                    dist_pdh = (pdh - entry_price) / atr
                    dist_pdl = (entry_price - pdl) / atr

                    # Label zones for multiple thresholds
                    zones = {}
                    for thresh in ZONE_THRESHOLDS:
                        if abs(dist_pdh) <= thresh:
                            zones[thresh] = "TOP_ZONE"
                        elif abs(dist_pdl) <= thresh:
                            zones[thresh] = "BOTTOM_ZONE"
                        else:
                            zones[thresh] = "MID"
                    labeled += 1
                else:
                    dist_pdh = None
                    dist_pdl = None
                    zones = {t: "NO_DATA" for t in ZONE_THRESHOLDS}

                # Parse entry time
                entry_time_str = rec.get("entry_time", "")
                try:
                    entry_ts = pd.Timestamp(entry_time_str)
                    if entry_ts.tzinfo is None:
                        entry_ts = entry_ts.tz_localize("UTC")
                except:
                    continue

                entry_et = entry_ts.tz_convert("US/Eastern")
                hour_decimal = entry_et.hour + entry_et.minute / 60.0
                tod_bucket = "unknown"
                for bname, (s, e) in TOD_BUCKETS.items():
                    if s <= hour_decimal < e:
                        tod_bucket = bname
                        break

                mfe = float(rec.get("max_gain_percent", 0))
                mae = abs(float(rec.get("max_drawdown_percent", 0)))

                if mae > 0 and mfe > 0:
                    rr = min(mfe / mae, 20.0)
                elif mae == 0 and mfe > 0:
                    rr = 20.0
                else:
                    rr = 0.0

                event_rec = {
                    "trade_id": rec.get("trade_id", ""),
                    "symbol": symbol,
                    "date": date_str,
                    "entry_time": entry_ts,
                    "entry_time_et": entry_et,
                    "hour_et": hour_decimal,
                    "tod_bucket": tod_bucket,
                    "entry_price": entry_price,
                    "atr": atr,
                    "pdh": pdh,
                    "pdl": pdl,
                    "dist_pdh": dist_pdh,
                    "dist_pdl": dist_pdl,
                    "pnl": pnl,
                    "pnl_percent": float(rec.get("pnl_percent", 0)),
                    "is_winner": 1 if pnl > 0 else 0,
                    "is_active": 1 if pnl != 0 else 0,
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                    "reward_risk": rr,
                    "hold_time_sec": int(rec.get("hold_time_seconds", 0)),
                    "entry_signal": rec.get("entry_signal", ""),
                    "volatility_regime": rec.get("volatility_regime", ""),
                    "exit_category": rec.get("primary_exit_category", rec.get("exit_reason", "")),
                }

                for thresh in ZONE_THRESHOLDS:
                    event_rec[f"zone_{thresh}"] = zones[thresh]

                events_data.append(event_rec)

    df = pd.DataFrame(events_data)
    print(f"  Total events: {len(df)}")
    print(f"  No ATR (skipped): {no_atr}")
    print(f"  No previous day data: {no_prev}")
    print(f"  Zone-labeled: {labeled}")

    return df


# ============================================================
# STEP 3: METRICS & STATISTICS
# ============================================================

def compute_metrics(df, label=""):
    """Compute metrics for a group."""
    n = len(df)
    if n == 0:
        return {"label": label, "n": 0, "win_rate": 0, "mean_pnl": 0,
                "median_pnl": 0, "total_pnl": 0, "median_mfe": 0, "mean_mfe": 0,
                "median_mae": 0, "mean_mae": 0, "median_rr": 0, "mean_rr": 0,
                "n_with_mae": 0, "median_rr_active": 0, "mean_hold_sec": 0,
                "unique_symbols": 0}

    winners = df[df["pnl"] > 0]
    has_mae = df[df["mae_pct"] > 0]

    return {
        "label": label,
        "n": n,
        "n_winners": len(winners),
        "win_rate": round(len(winners) / n * 100, 1),
        "mean_pnl": round(float(df["pnl"].mean()), 2),
        "median_pnl": round(float(df["pnl"].median()), 2),
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "median_mfe": round(float(df["mfe_pct"].median()), 3),
        "mean_mfe": round(float(df["mfe_pct"].mean()), 3),
        "median_mae": round(float(df["mae_pct"].median()), 3),
        "mean_mae": round(float(df["mae_pct"].mean()), 3),
        "median_rr": round(float(df["reward_risk"].median()), 3),
        "mean_rr": round(float(df["reward_risk"].mean()), 3),
        "n_with_mae": len(has_mae),
        "median_rr_active": round(float(has_mae["reward_risk"].median()), 3) if len(has_mae) > 0 else 0,
        "mean_hold_sec": round(float(df["hold_time_sec"].mean()), 1),
        "unique_symbols": int(df["symbol"].nunique()),
    }


def bootstrap_ci(arr, stat_fn=np.mean, n_boot=BOOTSTRAP_N, ci=95):
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
        if abs(stat_fn(perm[:na]) - stat_fn(perm[na:])) >= abs(obs):
            count += 1
    return round(count / n_perm, 4)


# ============================================================
# STEP 4: ZONE ANALYSIS
# ============================================================

def analyze_zones(df, zone_col, label=""):
    """Analyze metrics by zone for a given threshold column."""
    print(f"\n  --- Zone Analysis ({label}, column={zone_col}) ---")

    zones = ["TOP_ZONE", "BOTTOM_ZONE", "MID"]
    zone_data = {}

    for z in zones:
        zdf = df[df[zone_col] == z]
        m = compute_metrics(zdf, z)
        zone_data[z] = {
            "metrics": m,
            "df": zdf,
        }

        # Bootstrap CIs
        ci_wr = bootstrap_ci(zdf["is_winner"].values * 100) if len(zdf) >= 5 else (float("nan"), float("nan"))
        ci_mfe = bootstrap_ci(zdf["mfe_pct"].values, stat_fn=np.median) if len(zdf) >= 5 else (float("nan"), float("nan"))
        ci_rr = bootstrap_ci(zdf["reward_risk"].values, stat_fn=np.median) if len(zdf) >= 5 else (float("nan"), float("nan"))
        zone_data[z]["ci_wr"] = ci_wr
        zone_data[z]["ci_mfe"] = ci_mfe
        zone_data[z]["ci_rr"] = ci_rr

        print(f"    {z:<14} n={m['n']:>5}  WR={m['win_rate']:>5.1f}%  "
              f"PnL=${m['mean_pnl']:>+7.2f}  MFE={m['median_mfe']:.3f}%  "
              f"MAE={m['median_mae']:.3f}%  R:R={m['median_rr']:.3f}  "
              f"Syms={m['unique_symbols']}")
        print(f"                  CI(WR)=[{ci_wr[0]:.1f}%,{ci_wr[1]:.1f}%]  "
              f"CI(MFE)=[{ci_mfe[0]:.3f}%,{ci_mfe[1]:.3f}%]  "
              f"CI(R:R)=[{ci_rr[0]:.3f},{ci_rr[1]:.3f}]")

    # Compare TOP vs MID, BOTTOM vs MID
    mid_df = df[df[zone_col] == "MID"]
    comparisons = {}
    for z in ["TOP_ZONE", "BOTTOM_ZONE"]:
        zdf = df[df[zone_col] == z]
        if len(zdf) >= 5 and len(mid_df) >= 5:
            comp = {
                "p_wr": permutation_test(zdf["is_winner"].values, mid_df["is_winner"].values),
                "p_pnl": permutation_test(zdf["pnl"].values, mid_df["pnl"].values),
                "p_mfe": permutation_test(zdf["mfe_pct"].values, mid_df["mfe_pct"].values),
                "p_mae": permutation_test(zdf["mae_pct"].values, mid_df["mae_pct"].values),
                "p_rr": permutation_test(zdf["reward_risk"].values, mid_df["reward_risk"].values),
            }
        else:
            comp = {"p_wr": float("nan"), "p_pnl": float("nan"),
                    "p_mfe": float("nan"), "p_mae": float("nan"), "p_rr": float("nan")}
        comparisons[z] = comp

        sig = "***" if comp["p_wr"] < 0.01 else ("*" if comp["p_wr"] < 0.05 else "ns")
        print(f"\n    {z} vs MID: p(WR)={comp['p_wr']:.4f} {sig}  "
              f"p(PnL)={comp['p_pnl']:.4f}  p(MFE)={comp['p_mfe']:.4f}  "
              f"p(MAE)={comp['p_mae']:.4f}  p(R:R)={comp['p_rr']:.4f}")

    # Also compare COMBINED ZONES (TOP+BOTTOM) vs MID
    edge = df[df[zone_col].isin(["TOP_ZONE", "BOTTOM_ZONE"])]
    if len(edge) >= 5 and len(mid_df) >= 5:
        comp_edge = {
            "p_wr": permutation_test(edge["is_winner"].values, mid_df["is_winner"].values),
            "p_pnl": permutation_test(edge["pnl"].values, mid_df["pnl"].values),
            "p_mfe": permutation_test(edge["mfe_pct"].values, mid_df["mfe_pct"].values),
            "p_mae": permutation_test(edge["mae_pct"].values, mid_df["mae_pct"].values),
            "p_rr": permutation_test(edge["reward_risk"].values, mid_df["reward_risk"].values),
        }
        edge_m = compute_metrics(edge, "EDGE (TOP+BOT)")
        comparisons["EDGE_vs_MID"] = comp_edge

        sig = "***" if comp_edge["p_wr"] < 0.01 else ("*" if comp_edge["p_wr"] < 0.05 else "ns")
        print(f"\n    EDGE(TOP+BOT) n={edge_m['n']}  WR={edge_m['win_rate']:.1f}%  vs MID: "
              f"p(WR)={comp_edge['p_wr']:.4f} {sig}  "
              f"p(PnL)={comp_edge['p_pnl']:.4f}  p(MFE)={comp_edge['p_mfe']:.4f}")
        zone_data["EDGE"] = {"metrics": edge_m}

    return zone_data, comparisons


# ============================================================
# STEP 5: PRESSURE PRECURSOR ANALYSIS BY ZONE
# ============================================================

def pressure_by_zone(df_events, config, zone_col, zone_atr=0.25):
    """
    For events with valid pressure profiles, compare precursor characteristics by zone.
    """
    print(f"\n  --- Pressure Precursor Analysis ({zone_col}) ---")

    # Compute pressure profiles for zone-labeled events
    profiles = []

    zone_events = df_events[df_events[zone_col] != "NO_DATA"].copy()
    active_events = zone_events[zone_events["is_active"] == 1]

    print(f"  Processing {len(active_events)} active zone-labeled events for pressure...")

    matched = 0
    for i, (_, row) in enumerate(active_events.iterrows()):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(active_events)} (matched={matched})")

        evt = IgnitionEvent(
            trade_id=row["trade_id"],
            symbol=row["symbol"],
            entry_time=row["entry_time"],
            entry_price=row["entry_price"],
            entry_signal=row.get("entry_signal", ""),
            pnl=row["pnl"],
            pnl_percent=row["pnl_percent"],
            max_gain_percent=row["mfe_pct"],
            max_drawdown_percent=-row["mae_pct"],
            hold_time_seconds=row["hold_time_sec"],
            volatility_regime=row.get("volatility_regime", ""),
            momentum_score=0.0,
            momentum_state="",
            rvol=0.0,
            change_pct=0.0,
            spread_pct=0.0,
            primary_exit_category=row.get("exit_category", ""),
        )

        trades_df = load_raw_trades_around_event(
            config.databento_root, evt.symbol, evt.entry_time,
            config.pre_window_sec, config.post_window_sec,
        )
        if trades_df is None:
            continue

        profile = compute_pressure_profile(trades_df, evt, config, is_control=False)
        if profile is None:
            continue

        matched += 1
        profiles.append({
            "trade_id": row["trade_id"],
            "symbol": row["symbol"],
            "zone": row[zone_col],
            "is_winner": row["is_winner"],
            "pnl": row["pnl"],
            "mfe_pct": row["mfe_pct"],
            "mae_pct": row["mae_pct"],
            "peak_pressure_z": profile.peak_pressure_z_pre,
            "mean_pressure_z": profile.mean_pressure_z_pre,
            "pressure_direction": profile.pressure_direction_pre,
            "buildup_rate": profile.pressure_buildup_rate,
            "first_cross_sec": profile.first_threshold_cross_sec,
            "bars_above_threshold": profile.bars_above_threshold_pre,
            "pressure_consistency": profile.pressure_consistency,
            "volume_acceleration": profile.volume_acceleration,
        })

    print(f"  Matched {matched}/{len(active_events)} events with pressure profiles")

    if matched < 20:
        print(f"  [WARN] Too few pressure profiles for meaningful analysis")
        return pd.DataFrame(profiles), {}

    pdf = pd.DataFrame(profiles)

    # Compare pressure characteristics by zone
    results = {}
    pressure_cols = ["peak_pressure_z", "mean_pressure_z", "buildup_rate",
                     "volume_acceleration", "bars_above_threshold",
                     "pressure_consistency"]

    print(f"\n  Pressure metrics by zone:")
    print(f"  {'METRIC':<25} ", end="")
    for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
        n = len(pdf[pdf["zone"] == z])
        print(f"{'  ' + z + f'(n={n})':<22} ", end="")
    print()
    print(f"  {'-'*85}")

    for col in pressure_cols:
        print(f"  {col:<25} ", end="")
        zone_means = {}
        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
            zdf = pdf[pdf["zone"] == z]
            if len(zdf) > 0:
                val = zdf[col].mean()
                zone_means[z] = val
                print(f"  {val:>+10.4f}            ", end="")
            else:
                print(f"  {'N/A':>10}            ", end="")
        print()

        # Test each zone vs MID
        mid_vals = pdf[pdf["zone"] == "MID"][col].values
        for z in ["TOP_ZONE", "BOTTOM_ZONE"]:
            z_vals = pdf[pdf["zone"] == z][col].values
            if len(z_vals) >= 5 and len(mid_vals) >= 5:
                p = permutation_test(z_vals, mid_vals)
                sig = " *" if p < 0.05 else ""
                results[f"{col}_{z}_vs_MID"] = p
                if p < 0.10:
                    print(f"      {z} vs MID: p={p:.4f}{sig}")

    # Precursor frequency: how often is peak_pressure_z > threshold?
    print(f"\n  Precursor frequency (peak_pressure_z > 1.5):")
    for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
        zdf = pdf[pdf["zone"] == z]
        if len(zdf) == 0:
            continue
        has_precursor = (zdf["peak_pressure_z"] > 1.5).sum()
        pct = has_precursor / len(zdf) * 100
        print(f"    {z}: {has_precursor}/{len(zdf)} ({pct:.1f}%)")

    # Lead time (first_cross_sec) by zone
    print(f"\n  Lead time (first_cross_sec, precursor events only):")
    for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
        zdf = pdf[(pdf["zone"] == z) & (pdf["first_cross_sec"].notna()) & (pdf["first_cross_sec"] > 0)]
        if len(zdf) == 0:
            print(f"    {z}: no precursor events")
            continue
        mean_lt = zdf["first_cross_sec"].mean()
        median_lt = zdf["first_cross_sec"].median()
        print(f"    {z}: n={len(zdf)}  mean={mean_lt:.1f}s  median={median_lt:.1f}s")

    return pdf, results


# ============================================================
# STEP 6: FADE BEHAVIOR BY ZONE
# ============================================================

def fade_by_zone(pdf, zone_col):
    """
    Among profiled events, check if fade behavior (high pressure → reversal)
    differs by zone. Uses Phase 8 criteria: pressure_z >= 2.0.
    """
    print(f"\n  --- Fade Behavior by Zone ({zone_col}) ---")

    if len(pdf) == 0:
        print("  No pressure profiles available")
        return {}

    # Phase 8 fade criteria: peak_pressure_z >= 2.0
    fade_events = pdf[pdf["peak_pressure_z"] >= 2.0].copy()
    print(f"  High-pressure events (peak_z >= 2.0): {len(fade_events)}")

    if len(fade_events) < 10:
        print("  Too few fade events for zone analysis")
        return {}

    results = {}
    for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
        zdf = fade_events[fade_events["zone"] == z]
        if len(zdf) == 0:
            continue
        wr = zdf["is_winner"].mean() * 100
        mean_pnl = zdf["pnl"].mean()
        med_mfe = zdf["mfe_pct"].median()
        med_mae = zdf["mae_pct"].median()
        rr = zdf["reward_risk"].median() if "reward_risk" in zdf.columns else 0

        results[z] = {
            "n": len(zdf), "win_rate": wr, "mean_pnl": mean_pnl,
            "median_mfe": med_mfe, "median_mae": med_mae,
        }

        print(f"    {z:<14} n={len(zdf):>4}  WR={wr:>5.1f}%  "
              f"PnL=${mean_pnl:>+7.2f}  MFE={med_mfe:.3f}%  MAE={med_mae:.3f}%")

    # Test zone vs MID for fade events
    mid_fade = fade_events[fade_events["zone"] == "MID"]
    for z in ["TOP_ZONE", "BOTTOM_ZONE"]:
        z_fade = fade_events[fade_events["zone"] == z]
        if len(z_fade) >= 5 and len(mid_fade) >= 5:
            p = permutation_test(z_fade["is_winner"].values, mid_fade["is_winner"].values)
            sig = " *" if p < 0.05 else ""
            print(f"    {z} vs MID (fade WR): p={p:.4f}{sig}")
            results[f"p_{z}_vs_MID"] = p

    return results


# ============================================================
# STEP 7: STABILITY CHECKS
# ============================================================

def stability_ticker(df, zone_col):
    """Per-ticker zone analysis."""
    print(f"\n  --- Stability: Per-Ticker ({zone_col}) ---")

    syms = df["symbol"].value_counts()
    syms = syms[syms >= 5].index.tolist()
    print(f"  Tickers with n >= 5 active zone-labeled: {len(syms)}")

    print(f"\n  {'SYM':<8} {'N':>4} {'N_TOP':>6} {'N_BOT':>6} {'N_MID':>6} "
          f"{'WR_TOP':>7} {'WR_BOT':>7} {'WR_MID':>7}")
    print(f"  {'-'*70}")

    results = {}
    for sym in sorted(syms):
        sdf = df[df["symbol"] == sym]
        n = len(sdf)
        n_top = len(sdf[sdf[zone_col] == "TOP_ZONE"])
        n_bot = len(sdf[sdf[zone_col] == "BOTTOM_ZONE"])
        n_mid = len(sdf[sdf[zone_col] == "MID"])

        wr_top = sdf[sdf[zone_col] == "TOP_ZONE"]["is_winner"].mean() * 100 if n_top > 0 else float("nan")
        wr_bot = sdf[sdf[zone_col] == "BOTTOM_ZONE"]["is_winner"].mean() * 100 if n_bot > 0 else float("nan")
        wr_mid = sdf[sdf[zone_col] == "MID"]["is_winner"].mean() * 100 if n_mid > 0 else float("nan")

        results[sym] = {"n": n, "n_top": n_top, "n_bot": n_bot, "n_mid": n_mid,
                        "wr_top": wr_top, "wr_bot": wr_bot, "wr_mid": wr_mid}

        fmt_wr = lambda x: f"{x:>6.1f}%" if not np.isnan(x) else "    N/A"
        print(f"  {sym:<8} {n:>4} {n_top:>6} {n_bot:>6} {n_mid:>6} "
              f"{fmt_wr(wr_top)} {fmt_wr(wr_bot)} {fmt_wr(wr_mid)}")

    return results


def stability_tod(df, zone_col):
    """Time-of-day zone analysis."""
    print(f"\n  --- Stability: Time-of-Day ({zone_col}) ---")

    print(f"\n  {'BUCKET':<14} {'N':>5} {'N_TOP':>6} {'N_BOT':>6} {'N_MID':>6} "
          f"{'WR_TOP':>7} {'WR_BOT':>7} {'WR_MID':>7}")
    print(f"  {'-'*75}")

    results = {}
    for bname in ["pre_market", "open", "mid", "close", "after_hours"]:
        bdf = df[df["tod_bucket"] == bname]
        if len(bdf) == 0:
            continue
        n = len(bdf)
        n_top = len(bdf[bdf[zone_col] == "TOP_ZONE"])
        n_bot = len(bdf[bdf[zone_col] == "BOTTOM_ZONE"])
        n_mid = len(bdf[bdf[zone_col] == "MID"])

        wr_top = bdf[bdf[zone_col] == "TOP_ZONE"]["is_winner"].mean() * 100 if n_top > 0 else float("nan")
        wr_bot = bdf[bdf[zone_col] == "BOTTOM_ZONE"]["is_winner"].mean() * 100 if n_bot > 0 else float("nan")
        wr_mid = bdf[bdf[zone_col] == "MID"]["is_winner"].mean() * 100 if n_mid > 0 else float("nan")

        results[bname] = {"n": n, "n_top": n_top, "n_bot": n_bot, "n_mid": n_mid,
                          "wr_top": wr_top, "wr_bot": wr_bot, "wr_mid": wr_mid}

        fmt_wr = lambda x: f"{x:>6.1f}%" if not np.isnan(x) else "    N/A"
        print(f"  {bname:<14} {n:>5} {n_top:>6} {n_bot:>6} {n_mid:>6} "
              f"{fmt_wr(wr_top)} {fmt_wr(wr_bot)} {fmt_wr(wr_mid)}")

    return results


# ============================================================
# STEP 8: REPORT GENERATION
# ============================================================

def write_report(df_all, df_active, zone_results, pressure_df, pressure_results,
                 fade_results, tick_stab, tod_stab, path):
    """Write PHASE10B_StructuralZones.md."""
    L = []
    L.append("# Phase 10B — Structural Liquidity Zone Conditioning")
    L.append("")
    L.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    L.append(f"**Total events loaded:** {len(df_all)}")
    L.append(f"**Active events (PnL ≠ 0):** {len(df_active)}")
    n_zoned = len(df_active[df_active["zone_0.25"] != "NO_DATA"])
    L.append(f"**Zone-labeled (have PDH/PDL):** {n_zoned}")
    L.append(f"**Pressure-profiled:** {len(pressure_df) if pressure_df is not None else 0}")
    L.append("")

    # Power check
    L.append("## Power Check")
    L.append("")
    L.append("Validation gate requires n ≥ 500 per condition.")
    for thresh in ZONE_THRESHOLDS:
        zcol = f"zone_{thresh}"
        zoned = df_active[df_active[zcol] != "NO_DATA"]
        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
            n = len(zoned[zoned[zcol] == z])
            status = "✅ POWERED" if n >= 500 else (f"⚠️ UNDERPOWERED (n={n})" if n >= 50 else f"❌ INSUFFICIENT (n={n})")
            L.append(f"- ATR {thresh}: {z} → n={n} {status}")
    L.append("")
    L.append("---")
    L.append("")

    # Zone results for each threshold
    for thresh in ZONE_THRESHOLDS:
        zcol = f"zone_{thresh}"
        if zcol not in zone_results:
            continue
        zr, comps = zone_results[zcol]

        L.append(f"## Zone Analysis — ATR Threshold {thresh}")
        L.append("")
        L.append("| Zone | N | Win Rate | Mean PnL | Med MFE | Med MAE | Med R:R | CI95 WR | CI95 MFE | CI95 R:R | Symbols |")
        L.append("|------|---|----------|----------|---------|---------|---------|---------|----------|----------|---------|")

        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
            if z not in zr:
                continue
            m = zr[z]["metrics"]
            ci_wr = zr[z].get("ci_wr", (float("nan"), float("nan")))
            ci_mfe = zr[z].get("ci_mfe", (float("nan"), float("nan")))
            ci_rr = zr[z].get("ci_rr", (float("nan"), float("nan")))
            L.append(f"| {z} | {m['n']} | {m['win_rate']:.1f}% | ${m['mean_pnl']:+.2f} | "
                     f"{m['median_mfe']:.3f}% | {m['median_mae']:.3f}% | {m['median_rr']:.3f} | "
                     f"[{ci_wr[0]:.1f}%,{ci_wr[1]:.1f}%] | [{ci_mfe[0]:.3f}%,{ci_mfe[1]:.3f}%] | "
                     f"[{ci_rr[0]:.3f},{ci_rr[1]:.3f}] | {m['unique_symbols']} |")

        if "EDGE" in zr:
            m = zr["EDGE"]["metrics"]
            L.append(f"| **EDGE (TOP+BOT)** | {m['n']} | {m['win_rate']:.1f}% | ${m['mean_pnl']:+.2f} | "
                     f"{m['median_mfe']:.3f}% | {m['median_mae']:.3f}% | {m['median_rr']:.3f} | — | — | — | {m['unique_symbols']} |")

        L.append("")
        L.append("**Permutation tests vs MID:**")
        for comp_name, comp in comps.items():
            sig = " ★" if comp["p_wr"] < 0.05 else ""
            L.append(f"- {comp_name}: p(WR)={comp['p_wr']:.4f}{sig}, "
                     f"p(PnL)={comp['p_pnl']:.4f}, p(MFE)={comp['p_mfe']:.4f}, "
                     f"p(MAE)={comp['p_mae']:.4f}, p(R:R)={comp['p_rr']:.4f}")
        L.append("")
        L.append("---")
        L.append("")

    # Pressure analysis
    if pressure_df is not None and len(pressure_df) > 0:
        L.append("## Pressure Precursor Analysis by Zone")
        L.append("")
        L.append(f"Events with pressure profiles: {len(pressure_df)}")
        L.append("")

        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
            zdf = pressure_df[pressure_df["zone"] == z]
            if len(zdf) == 0:
                continue
            L.append(f"**{z} (n={len(zdf)}):**")
            L.append(f"- Mean peak_pressure_z: {zdf['peak_pressure_z'].mean():.3f}")
            L.append(f"- Mean buildup_rate: {zdf['buildup_rate'].mean():.4f}")
            L.append(f"- Mean volume_acceleration: {zdf['volume_acceleration'].mean():.2f}")
            has_pre = (zdf["peak_pressure_z"] > 1.5).sum()
            L.append(f"- Precursor frequency (peak > 1.5): {has_pre}/{len(zdf)} ({has_pre/len(zdf)*100:.1f}%)")
            pre_lt = zdf[(zdf["first_cross_sec"].notna()) & (zdf["first_cross_sec"] > 0)]
            if len(pre_lt) > 0:
                L.append(f"- Lead time: mean={pre_lt['first_cross_sec'].mean():.1f}s, median={pre_lt['first_cross_sec'].median():.1f}s")
            L.append("")

        # Significant pressure differences
        sig_results = {k: v for k, v in pressure_results.items() if v < 0.10}
        if sig_results:
            L.append("**Significant/trending pressure differences (p < 0.10):**")
            for k, v in sorted(sig_results.items(), key=lambda x: x[1]):
                sig = " ★" if v < 0.05 else ""
                L.append(f"- {k}: p={v:.4f}{sig}")
            L.append("")

        L.append("---")
        L.append("")

    # Fade analysis
    if fade_results:
        L.append("## Fade Behavior by Zone (peak_z >= 2.0)")
        L.append("")
        L.append("| Zone | N | Win Rate | Mean PnL | Med MFE | Med MAE |")
        L.append("|------|---|----------|----------|---------|---------|")
        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
            if z not in fade_results:
                continue
            r = fade_results[z]
            L.append(f"| {z} | {r['n']} | {r['win_rate']:.1f}% | ${r['mean_pnl']:+.2f} | "
                     f"{r['median_mfe']:.3f}% | {r['median_mae']:.3f}% |")
        L.append("")
        for k, v in fade_results.items():
            if k.startswith("p_"):
                sig = " ★" if v < 0.05 else ""
                L.append(f"- {k}: p={v:.4f}{sig}")
        L.append("")
        L.append("---")
        L.append("")

    # Stability
    L.append("## Stability: Time-of-Day")
    L.append("")
    if tod_stab:
        L.append("| Bucket | N | N Top | N Bot | N Mid | WR Top | WR Bot | WR Mid |")
        L.append("|--------|---|-------|-------|-------|--------|--------|--------|")
        for bname in ["pre_market", "open", "mid", "close", "after_hours"]:
            v = tod_stab.get(bname)
            if not v: continue
            fmt = lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A"
            L.append(f"| {bname} | {v['n']} | {v['n_top']} | {v['n_bot']} | {v['n_mid']} | "
                     f"{fmt(v['wr_top'])} | {fmt(v['wr_bot'])} | {fmt(v['wr_mid'])} |")
        L.append("")

    L.append("## Stability: Per-Ticker")
    L.append("")
    if tick_stab:
        L.append("| Sym | N | N Top | N Bot | N Mid | WR Top | WR Bot | WR Mid |")
        L.append("|-----|---|-------|-------|-------|--------|--------|--------|")
        for sym in sorted(tick_stab.keys()):
            v = tick_stab[sym]
            fmt = lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A"
            L.append(f"| {sym} | {v['n']} | {v['n_top']} | {v['n_bot']} | {v['n_mid']} | "
                     f"{fmt(v['wr_top'])} | {fmt(v['wr_bot'])} | {fmt(v['wr_mid'])} |")
        L.append("")

    # Conclusion
    L.append("---")
    L.append("")
    L.append("## Conclusion")
    L.append("")
    L.append("*Auto-generated. Interpret based on p-values, sample sizes, and stability.*")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Report written: {path}")
    return "\n".join(L)


# ============================================================
# MAIN
# ============================================================

def run_phase10b(config=None, symbols=None, max_events=None):
    if config is None:
        config = ReplayConfig()

    print("=" * 80)
    print("PHASE 10B — Structural Liquidity Zone Conditioning")
    print("=" * 80)

    # Step 1: Compute daily OHLC from Databento
    print("\n[STEP 1] Computing daily OHLC from Databento tick data...")
    ohlc = compute_daily_ohlc(config.databento_root)

    # Step 2: Load events with zone labels
    print("\n[STEP 2] Loading events and labeling zones...")
    df_all = load_events_with_zones(config.reports_root, ohlc)

    if len(df_all) == 0:
        print("[ERROR] No events loaded")
        return

    # Filter to active trades
    df_active = df_all[df_all["is_active"] == 1].copy()
    n_flat = len(df_all) - len(df_active)
    print(f"\n  Active trades: {len(df_active)} (excluded {n_flat} flat)")

    # Save raw event data
    raw_path = config.output_root / "phase10b_event_data.csv"
    df_active.to_csv(raw_path, index=False)
    print(f"  Saved: {raw_path}")

    # Zone distribution
    print(f"\n  Zone distribution (ATR 0.25):")
    for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID", "NO_DATA"]:
        n = len(df_active[df_active["zone_0.25"] == z])
        print(f"    {z}: {n} ({n/len(df_active)*100:.1f}%)")

    # Step 3: Zone analysis for each threshold
    print("\n[STEP 3] Zone analysis...")
    zone_results = {}
    for thresh in ZONE_THRESHOLDS:
        zcol = f"zone_{thresh}"
        print(f"\n{'='*80}")
        print(f"ZONE ANALYSIS — ATR Threshold = {thresh}")
        print(f"{'='*80}")

        # Filter to zone-labeled events
        zoned = df_active[df_active[zcol] != "NO_DATA"]
        if len(zoned) < 20:
            print(f"  [SKIP] Only {len(zoned)} zone-labeled events")
            continue

        zr, comps = analyze_zones(zoned, zcol, f"ATR={thresh}")
        zone_results[zcol] = (zr, comps)

    # Step 4: Pressure analysis by zone (use 0.25 ATR threshold)
    print(f"\n{'='*80}")
    print("PRESSURE PRECURSOR ANALYSIS BY ZONE")
    print(f"{'='*80}")

    zoned_active = df_active[df_active["zone_0.25"] != "NO_DATA"]
    pressure_df, pressure_results = pressure_by_zone(zoned_active, config, "zone_0.25")

    # Step 5: Fade behavior by zone
    print(f"\n{'='*80}")
    print("FADE BEHAVIOR BY ZONE")
    print(f"{'='*80}")
    fade_results = fade_by_zone(pressure_df, "zone")

    # Step 6: Stability
    print(f"\n{'='*80}")
    print("STABILITY CHECKS")
    print(f"{'='*80}")
    tick_stab = stability_ticker(zoned_active, "zone_0.25")
    tod_stab = stability_tod(zoned_active, "zone_0.25")

    # Step 7: Write report
    report_path = config.output_root.parent / "docs" / "Research" / "PHASE10B_StructuralZones.md"
    write_report(df_all, df_active, zone_results, pressure_df, pressure_results,
                 fade_results, tick_stab, tod_stab, report_path)

    # Save JSON
    json_path = config.output_root / "phase10b_results.json"
    json_data = {
        "n_total": len(df_all),
        "n_active": len(df_active),
        "n_ohlc_bars": len(ohlc),
        "zone_distribution": {
            thresh: {z: int(len(df_active[df_active[f"zone_{thresh}"] == z]))
                     for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID", "NO_DATA"]}
            for thresh in ZONE_THRESHOLDS
        },
        "pressure_profiled": len(pressure_df) if pressure_df is not None else 0,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    print("\n" + "=" * 80)
    print("PHASE 10B COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 10B — Structural Zones")
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
    run_phase10b(config=config, symbols=args.symbols, max_events=args.max_events)
