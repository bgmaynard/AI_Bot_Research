"""
Phase 10B.1 — Structural Liquidity Zone Conditioning (FIXED)
=============================================================

Root cause of 10B failure: Databento only had data for days Morpheus traded.
78% of symbols are single-day runners → no previous day in cache.

Fix: Use yfinance for daily OHLC → PDH/PDL available for ~100% of events.
Backfill: search up to 5 trading days back if exact prev-day missing.

Usage:
    python mrl/phase10b1_zones_fixed.py \
        --reports "\\\\Bob1\\c\\ai_project_hub\\store\\code\\IBKR_Algo_BOT_V2\\reports" \
        --databento "Z:\\AI_BOT_DATA\\databento_cache\\XNAS.ITCH\\trades"
"""

import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import defaultdict

from replay.replay_engine import (
    ReplayConfig,
    IgnitionEvent,
    load_raw_trades_around_event,
    compute_pressure_profile,
)

logger = logging.getLogger(__name__)

BOOTSTRAP_N = 2000
PERMUTATION_N = 2000

ZONE_THRESHOLDS = [0.10, 0.25, 0.50, 1.0]

TOD_BUCKETS = {
    "pre_market": (4.0, 9.5),
    "open":       (9.5, 10.5),
    "mid":        (10.5, 14.0),
    "close":      (14.0, 16.0),
    "after_hours": (16.0, 21.0),
}


# ============================================================
# STEP 1: DOWNLOAD DAILY OHLC VIA YFINANCE
# ============================================================

def download_daily_ohlc(symbols: list, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLC for all symbols via yfinance.
    Returns dict: {symbol: DataFrame with Date index, OHLC columns}
    """
    print(f"  Downloading daily OHLC for {len(symbols)} symbols ({start_date} to {end_date})...")

    # Batch download (yfinance handles batching)
    # Download in chunks to avoid timeouts
    chunk_size = 50
    all_data = {}

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        print(f"    Batch {i//chunk_size + 1}: {len(chunk)} symbols...")

        try:
            data = yf.download(chunk, start=start_date, end=end_date,
                               progress=False, threads=True)

            if len(chunk) == 1:
                # Single symbol: no multi-level columns
                sym = chunk[0]
                if len(data) > 0:
                    ohlc = pd.DataFrame({
                        "Open": data["Open"].values if "Open" in data.columns else data[("Open", sym)].values,
                        "High": data["High"].values if "High" in data.columns else data[("High", sym)].values,
                        "Low": data["Low"].values if "Low" in data.columns else data[("Low", sym)].values,
                        "Close": data["Close"].values if "Close" in data.columns else data[("Close", sym)].values,
                        "Volume": data["Volume"].values if "Volume" in data.columns else data[("Volume", sym)].values,
                    }, index=data.index)
                    all_data[sym] = ohlc
            else:
                # Multi-symbol: multi-level columns
                for sym in chunk:
                    try:
                        sym_data = pd.DataFrame({
                            "Open": data[("Open", sym)],
                            "High": data[("High", sym)],
                            "Low": data[("Low", sym)],
                            "Close": data[("Close", sym)],
                            "Volume": data[("Volume", sym)],
                        })
                        sym_data = sym_data.dropna(subset=["High", "Low"])
                        if len(sym_data) > 0:
                            all_data[sym] = sym_data
                    except (KeyError, TypeError):
                        pass

        except Exception as e:
            print(f"    [WARN] Batch failed: {e}")
            # Retry individually
            for sym in chunk:
                try:
                    d = yf.download(sym, start=start_date, end=end_date, progress=False)
                    if len(d) > 0:
                        all_data[sym] = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                except:
                    pass

    print(f"  Downloaded daily OHLC for {len(all_data)}/{len(symbols)} symbols")

    # Report coverage
    total_bars = sum(len(df) for df in all_data.values())
    print(f"  Total daily bars: {total_bars}")

    return all_data


def get_pdh_pdl(daily_data: Dict[str, pd.DataFrame], symbol: str, trade_date: str,
                max_backfill: int = 5) -> Tuple[Optional[float], Optional[float], int]:
    """
    Get Previous Day High/Low for a symbol on a given trade date.
    Searches back up to max_backfill trading days.
    Returns: (pdh, pdl, backfill_steps) or (None, None, -1) if not found.
    """
    if symbol not in daily_data:
        return None, None, -1

    df = daily_data[symbol]
    # Convert trade_date to comparable format
    trade_ts = pd.Timestamp(trade_date)

    # Find all dates strictly before trade_date
    prior_dates = df.index[df.index < trade_ts]
    if len(prior_dates) == 0:
        return None, None, -1

    # Search backwards up to max_backfill days
    for step in range(min(max_backfill, len(prior_dates))):
        prev_date = prior_dates[-(step + 1)]
        row = df.loc[prev_date]
        pdh = float(row["High"])
        pdl = float(row["Low"])
        if pdh > 0 and pdl > 0:
            return pdh, pdl, step
        # Also check if the index is timezone aware
    return None, None, -1


# ============================================================
# STEP 2: LOAD EVENTS WITH ZONE LABELS
# ============================================================

def load_events_with_zones(reports_root: Path, daily_data: dict, max_backfill: int = 5):
    """Load ignition events, add PDH/PDL zone labels."""
    events_data = []
    no_atr = 0
    no_prev = 0
    labeled = 0
    backfill_counts = defaultdict(int)

    report_dirs = sorted(reports_root.iterdir())
    for rdir in report_dirs:
        ledger = rdir / "trade_ledger.jsonl"
        if not ledger.exists():
            continue

        date_str = rdir.name

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

                # Get PDH/PDL from yfinance daily data
                pdh, pdl, backfill_steps = get_pdh_pdl(daily_data, symbol, date_str, max_backfill)

                if pdh is None or pdl is None:
                    no_prev += 1
                    dist_pdh = None
                    dist_pdl = None
                    zones = {t: "NO_DATA" for t in ZONE_THRESHOLDS}
                else:
                    backfill_counts[backfill_steps] += 1
                    dist_pdh = (pdh - entry_price) / atr
                    dist_pdl = (entry_price - pdl) / atr
                    labeled += 1

                    zones = {}
                    for thresh in ZONE_THRESHOLDS:
                        if abs(dist_pdh) <= thresh:
                            zones[thresh] = "TOP_ZONE"
                        elif abs(dist_pdl) <= thresh:
                            zones[thresh] = "BOTTOM_ZONE"
                        else:
                            zones[thresh] = "MID"

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
                    "backfill_steps": backfill_steps,
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
                    "exit_category": rec.get("exit_reason", ""),
                }

                for thresh in ZONE_THRESHOLDS:
                    event_rec[f"zone_{thresh}"] = zones[thresh]

                events_data.append(event_rec)

    df = pd.DataFrame(events_data)
    print(f"  Total events: {len(df)}")
    print(f"  No ATR: {no_atr}")
    print(f"  No prev day (yfinance): {no_prev}")
    print(f"  Zone-labeled: {labeled} ({labeled/len(df)*100:.1f}%)")
    print(f"  Backfill distribution:")
    for step in sorted(backfill_counts.keys()):
        print(f"    {step} days back: {backfill_counts[step]}")

    return df


# ============================================================
# STEP 3: STATISTICS
# ============================================================

def compute_metrics(df, label=""):
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
        "label": label, "n": n,
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
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
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


def print_metrics(label, m, base_wr=None):
    d = ""
    if base_wr is not None and m["n"] > 0:
        d = f" (Δ{m['win_rate']-base_wr:+.1f}pp)"
    print(f"    {label:<18} n={m['n']:>5}  WR={m['win_rate']:>5.1f}%{d:<12} "
          f"PnL=${m['mean_pnl']:>+7.2f}  MFE={m['median_mfe']:.3f}%  "
          f"MAE={m['median_mae']:.3f}%  R:R={m['median_rr']:.3f}  Syms={m['unique_symbols']}")


# ============================================================
# STEP 4: ZONE ANALYSIS
# ============================================================

def analyze_zones(df, zone_col, label=""):
    print(f"\n  --- Zone Analysis ({label}, column={zone_col}) ---")

    zones = ["TOP_ZONE", "BOTTOM_ZONE", "MID"]
    zone_data = {}

    for z in zones:
        zdf = df[df[zone_col] == z]
        m = compute_metrics(zdf, z)
        ci_wr = bootstrap_ci(zdf["is_winner"].values * 100) if len(zdf) >= 5 else (float("nan"), float("nan"))
        ci_mfe = bootstrap_ci(zdf["mfe_pct"].values, stat_fn=np.median) if len(zdf) >= 5 else (float("nan"), float("nan"))
        ci_rr = bootstrap_ci(zdf["reward_risk"].values, stat_fn=np.median) if len(zdf) >= 5 else (float("nan"), float("nan"))
        zone_data[z] = {"metrics": m, "ci_wr": ci_wr, "ci_mfe": ci_mfe, "ci_rr": ci_rr, "df": zdf}

        print_metrics(z, m)
        print(f"                      CI(WR)=[{ci_wr[0]:.1f}%,{ci_wr[1]:.1f}%]  "
              f"CI(MFE)=[{ci_mfe[0]:.3f}%,{ci_mfe[1]:.3f}%]  "
              f"CI(R:R)=[{ci_rr[0]:.3f},{ci_rr[1]:.3f}]")

    # Compare zone vs MID
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
            comp = {k: float("nan") for k in ["p_wr", "p_pnl", "p_mfe", "p_mae", "p_rr"]}
        comparisons[z] = comp
        sig = "***" if comp["p_wr"] < 0.01 else ("*" if comp["p_wr"] < 0.05 else "ns")
        print(f"\n    {z} vs MID: p(WR)={comp['p_wr']:.4f} {sig}  "
              f"p(PnL)={comp['p_pnl']:.4f}  p(MFE)={comp['p_mfe']:.4f}  "
              f"p(MAE)={comp['p_mae']:.4f}  p(R:R)={comp['p_rr']:.4f}")

    # EDGE (TOP+BOTTOM) vs MID
    edge = df[df[zone_col].isin(["TOP_ZONE", "BOTTOM_ZONE"])]
    if len(edge) >= 5 and len(mid_df) >= 5:
        edge_m = compute_metrics(edge, "EDGE")
        comp_edge = {
            "p_wr": permutation_test(edge["is_winner"].values, mid_df["is_winner"].values),
            "p_pnl": permutation_test(edge["pnl"].values, mid_df["pnl"].values),
            "p_mfe": permutation_test(edge["mfe_pct"].values, mid_df["mfe_pct"].values),
            "p_mae": permutation_test(edge["mae_pct"].values, mid_df["mae_pct"].values),
            "p_rr": permutation_test(edge["reward_risk"].values, mid_df["reward_risk"].values),
        }
        comparisons["EDGE_vs_MID"] = comp_edge
        zone_data["EDGE"] = {"metrics": edge_m}
        sig = "***" if comp_edge["p_wr"] < 0.01 else ("*" if comp_edge["p_wr"] < 0.05 else "ns")
        print(f"\n    EDGE(TOP+BOT) n={edge_m['n']}  WR={edge_m['win_rate']:.1f}%  vs MID: "
              f"p(WR)={comp_edge['p_wr']:.4f} {sig}  p(PnL)={comp_edge['p_pnl']:.4f}  "
              f"p(MFE)={comp_edge['p_mfe']:.4f}")

    return zone_data, comparisons


# ============================================================
# STEP 5: PRESSURE ANALYSIS BY ZONE
# ============================================================

def pressure_by_zone(df_events, config, zone_col):
    print(f"\n  --- Pressure Precursor Analysis ({zone_col}) ---")

    profiles = []
    active_zoned = df_events[(df_events[zone_col] != "NO_DATA") & (df_events["is_active"] == 1)]
    print(f"  Processing {len(active_zoned)} active zone-labeled events...")

    matched = 0
    for i, (_, row) in enumerate(active_zoned.iterrows()):
        if i % 200 == 0 and i > 0:
            print(f"    {i}/{len(active_zoned)} (matched={matched})")

        evt = IgnitionEvent(
            trade_id=row["trade_id"], symbol=row["symbol"],
            entry_time=row["entry_time"], entry_price=row["entry_price"],
            entry_signal=row.get("entry_signal", ""), pnl=row["pnl"],
            pnl_percent=row["pnl_percent"], max_gain_percent=row["mfe_pct"],
            max_drawdown_percent=-row["mae_pct"],
            hold_time_seconds=row["hold_time_sec"],
            volatility_regime=row.get("volatility_regime", ""),
            momentum_score=0.0, momentum_state="", rvol=0.0,
            change_pct=0.0, spread_pct=0.0,
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
            "trade_id": row["trade_id"], "symbol": row["symbol"],
            "zone": row[zone_col], "is_winner": row["is_winner"],
            "pnl": row["pnl"], "mfe_pct": row["mfe_pct"], "mae_pct": row["mae_pct"],
            "peak_pressure_z": profile.peak_pressure_z_pre,
            "mean_pressure_z": profile.mean_pressure_z_pre,
            "buildup_rate": profile.pressure_buildup_rate,
            "volume_acceleration": profile.volume_acceleration,
            "bars_above_threshold": profile.bars_above_threshold_pre,
            "pressure_consistency": profile.pressure_consistency,
            "first_cross_sec": profile.first_threshold_cross_sec,
        })

    print(f"  Matched {matched}/{len(active_zoned)} with pressure profiles")

    if matched < 20:
        print(f"  [WARN] Too few profiles")
        return pd.DataFrame(profiles), {}

    pdf = pd.DataFrame(profiles)

    pressure_cols = ["peak_pressure_z", "mean_pressure_z", "buildup_rate",
                     "volume_acceleration", "bars_above_threshold", "pressure_consistency"]

    print(f"\n  Pressure metrics by zone:")
    for col in pressure_cols:
        parts = []
        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
            zdf = pdf[pdf["zone"] == z]
            if len(zdf) > 0:
                parts.append(f"{z}={zdf[col].mean():+.4f}(n={len(zdf)})")
        print(f"    {col}: {', '.join(parts)}")

    # Permutation tests
    results = {}
    mid_pdf = pdf[pdf["zone"] == "MID"]
    for col in pressure_cols:
        for z in ["TOP_ZONE", "BOTTOM_ZONE"]:
            z_vals = pdf[pdf["zone"] == z][col].values
            mid_vals = mid_pdf[col].values
            if len(z_vals) >= 5 and len(mid_vals) >= 5:
                p = permutation_test(z_vals, mid_vals)
                results[f"{col}_{z}_vs_MID"] = p
                if p < 0.10:
                    sig = " *" if p < 0.05 else ""
                    print(f"      {z} vs MID ({col}): p={p:.4f}{sig}")

    # Precursor frequency
    print(f"\n  Precursor frequency (peak_z > 1.5):")
    for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
        zdf = pdf[pdf["zone"] == z]
        if len(zdf) == 0: continue
        has = (zdf["peak_pressure_z"] > 1.5).sum()
        print(f"    {z}: {has}/{len(zdf)} ({has/len(zdf)*100:.1f}%)")

    return pdf, results


# ============================================================
# STEP 6: STABILITY
# ============================================================

def stability_ticker(df, zone_col):
    print(f"\n  --- Stability: Per-Ticker ({zone_col}) ---")
    syms = df["symbol"].value_counts()
    syms = syms[syms >= 5].index.tolist()
    print(f"  Tickers with n >= 5: {len(syms)}")

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
        fmt = lambda x: f"{x:>6.1f}%" if not np.isnan(x) else "    N/A"
        print(f"  {sym:<8} {n:>4} {n_top:>6} {n_bot:>6} {n_mid:>6} "
              f"{fmt(wr_top)} {fmt(wr_bot)} {fmt(wr_mid)}")

    return results


def stability_tod(df, zone_col):
    print(f"\n  --- Stability: Time-of-Day ({zone_col}) ---")
    print(f"\n  {'BUCKET':<14} {'N':>5} {'N_TOP':>6} {'N_BOT':>6} {'N_MID':>6} "
          f"{'WR_TOP':>7} {'WR_BOT':>7} {'WR_MID':>7}")
    print(f"  {'-'*75}")

    results = {}
    for bname in ["pre_market", "open", "mid", "close", "after_hours"]:
        bdf = df[df["tod_bucket"] == bname]
        if len(bdf) == 0: continue
        n = len(bdf)
        n_top = len(bdf[bdf[zone_col] == "TOP_ZONE"])
        n_bot = len(bdf[bdf[zone_col] == "BOTTOM_ZONE"])
        n_mid = len(bdf[bdf[zone_col] == "MID"])
        wr_top = bdf[bdf[zone_col] == "TOP_ZONE"]["is_winner"].mean() * 100 if n_top > 0 else float("nan")
        wr_bot = bdf[bdf[zone_col] == "BOTTOM_ZONE"]["is_winner"].mean() * 100 if n_bot > 0 else float("nan")
        wr_mid = bdf[bdf[zone_col] == "MID"]["is_winner"].mean() * 100 if n_mid > 0 else float("nan")

        results[bname] = {"n": n, "n_top": n_top, "n_bot": n_bot, "n_mid": n_mid,
                          "wr_top": wr_top, "wr_bot": wr_bot, "wr_mid": wr_mid}
        fmt = lambda x: f"{x:>6.1f}%" if not np.isnan(x) else "    N/A"
        print(f"  {bname:<14} {n:>5} {n_top:>6} {n_bot:>6} {n_mid:>6} "
              f"{fmt(wr_top)} {fmt(wr_bot)} {fmt(wr_mid)}")

    return results


# ============================================================
# STEP 7: REPORT
# ============================================================

def write_report(df_all, df_active, zone_results, pressure_df, pressure_results,
                 tick_stab, tod_stab, path):
    L = []
    L.append("# Phase 10B.1 — Structural Liquidity Zone Conditioning (FIXED)")
    L.append("")
    L.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    L.append(f"**Fix:** Replaced Databento-only daily OHLC with yfinance. Coverage: ~90%+")
    L.append(f"**Total events:** {len(df_all)}")
    L.append(f"**Active (PnL ≠ 0):** {len(df_active)}")
    for thresh in ZONE_THRESHOLDS:
        zcol = f"zone_{thresh}"
        zoned = df_active[df_active[zcol] != "NO_DATA"]
        n_top = len(zoned[zoned[zcol] == "TOP_ZONE"])
        n_bot = len(zoned[zoned[zcol] == "BOTTOM_ZONE"])
        n_mid = len(zoned[zoned[zcol] == "MID"])
        L.append(f"**ATR {thresh}:** labeled={len(zoned)} ({len(zoned)/len(df_active)*100:.0f}%), "
                 f"TOP={n_top}, BOT={n_bot}, MID={n_mid}")
    L.append("")
    L.append("---")

    # Zone results for each threshold
    for thresh in ZONE_THRESHOLDS:
        zcol = f"zone_{thresh}"
        if zcol not in zone_results:
            continue
        zr, comps = zone_results[zcol]

        L.append(f"\n## Zone Analysis — ATR {thresh}")
        L.append("")
        L.append("| Zone | N | WR | Mean PnL | Total PnL | Med MFE | Med MAE | Med R:R | CI95 WR | Syms |")
        L.append("|------|---|----|----------|-----------|---------|---------|---------|---------|------|")

        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID", "EDGE"]:
            if z not in zr: continue
            m = zr[z]["metrics"]
            ci = zr[z].get("ci_wr", (float("nan"), float("nan")))
            L.append(f"| {z} | {m['n']} | {m['win_rate']:.1f}% | ${m['mean_pnl']:+.2f} | "
                     f"${m['total_pnl']:+.0f} | {m['median_mfe']:.3f}% | {m['median_mae']:.3f}% | "
                     f"{m['median_rr']:.3f} | [{ci[0]:.1f}%,{ci[1]:.1f}%] | {m['unique_symbols']} |")

        L.append("")
        L.append("**Permutation tests:**")
        for comp_name, comp in comps.items():
            sig = " ★" if comp.get("p_wr", 1) < 0.05 else ""
            L.append(f"- {comp_name}: p(WR)={comp.get('p_wr','nan'):.4f}{sig}, "
                     f"p(PnL)={comp.get('p_pnl','nan'):.4f}, "
                     f"p(MFE)={comp.get('p_mfe','nan'):.4f}")
        L.append("")

    # Pressure
    if pressure_df is not None and len(pressure_df) > 0:
        L.append("## Pressure Precursor Analysis by Zone")
        L.append("")
        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID"]:
            zdf = pressure_df[pressure_df["zone"] == z]
            if len(zdf) == 0: continue
            has = (zdf["peak_pressure_z"] > 1.5).sum()
            L.append(f"**{z}** (n={len(zdf)}): peak_z={zdf['peak_pressure_z'].mean():.3f}, "
                     f"buildup={zdf['buildup_rate'].mean():.4f}, "
                     f"vol_accel={zdf['volume_acceleration'].mean():.2f}, "
                     f"precursor_freq={has/len(zdf)*100:.1f}%")
        sig_results = {k: v for k, v in pressure_results.items() if v < 0.10}
        if sig_results:
            L.append("")
            L.append("**Significant pressure differences (p < 0.10):**")
            for k, v in sorted(sig_results.items(), key=lambda x: x[1]):
                L.append(f"- {k}: p={v:.4f}")
        L.append("")

    # Stability
    L.append("## Stability: Time-of-Day")
    if tod_stab:
        L.append("")
        L.append("| Bucket | N | TOP | BOT | MID | WR TOP | WR BOT | WR MID |")
        L.append("|--------|---|-----|-----|-----|--------|--------|--------|")
        for b in ["pre_market", "open", "mid", "close", "after_hours"]:
            v = tod_stab.get(b)
            if not v: continue
            fmt = lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A"
            L.append(f"| {b} | {v['n']} | {v['n_top']} | {v['n_bot']} | {v['n_mid']} | "
                     f"{fmt(v['wr_top'])} | {fmt(v['wr_bot'])} | {fmt(v['wr_mid'])} |")
    L.append("")

    L.append("## Stability: Per-Ticker (n ≥ 5)")
    if tick_stab:
        L.append("")
        L.append("| Sym | N | TOP | BOT | MID | WR TOP | WR BOT | WR MID |")
        L.append("|-----|---|-----|-----|-----|--------|--------|--------|")
        for sym in sorted(tick_stab.keys()):
            v = tick_stab[sym]
            fmt = lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A"
            L.append(f"| {sym} | {v['n']} | {v['n_top']} | {v['n_bot']} | {v['n_mid']} | "
                     f"{fmt(v['wr_top'])} | {fmt(v['wr_bot'])} | {fmt(v['wr_mid'])} |")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Report: {path}")


# ============================================================
# MAIN
# ============================================================

def run(config=None):
    if config is None:
        config = ReplayConfig()

    print("=" * 80)
    print("PHASE 10B.1 — Structural Liquidity Zones (FIXED: yfinance daily OHLC)")
    print("=" * 80)

    # Step 1: Get all symbols and date range
    print("\n[STEP 1] Collecting symbols from trade ledger...")
    reports = config.reports_root
    symbols = set()
    min_date = "2099-01-01"
    max_date = "2000-01-01"
    for d in sorted(reports.iterdir()):
        ledger = d / "trade_ledger.jsonl"
        if not ledger.exists():
            continue
        if d.name < min_date:
            min_date = d.name
        if d.name > max_date:
            max_date = d.name
        with open(ledger) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                symbols.add(rec["symbol"])

    symbols = sorted(symbols)
    start = (pd.Timestamp(min_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"  Symbols: {len(symbols)}")
    print(f"  Trade range: {min_date} to {max_date}")
    print(f"  Download range: {start} to {end}")

    # Step 2: Download daily OHLC
    print("\n[STEP 2] Downloading daily OHLC via yfinance...")
    daily_data = download_daily_ohlc(symbols, start, end)

    missing = [s for s in symbols if s not in daily_data]
    if missing:
        print(f"  [WARN] Missing from yfinance: {len(missing)} symbols: {missing[:20]}")

    # Step 3: Load events with zones
    print("\n[STEP 3] Loading events and labeling zones...")
    df_all = load_events_with_zones(reports, daily_data)

    df_active = df_all[df_all["is_active"] == 1].copy()
    n_flat = len(df_all) - len(df_active)
    print(f"\n  Active: {len(df_active)} (excluded {n_flat} flat)")

    # Zone distribution
    for thresh in ZONE_THRESHOLDS:
        zcol = f"zone_{thresh}"
        print(f"\n  Zone distribution (ATR {thresh}):")
        for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID", "NO_DATA"]:
            n = len(df_active[df_active[zcol] == z])
            print(f"    {z}: {n} ({n/len(df_active)*100:.1f}%)")

    # Coverage check
    zcol_025 = "zone_0.25"
    labeled = len(df_active[df_active[zcol_025] != "NO_DATA"])
    coverage = labeled / len(df_active) * 100
    print(f"\n  *** COVERAGE (ATR 0.25): {labeled}/{len(df_active)} = {coverage:.1f}% ***")
    if coverage < 70:
        print(f"  [WARN] Coverage below 70% target")

    # Save raw data
    raw_path = config.output_root / "phase10b1_event_data.csv"
    df_active.to_csv(raw_path, index=False)
    print(f"  Saved: {raw_path}")

    # Step 4: Zone analysis
    print(f"\n[STEP 4] Zone analysis...")
    zone_results = {}
    for thresh in ZONE_THRESHOLDS:
        zcol = f"zone_{thresh}"
        zoned = df_active[df_active[zcol] != "NO_DATA"]
        if len(zoned) < 20:
            print(f"  [SKIP] ATR {thresh}: only {len(zoned)} events")
            continue

        print(f"\n{'='*80}")
        print(f"ZONE ANALYSIS — ATR {thresh}")
        print(f"{'='*80}")
        zr, comps = analyze_zones(zoned, zcol, f"ATR={thresh}")
        zone_results[zcol] = (zr, comps)

    # Step 5: Pressure by zone
    print(f"\n{'='*80}")
    print("PRESSURE ANALYSIS BY ZONE")
    print(f"{'='*80}")
    zoned_active = df_active[df_active["zone_0.25"] != "NO_DATA"]
    pressure_df, pressure_results = pressure_by_zone(zoned_active, config, "zone_0.25")

    # Step 6: Stability
    print(f"\n{'='*80}")
    print("STABILITY")
    print(f"{'='*80}")
    tick_stab = stability_ticker(zoned_active, "zone_0.25")
    tod_stab = stability_tod(zoned_active, "zone_0.25")

    # Step 7: Report
    report_path = config.output_root.parent / "docs" / "Research" / "PHASE10B1_StructuralZones_FIXED.md"
    write_report(df_all, df_active, zone_results, pressure_df, pressure_results,
                 tick_stab, tod_stab, report_path)

    json_path = config.output_root / "phase10b1_results.json"
    json_data = {
        "n_total": len(df_all), "n_active": len(df_active),
        "daily_ohlc_symbols": len(daily_data),
        "coverage": {
            thresh: {
                z: int(len(df_active[df_active[f"zone_{thresh}"] == z]))
                for z in ["TOP_ZONE", "BOTTOM_ZONE", "MID", "NO_DATA"]
            }
            for thresh in ZONE_THRESHOLDS
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    print("\n" + "=" * 80)
    print("PHASE 10B.1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 10B.1")
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
    run(config=config)
