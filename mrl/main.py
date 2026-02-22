import databento as db
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================
# CONFIGURATION — Phase 8: Locked hypothesis, no sweeping
# ============================================================

DATA_ROOT = Path(r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades")
ENRICHED_ROOT = Path(r"Z:\AI_BOT_DATA\replays")
TICKER_FILE = Path(r"C:\AI_BOT_Research\tickers.txt")

BAR_SEC = 30
ROLLING_WINDOW_SECONDS = 200
ROLL_WIN = max(int(round(ROLLING_WINDOW_SECONDS / BAR_SEC)), 2)
PRESSURE_THRESHOLD = 2.0
MIN_TOTAL_VOLUME_PER_BAR = 500
MIN_TRADE_COUNT_PER_BAR = 5
EPS = 1e-9

SPREAD_DYNAMICS_THRESHOLD = 60.0
ABSORPTION_THRESHOLD = 0.4
L2_IMBALANCE_THRESHOLD = 0.7
MERGE_TOLERANCE_SEC = 2

# ============================================================
# TICKER LIST
# ============================================================

def load_tickers():
    if TICKER_FILE.exists():
        tickers = [
            line.strip().upper()
            for line in TICKER_FILE.read_text().splitlines()
            if line.strip() and line.strip().isalpha()
        ]
        if tickers:
            print(f"  Ticker file found: {len(tickers)} tickers -> {tickers}")
            return tickers
    print("  No tickers.txt found — processing all files.")
    return None


def filter_files(all_files, tickers):
    if tickers is None:
        return all_files
    filtered = []
    for f in all_files:
        name_upper = f.name.upper()
        for t in tickers:
            if name_upper.startswith(t + "_"):
                filtered.append(f)
                break
    return filtered


# ============================================================
# ENRICHED DATA LOADER (Morpheus JSONL)
# ============================================================

def load_enriched_data():
    enriched_files = sorted(ENRICHED_ROOT.glob("enriched_*.jsonl"))
    if not enriched_files:
        return None

    records = []
    for ef in enriched_files:
        print(f"  Loading enriched: {ef.name}")
        with open(ef) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    records.append({
                        "timestamp": pd.Timestamp(rec["timestamp"]),
                        "symbol": rec.get("symbol", ""),
                        "spread_dynamics": rec.get("spread_dynamics"),
                        "absorption": rec.get("absorption"),
                        "l2_pressure": rec.get("l2_pressure"),
                        "nofi": rec.get("nofi"),
                        "momentum_score": rec.get("momentum_score"),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

    if not records:
        return None

    edf = pd.DataFrame(records)
    edf["timestamp"] = pd.to_datetime(edf["timestamp"], utc=True).dt.as_unit("us")
    edf = edf.sort_values("timestamp").reset_index(drop=True)
    print(f"  Enriched records loaded: {len(edf)}")
    return edf


# ============================================================
# FILE PROCESSING — 30s bars, locked parameters
# ============================================================

def extract_symbol(file_path):
    return file_path.name.split("_")[0].upper()


def process_file(file_path):
    try:
        store = db.DBNStore.from_file(file_path)
        df = store.to_df()
    except Exception as e:
        return None
    if len(df) == 0:
        return None

    df["size"] = df["size"].astype("int64")
    df = df.reset_index()

    df["pressure"] = 0
    df.loc[df["side"] == "B", "pressure"] = df["size"]
    df.loc[df["side"] == "A", "pressure"] = -df["size"]
    df["buy_volume"] = 0
    df.loc[df["side"] == "B", "buy_volume"] = df["size"]
    df["sell_volume"] = 0
    df.loc[df["side"] == "A", "sell_volume"] = df["size"]
    df = df.set_index("ts_recv")

    resample_str = f"{BAR_SEC}s"
    combined = pd.DataFrame({
        "pressure": df["pressure"].resample(resample_str).sum(),
        "price": df["price"].resample(resample_str).last(),
        "buy_volume": df["buy_volume"].resample(resample_str).sum(),
        "sell_volume": df["sell_volume"].resample(resample_str).sum(),
        "total_volume": df["size"].resample(resample_str).sum(),
        "trade_count": df["size"].resample(resample_str).count(),
    }).dropna(subset=["price"])

    combined["active"] = (
        (combined["total_volume"] >= MIN_TOTAL_VOLUME_PER_BAR)
        & (combined["trade_count"] >= MIN_TRADE_COUNT_PER_BAR)
    )

    combined["price_change"] = combined["price"].diff()
    combined["vol_200s"] = combined["price_change"].rolling(ROLL_WIN).std()

    p_mean = combined["pressure"].rolling(ROLL_WIN).mean()
    p_std = combined["pressure"].rolling(ROLL_WIN).std()
    combined["pressure_z"] = (combined["pressure"] - p_mean) / (p_std + EPS)

    combined["forward_60s"] = combined["price"].shift(-2) - combined["price"]
    combined["forward_180s"] = combined["price"].shift(-6) - combined["price"]

    # MAE/MFE: forward-looking min/max price within horizon
    price_arr = combined["price"].values
    n = len(price_arr)
    for horizon_label, n_bars in [("60s", 2), ("180s", 6)]:
        max_ahead = np.full(n, np.nan)
        min_ahead = np.full(n, np.nan)
        for i in range(n):
            end = min(i + n_bars + 1, n)
            if i + 1 < end:
                window = price_arr[i + 1: end]
                if len(window) > 0:
                    max_ahead[i] = np.nanmax(window)
                    min_ahead[i] = np.nanmin(window)
        combined[f"max_price_{horizon_label}"] = max_ahead
        combined[f"min_price_{horizon_label}"] = min_ahead

    combined["symbol"] = extract_symbol(file_path)
    combined["source_file"] = file_path.name

    return combined


# ============================================================
# VOLATILITY REGIME ASSIGNMENT
# ============================================================

def assign_vol_regime(all_data):
    active_mask = all_data["active"]
    vol_vals = all_data.loc[active_mask, "vol_200s"]
    q66 = vol_vals.quantile(0.66)
    all_data["vol_regime"] = "OTHER"
    all_data.loc[all_data["vol_200s"] >= q66, "vol_regime"] = "HIGH"
    return all_data


# ============================================================
# LSI MERGE — Join Morpheus enriched features
# ============================================================

def merge_enriched(all_data, enriched_df):
    if enriched_df is None:
        return all_data

    all_data = all_data.reset_index()
    all_data["ts_recv"] = pd.to_datetime(all_data["ts_recv"], utc=True).dt.as_unit("us")

    merged_frames = []
    for symbol in all_data["symbol"].unique():
        trade_slice = all_data[all_data["symbol"] == symbol].copy()
        enrich_slice = enriched_df[enriched_df["symbol"] == symbol].copy()

        if len(enrich_slice) == 0:
            for col in ["spread_dynamics", "absorption", "l2_pressure", "nofi", "momentum_score"]:
                if col not in trade_slice.columns:
                    trade_slice[col] = np.nan
            merged_frames.append(trade_slice)
            continue

        trade_slice = trade_slice.sort_values("ts_recv")
        trade_slice["ts_recv"] = trade_slice["ts_recv"].dt.as_unit("us")
        enrich_slice = enrich_slice.sort_values("timestamp")
        enrich_slice["timestamp"] = enrich_slice["timestamp"].dt.as_unit("us")

        merged = pd.merge_asof(
            trade_slice,
            enrich_slice[["timestamp", "spread_dynamics", "absorption",
                          "l2_pressure", "nofi", "momentum_score"]],
            left_on="ts_recv",
            right_on="timestamp",
            tolerance=pd.Timedelta(seconds=MERGE_TOLERANCE_SEC),
            direction="nearest",
        )
        merged_frames.append(merged)

    result = pd.concat(merged_frames, axis=0)
    result = result.set_index("ts_recv").sort_index()

    if "timestamp" in result.columns:
        result = result.drop(columns=["timestamp"])

    return result


# ============================================================
# LSI FLAG COMPUTATION
# ============================================================

def compute_lsi_flags(all_data):
    has_enriched = "spread_dynamics" in all_data.columns and all_data["spread_dynamics"].notna().any()

    if not has_enriched:
        all_data["spread_widening_flag"] = False
        all_data["absorption_flag"] = False
        all_data["l2_imbalance_flag"] = False
        all_data["LSI"] = False
        return all_data, False

    all_data["spread_widening_flag"] = all_data["spread_dynamics"].fillna(0) >= SPREAD_DYNAMICS_THRESHOLD
    all_data["absorption_flag"] = all_data["absorption"].fillna(1.0) <= ABSORPTION_THRESHOLD
    all_data["l2_imbalance_flag"] = all_data["l2_pressure"].fillna(0).abs() >= L2_IMBALANCE_THRESHOLD

    all_data["LSI"] = (
        (all_data["pressure_z"].abs() >= PRESSURE_THRESHOLD)
        & all_data["spread_widening_flag"]
        & all_data["absorption_flag"]
    )

    return all_data, True


# ============================================================
# MAE / MFE COMPUTATION
# ============================================================

def compute_mae_mfe(events, horizon_label):
    max_col = f"max_price_{horizon_label}"
    min_col = f"min_price_{horizon_label}"

    valid = events.dropna(subset=[max_col, min_col, "price"]).copy()
    if len(valid) == 0:
        return None

    is_short = valid["pressure_z"] > 0

    mfe = np.where(
        is_short,
        valid["price"] - valid[min_col],
        valid[max_col] - valid["price"],
    )
    mae = np.where(
        is_short,
        valid[max_col] - valid["price"],
        valid["price"] - valid[min_col],
    )

    return {
        "n": len(valid),
        "mfe_mean": float(np.nanmean(mfe)),
        "mfe_median": float(np.nanmedian(mfe)),
        "mae_mean": float(np.nanmean(mae)),
        "mae_median": float(np.nanmedian(mae)),
        "mfe_std": float(np.nanstd(mfe)),
        "mae_std": float(np.nanstd(mae)),
        "reward_risk_ratio": float(np.nanmedian(mfe) / (np.nanmedian(mae) + EPS)),
    }


# ============================================================
# FADE EVALUATION
# ============================================================

def evaluate_fade(events, label=""):
    n = len(events)
    if n == 0:
        print(f"  {label}: n=0 — no events")
        return

    results = {"n": n}

    for horizon in ["60s", "180s"]:
        col = f"forward_{horizon}"
        valid = events.dropna(subset=[col, "pressure_z"])
        count = len(valid)

        if count == 0:
            results[f"hit_{horizon}"] = float("nan")
            results[f"med_ret_{horizon}"] = float("nan")
            results[f"mean_ret_{horizon}"] = float("nan")
            results[f"count_{horizon}"] = 0
            continue

        wins = (
            (valid["pressure_z"] > 0) & (valid[col] < 0)
        ) | (
            (valid["pressure_z"] < 0) & (valid[col] > 0)
        )
        results[f"hit_{horizon}"] = 100.0 * wins.sum() / count
        results[f"med_ret_{horizon}"] = valid[col].median()
        results[f"mean_ret_{horizon}"] = valid[col].mean()
        results[f"count_{horizon}"] = count

    print(f"\n  {label}")
    print(f"    Events:           {n}")
    for horizon in ["60s", "180s"]:
        hit = results.get(f"hit_{horizon}", float("nan"))
        med = results.get(f"med_ret_{horizon}", float("nan"))
        mean = results.get(f"mean_ret_{horizon}", float("nan"))
        cnt = results.get(f"count_{horizon}", 0)
        hit_str = f"{hit:.1f}%" if pd.notna(hit) else "N/A"
        med_str = f"{med:+.4f}" if pd.notna(med) else "N/A"
        mean_str = f"{mean:+.4f}" if pd.notna(mean) else "N/A"
        print(f"    {horizon}: n={cnt}  hit={hit_str}  median_ret={med_str}  mean_ret={mean_str}")

    for horizon in ["60s", "180s"]:
        mae_mfe = compute_mae_mfe(events, horizon)
        if mae_mfe:
            print(f"    {horizon} MAE/MFE:  MFE_med={mae_mfe['mfe_median']:+.4f}  "
                  f"MAE_med={mae_mfe['mae_median']:+.4f}  "
                  f"MFE_mean={mae_mfe['mfe_mean']:+.4f}  "
                  f"MAE_mean={mae_mfe['mae_mean']:+.4f}  "
                  f"R:R={mae_mfe['reward_risk_ratio']:.2f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PHASE 8 — LIQUIDITY SHOCK + INVENTORY FADE VALIDATION")
    print("=" * 80)
    print(f"Bar size:        {BAR_SEC}s")
    print(f"Rolling window:  {ROLL_WIN} bars (~{ROLL_WIN * BAR_SEC}s)")
    print(f"Threshold:       {PRESSURE_THRESHOLD}")
    print(f"Mode:            FADE only")
    print(f"Horizons:        60s, 180s")
    print(f"Vol regime:      HIGH only")
    print()

    tickers = load_tickers()
    all_files = sorted(DATA_ROOT.rglob("*.dbn.zst"))
    all_files = filter_files(all_files, tickers)
    print(f"  Files to process: {len(all_files)}")
    print()

    if len(all_files) == 0:
        print("[ERROR] No matching files found.")
        exit(1)

    frames = []
    processed = 0
    symbols_seen = set()
    for i, fpath in enumerate(all_files):
        result = process_file(fpath)
        if result is not None:
            frames.append(result)
            processed += 1
            symbols_seen.add(extract_symbol(fpath))

    print(f"  Files processed:   {processed}")
    print(f"  Tickers processed: {len(symbols_seen)} -> {sorted(symbols_seen)}")

    if processed == 0:
        print("[ERROR] No data produced.")
        exit(1)

    all_data = pd.concat(frames, axis=0).sort_index()
    all_data = assign_vol_regime(all_data)

    total_bars = len(all_data)
    active_bars = int(all_data["active"].sum())
    high_vol_active = int(((all_data["active"]) & (all_data["vol_regime"] == "HIGH")).sum())

    print(f"  Total bars:        {total_bars}")
    print(f"  Active bars:       {active_bars}")
    print(f"  HIGH vol active:   {high_vol_active}")

    print()
    print("=" * 80)
    print("LOADING ENRICHED DATA (Morpheus features)")
    print("=" * 80)
    enriched_df = load_enriched_data()

    all_data = merge_enriched(all_data, enriched_df)
    all_data, has_lsi = compute_lsi_flags(all_data)

    # ============================================================
    # A) BASELINE
    # ============================================================

    print()
    print("=" * 80)
    print("A) BASELINE — pressure_z FADE | HIGH VOL | THR 2.0")
    print("=" * 80)

    baseline_mask = (
        all_data["active"]
        & (all_data["vol_regime"] == "HIGH")
        & (all_data["pressure_z"].abs() >= PRESSURE_THRESHOLD)
    )
    baseline_events = all_data[baseline_mask].copy()

    evaluate_fade(baseline_events, label="BASELINE (pressure_z >= 2.0, HIGH vol, FADE)")

    # ============================================================
    # B) LSI FILTERED
    # ============================================================

    print()
    print("=" * 80)
    print("B) LSI FILTERED — Liquidity Shock Index")
    print("=" * 80)

    if has_lsi:
        lsi_mask = (
            all_data["active"]
            & (all_data["vol_regime"] == "HIGH")
            & all_data["LSI"]
        )
        lsi_events = all_data[lsi_mask].copy()

        print(f"  LSI config:")
        print(f"    spread_dynamics >= {SPREAD_DYNAMICS_THRESHOLD}")
        print(f"    absorption <= {ABSORPTION_THRESHOLD}")
        print(f"    |l2_pressure| >= {L2_IMBALANCE_THRESHOLD}")

        evaluate_fade(lsi_events, label="LSI FILTERED (pressure_z + spread_widening + weak_absorption)")

        enriched_count = all_data.loc[baseline_mask, "spread_dynamics"].notna().sum()
        print(f"\n  Enriched coverage on baseline events: {enriched_count}/{len(baseline_events)}")
    else:
        print("  No enriched Morpheus data available.")
        print("  LSI evaluation skipped.")
        print("  To enable: place enriched_*.jsonl files in Z:\\AI_BOT_DATA\\replays\\")

    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 80)
    print("PHASE 8 COMPLETE")
    print("=" * 80)
    print(f"  Files: {processed}  |  Tickers: {len(symbols_seen)}  |  Total bars: {total_bars}")
    print(f"  Active bars: {active_bars}  |  HIGH vol active: {high_vol_active}")
    print(f"  Baseline events: {len(baseline_events)}")
    if has_lsi:
        print(f"  LSI events: {len(lsi_events)}")
    print(f"  Enriched data: {'YES' if has_lsi else 'NO'}")
    print("=" * 80)