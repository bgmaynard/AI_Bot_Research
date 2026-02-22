"""
Replay Engine — HYP-013: Does pressure precede Morpheus ignition?

Core research question:
    Does pressure_z build BEFORE Morpheus fires an entry signal?
    If yes, by how much? Is the lead time statistically significant?

Architecture:
    1. Load Morpheus trade_ledger.jsonl → extract ignition events (entry_time, symbol)
    2. Load Databento raw trade data around each ignition event
    3. Compute pressure_z in a window BEFORE entry
    4. Measure pressure buildup characteristics
    5. Compare against random control windows (no ignition) to prove signal vs noise

Data Sources:
    - trade_ledger.jsonl: reports/{date}/trade_ledger.jsonl (from ai_project_hub)
    - entry_blocks.jsonl: reports/{date}/entry_blocks.jsonl (blocked signals)
    - Databento raw trades: Z:\\AI_BOT_DATA\\databento_cache\\XNAS.ITCH\\trades\\

Output:
    - Per-event pressure profile (pressure_z series before entry)
    - Aggregate statistics: mean lead time, pressure buildup rate
    - Control comparison: ignition events vs random windows
    - Statistical tests: permutation test, bootstrap CI
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ReplayConfig:
    """Configuration for replay engine."""

    # Data paths (adjust for your environment)
    reports_root: Path = Path(r"\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports")
    databento_root: Path = Path(r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades")
    output_root: Path = Path(r"C:\AI_Bot_Research\results")

    # Window around ignition events
    pre_window_sec: int = 300       # 5 minutes before entry
    post_window_sec: int = 60       # 1 minute after entry

    # Bar aggregation
    bar_sec: int = 5                # 5-second bars for finer resolution pre-ignition
    rolling_window_bars: int = 20   # Rolling window for pressure_z (20 bars = 100s at 5s bars)

    # Pressure thresholds
    pressure_z_threshold: float = 1.5   # Lower than Phase 8 (2.0) — we want to detect buildup
    min_volume_per_bar: int = 50        # Lower threshold for finer bars
    min_trades_per_bar: int = 3

    # Control sampling
    n_controls_per_event: int = 3   # Random control windows per ignition event
    control_min_gap_sec: int = 600  # Controls must be 10+ min away from any ignition

    # Statistical
    min_events: int = 30            # Minimum events to run analysis
    bootstrap_n: int = 1000         # Bootstrap iterations for CI
    permutation_n: int = 1000       # Permutation test iterations

    eps: float = 1e-9


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class IgnitionEvent:
    """A single Morpheus ignition event extracted from trade_ledger."""
    trade_id: str
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    entry_signal: str
    pnl: float
    pnl_percent: float
    max_gain_percent: float
    max_drawdown_percent: float
    hold_time_seconds: int
    volatility_regime: str
    momentum_score: float
    momentum_state: str
    rvol: float
    change_pct: float
    spread_pct: float
    primary_exit_category: str


@dataclass
class PressureProfile:
    """Pressure measurements in the window around an ignition event."""
    event: IgnitionEvent
    is_control: bool = False

    # Time series (bar-level)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    pressure_z: List[float] = field(default_factory=list)
    raw_pressure: List[float] = field(default_factory=list)
    buy_volume: List[int] = field(default_factory=list)
    sell_volume: List[int] = field(default_factory=list)
    total_volume: List[int] = field(default_factory=list)
    price: List[float] = field(default_factory=list)

    # Summary metrics (computed after populating time series)
    peak_pressure_z_pre: float = 0.0          # Max |pressure_z| before entry
    mean_pressure_z_pre: float = 0.0          # Mean pressure_z before entry
    pressure_direction_pre: str = ""           # "BUY" or "SELL" dominant
    pressure_buildup_rate: float = 0.0         # Rate of pressure increase (slope)
    first_threshold_cross_sec: float = float('nan')  # Seconds before entry that pressure first crossed threshold
    bars_above_threshold_pre: int = 0          # How many pre-entry bars exceeded threshold
    pressure_consistency: float = 0.0          # % of pre-entry bars with same-sign pressure
    volume_acceleration: float = 0.0           # Volume trend before entry

    # Data quality
    total_bars: int = 0
    pre_bars: int = 0
    post_bars: int = 0
    coverage_pct: float = 0.0


# ============================================================
# STEP 1: LOAD IGNITION EVENTS FROM TRADE LEDGER
# ============================================================

def load_ignition_events(reports_root: Path, symbols: List[str] = None) -> List[IgnitionEvent]:
    """
    Load all Morpheus trade entries from trade_ledger.jsonl files.
    Each trade = one ignition event (signal passed the entire funnel).
    """
    events = []

    # Search for trade_ledger.jsonl in all date directories
    ledger_files = sorted(reports_root.rglob("trade_ledger.jsonl"))

    if not ledger_files:
        logger.warning(f"No trade_ledger.jsonl files found in {reports_root}")
        return events

    for ledger_path in ledger_files:
        try:
            with open(ledger_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by symbol if specified
                    symbol = rec.get("symbol", "")
                    if symbols and symbol not in symbols:
                        continue

                    # Only include closed trades
                    if rec.get("status") != "closed":
                        continue

                    # Parse entry time
                    entry_time_str = rec.get("entry_time", "")
                    if not entry_time_str:
                        continue

                    try:
                        entry_time = pd.Timestamp(entry_time_str)
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.tz_localize("US/Eastern")
                        entry_time = entry_time.tz_convert("UTC")
                    except Exception:
                        continue

                    # Extract secondary triggers
                    sec = rec.get("secondary_triggers", {})

                    event = IgnitionEvent(
                        trade_id=rec.get("trade_id", ""),
                        symbol=symbol,
                        entry_time=entry_time,
                        entry_price=float(rec.get("entry_price", 0)),
                        entry_signal=rec.get("entry_signal", ""),
                        pnl=float(rec.get("pnl", 0)),
                        pnl_percent=float(rec.get("pnl_percent", 0)),
                        max_gain_percent=float(rec.get("max_gain_percent", 0)),
                        max_drawdown_percent=float(rec.get("max_drawdown_percent", 0)),
                        hold_time_seconds=int(rec.get("hold_time_seconds", 0)),
                        volatility_regime=rec.get("volatility_regime", ""),
                        momentum_score=float(rec.get("entry_momentum_score", 0)),
                        momentum_state=rec.get("entry_momentum_state", ""),
                        rvol=float(sec.get("relative_volume", 0) or rec.get("rvol_at_entry", 0)),
                        change_pct=float(sec.get("day_change_at_entry", 0)),
                        spread_pct=float(rec.get("entry_spread_pct", 0)),
                        primary_exit_category=rec.get("primary_exit_category", ""),
                    )
                    events.append(event)

        except Exception as e:
            logger.warning(f"Error reading {ledger_path}: {e}")

    logger.info(f"Loaded {len(events)} ignition events from {len(ledger_files)} ledger files")
    return events


# ============================================================
# STEP 2: LOAD RAW DATABENTO TRADE DATA AROUND AN EVENT
# ============================================================

def find_databento_file(databento_root: Path, symbol: str, event_time: pd.Timestamp) -> Optional[Path]:
    """
    Find the Databento .dbn.zst file that covers the event timestamp.
    Files are named like: SYMBOL_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS.dbn.zst
    """
    pattern = f"{symbol}_*.dbn.zst"
    candidates = list(databento_root.glob(pattern))

    if not candidates:
        # Try rglob for nested directories
        candidates = list(databento_root.rglob(pattern))

    if not candidates:
        return None

    # Match by date in filename
    event_date = event_time.strftime("%Y%m%d")
    for c in candidates:
        if event_date in c.name:
            return c

    # Fallback: return most recent file for this symbol
    return sorted(candidates)[-1] if candidates else None


def load_raw_trades_around_event(
    databento_root: Path,
    symbol: str,
    event_time: pd.Timestamp,
    pre_sec: int = 300,
    post_sec: int = 60,
) -> Optional[pd.DataFrame]:
    """
    Load raw Databento trade data in a window around an ignition event.

    Returns DataFrame with columns: ts_recv, price, size, side
    """
    try:
        import databento as db
    except ImportError:
        logger.error("databento package not installed. Run: pip install databento")
        return None

    file_path = find_databento_file(databento_root, symbol, event_time)
    if file_path is None:
        logger.debug(f"No Databento file found for {symbol} on {event_time.date()}")
        return None

    try:
        store = db.DBNStore.from_file(file_path)
        df = store.to_df()
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return None

    if len(df) == 0:
        return None

    df = df.reset_index()

    # Ensure UTC timestamps
    if df["ts_recv"].dt.tz is None:
        df["ts_recv"] = df["ts_recv"].dt.tz_localize("UTC")
    else:
        df["ts_recv"] = df["ts_recv"].dt.tz_convert("UTC")

    # Filter to window
    window_start = event_time - pd.Timedelta(seconds=pre_sec)
    window_end = event_time + pd.Timedelta(seconds=post_sec)

    mask = (df["ts_recv"] >= window_start) & (df["ts_recv"] <= window_end)
    window_df = df[mask].copy()

    if len(window_df) == 0:
        logger.debug(f"No trades in window for {symbol} around {event_time}")
        return None

    return window_df


# ============================================================
# STEP 3: COMPUTE PRESSURE PROFILE FOR A WINDOW
# ============================================================

def compute_pressure_profile(
    trades_df: pd.DataFrame,
    event: IgnitionEvent,
    config: ReplayConfig,
    is_control: bool = False,
) -> Optional[PressureProfile]:
    """
    Compute pressure_z time series from raw trades around an event.

    Process:
    1. Classify trades as buy/sell from 'side' column
    2. Aggregate into bar_sec bars
    3. Compute rolling pressure_z
    4. Split into pre-entry and post-entry
    5. Measure pressure buildup characteristics
    """
    df = trades_df.copy()
    df["size"] = df["size"].astype("int64")

    # Classify pressure
    df["pressure"] = 0
    df.loc[df["side"] == "B", "pressure"] = df["size"]
    df.loc[df["side"] == "A", "pressure"] = -df["size"]
    df["buy_vol"] = 0
    df.loc[df["side"] == "B", "buy_vol"] = df["size"]
    df["sell_vol"] = 0
    df.loc[df["side"] == "A", "sell_vol"] = df["size"]

    df = df.set_index("ts_recv")

    # Resample into bars
    resample_str = f"{config.bar_sec}s"
    bars = pd.DataFrame({
        "pressure": df["pressure"].resample(resample_str).sum(),
        "price": df["price"].resample(resample_str).last(),
        "buy_volume": df["buy_vol"].resample(resample_str).sum(),
        "sell_volume": df["sell_vol"].resample(resample_str).sum(),
        "total_volume": df["size"].resample(resample_str).sum(),
        "trade_count": df["size"].resample(resample_str).count(),
    }).dropna(subset=["price"])

    if len(bars) < config.rolling_window_bars + 2:
        return None

    # Compute pressure_z
    p_mean = bars["pressure"].rolling(config.rolling_window_bars).mean()
    p_std = bars["pressure"].rolling(config.rolling_window_bars).std()
    bars["pressure_z"] = (bars["pressure"] - p_mean) / (p_std + config.eps)

    # Drop NaN from rolling window warmup
    bars = bars.dropna(subset=["pressure_z"])

    if len(bars) == 0:
        return None

    # Split into pre and post entry
    entry_time = event.entry_time
    pre_bars = bars[bars.index < entry_time]
    post_bars = bars[bars.index >= entry_time]

    # Build profile
    profile = PressureProfile(
        event=event,
        is_control=is_control,
        timestamps=list(bars.index),
        pressure_z=list(bars["pressure_z"].values),
        raw_pressure=list(bars["pressure"].values),
        buy_volume=list(bars["buy_volume"].astype(int).values),
        sell_volume=list(bars["sell_volume"].astype(int).values),
        total_volume=list(bars["total_volume"].astype(int).values),
        price=list(bars["price"].values),
        total_bars=len(bars),
        pre_bars=len(pre_bars),
        post_bars=len(post_bars),
    )

    # Compute summary metrics from pre-entry bars
    if len(pre_bars) > 0:
        pz_pre = pre_bars["pressure_z"].values

        profile.peak_pressure_z_pre = float(np.max(np.abs(pz_pre)))
        profile.mean_pressure_z_pre = float(np.mean(pz_pre))
        profile.pressure_direction_pre = "BUY" if np.mean(pz_pre) > 0 else "SELL"

        # Pressure buildup rate: slope of |pressure_z| over pre-entry bars
        if len(pz_pre) >= 3:
            x = np.arange(len(pz_pre))
            abs_pz = np.abs(pz_pre)
            try:
                slope = np.polyfit(x, abs_pz, 1)[0]
                profile.pressure_buildup_rate = float(slope)
            except Exception:
                pass

        # First threshold crossing: how many seconds before entry?
        threshold = config.pressure_z_threshold
        above_threshold = np.abs(pz_pre) >= threshold
        if above_threshold.any():
            first_cross_idx = np.argmax(above_threshold)
            first_cross_time = pre_bars.index[first_cross_idx]
            lead_time = (entry_time - first_cross_time).total_seconds()
            profile.first_threshold_cross_sec = float(lead_time)
            profile.bars_above_threshold_pre = int(above_threshold.sum())

        # Pressure consistency: what % of pre-bars have same sign as dominant direction?
        if len(pz_pre) > 0:
            dominant_sign = np.sign(np.mean(pz_pre))
            same_sign = np.sum(np.sign(pz_pre) == dominant_sign)
            profile.pressure_consistency = float(same_sign / len(pz_pre))

        # Volume acceleration: slope of total_volume
        vol_pre = pre_bars["total_volume"].values.astype(float)
        if len(vol_pre) >= 3:
            try:
                vol_slope = np.polyfit(np.arange(len(vol_pre)), vol_pre, 1)[0]
                profile.volume_acceleration = float(vol_slope)
            except Exception:
                pass

    profile.coverage_pct = float(profile.total_bars / max(
        (config.pre_window_sec + config.post_window_sec) / config.bar_sec, 1
    ))

    return profile


# ============================================================
# STEP 4: GENERATE CONTROL WINDOWS
# ============================================================

def generate_control_events(
    real_events: List[IgnitionEvent],
    config: ReplayConfig,
) -> List[IgnitionEvent]:
    """
    Generate random control windows for comparison.
    Controls are at random times for the same symbol/date,
    but must be far from any real ignition event.
    """
    controls = []
    rng = np.random.default_rng(42)  # Reproducible

    # Group events by (symbol, date)
    by_sym_date: Dict[Tuple[str, str], List[IgnitionEvent]] = {}
    for evt in real_events:
        key = (evt.symbol, evt.entry_time.strftime("%Y-%m-%d"))
        by_sym_date.setdefault(key, []).append(evt)

    for (symbol, date_str), events in by_sym_date.items():
        # Trading hours window (UTC: ~11:30 to 20:00 for US Eastern 6:30-15:00)
        base_date = pd.Timestamp(date_str, tz="UTC")
        market_start = base_date + pd.Timedelta(hours=11, minutes=30)
        market_end = base_date + pd.Timedelta(hours=20)

        # Blocked zones: config.control_min_gap_sec around each event
        blocked = []
        for evt in events:
            blocked.append((
                evt.entry_time - pd.Timedelta(seconds=config.control_min_gap_sec),
                evt.entry_time + pd.Timedelta(seconds=config.control_min_gap_sec),
            ))

        # Generate random control times
        n_needed = config.n_controls_per_event * len(events)
        attempts = 0
        generated = 0

        while generated < n_needed and attempts < n_needed * 10:
            attempts += 1
            rand_sec = rng.uniform(0, (market_end - market_start).total_seconds())
            rand_time = market_start + pd.Timedelta(seconds=rand_sec)

            # Check not in blocked zone
            in_blocked = False
            for bstart, bend in blocked:
                if bstart <= rand_time <= bend:
                    in_blocked = True
                    break

            if not in_blocked:
                ctrl_event = IgnitionEvent(
                    trade_id=f"CTRL_{symbol}_{date_str}_{generated}",
                    symbol=symbol,
                    entry_time=rand_time,
                    entry_price=0.0,
                    entry_signal="CONTROL",
                    pnl=0.0,
                    pnl_percent=0.0,
                    max_gain_percent=0.0,
                    max_drawdown_percent=0.0,
                    hold_time_seconds=0,
                    volatility_regime="",
                    momentum_score=0.0,
                    momentum_state="CONTROL",
                    rvol=0.0,
                    change_pct=0.0,
                    spread_pct=0.0,
                    primary_exit_category="CONTROL",
                )
                controls.append(ctrl_event)
                generated += 1

    logger.info(f"Generated {len(controls)} control events for {len(real_events)} real events")
    return controls


# ============================================================
# STEP 5: ANALYZE RESULTS
# ============================================================

def analyze_pressure_profiles(
    real_profiles: List[PressureProfile],
    control_profiles: List[PressureProfile],
    config: ReplayConfig,
) -> Dict:
    """
    Compare pressure characteristics between real ignition events and controls.

    Key questions:
    - Is peak pressure_z higher before real ignitions than controls?
    - Is pressure buildup rate steeper before real ignitions?
    - How far ahead does pressure start building?
    - Is this statistically significant?
    """
    results = {
        "n_real": len(real_profiles),
        "n_control": len(control_profiles),
    }

    if len(real_profiles) < config.min_events:
        results["error"] = f"Insufficient events: {len(real_profiles)} < {config.min_events}"
        return results

    # Extract metrics
    def extract_metrics(profiles):
        return {
            "peak_pressure_z": [p.peak_pressure_z_pre for p in profiles],
            "mean_pressure_z": [p.mean_pressure_z_pre for p in profiles],
            "buildup_rate": [p.pressure_buildup_rate for p in profiles],
            "first_cross_sec": [p.first_threshold_cross_sec for p in profiles
                                if not np.isnan(p.first_threshold_cross_sec)],
            "bars_above_threshold": [p.bars_above_threshold_pre for p in profiles],
            "pressure_consistency": [p.pressure_consistency for p in profiles],
            "volume_acceleration": [p.volume_acceleration for p in profiles],
        }

    real_metrics = extract_metrics(real_profiles)
    ctrl_metrics = extract_metrics(control_profiles) if control_profiles else None

    # Summarize real events
    results["real_summary"] = {}
    for key, vals in real_metrics.items():
        arr = np.array(vals)
        if len(arr) > 0:
            results["real_summary"][key] = {
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "std": float(np.nanstd(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "n": len(arr),
            }

    # Summarize controls
    if ctrl_metrics:
        results["control_summary"] = {}
        for key, vals in ctrl_metrics.items():
            arr = np.array(vals)
            if len(arr) > 0:
                results["control_summary"][key] = {
                    "mean": float(np.nanmean(arr)),
                    "median": float(np.nanmedian(arr)),
                    "std": float(np.nanstd(arr)),
                    "n": len(arr),
                }

    # Statistical comparison: permutation test on key metrics
    if ctrl_metrics:
        results["statistical_tests"] = {}
        for metric_name in ["peak_pressure_z", "buildup_rate", "volume_acceleration"]:
            real_vals = np.array(real_metrics[metric_name])
            ctrl_vals = np.array(ctrl_metrics[metric_name])

            if len(real_vals) >= 10 and len(ctrl_vals) >= 10:
                observed_diff = np.nanmean(real_vals) - np.nanmean(ctrl_vals)

                # Permutation test
                combined = np.concatenate([real_vals, ctrl_vals])
                rng = np.random.default_rng(42)
                n_real = len(real_vals)
                perm_diffs = []
                for _ in range(config.permutation_n):
                    rng.shuffle(combined)
                    perm_diff = np.nanmean(combined[:n_real]) - np.nanmean(combined[n_real:])
                    perm_diffs.append(perm_diff)

                perm_diffs = np.array(perm_diffs)
                p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

                # Bootstrap CI on the difference
                boot_diffs = []
                for _ in range(config.bootstrap_n):
                    boot_real = rng.choice(real_vals, size=len(real_vals), replace=True)
                    boot_ctrl = rng.choice(ctrl_vals, size=len(ctrl_vals), replace=True)
                    boot_diffs.append(np.nanmean(boot_real) - np.nanmean(boot_ctrl))
                boot_diffs = np.array(boot_diffs)

                results["statistical_tests"][metric_name] = {
                    "observed_diff": float(observed_diff),
                    "p_value": p_value,
                    "significant_05": p_value < 0.05,
                    "significant_01": p_value < 0.01,
                    "ci_95_lower": float(np.percentile(boot_diffs, 2.5)),
                    "ci_95_upper": float(np.percentile(boot_diffs, 97.5)),
                }

    # Lead time analysis (key output for HYP-013)
    cross_secs = real_metrics["first_cross_sec"]
    if cross_secs:
        results["lead_time_analysis"] = {
            "n_events_with_precursor": len(cross_secs),
            "pct_events_with_precursor": float(len(cross_secs) / len(real_profiles) * 100),
            "mean_lead_sec": float(np.nanmean(cross_secs)),
            "median_lead_sec": float(np.nanmedian(cross_secs)),
            "std_lead_sec": float(np.nanstd(cross_secs)),
            "min_lead_sec": float(np.nanmin(cross_secs)),
            "max_lead_sec": float(np.nanmax(cross_secs)),
            "pct_above_5sec": float(np.mean(np.array(cross_secs) > 5) * 100),
            "pct_above_30sec": float(np.mean(np.array(cross_secs) > 30) * 100),
            "pct_above_60sec": float(np.mean(np.array(cross_secs) > 60) * 100),
        }

    # Winning vs losing trade comparison
    winners = [p for p in real_profiles if p.event.pnl > 0]
    losers = [p for p in real_profiles if p.event.pnl <= 0]

    if len(winners) >= 5 and len(losers) >= 5:
        results["winner_vs_loser"] = {
            "winners": {
                "n": len(winners),
                "mean_peak_pressure_z": float(np.nanmean([w.peak_pressure_z_pre for w in winners])),
                "mean_buildup_rate": float(np.nanmean([w.pressure_buildup_rate for w in winners])),
                "mean_consistency": float(np.nanmean([w.pressure_consistency for w in winners])),
                "pct_with_precursor": float(
                    sum(1 for w in winners if not np.isnan(w.first_threshold_cross_sec)) / len(winners) * 100
                ),
            },
            "losers": {
                "n": len(losers),
                "mean_peak_pressure_z": float(np.nanmean([l.peak_pressure_z_pre for l in losers])),
                "mean_buildup_rate": float(np.nanmean([l.pressure_buildup_rate for l in losers])),
                "mean_consistency": float(np.nanmean([l.pressure_consistency for l in losers])),
                "pct_with_precursor": float(
                    sum(1 for l in losers if not np.isnan(l.first_threshold_cross_sec)) / len(losers) * 100
                ),
            },
        }

    return results


# ============================================================
# STEP 6: MAIN REPLAY RUNNER
# ============================================================

def run_replay(
    config: ReplayConfig = None,
    symbols: List[str] = None,
    max_events: int = None,
    verbose: bool = True,
) -> Dict:
    """
    Full replay pipeline for HYP-013.

    Steps:
    1. Load ignition events from Morpheus trade ledger
    2. Generate control events
    3. For each event, load Databento data and compute pressure profile
    4. Analyze and compare real vs control
    5. Output results
    """
    if config is None:
        config = ReplayConfig()

    config.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HYP-013 REPLAY: Does pressure precede Morpheus ignition?")
    print("=" * 80)

    # Step 1: Load events
    print("\n[1/5] Loading ignition events from trade ledger...")
    events = load_ignition_events(config.reports_root, symbols)
    print(f"  Loaded: {len(events)} ignition events")

    if len(events) == 0:
        print("[ERROR] No events found. Check reports_root path.")
        return {"error": "No events found"}

    if max_events:
        events = events[:max_events]
        print(f"  Limited to: {max_events} events")

    # Show breakdown
    symbols_seen = set(e.symbol for e in events)
    dates_seen = set(e.entry_time.strftime("%Y-%m-%d") for e in events)
    print(f"  Symbols: {len(symbols_seen)} -> {sorted(symbols_seen)}")
    print(f"  Dates: {len(dates_seen)}")
    print(f"  Winners: {sum(1 for e in events if e.pnl > 0)}")
    print(f"  Losers: {sum(1 for e in events if e.pnl <= 0)}")

    # Step 2: Generate controls
    print("\n[2/5] Generating control events...")
    controls = generate_control_events(events, config)
    print(f"  Generated: {len(controls)} control events")

    # Step 3: Compute profiles
    print(f"\n[3/5] Computing pressure profiles (bar_sec={config.bar_sec}s, "
          f"pre={config.pre_window_sec}s, post={config.post_window_sec}s)...")

    real_profiles = []
    ctrl_profiles = []

    for i, evt in enumerate(events):
        if verbose and i % 50 == 0:
            print(f"  Processing real event {i+1}/{len(events)}: "
                  f"{evt.symbol} {evt.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")

        trades_df = load_raw_trades_around_event(
            config.databento_root, evt.symbol, evt.entry_time,
            config.pre_window_sec, config.post_window_sec,
        )
        if trades_df is None:
            continue

        profile = compute_pressure_profile(trades_df, evt, config, is_control=False)
        if profile is not None:
            real_profiles.append(profile)

    print(f"  Real profiles computed: {len(real_profiles)} / {len(events)}")

    for i, ctrl_evt in enumerate(controls):
        if verbose and i % 100 == 0:
            print(f"  Processing control {i+1}/{len(controls)}")

        trades_df = load_raw_trades_around_event(
            config.databento_root, ctrl_evt.symbol, ctrl_evt.entry_time,
            config.pre_window_sec, config.post_window_sec,
        )
        if trades_df is None:
            continue

        profile = compute_pressure_profile(trades_df, ctrl_evt, config, is_control=True)
        if profile is not None:
            ctrl_profiles.append(profile)

    print(f"  Control profiles computed: {len(ctrl_profiles)} / {len(controls)}")

    # Step 4: Analyze
    print("\n[4/5] Analyzing pressure profiles...")
    results = analyze_pressure_profiles(real_profiles, ctrl_profiles, config)

    # Step 5: Output
    print("\n[5/5] Writing results...")
    output_file = config.output_root / "hyp013_replay_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("HYP-013 RESULTS SUMMARY")
    print("=" * 80)

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return results

    rs = results.get("real_summary", {})
    if "peak_pressure_z" in rs:
        print(f"\n  REAL EVENTS (n={results['n_real']}):")
        print(f"    Peak |pressure_z| before entry:  mean={rs['peak_pressure_z']['mean']:.3f}  "
              f"median={rs['peak_pressure_z']['median']:.3f}")
        if "buildup_rate" in rs:
            print(f"    Pressure buildup rate:           mean={rs['buildup_rate']['mean']:.4f}")
        if "pressure_consistency" in rs:
            print(f"    Pressure consistency:             mean={rs['pressure_consistency']['mean']:.1%}")

    cs = results.get("control_summary", {})
    if "peak_pressure_z" in cs:
        print(f"\n  CONTROLS (n={results['n_control']}):")
        print(f"    Peak |pressure_z| before entry:  mean={cs['peak_pressure_z']['mean']:.3f}  "
              f"median={cs['peak_pressure_z']['median']:.3f}")

    lt = results.get("lead_time_analysis", {})
    if lt:
        print(f"\n  LEAD TIME ANALYSIS:")
        print(f"    Events with pressure precursor:  {lt['n_events_with_precursor']} "
              f"({lt['pct_events_with_precursor']:.1f}%)")
        print(f"    Mean lead time:   {lt['mean_lead_sec']:.1f}s")
        print(f"    Median lead time: {lt['median_lead_sec']:.1f}s")
        print(f"    > 5s lead:  {lt['pct_above_5sec']:.1f}%")
        print(f"    > 30s lead: {lt['pct_above_30sec']:.1f}%")
        print(f"    > 60s lead: {lt['pct_above_60sec']:.1f}%")

    st = results.get("statistical_tests", {})
    if st:
        print(f"\n  STATISTICAL SIGNIFICANCE:")
        for metric, test in st.items():
            sig = "***" if test["significant_01"] else ("*" if test["significant_05"] else "ns")
            print(f"    {metric}: diff={test['observed_diff']:.4f}  "
                  f"p={test['p_value']:.4f} {sig}  "
                  f"CI95=[{test['ci_95_lower']:.4f}, {test['ci_95_upper']:.4f}]")

    wl = results.get("winner_vs_loser", {})
    if wl:
        w = wl["winners"]
        l = wl["losers"]
        print(f"\n  WINNERS (n={w['n']}) vs LOSERS (n={l['n']}):")
        print(f"    Peak pressure:   winners={w['mean_peak_pressure_z']:.3f}  "
              f"losers={l['mean_peak_pressure_z']:.3f}")
        print(f"    Buildup rate:    winners={w['mean_buildup_rate']:.4f}  "
              f"losers={l['mean_buildup_rate']:.4f}")
        print(f"    With precursor:  winners={w['pct_with_precursor']:.1f}%  "
              f"losers={l['pct_with_precursor']:.1f}%")

    print("\n" + "=" * 80)

    return results


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HYP-013 Replay Engine")
    parser.add_argument("--reports", type=str, default=r"Z:\AI_BOT_DATA\reports",
                        help="Path to Morpheus reports directory")
    parser.add_argument("--databento", type=str,
                        default=r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades",
                        help="Path to Databento trade data")
    parser.add_argument("--output", type=str, default=r"C:\AI_Bot_Research\results",
                        help="Path to output results")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Filter to specific symbols")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limit number of events to process")
    parser.add_argument("--bar-sec", type=int, default=5,
                        help="Bar size in seconds (default: 5)")
    parser.add_argument("--pre-window", type=int, default=300,
                        help="Pre-entry window in seconds (default: 300)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    config = ReplayConfig(
        reports_root=Path(args.reports),
        databento_root=Path(args.databento),
        output_root=Path(args.output),
        bar_sec=args.bar_sec,
        pre_window_sec=args.pre_window,
    )

    results = run_replay(
        config=config,
        symbols=args.symbols,
        max_events=args.max_events,
        verbose=not args.quiet,
    )
