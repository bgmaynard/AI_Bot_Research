"""
RISK_APPROVED + Ignition Accelerator Coupling Study (v1)
=========================================================
Measures whether the ignition accelerator improves Morpheus's
74 v2-pass RISK_APPROVED signals by reducing entry lateness
while keeping Morpheus selectivity.

NOT evaluating ignition alone.
Evaluating: Morpheus signal + earlier price reference.

Usage:
  python -m ai.research.accel_coupling_study --date 2026-03-03
"""

import json
import sys
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE / "engine" / "output"
CACHE_DIR = BASE / "engine" / "cache" / "quotes"
COMMS_FILE = BASE / "comms" / "outbox_chatgpt.json"

PM_END_EPOCH = 1772548200.0  # 2026-03-03 14:30 UTC

# Sensitivity sweep parameters
# Lookback: how far BEFORE the signal we search for a preceding ignition
LOOKBACK_WINDOWS = [60, 180, 360]      # seconds to look backward from signal
ACCEL_OFFSETS    = [0.0, -0.20, -0.50] # % offset from ignition price
FILL_WINDOW      = 30                  # seconds to fill limit after ignition


# ── Data loading ───────────────────────────────────────────────────

def ts_to_epoch(date_str: str, ts_str: str) -> float:
    dt = datetime.strptime(f"{date_str} {ts_str}", "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def load_quotes_ffill(symbol: str) -> list[dict]:
    path = CACHE_DIR / f"{symbol}_quotes.json"
    with open(path) as f:
        data = json.load(f)
    quotes = data["quotes"]
    last_bid = None
    last_ask = None
    filled = []
    for q in quotes:
        bid = q.get("bid")
        ask = q.get("ask")
        if bid is not None:
            last_bid = bid
        if ask is not None:
            last_ask = ask
        filled.append({
            "epoch": q["epoch"],
            "bid": last_bid,
            "ask": last_ask,
            "last": q.get("last"),
            "mid": ((last_bid + last_ask) / 2) if last_bid and last_ask else q.get("last"),
        })
    return filled


def load_signals(date_str: str) -> list[dict]:
    """Load v2-pass signals merged with edge preservation exit data."""
    v2_path = OUTPUT_DIR / f"containment_v2_replay_{date_str}.json"
    ep_path = OUTPUT_DIR / f"edge_preservation_v2_{date_str}.json"

    with open(v2_path) as f:
        v2 = json.load(f)
    with open(ep_path) as f:
        ep = json.load(f)

    ep_by_num = {e["num"]: e for e in ep["signals"]}
    v2_pass = [s for s in v2["signals"] if s["v2_result"] == "PASS"]

    merged = []
    for s in v2_pass:
        num = s["num"]
        ep_sig = ep_by_num.get(num)
        if ep_sig is None:
            continue

        merged.append({
            "num": num,
            "symbol": s["symbol"],
            "timestamp": s["timestamp"],
            "epoch": ts_to_epoch(date_str, s["timestamp"]),
            "strategy": s["strategy"],
            "phase": s["phase"],
            "entry_price": s["entry_price"],
            "breakout_level": s.get("breakout_level", s["entry_price"]),
            "spread_pct": s["spread_pct"],
            "confidence": s.get("confidence", 0),
            "momentum_score": s.get("momentum_score", 0),
            # Baseline exit model from edge preservation
            "baseline_exit": ep_sig["exit_model"],
            "baseline_windows": ep_sig.get("windows", {}),
        })

    return merged


def load_ignition_events(date_str: str) -> dict:
    path = OUTPUT_DIR / f"ignition_events_{date_str}.json"
    with open(path) as f:
        return json.load(f)


# ── Streaming Ignition Detector (reused from accelerator module) ──

class StreamingIgnitionDetector:
    """Lightweight streaming detector: velocity + trend + freshness."""

    def __init__(self, vel_threshold=0.30, trend_threshold=0.50,
                 freshness_max_ms=1200.0):
        self.vel_thresh = vel_threshold
        self.trend_thresh = trend_threshold
        self.fresh_max = freshness_max_ms

    def scan_window(self, quotes: list[dict], start_epoch: float,
                    window_s: float) -> dict | None:
        """Scan quotes from start_epoch forward for up to window_s seconds.
        Returns first ignition event found, or None.
        """
        end_epoch = start_epoch + window_s
        buf = []

        for q in quotes:
            if q["epoch"] < start_epoch - 35.0:
                continue
            if q["epoch"] > end_epoch:
                break

            mid = q.get("mid")
            if mid is None or mid <= 0:
                continue

            buf.append(q)

            # Only evaluate after start_epoch (we buffer pre-start for lookback)
            if q["epoch"] < start_epoch:
                continue

            # Need 35s of buffer and at least 5 quotes
            window_buf = [b for b in buf if b["epoch"] >= q["epoch"] - 35.0]
            if len(window_buf) < 5:
                continue
            span = window_buf[-1]["epoch"] - window_buf[0]["epoch"]
            if span < 10.0:
                continue

            vel = self._velocity_30s(window_buf)
            trend = self._trend_persistence(window_buf)
            fresh = self._freshness_ms(window_buf)

            if abs(vel) >= self.vel_thresh and trend >= self.trend_thresh and fresh <= self.fresh_max:
                return {
                    "epoch": q["epoch"],
                    "price": mid,
                    "direction": "up" if vel > 0 else "down",
                    "velocity_30s": round(vel, 4),
                    "trend_persistence": round(trend, 4),
                    "freshness_ms": round(fresh, 1),
                }

        return None

    def _velocity_30s(self, buf: list) -> float:
        now = buf[-1]
        target = now["epoch"] - 30.0
        best = buf[0]
        for q in buf:
            if abs(q["epoch"] - target) < abs(best["epoch"] - target):
                best = q
        if best["mid"] is None or best["mid"] <= 0:
            return 0.0
        return ((now["mid"] - best["mid"]) / best["mid"]) * 100.0

    def _trend_persistence(self, buf: list) -> float:
        mids = [q["mid"] for q in buf if q["mid"] is not None]
        if len(mids) < 3:
            return 0.0
        overall = mids[-1] - mids[0]
        if overall == 0:
            return 0.0
        same = 0
        total = 0
        for i in range(1, len(mids)):
            d = mids[i] - mids[i - 1]
            if d == 0:
                continue
            total += 1
            if (d > 0 and overall > 0) or (d < 0 and overall < 0):
                same += 1
        return same / total if total > 0 else 0.0

    def _freshness_ms(self, buf: list) -> float:
        if len(buf) < 2:
            return 99999.0
        recent = buf[-min(10, len(buf)):]
        gaps = [(recent[i]["epoch"] - recent[i - 1]["epoch"]) * 1000.0
                for i in range(1, len(recent))]
        return statistics.mean(gaps) if gaps else 99999.0


# ── Exit model simulation ─────────────────────────────────────────

def simulate_exit(quotes: list[dict], fill_epoch: float, fill_price: float,
                  direction: str = "up") -> dict:
    """Hard stop -1%, trail +0.8%/0.4%, time exit 300s."""
    STOP = -1.0
    TRAIL_ACT = 0.8
    TRAIL_DIST = 0.4
    TIME_EXIT = 300.0

    deadline = fill_epoch + TIME_EXIT
    peak_ret = 0.0
    trough_ret = 0.0
    trail_active = False
    trail_peak = 0.0
    d = 1.0 if direction == "up" else -1.0

    last_q = None
    for q in quotes:
        if q["epoch"] <= fill_epoch:
            continue
        mid = q.get("mid")
        if mid is None:
            continue

        raw = ((mid - fill_price) / fill_price) * 100.0
        dr = raw * d

        if dr > peak_ret:
            peak_ret = dr
        if dr < trough_ret:
            trough_ret = dr

        if dr <= STOP:
            return {"exit_type": "STOP", "exit_return_pct": round(dr, 4),
                    "exit_price": mid, "exit_time_sec": round(q["epoch"] - fill_epoch, 2),
                    "trail_activated": trail_active, "peak_pct": round(peak_ret, 4),
                    "trough_pct": round(trough_ret, 4)}

        if dr >= TRAIL_ACT:
            trail_active = True
            if dr > trail_peak:
                trail_peak = dr

        if trail_active and (trail_peak - dr) >= TRAIL_DIST:
            return {"exit_type": "TRAIL", "exit_return_pct": round(dr, 4),
                    "exit_price": mid, "exit_time_sec": round(q["epoch"] - fill_epoch, 2),
                    "trail_activated": True, "peak_pct": round(peak_ret, 4),
                    "trough_pct": round(trough_ret, 4)}

        if q["epoch"] >= deadline:
            return {"exit_type": "TIME", "exit_return_pct": round(dr, 4),
                    "exit_price": mid, "exit_time_sec": round(q["epoch"] - fill_epoch, 2),
                    "trail_activated": trail_active, "peak_pct": round(peak_ret, 4),
                    "trough_pct": round(trough_ret, 4)}

        last_q = q

    if last_q and last_q["mid"] is not None:
        raw = ((last_q["mid"] - fill_price) / fill_price) * 100.0
        dr = raw * d
        return {"exit_type": "EOD", "exit_return_pct": round(dr, 4),
                "exit_price": last_q["mid"],
                "exit_time_sec": round(last_q["epoch"] - fill_epoch, 2),
                "trail_activated": trail_active, "peak_pct": round(peak_ret, 4),
                "trough_pct": round(trough_ret, 4)}

    return {"exit_type": "NO_DATA", "exit_return_pct": 0.0, "exit_price": fill_price,
            "exit_time_sec": 0, "trail_activated": False, "peak_pct": 0.0, "trough_pct": 0.0}


# ── Fill simulation ───────────────────────────────────────────────

def try_limit_fill(quotes: list[dict], after_epoch: float, limit_price: float,
                   window_s: float, direction: str = "up") -> tuple:
    """Try to fill a limit order.
    For 'up' direction: buy limit = fill when mid/ask <= limit.
    For 0% offset: limit >= current mid, so fills immediately.
    Returns (fill_epoch, fill_price) or (None, None).
    """
    deadline = after_epoch + window_s
    for q in quotes:
        if q["epoch"] <= after_epoch:
            continue
        if q["epoch"] > deadline:
            break
        mid = q.get("mid")
        if mid is None:
            continue

        if direction == "up":
            if mid <= limit_price:
                return q["epoch"], mid
        else:
            if mid >= limit_price:
                return q["epoch"], mid

    return None, None


# ── Core analysis ──────────────────────────────────────────────────

def find_preceding_ignition(ign_events: list[dict], sig_epoch: float,
                            lookback_s: float) -> dict | None:
    """Find the closest ignition event BEFORE signal within lookback window.

    Ignition precedes the Morpheus signal (median ~180s before).
    We search backward from the signal time to find the nearest ignition.
    Returns the closest (most recent) ignition within the window.
    """
    earliest = sig_epoch - lookback_s
    best = None
    best_dist = float("inf")

    for evt in ign_events:
        e = evt["epoch"]
        if e > sig_epoch:
            break  # events are sorted; past signal time
        if e < earliest:
            continue
        dist = sig_epoch - e
        if dist < best_dist:
            best_dist = dist
            best = evt

    return best


def run_coupling(signals: list[dict], quote_cache: dict[str, list],
                 ign_by_symbol: dict[str, list],
                 lookback_s: float, offset: float) -> list[dict]:
    """Run coupled analysis for one (lookback, offset) combination.

    For each Morpheus signal:
      1. Look BACKWARD to find the ignition that preceded this signal
      2. Compute accelerated limit price at ignition_price * (1 + offset%)
      3. Try to fill that limit within FILL_WINDOW seconds after ignition
      4. If filled, simulate exit model and compare to baseline
    """
    results = []

    for sig in signals:
        sym = sig["symbol"]
        quotes = quote_cache.get(sym)
        if quotes is None:
            continue

        sig_epoch = sig["epoch"]
        entry_price = sig["entry_price"]
        baseline_exit = sig["baseline_exit"]

        rec = {
            "num": sig["num"],
            "symbol": sym,
            "timestamp": sig["timestamp"],
            "strategy": sig["strategy"],
            "phase": sig["phase"],
            "entry_price": entry_price,
            "lookback_s": lookback_s,
            "offset_pct": offset,
            # Baseline
            "baseline_exit_type": baseline_exit["exit_type"],
            "baseline_exit_return": baseline_exit["exit_return_pct"],
            "baseline_exit_time_sec": baseline_exit.get("exit_time_sec", None),
            "baseline_peak_pct": baseline_exit.get("peak_return_pct",
                                                    baseline_exit.get("peak_pct", 0)),
        }

        # Step 1: Find preceding ignition within lookback window
        sym_ignitions = ign_by_symbol.get(sym, [])
        ign = find_preceding_ignition(sym_ignitions, sig_epoch, lookback_s)

        if ign is None:
            rec["ignition_found"] = False
            rec["accel_outcome"] = "NO_IGNITION"
            rec["accel_filled"] = False
            rec["accel_exit_return"] = None
            rec["entry_lateness_before"] = None
            rec["entry_lateness_after"] = None
            rec["return_delta"] = None
            results.append(rec)
            continue

        ign_epoch = ign["epoch"]
        ign_price = ign["price"]
        entry_lateness = sig_epoch - ign_epoch  # how late Morpheus is vs ignition

        rec["ignition_found"] = True
        rec["ignition_epoch"] = ign_epoch
        rec["ignition_time"] = ign["time"]
        rec["ignition_price"] = ign_price
        rec["entry_lateness_before"] = round(entry_lateness, 1)
        rec["ignition_velocity"] = ign.get("velocity_3s", 0)

        # Step 2: Compute accelerated limit price
        # For momentum breakouts (long), offset is negative = buy below ignition price
        accel_limit = ign_price * (1.0 + offset / 100.0)
        rec["accel_limit_price"] = round(accel_limit, 6)

        # Step 3: Try to fill within FILL_WINDOW after ignition
        fill_epoch, fill_price = try_limit_fill(
            quotes, ign_epoch, accel_limit, FILL_WINDOW, direction="up"
        )

        if fill_epoch is None:
            rec["accel_outcome"] = "IGNITION_NO_FILL"
            rec["accel_filled"] = False
            rec["accel_exit_return"] = None
            rec["entry_lateness_after"] = round(entry_lateness, 1)  # unchanged
            rec["return_delta"] = None
            results.append(rec)
            continue

        rec["accel_outcome"] = "FILLED"
        rec["accel_filled"] = True
        rec["accel_fill_epoch"] = fill_epoch
        rec["accel_fill_price"] = fill_price
        rec["accel_fill_delay_from_ignition"] = round(fill_epoch - ign_epoch, 2)

        # Entry lateness improvement
        new_lateness = fill_epoch - ign_epoch  # time from ignition to accel fill
        rec["entry_lateness_after"] = round(new_lateness, 1)
        rec["lateness_improvement_sec"] = round(entry_lateness - new_lateness, 1)

        # Slippage: difference between ignition price and fill price
        rec["slippage_pct"] = round(
            ((fill_price - ign_price) / ign_price) * 100.0, 4
        )

        # Price improvement: accel fill vs baseline entry price
        rec["price_improvement_pct"] = round(
            ((entry_price - fill_price) / entry_price) * 100.0, 4
        )

        # Step 4: Simulate exit from accel fill
        exit_res = simulate_exit(quotes, fill_epoch, fill_price, direction="up")
        rec["accel_exit_type"] = exit_res["exit_type"]
        rec["accel_exit_return"] = exit_res["exit_return_pct"]
        rec["accel_exit_time_sec"] = exit_res["exit_time_sec"]
        rec["accel_peak_pct"] = exit_res["peak_pct"]
        rec["accel_trough_pct"] = exit_res["trough_pct"]
        rec["accel_trail_activated"] = exit_res["trail_activated"]

        # Delta vs baseline
        if baseline_exit["exit_return_pct"] is not None:
            rec["return_delta"] = round(
                exit_res["exit_return_pct"] - baseline_exit["exit_return_pct"], 4
            )
        else:
            rec["return_delta"] = None

        results.append(rec)

    return results


# ── Aggregate statistics ───────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    """Compute aggregate stats for one (window, offset) combination."""
    n = len(results)
    if n == 0:
        return {}

    ign_found = [r for r in results if r["ignition_found"]]
    filled = [r for r in results if r["accel_filled"]]
    no_ign = [r for r in results if not r["ignition_found"]]
    ign_no_fill = [r for r in results if r["ignition_found"] and not r["accel_filled"]]

    # Baseline stats (all signals)
    bl_returns = [r["baseline_exit_return"] for r in results
                  if r["baseline_exit_return"] is not None]
    bl_wins = [r for r in bl_returns if r > 0]
    bl_losses = [r for r in bl_returns if r < 0]

    # Accel stats (filled only)
    ac_returns = [r["accel_exit_return"] for r in filled
                  if r["accel_exit_return"] is not None]
    ac_wins = [r for r in ac_returns if r > 0]
    ac_losses = [r for r in ac_returns if r < 0]

    # Deltas (filled signals only — compare same signals)
    filled_bl = [r["baseline_exit_return"] for r in filled
                 if r["baseline_exit_return"] is not None]
    filled_bl_wins = [r for r in filled_bl if r > 0]
    filled_bl_losses = [r for r in filled_bl if r < 0]

    def pf(wins, losses):
        if not losses or sum(losses) == 0:
            return 999.0 if wins else 0.0
        return sum(wins) / abs(sum(losses))

    def wr(wins, total):
        return len(wins) / len(total) * 100.0 if total else 0.0

    s = {
        "n_signals": n,
        "n_ignition_found": len(ign_found),
        "n_filled": len(filled),
        "n_no_ignition": len(no_ign),
        "n_ignition_no_fill": len(ign_no_fill),
        "ignition_rate": round(len(ign_found) / n * 100, 1),
        "fill_rate": round(len(filled) / n * 100, 1),
        "miss_rate": round((len(no_ign) + len(ign_no_fill)) / n * 100, 1),
        # Baseline (all 74)
        "baseline_wr": round(wr(bl_wins, bl_returns), 1),
        "baseline_avg_return": round(statistics.mean(bl_returns), 4) if bl_returns else 0,
        "baseline_pf": round(pf(bl_wins, bl_losses), 3),
        # Accel (filled only)
        "accel_wr": round(wr(ac_wins, ac_returns), 1) if ac_returns else 0,
        "accel_avg_return": round(statistics.mean(ac_returns), 4) if ac_returns else 0,
        "accel_pf": round(pf(ac_wins, ac_losses), 3) if ac_returns else 0,
        # Same-signal baseline (filled signals baseline performance)
        "same_signal_baseline_wr": round(wr(filled_bl_wins, filled_bl), 1) if filled_bl else 0,
        "same_signal_baseline_avg": round(statistics.mean(filled_bl), 4) if filled_bl else 0,
        "same_signal_baseline_pf": round(pf(filled_bl_wins, filled_bl_losses), 3) if filled_bl else 0,
    }

    # Deltas (accel vs same-signal baseline)
    if ac_returns and filled_bl:
        s["wr_delta"] = round(s["accel_wr"] - s["same_signal_baseline_wr"], 1)
        s["avg_return_delta"] = round(s["accel_avg_return"] - s["same_signal_baseline_avg"], 4)
        s["pf_delta"] = round(s["accel_pf"] - s["same_signal_baseline_pf"], 3)
    else:
        s["wr_delta"] = 0
        s["avg_return_delta"] = 0
        s["pf_delta"] = 0

    # Entry delay stats
    lateness_before = [r["entry_lateness_before"] for r in filled
                       if r.get("entry_lateness_before") is not None]
    lateness_after = [r["entry_lateness_after"] for r in filled
                      if r.get("entry_lateness_after") is not None]
    improvements = [r["lateness_improvement_sec"] for r in filled
                    if r.get("lateness_improvement_sec") is not None]
    price_imps = [r["price_improvement_pct"] for r in filled
                  if r.get("price_improvement_pct") is not None]

    if lateness_before:
        s["median_lateness_before"] = round(statistics.median(lateness_before), 1)
    if lateness_after:
        s["median_lateness_after"] = round(statistics.median(lateness_after), 1)
    if improvements:
        s["median_lateness_improvement"] = round(statistics.median(improvements), 1)
    if price_imps:
        s["median_price_improvement_pct"] = round(statistics.median(price_imps), 4)

    return s


def compute_symbol_breakdown(results: list[dict]) -> dict:
    """Breakdown by symbol for one configuration."""
    by_sym = defaultdict(list)
    for r in results:
        by_sym[r["symbol"]].append(r)

    breakdown = {}
    for sym in sorted(by_sym.keys()):
        recs = by_sym[sym]
        filled = [r for r in recs if r["accel_filled"]]

        bl_rets = [r["baseline_exit_return"] for r in recs if r["baseline_exit_return"] is not None]
        ac_rets = [r["accel_exit_return"] for r in filled if r["accel_exit_return"] is not None]

        bl_wins = [r for r in bl_rets if r > 0]
        bl_losses = [r for r in bl_rets if r < 0]
        ac_wins = [r for r in ac_rets if r > 0]
        ac_losses = [r for r in ac_rets if r < 0]

        def pf(w, l):
            if not l or sum(l) == 0:
                return 999.0 if w else 0.0
            return sum(w) / abs(sum(l))

        breakdown[sym] = {
            "n_signals": len(recs),
            "n_filled": len(filled),
            "fill_rate": round(len(filled) / len(recs) * 100, 1) if recs else 0,
            "baseline_wr": round(len(bl_wins) / len(bl_rets) * 100, 1) if bl_rets else 0,
            "baseline_avg": round(statistics.mean(bl_rets), 4) if bl_rets else 0,
            "baseline_pf": round(pf(bl_wins, bl_losses), 3),
            "accel_wr": round(len(ac_wins) / len(ac_rets) * 100, 1) if ac_rets else 0,
            "accel_avg": round(statistics.mean(ac_rets), 4) if ac_rets else 0,
            "accel_pf": round(pf(ac_wins, ac_losses), 3),
        }

    return breakdown


# ── Report generation ──────────────────────────────────────────────

def generate_report(date_str: str, sweep_results: dict, signals: list) -> str:
    lines = []
    lines.append(f"# RISK_APPROVED + Ignition Accelerator Coupling Report - {date_str}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append("**Method:** Couple v1 ignition detector with 74 v2-pass RISK_APPROVED signals")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Baseline reference
    bl_rets = [s["baseline_exit"]["exit_return_pct"] for s in signals
               if s["baseline_exit"]["exit_return_pct"] is not None]
    bl_wins = [r for r in bl_rets if r > 0]
    bl_losses = [r for r in bl_rets if r < 0]
    bl_wr = len(bl_wins) / len(bl_rets) * 100 if bl_rets else 0
    bl_avg = statistics.mean(bl_rets) if bl_rets else 0
    bl_pf = (sum(bl_wins) / abs(sum(bl_losses))) if bl_losses and sum(bl_losses) != 0 else 0

    lines.append("## Baseline Reference (74 v2-pass signals, current entry)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Signals | {len(signals)} |")
    lines.append(f"| Win Rate | {bl_wr:.1f}% |")
    lines.append(f"| Avg Return | {bl_avg:+.4f}% |")
    lines.append(f"| Profit Factor | {bl_pf:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sensitivity sweep table
    lines.append("## Sensitivity Sweep: Lookback Window x Offset")
    lines.append("")
    lines.append("| Lookback | Offset | Ign Found | Filled | Fill% | "
                 "Accel WR | Accel Avg | Accel PF | "
                 "Same-Sig BL WR | Same-Sig BL Avg | Same-Sig BL PF | "
                 "WR Delta | Avg Delta | PF Delta |")
    lines.append("|----------|--------|-----------|--------|-------|"
                 "----------|-----------|----------|"
                 "----------------|-----------------|----------------|"
                 "----------|-----------|----------|")

    best_key = None
    best_pf_delta = -999

    for w in LOOKBACK_WINDOWS:
        for o in ACCEL_OFFSETS:
            key = (w, o)
            if key not in sweep_results:
                continue
            stats = sweep_results[key]["stats"]

            wr_d = stats.get("wr_delta", 0)
            avg_d = stats.get("avg_return_delta", 0)
            pf_d = stats.get("pf_delta", 0)

            lines.append(
                f"| {w}s | {o:+.2f}% | {stats['n_ignition_found']}/{stats['n_signals']} | "
                f"{stats['n_filled']} | {stats['fill_rate']:.1f}% | "
                f"{stats['accel_wr']:.1f}% | {stats['accel_avg_return']:+.4f}% | "
                f"{stats['accel_pf']:.3f} | "
                f"{stats['same_signal_baseline_wr']:.1f}% | "
                f"{stats['same_signal_baseline_avg']:+.4f}% | "
                f"{stats['same_signal_baseline_pf']:.3f} | "
                f"{wr_d:+.1f}pp | {avg_d:+.4f}% | {pf_d:+.3f} |"
            )

            # Track best by PF delta (among configs with >= 15% fill rate)
            if stats["fill_rate"] >= 15 and pf_d > best_pf_delta:
                best_pf_delta = pf_d
                best_key = key

    lines.append("")

    # Best configuration
    if best_key:
        bw, bo = best_key
        bs = sweep_results[best_key]["stats"]
        lines.append("## Best Configuration")
        lines.append("")
        lines.append(f"| Parameter | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Lookback window | {bw}s |")
        lines.append(f"| Offset | {bo:+.2f}% |")
        lines.append(f"| Fill rate | {bs['fill_rate']:.1f}% ({bs['n_filled']}/{bs['n_signals']}) |")
        lines.append(f"| Accel WR | {bs['accel_wr']:.1f}% (vs {bs['same_signal_baseline_wr']:.1f}% baseline) |")
        lines.append(f"| Accel Avg Return | {bs['accel_avg_return']:+.4f}% (vs {bs['same_signal_baseline_avg']:+.4f}%) |")
        lines.append(f"| Accel PF | {bs['accel_pf']:.3f} (vs {bs['same_signal_baseline_pf']:.3f}) |")
        lines.append(f"| PF Delta | {bs.get('pf_delta', 0):+.3f} |")
        if bs.get("median_lateness_before") is not None:
            lines.append(f"| Median entry lateness BEFORE | {bs['median_lateness_before']:.1f}s |")
        if bs.get("median_lateness_after") is not None:
            lines.append(f"| Median entry lateness AFTER | {bs['median_lateness_after']:.1f}s |")
        if bs.get("median_lateness_improvement") is not None:
            lines.append(f"| Median lateness improvement | {bs['median_lateness_improvement']:.1f}s |")
        if bs.get("median_price_improvement_pct") is not None:
            lines.append(f"| Median price improvement | {bs['median_price_improvement_pct']:+.4f}% |")
        lines.append("")

    # Per-symbol breakdown for best config
    if best_key and best_key in sweep_results:
        sym_bd = sweep_results[best_key].get("symbol_breakdown", {})
        if sym_bd:
            lines.append(f"## Per-Symbol Breakdown (lookback={best_key[0]}s, offset={best_key[1]:+.2f}%)")
            lines.append("")
            lines.append("| Symbol | Signals | Filled | Fill% | BL WR | BL Avg | BL PF | "
                         "Accel WR | Accel Avg | Accel PF |")
            lines.append("|--------|---------|--------|-------|-------|--------|-------|"
                         "----------|-----------|----------|")
            for sym in sorted(sym_bd.keys(), key=lambda s: -sym_bd[s]["n_signals"]):
                sb = sym_bd[sym]
                lines.append(
                    f"| {sym} | {sb['n_signals']} | {sb['n_filled']} | {sb['fill_rate']:.1f}% | "
                    f"{sb['baseline_wr']:.1f}% | {sb['baseline_avg']:+.4f}% | {sb['baseline_pf']:.3f} | "
                    f"{sb['accel_wr']:.1f}% | {sb['accel_avg']:+.4f}% | {sb['accel_pf']:.3f} |"
                )
            lines.append("")

    # Strategy breakdown for best config
    if best_key and best_key in sweep_results:
        recs = sweep_results[best_key]["results"]
        by_strat = defaultdict(list)
        for r in recs:
            by_strat[r["strategy"]].append(r)

        if len(by_strat) > 1:
            lines.append(f"## Per-Strategy Breakdown (lookback={best_key[0]}s, offset={best_key[1]:+.2f}%)")
            lines.append("")
            lines.append("| Strategy | Signals | Filled | BL Avg | Accel Avg | Delta |")
            lines.append("|----------|---------|--------|--------|-----------|-------|")
            for strat, recs_s in sorted(by_strat.items()):
                filled_s = [r for r in recs_s if r["accel_filled"]]
                bl_r = [r["baseline_exit_return"] for r in recs_s if r["baseline_exit_return"] is not None]
                ac_r = [r["accel_exit_return"] for r in filled_s if r["accel_exit_return"] is not None]
                bl_a = statistics.mean(bl_r) if bl_r else 0
                ac_a = statistics.mean(ac_r) if ac_r else 0
                d = ac_a - bl_a if ac_r else 0
                lines.append(f"| {strat} | {len(recs_s)} | {len(filled_s)} | "
                             f"{bl_a:+.4f}% | {ac_a:+.4f}% | {d:+.4f}% |")
            lines.append("")

    # Failure modes
    lines.append("## Failure Mode Analysis")
    lines.append("")

    # Use 120s/0% as the main config for failure analysis
    main_key = best_key or (120, 0.0)
    if main_key in sweep_results:
        recs = sweep_results[main_key]["results"]
        no_ign = [r for r in recs if not r["ignition_found"]]
        ign_no_fill = [r for r in recs if r["ignition_found"] and not r["accel_filled"]]
        filled = [r for r in recs if r["accel_filled"]]
        stopped = [r for r in filled if r.get("accel_exit_type") == "STOP"]

        lines.append(f"### Configuration: lookback={main_key[0]}s, offset={main_key[1]:+.2f}%")
        lines.append("")
        lines.append("| Failure Mode | Count | % of 74 | Description |")
        lines.append("|-------------|-------|---------|-------------|")
        lines.append(f"| No preceding ignition | {len(no_ign)} | {len(no_ign)/74*100:.1f}% | "
                     f"No ignition event within {main_key[0]}s before signal |")
        lines.append(f"| Ignition but no fill | {len(ign_no_fill)} | {len(ign_no_fill)/74*100:.1f}% | "
                     f"Price didn't touch limit within {FILL_WINDOW}s after ignition |")
        lines.append(f"| Filled but stopped out | {len(stopped)} | {len(stopped)/74*100:.1f}% | "
                     f"Entry filled but hit -1.0% stop |")
        lines.append(f"| Successfully filled | {len(filled)} | {len(filled)/74*100:.1f}% | "
                     f"Accelerated entry filled and completed exit |")
        lines.append("")

        # Which symbols have the worst miss rates?
        if no_ign:
            miss_syms = defaultdict(int)
            for r in no_ign:
                miss_syms[r["symbol"]] += 1
            lines.append("**No-ignition signals by symbol:**")
            for sym, c in sorted(miss_syms.items(), key=lambda x: -x[1]):
                total_sym = sum(1 for r in recs if r["symbol"] == sym)
                lines.append(f"- {sym}: {c}/{total_sym} missed ({c/total_sym*100:.0f}%)")
            lines.append("")

    # Conclusions
    lines.append("## Key Findings")
    lines.append("")

    if best_key:
        bs = sweep_results[best_key]["stats"]
        improves = bs.get("pf_delta", 0) > 0

        lines.append(f"1. **Does coupling improve PF vs baseline?** "
                     f"{'YES' if improves else 'NO'} - "
                     f"PF delta = {bs.get('pf_delta', 0):+.3f} "
                     f"(baseline {bs['same_signal_baseline_pf']:.3f} vs accel {bs['accel_pf']:.3f})")

        if improves:
            lines.append(f"2. **Recommended conservative defaults:**")
            lines.append(f"   - `ignition_accelerator_enabled`: true")
            lines.append(f"   - `ignition_accelerator_offset_pct`: {best_key[1]}")
            lines.append(f"   - `ignition_lookback_seconds`: {best_key[0]}")
            lines.append(f"   - `fill_window_seconds`: {FILL_WINDOW}")
            lines.append(f"   - Apply ONLY when spread < 0.9% (PM) / 0.6% (RTH)")
            lines.append(f"   - Safety: accelerator only modifies entry_reference when "
                         f"signal already passed Morpheus scoring/gates")
        else:
            lines.append(f"2. **Recommended conservative defaults:** "
                         f"Do NOT enable accelerator yet. PF does not improve.")
            lines.append(f"   - Need more data days or different detector thresholds")

        miss_rate = bs.get("miss_rate", 0)
        lines.append(f"3. **Failure modes:**")
        lines.append(f"   - Miss rate: {miss_rate:.1f}% "
                     f"({bs['n_signals'] - bs['n_filled']}/{bs['n_signals']} signals)")
        lines.append(f"   - Primary failure: no ignition detected within window "
                     f"({bs['n_no_ignition']}/{bs['n_signals']})")
        if bs.get("n_ignition_no_fill", 0) > 0:
            lines.append(f"   - Secondary failure: ignition found but limit not filled "
                         f"({bs['n_ignition_no_fill']}/{bs['n_signals']})")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This study is research-only. No production changes applied.*")

    return "\n".join(lines)


def update_comms(date_str: str, sweep_results: dict, best_key: tuple | None):
    """Append comms message."""
    with open(COMMS_FILE, encoding="utf-8") as f:
        comms = json.load(f)

    summary = ["RISK_APPROVED + IGNITION ACCELERATOR COUPLING STUDY COMPLETE\n"]
    summary.append("== SENSITIVITY SWEEP (lookback x offset) ==")
    for w in LOOKBACK_WINDOWS:
        for o in ACCEL_OFFSETS:
            key = (w, o)
            if key not in sweep_results:
                continue
            s = sweep_results[key]["stats"]
            summary.append(
                f"  LB={w}s, offset={o:+.2f}%: "
                f"fill={s['fill_rate']:.1f}%, "
                f"accel_WR={s['accel_wr']:.1f}%, "
                f"accel_PF={s['accel_pf']:.3f}, "
                f"PF_delta={s.get('pf_delta', 0):+.3f}"
            )

    if best_key:
        bs = sweep_results[best_key]["stats"]
        improves = bs.get("pf_delta", 0) > 0
        summary.append(f"\n== BEST CONFIG: lookback={best_key[0]}s, offset={best_key[1]:+.2f}% ==")
        summary.append(f"  Fill rate: {bs['fill_rate']:.1f}%")
        summary.append(f"  Accel PF: {bs['accel_pf']:.3f} (baseline same-signal: {bs['same_signal_baseline_pf']:.3f})")
        summary.append(f"  PF delta: {bs.get('pf_delta', 0):+.3f}")
        summary.append(f"  Coupling improves PF: {'YES' if improves else 'NO'}")
        if improves:
            summary.append(f"\n== RECOMMENDED CONSERVATIVE DEFAULTS ==")
            summary.append(f"  ignition_accelerator_enabled: true")
            summary.append(f"  offset: {best_key[1]}%")
            summary.append(f"  ignition_lookback: {best_key[0]}s")
            summary.append(f"  fill_window: {FILL_WINDOW}s")
            summary.append(f"  Spread gates: PM < 0.9%, RTH < 0.6%")
        else:
            summary.append(f"\n== RECOMMENDATION: DO NOT ENABLE YET ==")
            summary.append(f"  Need more data days or tuned detector thresholds")

    summary.append(f"\n== FAILURE MODES ==")
    if best_key and best_key in sweep_results:
        bs = sweep_results[best_key]["stats"]
        summary.append(f"  No ignition: {bs['n_no_ignition']}/{bs['n_signals']} signals")
        summary.append(f"  Ignition no fill: {bs.get('n_ignition_no_fill', 0)}/{bs['n_signals']} signals")
        summary.append(f"  Miss rate: {bs['miss_rate']:.1f}%")

    summary.append(f"\n== FILES ==")
    summary.append(f"  JSON: engine/output/risk_approved_accel_coupling_{date_str}.json")
    summary.append(f"  Report: engine/output/risk_approved_accel_coupling_report_{date_str}.md")
    summary.append("\nAccelerator only modifies entry_reference when signal already passed Morpheus scoring/gates.")
    summary.append("Production remains frozen. All analysis is read-only research.")

    msg = {
        "id": f"msg_chatgpt_{date_str.replace('-', '')}_011",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": "claude",
        "to": "chatgpt",
        "type": "research_result",
        "subject": f"Coupling Study ({date_str}): Morpheus signals + ignition accelerator",
        "body": "\n".join(summary),
        "references": [
            f"engine/output/risk_approved_accel_coupling_{date_str}.json",
            f"engine/output/risk_approved_accel_coupling_report_{date_str}.md",
        ],
    }

    comms["messages"].append(msg)
    with open(COMMS_FILE, "w", encoding="utf-8") as f:
        json.dump(comms, f, indent=2)
    print(f"  Appended msg_011 to comms outbox")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RISK_APPROVED + Ignition Accelerator Coupling Study")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date_str = args.date

    print(f"RISK_APPROVED + Ignition Accelerator Coupling Study - {date_str}")
    print("=" * 70)

    # Load signals
    signals = load_signals(date_str)
    print(f"\nLoaded {len(signals)} v2-pass signals")

    # Load quote caches
    symbols = sorted(set(s["symbol"] for s in signals))
    print(f"Symbols: {', '.join(symbols)}")
    quote_cache = {}
    for sym in symbols:
        try:
            quote_cache[sym] = load_quotes_ffill(sym)
            print(f"  {sym}: {len(quote_cache[sym])} quotes")
        except FileNotFoundError:
            print(f"  {sym}: MISSING - signals for this symbol will be skipped")

    # Load pre-computed ignition events (backward matching)
    ign_data = load_ignition_events(date_str)
    ign_by_symbol = {}
    total_ign = 0
    for sym, sym_data in ign_data["events_per_symbol"].items():
        events = sym_data["events"]
        # Sort by epoch for binary-search-style lookups
        events.sort(key=lambda e: e["epoch"])
        ign_by_symbol[sym] = events
        total_ign += len(events)
    print(f"\nLoaded {total_ign} pre-computed ignition events across "
          f"{len(ign_by_symbol)} symbols")

    # Detector (kept for reference but backward matching is primary)
    detector = StreamingIgnitionDetector()

    # Run sensitivity sweep
    sweep_results = {}
    for w in LOOKBACK_WINDOWS:
        for o in ACCEL_OFFSETS:
            print(f"\n--- Lookback={w}s, Offset={o:+.2f}% ---")
            results = run_coupling(signals, quote_cache, ign_by_symbol, w, o)

            stats = compute_stats(results)
            sym_bd = compute_symbol_breakdown(results)

            sweep_results[(w, o)] = {
                "results": results,
                "stats": stats,
                "symbol_breakdown": sym_bd,
            }

            print(f"  Ignition found: {stats['n_ignition_found']}/{stats['n_signals']} "
                  f"({stats['ignition_rate']:.1f}%)")
            print(f"  Filled: {stats['n_filled']} ({stats['fill_rate']:.1f}%)")
            if stats["n_filled"] > 0:
                print(f"  Accel: WR={stats['accel_wr']:.1f}%, "
                      f"avg={stats['accel_avg_return']:+.4f}%, "
                      f"PF={stats['accel_pf']:.3f}")
                print(f"  Same-signal BL: WR={stats['same_signal_baseline_wr']:.1f}%, "
                      f"avg={stats['same_signal_baseline_avg']:+.4f}%, "
                      f"PF={stats['same_signal_baseline_pf']:.3f}")
                print(f"  Deltas: WR={stats['wr_delta']:+.1f}pp, "
                      f"avg={stats['avg_return_delta']:+.4f}%, "
                      f"PF={stats['pf_delta']:+.3f}")

    # Find best configuration
    best_key = None
    best_pf_delta = -999
    for key, data in sweep_results.items():
        s = data["stats"]
        if s["fill_rate"] >= 15 and s.get("pf_delta", -999) > best_pf_delta:
            best_pf_delta = s["pf_delta"]
            best_key = key

    # Save per-signal JSON (for best config)
    if best_key:
        all_signal_data = []
        for r in sweep_results[best_key]["results"]:
            # Clean for JSON serialization
            clean = {k: v for k, v in r.items() if v is not None}
            all_signal_data.append(clean)
    else:
        # Save 120s/0% as default
        all_signal_data = []
        dk = (120, 0.0)
        if dk in sweep_results:
            for r in sweep_results[dk]["results"]:
                clean = {k: v for k, v in r.items() if v is not None}
                all_signal_data.append(clean)

    json_path = OUTPUT_DIR / f"risk_approved_accel_coupling_{date_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "risk_approved_accel_coupling",
            "best_config": {"window": best_key[0], "offset": best_key[1]} if best_key else None,
            "sweep_summary": {
                f"LB{w}_O{o}": sweep_results[(w, o)]["stats"]
                for w in LOOKBACK_WINDOWS for o in ACCEL_OFFSETS
                if (w, o) in sweep_results
            },
            "signals": all_signal_data,
        }, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Generate report
    report = generate_report(date_str, sweep_results, signals)
    report_path = OUTPUT_DIR / f"risk_approved_accel_coupling_report_{date_str}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Comms
    update_comms(date_str, sweep_results, best_key)

    # Final
    print("\n" + "=" * 70)
    print("COUPLING STUDY COMPLETE")
    print("=" * 70)
    if best_key:
        bs = sweep_results[best_key]["stats"]
        improves = bs.get("pf_delta", 0) > 0
        print(f"\n  Best config: lookback={best_key[0]}s, offset={best_key[1]:+.2f}%")
        print(f"  PF delta: {bs.get('pf_delta', 0):+.3f}")
        print(f"  Coupling improves PF: {'YES' if improves else 'NO'}")


if __name__ == "__main__":
    main()
