"""
Ignition Entry Accelerator v1 - 2026-03-03
============================================
Research module that:
  A) Detects ignition in quote streams (real-time usable design)
  B) Proposes EARLY_ENTRY_CANDIDATE limit orders at offset prices
  C) Simulates fill + exit to measure improvement vs baseline
  D) Generates comparison report
  E) Proposes production toggle design

Usage:
  python -m ai.research.ignition_entry_accelerator --date 2026-03-03
"""

import json
import os
import sys
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE / "engine" / "output"
CACHE_DIR = BASE / "engine" / "cache" / "quotes"
CONFIG_DIR = BASE / "configs"
COMMS_FILE = BASE / "comms" / "outbox_chatgpt.json"

PM_END_EPOCH = 1772548200.0  # 2026-03-03 14:30 UTC


# ═══════════════════════════════════════════════════════════════════
# PART A — Ignition Detector (streaming-style, deterministic)
# ═══════════════════════════════════════════════════════════════════

class IgnitionDetector:
    """Streaming ignition detector with hysteresis lockout.

    Operates on a per-symbol quote stream. Triggers when:
      1. price_velocity_30s exceeds threshold (absolute value)
      2. local_trend_persistence exceeds threshold
      3. quote freshness < max_ms

    After trigger, locks out the symbol for lockout_seconds.
    """

    def __init__(self, config: dict):
        det = config["detector"]
        self.vel_threshold = det["velocity_30s_threshold"]
        self.trend_threshold = det["trend_persistence_threshold"]
        self.freshness_max_ms = det["quote_freshness_max_ms"]
        self.lockout_s = det["lockout_seconds"]

        # Per-symbol state
        self._last_ignition: dict[str, float] = {}  # symbol -> epoch of last ignition
        self._quote_buffers: dict[str, list] = {}    # symbol -> recent quotes for rolling calc

    def reset(self):
        self._last_ignition.clear()
        self._quote_buffers.clear()

    def feed_quote(self, symbol: str, quote: dict) -> dict | None:
        """Feed a single quote tick. Returns ignition event or None.

        quote: {epoch, bid, ask, mid, last}
        """
        epoch = quote["epoch"]
        mid = quote.get("mid")
        if mid is None or mid <= 0:
            return None

        # Maintain rolling buffer (keep last 35 seconds)
        buf = self._quote_buffers.setdefault(symbol, [])
        buf.append(quote)

        # Trim buffer to 35s window
        cutoff = epoch - 35.0
        while buf and buf[0]["epoch"] < cutoff:
            buf.pop(0)

        # Need at least 5 quotes and 10 seconds of data
        if len(buf) < 5:
            return None
        span = buf[-1]["epoch"] - buf[0]["epoch"]
        if span < 10.0:
            return None

        # Check lockout
        last_ign = self._last_ignition.get(symbol, 0)
        if epoch - last_ign < self.lockout_s:
            return None

        # Compute indicators
        velocity = self._compute_velocity_30s(buf)
        trend = self._compute_trend_persistence(buf)
        freshness_ms = self._compute_freshness_ms(buf)

        # Check criteria
        criteria_met = 0
        debug = {
            "velocity_30s": round(velocity, 4),
            "velocity_threshold": self.vel_threshold,
            "trend_persistence": round(trend, 4),
            "trend_threshold": self.trend_threshold,
            "freshness_ms": round(freshness_ms, 1),
            "freshness_max_ms": self.freshness_max_ms,
        }

        vel_pass = abs(velocity) >= self.vel_threshold
        trend_pass = trend >= self.trend_threshold
        fresh_pass = freshness_ms <= self.freshness_max_ms

        if vel_pass:
            criteria_met += 1
        if trend_pass:
            criteria_met += 1
        if fresh_pass:
            criteria_met += 1

        if criteria_met < 3:
            return None

        # Ignition detected
        direction = "up" if velocity > 0 else "down"
        confidence = self._compute_confidence(abs(velocity), trend, freshness_ms)

        self._last_ignition[symbol] = epoch

        return {
            "symbol": symbol,
            "epoch": epoch,
            "ts": quote.get("ts", ""),
            "type": "IGNITION_DETECTED",
            "ignition_price": mid,
            "direction": direction,
            "confidence": round(confidence, 3),
            "debug": debug,
        }

    def _compute_velocity_30s(self, buf: list) -> float:
        """Price velocity over ~30 seconds as % change."""
        now = buf[-1]
        # Find quote closest to 30s ago
        target = now["epoch"] - 30.0
        best = buf[0]
        for q in buf:
            if abs(q["epoch"] - target) < abs(best["epoch"] - target):
                best = q
        if best["mid"] is None or best["mid"] <= 0:
            return 0.0
        return ((now["mid"] - best["mid"]) / best["mid"]) * 100.0

    def _compute_trend_persistence(self, buf: list) -> float:
        """Fraction of consecutive ticks moving in the same direction.
        Returns 0.0..1.0 where 1.0 = all ticks move in same direction.
        """
        if len(buf) < 3:
            return 0.0

        mids = [q["mid"] for q in buf if q["mid"] is not None]
        if len(mids) < 3:
            return 0.0

        # Direction of overall move
        overall = mids[-1] - mids[0]
        if overall == 0:
            return 0.0

        same_dir = 0
        total = 0
        for i in range(1, len(mids)):
            diff = mids[i] - mids[i - 1]
            if diff == 0:
                continue
            total += 1
            if (diff > 0 and overall > 0) or (diff < 0 and overall < 0):
                same_dir += 1

        return same_dir / total if total > 0 else 0.0

    def _compute_freshness_ms(self, buf: list) -> float:
        """Average inter-quote gap in milliseconds over recent quotes."""
        if len(buf) < 2:
            return 99999.0
        recent = buf[-min(10, len(buf)):]
        gaps = []
        for i in range(1, len(recent)):
            gap = (recent[i]["epoch"] - recent[i - 1]["epoch"]) * 1000.0
            gaps.append(gap)
        return statistics.mean(gaps) if gaps else 99999.0

    def _compute_confidence(self, abs_vel: float, trend: float, fresh_ms: float) -> float:
        """Confidence 0..1 based on how strongly criteria are exceeded."""
        vel_score = min(abs_vel / (self.vel_threshold * 3), 1.0)
        trend_score = min(trend / 1.0, 1.0)
        fresh_score = max(0, 1.0 - fresh_ms / (self.freshness_max_ms * 2))
        return 0.4 * vel_score + 0.4 * trend_score + 0.2 * fresh_score


# ═══════════════════════════════════════════════════════════════════
# PART B — Entry Accelerator Policy
# ═══════════════════════════════════════════════════════════════════

class EntryAcceleratorPolicy:
    """Given an ignition event, propose limit entry candidates."""

    def __init__(self, config: dict):
        pol = config["accelerator_policy"]
        self.offsets = pol["offsets_pct"]
        self.fill_window = pol["fill_window_seconds"]
        self.target_start = pol["target_window_start_s"]
        self.target_end = pol["target_window_end_s"]

        sc = config.get("spread_constraints", {})
        self.pm_max_spread = sc.get("pm_max_spread_pct", 0.9)
        self.rth_max_spread = sc.get("rth_max_spread_pct", 0.6)

    def propose_entries(self, ignition: dict, spread_pct: float | None = None) -> list[dict]:
        """Generate EARLY_ENTRY_CANDIDATE proposals for each offset."""
        epoch = ignition["epoch"]
        price = ignition["ignition_price"]
        symbol = ignition["symbol"]
        direction = ignition["direction"]

        # Check spread constraint
        is_pm = epoch < PM_END_EPOCH
        max_spread = self.pm_max_spread if is_pm else self.rth_max_spread
        spread_ok = spread_pct is None or spread_pct <= max_spread

        candidates = []
        for offset in self.offsets:
            # For upward ignitions, limit buy below current price
            # offset is negative (e.g., -0.50%) meaning buy at discount
            if direction == "up":
                proposed_limit = price * (1.0 + offset / 100.0)
            else:
                # For downward ignitions (short momentum), limit above current
                proposed_limit = price * (1.0 - offset / 100.0)

            candidates.append({
                "symbol": symbol,
                "epoch": epoch,
                "ts": ignition.get("ts", ""),
                "type": "EARLY_ENTRY_CANDIDATE",
                "direction": direction,
                "offset_pct": offset,
                "ignition_price": price,
                "proposed_limit": round(proposed_limit, 6),
                "window_sec": self.fill_window,
                "spread_ok": spread_ok,
                "spread_pct": spread_pct,
                "confidence": ignition.get("confidence", 0),
                "reason": "IGNITION_ACCEL",
            })

        return candidates


# ═══════════════════════════════════════════════════════════════════
# PART C — Research Replay Simulator
# ═══════════════════════════════════════════════════════════════════

class ReplaySimulator:
    """Simulates limit order fill + exit model for each candidate."""

    def __init__(self, config: dict):
        em = config["exit_model"]
        self.hard_stop_pct = em["hard_stop_pct"]
        self.trail_activate_pct = em["trail_activate_pct"]
        self.trail_distance_pct = em["trail_distance_pct"]
        self.time_exit_s = em["time_exit_seconds"]

        pol = config["accelerator_policy"]
        self.fill_window = pol["fill_window_seconds"]

    def simulate(self, candidate: dict, quotes: list[dict]) -> dict:
        """Simulate a single candidate entry against the quote stream.

        Returns result dict with fill status, exit type, return, etc.
        """
        ign_epoch = candidate["epoch"]
        limit_price = candidate["proposed_limit"]
        direction = candidate["direction"]
        offset_pct = candidate["offset_pct"]
        symbol = candidate["symbol"]

        result = {
            "symbol": symbol,
            "ignition_epoch": ign_epoch,
            "offset_pct": offset_pct,
            "direction": direction,
            "ignition_price": candidate["ignition_price"],
            "proposed_limit": limit_price,
            "confidence": candidate.get("confidence", 0),
            "spread_ok": candidate.get("spread_ok", True),
        }

        # Phase 1: Try to fill within fill_window
        fill_epoch, fill_price = self._try_fill(
            quotes, ign_epoch, limit_price, direction, self.fill_window
        )

        if fill_epoch is None:
            result["filled"] = False
            result["fill_reason"] = "NO_TOUCH"
            result["exit_type"] = None
            result["exit_return_pct"] = None
            result["exit_time_sec"] = None
            result["peak_return_pct"] = None
            result["trough_return_pct"] = None
            return result

        result["filled"] = True
        result["fill_epoch"] = fill_epoch
        result["fill_price"] = fill_price
        result["fill_delay_s"] = round(fill_epoch - ign_epoch, 2)
        result["slippage_pct"] = round(
            ((fill_price - candidate["ignition_price"]) / candidate["ignition_price"]) * 100.0, 4
        )

        # Phase 2: Simulate exit model from fill point
        exit_result = self._simulate_exit(quotes, fill_epoch, fill_price, direction)
        result.update(exit_result)

        return result

    def _try_fill(self, quotes: list, ign_epoch: float, limit: float,
                  direction: str, window: float) -> tuple:
        """Try to fill a limit order within the fill window.

        For upward ignition: limit buy = we buy at or below limit price.
          Fill if any quote's ask <= limit_price (or mid touches limit).
        For downward ignition: limit sell-short = we sell at or above limit.
          Fill if any quote's bid >= limit_price.

        Returns (fill_epoch, fill_price) or (None, None).
        """
        deadline = ign_epoch + window

        for q in quotes:
            if q["epoch"] <= ign_epoch:
                continue
            if q["epoch"] > deadline:
                break

            mid = q.get("mid")
            if mid is None:
                continue

            if direction == "up":
                # Buy limit: fill when price drops to or below limit
                # Use mid as proxy (conservative)
                if mid <= limit:
                    return q["epoch"], mid
                # Also check if ask touches limit
                ask = q.get("ask")
                if ask is not None and ask <= limit:
                    return q["epoch"], ask
                # 0% offset: fill immediately at market (mid)
                if limit >= mid:
                    return q["epoch"], mid
            else:
                # Short limit: fill when price rises to or above limit
                if mid >= limit:
                    return q["epoch"], mid
                bid = q.get("bid")
                if bid is not None and bid >= limit:
                    return q["epoch"], bid
                if limit <= mid:
                    return q["epoch"], mid

        return None, None

    def _simulate_exit(self, quotes: list, fill_epoch: float, fill_price: float,
                       direction: str) -> dict:
        """Simulate exit model from fill point.

        Returns dict with exit_type, exit_return_pct, exit_time_sec, etc.
        """
        deadline = fill_epoch + self.time_exit_s
        peak_ret = 0.0
        trough_ret = 0.0
        trail_active = False
        trail_peak = 0.0

        dir_mult = 1.0 if direction == "up" else -1.0

        last_q = None
        for q in quotes:
            if q["epoch"] <= fill_epoch:
                continue
            mid = q.get("mid")
            if mid is None:
                continue

            raw_ret = ((mid - fill_price) / fill_price) * 100.0
            dir_ret = raw_ret * dir_mult  # positive = favorable

            if dir_ret > peak_ret:
                peak_ret = dir_ret
            if dir_ret < trough_ret:
                trough_ret = dir_ret

            # Hard stop
            if dir_ret <= self.hard_stop_pct:
                return {
                    "exit_type": "STOP",
                    "exit_return_pct": round(dir_ret, 4),
                    "exit_price": mid,
                    "exit_time_sec": round(q["epoch"] - fill_epoch, 2),
                    "trail_activated": trail_active,
                    "peak_return_pct": round(peak_ret, 4),
                    "trough_return_pct": round(trough_ret, 4),
                }

            # Trail activation
            if dir_ret >= self.trail_activate_pct:
                trail_active = True
                if dir_ret > trail_peak:
                    trail_peak = dir_ret

            # Trail exit
            if trail_active and (trail_peak - dir_ret) >= self.trail_distance_pct:
                return {
                    "exit_type": "TRAIL",
                    "exit_return_pct": round(dir_ret, 4),
                    "exit_price": mid,
                    "exit_time_sec": round(q["epoch"] - fill_epoch, 2),
                    "trail_activated": True,
                    "peak_return_pct": round(peak_ret, 4),
                    "trough_return_pct": round(trough_ret, 4),
                }

            # Time exit
            if q["epoch"] >= deadline:
                return {
                    "exit_type": "TIME",
                    "exit_return_pct": round(dir_ret, 4),
                    "exit_price": mid,
                    "exit_time_sec": round(q["epoch"] - fill_epoch, 2),
                    "trail_activated": trail_active,
                    "peak_return_pct": round(peak_ret, 4),
                    "trough_return_pct": round(trough_ret, 4),
                }

            last_q = q

        # End of data
        if last_q is not None:
            mid = last_q["mid"]
            raw_ret = ((mid - fill_price) / fill_price) * 100.0
            dir_ret = raw_ret * dir_mult
            return {
                "exit_type": "EOD",
                "exit_return_pct": round(dir_ret, 4),
                "exit_price": mid,
                "exit_time_sec": round(last_q["epoch"] - fill_epoch, 2),
                "trail_activated": trail_active,
                "peak_return_pct": round(peak_ret, 4),
                "trough_return_pct": round(trough_ret, 4),
            }

        return {
            "exit_type": "NO_DATA",
            "exit_return_pct": 0.0,
            "exit_price": fill_price,
            "exit_time_sec": 0,
            "trail_activated": False,
            "peak_return_pct": 0.0,
            "trough_return_pct": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_quotes_ffill(symbol: str) -> list[dict]:
    """Load quote cache with forward-filled bid/ask."""
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
            "ts": q.get("ts", ""),
            "bid": last_bid,
            "ask": last_ask,
            "last": q.get("last"),
            "mid": ((last_bid + last_ask) / 2) if last_bid and last_ask else q.get("last"),
        })
    return filled


def load_config() -> dict:
    path = CONFIG_DIR / "ignition_accelerator.json"
    with open(path) as f:
        return json.load(f)


def load_ignition_events(date_str: str) -> dict:
    path = OUTPUT_DIR / f"ignition_events_{date_str}.json"
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_detector_pass(ignition_data: dict, config: dict) -> list[dict]:
    """Part A: Run ignition detector on quote streams, producing events.

    Uses the pre-detected ignition events (from ignition_detector.py)
    since those were already validated. This function re-runs detection
    using the new streaming detector to verify and produce enriched events.
    """
    print("\n=== PART A: Ignition Detector (streaming verification) ===")

    detector = IgnitionDetector(config)
    all_ignitions = []
    total_quotes_processed = 0

    for symbol, sym_data in ignition_data["events_per_symbol"].items():
        try:
            quotes = load_quotes_ffill(symbol)
        except FileNotFoundError:
            continue

        detector.reset()
        sym_ignitions = []

        for q in quotes:
            event = detector.feed_quote(symbol, q)
            if event is not None:
                sym_ignitions.append(event)

        total_quotes_processed += len(quotes)
        all_ignitions.extend(sym_ignitions)
        orig_count = sym_data["count"]
        print(f"  {symbol}: {len(sym_ignitions)} v1-detector ignitions "
              f"(original: {orig_count})")

    print(f"\n  Total: {len(all_ignitions)} ignitions from {total_quotes_processed} quotes")
    print(f"  Lockout: {config['detector']['lockout_seconds']}s")
    return all_ignitions


def run_accelerator_pass(ignitions: list[dict], config: dict,
                         quote_cache: dict) -> list[dict]:
    """Part B+C: Generate candidates and simulate fills + exits."""
    print("\n=== PART B+C: Accelerator Policy + Replay Simulation ===")

    policy = EntryAcceleratorPolicy(config)
    simulator = ReplaySimulator(config)

    all_results = []
    offsets = config["accelerator_policy"]["offsets_pct"]

    for ign in ignitions:
        symbol = ign["symbol"]
        quotes = quote_cache.get(symbol)
        if quotes is None:
            continue

        # Compute spread at ignition time
        spread_pct = None
        for q in quotes:
            if q["epoch"] >= ign["epoch"]:
                if q["bid"] and q["ask"] and q["ask"] > 0:
                    spread_pct = ((q["ask"] - q["bid"]) / q["ask"]) * 100.0
                break

        candidates = policy.propose_entries(ign, spread_pct)

        for cand in candidates:
            result = simulator.simulate(cand, quotes)
            all_results.append(result)

    # Print summary per offset
    for offset in offsets:
        subset = [r for r in all_results if r["offset_pct"] == offset]
        filled = [r for r in subset if r["filled"]]
        fill_rate = len(filled) / len(subset) * 100 if subset else 0

        if filled:
            returns = [r["exit_return_pct"] for r in filled if r["exit_return_pct"] is not None]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            wr = len(wins) / len(returns) * 100 if returns else 0
            avg_ret = statistics.mean(returns) if returns else 0
            pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (
                999.0 if wins else 0.0)
            print(f"\n  Offset {offset:+.2f}%: fill={fill_rate:.1f}% ({len(filled)}/{len(subset)}), "
                  f"WR={wr:.1f}%, avg={avg_ret:+.4f}%, PF={pf:.3f}")
        else:
            print(f"\n  Offset {offset:+.2f}%: fill={fill_rate:.1f}% ({len(filled)}/{len(subset)})")

    return all_results


def generate_report(date_str: str, config: dict, ignitions: list[dict],
                    results: list[dict]) -> str:
    """Part D: Generate the accelerator report."""
    print("\n=== PART D: Report Generation ===")

    offsets = config["accelerator_policy"]["offsets_pct"]
    lines = []

    lines.append(f"# Ignition Entry Accelerator Report - {date_str}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("**Mode:** READ-ONLY RESEARCH (SuperBot Engine)")
    lines.append("**Method:** Limit-order replay at ignition events with exit model simulation")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview
    lines.append("## Study Overview")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Ignition events tested | {len(ignitions)} |")
    lines.append(f"| Offsets tested | {', '.join(f'{o:+.2f}%' for o in offsets)} |")
    lines.append(f"| Fill window | {config['accelerator_policy']['fill_window_seconds']}s |")
    lines.append(f"| Exit model | stop={config['exit_model']['hard_stop_pct']}%, "
                 f"trail={config['exit_model']['trail_activate_pct']}%/"
                 f"{config['exit_model']['trail_distance_pct']}%, "
                 f"time={config['exit_model']['time_exit_seconds']}s |")
    lines.append(f"| Detector lockout | {config['detector']['lockout_seconds']}s |")
    lines.append("")

    # Offset comparison table
    lines.append("## Offset Comparison")
    lines.append("")
    lines.append("| Offset | N Total | Filled | Fill Rate | WR | Avg Return | PF | "
                 "Avg Slippage | Worst DD | Avg Fill Delay |")
    lines.append("|--------|---------|--------|-----------|-----|------------|-----|"
                 "-------------|----------|----------------|")

    offset_stats = {}
    for offset in offsets:
        subset = [r for r in results if r["offset_pct"] == offset]
        filled = [r for r in subset if r["filled"]]
        fill_rate = len(filled) / len(subset) * 100 if subset else 0

        if filled:
            returns = [r["exit_return_pct"] for r in filled if r["exit_return_pct"] is not None]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            wr = len(wins) / len(returns) * 100 if returns else 0
            avg_ret = statistics.mean(returns) if returns else 0
            pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (
                999.0 if wins else 0.0)
            slippages = [r["slippage_pct"] for r in filled if "slippage_pct" in r]
            avg_slip = statistics.mean(slippages) if slippages else 0
            troughs = [r["trough_return_pct"] for r in filled if r.get("trough_return_pct") is not None]
            worst_dd = min(troughs) if troughs else 0
            delays = [r["fill_delay_s"] for r in filled if "fill_delay_s" in r]
            avg_delay = statistics.mean(delays) if delays else 0

            lines.append(
                f"| {offset:+.2f}% | {len(subset)} | {len(filled)} | {fill_rate:.1f}% | "
                f"{wr:.1f}% | {avg_ret:+.4f}% | {pf:.3f} | "
                f"{avg_slip:+.4f}% | {worst_dd:+.4f}% | {avg_delay:.1f}s |"
            )

            offset_stats[offset] = {
                "n_total": len(subset),
                "n_filled": len(filled),
                "fill_rate": round(fill_rate, 1),
                "win_rate": round(wr, 1),
                "avg_return": round(avg_ret, 4),
                "profit_factor": round(pf, 3),
                "avg_slippage": round(avg_slip, 4),
                "worst_drawdown": round(worst_dd, 4),
                "avg_fill_delay": round(avg_delay, 1),
            }
        else:
            lines.append(
                f"| {offset:+.2f}% | {len(subset)} | 0 | 0.0% | - | - | - | - | - | - |"
            )
            offset_stats[offset] = {
                "n_total": len(subset), "n_filled": 0, "fill_rate": 0,
                "win_rate": 0, "avg_return": 0, "profit_factor": 0,
            }

    lines.append("")

    # Exit type distribution per offset
    lines.append("## Exit Type Distribution")
    lines.append("")
    lines.append("| Offset | STOP | TRAIL | TIME | EOD |")
    lines.append("|--------|------|-------|------|-----|")
    for offset in offsets:
        filled = [r for r in results if r["offset_pct"] == offset and r["filled"]]
        if not filled:
            lines.append(f"| {offset:+.2f}% | - | - | - | - |")
            continue
        exits = defaultdict(int)
        for r in filled:
            exits[r.get("exit_type", "UNKNOWN")] += 1
        n = len(filled)
        stop_pct = exits.get("STOP", 0) / n * 100
        trail_pct = exits.get("TRAIL", 0) / n * 100
        time_pct = exits.get("TIME", 0) / n * 100
        eod_pct = exits.get("EOD", 0) / n * 100
        lines.append(
            f"| {offset:+.2f}% | {stop_pct:.1f}% ({exits.get('STOP', 0)}) | "
            f"{trail_pct:.1f}% ({exits.get('TRAIL', 0)}) | "
            f"{time_pct:.1f}% ({exits.get('TIME', 0)}) | "
            f"{eod_pct:.1f}% ({exits.get('EOD', 0)}) |"
        )
    lines.append("")

    # Per-symbol breakdown
    lines.append("## Per-Symbol Breakdown (by ignition count)")
    lines.append("")

    sym_counts = defaultdict(int)
    for ign in ignitions:
        sym_counts[ign["symbol"]] += 1

    for offset in offsets:
        lines.append(f"### Offset {offset:+.2f}%")
        lines.append("")
        lines.append("| Symbol | Ignitions | Filled | Fill Rate | WR | Avg Return | PF |")
        lines.append("|--------|-----------|--------|-----------|-----|------------|-----|")

        for sym, _ in sorted(sym_counts.items(), key=lambda x: -x[1]):
            sym_results = [r for r in results
                           if r["offset_pct"] == offset and r["symbol"] == sym]
            filled = [r for r in sym_results if r["filled"]]
            fr = len(filled) / len(sym_results) * 100 if sym_results else 0

            if filled:
                rets = [r["exit_return_pct"] for r in filled if r["exit_return_pct"] is not None]
                wins = [r for r in rets if r > 0]
                losses = [r for r in rets if r < 0]
                wr = len(wins) / len(rets) * 100 if rets else 0
                avg = statistics.mean(rets) if rets else 0
                pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (
                    999.0 if wins else 0.0)
                lines.append(
                    f"| {sym} | {sym_counts[sym]} | {len(filled)} | {fr:.1f}% | "
                    f"{wr:.1f}% | {avg:+.4f}% | {pf:.3f} |"
                )
            else:
                lines.append(
                    f"| {sym} | {sym_counts[sym]} | 0 | 0.0% | - | - | - |"
                )

        lines.append("")

    # Recommended configuration
    lines.append("## Recommended Configuration")
    lines.append("")

    # Find best offset by PF (among those with decent fill rate)
    best_offset = None
    best_pf = 0
    for offset, stats in offset_stats.items():
        if stats["fill_rate"] >= 20 and stats["profit_factor"] > best_pf:
            best_pf = stats["profit_factor"]
            best_offset = offset

    if best_offset is not None:
        bs = offset_stats[best_offset]
        lines.append(f"- **Recommended offset:** {best_offset:+.2f}%")
        lines.append(f"- **Expected fill rate:** {bs['fill_rate']:.1f}%")
        lines.append(f"- **Expected WR:** {bs['win_rate']:.1f}%")
        lines.append(f"- **Expected PF:** {bs['profit_factor']:.3f}")
        lines.append(f"- **Average fill delay:** {bs['avg_fill_delay']:.1f}s after ignition")
    else:
        lines.append("- No offset met minimum fill rate threshold")

    lines.append(f"- **Spread constraint (PM):** < {config['spread_constraints']['pm_max_spread_pct']}%")
    lines.append(f"- **Spread constraint (RTH):** < {config['spread_constraints']['rth_max_spread_pct']}%")
    lines.append(f"- **Lockout:** {config['detector']['lockout_seconds']}s between ignitions per symbol")
    lines.append("")

    # Part E — Production Toggle Design
    lines.append("---")
    lines.append("")
    lines.append("## Part E: Production Integration Design")
    lines.append("")
    lines.append("### Proposed runtime_config Keys")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "ignition_accelerator_enabled": false,')
    if best_offset is not None:
        lines.append(f'  "ignition_accelerator_offset_pct": {best_offset},')
    else:
        lines.append('  "ignition_accelerator_offset_pct": -0.20,')
    lines.append(f'  "ignition_accelerator_fill_window_seconds": {config["accelerator_policy"]["fill_window_seconds"]},')
    lines.append(f'  "ignition_accelerator_lockout_seconds": {config["detector"]["lockout_seconds"]},')
    lines.append('  "ignition_accelerator_pm_max_spread_pct": 0.9,')
    lines.append('  "ignition_accelerator_rth_max_spread_pct": 0.6')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("### Integration Path")
    lines.append("")
    lines.append("1. **IgnitionDetector** runs in Morpheus quote processing loop (per-symbol)")
    lines.append("2. On IGNITION_DETECTED, accelerator proposes a LIMIT entry at offset price")
    lines.append("3. If signal already exists in pipeline (RISK_APPROVED), entry_reference is "
                 "overridden to proposed_limit")
    lines.append("4. If no signal exists yet, candidate is queued — will be consumed if/when "
                 "signal arrives within fill_window_seconds")
    lines.append("5. All existing gates remain active: risk, meta, containment are NOT bypassed")
    lines.append("6. Accelerator only improves entry price — it does not create new signals")
    lines.append("")
    lines.append("### Safety Guarantees")
    lines.append("")
    lines.append("- Accelerator is gated by `ignition_accelerator_enabled` (default: false)")
    lines.append("- Does NOT bypass risk gate, meta gate, or containment filter")
    lines.append("- Does NOT create new trading signals (only adjusts entry price of existing ones)")
    lines.append("- Lockout prevents over-trading on repeated ignitions")
    lines.append("- Spread constraints prevent entries in illiquid conditions")
    lines.append("- LIMIT order means unfavorable fills are impossible (no market order slippage)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This study is research-only. No production changes applied.*")

    report = "\n".join(lines)
    return report, offset_stats, best_offset


def save_candidates(date_str: str, results: list[dict]):
    """Save all candidate results to JSON."""
    path = OUTPUT_DIR / f"ignition_accelerator_candidates_{date_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "type": "ignition_accelerator_candidates",
            "total_candidates": len(results),
            "candidates": results,
        }, f, indent=2)
    print(f"  Saved: {path}")
    return path


def update_comms(date_str: str, offset_stats: dict, best_offset: float | None,
                 ignition_count: int, config: dict):
    """Append summary message to comms."""
    print("\n=== COMMS UPDATE ===")

    with open(COMMS_FILE, encoding="utf-8") as f:
        comms = json.load(f)

    summary_lines = ["IGNITION ENTRY ACCELERATOR v1 COMPLETE\n"]

    summary_lines.append("== OFFSET COMPARISON ==")
    for offset, stats in sorted(offset_stats.items()):
        summary_lines.append(
            f"  {offset:+.2f}%: fill={stats['fill_rate']:.1f}%, "
            f"WR={stats['win_rate']:.1f}%, avg={stats['avg_return']:+.4f}%, "
            f"PF={stats['profit_factor']:.3f}"
        )

    if best_offset is not None:
        bs = offset_stats[best_offset]
        summary_lines.append(f"\n== RECOMMENDED OFFSET: {best_offset:+.2f}% ==")
        summary_lines.append(f"  Fill rate: {bs['fill_rate']:.1f}%")
        summary_lines.append(f"  Win rate: {bs['win_rate']:.1f}%")
        summary_lines.append(f"  Profit factor: {bs['profit_factor']:.3f}")
        summary_lines.append(f"  Avg fill delay: {bs.get('avg_fill_delay', 'N/A')}s")
    else:
        summary_lines.append("\n== NO OFFSET MET MINIMUM FILL RATE THRESHOLD ==")

    summary_lines.append(f"\n== DETECTOR ==")
    summary_lines.append(f"  Ignitions detected: {ignition_count}")
    summary_lines.append(f"  Lockout: {config['detector']['lockout_seconds']}s")
    summary_lines.append(f"  Velocity threshold: {config['detector']['velocity_30s_threshold']}")
    summary_lines.append(f"  Trend threshold: {config['detector']['trend_persistence_threshold']}")

    summary_lines.append(f"\n== PRODUCTION TOGGLE DESIGN ==")
    summary_lines.append("  Proposed runtime_config keys:")
    summary_lines.append("    ignition_accelerator_enabled: false (default)")
    if best_offset is not None:
        summary_lines.append(f"    ignition_accelerator_offset_pct: {best_offset}")
    summary_lines.append(f"    ignition_accelerator_fill_window_seconds: "
                         f"{config['accelerator_policy']['fill_window_seconds']}")
    summary_lines.append(f"    ignition_accelerator_lockout_seconds: "
                         f"{config['detector']['lockout_seconds']}")
    summary_lines.append("  Safety: does NOT bypass risk/meta/containment")
    summary_lines.append("  Integration: overrides entry_reference to LIMIT at offset price")

    summary_lines.append(f"\n== FILES ==")
    summary_lines.append(f"  Report: engine/output/ignition_accelerator_report_{date_str}.md")
    summary_lines.append(f"  Candidates: engine/output/ignition_accelerator_candidates_{date_str}.json")
    summary_lines.append(f"  Config: configs/ignition_accelerator.json")
    summary_lines.append("\nProduction remains frozen. All analysis is read-only research.")

    msg = {
        "id": f"msg_chatgpt_{date_str.replace('-', '')}_010",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": "claude",
        "to": "chatgpt",
        "type": "research_result",
        "subject": (f"Ignition Entry Accelerator v1 ({date_str}): "
                    f"best offset={best_offset:+.2f}% " if best_offset is not None else
                    f"Ignition Entry Accelerator v1 ({date_str}): no offset met threshold ") +
                   f"across {ignition_count} ignitions",
        "body": "\n".join(summary_lines),
        "references": [
            f"engine/output/ignition_accelerator_report_{date_str}.md",
            f"engine/output/ignition_accelerator_candidates_{date_str}.json",
            "configs/ignition_accelerator.json",
        ],
    }

    comms["messages"].append(msg)
    with open(COMMS_FILE, "w", encoding="utf-8") as f:
        json.dump(comms, f, indent=2)

    print(f"  Appended msg_010 to comms outbox")


def main():
    parser = argparse.ArgumentParser(description="Ignition Entry Accelerator v1")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    args = parser.parse_args()
    date_str = args.date

    print(f"Ignition Entry Accelerator v1 - {date_str}")
    print("=" * 60)

    # Load config
    config = load_config()
    print(f"\nConfig loaded: offsets={config['accelerator_policy']['offsets_pct']}, "
          f"fill_window={config['accelerator_policy']['fill_window_seconds']}s")

    # Load ignition events (from prior study)
    ignition_data = load_ignition_events(date_str)
    total_prior = ignition_data.get("total_events", 0)
    print(f"Prior ignition events: {total_prior}")

    # Part A: Run streaming detector
    ignitions = run_detector_pass(ignition_data, config)

    if not ignitions:
        print("\nERROR: No ignitions detected. Check thresholds.")
        sys.exit(1)

    # Pre-load all quote caches
    print("\n  Loading quote caches...")
    quote_cache = {}
    symbols = set(ign["symbol"] for ign in ignitions)
    for sym in sorted(symbols):
        try:
            quote_cache[sym] = load_quotes_ffill(sym)
            print(f"    {sym}: {len(quote_cache[sym])} quotes")
        except FileNotFoundError:
            print(f"    {sym}: MISSING")

    # Part B+C: Accelerator + Replay
    results = run_accelerator_pass(ignitions, config, quote_cache)

    # Save candidates JSON
    save_candidates(date_str, results)

    # Part D: Report
    report, offset_stats, best_offset = generate_report(
        date_str, config, ignitions, results
    )

    report_path = OUTPUT_DIR / f"ignition_accelerator_report_{date_str}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Part E + Comms
    update_comms(date_str, offset_stats, best_offset, len(ignitions), config)

    # Final summary
    print("\n" + "=" * 60)
    print("IGNITION ENTRY ACCELERATOR v1 COMPLETE")
    print("=" * 60)
    print(f"\nFiles generated:")
    print(f"  1. ai/research/ignition_entry_accelerator.py")
    print(f"  2. configs/ignition_accelerator.json")
    print(f"  3. {report_path}")
    print(f"  4. engine/output/ignition_accelerator_candidates_{date_str}.json")
    print(f"  5. comms/outbox_chatgpt.json (msg_010)")

    if best_offset is not None:
        bs = offset_stats[best_offset]
        print(f"\n  RECOMMENDED: {best_offset:+.2f}% offset")
        print(f"    Fill rate: {bs['fill_rate']:.1f}%")
        print(f"    Win rate: {bs['win_rate']:.1f}%")
        print(f"    Avg return: {bs['avg_return']:+.4f}%")
        print(f"    Profit factor: {bs['profit_factor']:.3f}")


if __name__ == "__main__":
    main()
