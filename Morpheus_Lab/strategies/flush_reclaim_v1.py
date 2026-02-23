"""
Morpheus Lab — Flush Reclaim v1 (Standalone)
================================================
Pure flush-reclaim strategy. No breakout logic. No pullback logic.

Structural Definition:
  A "flush reclaim" is a specific microstructure pattern where:
  1. FLUSH: Price drops sharply below VWAP (>= flush_pct within flush_window ticks)
  2. SPIKE: Volume during flush exceeds vol_surge x rolling average
  3. RECLAIM: Price recovers back above VWAP within reclaim_window ticks
  4. ENTRY: At the moment of VWAP reclaim, go long
  5. STOP: At the low of the flush (structural, not percentage)
  6. TARGET: Entry + reward_multiple x (entry - flush_low)

Why this works (hypothesis):
  - Flushes trigger stop hunts / weak hand liquidation
  - Volume spike confirms real selling pressure (not just drift)
  - Reclaim proves supply is absorbed -- buyers stepped in
  - Structural stop at flush low = tight risk, defined invalidation
  - The pattern is mean-reversion, not momentum

Regime Gate:
  Only trades in TREND_DOWN and LOW_VOL_CHOP by default.
  These regimes showed highest PF for this entry type.

Parameters:
  lookback:         Rolling window for VWAP calculation
  flush_pct:        Min % drop below VWAP to qualify as flush (default 0.5%)
  flush_window:     Max ticks for flush to develop (default 50)
  vol_surge:        Volume multiplier for spike detection (default 2.0)
  reclaim_window:   Max ticks after flush low to reclaim VWAP (default 100)
  reward_multiple:  Target = risk x N (default 2.0)
  min_risk_pct:     Min risk % to avoid micro-stops (default 0.3%)
  max_risk_pct:     Max risk % to avoid outsized stops (default 3.0%)
  cooldown:         Min ticks between entries (default 50)
  share_size:       Position size (default 100)
  allowed_regimes:  Comma-separated regimes (default: TREND_DOWN,LOW_VOL_CHOP)
  regime_window:    Lookback for regime classifier (default 200)
"""

import numpy as np
from typing import List, Optional, Set

from strategies.batch_strategy import BatchStrategy, BatchTrade
from engine.regime_classifier import (
    classify_tick_regime,
    LOW_VOL_CHOP, TREND_UP, TREND_DOWN,
    HIGH_VOL_BREAKOUT, PARABOLIC_EXTEND, LIQUIDITY_FADE,
)


DEFAULT_ALLOWED_REGIMES = {TREND_DOWN, LOW_VOL_CHOP}


class FlushReclaimV1(BatchStrategy):
    """Standalone flush-reclaim strategy with structural stops."""

    name = "flush_reclaim_v1"

    def __init__(
        self,
        lookback: int = 200,
        flush_pct: float = 0.5,
        flush_window: int = 50,
        vol_surge: float = 2.0,
        reclaim_window: int = 100,
        reward_multiple: float = 2.0,
        min_risk_pct: float = 0.3,
        max_risk_pct: float = 3.0,
        cooldown: int = 50,
        share_size: int = 100,
        allowed_regimes: Optional[str] = None,
        regime_window: int = 200,
    ):
        super().__init__(
            lookback=lookback,
            flush_pct=flush_pct,
            flush_window=flush_window,
            vol_surge=vol_surge,
            reclaim_window=reclaim_window,
            reward_multiple=reward_multiple,
            min_risk_pct=min_risk_pct,
            max_risk_pct=max_risk_pct,
            cooldown=cooldown,
            share_size=share_size,
        )
        self.lookback = lookback
        self.flush_pct = flush_pct / 100.0
        self.flush_window = flush_window
        self.vol_surge = vol_surge
        self.reclaim_window = reclaim_window
        self.reward_multiple = reward_multiple
        self.min_risk_pct = min_risk_pct / 100.0
        self.max_risk_pct = max_risk_pct / 100.0
        self.cooldown = cooldown
        self.share_size = share_size
        self.regime_window = regime_window

        if allowed_regimes:
            self.allowed_regimes = set(r.strip() for r in allowed_regimes.split(","))
        else:
            self.allowed_regimes = DEFAULT_ALLOWED_REGIMES.copy()

    def on_batch(
        self,
        ts: np.ndarray,
        price: np.ndarray,
        size: np.ndarray,
        symbol: str,
    ) -> List[BatchTrade]:
        """
        Vectorized flush detection + reclaim entry scanning.

        Phase 1 (vectorized): Compute VWAP, volume stats, regimes
        Phase 2 (vectorized): Find flush candidates
        Phase 3 (scan loop): For each flush, scan forward for VWAP reclaim
        """
        n = len(price)
        if n < self.lookback + self.flush_window + self.reclaim_window:
            return []

        lb = self.lookback

        # == PHASE 1: COMPUTE INFRASTRUCTURE (vectorized) ==

        # Rolling VWAP
        dollar_vol = price * size.astype(np.float64)
        cum_dollar = np.cumsum(dollar_vol)
        cum_size = np.cumsum(size.astype(np.float64))

        roll_dollar = np.zeros(n)
        roll_size_arr = np.zeros(n)
        roll_dollar[lb:] = cum_dollar[lb:] - cum_dollar[:-lb]
        roll_size_arr[lb:] = cum_size[lb:] - cum_size[:-lb]

        vwap = np.zeros(n)
        valid = roll_size_arr > 0
        vwap[valid] = roll_dollar[valid] / roll_size_arr[valid]

        # Rolling average volume
        avg_vol = np.zeros(n)
        avg_vol[lb:] = roll_size_arr[lb:] / lb

        # Regime classification
        regimes = classify_tick_regime(price, size, window=self.regime_window)

        # == PHASE 2: FIND FLUSH CANDIDATES (vectorized) ==

        # A flush tick is where:
        #   1. VWAP is valid (> 0)
        #   2. Price is below VWAP by at least flush_pct
        #   3. Volume is surging (current tick > vol_surge x avg)

        flush_threshold = vwap * (1 - self.flush_pct)
        is_below_vwap = (price < flush_threshold) & (vwap > 0)
        is_vol_surge = size.astype(np.float64) > (avg_vol * self.vol_surge)

        flush_ticks = is_below_vwap & is_vol_surge
        flush_indices = np.where(flush_ticks)[0]

        # Filter to valid range
        min_idx = lb
        max_idx = n - self.reclaim_window - 1
        flush_indices = flush_indices[(flush_indices >= min_idx) & (flush_indices <= max_idx)]

        if len(flush_indices) == 0:
            return []

        # == PHASE 3: RECLAIM SCANNING (per-flush loop) ==

        trades: List[BatchTrade] = []
        next_allowed = 0
        _allowed = self.allowed_regimes

        # Group flush ticks into flush "events"
        flush_events = _cluster_flush_events(flush_indices, self.flush_window)

        for event_start, event_end in flush_events:
            # Skip if in cooldown from prior trade
            if event_start < next_allowed:
                continue

            # Regime gate at flush start
            regime_at_flush = str(regimes[event_start])
            if regime_at_flush not in _allowed:
                continue

            # Find the flush low (lowest price in the flush event range)
            flush_slice = price[event_start:event_end + 1]
            flush_low_offset = np.argmin(flush_slice)
            flush_low_idx = event_start + flush_low_offset
            flush_low = float(price[flush_low_idx])

            if flush_low <= 0:
                continue

            # VWAP at flush start (reference level)
            reclaim_level = float(vwap[event_start])
            if reclaim_level <= 0:
                continue

            # Scan forward from flush low for VWAP reclaim
            scan_start = flush_low_idx + 1
            scan_end = min(flush_low_idx + self.reclaim_window, n)

            if scan_start >= scan_end:
                continue

            scan_slice = price[scan_start:scan_end]
            vwap_slice = vwap[scan_start:scan_end]

            # Reclaim = price crosses back above evolving VWAP
            reclaim_mask = scan_slice > vwap_slice
            reclaim_hits = np.where(reclaim_mask)[0]

            if len(reclaim_hits) == 0:
                continue  # No reclaim within window

            # Entry at first reclaim tick
            reclaim_offset = reclaim_hits[0]
            entry_idx = scan_start + reclaim_offset
            entry_price = float(price[entry_idx])
            entry_ts = int(ts[entry_idx])

            # == STRUCTURAL STOP: flush low ==
            stop_price = flush_low

            # Risk validation
            risk = entry_price - stop_price
            risk_pct = risk / entry_price if entry_price > 0 else 0

            if risk_pct < self.min_risk_pct:
                continue  # Stop too tight, likely noise
            if risk_pct > self.max_risk_pct:
                continue  # Stop too wide, risk too large

            # == TARGET: reward_multiple x risk ==
            target_price = entry_price + (self.reward_multiple * risk)

            # == EXIT SCAN (vectorized forward from entry) ==
            exit_slice = price[entry_idx + 1:]
            if len(exit_slice) == 0:
                continue

            hit_target = exit_slice >= target_price
            hit_stop = exit_slice <= stop_price
            hit_either = hit_target | hit_stop

            hit_indices_arr = np.where(hit_either)[0]

            if len(hit_indices_arr) > 0:
                exit_offset = hit_indices_arr[0]
                exit_idx = entry_idx + 1 + exit_offset
                exit_price = float(price[exit_idx])
                exit_ts = int(ts[exit_idx])
                reason = "target" if exit_price >= target_price else "stop"
            else:
                exit_idx = len(price) - 1
                exit_price = float(price[exit_idx])
                exit_ts = int(ts[exit_idx])
                reason = "eod"

            trades.append(BatchTrade(
                symbol=symbol,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                entry_price=entry_price,
                exit_price=exit_price,
                size=self.share_size,
                direction=1,  # long only
                exit_reason=reason,
                entry_regime=regime_at_flush,
                entry_type="flush_reclaim",
            ))

            next_allowed = exit_idx + self.cooldown

        return trades


def _cluster_flush_events(
    flush_indices: np.ndarray,
    max_gap: int,
) -> List[tuple]:
    """
    Cluster consecutive flush tick indices into discrete flush events.

    A new event starts when the gap between flush ticks exceeds max_gap.

    Returns: List of (start_idx, end_idx) tuples in original array coordinates.
    """
    if len(flush_indices) == 0:
        return []

    events = []
    event_start = flush_indices[0]
    prev = flush_indices[0]

    for idx in flush_indices[1:]:
        if idx - prev > max_gap:
            events.append((int(event_start), int(prev)))
            event_start = idx
        prev = idx

    events.append((int(event_start), int(prev)))

    return events
