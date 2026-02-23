"""
Morpheus Lab — Momentum Continuation v2 (Regime-Gated)
=========================================================
Evolution of momentum_breakout with:
  1. Regime gate: only trades in favorable regimes
  2. Entry type classification: tags each entry as breakout/pullback/flush_reclaim
  3. Same vectorized engine, structurally filtered

Regime Gate (configurable):
  ALLOW: TREND_DOWN, TREND_UP, LOW_VOL_CHOP (by default)
  BLOCK: PARABOLIC_EXTEND, LIQUIDITY_FADE (by default)
  HIGH_VOL_BREAKOUT: configurable (default: allow)

Entry Type Classification:
  true_breakout:     Price at new rolling high + volume surge
  pullback:          Price below recent high, bouncing off VWAP support
  flush_reclaim:     Price was below VWAP, reclaimed with volume

This classification is diagnostic — it tags trades for analysis
but does NOT filter them. We measure first, then decide.

Parameters (inherited):
  lookback, vol_surge, breakout_pct, target_pct, stop_pct, cooldown, share_size

New parameters:
  allowed_regimes:   List of regimes to trade in
  classify_entries:  Whether to tag entry types (small perf cost)
"""

import numpy as np
from typing import List, Optional, Set

from strategies.batch_strategy import BatchStrategy, BatchTrade
from engine.regime_classifier import (
    classify_tick_regime,
    LOW_VOL_CHOP, TREND_UP, TREND_DOWN,
    HIGH_VOL_BREAKOUT, PARABOLIC_EXTEND, LIQUIDITY_FADE,
    ALL_REGIMES,
)


# Default regime gate
DEFAULT_ALLOWED = {TREND_DOWN, TREND_UP, LOW_VOL_CHOP, HIGH_VOL_BREAKOUT}
DEFAULT_BLOCKED = {PARABOLIC_EXTEND, LIQUIDITY_FADE}


class MomentumContinuationV2(BatchStrategy):
    """Regime-gated momentum strategy with entry type classification."""

    name = "momentum_continuation_v2"

    def __init__(
        self,
        lookback: int = 200,
        vol_surge: float = 2.0,
        breakout_pct: float = 0.5,
        target_pct: float = 2.0,
        stop_pct: float = 1.0,
        cooldown: int = 50,
        share_size: int = 100,
        allowed_regimes: Optional[str] = None,
        regime_window: int = 200,
        classify_entries: bool = True,
    ):
        super().__init__(
            lookback=lookback,
            vol_surge=vol_surge,
            breakout_pct=breakout_pct,
            target_pct=target_pct,
            stop_pct=stop_pct,
            cooldown=cooldown,
            share_size=share_size,
        )
        self.lookback = lookback
        self.vol_surge = vol_surge
        self.breakout_pct = breakout_pct / 100.0
        self.target_pct = target_pct / 100.0
        self.stop_pct = stop_pct / 100.0
        self.cooldown = cooldown
        self.share_size = share_size
        self.regime_window = regime_window
        self.classify_entries = classify_entries

        # Parse allowed regimes
        if allowed_regimes:
            self.allowed_regimes = set(r.strip() for r in allowed_regimes.split(","))
        else:
            self.allowed_regimes = DEFAULT_ALLOWED.copy()

    def on_batch(
        self,
        ts: np.ndarray,
        price: np.ndarray,
        size: np.ndarray,
        symbol: str,
    ) -> List[BatchTrade]:
        """
        Regime-gated signal detection with entry classification.

        1. Compute regimes for entire batch (vectorized)
        2. Detect signals (same as v1)
        3. Gate: skip entries in blocked regimes
        4. Classify: tag each entry type
        5. Exit scan (same as v1)
        """
        n = len(price)
        if n < self.lookback + 10:
            return []

        lb = self.lookback

        # ── REGIME CLASSIFICATION (vectorized, once per batch) ──

        regimes = classify_tick_regime(
            price, size, window=self.regime_window
        )

        # ── VECTORIZED SIGNAL DETECTION (same as v1) ────────

        dollar_vol = price * size.astype(np.float64)
        cum_dollar = np.cumsum(dollar_vol)
        cum_size = np.cumsum(size.astype(np.float64))

        roll_dollar = cum_dollar[lb:] - cum_dollar[:-lb]
        roll_size = cum_size[lb:] - cum_size[:-lb]

        valid = roll_size > 0
        vwap = np.zeros(n - lb)
        vwap[valid] = roll_dollar[valid] / roll_size[valid]

        price_aligned = price[lb:]
        avg_vol = roll_size / lb
        current_size = size[lb:].astype(np.float64)

        # Signal conditions
        breakout = price_aligned > vwap * (1 + self.breakout_pct)
        vol_spike = current_size > (avg_vol * self.vol_surge)
        signals = breakout & vol_spike & (vwap > 0)

        signal_indices = np.where(signals)[0] + lb

        if len(signal_indices) == 0:
            return []

        # ── PRE-COMPUTE ENTRY CLASSIFICATION DATA ────────────

        if self.classify_entries:
            # Rolling high over lookback (for breakout detection)
            rolling_high = _rolling_max(price, lb)

            # Price relative to VWAP (for pullback/reclaim detection)
            # vwap_full[i] = vwap for index i (only valid from lb onward)
            vwap_full = np.zeros(n)
            vwap_full[lb:] = vwap

            # Recent price trajectory: was price below VWAP recently?
            # Check if any of the last `lookback//4` ticks were below VWAP
            check_window = max(lb // 4, 10)

        # ── REGIME-GATED EXIT SCANNING ───────────────────────

        trades: List[BatchTrade] = []
        next_allowed = 0
        _allowed = self.allowed_regimes

        for entry_idx in signal_indices:
            if entry_idx < next_allowed:
                continue

            # ── REGIME GATE ──────────────────────────────────
            regime_at_entry = str(regimes[entry_idx])
            if regime_at_entry not in _allowed:
                continue

            entry_price = float(price[entry_idx])
            entry_ts = int(ts[entry_idx])

            if entry_price <= 0:
                continue

            # ── ENTRY TYPE CLASSIFICATION ────────────────────
            entry_type = ""
            if self.classify_entries:
                entry_type = _classify_entry(
                    price, vwap_full, rolling_high,
                    entry_idx, lb, check_window
                )

            # ── EXIT SCAN (same as v1) ───────────────────────
            target_price = entry_price * (1 + self.target_pct)
            stop_price = entry_price * (1 - self.stop_pct)

            exit_slice = price[entry_idx + 1:]
            if len(exit_slice) == 0:
                continue

            hit_target = exit_slice >= target_price
            hit_stop = exit_slice <= stop_price
            hit_either = hit_target | hit_stop

            hit_indices = np.where(hit_either)[0]

            if len(hit_indices) > 0:
                exit_offset = hit_indices[0]
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
                direction=1,
                exit_reason=reason,
                entry_regime=regime_at_entry,
                entry_type=entry_type,
            ))

            next_allowed = exit_idx + self.cooldown

        return trades


def _classify_entry(
    price: np.ndarray,
    vwap_full: np.ndarray,
    rolling_high: np.ndarray,
    entry_idx: int,
    lookback: int,
    check_window: int,
) -> str:
    """
    Classify an entry into one of three types:

    true_breakout:     Price at/near rolling high AND above VWAP
                       (new high breakout with momentum)

    pullback:          Price was recently higher, pulled back toward VWAP,
                       now bouncing (price > VWAP but below recent high)

    flush_reclaim:     Price was below VWAP within check_window,
                       now reclaimed above (mean reversion entry)
    """
    ep = price[entry_idx]
    ev = vwap_full[entry_idx]
    rh = rolling_high[entry_idx]

    if ev <= 0 or rh <= 0:
        return "unclassified"

    # Was price below VWAP recently?
    start = max(0, entry_idx - check_window)
    recent_prices = price[start:entry_idx]
    recent_vwap = vwap_full[start:entry_idx]

    was_below_vwap = False
    if len(recent_prices) > 0 and len(recent_vwap) > 0:
        valid_mask = recent_vwap > 0
        if np.any(valid_mask):
            was_below_vwap = np.any(recent_prices[valid_mask] < recent_vwap[valid_mask])

    # Classification logic
    near_high_threshold = 0.995  # within 0.5% of rolling high

    if ep >= rh * near_high_threshold:
        # At or near rolling high
        return "true_breakout"
    elif was_below_vwap:
        # Was below VWAP recently, now above = reclaim
        return "flush_reclaim"
    else:
        # Above VWAP but below recent high = pullback bounce
        return "pullback"


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum over window. Returns array of same length."""
    n = len(arr)
    result = np.zeros(n)

    if n < window:
        # Just expanding max
        result = np.maximum.accumulate(arr)
        return result

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(arr, window)
    w_max = np.max(windows, axis=1)

    # windows[i] = arr[i:i+window], result at i+window-1
    result[:window] = np.maximum.accumulate(arr[:window])
    result[window - 1: window - 1 + len(w_max)] = w_max

    return result
