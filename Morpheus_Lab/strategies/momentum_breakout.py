"""
Morpheus Lab — Momentum Breakout Strategy (Vectorized)
=========================================================
Research prototype: detects volume-confirmed price breakouts
using pure numpy operations. No per-tick Python loops in signal
detection.

Logic:
  1. Compute rolling VWAP over lookback window
  2. Detect volume surges (rolling vol > N× average)
  3. Entry when: price breaks above VWAP + threshold AND volume surge
  4. Exit: fixed % target or stop, scanned vectorized
  5. One position at a time per symbol

Parameters:
  lookback:       Rolling window for VWAP (trade count)
  vol_surge:      Volume multiplier for surge detection
  breakout_pct:   Price above VWAP to trigger entry (%)
  target_pct:     Profit target from entry (%)
  stop_pct:       Stop loss from entry (%)
  cooldown:       Min trades between entries
"""

import numpy as np
from typing import List

from strategies.batch_strategy import BatchStrategy, BatchTrade


class MomentumBreakout(BatchStrategy):
    """Vectorized momentum breakout for low-float stocks."""

    name = "momentum_breakout"

    def __init__(
        self,
        lookback: int = 200,
        vol_surge: float = 2.0,
        breakout_pct: float = 0.5,
        target_pct: float = 2.0,
        stop_pct: float = 1.0,
        cooldown: int = 50,
        share_size: int = 100,
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

    def on_batch(
        self,
        ts: np.ndarray,
        price: np.ndarray,
        size: np.ndarray,
        symbol: str,
    ) -> List[BatchTrade]:
        """
        Vectorized signal detection + forward-scan exit.

        Signal detection is fully vectorized (numpy).
        Exit scanning uses a tight loop only over signal indices.
        """
        n = len(price)
        if n < self.lookback + 10:
            return []

        # ── VECTORIZED SIGNAL DETECTION ──────────────────────

        # Rolling VWAP: cumulative(price*size) / cumulative(size)
        dollar_vol = price * size.astype(np.float64)
        cum_dollar = np.cumsum(dollar_vol)
        cum_size = np.cumsum(size.astype(np.float64))

        lb = self.lookback
        # Rolling sums via shifted cumsum
        roll_dollar = cum_dollar[lb:] - cum_dollar[:-lb]
        roll_size = cum_size[lb:] - cum_size[:-lb]

        # Avoid division by zero
        valid = roll_size > 0
        vwap = np.zeros(n - lb)
        vwap[valid] = roll_dollar[valid] / roll_size[valid]

        # Align: vwap[i] corresponds to price[i + lb]
        # Price slice for comparison
        price_aligned = price[lb:]

        # Rolling average volume
        avg_vol = roll_size / lb  # average size per trade in window
        current_size = size[lb:].astype(np.float64)

        # ── SIGNAL CONDITIONS (all vectorized) ───────────────

        # Price above VWAP + threshold
        breakout = price_aligned > vwap * (1 + self.breakout_pct)

        # Volume surge: current trade size > vol_surge × rolling average
        vol_spike = current_size > (avg_vol * self.vol_surge)

        # Combined signal
        signals = breakout & vol_spike & (vwap > 0)

        # Signal indices (in original array coordinates)
        signal_indices = np.where(signals)[0] + lb

        if len(signal_indices) == 0:
            return []

        # ── EXIT SCANNING (tight loop over signals only) ─────

        trades: List[BatchTrade] = []
        next_allowed = 0  # cooldown tracker

        for entry_idx in signal_indices:
            if entry_idx < next_allowed:
                continue

            entry_price = float(price[entry_idx])
            entry_ts = int(ts[entry_idx])

            if entry_price <= 0:
                continue

            target_price = entry_price * (1 + self.target_pct)
            stop_price = entry_price * (1 - self.stop_pct)

            # Forward scan from entry to end of batch
            exit_slice = price[entry_idx + 1:]
            if len(exit_slice) == 0:
                continue

            # Vectorized: find first target or stop hit
            hit_target = exit_slice >= target_price
            hit_stop = exit_slice <= stop_price
            hit_either = hit_target | hit_stop

            hit_indices = np.where(hit_either)[0]

            if len(hit_indices) > 0:
                exit_offset = hit_indices[0]
                exit_idx = entry_idx + 1 + exit_offset
                exit_price = float(price[exit_idx])
                exit_ts = int(ts[exit_idx])

                if exit_price >= target_price:
                    reason = "target"
                else:
                    reason = "stop"
            else:
                # No exit found — close at end of batch
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
            ))

            next_allowed = exit_idx + self.cooldown

        return trades
