"""
Morpheus Lab — Regime Classifier
====================================
Vectorized market regime detection from trade tick data.

Classifies each tick into one of six regimes based on:
  - Rolling volatility (price range / mean)
  - Trend direction (linear slope)
  - Volume intensity (rolling vs baseline)
  - Range expansion (current vs average)

Regimes:
  LOW_VOL_CHOP      - Tight range, no trend, normal volume
  TREND_UP           - Positive slope, moderate volatility
  TREND_DOWN         - Negative slope, moderate volatility
  HIGH_VOL_BREAKOUT  - Range expansion + volume surge + any direction
  PARABOLIC_EXTEND   - Extreme price acceleration + high volume
  LIQUIDITY_FADE     - Volume collapse + widening noise

All operations are vectorized numpy. No per-tick Python loops.
"""

import numpy as np
from typing import Optional

# Regime constants
LOW_VOL_CHOP = "LOW_VOL_CHOP"
TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
HIGH_VOL_BREAKOUT = "HIGH_VOL_BREAKOUT"
PARABOLIC_EXTEND = "PARABOLIC_EXTEND"
LIQUIDITY_FADE = "LIQUIDITY_FADE"

ALL_REGIMES = [
    LOW_VOL_CHOP,
    TREND_UP,
    TREND_DOWN,
    HIGH_VOL_BREAKOUT,
    PARABOLIC_EXTEND,
    LIQUIDITY_FADE,
]


def classify_tick_regime(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 200,
    vol_surge_mult: float = 2.0,
    trend_threshold: float = 0.0003,
    parabolic_threshold: float = 0.001,
    range_expansion_mult: float = 2.0,
    fade_vol_pct: float = 0.3,
    breakout_fast_window: int = 20,
    breakout_price_pct: float = 0.02,
) -> np.ndarray:
    """
    Classify each tick into a market regime.

    Args:
        prices: float64 price array
        volumes: int/float volume (trade size) array
        window: lookback window for rolling calculations
        vol_surge_mult: volume > N× rolling avg = surge
        trend_threshold: min slope per tick for trend
        parabolic_threshold: slope for parabolic detection
        range_expansion_mult: range > N× avg = expansion
        fade_vol_pct: volume < N% of avg = liquidity fade
        breakout_fast_window: short window for breakout price move
        breakout_price_pct: min price change % in fast window for breakout

    Returns:
        string array of regime labels, same length as prices.
        First `window` elements are labeled LOW_VOL_CHOP (insufficient data).
    """
    n = len(prices)
    regimes = np.full(n, LOW_VOL_CHOP, dtype=object)

    if n < window + 10:
        return regimes

    # ── ROLLING CALCULATIONS (all vectorized) ────────────

    # 1. Rolling volatility: std(returns) over window
    log_prices = np.log(np.maximum(prices, 1e-8))
    returns = np.diff(log_prices)
    returns = np.concatenate([[0.0], returns])

    rolling_vol = _rolling_std(returns, window)

    # 2. Rolling trend slope (normalized)
    rolling_slope = _rolling_slope(prices, window)

    # 3. Rolling volume mean
    vol_float = volumes.astype(np.float64)
    rolling_vol_mean = _rolling_mean(vol_float, window)
    current_vol = vol_float

    # 4. Rolling price range ratio
    rolling_range = _rolling_range_ratio(prices, window)
    baseline_window = min(window * 3, n)
    baseline_range = _rolling_range_ratio(prices, baseline_window)

    # 5. Fast price acceleration: |price change| over short window / price
    #    This catches breakouts immediately rather than waiting for range expansion
    fast_w = min(breakout_fast_window, window)
    fast_price_change = np.zeros(n)
    fast_price_change[fast_w:] = np.abs(prices[fast_w:] - prices[:-fast_w]) / np.maximum(prices[:-fast_w], 1e-8)

    # ── REGIME CLASSIFICATION (vectorized boolean masks) ──

    start = window

    vol = rolling_vol[start:]
    slope = rolling_slope[start:]
    vmean = rolling_vol_mean[start:]
    cvol = current_vol[start:]
    rng = rolling_range[start:]
    brng = baseline_range[start:]
    fast_pct = fast_price_change[start:]

    vmean_safe = np.maximum(vmean, 1e-8)
    brng_safe = np.maximum(brng, 1e-8)

    vol_ratio = cvol / vmean_safe
    range_ratio = rng / brng_safe

    # ── MASKS (priority order: most specific first) ──────

    # PARABOLIC: extreme slope + high volume
    is_parabolic = (
        (np.abs(slope) > parabolic_threshold) &
        (vol_ratio > vol_surge_mult * 0.8)
    )

    # HIGH_VOL_BREAKOUT: either range expansion OR fast price move, + volume surge
    is_breakout = (
        ~is_parabolic &
        (vol_ratio > vol_surge_mult) &
        ((range_ratio > range_expansion_mult) | (fast_pct > breakout_price_pct))
    )

    # LIQUIDITY_FADE: volume collapse
    is_fade = (
        ~is_parabolic &
        ~is_breakout &
        (vol_ratio < fade_vol_pct)
    )

    # TREND_UP: positive slope
    is_trend_up = (
        ~is_parabolic &
        ~is_breakout &
        ~is_fade &
        (slope > trend_threshold)
    )

    # TREND_DOWN: negative slope
    is_trend_down = (
        ~is_parabolic &
        ~is_breakout &
        ~is_fade &
        (slope < -trend_threshold)
    )

    # LOW_VOL_CHOP: everything else (already default)

    # ── APPLY LABELS ─────────────────────────────────────

    regime_slice = regimes[start:]
    regime_slice[is_trend_up] = TREND_UP
    regime_slice[is_trend_down] = TREND_DOWN
    regime_slice[is_fade] = LIQUIDITY_FADE
    regime_slice[is_breakout] = HIGH_VOL_BREAKOUT
    regime_slice[is_parabolic] = PARABOLIC_EXTEND

    return regimes


def get_regime_at_index(
    regimes: np.ndarray,
    index: int,
) -> str:
    """Get regime label at a specific index."""
    if index < 0 or index >= len(regimes):
        return LOW_VOL_CHOP
    return str(regimes[index])


def regime_summary(regimes: np.ndarray) -> dict:
    """
    Summarize regime distribution.

    Returns: {regime_name: {'count': N, 'pct': float}}
    """
    n = len(regimes)
    summary = {}
    for regime in ALL_REGIMES:
        count = int(np.sum(regimes == regime))
        summary[regime] = {
            'count': count,
            'pct': round(count / n * 100, 1) if n > 0 else 0.0,
        }
    return summary


# ── ROLLING HELPER FUNCTIONS (vectorized) ────────────────

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean using cumsum trick. Returns array of same length."""
    n = len(arr)
    result = np.zeros(n)
    cumsum = np.cumsum(arr)

    result[window:] = (cumsum[window:] - cumsum[:-window]) / window
    # Fill early values with expanding mean
    if window > 0:
        result[:window] = cumsum[:window] / np.arange(1, window + 1)

    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation. Returns array of same length."""
    n = len(arr)
    result = np.zeros(n)

    # Use cumsum for mean, cumsum of squares for variance
    cumsum = np.cumsum(arr)
    cumsum_sq = np.cumsum(arr ** 2)

    for_mean = np.zeros(n)
    for_var = np.zeros(n)

    for_mean[window:] = (cumsum[window:] - cumsum[:-window]) / window
    for_var[window:] = (cumsum_sq[window:] - cumsum_sq[:-window]) / window

    variance = for_var - for_mean ** 2
    variance = np.maximum(variance, 0)  # numerical safety
    result[window:] = np.sqrt(variance[window:])

    return result


def _rolling_slope(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling linear regression slope, normalized by mean price.

    Uses the fast formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    where x = [0, 1, ..., window-1] and y = prices in window.

    Returns per-tick slope (price change per tick, normalized).
    """
    n = len(prices)
    result = np.zeros(n)

    if n < window:
        return result

    # x values are constant: 0, 1, ..., window-1
    # Σx = window*(window-1)/2
    # Σx² = window*(window-1)*(2*window-1)/6
    w = window
    sum_x = w * (w - 1) / 2
    sum_x2 = w * (w - 1) * (2 * w - 1) / 6
    denom = w * sum_x2 - sum_x ** 2

    if denom == 0:
        return result

    # We need rolling Σy and rolling Σ(x*y)
    # For Σy: just rolling sum of prices
    cum_y = np.cumsum(prices)

    # For Σ(x*y): x is the position within window
    # x[i] = i for i in [0, window)
    # Trick: Σ(x*y) for window ending at j = Σ(i * price[j-w+1+i]) for i in [0,w)
    # = Σ(i * price[j-w+1+i])
    # This can be computed as: j*Σy - rolling_sum(cumsum(prices))
    # Actually, use the weighted sum approach:

    # weighted[k] = k * prices[k] ... but k resets per window
    # Simpler: use the identity
    # Σ(i * y[start+i]) = (start+0)*y[start] + (start+1)*y[start+1] + ...
    # No, we need position within window, not absolute.

    # Fast approach: use two cumulative sums
    # Let idx = np.arange(n)
    # weighted_prices = idx * prices
    # cum_wy = cumsum(weighted_prices)
    # rolling_wy[j] = cum_wy[j] - cum_wy[j-w]  (sum of absolute_idx * price)
    # But we need relative_idx within window, so:
    # Σ(relative_i * y[j-w+1+i]) = Σ((j-w+1+i) * y[...]) - (j-w+1)*Σy
    #                              = rolling_wy - (j-w+1) * rolling_y

    idx = np.arange(n, dtype=np.float64)
    weighted = idx * prices
    cum_wy = np.cumsum(weighted)

    # Rolling sums starting from index `window`
    rolling_y = np.zeros(n)
    rolling_wy = np.zeros(n)

    rolling_y[w:] = cum_y[w:] - cum_y[:-w]
    rolling_y[w - 1] = cum_y[w - 1]

    rolling_wy[w:] = cum_wy[w:] - cum_wy[:-w]
    rolling_wy[w - 1] = cum_wy[w - 1]

    # Convert absolute-index weighted sum to relative-index weighted sum
    # window_start[j] = j - w + 1
    window_start = idx - w + 1
    sum_xy = rolling_wy - window_start * rolling_y  # relative position * price

    # Slope = (n * Σxy - Σx * Σy) / denom
    raw_slope = (w * sum_xy - sum_x * rolling_y) / denom

    # Normalize by rolling mean price
    rolling_mean_price = np.zeros(n)
    rolling_mean_price[w:] = rolling_y[w:] / w
    rolling_mean_price[w - 1] = rolling_y[w - 1] / w
    rolling_mean_price = np.maximum(rolling_mean_price, 1e-8)

    result[w:] = raw_slope[w:] / rolling_mean_price[w:]
    if w > 0 and w - 1 < n:
        result[w - 1] = raw_slope[w - 1] / rolling_mean_price[w - 1]

    return result


def _rolling_range_ratio(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling (max-min)/mean over window.

    Measures price range relative to price level.
    """
    n = len(prices)
    result = np.zeros(n)

    if n < window:
        return result

    # Sliding window max/min — use stride tricks for small windows,
    # or iterate for arbitrary windows. For our use case (window=200-600),
    # a simple approach with numpy is fine.

    # Efficient approach: use a rolling buffer
    # For windows up to ~1000, this is fast enough
    from numpy.lib.stride_tricks import sliding_window_view

    if n >= window:
        windows = sliding_window_view(prices, window)
        # windows[i] = prices[i:i+window]
        w_max = np.max(windows, axis=1)
        w_min = np.min(windows, axis=1)
        w_mean = np.mean(windows, axis=1)
        w_mean = np.maximum(w_mean, 1e-8)

        ratio = (w_max - w_min) / w_mean
        # windows[i] corresponds to prices[i:i+window], so result at index i+window-1
        result[window - 1: window - 1 + len(ratio)] = ratio

    return result
