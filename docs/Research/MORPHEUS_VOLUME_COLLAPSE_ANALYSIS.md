# Morpheus Trade Volume Collapse Analysis
## Why did IBKR go from 150-250 trades/day to 2-21?

**Date of change:** Feb 10-17, 2026 (gradual, multiple commits)

---

## Root Causes Identified (from git history)

### 1. STABILITY LOCK — Feb 17 (commit 89e447c1)
**7-phase infrastructure hardening permanently changed the system:**
- **Phase 3: Finviz scanner REMOVED.** IBKR's continuous_discovery.py previously ran
  Finviz + Schwab movers as dual scanner sources. Finviz was permanently removed because
  its synchronous web scraping blocked the event loop for 30-60s, crashing the server.
  Now only Max_AI advisory buffer + Schwab movers feed symbols.
- **Impact:** Fewer symbols entering the watchlist = fewer signals = fewer trades.

### 2. Regime Context Multiplier — Feb ~17 (commit 4c69b189)
**Dead-cat bounce prevention via EMA/MACD regime scoring:**
- Stocks down -21% with a +3% micro bounce now get regime_mult ~0.78 (reduced sizing
  and shorter hold) instead of full conviction.
- Regime states: VOL_SPIKE (<0.33), CHOP (0.33-0.66), RECOVERY_CONFIRMED (>0.66)
- Multiplier range [0.70, 1.15] applied to micro_score AND position sizing.
- **Impact:** Reduces conviction on a large portion of low-float momentum stocks that
  are inherently volatile. Many signals that previously scored above threshold now score below.

### 3. Falling-Knife Gate — Feb ~18 (commit a7c792eb)
**Hard blocks entries when day_change <= -8% AND price < VWAP:**
- Emits FALLING_KNIFE_DAILY_DOWNTREND block reason.
- **Impact:** Blocks a category of mean-reversion bounce entries that previously fired.

### 4. HARD_STOP_ADAPTIVE disabled — same commit
- When false, skips ALL hard stop exits. Falls through to trailing stop, max hold,
  momentum decay. Currently set to false.
- **Impact:** Changes exit behavior, not entry volume directly.

### 5. Extension Filter — Feb 11 (commit 89110c28)
- Blocks pre-market entries >30% extension without catalyst.
- **Impact:** Blocks a subset of the most extended runners pre-market.

### 6. Scalper stability hardening — Feb 12 (commit 69bc1add)
- Disable tracking, flip-flop guard, soft pause mechanisms.
- Scalper self-disabled 3x on Feb 11 before fix.
- **Impact:** Stability improvements that may have incidentally reduced trade frequency.

---

## Morpheus_AI Block Reasons (from 5-day payload)

The Morpheus_AI funnel shows WHY it's so selective:
- `regime_position_limit`: 1,184 blocks — position limits hit quickly with few slots
- `DAILY_LOSS_LIMIT: -0.1311 < -0.02`: 671 blocks — VERY tight daily loss limit (-2%)
- `LOW_SCORE: XX < 60.0`: Thousands of blocks — min score threshold = 60.0
- `NEGATIVE_L2_PRESSURE`: L2 order book pressure filtering
- `insufficient_room_to_profit`: 358 blocks
- `Signal direction misaligned with regime`: 191 blocks
- `hard_veto: 50.9% >= 45.0% no catalyst`: 157 blocks

## IBKR Block Reasons
- `SIGNAL_NOT_CONFIRMED`: 33,580 blocks — the primary gate (confluence < 70.0)
- `SPREAD_UNSTABLE`: 7,399 blocks
- `POST_STOP_COOLDOWN`: 54 blocks

---

## Key Insight

**The system that generated our 1,944-trade research dataset (Jan 28 - Feb 9) no longer exists.**

The pre-Feb-10 Morpheus was a high-frequency momentum scalper firing 150-250 trades/day with
a broad scanner (Finviz + Schwab), no regime context multiplier, no falling-knife gate, and
no extension filter.

The post-Feb-17 Morpheus is a highly selective system firing 2-21 trades/day with a narrower
scanner (Max_AI + Schwab only), regime-weighted scoring, dead-cat bounce prevention, and
multiple additional gates.

## Implications for Research Findings

1. **Price >= $5 filter** — Still relevant conceptually. The new system naturally trades fewer
   sub-$1.50 stocks due to tighter gating, but the min_price config is still 1.0. The finding
   could still be implemented as a hard floor.

2. **Hold time sweet spot (30-300s)** — max_hold_seconds is 300, so the upper bound matches.
   The 10-30s early exit issue may have changed with HARD_STOP_ADAPTIVE disabled.

3. **Hard stop deficit (-$1,911)** — HARD_STOP_ADAPTIVE is now DISABLED. The system already
   responded to this problem (though perhaps not based on our research).

4. **Exit timing** — With hard stops disabled, the exit mix has shifted to trailing stops,
   momentum decay, and max hold timeout. Different exit profile than our research data.

## Recommendation

The research findings are **directionally valid** but need re-validation on the current
system's output. At current trade rates (5-20/day combined), we need 2-4 weeks to accumulate
enough data for meaningful out-of-sample testing.

The most immediately actionable item is the **min_price floor** — changing from 1.0 to 3.0
or 5.0 in scalper_config.json is a simple config change that aligns with our research AND
the system's current selective philosophy.
