# MPAI Research — What Actually Works
## Cross-Phase Data Mining Results

**Date:** 2026-02-22
**Dataset:** 1,914 Morpheus trades (1,121 active, 793 scratch) across 18 days, 142 symbols

---

## Executive Summary

After 10 phases of microstructure research, the Databento tick-level data (pressure_z, volume acceleration, structural zones) produced **zero actionable trade filters** for Morpheus momentum scalping.

However, mining the Morpheus trade ledger itself revealed **three statistically significant patterns** that don't require any external data:

| Finding | WR Improvement | p-value | Actionable at Entry? |
|---------|---------------|---------|---------------------|
| **Price ≥ $5** | +8.2pp (53.9% vs 45.7%) | **0.0000** | ✅ YES |
| **Hold 30–300s** | +10.9pp (54.3% vs 43.4%) | **0.0000** | ❌ Post-hoc (exit logic) |
| **Price ≥ $5 + Hold 30–300s** | +9.8pp (56.5% vs 46.7%) | **0.0070** | ⚠️ Partially |

---

## Finding 1: Price Level Filter (ENTRY-TIME ACTIONABLE)

### The Signal

Stocks priced ≥ $5 at entry produce significantly higher win rates than stocks under $5.

| Group | N | Win Rate | Mean PnL | Total PnL | Med MFE | Med MAE | Hard Stop % |
|-------|---|----------|----------|-----------|---------|---------|-------------|
| Price ≥ $5 | 358 | **53.9%** | $-0.14 | $-48 | 0.098% | 0.047% | **6%** |
| Price < $5 | 763 | 45.7% | $-0.23 | $-176 | 0.012% | 0.286% | 16% |

**Permutation test: p = 0.0000 (WR), p = 0.0000 (Hard Stop rate)**

### Why It Matters

- Hard stop rate drops from 16% → 6%. That's 2.7x fewer catastrophic exits.
- Median MAE drops from 0.286% → 0.047%. Trades move against you 6x less.
- Median MFE jumps from 0.012% → 0.098%. Trades move in your favor 8x more.
- The effect is NOT just "fewer hard stops" — win rate across all exit types is higher.

### The Toxic Zone: Under $1.50

| Group | N | Win Rate | Total PnL | Hard Stop % |
|-------|---|----------|-----------|-------------|
| < $1.50 | 125 | **41.6%** | **-$272** | **37%** |
| ≥ $1.50 | 996 | 49.2% | +$47 | 10% |

Stocks under $1.50 are destroying Morpheus. Over one-third hit hard stops. These 125 trades cost $272 in 18 days.

### Price Tier Breakdown

| Price Tier | N | Win Rate | Total PnL | Hard Stop % | Timeout % |
|-----------|---|----------|-----------|-------------|-----------|
| < $1.50 | 125 | 41.6% | -$272 | 37% | 10% |
| $1.50–3 | 506 | 46.6% | +$444 | 13% | 10% |
| $3–5 | 132 | 46.2% | -$349 | 11% | 26% |
| **$5–8** | **198** | **53.0%** | **+$21** | **7%** | **8%** |
| **$8–12** | **61** | **52.5%** | **+$27** | **5%** | **10%** |
| **$12–20** | **99** | **56.6%** | **-$96** | **7%** | **13%** |

The $5+ range consistently outperforms. The $1.50–3 range has high volume but marginal quality. Under $1.50 is toxic.

### Interpretation

Low-priced stocks (< $5, especially < $1.50) are:
- More likely to be ultra-low-float penny stocks with erratic moves
- More likely to gap through stop levels (the 1% hard stop is too tight for these)
- More subject to spread costs relative to price
- More likely to be manipulated or have thin order books

This is a Morpheus configuration finding: the ignition funnel should either skip or apply different parameters to sub-$5 (or at minimum sub-$1.50) entries.

---

## Finding 2: Hold Time Sweet Spot (EXIT LOGIC INSIGHT)

### The Signal

Trades held 30–300 seconds significantly outperform those exiting faster or slower.

| Hold Time | N | Win Rate | Total PnL | Med MFE |
|-----------|---|----------|-----------|---------|
| 0–10s | 5 | 20.0% | -$10 | 0.000% |
| 10–30s | 433 | 46.2% | **-$543** | 0.000% |
| **30–60s** | **164** | **54.3%** | **+$164** | **0.127%** |
| 1–2min | 137 | 48.2% | -$405 | 0.063% |
| **2–5min** | **202** | **58.9%** | **+$503** | **0.450%** |
| 5min+ | 180 | 37.2% | +$67 | 0.037% |

**Permutation test (30–300s vs rest): p = 0.0000**

### Why It Matters

This is NOT an entry-time filter (you can't know hold time at entry). But it reveals:

1. **Early exits (< 30s) are too aggressive.** 433 trades exit within 30 seconds with 46.2% WR and -$543 total. Morpheus's decay detectors may be triggering too fast on normal momentum fluctuations.

2. **The 2–5 minute window is the most profitable.** 58.9% WR, +$503 total, highest median MFE (0.450%). Trades that have room to develop perform best.

3. **Max hold timeouts (5min+) are the second biggest drag.** 37.2% WR. Morpheus is holding losing positions too long when momentum has already failed.

### Actionable Interpretation

- Morpheus's exit sensitivity in the first 30 seconds may need tuning — too many premature exits
- The 2–5 minute hold duration is where momentum plays out. The exit logic should protect this window.
- Max hold timeout trades (377 total, 19.7% of all) are a 20% tax on the system.

---

## Finding 3: Exit Reason Profitability (SYSTEM DIAGNOSTIC)

### Exit Category Performance

| Exit Category | N | Win Rate | Mean PnL | Total PnL | Med Hold |
|--------------|---|----------|----------|-----------|----------|
| DECAY_VOLUME | 39 | **87.2%** | +$5.21 | +$203 | 102s |
| REVERSAL_EXIT | 15 | **80.0%** | +$9.20 | +$138 | 189s |
| DECAY_IMBALANCE | 26 | **76.9%** | +$10.78 | +$280 | 78s |
| DECAY_VELOCITY | 286 | **66.1%** | +$4.57 | **+$1,307** | 55s |
| DECAY_VEL+VOL | 319 | 56.7% | +$0.52 | +$167 | 29s |
| TRAIL_STOP | 151 | 45.0% | +$3.06 | +$463 | 28s |
| MAX_HOLD_TIMEOUT | 134 | **28.4%** | -$5.11 | **-$685** | 315s |
| HARD_STOP | 148 | **0.0%** | -$12.91 | **-$1,911** | 91s |

### Key Observations

1. **Momentum decay exits are the money makers.** Velocity drop alone: +$1,307. Volume collapse: +$203. Imbalance reversal: +$280. The system's momentum sensing works.

2. **Hard stops are the biggest single drag: -$1,911.** 148 trades, 0% win rate, -$12.91 average loss. This is 13% of all active trades and accounts for the bulk of system losses.

3. **Max hold timeouts: -$685.** Another 12% of active trades with 28.4% WR. These are trades where momentum never developed.

4. **Combined hard stop + timeout = 25% of active trades, -$2,596 total.** This is the entire system deficit and then some. Without these, Morpheus would be profitable.

---

## Finding 4: The 41% Scratch Rate (SYSTEM DESIGN)

793 out of 1,914 trades (41.4%) exit with exactly $0 PnL.

| Scratch Exit Reason | Count | % of Scratches |
|--------------------|-------|----------------|
| MAX_HOLD_TIMEOUT | 243 | 30.6% |
| DECAY_VEL+VOL | 218 | 27.5% |
| TRAIL_STOP | 218 | 27.5% |
| DECAY_VELOCITY | 76 | 9.6% |
| DECAY_VOLUME | 33 | 4.2% |

**41% of all trades produce zero return while incurring full commission costs.** 

The scratch rate by price level varies enormously:
- < $1.50: 26.5% scratch
- $1.50–3: **47.6%** scratch (nearly half)
- $5–8: **21.4%** scratch (lowest)
- $8–12: 61.4% scratch

The $1.50–3 range has the highest scratch rate and is where the bulk of Morpheus trades occur.

---

## What Failed (Phases 1–10)

| Research Direction | Result | Why |
|-------------------|--------|-----|
| Pressure direction → price continuation | FALSIFIED | Pressure is ubiquitous in momentum stocks |
| DPI alignment → improved accuracy | FALSIFIED | — |
| Pressure timing → ignition prediction | FALSIFIED (p=0.229) | High-pressure environment is permanent |
| Pressure slope filter | FALSIFIED (p=0.464) | Descriptive ≠ predictive |
| Volume acceleration filter | FALSIFIED (p=0.456) | Precursor but not quality signal |
| Combined gate (slope + vol_accel) | FALSIFIED (p=0.885) | — |
| PDH/PDL structural zones | NOT APPLICABLE | Stocks gap 5+ ATR from prior day levels |

**Core lesson:** Databento XNAS.ITCH trade-level microstructure does not contain actionable quality signals for $1–$20 low-float momentum scalping. These stocks are too volatile, too retail-driven, and too disconnected from typical microstructure assumptions.

---

## What Survived (HYP-003)

**Inventory FADE in high volatility reverts.** Phase 8 confirmed: 56.1% hit rate, 2.22 R:R, n=435. This is a standalone reversion signal, not a Morpheus integration signal. It could potentially be developed as its own strategy on different instruments.

---

## Recommended Actions for Morpheus

These are derived from the trade ledger analysis above. None require Databento data.

### Immediate (Configuration Changes)

1. **Evaluate minimum price threshold.** Sub-$1.50 stocks have 41.6% WR, 37% hard stop rate, and -$272 in 18 days. Consider excluding or adding a price floor parameter.

2. **Review hard stop sizing for low-priced stocks.** A 1% hard stop on a $1.20 stock is $0.012 — one tick can trigger it. The stop may need to be ATR-based or percentage-adjusted by price tier.

### Short-Term (Exit Logic Analysis)

3. **Analyze 10–30s early exit pattern.** 433 trades exit in under 30s with 46.2% WR and -$543. Are the decay velocity/volume detectors firing too aggressively on normal momentum noise?

4. **Investigate max hold timeout trades.** 377 timeouts (19.7% of all trades, -$685 for active ones). What do these look like? Can momentum state at the 3-minute mark predict whether to keep holding or cut early?

### Medium-Term (Research)

5. **HYP-003 standalone development.** The inventory fade reversion signal (56.1% WR, 2.22 R:R) could be a separate strategy targeting liquid mid/large cap names where microstructure signals are stronger.

---

## Files

| File | Description |
|------|-------------|
| `results/phase9v2_event_data.csv` | Per-event data with pressure profiles (420 events) |
| `results/phase10b_event_data.csv` | Events with zone labels |
| `results/phase9v2_results.json` | Phase 9 v2 experiment results |
| `results/phase10b_results.json` | Phase 10B zone results |
| `docs/Research/PHASE9_Results.md` | Phase 9 v2 formal report |
| `docs/Research/PHASE10B_StructuralZones.md` | Phase 10B structural zones report |

---

*Generated by Claude AI — 2026-02-22*
*Research Server: C:\AI_Bot_Research | GitHub: bgmaynard/AI_Bot_Research*
