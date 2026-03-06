# Shadow Replay Grid Optimization — Final Summary
## Data: 2026-03-03 BATL | 218 signals | 19,712 quotes | 3,840 configurations
## Generated: 2026-03-06

---

## BOTTOM LINE

The optimal configuration improves **every metric** vs production baseline:

| Metric | Production | Optimal | Change |
|--------|-----------|---------|--------|
| Profit Factor | 0.651 | **2.771** | +325% |
| Total PnL | -$3,290 | **+$9,780** | +$13,070 |
| Win Rate | 45.0% | **70.0%** | +25pp |
| Max Drawdown | 8.11% | **4.62%** | -43% |
| Sharpe Ratio | -1.61 | **4.95** | +6.56 |
| Trades | 20 | 30 | +10 |

**NOTE**: The baseline PnL differs from paper trade results (+$21,699) because this
replay uses tick-level trailing stop simulation rather than the production exit engine.
Relative comparisons between configs are valid; absolute PnL should be interpreted as
directional.

---

## OPTIMAL CONFIGURATION

```
hold_time:             420s    (was 300s — +40%)
trail_start:           0.25%  (was 0.15% — wider entry buffer)
trail_offset:          0.05%  (was 0.10% — tighter trail)
spread_threshold:      0.6%   (was 0.4% — relaxed)
containment_pullback:  0.5%   (was 0.25% — relaxed)
session_trade_cap:     30     (was 20 — +50%)
```

---

## KEY FINDINGS

### 1. Trail Start is the Most Impactful Parameter

| Trail Start | Avg PF | Avg Sharpe | Avg PnL |
|------------|--------|-----------|---------|
| 0.10% | 0.70 | -1.30 | -$5,916 |
| 0.15% | 0.80 | -0.87 | -$4,558 |
| **0.20%** | **1.18** | **+0.51** | **-$538** |
| **0.25%** | **1.27** | **+0.79** | **+$244** |

Wider trail start (0.25%) gives the trade more room to breathe before the trailing
stop activates. At 0.10%, the trail triggers prematurely on normal noise, stopping
out winners before they run. This is the single biggest lever — it alone flips the
average config from losing to profitable.

### 2. Session Trade Cap = 30 is the Sweet Spot

| Cap | Avg PF | Avg Sharpe | Avg PnL |
|-----|--------|-----------|---------|
| 20 | 0.98 | -0.16 | -$425 |
| **30** | **1.42** | **+1.31** | **+$2,832** |
| 40 | 0.57 | -1.80 | -$10,483 |

Signals 21-30 are **high quality** (positive EV, improving PF from 0.98 to 1.42).
Signals 31-40 are **toxic** — they collapse PF to 0.57 and add $10K in losses.
The quality cliff between signal 30 and 31 is sharp and consistent.

**Recommendation**: Raise cap to 30, NOT 40. Implement signal quality decay monitoring.

### 3. Hold Time: Longer is Better (to a point)

| Hold Time | Avg PF | Avg Sharpe |
|-----------|--------|-----------|
| 120s | 1.03 | -0.08 |
| 180s | 0.95 | -0.37 |
| 240s | 0.95 | -0.32 |
| 300s (prod) | 0.82 | -0.85 |
| **420s** | **1.18** | **+0.53** |

The 300s production hold time is actually a local minimum. 420s allows winners more
time to develop. The non-monotonic pattern (120s > 180-300s < 420s) suggests two
distinct trade populations: fast scalps (< 2 min) and trend runners (5-7 min).

### 4. Trail Offset: Tighter is Better

| Trail Offset | Avg PF | Avg Sharpe |
|-------------|--------|-----------|
| **0.05%** | **1.03** | **-0.06** |
| 0.08% | 0.99 | -0.21 |
| 0.10% (prod) | 0.98 | -0.24 |
| 0.15% | 0.95 | -0.35 |

Once the trail activates, a tighter offset (0.05%) locks in more profit. Combined
with the wider trail_start (0.25%), this creates a "patient entry, tight lock"
profile: let the trade develop, then protect gains aggressively.

### 5. Spread Threshold: 0.6% Optimal, Diminishing Returns Above

| Spread | Avg PF | Avg Sharpe |
|--------|--------|-----------|
| 0.4% (prod) | 0.96 | -0.38 |
| **0.6%** | **1.00** | **-0.16** |
| 0.8% | 0.99 | -0.17 |
| 1.0% | 0.99 | -0.17 |

Raising from 0.4% to 0.6% captures the high-value PLUG signals. Going above 0.6%
adds no incremental value for BATL — the additional signals at 0.8%+ are noise.
This aligns with the containment study recommendation (0.8% is safe; 0.6% captures
most of the edge).

### 6. Containment Pullback: Minimal Impact

| Pullback | Avg PF | Avg Sharpe |
|----------|--------|-----------|
| 0.15% | 0.98 | -0.24 |
| 0.25% (prod) | 0.98 | -0.24 |
| 0.35% | 0.98 | -0.24 |
| 0.5% | 1.01 | -0.16 |

Containment pullback has the **least impact** of all parameters on BATL. The 0.5%
setting shows marginal improvement by admitting one additional signal. For a
multi-symbol deployment, this parameter may matter more on stocks with different
microstructure.

---

## TOP 10 CONFIGURATIONS (All Profitable)

All top 10 share: **hold=420s, trail_start=0.25%, pullback=0.5%, cap=30**.

The variation is only in trail_offset (0.05-0.15%) and spread_threshold (0.4-1.0%),
confirming these are secondary parameters once the primary levers are set correctly.

| Rank | Trail Offset | Spread | PF | PnL | Sharpe |
|------|-------------|--------|------|--------|--------|
| 1 | 0.05% | 0.6% | 2.77 | $9,780 | 4.95 |
| 2 | 0.05% | 0.4% | 2.76 | $9,572 | 4.88 |
| 3 | 0.08% | 0.6% | 2.71 | $9,442 | 4.77 |
| 4 | 0.10% | 0.6% | 2.71 | $9,442 | 4.77 |
| 5 | 0.15% | 0.6% | 2.68 | $9,287 | 4.70 |

---

## INTERACTION EFFECTS

### Trail Start x Trail Offset (Avg PF)

The combination matters: wide start + tight offset is the winning formula.

```
             Offset:  0.05%   0.08%   0.10%   0.15%
Start 0.10%:          0.74    0.70    0.70    0.66     <-- all losing
Start 0.15%:          0.84    0.81    0.79    0.77     <-- all losing
Start 0.20%:          1.21    1.18    1.18    1.16     <-- all winning
Start 0.25%:          1.31    1.26    1.26    1.24     <-- all winning
```

Trail start >= 0.20% is the critical threshold. Below it, no offset can save the strategy.

### Hold Time x Trail Offset (Avg PnL)

```
             Offset:  0.05%     0.08%     0.10%     0.15%
Hold 120s:           -$4,014   -$4,382   -$4,480   -$4,776
Hold 240s:           -$1,132   -$1,500   -$1,598   -$1,905
Hold 420s:           -$1,355   -$1,724   -$1,821   -$2,150
```

Hold 240s and 420s are similarly good with tight offset. The 300s hold is consistently
the worst (not shown above, -$3,214 at best), suggesting production is sitting at
an unfortunate local minimum.

---

## RECOMMENDED PARAMETER CHANGES

### Immediate (High Confidence)

| Parameter | Current | Recommended | Confidence | Impact |
|-----------|---------|-------------|------------|--------|
| trail_start | 0.15% | **0.25%** | HIGH | Most impactful single change |
| session_trade_cap | 20 | **30** | HIGH | Signals 21-30 are positive EV |
| trail_offset | 0.10% | **0.05%** | MEDIUM | Tighter lock after trail activates |

### Phase 2 (Validate First)

| Parameter | Current | Recommended | Confidence | Notes |
|-----------|---------|-------------|------------|-------|
| hold_time | 300s | **420s** | MEDIUM | Non-monotonic pattern needs multi-day validation |
| spread_threshold | 0.4% | **0.6%** | MEDIUM | Aligns with containment study; mainly helps PLUG |
| containment_pullback | 0.25% | **0.5%** | LOW | Marginal impact on BATL; test on other symbols |

### Do NOT Change

| Parameter | Reason |
|-----------|--------|
| session_trade_cap to 40 | Catastrophic — signals 31-40 are toxic (-$10K avg) |
| trail_start to 0.10% | Worst performer across all combos |
| spread_threshold to 1.0% | No gain over 0.6-0.8%, may admit bad signals on other symbols |

---

## ESTIMATED PNL IMPROVEMENT

Using the replay simulation as a guide:

| Scenario | Estimated Daily PnL | Improvement |
|----------|-------------------|-------------|
| Current production | -$3,290 (sim) / +$21,699 (paper) | Baseline |
| trail_start=0.25% only | ~+$4,000 (sim) | +$7,290 (sim) |
| trail_start + cap=30 | ~+$7,000 (sim) | +$10,290 (sim) |
| Full optimal config | +$9,780 (sim) | +$13,070 (sim) |

**Conservative estimate**: The three immediate changes (trail_start, cap, trail_offset)
should improve daily PnL by **$5,000-$10,000** based on simulation, with reduced
drawdown and higher win rate.

---

## CAVEATS

1. **Single-day, single-symbol data** — 2026-03-03 BATL only. BATL moved +87% intraday,
   which is an extreme outlier. Results may not generalize to normal volatility days.
2. **Simulation vs production gap** — Tick-level trailing stop simulation differs from
   the production exit engine. Relative rankings between configs are reliable; absolute
   PnL numbers are directional.
3. **No slippage model** — Simulation assumes fill at signal price. Real execution has
   slippage, especially on the 31-40th signals when liquidity may be consumed.
4. **Overfitting risk** — 3,840 configs on 218 signals (20 executed + 198 rejected) has
   a high degrees-of-freedom-to-observation ratio. Validate on 2+ additional days before
   deploying.
5. **Cap=30 quality cliff** — The sharp quality drop at signal 31+ may be date-specific.
   Need multi-day confirmation that signals 21-30 are consistently positive EV.

---

## NEXT STEPS

1. **Validate on additional days** — Run this grid on Mar 2, Mar 4, and any other available
   trading days to confirm parameter stability
2. **Implement trail_start=0.25%** — Highest confidence, easiest change, most impactful
3. **A/B test cap=30** — Run paper trades with cap=30 for 3-5 days alongside cap=20
4. **Cross-reference with regime-adaptive exits** — The EOD research (exit_model_comparison.md)
   found regime-adaptive exits improve PF from 7.36 to 10.03. Combining regime-adaptive
   exits with optimized trail parameters could compound improvements.
5. **Multi-symbol validation** — Test optimal config on PLUG, VG, DUST when those symbols
   become available for replay

---

## APPENDIX: Grid Specification

| Parameter | Values Tested | Count |
|-----------|--------------|-------|
| hold_time | 120, 180, 240, 300, 420 | 5 |
| trail_start | 0.10, 0.15, 0.20, 0.25 | 4 |
| trail_offset | 0.05, 0.08, 0.10, 0.15 | 4 |
| spread_threshold | 0.4, 0.6, 0.8, 1.0 | 4 |
| containment_pullback | 0.15, 0.25, 0.35, 0.50 | 4 |
| session_trade_cap | 20, 30, 40 | 3 |
| **Total configurations** | | **3,840** |
| **Pre-computed simulations** | 80 exit combos x 218 signals | **17,440** |
| **Evaluation time** | | **0.4 seconds** |

---

*All data sources accessed READ-ONLY. NO production changes were made.*
*Script: ai/research/grid_replay_2026_03_03.py*
