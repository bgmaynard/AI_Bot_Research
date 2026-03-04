# Momentum Edge Diagnostics - 2026-03-03

**Generated:** 2026-03-04T04:11:06Z
**Mode:** READ-ONLY RESEARCH (SuperBot Engine)
**Basis:** 74 v2-pass signals from 2026-03-03
**Exit model:** stop=-1.0%, trail=+0.8%/0.4%, fallback=5m

---

## Study 1: Entry Timing Offset

Tests whether entering earlier (lower price) or later (higher price) improves results.
Negative offset = earlier/better entry. Positive offset = later/worse entry.

| Offset | N | Win Rate | Avg Return | Expectancy | PF | Avg MFE | Avg MAE | Avg Win | Avg Loss |
|--------|---|----------|------------|------------|-----|---------|---------|---------|----------|
| -0.50% **<-- best** | 74 | 74.3% | +0.3363% | +0.3363 | 2.304 | 1.714% | -0.685% | +0.800% | -1.005% |
| -0.40% | 74 | 70.3% | +0.2238% | +0.2238 | 1.739 | 1.612% | -0.785% | +0.749% | -1.018% |
| -0.30% | 74 | 66.2% | +0.1211% | +0.1211 | 1.330 | 1.510% | -0.884% | +0.737% | -1.087% |
| -0.20% | 74 | 62.2% | +0.0384% | +0.0384 | 1.087 | 1.408% | -0.984% | +0.771% | -1.166% |
| -0.10% | 74 | 54.1% | -0.1571% | -0.1571 | 0.686 | 1.306% | -1.083% | +0.635% | -1.089% |
| +0.00% **<-- current** | 74 | 45.9% | -0.2227% | -0.2227 | 0.591 | 1.205% | -1.182% | +0.699% | -1.006% |
| +0.10% | 74 | 40.5% | -0.2886% | -0.2886 | 0.500 | 1.104% | -1.280% | +0.711% | -0.970% |

**Best offset:** -0.50%
**Best expectancy:** +0.3363
**Best avg return:** +0.3363%
**Current avg return:** -0.2227%
**Improvement vs current:** +0.5590%

**Finding:** Earlier entry by 0.50% improves returns by +0.5590%. Signals are entering LATE.

---

## Study 2: Hold Time Sweep

Tests fixed-time exits (stop + time only, no trailing stop) at various hold periods.

| Hold Time | N | Win Rate | Avg Return | Expectancy | PF | Avg MFE | Avg MAE | Trail WR | Trail Avg |
|-----------|---|----------|------------|------------|-----|---------|---------|----------|-----------|
| 30s | 74 | 33.8% | -0.2076% | -0.2076 | 0.323 | 0.277% | -0.399% | 33.8% | -0.2038% |
| 45s | 74 | 37.8% | -0.1334% | -0.1334 | 0.582 | 0.371% | -0.497% | 37.8% | -0.1415% |
| 60s | 74 | 45.9% | -0.0509% | -0.0509 | 0.832 | 0.541% | -0.578% | 45.9% | -0.0746% |
| 90s | 74 | 41.9% | -0.0952% | -0.0952 | 0.758 | 0.683% | -0.696% | 41.9% | -0.1698% |
| 120s | 74 | 37.8% | -0.1814% | -0.1814 | 0.612 | 0.732% | -0.782% | 37.8% | -0.2478% |
| 180s | 74 | 41.9% | -0.0356% | -0.0356 | 0.929 | 0.949% | -0.925% | 44.6% | -0.2192% |
| 300s **<-- optimal** | 74 | 40.5% | -0.0276% | -0.0276 | 0.953 | 1.205% | -1.182% | 45.9% | -0.2227% |

**Optimal hold time:** 300s
**Optimal expectancy:** -0.0276
**Optimal avg return:** -0.0276%

### Return Decay Curve

```
    30s -0.2076%  ########################################|
    45s -0.1334%                 #########################|
    60s -0.0509%                                 #########|
    90s -0.0952%                        ##################|
   120s -0.1814%        ##################################|
   180s -0.0356%                                    ######|
   300s -0.0276%                                     #####|
```

---

## Study 3: Symbol Expectancy Ranking

| Rank | Symbol | Signals | Win Rate | Avg Return | Total Return | PF | Avg MFE | Avg MAE | Classification |
|------|--------|---------|----------|------------|-------------|-----|---------|---------|----------------|
| 1 | PLUG | 17 | 70.6% | +0.3566% | +6.062% | 5.290 | 1.207% | -0.493% | STRONG_EDGE |
| 2 | BATL | 2 | 50.0% | +0.3197% | +0.639% | 1.612 | 3.177% | -1.650% | STRONG_EDGE |
| 3 | DUST | 6 | 66.7% | +0.1805% | +1.083% | 1.492 | 0.931% | -0.549% | STRONG_EDGE |
| 4 | MSTZ | 6 | 66.7% | +0.1350% | +0.810% | 1.369 | 0.689% | -0.751% | STRONG_EDGE |
| 5 | VG | 1 | 100.0% | +0.0025% | +0.003% | inf | 0.412% | -0.571% | NEUTRAL |
| 6 | UVIX | 7 | 28.6% | -0.3663% | -2.564% | 0.324 | 0.592% | -1.215% | NEGATIVE_EDGE |
| 7 | SOXS | 13 | 38.5% | -0.3722% | -4.838% | 0.277 | 0.386% | -0.920% | NEGATIVE_EDGE |
| 8 | TMDE | 10 | 30.0% | -0.5497% | -5.497% | 0.423 | 3.756% | -2.156% | NEGATIVE_EDGE |
| 9 | IONZ | 6 | 33.3% | -0.5646% | -3.387% | 0.265 | 0.874% | -1.704% | NEGATIVE_EDGE |
| 10 | USEG | 2 | 0.0% | -1.3202% | -2.640% | 0.000 | -0.058% | -1.833% | NEGATIVE_EDGE |
| 11 | CRCD | 4 | 0.0% | -1.5367% | -6.147% | 0.000 | 0.089% | -2.870% | NEGATIVE_EDGE |

### Edge Concentration

| Category | Symbols | Total Return |
|----------|---------|-------------|
| STRONG_EDGE | 4 | +8.594% |
| NEUTRAL | 1 | - |
| NEGATIVE_EDGE | 6 | -25.074% |
| **ALL** | **11** | **-16.477%** |

**Allow-list candidates:** PLUG, BATL, DUST, MSTZ
**Blacklist candidates:** UVIX, SOXS, TMDE, IONZ, USEG, CRCD

### Symbol Edge Map

```
   PLUG  +0.3566%  |+++++++
   BATL  +0.3197%  |++++++
   DUST  +0.1805%  |+++
   MSTZ  +0.1350%  |++
     VG  +0.0025%  |
   UVIX  -0.3663%  -------|
   SOXS  -0.3722%  -------|
   TMDE  -0.5497%  ----------|
   IONZ  -0.5646%  -----------|
   USEG  -1.3202%  --------------------------|
   CRCD  -1.5367%  ------------------------------|
```

---

## Final Diagnosis

**Performance issues ranked by severity:**

1. **BAD SYMBOL SELECTION** (impact: 25.074%) - 6 symbols with negative edge, costing -25.074%
2. **LATE ENTRY** (impact: 0.559%) - Best offset is -0.50%, improvement of +0.5590%

**Primary cause:** BAD SYMBOL SELECTION

### Recommendations

1. **Entry timing:** Consider limit-order entries 0.50% below signal price
3. **Symbol filter:** Consider blacklisting: UVIX, SOXS, TMDE, IONZ, USEG, CRCD
4. **Symbol focus:** Prioritize: PLUG, BATL, DUST, MSTZ

---

*This study is research-only. No production changes applied.*
*All recommendations require review and approval before implementation.*