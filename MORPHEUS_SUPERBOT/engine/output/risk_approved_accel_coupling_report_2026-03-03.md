# RISK_APPROVED + Ignition Accelerator Coupling Report - 2026-03-03

**Generated:** 2026-03-04T05:40:57Z
**Mode:** READ-ONLY RESEARCH (SuperBot Engine)
**Method:** Couple v1 ignition detector with 74 v2-pass RISK_APPROVED signals

---

## Baseline Reference (74 v2-pass signals, current entry)

| Metric | Value |
|--------|-------|
| Signals | 74 |
| Win Rate | 45.9% |
| Avg Return | -0.2227% |
| Profit Factor | 0.591 |

---

## Sensitivity Sweep: Lookback Window x Offset

| Lookback | Offset | Ign Found | Filled | Fill% | Accel WR | Accel Avg | Accel PF | Same-Sig BL WR | Same-Sig BL Avg | Same-Sig BL PF | WR Delta | Avg Delta | PF Delta |
|----------|--------|-----------|--------|-------|----------|-----------|----------|----------------|-----------------|----------------|----------|-----------|----------|
| 60s | +0.00% | 5/74 | 4 | 5.4% | 50.0% | +0.1183% | 1.445 | 25.0% | -0.5388% | 0.167 | +25.0pp | +0.6571% | +1.278 |
| 60s | -0.20% | 5/74 | 4 | 5.4% | 50.0% | +0.1763% | 1.663 | 25.0% | -0.5388% | 0.167 | +25.0pp | +0.7151% | +1.496 |
| 60s | -0.50% | 5/74 | 1 | 1.4% | 100.0% | +1.1432% | 999.000 | 0.0% | -1.0444% | 0.000 | +100.0pp | +2.1876% | +999.000 |
| 180s | +0.00% | 13/74 | 9 | 12.2% | 44.4% | +0.0478% | 1.119 | 22.2% | -0.5638% | 0.149 | +22.2pp | +0.6116% | +0.970 |
| 180s | -0.20% | 13/74 | 8 | 10.8% | 50.0% | +0.1325% | 1.306 | 25.0% | -0.6172% | 0.152 | +25.0pp | +0.7497% | +1.154 |
| 180s | -0.50% | 13/74 | 1 | 1.4% | 100.0% | +1.1432% | 999.000 | 0.0% | -1.0444% | 0.000 | +100.0pp | +2.1876% | +999.000 |
| 360s | +0.00% | 20/74 | 12 | 16.2% | 58.3% | +0.2925% | 1.969 | 33.3% | -0.3132% | 0.369 | +25.0pp | +0.6057% | +1.600 |
| 360s | -0.20% | 20/74 | 11 | 14.9% | 63.6% | +0.3975% | 2.264 | 36.4% | -0.3294% | 0.378 | +27.2pp | +0.7269% | +1.886 |
| 360s | -0.50% | 20/74 | 1 | 1.4% | 100.0% | +1.1432% | 999.000 | 0.0% | -1.0444% | 0.000 | +100.0pp | +2.1876% | +999.000 |

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Lookback window | 360s |
| Offset | +0.00% |
| Fill rate | 16.2% (12/74) |
| Accel WR | 58.3% (vs 33.3% baseline) |
| Accel Avg Return | +0.2925% (vs -0.3132%) |
| Accel PF | 1.969 (vs 0.369) |
| PF Delta | +1.600 |
| Median entry lateness BEFORE | 102.1s |
| Median entry lateness AFTER | 10.0s |
| Median lateness improvement | 93.8s |
| Median price improvement | +0.1598% |

## Per-Symbol Breakdown (lookback=360s, offset=+0.00%)

| Symbol | Signals | Filled | Fill% | BL WR | BL Avg | BL PF | Accel WR | Accel Avg | Accel PF |
|--------|---------|--------|-------|-------|--------|-------|----------|-----------|----------|
| PLUG | 17 | 3 | 17.6% | 70.6% | +0.3566% | 5.290 | 66.7% | +0.9256% | 7.346 |
| SOXS | 13 | 2 | 15.4% | 38.5% | -0.3722% | 0.277 | 50.0% | +0.4963% | 999.000 |
| TMDE | 10 | 0 | 0.0% | 30.0% | -0.5497% | 0.423 | 0.0% | +0.0000% | 0.000 |
| UVIX | 7 | 3 | 42.9% | 28.6% | -0.3663% | 0.324 | 0.0% | -1.0622% | 0.000 |
| DUST | 6 | 0 | 0.0% | 66.7% | +0.1805% | 1.492 | 0.0% | +0.0000% | 0.000 |
| IONZ | 6 | 3 | 50.0% | 33.3% | -0.5646% | 0.265 | 100.0% | +0.6955% | 999.000 |
| MSTZ | 6 | 0 | 0.0% | 66.7% | +0.1350% | 1.369 | 0.0% | +0.0000% | 0.000 |
| CRCD | 4 | 0 | 0.0% | 0.0% | -1.5367% | 0.000 | 0.0% | +0.0000% | 0.000 |
| BATL | 2 | 1 | 50.0% | 50.0% | +0.3197% | 1.612 | 100.0% | +0.8408% | 999.000 |
| USEG | 2 | 0 | 0.0% | 0.0% | -1.3202% | 0.000 | 0.0% | +0.0000% | 0.000 |
| VG | 1 | 0 | 0.0% | 100.0% | +0.0025% | 999.000 | 0.0% | +0.0000% | 0.000 |

## Per-Strategy Breakdown (lookback=360s, offset=+0.00%)

| Strategy | Signals | Filled | BL Avg | Accel Avg | Delta |
|----------|---------|--------|--------|-----------|-------|
| catalyst_momentum | 72 | 12 | -0.1820% | +0.2925% | +0.4745% |
| premarket_breakout | 2 | 0 | -1.6872% | +0.0000% | +0.0000% |

## Failure Mode Analysis

### Configuration: lookback=360s, offset=+0.00%

| Failure Mode | Count | % of 74 | Description |
|-------------|-------|---------|-------------|
| No preceding ignition | 54 | 73.0% | No ignition event within 360s before signal |
| Ignition but no fill | 8 | 10.8% | Price didn't touch limit within 30s after ignition |
| Filled but stopped out | 3 | 4.1% | Entry filled but hit -1.0% stop |
| Successfully filled | 12 | 16.2% | Accelerated entry filled and completed exit |

**No-ignition signals by symbol:**
- PLUG: 14/17 missed (82%)
- SOXS: 11/13 missed (85%)
- TMDE: 10/10 missed (100%)
- DUST: 6/6 missed (100%)
- MSTZ: 4/6 missed (67%)
- UVIX: 3/7 missed (43%)
- IONZ: 3/6 missed (50%)
- USEG: 2/2 missed (100%)
- VG: 1/1 missed (100%)

## Key Findings

1. **Does coupling improve PF vs baseline?** YES - PF delta = +1.600 (baseline 0.369 vs accel 1.969)
2. **Recommended conservative defaults:**
   - `ignition_accelerator_enabled`: true
   - `ignition_accelerator_offset_pct`: 0.0
   - `ignition_lookback_seconds`: 360
   - `fill_window_seconds`: 30
   - Apply ONLY when spread < 0.9% (PM) / 0.6% (RTH)
   - Safety: accelerator only modifies entry_reference when signal already passed Morpheus scoring/gates
3. **Failure modes:**
   - Miss rate: 83.8% (62/74 signals)
   - Primary failure: no ignition detected within window (54/74)
   - Secondary failure: ignition found but limit not filled (8/74)

---

*This study is research-only. No production changes applied.*