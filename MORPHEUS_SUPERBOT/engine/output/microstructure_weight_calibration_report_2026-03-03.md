# Microstructure Weight Calibration Report - 2026-03-03

**Generated:** 2026-03-04T04:34:53Z
**Mode:** READ-ONLY RESEARCH (SuperBot Engine)
**Method:** RandomForest + GradientBoosting ensemble feature importance

---

## Top Features Driving Momentum Edge

| Rank | Feature | Importance | Direction | Weight | Correlation |
|------|---------|------------|-----------|--------|-------------|
| 1 | volatility_expansion | 0.1911 | negative | -3.0000 | -0.200 |
| 2 | local_trend_persistence | 0.1192 | positive | +1.8713 | +0.235 |
| 3 | local_volatility | 0.1146 | negative | -1.7991 | -0.530 |
| 4 | local_spread_std | 0.1021 | positive | +1.6028 | +0.110 |
| 5 | entry_lateness | 0.0911 | negative | -1.4301 | -0.405 |
| 6 | price_velocity_30s | 0.0742 | positive | +1.1648 | +0.495 |
| 7 | spread_pct | 0.0522 | positive | +0.8195 | +0.107 |
| 8 | local_gap_p90_ms | 0.0355 | negative | -0.5573 | -0.145 |
| 9 | local_stale_count | 0.0278 | positive | +0.4364 | +0.282 |
| 10 | momentum_score | 0.0221 | positive | +0.3469 | +0.117 |
| 11 | sym_spread_stability | 0.0215 | negative | -0.3375 | -0.053 |
| 12 | local_tick_rate | 0.0189 | positive | +0.2967 | +0.095 |
| 13 | confidence | 0.0188 | negative | -0.2951 | -0.123 |
| 14 | sym_volatility_5s | 0.0167 | positive | +0.2622 | +0.050 |
| 15 | sym_profit_margin_ratio | 0.0165 | positive | +0.2590 | +0.055 |

## Ranking Comparison: Original vs Learned

| Symbol | Actual Return | Orig Rank | Orig Score | Learned Rank | Learned Score | Improved? |
|--------|--------------|-----------|------------|-------------|--------------|-----------|
| PLUG | +0.3566% | 9 | -4.523 | 4 | +1.693 | YES |
| BATL | +0.3197% | 1 | +16.244 | 1 | +11.092 | NO |
| DUST | +0.1805% | 5 | -0.733 | 8 | -1.787 | NO |
| MSTZ | +0.1350% | 2 | +6.071 | 7 | -1.472 | NO |
| VG | +0.0025% | 6 | -3.287 | 9 | -2.657 | NO |
| UVIX | -0.3663% | 4 | +3.321 | 2 | +3.427 | NO |
| SOXS | -0.3722% | 7 | -3.949 | 6 | +0.361 | NO |
| TMDE | -0.5497% | 3 | +4.355 | 11 | -10.939 | YES |
| IONZ | -0.5646% | 8 | -4.420 | 5 | +0.586 | NO |
| USEG | -1.3202% | 11 | -6.929 | 3 | +2.721 | NO |
| CRCD | -1.5367% | 10 | -6.149 | 10 | -3.025 | NO |

## Correlation Metrics

| Metric | Original | Learned | Improvement |
|--------|----------|---------|-------------|
| Sign accuracy | 6/11 (55%) | 4/11 (36%) | -18pp |
| Score-outcome r | +0.5263 | +0.2641 | -0.2622 |

## Top-Half vs Bottom-Half Performance

| Model | Set | N | Win Rate | Avg Return | PF | 1m Avg |
|-------|-----|---|----------|------------|-----|--------|
| Original | Top | 31 | 45.2% | -0.1783% | 0.705 | 0.0623 |
| Original | Bottom | 43 | 46.5% | -0.2546% | 0.491 | 0.0034 |
| **Learned** | **Top** | **34** | **50.0%** | **-0.0556%** | **0.86** | **0.11** |
| **Learned** | **Bottom** | **40** | **42.5%** | **-0.3647%** | **0.455** | **-0.0415** |

**Original top-bottom spread:** +0.0763%
**Learned top-bottom spread:** +0.3091%
**Improvement:** +0.2328% wider separation

## Updated Preferred Symbols (Learned Model)

- **BATL**: +11.092 [PREFERRED]
- **UVIX**: +3.427 [PREFERRED]
- **USEG**: +2.721 [PREFERRED]
- **PLUG**: +1.693 [PREFERRED]
- **IONZ**: +0.586 [PREFERRED]
- **SOXS**: +0.361 [PREFERRED]
- **MSTZ**: -1.472 [AVOID]
- **DUST**: -1.787 [AVOID]
- **VG**: -2.657 [AVOID]
- **CRCD**: -3.025 [AVOID]
- **TMDE**: -10.939 [AVOID]

---

*This calibration is research-only. No production changes applied.*