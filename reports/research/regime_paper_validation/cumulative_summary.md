# Regime Filter Cumulative Summary
## Validation Period: 2026-03-02 to 2026-03-04
## Generated: 2026-03-06

---

## EXECUTIVE SUMMARY

The combined regime filter (vol >= medium, spread <= 0.6%%, OFI >= moderate,
suppress LOW_VOLATILITY, suppress POWER_HOUR) was validated against 2321
simulated trades across 11 symbols on 2026-03-03.

### Key Results (Uncapped, All Signals)

| Metric | Baseline | Filtered | Improvement |
|--------|----------|----------|-------------|
| Trades | 2321 | 322 | -1999 (86% reduction) |
| Win Rate | 33.7% | 42.5% | +8.8 pp |
| Profit Factor | 0.73 | 1.3 | +0.57 |
| Total PnL | $-493065 | $+53772 | $+546836 |

### Key Results (Capped at 20, Production-Like)

| Metric | Baseline | Filtered |
|--------|----------|----------|
| Trades | 20 | 20 |
| Win Rate | 20.0% | 85.0% |
| Profit Factor | 0.04 | 9.65 |
| Total PnL | $-8850 | $+55175 |

### Actual Paper Trades (20 BATL, Production Engine)

| Metric | Would PASS | Would BLOCK |
|--------|-----------|-------------|
| Trades | 3 | 17 |
| Win Rate | 66.7% | 52.9% |
| Profit Factor | 7.54 | 3.35 |
| Total PnL | $+7923 | $+13776 |

---

## FILTER COMPONENT EFFECTIVENESS

| Component | Blocked Signals | %% of All Blocks | Standalone PF Lift |
|-----------|----------------|-----------------|-------------------|
| LOW_VOL | 1316 | 56.7% | +0.23 |
| HIGH_SPREAD | 814 | 35.1% | +0.02 |
| WEAK_OFI | 389 | 16.8% | +0.09 |
| SUPPRESS_REGIME | 1485 | 64.0% | +0.28 |
| SUPPRESS_SESSION | 119 | 5.1% | +0.03 |

---

## DEPLOYMENT READINESS

| Criterion | Status | Required |
|-----------|--------|----------|
| Days of full data | 1 | 5-10 |
| Filter PF > Baseline PF | YES | Consistent across days |
| Filter WR > Baseline WR | YES | Consistent across days |
| Missed high-MFE trades < 10% | NO | Acceptable miss rate |
| Multi-symbol validation | YES (11 symbols) | At least 3 symbols |
| Cross-day consistency | UNKNOWN | Need more days |

**VERDICT: PROMISING but NOT READY for production.**
Continue collecting daily data and re-running validation.

---

*All data sources accessed READ-ONLY. NO production changes were made.*
*Script: ai/research/regime_paper_validation.py*