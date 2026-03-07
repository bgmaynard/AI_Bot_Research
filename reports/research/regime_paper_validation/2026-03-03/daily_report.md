# Regime Filter Validation — Daily Report
## Date: 2026-03-03
## Generated: 2026-03-06

---

## FILTER CONFIGURATION

```
volatility_1m >= 0.3% | spread <= 0.6% | OFI >= -0.2
suppress: LOW_VOLATILITY | power_hour
simulation: trail=1.0%%, max_hold=300s
```

---

## SIDE-BY-SIDE (Uncapped)

| Metric | Baseline | Filtered | Blocked | Filter Edge |
|--------|----------|----------|---------|-------------|
| Trades | 2321 | 322 | 1999 | - |
| Win Rate | 33.7% | 42.5% | 32.3% | +8.8 pp |
| Profit Factor | 0.73 | 1.3 | 0.66 | +0.57 |
| Total PnL | $-493,065 | $+53,772 | $-546,836 | $+546,836 |
| Avg PnL | $-212 | $+167 | $-274 | $+379 |
| Max DD | $+808,814 | $+69,625 | - | $+739,190 |

---

## BLOCK REASON DISTRIBUTION

| Reason | Count | %% of Blocked |
|--------|-------|--------------:|
| SUPPRESS_REGIME | 1485 | 36.0% |
| LOW_VOL | 1316 | 31.9% |
| HIGH_SPREAD | 814 | 19.7% |
| WEAK_OFI | 389 | 9.4% |
| SUPPRESS_SESSION | 119 | 2.9% |

---

## PER-SYMBOL

| Symbol | Base n | Base PF | Filt n | Filt PF | Blocked n | Blocked PF | Edge |
|--------|--------|---------|--------|---------|-----------|------------|------|
| BATL | 1744 | 0.73 | 264 | 1.28 | 1480 | 0.67 | YES |
| CRCD | 21 | 0.66 | 1 | INF | 20 | 0.64 | NO |
| DUST | 3 | 0.5 | 0 | 0 | 3 | 0.5 | N/A |
| IONZ | 238 | 0.69 | 14 | 1.86 | 224 | 0.52 | YES |
| MSTZ | 48 | 1.25 | 0 | 0 | 48 | 1.25 | N/A |
| PLUG | 3 | 0.43 | 1 | 0.0 | 2 | 0.6 | NO |
| SOXS | 1 | INF | 0 | 0 | 1 | INF | N/A |
| TMDE | 156 | 0.62 | 39 | 0.54 | 117 | 0.66 | NO |
| USEG | 8 | 0.33 | 0 | 0 | 8 | 0.33 | N/A |
| UVIX | 74 | 0.72 | 2 | 0.0 | 72 | 0.72 | NO |
| VG | 25 | 0.33 | 1 | INF | 24 | 0.25 | NO |

---

## PER-SESSION

| Session | Base n | Base PF | Filt n | Filt PF | Edge |
|---------|--------|---------|--------|---------|------|
| premarket | 1170 | 0.81 | 160 | 1.28 | YES |
| open | 203 | 0.56 | 36 | 1.94 | YES |
| midday | 829 | 0.73 | 126 | 1.15 | YES |
| power_hour | 119 | 0.11 | 0 | 0 | SUPPRESSED |

---
*All data READ-ONLY. NO production changes.*