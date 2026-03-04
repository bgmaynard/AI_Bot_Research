# Microstructure Validation - 2026-03-03

**Generated:** 2026-03-04T04:25:08Z
**Mode:** READ-ONLY RESEARCH (SuperBot Engine)
**Method:** Forward-window replay, preferred vs avoid sets

---

## Preferred vs Avoid: Exit Model

| Metric | Preferred | Avoid | Delta |
|--------|-----------|-------|-------|
| win_rate | 47.2% | 45.8% | +1.4000% |
| avg_return | -0.1922% | -0.2377% | +0.0455% |
| total_return | -13.8366% | -17.1166% | +3.2800% |
| signals | 72 | 72 | - |

## Preferred vs Avoid: Forward Windows

| Window | Pref WR | Pref Avg | Avoid WR | Avoid Avg | WR Delta | Return Delta |
|--------|---------|----------|----------|-----------|----------|--------------|
| 1m | 48.6% | +0.0545% | 47.2% | +0.0172% | +1.4pp | +0.0373% |
| 5m | 45.8% | -0.0902% | 44.4% | -0.1471% | +1.4pp | +0.0569% |
| 10m | 38.9% | -0.5931% | 37.5% | -0.6903% | +1.4pp | +0.0972% |

## Validation Verdict

**VALIDATED**: Preferred set outperforms avoid set at 1m window (+0.0545% vs +0.0172%)
**EXIT MODEL VALIDATED**: Preferred -0.1922% vs Avoid -0.2377% (delta: +0.0455%)

## Score vs Outcome Correlation

| Symbol | Micro Score | Exit Avg Return | Correlated? |
|--------|------------|-----------------|-------------|
| BATL | +16.244 | +0.3197% | YES |
| MSTZ | +6.071 | +0.1350% | YES |
| TMDE | +4.355 | -0.5497% | NO |
| UVIX | +3.321 | -0.3663% | NO |
| DUST | -0.733 | +0.1805% | NO |
| VG | -3.287 | +0.0025% | NO |
| SOXS | -3.949 | -0.3722% | YES |
| IONZ | -4.420 | -0.5646% | YES |
| PLUG | -4.523 | +0.3566% | NO |
| CRCD | -6.149 | -1.5367% | YES |
| USEG | -6.929 | -1.3202% | YES |

**Correlation accuracy:** 6/11 (55%)

---

*This validation is research-only. No production changes applied.*