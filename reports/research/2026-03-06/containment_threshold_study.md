# 5. Containment Filter Study
## Data: Spread Gate Relaxation Study + last_mile_results.json (2026-03-03 to 2026-03-04)
## Generated: 2026-03-06

---

## Executive Summary

Containment gates blocked **120 signals** across Mar 3-4 that had already passed all
quality filters (ignition, meta). SPREAD is the dominant veto (53% of containment blocks).
Raising the spread threshold from 0.4% to 0.8% would pass **46 additional signals** with
near-breakeven PnL (-0.001% avg) and manageable risk (MAE avg -0.47%).

The containment gate is **too strict for low-price stocks** where penny-wide spreads
represent a structurally higher percentage.

---

## 1. Current Production Thresholds

| Parameter | RTH Value | Premarket Value |
|-----------|-----------|-----------------|
| Spread threshold | 0.4% | 0.6% |
| Recheck hold | Active | Active |
| Pullback spike required | Yes | Yes |
| Price below check | Active | Active |

---

## 2. Containment Event Breakdown (Mar 3-4)

| Reason | Mar 3 | Mar 4 | Total | % |
|--------|-------|-------|-------|---|
| SPREAD | 56 | 7 | 63 | 52.5% |
| RECHECK_HOLD | 22 | 9 | 31 | 25.8% |
| NO_PULLBACK_SPIKE | 11 | 5 | 16 | 13.3% |
| RECHECK_PRICE_BELOW | 8 | 2 | 10 | 8.3% |
| **Total** | **97** | **23** | **120** | **100%** |

---

## 3. Spread Distribution at Time of Veto

From the 64 SPREAD-blocked events with price data:

| Metric | Value |
|--------|-------|
| Min spread | 0.419% |
| P25 | 0.466% |
| **Median** | **0.498%** |
| P75 | 0.744% |
| Max | 5.677% |
| Mean spread (cents) | 0.025c |
| Median spread (cents) | 0.010c |

**Key insight**: Most vetoed signals have spreads of 0.42-0.75% — just above the 0.4%
threshold. These are penny-wide spreads ($0.01) on low-price stocks ($1-$3), which
structurally represent a higher percentage.

---

## 4. Spread Veto by Symbol

| Symbol | Count | Avg Spread% | Avg Spread (cents) | Avg Price | Structural? |
|--------|-------|------------|-------------------|-----------|-------------|
| PLUG | 18 | 0.452% | 0.010c | $2.19 | YES - penny on $2 |
| USEG | 11 | 0.720% | 0.010c | $1.39 | YES - penny on $1.40 |
| SOXS | 10 | 0.483% | 0.010c | $2.07 | YES - penny on $2 |
| TMDE | 7 | 0.634% | 0.024c | $3.84 | Partial |
| STAK | 5 | 1.005% | 0.012c | $1.19 | YES - penny on $1.20 |
| NPT | 4 | 1.769% | 0.146c | $8.21 | NO - genuinely wide |
| TPET | 3 | 0.726% | N/A | N/A | Unknown |
| BATL | 2 | 0.864% | 0.120c | $24.64 | NO - genuinely wide |
| IONZ | 1 | 0.464% | 0.080c | $17.34 | Borderline |
| CRCD | 1 | 0.438% | 0.040c | $9.14 | Borderline |

**39 of 64 spread vetoes** (61%) are on stocks < $3 where the spread is structurally
1 penny — the % threshold penalizes low-price stocks disproportionately.

---

## 5. Threshold Sweep — How Many Pass at Each Level?

| Threshold | Signals Pass | With Price Data | WR (60s) | Avg PnL | Total PnL | Avg MAE |
|-----------|-------------|----------------|----------|---------|-----------|---------|
| **0.4%** (current) | 0 | 0 | N/A | N/A | N/A | N/A |
| **0.6%** | 36 | 34 | 23.5% | -0.091% | -3.099% | -0.447% |
| **0.7%** | 41 | 38 | 26.3% | -0.003% | -0.121% | -0.486% |
| **0.8%** | 52 | 46 | 28.3% | -0.001% | -0.059% | -0.466% |
| **1.0%** | 56 | 50 | 26.0% | -0.001% | -0.059% | -0.466% |

### Incremental Gains Per Step:

```
0.4% -> 0.6%: +36 trades  (PLUG:18, SOXS:10, TMDE:5, IONZ:1, CRCD:1)
0.6% -> 0.7%: + 5 trades  (USEG:4, MOBX:1)
0.7% -> 0.8%: +11 trades  (USEG:7, TPET:3, TMDE:1)
0.8% -> 1.0%: + 4 trades  (STAK:3, STAK:1)
```

**Sweet spot: 0.8%** — captures 52 of 64 signals (81%) with near-zero avg PnL drag.
Going above 0.8% adds only 4 more trades (diminishing returns).

---

## 6. Spread Bucket vs Outcome

| Spread Bucket | N | Avg PnL | WR(60s) | Avg MFE | Avg MAE |
|---------------|---|---------|---------|---------|---------|
| 0.4-0.5% | 30 | +0.030% | 26.7% | 0.754% | -0.393% |
| 0.5-0.6% | 4 | -1.000% | 0.0% | 0.201% | -0.847% |
| 0.6-0.7% | 4 | +0.744% | 50.0% | 1.651% | -0.817% |
| 0.7-0.8% | 8 | +0.008% | 37.5% | 0.595% | -0.371% |
| 0.8-0.9% | 4 | N/A | 0.0% | N/A | N/A |
| 0.9%+ | 6 | +0.115% | 16.7% | 0.915% | -0.340% |

The 0.4-0.5% bucket (just above threshold) is actually **net positive** (+0.030% avg PnL).
These are being incorrectly blocked.

---

## 7. Micro-Wait Recheck Results

49 of 64 events were candidates for spread recheck (within 0.15% of threshold).
**None improved** to passing within 1 second — spreads are structural, not momentary.

This confirms: a threshold change is needed, not a timing optimization.

---

## 8. Price-Adjusted Threshold Recommendation

Rather than a flat % threshold, consider a **cents-based floor**:

| Policy | Equivalent | Effect |
|--------|-----------|--------|
| Current: 0.4% flat | $0.01 on $2.50 stock | Blocks all sub-$2.50 penny-spread stocks |
| Proposed: max(0.4%, $0.02 spread) | Allows penny spreads on stocks > $2 | Unblocks PLUG, SOXS, most USEG |
| Alternative: 0.8% flat | Simple, captures 81% of blocks | Simpler to implement |

---

## 9. Recommendations

1. **P0**: Raise RTH spread threshold from 0.4% to **0.8%** — unblocks 52 signals with near-zero PnL drag
2. **P1**: Consider dual threshold: 0.4% OR $0.02 absolute spread — handles low-price stocks structurally
3. **P2**: Raise premarket threshold from 0.6% to **0.8%** (marginal gain: +1 trade)
4. **P3**: RECHECK_HOLD (31 events) needs separate study — these passed quality gates then were held

---

*Data sources: SPREAD_GATE_RELAXATION_STUDY_2026-03-03_to_2026-03-04.md,
last_mile_results.json, containment events (READ-ONLY)*
*NO production changes were made. This is research-only analysis.*
