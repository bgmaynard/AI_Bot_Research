# 4. Ignition Funnel Autopsy
## Data: last_mile_results.json (2026-03-02 to 2026-03-04)
## Generated: 2026-03-06

---

## Executive Summary

**Zero trades executed across 3 consecutive days** despite 241 risk-approved signals.
The pipeline processes ~30,000 signals/day but applies cascading filters that reduce
the stream to <200 at ignition, then <30 at risk approval. The final mile (execution)
blocks the remaining 100% via data staleness and containment rechecks.

---

## 1. Full Pipeline Reconstruction

### 2026-03-02 (Dead Feed Day)

```
signals_scored:     31,517  (100.0%)
  |- EXTENSION_GATE:  -8,933  (28.3%)  hard_veto: no catalyst
extension_passed:   22,584  (71.7%)
  |- IGNITION_GATE:  -22,470  (71.3%)  4 sub-reasons
ignition_passed:       114  ( 0.4%)
  |- META_GATE:          -30  ( 0.1%)
meta_approved:         279  ( 0.9%)  [funnel mismatch in EOD data]
  |- RISK:             -169  ( 0.5%)  INSUFFICIENT_ROOM
risk_approved:         110  ( 0.3%)
  |- EXECUTION:        -110  ( 0.3%)  DATA_STALE: 100%
executed:                0  ( 0.0%)
```

### 2026-03-03 (Containment Squeeze Day)

```
signals_scored:     29,742  (100.0%)
  |- EXTENSION_GATE:  -11,340  (38.1%)  hard_veto: no catalyst
extension_passed:   18,402  (61.9%)
  |- IGNITION_GATE:  -18,237  (61.3%)  4 sub-reasons
ignition_passed:       165  ( 0.6%)
  |- META_GATE:          -54  ( 0.2%)
  |- CONTAINMENT:        -97  ( 0.3%)
  |- RISK:                -7  ( 0.0%)
risk_approved:         104  ( 0.3%)
  |- EXECUTION:           -7  ( 0.0%)
executed:                0  ( 0.0%)
```

### 2026-03-04 (Momentum Blackout Day)

```
signals_scored:     31,898  (100.0%)
  |- EXTENSION_GATE:   -4,075  (12.8%)
extension_passed:   27,823  (87.2%)
  |- IGNITION_GATE:  -27,643  (86.7%)  NO_MOMENTUM_DATA dominant
ignition_passed:       180  ( 0.6%)
  |- META_GATE:         -104  ( 0.3%)
  |- CONTAINMENT:        -23  ( 0.1%)
  |- RISK:               -92  ( 0.3%)
risk_approved:          27  ( 0.1%)
  |- EXECUTION:           -4  ( 0.0%)
executed:                0  ( 0.0%)
```

---

## 2. Stage-by-Stage Pass Rates

| Stage | Mar 2 | Mar 3 | Mar 4 | 3-Day Avg |
|-------|-------|-------|-------|-----------|
| Extension Pass Rate | 71.7% | 61.9% | 87.2% | 73.6% |
| Ignition Pass Rate (of ext_pass) | 0.50% | 0.90% | 0.65% | 0.68% |
| Meta Pass Rate (of ign_pass) | 73.7% | 67.3% | 42.2% | 61.1% |
| Risk Pass Rate (of meta_pass) | 39.4% | 93.3% | 29.7% | 54.1% |
| Execution Pass Rate | 0% | 0% | 0% | **0%** |

**Critical bottleneck: Ignition Gate** passes only 0.5-0.9% of extension-passed signals.
This is by far the most aggressive filter in the pipeline.

---

## 3. Ignition Gate Veto Breakdown

| Reason | Mar 2 | Mar 3 | Mar 4 | Total | % |
|--------|-------|-------|-------|-------|---|
| LOW_CONFIDENCE | 11,027 | 7,151 | 2,546 | 20,724 | 30.3% |
| NO_MOMENTUM_DATA | 4,456 | 379 | 19,769 | 24,604 | 36.0% |
| HIGH_SPREAD | 3,931 | 5,143 | 2,765 | 11,839 | 17.3% |
| LOW_SCORE | 3,056 | 5,513 | 2,558 | 11,127 | 16.3% |
| **Total** | **22,470** | **18,237** | **27,643** | **68,350** | **100%** |

### Dominant Veto by Day:
- **Mar 2**: LOW_CONFIDENCE (49%) — scorer producing low values
- **Mar 3**: LOW_CONFIDENCE (39%) + LOW_SCORE (30%) — quality problem
- **Mar 4**: NO_MOMENTUM_DATA (72%) — **infrastructure failure**, not signal quality

---

## 4. Containment Gate Analysis (New gate, active Mar 3+)

| Reason | Mar 3 | Mar 4 | Total |
|--------|-------|-------|-------|
| SPREAD | 56 | 7 | 63 |
| RECHECK_HOLD | 22 | 9 | 31 |
| NO_PULLBACK_SPIKE | 11 | 5 | 16 |
| RECHECK_PRICE_BELOW | 8 | 2 | 10 |
| **Total** | **97** | **23** | **120** |

SPREAD is the dominant containment veto. These are signals that passed ignition
(quality filters) but were blocked by microstructure checks.

---

## 5. Execution Block Breakdown

| Reason | Mar 2 | Mar 3 | Mar 4 | Total | % |
|--------|-------|-------|-------|-------|---|
| DATA_STALE | 110 | 0 | 1 | 111 | 91.7% |
| QUOTE_STALE | 0 | 3 | 3 | 6 | 5.0% |
| CONTAINMENT_RECHECK_HOLD | 0 | 4 | 0 | 4 | 3.3% |
| **Total** | **110** | **7** | **4** | **121** | **100%** |

Mar 2 is anomalous (server crash, all feeds dead). Mar 3-4 show 11 execution
blocks total — these are the "last inch" failures.

---

## 6. Risk Gate Analysis

| Reason | Mar 2 | Mar 4 | Notes |
|--------|-------|-------|-------|
| INSUFFICIENT_ROOM | 169 | 89 | Signal entry too close to target |
| Other | 0 | 3 | Misc risk rules |

169 signals on Mar 2 passed quality but failed profit-room check — the entry price
was too close to the profit target to justify the stop risk. This suggests either:
1. Profit targets are too tight, or
2. Scanner is discovering stocks after the initial move

---

## 7. Signal Density by Symbol (Top 10)

| Symbol | Mar 2 | Mar 3 | Mar 4 | Total Blocks | Notes |
|--------|-------|-------|-------|-------------|-------|
| TPET | - | 4,639 | - | 4,639 | Mar 3 dominant |
| BATL | 2,977 | 4,415 | - | 7,392 | Consistent high signal |
| NPT | - | 2,955 | - | 2,955 | Mar 3 only |
| TMDE | 2,531 | 2,666 | - | 5,197 | Both days |
| TURB | 2,515 | - | - | 2,515 | Mar 2 only |
| AGIG | 2,044 | - | - | 2,044 | Mar 2 only |
| USEG | 1,920 | 1,615 | - | 3,535 | Both days |
| STAK | - | 1,481 | - | 1,481 | Mar 3 only |
| RUBI | - | 1,359 | - | 1,359 | Mar 3 only |
| UVIX | - | 1,297 | - | 1,297 | Mar 3 only |

---

## 8. Key Findings

1. **Ignition gate is 99.3% kill rate** — intentionally aggressive but may be over-filtering
2. **NO_MOMENTUM_DATA** blocked 72% of ignition on Mar 4 — this is an infra bug, not a quality filter
3. **Data staleness** killed 100% of execution on Mar 2 and remains a recurring issue
4. **Containment** is a new gate (Mar 3+) that blocks 97-120 signals that passed quality gates
5. **Zero execution across 3 days** despite 241 risk-approved signals — the system needs to trade
6. **Pipeline is tightening**: Risk approved declining 110 -> 104 -> 27

---

*Data sources: last_mile_results.json, downstream_blocks_summary.json (READ-ONLY)*
*NO production changes were made. This is research-only analysis.*
