# Phase 10B — Structural Liquidity Zone Conditioning

**Date:** 2026-02-21
**Total events loaded:** 1944
**Active events (PnL ≠ 0):** 1148
**Zone-labeled (have PDH/PDL):** 216
**Pressure-profiled:** 73

## Power Check

Validation gate requires n ≥ 500 per condition.
- ATR 0.1: TOP_ZONE → n=1 ❌ INSUFFICIENT (n=1)
- ATR 0.1: BOTTOM_ZONE → n=1 ❌ INSUFFICIENT (n=1)
- ATR 0.1: MID → n=214 ⚠️ UNDERPOWERED (n=214)
- ATR 0.25: TOP_ZONE → n=3 ❌ INSUFFICIENT (n=3)
- ATR 0.25: BOTTOM_ZONE → n=1 ❌ INSUFFICIENT (n=1)
- ATR 0.25: MID → n=212 ⚠️ UNDERPOWERED (n=212)
- ATR 0.5: TOP_ZONE → n=5 ❌ INSUFFICIENT (n=5)
- ATR 0.5: BOTTOM_ZONE → n=4 ❌ INSUFFICIENT (n=4)
- ATR 0.5: MID → n=207 ⚠️ UNDERPOWERED (n=207)

---

## Zone Analysis — ATR Threshold 0.1

| Zone | N | Win Rate | Mean PnL | Med MFE | Med MAE | Med R:R | CI95 WR | CI95 MFE | CI95 R:R | Symbols |
|------|---|----------|----------|---------|---------|---------|---------|----------|----------|---------|
| TOP_ZONE | 1 | 0.0% | $-1.15 | 0.000% | 0.174% | 0.000 | [nan%,nan%] | [nan%,nan%] | [nan,nan] | 1 |
| BOTTOM_ZONE | 1 | 100.0% | $+16.40 | 0.000% | 0.000% | 0.000 | [nan%,nan%] | [nan%,nan%] | [nan,nan] | 1 |
| MID | 214 | 43.0% | $-0.62 | 0.000% | 0.272% | 0.000 | [36.4%,49.5%] | [0.000%,0.119%] | [0.000,0.606] | 26 |

**Permutation tests vs MID:**
- TOP_ZONE: p(WR)=nan, p(PnL)=nan, p(MFE)=nan, p(MAE)=nan, p(R:R)=nan
- BOTTOM_ZONE: p(WR)=nan, p(PnL)=nan, p(MFE)=nan, p(MAE)=nan, p(R:R)=nan

---

## Zone Analysis — ATR Threshold 0.25

| Zone | N | Win Rate | Mean PnL | Med MFE | Med MAE | Med R:R | CI95 WR | CI95 MFE | CI95 R:R | Symbols |
|------|---|----------|----------|---------|---------|---------|---------|----------|----------|---------|
| TOP_ZONE | 3 | 0.0% | $-4.52 | 0.000% | 0.348% | 0.000 | [nan%,nan%] | [nan%,nan%] | [nan,nan] | 2 |
| BOTTOM_ZONE | 1 | 100.0% | $+16.40 | 0.000% | 0.000% | 0.000 | [nan%,nan%] | [nan%,nan%] | [nan,nan] | 1 |
| MID | 212 | 43.4% | $-0.57 | 0.000% | 0.259% | 0.000 | [36.3%,50.0%] | [0.000%,0.113%] | [0.000,0.588] | 26 |

**Permutation tests vs MID:**
- TOP_ZONE: p(WR)=nan, p(PnL)=nan, p(MFE)=nan, p(MAE)=nan, p(R:R)=nan
- BOTTOM_ZONE: p(WR)=nan, p(PnL)=nan, p(MFE)=nan, p(MAE)=nan, p(R:R)=nan

---

## Zone Analysis — ATR Threshold 0.5

| Zone | N | Win Rate | Mean PnL | Med MFE | Med MAE | Med R:R | CI95 WR | CI95 MFE | CI95 R:R | Symbols |
|------|---|----------|----------|---------|---------|---------|---------|----------|----------|---------|
| TOP_ZONE | 5 | 20.0% | $-2.93 | 0.000% | 0.348% | 0.000 | [0.0%,60.0%] | [0.000%,0.697%] | [0.000,20.000] | 2 |
| BOTTOM_ZONE | 4 | 75.0% | $+12.98 | 0.000% | 0.000% | 0.000 | [nan%,nan%] | [nan%,nan%] | [nan,nan] | 3 |
| MID | 207 | 43.0% | $-0.75 | 0.000% | 0.271% | 0.000 | [36.2%,50.2%] | [0.000%,0.121%] | [0.000,0.702] | 26 |
| **EDGE (TOP+BOT)** | 9 | 44.4% | $+4.14 | 0.000% | 0.174% | 0.000 | — | — | — | 5 |

**Permutation tests vs MID:**
- TOP_ZONE: p(WR)=0.3985, p(PnL)=0.6220, p(MFE)=0.4745, p(MAE)=0.6675, p(R:R)=0.5975
- BOTTOM_ZONE: p(WR)=nan, p(PnL)=nan, p(MFE)=nan, p(MAE)=nan, p(R:R)=nan
- EDGE_vs_MID: p(WR)=1.0000, p(PnL)=0.2005, p(MFE)=0.2025, p(MAE)=0.3630, p(R:R)=0.2340

---

## Pressure Precursor Analysis by Zone

Events with pressure profiles: 73

**BOTTOM_ZONE (n=1):**
- Mean peak_pressure_z: 1.882
- Mean buildup_rate: 0.0063
- Mean volume_acceleration: -47.12
- Precursor frequency (peak > 1.5): 1/1 (100.0%)
- Lead time: mean=156.3s, median=156.3s

**MID (n=72):**
- Mean peak_pressure_z: 2.309
- Mean buildup_rate: 0.0028
- Mean volume_acceleration: -14.58
- Precursor frequency (peak > 1.5): 52/72 (72.2%)
- Lead time: mean=121.5s, median=136.5s

---

## Fade Behavior by Zone (peak_z >= 2.0)

| Zone | N | Win Rate | Mean PnL | Med MFE | Med MAE |
|------|---|----------|----------|---------|---------|
| MID | 45 | 46.7% | $-0.57 | 0.000% | 0.245% |


---

## Stability: Time-of-Day

| Bucket | N | N Top | N Bot | N Mid | WR Top | WR Bot | WR Mid |
|--------|---|-------|-------|-------|--------|--------|--------|
| pre_market | 77 | 3 | 1 | 73 | 0.0% | 100.0% | 37.0% |
| open | 27 | 0 | 0 | 27 | N/A | N/A | 44.4% |
| mid | 43 | 0 | 0 | 43 | N/A | N/A | 44.2% |
| close | 31 | 0 | 0 | 31 | N/A | N/A | 54.8% |

## Stability: Per-Ticker

| Sym | N | N Top | N Bot | N Mid | WR Top | WR Bot | WR Mid |
|-----|---|-------|-------|-------|--------|--------|--------|
| ANL | 8 | 0 | 0 | 8 | N/A | N/A | 50.0% |
| AQST | 11 | 0 | 0 | 11 | N/A | N/A | 45.5% |
| BATL | 21 | 0 | 0 | 21 | N/A | N/A | 38.1% |
| BOXL | 11 | 0 | 0 | 11 | N/A | N/A | 36.4% |
| CATX | 7 | 0 | 0 | 7 | N/A | N/A | 42.9% |
| CISS | 38 | 0 | 0 | 38 | N/A | N/A | 47.4% |
| ELPW | 17 | 0 | 0 | 17 | N/A | N/A | 41.2% |
| LIMN | 10 | 0 | 0 | 10 | N/A | N/A | 30.0% |
| MRNO | 13 | 0 | 0 | 13 | N/A | N/A | 23.1% |
| MSTZ | 8 | 0 | 0 | 8 | N/A | N/A | 50.0% |
| NAMM | 16 | 0 | 0 | 16 | N/A | N/A | 43.8% |
| RDW | 8 | 0 | 0 | 8 | N/A | N/A | 37.5% |
| SOLT | 11 | 1 | 0 | 10 | 0.0% | N/A | 50.0% |
| VHUB | 9 | 0 | 0 | 9 | N/A | N/A | 66.7% |
| ZSL | 10 | 2 | 0 | 8 | 0.0% | N/A | 37.5% |

---

## Conclusion

*Auto-generated. Interpret based on p-values, sample sizes, and stability.*
