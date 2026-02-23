# Phase 10B.1 — Structural Liquidity Zone Conditioning (FIXED)

**Date:** 2026-02-22
**Fix:** Replaced Databento-only daily OHLC with yfinance. Coverage: ~90%+
**Total events:** 1944
**Active (PnL ≠ 0):** 1148
**ATR 0.1:** labeled=1148 (100%), TOP=23, BOT=3, MID=1122
**ATR 0.25:** labeled=1148 (100%), TOP=63, BOT=6, MID=1079
**ATR 0.5:** labeled=1148 (100%), TOP=122, BOT=13, MID=1013
**ATR 1.0:** labeled=1148 (100%), TOP=230, BOT=38, MID=880

---

## Zone Analysis — ATR 0.1

| Zone | N | WR | Mean PnL | Total PnL | Med MFE | Med MAE | Med R:R | CI95 WR | Syms |
|------|---|----|----------|-----------|---------|---------|---------|---------|------|
| TOP_ZONE | 23 | 47.8% | $-1.17 | $-27 | 0.000% | 0.040% | 0.000 | [26.1%,69.6%] | 8 |
| BOTTOM_ZONE | 3 | 66.7% | $+0.26 | $+1 | 0.813% | 0.000% | 20.000 | [nan%,nan%] | 2 |
| MID | 1122 | 48.5% | $-0.25 | $-283 | 0.037% | 0.173% | 0.198 | [45.5%,51.3%] | 135 |
| EDGE | 26 | 50.0% | $-1.01 | $-26 | 0.001% | 0.020% | 10.000 | [nan%,nan%] | 10 |

**Permutation tests:**
- TOP_ZONE: p(WR)=1.0000, p(PnL)=0.7295, p(MFE)=0.0965
- BOTTOM_ZONE: p(WR)=nan, p(PnL)=nan, p(MFE)=nan
- EDGE_vs_MID: p(WR)=1.0000, p(PnL)=0.7780, p(MFE)=0.1070


## Zone Analysis — ATR 0.25

| Zone | N | WR | Mean PnL | Total PnL | Med MFE | Med MAE | Med R:R | CI95 WR | Syms |
|------|---|----|----------|-----------|---------|---------|---------|---------|------|
| TOP_ZONE | 63 | 52.4% | $-0.60 | $-38 | 0.006% | 0.040% | 1.000 | [39.7%,65.1%] | 17 |
| BOTTOM_ZONE | 6 | 66.7% | $-0.16 | $-1 | 0.070% | 0.173% | 10.056 | [33.3%,100.0%] | 4 |
| MID | 1079 | 48.2% | $-0.25 | $-271 | 0.033% | 0.175% | 0.171 | [45.2%,51.3%] | 134 |
| EDGE | 69 | 53.6% | $-0.56 | $-39 | 0.039% | 0.040% | 1.000 | [nan%,nan%] | 20 |

**Permutation tests:**
- TOP_ZONE: p(WR)=0.5965, p(PnL)=0.8720, p(MFE)=0.0745
- BOTTOM_ZONE: p(WR)=0.4265, p(PnL)=0.9805, p(MFE)=0.4970
- EDGE_vs_MID: p(WR)=0.3920, p(PnL)=0.8710, p(MFE)=0.0770


## Zone Analysis — ATR 0.5

| Zone | N | WR | Mean PnL | Total PnL | Med MFE | Med MAE | Med R:R | CI95 WR | Syms |
|------|---|----|----------|-----------|---------|---------|---------|---------|------|
| TOP_ZONE | 122 | 50.0% | $-0.31 | $-37 | 0.005% | 0.064% | 0.773 | [41.0%,58.2%] | 25 |
| BOTTOM_ZONE | 13 | 46.2% | $+4.18 | $+54 | 0.000% | 0.959% | 0.000 | [22.9%,76.9%] | 7 |
| MID | 1013 | 48.4% | $-0.32 | $-326 | 0.037% | 0.175% | 0.196 | [45.3%,51.4%] | 132 |
| EDGE | 135 | 49.6% | $+0.12 | $+17 | 0.003% | 0.089% | 0.360 | [nan%,nan%] | 30 |

**Permutation tests:**
- TOP_ZONE: p(WR)=0.7725, p(PnL)=0.9930, p(MFE)=0.0950
- BOTTOM_ZONE: p(WR)=1.0000, p(PnL)=0.2345, p(MFE)=0.1810
- EDGE_vs_MID: p(WR)=0.8570, p(PnL)=0.7615, p(MFE)=0.1135


## Zone Analysis — ATR 1.0

| Zone | N | WR | Mean PnL | Total PnL | Med MFE | Med MAE | Med R:R | CI95 WR | Syms |
|------|---|----|----------|-----------|---------|---------|---------|---------|------|
| TOP_ZONE | 230 | 48.3% | $-0.51 | $-118 | 0.000% | 0.040% | 0.000 | [41.7%,54.8%] | 33 |
| BOTTOM_ZONE | 38 | 42.1% | $+0.27 | $+10 | 0.000% | 0.373% | 0.000 | [26.3%,57.9%] | 16 |
| MID | 880 | 48.9% | $-0.23 | $-202 | 0.067% | 0.194% | 0.333 | [45.6%,52.2%] | 123 |
| EDGE | 268 | 47.4% | $-0.40 | $-107 | 0.000% | 0.086% | 0.000 | [nan%,nan%] | 45 |

**Permutation tests:**
- TOP_ZONE: p(WR)=0.8790, p(PnL)=0.8465, p(MFE)=0.0080
- BOTTOM_ZONE: p(WR)=0.4835, p(PnL)=0.8365, p(MFE)=0.5710
- EDGE_vs_MID: p(WR)=0.6595, p(PnL)=0.9065, p(MFE)=0.0100

## Pressure Precursor Analysis by Zone

**TOP_ZONE** (n=23): peak_z=1.999, buildup=-0.0119, vol_accel=7.81, precursor_freq=60.9%
**BOTTOM_ZONE** (n=1): peak_z=3.319, buildup=-0.0224, vol_accel=-14.85, precursor_freq=100.0%
**MID** (n=252): peak_z=2.282, buildup=0.0067, vol_accel=40.17, precursor_freq=68.3%

## Stability: Time-of-Day

| Bucket | N | TOP | BOT | MID | WR TOP | WR BOT | WR MID |
|--------|---|-----|-----|-----|--------|--------|--------|
| pre_market | 355 | 14 | 1 | 340 | 50.0% | 0.0% | 46.2% |
| open | 135 | 3 | 0 | 132 | 33.3% | N/A | 52.3% |
| mid | 201 | 27 | 0 | 174 | 51.9% | N/A | 47.7% |
| close | 182 | 11 | 4 | 167 | 72.7% | 75.0% | 46.7% |

## Stability: Per-Ticker (n ≥ 5)

| Sym | N | TOP | BOT | MID | WR TOP | WR BOT | WR MID |
|-----|---|-----|-----|-----|--------|--------|--------|
| ALXO | 45 | 1 | 0 | 44 | 100.0% | N/A | 56.8% |
| ANL | 9 | 0 | 0 | 9 | N/A | N/A | 55.6% |
| AQST | 17 | 0 | 0 | 17 | N/A | N/A | 52.9% |
| AREB | 20 | 0 | 0 | 20 | N/A | N/A | 55.0% |
| ASTI | 12 | 0 | 0 | 12 | N/A | N/A | 33.3% |
| AUID | 13 | 0 | 0 | 13 | N/A | N/A | 69.2% |
| BATL | 21 | 0 | 0 | 21 | N/A | N/A | 38.1% |
| BGMS | 5 | 0 | 0 | 5 | N/A | N/A | 40.0% |
| BMNU | 8 | 0 | 0 | 8 | N/A | N/A | 62.5% |
| BOXL | 18 | 0 | 0 | 18 | N/A | N/A | 27.8% |
| BTCZ | 7 | 0 | 0 | 7 | N/A | N/A | 57.1% |
| BTQ | 24 | 1 | 0 | 23 | 0.0% | N/A | 34.8% |
| CATX | 13 | 0 | 0 | 13 | N/A | N/A | 46.2% |
| CDIO | 5 | 1 | 0 | 4 | 0.0% | N/A | 50.0% |
| CISS | 143 | 11 | 0 | 132 | 45.5% | N/A | 42.4% |
| CONL | 22 | 3 | 0 | 19 | 0.0% | N/A | 47.4% |
| DCX | 10 | 0 | 0 | 10 | N/A | N/A | 40.0% |
| DFDV | 14 | 0 | 0 | 14 | N/A | N/A | 50.0% |
| DHX | 16 | 0 | 0 | 16 | N/A | N/A | 43.8% |
| DOGZ | 34 | 0 | 0 | 34 | N/A | N/A | 38.2% |
| DPRO | 30 | 7 | 0 | 23 | 28.6% | N/A | 43.5% |
| DRMA | 10 | 0 | 1 | 9 | N/A | 100.0% | 44.4% |
| DUO | 44 | 0 | 0 | 44 | N/A | N/A | 47.7% |
| EGHT | 18 | 2 | 0 | 16 | 50.0% | N/A | 50.0% |
| ELPW | 24 | 0 | 2 | 22 | N/A | 50.0% | 50.0% |
| FATBB | 9 | 0 | 0 | 9 | N/A | N/A | 66.7% |
| FSLY | 8 | 0 | 0 | 8 | N/A | N/A | 50.0% |
| FUSE | 21 | 0 | 0 | 21 | N/A | N/A | 28.6% |
| GDXD | 20 | 0 | 0 | 20 | N/A | N/A | 75.0% |
| GMM | 8 | 0 | 0 | 8 | N/A | N/A | 25.0% |
| JDZG | 10 | 0 | 0 | 10 | N/A | N/A | 80.0% |
| LFS | 7 | 0 | 0 | 7 | N/A | N/A | 42.9% |
| LIMN | 24 | 0 | 0 | 24 | N/A | N/A | 37.5% |
| LUNR | 10 | 1 | 0 | 9 | 100.0% | N/A | 66.7% |
| MAXN | 6 | 0 | 0 | 6 | N/A | N/A | 83.3% |
| MRNO | 17 | 0 | 0 | 17 | N/A | N/A | 29.4% |
| MSTZ | 9 | 0 | 1 | 8 | N/A | 100.0% | 50.0% |
| NAMM | 35 | 0 | 0 | 35 | N/A | N/A | 48.6% |
| OCG | 9 | 0 | 0 | 9 | N/A | N/A | 55.6% |
| PINS | 10 | 8 | 0 | 2 | 50.0% | N/A | 50.0% |
| PLRZ | 15 | 3 | 0 | 12 | 66.7% | N/A | 50.0% |
| RDW | 10 | 0 | 0 | 10 | N/A | N/A | 40.0% |
| RGTI | 11 | 0 | 0 | 11 | N/A | N/A | 81.8% |
| RIVN | 6 | 1 | 0 | 5 | 100.0% | N/A | 60.0% |
| RNAZ | 9 | 0 | 0 | 9 | N/A | N/A | 33.3% |
| SMCL | 56 | 19 | 0 | 37 | 73.7% | N/A | 48.6% |
| SOLT | 13 | 1 | 2 | 10 | 0.0% | 50.0% | 50.0% |
| STIM | 8 | 0 | 0 | 8 | N/A | N/A | 62.5% |
| VHUB | 12 | 0 | 0 | 12 | N/A | N/A | 50.0% |
| VIVS | 8 | 0 | 0 | 8 | N/A | N/A | 62.5% |
| WATT | 24 | 0 | 0 | 24 | N/A | N/A | 58.3% |
| WHLR | 5 | 0 | 0 | 5 | N/A | N/A | 40.0% |
| WHLRP | 10 | 0 | 0 | 10 | N/A | N/A | 60.0% |
| ZSL | 12 | 0 | 0 | 12 | N/A | N/A | 41.7% |
