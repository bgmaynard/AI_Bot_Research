# Phase 9 — Ignition Precursors + Quality Filters: Results

**Date:** 2026-02-21
**Dataset:** 420 trades with valid pressure profiles / 1,944 total Morpheus trades
**Period:** 18 trading days (2026-01-29 to 2026-02-20)
**Symbols:** 64 unique tickers

---

## Executive Summary

**Verdict:** **CONDITIONAL GO** — promising but needs more data
**Basis:** Combined gate improves win rate (44.3% vs 43.8% baseline) but p=0.8850. Stable in 3/4 ToD buckets.

**Best slope filter:** slope >= 0.005 (mild)
  — n=156, WR=44.2% (baseline 43.8%), p(WR)=0.9125, CI95=[36.5%, 51.9%]

**Best volume filter:** vol_accel >= P80 (48.0)
  — n=84, WR=47.6% (baseline 43.8%), p(WR)=0.4745, CI95=[36.9%, 59.5%]

**Combined gate:** slope >= 0.0, vol_accel >= P80 (48.0)
  — n=70, WR=44.3% (baseline 43.8%), R:R=0.000 (baseline 0.000)
  — CI95(WR): [32.9%, 55.7%]
  — CI95(R:R): [0.000, 10.000]
  — Ticker stability: 5/11 improved
  — ToD stability: 3/4 buckets improved

---

## Baseline (All Trades With Profiles)

| Metric | Value |
|--------|-------|
| Total trades | 420 |
| Win rate | 43.8% |
| Mean PnL | $0.00 |
| Total PnL | $0.33 |
| Median MFE | 0.00% |
| Median MAE | 0.08% |
| Median R:R | 0.000 |
| Unique symbols | 64 |
| Unique dates | 15 |

---

## Experiment 1 — HYP-023: Pressure Slope Filter

| Threshold | N Pass | WR Pass | WR Reject | Δ WR | p(WR) | Mean PnL Pass | R:R Pass |
|-----------|--------|---------|-----------|------|-------|---------------|----------|
| slope >= -0.01 (loose) | 286 | 41.6% | 48.5% | -6.9pp | 0.2065 | $-0.03 | 0.000 |
| slope >= 0 (neutral) | 243 | 41.6% | 46.9% | -5.3pp | 0.3195 | $+0.36 | 0.000 |
| slope >= 0.005 (mild) | 156 | 44.2% | 43.6% | +0.7pp | 0.9125 | $+0.86 | 0.000 |
| slope >= 0.01 (moderate) | 129 | 44.2% | 43.6% | +0.5pp | 1.0000 | $+1.45 | 0.000 |
| slope >= 0.02 (strong) | 91 | 44.0% | 43.8% | +0.2pp | 1.0000 | $+3.67 | 0.000 |

---

## Experiment 2 — HYP-024: Volume Acceleration Threshold

| Quantile | Threshold | N Pass | WR Pass | WR Reject | Δ WR | p(WR) | Mean PnL Pass | R:R Pass |
|----------|-----------|--------|---------|-----------|------|-------|---------------|----------|
| P50 | 0.0 | 232 | 43.5% | 44.1% | -0.6pp | 0.9220 | $-1.04 | 0.000 |
| P60 | 0.0 | 232 | 43.5% | 44.1% | -0.6pp | 0.9220 | $-1.04 | 0.000 |
| P70 | 15.2 | 126 | 46.8% | 42.5% | +4.3pp | 0.4560 | $-1.46 | 0.000 |
| P80 | 48.0 | 84 | 47.6% | 42.9% | +4.8pp | 0.4745 | $-1.81 | 0.000 |
| P90 | 147.5 | 42 | 42.9% | 43.9% | -1.1pp | 1.0000 | $-4.04 | 0.000 |

---

## Experiment 3 — HYP-025: Combined Gate

Gate: `buildup_rate >= 0.0` AND `volume_acceleration >= 48.0` (P80)

| Group | N | Win Rate | Mean PnL | Median MFE | Median MAE | Median R:R |
|-------|---|----------|----------|------------|------------|------------|
| ALL (baseline) | 420 | 43.8% | $+0.00 | 0.00% | 0.08% | 0.000 |
| BOTH PASS | 70 | 44.3% | $-2.84 | 0.00% | 0.13% | 0.000 |
| SLOPE ONLY | 173 | 40.5% | $+1.65 | 0.00% | 0.05% | 0.000 |
| VOL ONLY | 14 | 64.3% | $+3.30 | 0.16% | 0.00% | 0.750 |
| NEITHER | 163 | 45.4% | $-0.82 | 0.00% | 0.15% | 0.000 |

**BOTH vs NEITHER:** p(WR)=0.8850 (ns), p(PnL)=0.2740
**BOTH vs BASELINE:** p(WR)=1.0000, p(PnL)=0.2975

---

## Stability: Time-of-Day

| Bucket | N All | N Pass | WR All | WR Pass | Δ WR | PnL All | PnL Pass |
|--------|-------|--------|--------|---------|------|---------|----------|
| pre_market | 163 | 23 | 41.1% | 26.1% | -15.0pp | $-0.13 | $-8.33 |
| open | 90 | 16 | 48.9% | 56.2% | +7.4pp | $+0.21 | $-0.24 |
| mid | 116 | 21 | 45.7% | 52.4% | +6.7pp | $+0.13 | $+0.48 |
| close | 51 | 10 | 39.2% | 50.0% | +10.8pp | $-0.22 | $-1.30 |

---

## Stability: Per-Ticker (n >= 10)

| Symbol | N All | N Pass | WR All | WR Pass | Δ WR |
|--------|-------|--------|--------|---------|------|
| ALXO | 13 | 2 | 53.8% | 50.0% | -3.8pp |
| AQST | 16 | 5 | 25.0% | 40.0% | +15.0pp |
| BOXL | 12 | 1 | 33.3% | 100.0% | +66.7pp |
| CATX | 10 | 0 | 60.0% | N/A | N/A |
| CISS | 99 | 15 | 41.4% | 26.7% | -14.7pp |
| CONL | 16 | 4 | 37.5% | 75.0% | +37.5pp |
| DPRO | 12 | 0 | 25.0% | N/A | N/A |
| LIMN | 12 | 1 | 25.0% | 0.0% | -25.0pp |
| LUNR | 12 | 1 | 58.3% | 100.0% | +41.7pp |
| MSTZ | 11 | 0 | 45.5% | N/A | N/A |
| NAMM | 13 | 2 | 30.8% | 0.0% | -30.8pp |
| PINS | 12 | 3 | 41.7% | 33.3% | -8.3pp |
| RDW | 11 | 1 | 36.4% | 0.0% | -36.4pp |
| RGTI | 11 | 0 | 81.8% | N/A | N/A |
| SMCL | 24 | 6 | 54.2% | 83.3% | +29.2pp |
