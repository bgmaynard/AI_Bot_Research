# 1. Trade Outcome Analysis
## Data: 2026-03-03 Paper Trades (20 executed) + 2026-03-02 Trade Ledger (30 records)
## Generated: 2026-03-06

---

## Executive Summary

**20 paper trades executed on 2026-03-03**, all on BATL via HYBRID_ENTRY strategy.
55% win rate, profit factor 4.07, total PnL **+$21,698.98** on $100K account (+21.7%).
Average hold time 80.3 seconds. 130 additional signals rejected by MAX_TRADES_PER_DAY cap.

**Zero live trades executed Mar 2-4** due to last-mile execution blocks (data staleness,
containment rechecks). All analysis below uses the paper trade simulation.

---

## 1. PnL Distribution

| Bucket | Count | % |
|--------|-------|---|
| > +$3,000 | 4 | 20% |
| +$500 to +$3,000 | 3 | 15% |
| $0 to +$500 | 4 | 20% |
| -$500 to $0 | 2 | 10% |
| -$1,000 to -$500 | 4 | 20% |
| < -$1,000 | 3 | 15% |

**PnL Statistics:**
- Mean: +$1,084.95
- Median: +$53.77
- Std Dev: ~$2,600
- Max Win: +$8,613.50 (Trade #10, +7.80%)
- Max Loss: -$1,360.32 (Trade #13, -1.16%)
- Skew: Heavily positive (fat right tail from 4 large winners)

---

## 2. Hold Time Distribution

| Bucket | Count | % | Avg PnL |
|--------|-------|---|---------|
| < 20s | 3 | 15% | -$909.27 |
| 20-60s | 5 | 25% | +$1,517.33 |
| 60-120s | 4 | 20% | +$3,951.38 |
| 120-180s | 5 | 25% | +$1,299.99 |
| > 180s | 3 | 15% | +$691.00 |

**Key Finding**: Trades held < 20s are universally losers (avg -$909). Best returns
cluster in the 30-130s window. The trail stop is cutting losers quickly (good) but
may also be clipping some winners early.

---

## 3. Win vs Loss Analysis

| Metric | Winners (11) | Losers (9) |
|--------|-------------|-----------|
| Count | 11 (55%) | 9 (45%) |
| Avg PnL | +$2,699.63 | -$886.57 |
| Avg Hold | 102.7s | 52.9s |
| Avg Return | +2.60% | -0.70% |
| Max | +$8,613.50 | -$1,360.32 |
| Min | +$27.65 | -$117.98 |

**Win/Loss Ratio: 3.05x** -- Winners are 3x the size of losers on average.
This is the core edge: small losers cut fast, big winners ride the trail.

---

## 4. Per-Symbol Performance

All 20 trades were on BATL. No diversification.

| Symbol | Trades | WR | Total PnL | Avg PnL | Avg Hold |
|--------|--------|-----|-----------|---------|----------|
| BATL | 20 | 55% | +$21,698.98 | +$1,084.95 | 80.3s |

**130 rejected BATL signals** (MAX_TRADES_PER_DAY cap of 20) had entry prices
ranging from $20.64 to $33.48 — BATL ran from $17.86 to $33.48 intraday.
The rejected signals in the $25-33 range represent massive unrealized alpha.

---

## 5. Per-Strategy Performance

| Strategy | Trades | WR | Total PnL | Notes |
|----------|--------|-----|-----------|-------|
| HYBRID_ENTRY | 20 | 55% | +$21,698.98 | Only strategy that executed |

No other strategies produced executed trades. All other strategies failed at
ignition gate or earlier. See Strategy Coverage Analysis (Module 7) for details.

---

## 6. Regime Performance

| Regime | Trades | WR | Avg Return | Total PnL |
|--------|--------|----|-----------|-----------|
| RANGE_BOUND | 11 | 64% | +2.24% | +$12,719.05 |
| LOW_VOLATILITY | 9 | 44% | +0.34% | +$8,979.93 |

RANGE_BOUND regime produced significantly better results. LOW_VOLATILITY trades
had lower win rate but still net positive due to large outlier winners (Trade #19: +2.07%).

---

## 7. Equity Curve

```
Trade #:   1     5     10    15    20
PnL ($): -588  +5,032 +19,047 +14,295 +21,699
         ▼     ▲▲▲   ▲▲▲▲▲  ▼▼    ▲▲▲
```

Peak equity reached **$119,047** after Trade #10 (the +$8,614 monster). Drawdown
to $114,295 after trades 11-15 (5 of 6 losing), then recovery to $121,699.

**Max drawdown: 3.99%** ($3,991 from peak). Well within risk limits.

---

*Data sources: paper_trades.json, broker_execution_log.json, trades_2026-03-02.json (READ-ONLY)*
*NO production changes were made. This is research-only analysis.*
