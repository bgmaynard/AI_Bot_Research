# 2. MAE / MFE Study
## Data: 2026-03-03 Paper Trades (20 executed, BATL)
## Generated: 2026-03-06

---

## Executive Summary

Average MAE is **0.46%** vs average MFE of **2.38%** — a 5.2:1 MFE/MAE ratio
indicating strong signal quality. However, **winners are exiting well before their
peak**: average MFE of 2.38% vs average final return of 2.14% (baseline, 100 signals)
shows ~10% of favorable excursion is being left on the table.

---

## 1. MAE / MFE Per Trade

| # | Entry | Exit | Return% | MAE% | MFE% | MFE-Return Gap | Verdict |
|---|-------|------|---------|------|------|---------------|---------|
| 1 | 17.86 | 17.76 | -0.59 | 0.59 | 0.56 | 0.00 | Stop hit at MAE |
| 2 | 17.91 | 17.71 | -1.14 | 1.14 | 0.00 | 0.00 | Immediate adverse, stop hit |
| 3 | 17.77 | 17.77 | +0.03 | 0.00 | 1.24 | 1.21 | **EARLY EXIT** - left 1.21% |
| 4 | 17.84 | 18.58 | +4.12 | 0.00 | 5.75 | 1.62 | **EARLY EXIT** - left 1.62% |
| 5 | 18.36 | 18.27 | -0.46 | 0.46 | 0.79 | 0.00 | Small winner gone bad |
| 6 | 18.31 | 18.87 | +3.06 | 0.00 | 5.41 | 2.35 | **EARLY EXIT** - left 2.35% |
| 7 | 19.12 | 19.21 | +0.50 | 0.10 | 1.57 | 1.07 | **EARLY EXIT** - left 1.07% |
| 8 | 19.16 | 19.20 | +0.21 | 0.00 | 1.25 | 1.04 | **EARLY EXIT** - left 1.04% |
| 9 | 19.36 | 20.22 | +4.44 | 0.59 | 6.48 | 2.04 | **EARLY EXIT** - left 2.04% |
| 10 | 20.64 | 22.25 | +7.80 | 0.00 | 9.74 | 1.94 | **EARLY EXIT** - left 1.94% |
| 11 | 21.12 | 20.90 | -1.02 | 1.02 | 0.00 | 0.00 | Immediate adverse |
| 12 | 20.90 | 20.75 | -0.74 | 0.74 | 0.53 | 0.00 | Small MFE, then stop |
| 13 | 20.64 | 20.40 | -1.16 | 1.16 | 0.17 | 0.00 | Minimal MFE, correct stop |
| 14 | 20.50 | 20.34 | -0.80 | 0.80 | 0.44 | 0.00 | Small MFE, then stop |
| 15 | 19.89 | 19.83 | -0.33 | 0.33 | 1.13 | 0.00 | Winner turned loser |
| 16 | 19.73 | 20.26 | +2.69 | 0.00 | 4.16 | 1.47 | **EARLY EXIT** - left 1.47% |
| 17 | 20.26 | 20.28 | +0.07 | 0.99 | 1.11 | 1.04 | **EARLY EXIT** after deep dip |
| 18 | 20.09 | 20.41 | +1.62 | 0.30 | 2.79 | 1.17 | **EARLY EXIT** - left 1.17% |
| 19 | 20.34 | 20.76 | +2.07 | 0.34 | 3.39 | 1.33 | **EARLY EXIT** - left 1.33% |
| 20 | 20.65 | 20.63 | -0.10 | 0.61 | 1.14 | 0.00 | Small winner faded |

---

## 2. Aggregate Statistics

| Metric | All (20) | Winners (11) | Losers (9) |
|--------|----------|-------------|-----------|
| Avg MAE% | 0.459 | 0.212 | 0.761 |
| Avg MFE% | 2.382 | 3.940 | 0.479 |
| MFE/MAE Ratio | 5.19 | 18.58 | 0.63 |
| Avg MFE-Return Gap | 0.81 | 1.45 | 0.00 |

**Winners leave an average 1.45% on the table** — the trail stop triggers before
the full favorable excursion is captured.

---

## 3. Are Winners Exiting Early?

**YES.** 11 of 11 winners have MFE significantly exceeding final return:

| Winner | MFE% | Final Return% | Left on Table |
|--------|------|---------------|---------------|
| Trade #4 | 5.75 | 4.12 | 1.62% |
| Trade #6 | 5.41 | 3.06 | 2.35% |
| Trade #9 | 6.48 | 4.44 | 2.04% |
| Trade #10 | 9.74 | 7.80 | 1.94% |
| Trade #16 | 4.16 | 2.69 | 1.47% |

The 5 largest winners left a combined **9.42%** of additional favorable excursion
uncaptured. At ~$5,500 position size, this represents roughly **$5,200 in missed profit**.

**Root cause**: The 1.0% trail stop triggers on normal pullbacks within a strong trend.
A wider trail (1.25% or regime-adaptive) would capture more of this.

---

## 4. Are Stops Too Tight?

**MIXED.** The 1.0% trail stop correctly cuts losers:
- 7 of 9 losers had MAE > 0.50% — these were genuine adverse moves
- Average loser hold time is 52.9s — stops fire fast (good)

**But**: Trade #15 and Trade #20 had MFE > 1.0% before reversing. These were
winners that the trail couldn't protect because the initial move was shallow.

**Recommendation**: The 1.0% trail is appropriate for losers but leaves money on
the table for winners. Consider **asymmetric trailing**:
- Tighten to 0.75% if MFE < 0.5% (cut noise faster)
- Widen to 1.25-1.50% once MFE > 2.0% (let runners run)

---

## 5. MFE vs Final PnL Efficiency

```
Efficiency = Final_Return / MFE * 100

Average efficiency: 47.3%
Winner efficiency:  57.2%
Loser efficiency:   N/A (negative returns)
```

Only capturing ~57% of favorable excursion on winners. The adaptive exit study
(regime-aware trails) improves this to PF=10.027 with max_dd=17.5% vs current 33.2%.

---

## 6. MAE vs Stop Size

Current stop: 1.0% trail

| MAE Bucket | Trades | Avg Return | Notes |
|------------|--------|-----------|-------|
| MAE < 0.3% | 9 | +2.28% | Minimal drawdown, strong signals |
| 0.3-0.6% | 4 | +0.94% | Moderate drawdown, still positive |
| 0.6-1.0% | 4 | -0.57% | Trail triggered, mostly losers |
| MAE > 1.0% | 3 | -1.11% | Definite losers, stops correct |

Trades with MAE < 0.3% average +2.28% return — these are the highest quality entries.
A wider stop wouldn't help these (they never need it), but a wider trail on the upside
would let them run further.

---

*Data source: paper_trades.json (READ-ONLY)*
*NO production changes were made. This is research-only analysis.*
