# Missed Trades Analysis
## Trades the filter would block that had high MFE
## Generated: 2026-03-06

---

## TOP 20 BLOCKED TRADES BY MFE (>=1%%)

| # | Symbol | Price | MFE%% | Actual PnL | Block Reasons | Session | Regime |
|---|--------|-------|-------|-----------|--------------|---------|--------|
| 1 | TMDE | $3.42 | 25.82% | $-25 | LOW_VOL, SUPPRESS_REGIME | premarket | LOW_VOLATILITY |
| 2 | USEG | $1.24 | 11.29% | $-3 | HIGH_SPREAD, SUPPRESS_REGIME | premarket | LOW_VOLATILITY |
| 3 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 4 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 5 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 6 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 7 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 8 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 9 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 10 | BATL | $27.51 | 10.76% | $+13200 | LOW_VOL | premarket | RANGE_BOUND |
| 11 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 12 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 13 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 14 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 15 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 16 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 17 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 18 | BATL | $22.72 | 7.26% | $+6850 | HIGH_SPREAD | premarket | RANGE_BOUND |
| 19 | BATL | $21.21 | 7.09% | $+6225 | LOW_VOL, HIGH_SPREAD, SUPPRESS_REGIME | premarket | LOW_VOLATILITY |
| 20 | BATL | $21.21 | 7.09% | $+6225 | LOW_VOL, HIGH_SPREAD, SUPPRESS_REGIME | premarket | LOW_VOLATILITY |

## MISSED vs AVOIDED

| Metric | Value |
|--------|-------|
| Total blocked trades | 1999 |
| Blocked winners | 645 |
| Blocked losers | 1354 |
| Missed profit (winners blocked) | $+1070185 |
| Avoided losses (losers blocked) | $+1617021 |
| Net filter value | $+546837 |

**The filter avoids $+546837 more in losses than it misses in profits.**

---

*Data source: live_signals.json, *_quotes.json (READ-ONLY)*
*NO production changes were made.*