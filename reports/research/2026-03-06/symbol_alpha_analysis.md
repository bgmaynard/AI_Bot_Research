# 6. Symbol Opportunity Analysis
## Data: Symbol Edge Gate, Signal Ledger, Paper Trades, Quote Cache (2026-03-03)
## Generated: 2026-03-06

---

## Executive Summary

Only **BATL** produced executed trades (20 paper trades, +$21,699). The symbol edge gate
classifies BATL and PLUG as PREFERRED, while USEG, TMDE, and CRCD are classified AVOID.
Key symbols RDW, BBAI, and RGTI were **not present** in the 2026-03-03 scanner data.
BATL concentration risk is extreme — 100% of alpha from one symbol.

---

## 1. Symbol Edge Gate Rankings

| Symbol | Composite Score | Classification | Expectancy | Microstructure | Edge WR |
|--------|----------------|---------------|------------|---------------|---------|
| **BATL** | **+0.586** | **PREFERRED** | 0.961 | 1.000 | 0.000 |
| **PLUG** | **+0.469** | **PREFERRED** | 1.000 | 0.147 | 0.412 |
| VG | +0.357 | NEUTRAL | 0.626 | -0.248 | 1.000 |
| DUST | +0.309 | NEUTRAL | 0.814 | -0.169 | 0.334 |
| MSTZ | +0.300 | NEUTRAL | 0.766 | -0.141 | 0.334 |
| UVIX | +0.073 | NEUTRAL | 0.236 | 0.304 | -0.428 |
| SOXS | +0.041 | NEUTRAL | 0.230 | 0.026 | -0.230 |
| IONZ | -0.046 | NEUTRAL | 0.027 | 0.046 | -0.334 |
| **TMDE** | **-0.315** | **AVOID** | 0.043 | -1.000 | -0.400 |
| **USEG** | **-0.410** | **AVOID** | -0.771 | 0.240 | -1.000 |
| **CRCD** | **-0.620** | **AVOID** | -1.000 | -0.282 | -1.000 |

**BATL dominates** on both expectancy (0.961) and microstructure (1.000 — highest
micro_score of 11.09). PLUG has the best raw expectancy but lower microstructure.

---

## 2. Signal Density by Symbol (2026-03-03)

| Symbol | Total Blocks | Signal Count | Ign. Pass | Risk Apprvd | Exec Blocks |
|--------|-------------|-------------|-----------|-------------|-------------|
| TPET | 4,639 | ~4,600+ | - | - | 0 |
| BATL | 4,415 | ~4,400+ | Many | Many | 0 |
| NPT | 2,955 | ~2,900+ | - | - | 0 |
| TMDE | 2,666 | ~2,600+ | Some | Some | 2 |
| USEG | 1,615 | ~1,600+ | Some | Some | 2 |
| STAK | 1,481 | ~1,400+ | - | - | 0 |
| RUBI | 1,359 | ~1,300+ | - | - | 0 |
| UVIX | 1,297 | ~1,200+ | - | - | 0 |
| PLUG | 1,182 | ~1,100+ | Some | Some | 1 |
| ZSL | 1,029 | ~1,000+ | - | - | 0 |

TPET generated the most signals but has NO executed trades and NO edge gate score —
likely blocked entirely at extension/ignition.

---

## 3. Execution Rate

| Symbol | Paper Trades | PnL | Notes |
|--------|-------------|-----|-------|
| BATL | 20 | +$21,699 | Only symbol to execute |
| All others | 0 | $0 | Blocked at various gates |

**Execution rate: 1 of 11 symbols (9%)**. This is a diversification failure.

---

## 4. BATL Deep Dive — Is It Producing Alpha?

| Metric | Value |
|--------|-------|
| Trades | 20 |
| Win Rate | 55% |
| Profit Factor | 4.07 |
| Total PnL | +$21,698.98 |
| Avg Hold | 80.3s |
| Avg MFE | 2.38% |
| Avg MAE | 0.46% |
| Micro Score | 11.092 (highest) |
| Expectancy Raw | 0.320 |

**Yes — BATL is producing strong alpha.** PF of 4.07 with 55% WR means winners are
3x the size of losers. The microstructure score of 11.09 (best in universe) indicates
tight spreads, good liquidity, and clean price action.

**But**: 130 BATL signals were rejected (MAX_TRADES_PER_DAY cap). With BATL running
from $17.86 to $33.48 (+87% intraday), significant alpha was left uncaptured in the
$25-$33 range.

---

## 5. Focus Symbols: RDW, BBAI, RGTI

**None of these symbols appear in the 2026-03-03 data set.** They were not discovered
by the MAX_AI scanner on this date.

| Symbol | In Signal Ledger | In Quote Cache | In Edge Gate | Assessment |
|--------|-----------------|---------------|-------------|------------|
| RDW | No | No | No | Not scanned |
| BBAI | No | No | No | Not scanned |
| RGTI | No | No | No | Not scanned |

These may be symbols from a different trading day or added to the scanner's universe
after Mar 3. Cannot assess their alpha contribution without data.

---

## 6. Over-Filtered Symbols

| Symbol | Signals | Edge Score | Class | Why Filtered | Potentially Tradeable? |
|--------|---------|-----------|-------|-------------|----------------------|
| PLUG | 1,182 | +0.469 | PREFERRED | Spread gate (18 blocks) | **YES** — best expectancy |
| VG | - | +0.357 | NEUTRAL | Unknown | Maybe |
| DUST | - | +0.309 | NEUTRAL | Unknown | Maybe |
| SOXS | - | +0.041 | NEUTRAL | Spread gate (10 blocks) | Marginal |

**PLUG is the most over-filtered symbol** — it has the highest raw expectancy (0.357)
and is classified PREFERRED, but 18 spread-gate blocks prevented any execution.
Relaxing the spread threshold to 0.8% would unblock most PLUG signals.

---

## 7. Concentration Risk

| Metric | Current | Ideal |
|--------|---------|-------|
| Symbols executing | 1 | 3-5 |
| % PnL from top symbol | 100% | <40% |
| Avg correlation | N/A (1 symbol) | <0.5 |

**Extreme concentration risk.** If BATL had a bad day, the entire strategy returns nothing.
PLUG should be the #2 symbol (PREFERRED classification, good expectancy) but is blocked
by spread gates.

---

## 8. Recommendations

1. **P0**: Unblock PLUG by relaxing spread threshold (see Containment Study)
2. **P1**: Raise MAX_TRADES_PER_DAY cap from 20 to 30-40 for PREFERRED symbols
3. **P2**: Add RDW, BBAI, RGTI to scanner universe if not already present
4. **P3**: Implement per-symbol position limits instead of global trade cap
5. **P4**: Monitor AVOID symbols (USEG, TMDE, CRCD) — they consume signal bandwidth with negative expectancy

---

*Data sources: symbol_edge_gate_2026-03-03.json, paper_trades.json,
last_mile_results.json, watchlist classifier (READ-ONLY)*
*NO production changes were made. This is research-only analysis.*
