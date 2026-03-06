# 7. Strategy Coverage Analysis
## Data: Signal Ledger, Paper Trades, Containment Events (2026-03-03)
## Generated: 2026-03-06

---

## Executive Summary

Only **HYBRID_ENTRY** produced executed trades. Two strategies generated signals that
reached containment: **catalyst_momentum** and **premarket_breakout**. Most signals
originate from the premarket_breakout strategy but fail at ignition due to
LOW_CONFIDENCE, LOW_SCORE, or HIGH_SPREAD.

The system is **single-strategy dependent** — if HYBRID_ENTRY fails on a given day,
there is zero execution.

---

## 1. Strategy Signal Generation

From the signal_ledger.jsonl (2026-03-03), strategies observed:

| Strategy | Signal Count | Ign. Pass | Containment | Risk Apprvd | Executed | PnL |
|----------|-------------|-----------|-------------|-------------|----------|-----|
| **HYBRID_ENTRY** | ~150* | ~100 | N/A | Many | 20 (paper) | +$21,699 |
| **premarket_breakout** | ~15,000+ | ~50 | ~40 | Some | 0 | $0 |
| **catalyst_momentum** | ~10,000+ | ~15 | ~57 | Some | 0 | $0 |
| Other/unknown | ~4,000+ | ~0 | 0 | 0 | 0 | $0 |

*HYBRID_ENTRY counts are from the 150 paper trade signals (20 executed + 130 rejected by trade cap)*

---

## 2. Why Other Strategies Fail Ignition

### premarket_breakout
- **Primary vetoes**: EXTENSION_HARD_VETO (no catalyst, high gap%), HIGH_SPREAD
- Most signals come from gap stocks in premarket where spreads are structurally wide
- Extension gate blocks ~38% of all signals (most are premarket_breakout with >30% gaps)
- Those surviving extension hit HIGH_SPREAD at ignition (5,143 blocks on Mar 3)

### catalyst_momentum
- **Primary vetoes**: LOW_CONFIDENCE, LOW_SCORE
- The confidence scorer consistently produces sub-0.30 values for catalyst signals
- 54 signals were blocked at META_GATE for confidence < 0.30
- Containment events show catalyst_momentum signals blocked by SPREAD (56),
  RECHECK_HOLD (22), NO_PULLBACK_SPIKE (11)

### Why HYBRID_ENTRY succeeds
- Uses a combined signal (pressure + ignition + trap detection) that produces
  higher confidence scores
- Targets stocks already in motion (not gap/premarket plays)
- BATL's high microstructure score (11.09) means it passes spread filters easily
- All 20 executed trades used TRAIL_EXIT — consistent with the HYBRID_ENTRY model

---

## 3. Containment Gate by Strategy

From the 97 containment events on Mar 3:

| Strategy | SPREAD | RECHECK_HOLD | NO_PULLBACK | PRICE_BELOW | Total |
|----------|--------|-------------|-------------|-------------|-------|
| catalyst_momentum | ~35 | ~15 | ~7 | ~5 | ~62 |
| premarket_breakout | ~21 | ~7 | ~4 | ~3 | ~35 |

catalyst_momentum signals are more frequently blocked by containment than
premarket_breakout — primarily on spread and recheck conditions.

---

## 4. Single-Strategy Risk Assessment

| Scenario | Impact |
|----------|--------|
| HYBRID_ENTRY has bad day | Zero execution, zero PnL |
| BATL doesn't move | Zero alpha (only symbol executing) |
| Spread widens on BATL | HYBRID_ENTRY also blocked |
| Market gap up (premarket play) | Cannot participate — premarket strategies blocked |

**The system has ONE point of failure**: HYBRID_ENTRY on BATL. If either the strategy
or the symbol underperforms, there is no fallback.

---

## 5. Strategy Improvement Opportunities

### Enable catalyst_momentum
1. Confidence scorer appears miscalibrated — producing < 0.30 for valid catalyst signals
2. 62 containment blocks suggest these signals reach deep in the pipeline
3. Relaxing spread threshold to 0.8% would unblock ~35 catalyst signals
4. If even 25% of those are profitable, it diversifies alpha away from pure HYBRID_ENTRY

### Enable premarket_breakout
1. EXTENSION_HARD_VETO blocks most signals — gap stocks without catalyst tag
2. Consider: allow gaps 10-30% through extension if spread < 0.5%
3. HIGH_SPREAD at ignition blocks survivors — same spread threshold issue
4. Premarket strategy needs either spread relaxation or price-adjusted thresholds

### New strategies to consider
- **Momentum continuation**: Enter after first pullback on strong open (not gap)
- **Volume breakout**: Pure volume-driven entry on stocks with rvol > 5x
- These would diversify beyond HYBRID_ENTRY's pressure-based model

---

## 6. Strategy Correlation

| Strategy Pair | Expected Correlation | Notes |
|--------------|---------------------|-------|
| HYBRID_ENTRY vs catalyst_momentum | Low | Different trigger conditions |
| HYBRID_ENTRY vs premarket_breakout | Low | Different time windows |
| catalyst_momentum vs premarket_breakout | Medium | Both target gap/news stocks |

Adding catalyst_momentum or premarket_breakout would **reduce portfolio correlation** and
provide alpha on days when HYBRID_ENTRY's conditions aren't met.

---

## 7. Recommendations

1. **P0**: Investigate confidence scorer calibration — 54 META_GATE blocks at < 0.30 threshold suggests miscalibration
2. **P1**: Relax spread thresholds to enable catalyst_momentum signals (see Containment Study)
3. **P2**: Consider per-strategy spread thresholds (tighter for HYBRID_ENTRY, looser for catalyst_momentum)
4. **P3**: Add momentum continuation strategy for mid-session diversification
5. **P4**: Track strategy PnL separately to identify which strategies justify gate relaxation

---

*Data sources: signal_ledger.jsonl, paper_trades.json, containment events,
last_mile_results.json (READ-ONLY)*
*NO production changes were made. This is research-only analysis.*
