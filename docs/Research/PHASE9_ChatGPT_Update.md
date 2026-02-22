# MPAI Research Update — Phase 9 Full Results
## For ChatGPT Continuation Context

**Date:** 2026-02-21
**Researcher:** bgmaynard
**Engine:** Claude AI (Anthropic) on Research Server
**Repository:** https://github.com/bgmaynard/AI_Bot_Research
**Previous Update:** `docs/Research/HYP013_Full_Results_ChatGPT_Update.md`

---

## 1. What Was Tested

**Phase 9 — Pivot: From Pressure Timing to Ignition Precursors + Quality Filters**

After HYP-013 falsified the idea that pressure_z predicts ignition timing, two secondary findings emerged from that test:

1. **Volume acceleration** was statistically significant as a precursor (p=0.038) — volume ramps up before real ignitions more than at random times.
2. **Pressure buildup rate** showed a winner/loser divergence — winners had stable/flat pressure (0.000), losers had fading pressure (-0.003).

Phase 9 asked: **Can these findings be turned into actionable trade quality filters?**

Three hypotheses were tested:
- **HYP-023:** Does filtering for pressure_slope ≥ 0 improve trade quality?
- **HYP-024:** Does filtering for volume_acceleration above a threshold improve trade quality?
- **HYP-025:** Does a combined gate (both filters together) improve trade quality?

---

## 2. Phase 9 Verdict: NO-GO

**All three hypotheses failed.** None of the filters produced a statistically significant improvement in win rate, PnL, or reward:risk ratio. The secondary findings from HYP-013 were real as descriptive statistics but do not function as predictive filters.

---

## 3. Dataset

Same dataset as HYP-013 — no new data collection.

- **Total Morpheus trades:** 1,944 (closed, from trade_ledger.jsonl)
- **Trades with valid pressure profiles:** 420 (21.6% coverage)
- **Symbols:** 142 unique tickers, $1-$20 low-float momentum stocks
- **Trading days:** 18 (2026-01-29 through 2026-02-20)
- **Databento files:** 187 downloaded (XNAS.ITCH trades schema)
- **Winners in profiled set:** 184 (43.8%)
- **Losers in profiled set:** 236 (56.2%)

### Why Only 21.6% Coverage?

1,524 out of 1,944 events couldn't compute valid pressure profiles. Primary cause: low-float momentum stocks have sparse tick data outside of their active spike windows. A 5-second bar aggregation over a 5-minute pre-window requires at least 22 non-empty bars (rolling window warmup + 2). Many events occurred during thin periods where Databento recorded too few trades to fill the bars.

---

## 4. Test Methodology

### Script: `mrl/phase9_experiment.py`

The experiment reuses the pressure profiles from the HYP-013 replay engine (`mrl/replay/replay_engine.py`) and adds three filter experiments with statistical testing.

### Configuration (inherited from replay engine)
```
bar_sec = 5                    # 5-second bars
rolling_window_bars = 20       # 100s rolling window for pressure_z
pressure_z_threshold = 1.5     # Threshold for precursor detection
pre_window_sec = 300           # 5 minutes before entry
post_window_sec = 60           # 1 minute after entry
```

### Per-Event Data Collected
For each of the 420 profiled events, the script computed and stored:

**Trade outcomes:** pnl, pnl_percent, is_winner, max_gain_pct (MFE), max_drawdown_pct (MAE), hold_time_sec, reward_risk (MFE/MAE), exit_category, volatility_regime

**Pressure profile metrics:** peak_pressure_z, mean_pressure_z, pressure_direction, buildup_rate (slope of |pressure_z| over pre-entry bars), first_cross_sec (lead time), bars_above_threshold, pressure_consistency, volume_acceleration (slope of total_volume over pre-entry bars)

**Context:** symbol, entry_time, hour_et, tod_bucket (pre_market/open/mid/close/after_hours), date, rvol, change_pct, spread_pct

Raw data saved to: `results/phase9_event_data.csv`

### Statistical Methods

**For each filter threshold:**
1. Split trades into PASS (meets filter) and REJECT (doesn't)
2. **Permutation test** (n=2,000): Compare win rates between PASS and REJECT groups. Shuffle labels, measure how often random shuffle produces as large a difference as observed.
3. **Bootstrap 95% CI** (n=2,000): Resample PASS group with replacement, compute CI on win rate and reward:risk.

**Stability checks:**
- Per-ticker breakdown for tickers with n ≥ 10 (15 tickers qualified)
- Time-of-day buckets: pre_market (<9:30 ET), open (9:30-10:30), mid (10:30-14:00), close (14:00-16:00)

---

## 5. Experiment 1 — HYP-023: Pressure Slope Filter

**Question:** Does filtering for buildup_rate ≥ threshold improve win rate or PnL?

### Thresholds Tested

| Threshold | N Pass | WR Pass | WR Reject | Δ WR | p-value | Significant? |
|-----------|--------|---------|-----------|------|---------|-------------|
| slope ≥ -0.01 (loose) | 286 | 41.6% | 48.5% | -6.9pp | 0.207 | NO |
| slope ≥ 0 (neutral) | 243 | 41.6% | 46.9% | -5.3pp | 0.320 | NO |
| slope ≥ 0.005 (mild) | 156 | 44.2% | 43.6% | +0.6pp | 0.913 | NO |
| slope ≥ 0.01 (moderate) | 129 | 44.2% | 43.6% | +0.6pp | 1.000 | NO |
| slope ≥ 0.02 (strong) | 91 | 44.0% | 43.8% | +0.2pp | 1.000 | NO |

### Key Finding

**The slope filter is completely non-predictive.** At the "neutral" threshold (≥ 0), the REJECT group actually had a *higher* win rate (46.9%) than the PASS group (41.6%). The winner/loser divergence observed in HYP-013 (winners: 0.000, losers: -0.003) was a descriptive artifact that does not translate to a predictive filter.

The only positive signal was in mean PnL: slope ≥ 0.02 showed +$3.67 mean PnL for PASS vs -$1.01 for REJECT. But with p=1.000 and n=91, this is not statistically meaningful.

### Why the HYP-013 Observation Didn't Translate

The HYP-013 analysis compared *already-known* winners vs losers and measured their pre-entry pressure slope. That's a backward-looking split — it tells you that winners *happened to have* flat pressure, not that flat pressure *predicts* winners. This is a classic descriptive-vs-predictive trap. The distributions overlap too much for slope to function as a filter.

---

## 6. Experiment 2 — HYP-024: Volume Acceleration Threshold

**Question:** Does filtering for volume_acceleration above a quantile threshold improve trade quality?

### Volume Acceleration Distribution
```
mean = 13.62
median = 0.00 (half of all events had zero or negative volume acceleration)
std = 400.62 (extremely high variance)
P50 = 0.00, P60 = 0.00, P70 = 15.18, P80 = 48.00, P90 = 147.54
```

### Thresholds Tested

| Quantile | Threshold | N Pass | WR Pass | WR Reject | Δ WR | p-value | Significant? |
|----------|-----------|--------|---------|-----------|------|---------|-------------|
| P50 | 0.0 | 232 | 43.5% | 44.1% | -0.6pp | 0.922 | NO |
| P60 | 0.0 | 232 | 43.5% | 44.1% | -0.6pp | 0.922 | NO |
| P70 | 15.2 | 126 | 46.8% | 42.5% | +4.3pp | 0.456 | NO |
| P80 | 48.0 | 84 | 47.6% | 42.9% | +4.8pp | 0.475 | NO |
| P90 | 147.5 | 42 | 42.9% | 43.9% | -1.1pp | 1.000 | NO |

### Key Finding

**Volume acceleration does not predict trade quality.** Despite being a statistically significant precursor (p=0.038 in HYP-013, confirming that volume ramps up before real ignitions), it has zero filtering power. Knowing that volume is accelerating tells you "something is about to happen" but NOT "what about to happen will be profitable."

The P70-P80 range showed a slight win rate uplift (~+4-5pp) but with p > 0.45, this is well within noise. At P90, the effect disappears entirely.

### Why HYP-024's Precursor Status Didn't Translate

Volume acceleration is a precursor to *any* Morpheus ignition — winners and losers alike. It's like detecting "the starting gun fired" — it's real, it's measurable, but it doesn't tell you who wins the race. The HYP-013 result (p=0.038 vs controls) proved volume ramps up before ignition, but that's an ignition-detection signal, not a quality signal.

---

## 7. Experiment 3 — HYP-025: Combined Gate

**Question:** Does the intersection of both filters (slope ≥ 0 AND volume_accel ≥ P80) outperform?

### Gate Parameters
- Pressure slope threshold: ≥ 0.0
- Volume acceleration threshold: ≥ 48.0 (P80)

### Results by Group

| Group | N | Win Rate | Mean PnL | Median MFE | Median MAE | Median R:R |
|-------|---|----------|----------|------------|------------|------------|
| ALL (baseline) | 420 | 43.8% | $0.00 | 0.00% | 0.08% | 0.000 |
| BOTH PASS | 70 | 44.3% | -$2.84 | 0.00% | 0.13% | 0.000 |
| SLOPE ONLY | 173 | 40.5% | +$1.65 | 0.00% | 0.05% | 0.000 |
| VOL ONLY | 14 | 64.3% | +$3.30 | 0.16% | 0.00% | 0.750 |
| NEITHER | 163 | 45.4% | -$0.82 | 0.00% | 0.15% | 0.000 |

### Statistical Tests

| Comparison | p (Win Rate) | p (PnL) | Significant? |
|-----------|-------------|---------|-------------|
| BOTH vs NEITHER | 0.885 | 0.274 | NO |
| BOTH vs BASELINE | 1.000 | 0.298 | NO |

CI95 for BOTH PASS win rate: [32.9%, 55.7%] — extremely wide, includes baseline.
CI95 for BOTH PASS R:R: [0.000, 10.000] — meaningless spread.

### Key Finding

**The combined gate performs no better than random.** The BOTH PASS group (44.3% WR) was essentially identical to baseline (43.8%) and marginally worse than NEITHER (45.4%). The combined filter actually had the worst mean PnL of any group (-$2.84).

**VOL ONLY (n=14) showed 64.3% WR** — but with only 14 trades, this is a statistical fluke. The 95% CI would span roughly 35-87%.

### MFE/MAE Data Quality Issue

Median MFE across the board is 0.00%, which means most trades recorded zero maximum favorable excursion. This is almost certainly a data issue in the Morpheus `trade_ledger.jsonl` — the `max_gain_percent` field appears to not track intra-trade maximum price excursion correctly. This broke the reward:risk analysis entirely (R:R = 0.000 for most groups). Any future MFE/MAE analysis needs to reconstruct these values from raw Databento tick data rather than relying on the trade ledger fields.

---

## 8. Stability Checks

### Time-of-Day (Combined Gate: slope ≥ 0 AND vol_accel ≥ P80)

| Bucket | N All | N Pass | WR All | WR Pass | Δ WR |
|--------|-------|--------|--------|---------|------|
| pre_market | 163 | 23 | 41.1% | 26.1% | **-15.0pp** |
| open | 90 | 16 | 48.9% | 56.2% | +7.4pp |
| mid | 116 | 21 | 45.7% | 52.4% | +6.7pp |
| close | 51 | 10 | 39.2% | 50.0% | +10.8pp |

**3 of 4 market-hours buckets showed improvement**, but pre_market collapsed by -15pp. Pre-market is also the largest bucket (163 events, 39% of all data). The market-hours improvement is interesting but sample sizes per bucket (10-21) are far too small for any confidence.

### Per-Ticker (n ≥ 10 tickers only, 15 qualified)

| Improved | Worsened | No Data |
|----------|----------|---------|
| 5 | 6 | 4 |

**Essentially a coin flip.** The filter improved 5/11 tickers with data — no better than random. Notable results:
- SMCL: +29.2pp (83.3% WR with filter, but n=6 in pass group)
- CISS: -14.7pp (26.7% WR with filter vs 41.4% baseline — the biggest ticker was hurt by the filter)
- CONL: +37.5pp (75.0% WR, but n=4 in pass group)

All per-ticker improvements had n < 10 in the pass group — statistically meaningless.

---

## 9. Conclusions

### HYP-023: FALSIFIED
Pressure slope does not predict trade quality. The winner/loser divergence from HYP-013 was descriptive, not predictive. Filtering by slope ≥ 0 actually selected for slightly *worse* win rates.

### HYP-024: NOT SUPPORTED as a quality filter
Volume acceleration is a confirmed ignition precursor (p=0.038 vs controls) but has zero trade quality filtering power. It detects that ignition is coming, not that the trade will be profitable.

### HYP-025: FALSIFIED
Combined gate (slope ≥ 0 AND vol_accel ≥ P80) produced 44.3% WR vs 45.4% for NEITHER group. p=0.885. Completely dead.

### MFE/MAE Data Issue
The `max_gain_percent` field in `trade_ledger.jsonl` reads 0.00% for most trades. This broke all reward:risk calculations. Future MFE/MAE work must reconstruct maximum favorable/adverse excursion from raw Databento tick data during the hold period of each trade.

---

## 10. Updated Hypothesis Registry

| ID | Hypothesis | Status | Notes |
|----|-----------|--------|-------|
| HYP-001 | Raw pressure predicts continuation | **FALSIFIED** | Phase 1-5 |
| HYP-002 | DPI alignment improves accuracy | **FALSIFIED** | Phase 6-7 |
| HYP-003 | FADE in high vol reverts | **CONFIRMED** | Phase 8. 56.1%, 2.22 R:R, n=435 |
| HYP-013 | Pressure buildup precedes Morpheus ignition | **FALSIFIED** | n=420 real vs 912 control. p=0.229 |
| HYP-018 | Trades with pressure precursor have better MFE/MAE | **INCONCLUSIVE** | MFE data quality issue. Cannot evaluate. |
| HYP-023 | Pressure slope ≥ 0 filter improves trade quality | **FALSIFIED** | Phase 9. All thresholds ns. Best p=0.207 (wrong direction). |
| HYP-024 | Volume acceleration threshold improves quality | **FALSIFIED as filter** | Phase 9. All thresholds ns. Best p=0.456. Confirmed as precursor (p=0.038) but not as quality filter. |
| HYP-025 | Combined gate improves outcomes | **FALSIFIED** | Phase 9. n=70 vs 163. p=0.885. NEITHER group was better. |

---

## 11. Where This Leaves MPAI

### What Has Been Tested and Failed (Phases 1-9)
1. ❌ Raw pressure direction predicts price continuation
2. ❌ DPI (directional pressure index) alignment improves accuracy
3. ❌ Pressure_z buildup precedes Morpheus ignition timing
4. ❌ Pressure slope filters improve trade quality
5. ❌ Volume acceleration filters improve trade quality
6. ❌ Combined pressure slope + volume acceleration gate

### What Has Worked
1. ✅ **HYP-003:** Inventory FADE in high volatility reverts at 56.1% hit rate, 2.22 R:R, n=435. This is the only validated finding in the entire MPAI research program.
2. ✅ **Volume acceleration as ignition precursor** (p=0.038). Real but not useful as a quality filter.

### The Honest Assessment
The MPAI concept — using Databento microstructure data (buy/sell classified tick trades) to improve Morpheus momentum scalping entries — has been tested across 9 phases and has not produced a single actionable trade filter. The only confirmed finding (HYP-003, inventory fade) is a standalone reversion signal unrelated to the core MPAI→Morpheus integration concept.

### Possible Directions (Not Recommended Without Strong Prior)
- **Fix MFE/MAE data:** Reconstruct true intra-trade MFE/MAE from Databento ticks during hold period. The trade ledger's `max_gain_percent` field is unreliable. This might unlock HYP-018 (do pressure precursors correlate with better MFE?).
- **Level 2 / order book data:** XNAS.ITCH trades schema only has trade executions with side. Full order book (bid/ask depth, queue position, order flow imbalance) is a fundamentally different signal than what we've been testing.
- **Different asset class:** The $1-$20 low-float momentum universe may simply be too noisy and retail-driven for microstructure signals to matter. Institutional flow detection might work on mid/large-cap liquid names.
- **Abandon MPAI for Morpheus integration.** Focus research resources on improving Morpheus's existing gates (ignition funnel, central gating, extension FSM) using signals already available in the trading system.

---

## 12. Code & Files Reference

### Files Created/Modified This Session

```
C:\AI_Bot_Research\
├── docs/Research/
│   ├── MPAI_whitepaper V3.md              # Updated to v3.1
│   ├── HYP013_Full_Results_ChatGPT_Update.md  # Previous update
│   └── PHASE9_IgnitionPrecursors_Results.md   # Phase 9 auto-generated report
├── mrl/
│   ├── phase9_experiment.py               # Phase 9 experiment script (906 lines)
│   ├── download_data.py                   # Smart Databento downloader
│   └── replay/
│       └── replay_engine.py               # HYP-013 replay engine
└── results/
    ├── phase9_event_data.csv              # Raw per-event data (420 rows)
    ├── phase9_results.json                # Full Phase 9 JSON output
    └── hyp013_replay_results.json         # HYP-013 JSON output
```

### Shared Docs (copied to trading PC)
```
\\BOB1\AI_Bot_Data\AI_BOT_DATA\Shared Docs\
├── MPAI_whitepaper V3.md
└── PHASE9_IgnitionPrecursors_Results.md
```

### Data Paths
- **Morpheus reports:** `\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports\`
- **Databento cache:** `Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades\` (187 files)
- **Research output:** `C:\AI_Bot_Research\results\`

### Whitepaper v3.1 Changes
- Version 3.0 → 3.1
- HYP-013: UNTESTED → FALSIFIED
- HYP-018: UNTESTED → PARTIALLY SUPPORTED (then inconclusive due to MFE data issue)
- Added HYP-023, HYP-024, HYP-025 to registry (all FALSIFIED)
- Added Section 13: Phase 9 pivot description and experiment design
- Updated validation gates: V2.5 FALSIFIED, V2.5R added and FAILED
- Updated priority research path to reflect Phase 9 completion

---

## 13. Session Summary (2026-02-21)

### Infrastructure Built
1. Git v2.53.0 installed on research server
2. GitHub repo `AI_Bot_Research` created and connected
3. Smart Databento downloader that reads Morpheus trade ledger
4. 187 Databento files downloaded (all Morpheus-traded symbols)
5. HYP-013 replay engine (877 lines)
6. Phase 9 experiment framework (906 lines)

### Tests Executed
1. HYP-013 full test: 420 real events vs 912 controls → **FALSIFIED** (pressure doesn't predict ignition)
2. HYP-023 pressure slope sweep (5 thresholds) → **FALSIFIED**
3. HYP-024 volume acceleration sweep (5 quantiles) → **FALSIFIED as filter**
4. HYP-025 combined gate → **FALSIFIED** (p=0.885)
5. Stability checks: per-ticker (5/11 improved = coin flip) and time-of-day (3/4 market-hours improved but pre-market collapsed, all n too small)

### Key Takeaway
The Databento XNAS.ITCH trade-level microstructure data does not contain actionable signals for improving Morpheus momentum scalping entries on $1-$20 low-float stocks. Nine phases of testing have produced one standalone finding (inventory fade reversion) and zero Morpheus integration signals.

---

*End of Update — Generated by Claude AI on 2026-02-21*
*Research Server: AI_Bot_Research | Trading PC: IBKR_Algo_BOT_V2 (Bob1)*
