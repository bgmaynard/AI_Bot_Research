# Morpheus SuperBot Research Playbook

## When to Run What, and Why

---

## THE BIG PICTURE

```
MARKET HOURS                    AFTER CLOSE                    AS-NEEDED
─────────────                   ───────────                    ─────────
Watchlist Modules               Nightly Pipeline               Deep-Dive Studies
(classify, track, vet)          (automated, 4 modules)         (manual, targeted)
       │                               │                              │
       ▼                               ▼                              ▼
 "What to trade today"          "How did we do?"              "How to improve"
```

There are **three layers** of analysis, each with a different cadence and purpose.

---

## LAYER 1: REAL-TIME (During Market Hours)

**Purpose:** Decide what's worth trading right now.

```
Scanner discovers stock
        │
        ▼
┌─────────────────────────┐
│  stock_classifier.py    │  Score 0-100, assign A/B/C tier
│  (watchlist/)           │  Catalyst, RVol, Gap%, Spread, Confidence, Momentum
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  daily_tracker.py       │  Track price from discovery → EOD
│  (watchlist/)           │  Outcome: winner / active / faded / loser / noise
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  vetted_list.py         │  Curated top-10 tradeable list
│  (watchlist/)           │  A-class auto-adds, B-class manual promote, C rejected
└─────────────────────────┘
```

**When:** Runs continuously via research_server.py. Ingests signals on server startup.

**API:**
- `GET /api/watchlist/classified` — All classified symbols with tiers
- `GET /api/watchlist/vetted` — Current vetted list
- `GET /api/watchlist/tracker` — Tracked symbols with price updates
- `GET /api/watchlist/report` — EOD performance report
- `POST /api/watchlist/add` — Manually promote a symbol

---

## LAYER 2: NIGHTLY AUTOMATION (After Market Close)

**Purpose:** Automated validation — did the filters work? What's the optimal config?

**Trigger:** `python -m ai.research.nightly_pipeline` or `POST /api/research/nightly`

```
Auto-detect latest trading day from quote cache
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  MODULE 1: Shadow Replay Grid Optimization                   │
│  3,840 configs: hold × trail × spread × pullback × cap       │
│  Output: reports/research/replay/{date}/                     │
│  Answers: "What exit parameters maximize PF today?"          │
├──────────────────────────────────────────────────────────────┤
│  MODULE 2: Alpha Heatmap                                     │
│  Vol/Spread/OFI matrices + per-symbol + per-session           │
│  Output: reports/research/alpha_heatmap/{date}/              │
│  Answers: "Which market conditions produced alpha?"           │
├──────────────────────────────────────────────────────────────┤
│  MODULE 3: Regime Filter Validation                          │
│  Baseline vs filtered (vol≥0.3%, spread≤0.6%, OFI≥-0.2)     │
│  Output: reports/research/regime_paper_validation/{date}/    │
│  Answers: "Does the filter improve results?"                  │
├──────────────────────────────────────────────────────────────┤
│  MODULE 4: Research Dashboard                                │
│  Unified summary of all modules                              │
│  Output: reports/research/daily_summary.md                   │
│  Answers: "What's the one-page view?"                        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
Rolling scorecard tracks deployment readiness (need 5+ days)
```

**Key files:**
- `ai/research/nightly_pipeline.py` — Orchestrator
- `reports/research/daily_summary.md` — Dashboard
- `reports/research/regime_paper_validation/regime_filter_daily_scorecard.md` — Rolling tracker
- `reports/research/pipeline_results.json` — Machine-readable results

---

## LAYER 3: DEEP-DIVE STUDIES (Manual, As-Needed)

This is where the 33 research scripts live. They form a **logical pipeline** — each phase
produces JSON artifacts that downstream phases consume. You don't run all of them every day.
**Run them when you need to answer a specific question.**

---

### PHASE 1: Signal Quality — "Is the ignition signal reliable?"

**Run when:** You suspect signal quality is degrading, or after changing ignition parameters.

```
START HERE
    │
    ▼
ignition_detector.py              "Does ignition fire at the right time?"
    │ outputs: edge_preservation_v2_{date}.json
    ▼
ignition_detector_v2.py           "Two-tier detection: HARD (3/5) vs SOFT (2/4)"
    │ outputs: ignition_events_v2_{date}.json
    ▼
ignition_tier_sweep.py            "What thresholds give PF≥1.15 at coverage≥40%?"
    │ outputs: ignition_tier_sweep_{date}.json
    ▼
accel_coupling_study_v2.py        "Does coupling ignition + acceleration help?"
    │ outputs: ignition_coverage_report.md
    ▼
soft_salvage_study.py             "Can we rescue SOFT ignitions with extra filters?"
    │ outputs: SOFT+ candidate rules
    ▼
multiday_ignition_validation.py   "Does this hold across multiple days?"
    │ outputs: shadow_deploy/ pack (production-ready config)
    ▼
DECISION: If PF≥1.15 and stable across days → deploy to shadow pack
```

**Key question answered:** "What % of real moves do we catch, and how many false signals do we fire?"

---

### PHASE 2: Entry Timing — "When exactly should we enter?"

**Run when:** You see high MAE (adverse excursion) suggesting entries are too early/late.

```
extension_continuation.py         "Does momentum continue after ignition?"
    │ outputs: momentum decay curve at 7 time windows
    ▼
microstructure_discovery.py       "Which microstructure features predict fills?"
    │ outputs: microstructure_features_{date}.json, microstructure_rank_{date}.json
    ▼
microstructure_calibration.py     "Learn feature weights from historical outcomes"
    │ outputs: microstructure_rank_learned_{date}.json
    ▼
entry_offset_optimizer.py         "Enter at signal price, or offset by -0.5%?"
    │ outputs: entry_offset_optimizer_{date}.json
    │ KEY FINDING: entering 0.50% lower → WR from 45.9% to 74.3%
    ▼
DECISION: If offset improves WR without killing coverage → update entry logic
```

---

### PHASE 3: Pressure Analysis — "Is the L2 pressure signal real?"

**Run when:** You have new L2 data, or suspect pressure events are generating false signals.

```
pressure_timing_study.py          "How far apart are pressure → ignition → entry?"
    │
    ▼
pressure_quality_study.py         "Which pressure features predict highest PF?"
    │ outputs: 7-dimension feature ranking, tier classification
    ▼
pressure_trap_study.py            "Which pressure events are traps (no ignition follows)?"
    │ outputs: trap probability model
    ▼
liquidity_vacuum_study.py         "Does order book exhaustion predict moves?"
    │ outputs: vacuum event correlation to fills
    ▼
pressure_entry_simulation.py      "Compare: signal entry vs ignition entry vs pressure entry"
    │ outputs: per-model PnL comparison
    ▼
DECISION: If trap filter reduces FP >30% without killing coverage → deploy
```

---

### PHASE 4: Strategy Assembly — "How do all the signals combine?"

**Run when:** You've updated any upstream component and need to re-validate the combined strategy.

```
┌──────────────────────────────────────────────────────────┐
│  hybrid_entry_study.py    ★ KEY HUB MODULE ★             │
│                                                          │
│  Combines: pressure + trap<0.5 + ignition + vacuum       │
│  Compares hybrid vs each individual signal model          │
│  OUTPUT: hybrid_entry_study_{date}.json                  │
│                                                          │
│  THIS FILE IS CONSUMED BY ALMOST EVERYTHING DOWNSTREAM   │
└───────────────────────────┬──────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
    position_mgmt    market_regime    walk_forward
    _study.py        _sensitivity.py  _validation.py
    "6 exit models"  "Performance     "5-fold OOS test
     compared"        by regime"       IS vs OOS exp."
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                   VALIDATED STRATEGY
                   (or back to drawing board)
```

**Decision tree after walk-forward:**
- Degradation <20%: Strong signal, proceed to gate optimization
- Degradation 20-30%: Marginal, collect more data before deploying
- Degradation >30%: Overfitting detected, revisit assumptions

---

### PHASE 5: Gate Optimization — "How to filter out the bad trades?"

**Run when:** Walk-forward passes, and you want to maximize PF by removing losing conditions.

**These 5 modules are PARALLELIZABLE — run them all at once.**

```
hybrid_entry_study.json (from Phase 4)
    │
    ├───► symbol_edge_gate.py           "Which symbols have negative edge?"
    │     Impact: +25% PnL by filtering 6 bad symbols
    │
    ├───► time_of_day_gate.py           "Which 30-min windows are toxic?"
    │     Impact: Remove ~71 trades in 8 losing windows
    │
    ├───► softplus_ignition_coupling.py "Can SOFT+ ignitions reduce 73% miss rate?"
    │     Impact: PF 6.379 at 40.5% coverage
    │
    ├───► entry_offset_optimizer.py     "Best price offset per symbol?"
    │     Impact: +0.56% avg return
    │
    └───► adaptive_exit_study.py        "Regime-adaptive trailing stop?"
          Impact: +0.338 PF improvement
          │
          ▼
    daily_research_cycle_v2.py          "Run all 5 gates combined"
    Outputs: Combined strategy performance with all gates active
```

---

### PHASE 6: Validation & Cross-Day — "Is this real?"

**Run when:** You have multiple days of data and want to confirm stability.

```
alpha_heatmap_study.py            "Single-day: which conditions produce alpha?"
    │ outputs: reports/research/alpha_heatmap/
    ▼
multiday_alpha_validation.py      "Does this hold across 11 symbols, 2321 trades?"
    │ outputs: Combined filter PF=1.23 vs baseline PF=0.63
    ▼
regime_paper_validation.py        "Side-by-side: does the filter improve live results?"
    │ outputs: reports/research/regime_paper_validation/
    ▼
grid_replay_2026_03_03.py         "Optimal exit params via 3,840-config grid search"
    │ outputs: Best config table + sensitivity analysis
    ▼
DEPLOYMENT DECISION:
  - Scorecard shows 5+ days where filter helps → READY
  - <5 days or inconsistent → KEEP PAPER TRADING
```

---

### PHASE 7: Containment & Spread — "Are we blocking good trades?"

**Run when:** You suspect the containment/spread gates are too aggressive.

```
containment_v2.py                 "Single-pass containment model (replaces v1 3-stage)"
    │
    ▼
analysis/containment_fp_study.py  "Which containment vetoes blocked profitable trades?"
    │
    ▼
analysis/containment_continuation_study.py  "Are we blocking valid breakouts?"
    │
    ▼
analysis/spread_gate_relaxation_study.py    "What if spread threshold goes 0.6% → 0.8%?"
    │
    ▼
analysis/spread_bucket_performance.py       "Win rate by spread bucket (tight/med/wide)"
    │
    ▼
analysis/spread_gate_expectancy_sim.py      "Expected PnL impact of threshold change"
    │
    ▼
DECISION: If relaxing spread gate recovers >20% blocked winners
          with <10% increase in losers → update threshold
```

---

## ANALYSIS SCRIPTS (Standalone Audits)

These are one-off diagnostic scripts. Run them for specific investigations.

| Script | Question It Answers | When to Run |
|--------|---------------------|-------------|
| `analysis/max_ai_scanner_audit.py` | "Is MAX_AI finding good stocks?" | After scanner changes |
| `analysis/last_mile_analysis.py` | "Why did risk-approved signals never execute?" | After 0-trade days |
| `analysis/containment_fp_study.py` | "How many good trades does containment block?" | After high containment block rate |
| `analysis/spread_gate_*.py` (3 files) | "Is spread gate too strict?" | After spread-blocked signals spike |

---

## DATA FLOW MAP

```
PRODUCTION BOTS (read-only pull)
│
├── engine/cache/quotes/{SYMBOL}_quotes.json     ← Quote ticks
├── engine/cache/trades/trades_{date}.json        ← Shadow trades
├── engine/output/live_signals.json               ← All signals for the day
├── engine/output/paper_trades.json               ← Simulated trades
├── engine/output/pressure_events_{date}.json     ← L2 pressure events
├── data_cache/morpheus_reports/{date}/
│   ├── signal_ledger.jsonl                       ← Per-signal decision log
│   └── gating_blocks.jsonl                       ← Block reasons per signal
│
▼ CONSUMED BY ▼
│
├── WATCHLIST (real-time)
│   └── signal_ledger.jsonl → classifier → tracker → vetted list
│
├── NIGHTLY PIPELINE (automated)
│   └── quote cache + live_signals → 4 modules → date-stamped reports
│
├── PHASE 1-3 STUDIES (manual)
│   └── quote cache + signals → ignition/pressure/microstructure analysis
│
├── PHASE 4 (manual)
│   └── ★ hybrid_entry_study.json ★ (central hub — consumed by everything below)
│
├── PHASE 5 GATES (manual, parallelizable)
│   └── hybrid_entry + other artifacts → 5 gate optimizations
│
└── PHASE 6-7 VALIDATION (manual)
    └── multi-day data → stability checks → deployment decision
```

### Key Intermediate Artifacts

| Artifact | Producer | Consumers |
|----------|----------|-----------|
| `edge_preservation_v2_{date}.json` | ignition_detector | symbol_edge_gate, ignition_detector_v2 |
| `ignition_events_v2_{date}.json` | ignition_detector_v2 | accel_coupling_v2, softplus_ignition, multiday_validation |
| `hybrid_entry_study_{date}.json` | hybrid_entry_study | **9 downstream studies** (gates, position mgmt, regime sensitivity, walk-forward) |
| `walk_forward_validation.json` | walk_forward_validation | time_of_day_gate |
| `microstructure_features_{date}.json` | microstructure_discovery | symbol_edge_gate, microstructure_calibration |
| `market_regime_sensitivity_{date}.json` | market_regime_sensitivity | adaptive_exit_study |

---

## DECISION TREE: "What Should I Run?"

```
START: What's the problem?
│
├── "No trades executed today"
│   └── Run: analysis/last_mile_analysis.py
│       └── Then: containment_fp_study.py (if containment blocked them)
│       └── Then: spread_gate_relaxation_study.py (if spread blocked them)
│
├── "Win rate is dropping"
│   ├── Check: nightly pipeline dashboard first
│   ├── Run: market_regime_sensitivity.py (is the regime unfavorable?)
│   ├── Run: time_of_day_gate.py (are we trading in bad windows?)
│   └── Run: symbol_edge_gate.py (are we trading bad symbols?)
│
├── "Signals fire but price doesn't move"
│   ├── Run: ignition_detector_v2.py (is ignition real?)
│   ├── Run: pressure_trap_study.py (are we entering on traps?)
│   └── Run: extension_continuation.py (does momentum continue?)
│
├── "Good entries but bad exits"
│   ├── Run: adaptive_exit_study.py (test different exit params)
│   ├── Run: position_management_study.py (compare 6 exit models)
│   └── Run: grid_replay (nightly) (optimal trail/hold/cap)
│
├── "Want to add a new symbol"
│   ├── Run: watchlist/stock_classifier.py (get tier)
│   ├── Run: microstructure_discovery.py (check microstructure quality)
│   └── Run: symbol_edge_gate.py (does it have positive edge?)
│
├── "Want to deploy filter to production"
│   ├── Check: regime_filter_daily_scorecard.md (need 5+ days)
│   ├── Run: walk_forward_validation.py (OOS degradation <30%?)
│   └── Run: multiday_alpha_validation.py (stable across symbols?)
│
└── "Routine nightly check"
    └── Run: python -m ai.research.nightly_pipeline
        └── Read: reports/research/daily_summary.md
```

---

## DAILY RESEARCH WORKFLOW (Recommended)

### Morning (Pre-Market)
1. Check `daily_summary.md` from last night's pipeline
2. Review scorecard for regime filter deployment readiness
3. Research server auto-ingests new signals → watchlist classifies

### During Market
4. Monitor `GET /api/watchlist/vetted` for tradeable symbols
5. Track `GET /api/watchlist/tracker` for discovery-to-EOD performance

### After Close
6. Trigger nightly pipeline: `python -m ai.research.nightly_pipeline`
7. Review dashboard: `reports/research/daily_summary.md`
8. If anomalies detected → run targeted deep-dive from decision tree above

### Weekly
9. Run `daily_research_cycle_v2.py` for comprehensive gate analysis
10. Review rolling scorecard for 5-day deployment readiness
11. If ready → `walk_forward_validation.py` for final OOS check

---

## ORCHESTRATOR COMPARISON

| Script | Scope | When to Use |
|--------|-------|-------------|
| `nightly_pipeline.py` | 4 modules (replay, heatmap, regime, dashboard) | **Every night** — automated validation |
| `daily_research_cycle.py` | Ignition validation + shadow deploy pack | When updating ignition config |
| `daily_research_cycle_v2.py` | Phase 1 + 5 gate optimizations + combined sim | Weekly or after strategy changes |
| `engine/superbot.py` | Engine-level replay + parameter tuning + walk-forward | When proposing config changes to production |

---

## FILE LOCATIONS CHEAT SHEET

```
MORPHEUS_SUPERBOT/
├── research_server.py              Server (port 9200, all APIs)
├── config.json                     Bot paths, safety config
├── RESEARCH_PLAYBOOK.md            ← YOU ARE HERE
│
├── watchlist/                      Real-time classification
│   ├── stock_classifier.py         A/B/C tier scoring
│   ├── daily_tracker.py            Discovery → EOD tracking
│   └── vetted_list.py              Curated top-10
│
├── ai/research/                    33 research studies
│   ├── nightly_pipeline.py         ★ Nightly automation (run daily)
│   ├── daily_research_cycle_v2.py  ★ Full gate optimization (run weekly)
│   ├── hybrid_entry_study.py       ★ Central hub (produces key artifact)
│   ├── walk_forward_validation.py  ★ Final OOS check (before deployment)
│   └── [29 other studies]          Deep-dive as-needed
│
├── analysis/                       7 standalone audits
│   ├── max_ai_scanner_audit.py     Scanner quality check
│   ├── containment_fp_study.py     Containment gate audit
│   └── spread_gate_*.py            Spread threshold analysis
│
├── engine/                         Core replay + tuning engine
│   ├── superbot.py                 Main orchestrator
│   ├── cache/quotes/               Quote tick data (per-symbol)
│   └── output/                     Signals, trades, study results
│
└── reports/research/               All generated reports
    ├── daily_summary.md            ★ Start here every morning
    ├── pipeline_results.json       Machine-readable pipeline output
    ├── replay/{date}/              Grid optimization results
    ├── alpha_heatmap/{date}/       Performance matrices
    └── regime_paper_validation/    Filter validation + scorecard
```

---

*All analysis is READ-ONLY. No production changes are ever made by research scripts.*
*Last updated: 2026-03-06*
