# CLAUDE.md — AI_Bot_Research Project Context

## Project Owner
- **GitHub:** bgmaynard
- **Email:** bgmaynard@hotmail.com
- **Repo:** https://github.com/bgmaynard/AI_Bot_Research

## Infrastructure

### Research Server (this machine)
- **Purpose:** Standalone research & development — completely isolated from live trading
- **Repo Path:** `C:\AI_Bot_Research`
- **Access:** Read-only access to trading data via network mount
- **Git:** v2.53.0 installed, authenticated via browser OAuth
- **Branch:** `main`

### Trading PC (separate machine — BOB1)
- **Purpose:** Live trading only — NO research activity touches this
- **Platform:** IBKR (Interactive Brokers)
- **Software:** Morpheus Trading Platform (localhost:9100)
- **Project Path:** `C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2`
- **Reports:** `\\Bob1\c\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports`
- **Shared Docs:** `\\BOB1\AI_Bot_Data\AI_BOT_DATA\Shared Docs\`
- **Databento Cache:** `Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades`
- **Rule:** Research server NEVER writes to or modifies anything on the trading PC

## Project: Morpheus AI Trading Platform
- Algorithmic momentum stock trading system running on IBKR
- Uses RSI, MFE/MAE profiles, momentum_score, and ignition gates
- **Style:** Warrior Trading methodology, 5 pillars of stock selection
- **Price range:** $1-$20 stocks (low-float, low-price momentum)

### Morpheus Trading Profile
- **Trade data:** 1,944 trades (1,148 active, 796 scratch) across 142 symbols, 18 trading days (2026-01-28 to 2026-02-20)
- **Baseline WR:** 48.3% (active trades), 48.5% (all trades)
- **Entry signal:** `momentum_spike` via ignition funnel pipeline (all trades same signal type)
- **Volatility regime:** 100% HIGH (all trades)
- **41.4% scratch rate:** 796 trades exit with PnL=0 (commission drag)
- **Reports location:** `reports/{date}/trade_ledger.jsonl`
- **Trade ledger key fields:** entry_time, entry_price, atr, entry_signal, pnl, pnl_percent, max_gain_percent, max_drawdown_percent, hold_time_seconds, volatility_regime, exit_reason, secondary_triggers

## Project: MPAI (Microstructure Pressure & Arbitrage Index)

### Purpose
Early warning pressure detection layer that sits UNDERNEATH Morpheus. Goal: detect building pressure in microstructure data BEFORE momentum ignites for earlier, better entries.

### Research Status: 10 PHASES COMPLETE — NO MORPHEUS INTEGRATION SIGNAL FOUND

**After 10 phases of rigorous testing, Databento XNAS.ITCH trade-level microstructure data produced zero actionable trade filters for Morpheus momentum scalping on $1-$20 low-float stocks.**

### Hypotheses Tested and Results

| ID | Hypothesis | Status | Key Result |
|----|-----------|--------|------------|
| HYP-001 | Pressure direction predicts continuation | **FALSIFIED** | Phases 1-5 |
| HYP-002 | DPI alignment improves accuracy | **FALSIFIED** | Phases 6-7 |
| HYP-003 | Inventory fade in high-vol reverts | **CONFIRMED** | 56.1% WR, 2.22 R:R, n=435. Standalone signal, NOT Morpheus integration. |
| HYP-013 | Pressure buildup precedes ignition timing | **FALSIFIED** | p=0.229. Volume acceleration confirmed as precursor (p=0.038) but not timing predictor. |
| HYP-023 | Pressure slope >= 0 filter improves quality | **FALSIFIED** | Phase 9 v2. All thresholds ns. Best p=0.464. |
| HYP-024 | Volume acceleration filter improves quality | **FALSIFIED as filter** | Confirmed precursor (p=0.038) but zero quality filtering power. |
| HYP-025 | Combined gate (slope + vol_accel) | **FALSIFIED** | Phase 9 v2. p=0.888. NEITHER group outperformed BOTH PASS. |
| HYP-023A | PDH/PDL structural zones alter pressure behavior | **FALSIFIED** | Phase 10B.1. WR ns at all 4 ATR thresholds. MAE lower near PDH but MFE also lower = compressed moves, not edge. |

### Key Insight: Why Microstructure Failed for This Asset Class
Morpheus trades $1-$20 low-float momentum stocks that:
- Live in permanently chaotic, HIGH volatility microstructure
- Are retail-driven with erratic order flow
- Gap 5+ ATR from prior-day levels (structural zones irrelevant)
- 78% of symbols are single-day runners
- Pressure_z spikes are ubiquitous and cannot separate winners from losers

### Only Surviving Signal
**HYP-003 (Inventory Fade Reversion):** 56.1% hit rate, 2.22 R:R, n=435. Standalone reversion signal in high-volatility regimes. Could be developed as its own strategy on more liquid instruments.

## ACTIONABLE FINDINGS FROM TRADE LEDGER MINING (2026-02-22)

These are the real discoveries. None require Databento data.

### Finding 1: Price >= $5 Filter (HIGHEST PRIORITY — ENTRY-TIME ACTIONABLE)
- **Price >= $5:** 53.9% WR vs 45.7% for <$5. **p=0.0000**
- Hard stop rate drops 16% -> 6% (2.7x improvement)
- Median MAE drops 0.286% -> 0.047% (6x less drawdown)
- Median MFE jumps 0.012% -> 0.098% (8x more upside)
- **Sub-$1.50 is TOXIC:** 41.6% WR, 37% hard stop rate, -$272 in 18 days from 125 trades

### Finding 2: Hold Time Sweet Spot (EXIT LOGIC INSIGHT)
- 30-300 seconds: 54.3% WR vs 43.4% rest. **p=0.0000**
- 10-30s exits (n=433): 46.2% WR, -$543 total -> exit logic may be too aggressive early
- 2-5 minute window: 58.9% WR, +$503 -> where the real money is
- Not actionable at entry, reveals exit sensitivity issues

### Finding 3: Hard Stop + Timeout = Entire System Deficit
- Hard stops: 148 trades, 0% WR, -$1,911
- Max hold timeouts: 134 trades, 28.4% WR, -$685
- Combined: 25% of active trades, **-$2,596 total** (everything else is profitable)
- 41% scratch rate (793 trades) = additional commission drag

### Finding 4: Exit Category Performance
- DECAY_VELOCITY: n=286, 66.1% WR, +$1,307 (the money maker)
- DECAY_VOLUME: n=39, 87.2% WR, +$203
- DECAY_IMBALANCE: n=26, 76.9% WR, +$280
- TRAIL_STOP: n=151, 45.0% WR, +$463
- HARD_STOP: n=148, 0.0% WR, -$1,911 (biggest drag)
- MAX_HOLD_TIMEOUT: n=134, 28.4% WR, -$685

### Recommended Morpheus Actions
1. **Evaluate minimum price threshold** — sub-$1.50 is destroying the system
2. **Review hard stop sizing** for low-priced stocks (1% stop on $1.20 = one tick)
3. **Analyze 10-30s early exit pattern** — decay detectors may fire too fast
4. **Investigate max hold timeout** trades — can momentum state at 3min predict keep/cut?

## Repo Structure
```
C:\AI_Bot_Research\
├── CLAUDE.md                          # This file
├── README.md
├── setup.ps1
├── tickers.txt
├── docs/Research/
│   ├── MPAI_whitepaper V3.md          # Main whitepaper (v3.1)
│   ├── MPAI_What_Actually_Works.md    # Cross-phase findings summary
│   ├── PHASE9_Results.md              # Phase 9 v2 report
│   ├── PHASE10B_StructuralZones.md    # Phase 10B (invalid - 81% NO_DATA)
│   └── PHASE10B1_StructuralZones_FIXED.md  # Phase 10B.1 (yfinance, falsified)
├── results/
│   ├── phase10b_event_data.csv
│   ├── phase10b_results.json
│   ├── phase10b1_event_data.csv       # Full dataset with zone labels
│   └── phase10b1_results.json
└── mrl/
    ├── __init__.py
    ├── config.py
    ├── download_data.py               # Smart Databento downloader
    ├── main.py
    ├── phase9_experiment.py            # Phase 9 quality filters (v2)
    ├── phase10b_structural_zones.py    # Phase 10B (Databento-only, broken)
    ├── phase10b1_zones_fixed.py        # Phase 10B.1 (yfinance fix, working)
    ├── events/mfe_mae.py
    ├── features/
    │   ├── doi_ttl.py
    │   ├── pressure.py
    │   └── vdi.py
    ├── replay/
    │   └── replay_engine.py            # Core: Databento ticks, pressure profiles
    └── store/writer.py
```

## Project: HRDC (Halt Resumption Directional Confirmation)
- **Purpose:** Halt resumption trading strategy within Morpheus
- **Framework:** Regime-based — Early Morning Momentum, Post-Open Transition, Midday, Power Hour
- **Key Parameters:** 3s evaluation window (gap-up), 0s exit (resume below halt)
- **Decision States:** EXIT_IMMEDIATELY | MOMENTUM_ALLOWED | FLAT_ONLY

## Related Repository
- **ai_project_hub:** https://github.com/bgmaynard/ai_project_hub
- Contains full IBKR_Algo_BOT_V2 codebase (Morpheus)
- Key files: `morpheus_trading_api.py`, `ai/ignition_funnel.py`, `ai/momentum_scorer.py`

## Key Principles
1. **Strict isolation** between research and live trading systems
2. **Data first, always** — no implementation without statistical significance
3. **Capital preservation first** — failures exit immediately
4. **Research is read-only** — never writes to trading infrastructure
5. **Permutation tests + bootstrap CIs** — standard validation throughout

## Session History
- **2025-02-21:** Repo created. Built replay_engine.py for HYP-013. Smart Databento downloader.
- **2026-02-22:** Marathon session. Phase 9 v2 (MFE/MAE fix, all 3 quality filters falsified). Phase 10B (PDH/PDL zones DOA at 81% NO_DATA). Phase 10B.1 (yfinance fix, 100% coverage, HYP-023A falsified). Cross-phase trade ledger mining discovered 3 statistically significant patterns. Whitepaper v3.1.

## NEXT SESSION PRIORITIES
1. **Share findings with ChatGPT co-analyst** — MPAI_What_Actually_Works.md
2. **Decide research direction:**
   - Option A: Price >= $5 filter study -> Morpheus config change proposal
   - Option B: Deep dive exit timing (10-30s early exits, max hold timeouts)
   - Option C: Develop HYP-003 (inventory fade) as standalone strategy
   - Option D: Close MPAI microstructure research, pivot to Morpheus internal optimization
3. **Whitepaper** — formal closing of Databento track, pivot to trade ledger insights
