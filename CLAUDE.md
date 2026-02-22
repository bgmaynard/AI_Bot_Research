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

### Trading PC (separate machine)
- **Purpose:** Live trading only — NO research activity touches this
- **Platform:** IBKR (Interactive Brokers)
- **Software:** Morpheus Trading Platform (localhost:9100)
- **Project Path:** `C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2`
- **Rule:** Research server NEVER writes to or modifies anything on the trading PC

## Project: Morpheus AI Trading Platform
- Algorithmic momentum stock trading system running on IBKR
- Uses RSI, MFE/MAE profiles, momentum_score, and ignition gates

## Project: MPAI (Microstructure Pressure & Arbitrage Index)
- **Purpose:** Early warning pressure detection layer that sits UNDERNEATH Morpheus
- **Problem it solves:** By the time Morpheus detects momentum ignition and fires an entry signal, some of the profitable move has already occurred
- **Goal:** Detect building pressure in market microstructure data BEFORE price moves become visible — millisecond-level edge for earlier entries
- **Inspiration:** Don Kaufman's TheoTrade "Ghost Prints" methodology (institutional pressure detection between brokers and market makers), adapted for equity momentum trading
- **Components under investigation:** 18 pressure detection components including aggressor flow analysis, dark pool divergence detection, and others
- **Validation approach:** Correlate with existing Morpheus indicators, strict hypothesis testing protocols

## Project: HRDC (Halt Resumption Directional Confirmation)
- **Purpose:** Halt resumption trading strategy within Morpheus
- **Framework:** Regime-based — Early Morning Momentum, Post-Open Transition, Midday, Power Hour
- **Key Parameters:**
  - 3-second evaluation window for gap-up scenarios
  - 0-second (immediate) exit for any resume below halt price
  - Philosophy: "Failures get 0 seconds, winners get 3 seconds to prove themselves"
- **Decision States:** EXIT_IMMEDIATELY | MOMENTUM_ALLOWED | FLAT_ONLY

## Repo Structure (as of initial commit)
```
C:\AI_Bot_Research\
├── .gitignore
├── README.md
├── setup.ps1
├── tickers.txt
└── mrl/
    ├── __init__.py
    ├── config.py
    ├── download_data.py
    ├── main.py
    ├── events/
    │   └── mfe_mae.py
    ├── features/
    │   ├── doi_ttl.py
    │   ├── pressure.py
    │   └── vdi.py
    ├── replay/
    │   └── replay_engine.py
    └── store/
        └── writer.py
```

## Key Principles
1. **Strict isolation** between research and live trading systems
2. **Validate and correlate data BEFORE implementing any trading logic**
3. **Capital preservation first** — failures exit immediately
4. **Research is read-only** — never writes to trading infrastructure

## Morpheus Trading Profile
- **Style:** Warrior Trading methodology, 5 pillars of stock selection
- **Price range:** $1-$20 stocks (low-float, low-price momentum)
- **Trade data:** 1,914 trades across 132 unique symbols, 17 trading days
- **Top symbols:** CISS (411), ANL (81), OCG (79), ALXO (68), SMCL (68), PLRZ (67)
- **Entry signal:** `momentum_spike` via ignition funnel pipeline
- **Reports location:** `reports/{date}/trade_ledger.jsonl` (IBKR_Algo_BOT_V2)
- **Trade ledger fields:** entry_time, entry_price, entry_signal, pnl, max_gain_percent, max_drawdown_percent, entry_momentum_score, entry_momentum_state, volatility_regime, secondary_triggers (spread, rvol, change%, float, volume)

## Related Repository
- **ai_project_hub:** https://github.com/bgmaynard/ai_project_hub
- Contains full IBKR_Algo_BOT_V2 codebase (Morpheus)
- Key files: `morpheus_trading_api.py`, `ai/ignition_funnel.py`, `ai/momentum_scorer.py`
- Reports: `store/code/IBKR_Algo_BOT_V2/reports/`

## Current Research Status
- **Phase 8:** COMPLETE — inventory fade validated (HYP-003: 56.1%, 2.22 R:R, n=435)
- **Next gate:** V2.5 — Prove pressure precedes Morpheus ignition (HYP-013)
- **Blocker resolved:** Databento data needs to match Morpheus-traded symbols (not AAPL/TSLA)
- **Data gap:** Need to download Databento XNAS.ITCH trades for 132 momentum symbols across 174 symbol-date pairs

## Setup History
- **2025-02-21:** GitHub repo `AI_Bot_Research` created, Git installed on research server, initial commit with 14 files (mrl package), pushed to main branch successfully.
- **2025-02-21:** Built replay_engine.py for HYP-013 (pressure precedes ignition?). Rebuilt download_data.py as smart downloader that reads trade_ledger.jsonl to download only Morpheus-traded symbols from Databento. Discovered data mismatch: Morpheus trades $1-$20 momentum stocks, original Databento download had AAPL/TSLA/NVDA.
