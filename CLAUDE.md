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

## Setup History
- **2025-02-21:** GitHub repo `AI_Bot_Research` created, Git installed on research server, initial commit with 14 files (mrl package), pushed to main branch successfully.
