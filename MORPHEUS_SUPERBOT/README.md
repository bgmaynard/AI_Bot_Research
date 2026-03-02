# Morpheus SuperBot Research Lab

**Standalone research environment for trading bot analysis.**

## Important

This system is:
- **READ-ONLY** with respect to production bots
- **ISOLATED** from live trading systems
- Located at `C:\AI_Bot_Research` on Windows

## Directory Structure

```
C:\AI_Bot_Research\
├── research/           # Core research modules
├── ui/                 # Research Lab UI
├── data/               # Ingested data (read from production)
├── logs/               # Research logs
├── proposals/          # Generated improvement proposals
├── research_server.py  # Main server
└── config.json         # Configuration
```

## Data Sources (READ-ONLY - NEVER MODIFIED)

The research system reads data from:

| Bot | Path | Data |
|-----|------|------|
| **Morpheus** (IBKR) | `D:\Trading\IBKR_Algo_BOT_V2` | Trades, signals, regimes |
| **Morpheus AI** | `C:\Morpheus\Morpheus_AI` | Trades, signals, regimes |
| **Max AI** | `D:\Trading\Max_AI` | Trades, signals |

**CRITICAL: The Research Lab NEVER modifies these sources.**
- Read-only file access
- No write operations to bot directories
- No config changes to production bots
- Data is copied locally for analysis
- Pull-only architecture

## Quick Start

```cmd
cd C:\AI_Bot_Research
python research_server.py
```

Access: http://localhost:9200/research_lab.html

## Safety

- No connection to live order endpoints
- No execution capability
- No production config modification
- All proposals require manual supervisor approval
