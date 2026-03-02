# CLAUDE.md - SuperBot Research Lab

This file provides guidance to Claude Code when working with the Research Lab.

## Overview

**SuperBot Research Lab** - Standalone research environment for trading bot analysis.

**Location:** `C:\AI_Bot_Research`
**Port:** 9200
**Access:** `http://localhost:9200/research_lab.html`

## Architecture (Strict Isolation)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION BOTS (NEVER MODIFIED)                  │
├─────────────────────────────────────────────────────────────────────┤
│  Morpheus (IBKR)     │ D:\Trading\IBKR_Algo_BOT_V2                  │
│  Morpheus AI         │ C:\Morpheus\Morpheus_AI                      │
│  Max AI              │ D:\Trading\Max_AI                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ EXPORT (append-only logs)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SHARED EXPORT BRIDGE                              │
│                    \\AI_SHARE\morpheus_exports\                      │
│                    (Production WRITES, Research READS)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ READ-ONLY
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RESEARCH LAB (This System)                        │
│                    C:\AI_Bot_Research                                │
│                    Port 9200                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ WRITE (proposals only)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PROPOSALS                                         │
│                    \\AI_SHARE\superbot_proposals\                    │
│                    (Research WRITES, Supervisor REVIEWS)             │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Sources

| Bot | Path | Access |
|-----|------|--------|
| **Morpheus** (IBKR) | `D:\Trading\IBKR_Algo_BOT_V2` | READ-ONLY via exports |
| **Morpheus AI** | `C:\Morpheus\Morpheus_AI` | READ-ONLY via exports |
| **Max AI** | `D:\Trading\Max_AI` | READ-ONLY via exports |

**CRITICAL: This system NEVER modifies production bot data.**

## Directory Structure

```
C:\AI_Bot_Research\
├── research_server.py      # Main HTTP server (port 9200)
├── config.json             # Configuration
├── CLAUDE.md               # This file
├── ui/
│   ├── research_lab.html   # Visibility panel
│   └── js/
│       └── research_lab.js # Controller logic
├── data/                   # Local research data
│   ├── runs.json           # Shadow replay runs
│   └── proposals.json      # Local proposals
├── proposals/              # Generated improvement proposals
├── logs/                   # Research logs
└── docs/
    └── EXPORT_CONTRACT.md  # Data contract specification
```

## Commands

```cmd
# Start Research Lab
cd C:\AI_Bot_Research
python research_server.py

# With explicit research mode
set SUPERBOT_MODE=RESEARCH_ONLY
python research_server.py
```

## Safety Guardrails

### Execution Fence

The server enforces strict isolation:

1. **Environment Check**: `SUPERBOT_MODE=RESEARCH_ONLY`
2. **Hostname Block**: Cannot run on production servers
3. **Path Block**: Cannot run from production directories
4. **API Block**: `ALLOW_TRADING_APIS=true` causes exit

### What SuperBot CAN Do

- Read exported log files (via shared folder)
- Generate improvement proposals
- Run shadow replays locally
- Perform walk-forward validation
- Display visibility dashboards

### What SuperBot CANNOT Do

- Push config updates to production
- Modify production parameters
- Call trading endpoints
- Call order endpoints
- Register routes in production servers
- Share execution authentication
- Write to production bot directories

## UI Sections

| Section | Purpose |
|---------|---------|
| A. Status Dashboard | Ingestion stats, data hash, git SHA, replay accuracy |
| B. Replay Integrity | Production vs replay comparison (green/yellow/red) |
| C. Shadow Runs Table | Clickable rows with equity curve modal |
| D. Walk-Forward Results | IS/OOS expectancy, degradation warning |
| E. Proposal Queue | View evidence, approve/reject with confirmation |

## API Endpoints (Port 9200)

```bash
# Research status
GET  /api/research/status
GET  /api/research/replay/integrity
GET  /api/research/runs
GET  /api/research/runs/{id}
GET  /api/research/runs/{id}/walkforward

# Proposals
GET  /api/superbot/proposals
GET  /api/research/proposals/{id}/evidence
POST /api/superbot/proposals/{id}/approve
POST /api/superbot/proposals/{id}/reject
```

## Configuration

`config.json`:
```json
{
  "data_sources": {
    "morpheus": {"path": "D:\\Trading\\IBKR_Algo_BOT_V2", "read_only": true},
    "morpheus_ai": {"path": "C:\\Morpheus\\Morpheus_AI", "read_only": true},
    "max_ai": {"path": "D:\\Trading\\Max_AI", "read_only": true}
  },
  "safety": {
    "read_only_mode": true,
    "never_modify_sources": true,
    "no_live_orders": true,
    "no_execution": true
  }
}
```

## Key Principles

1. **Production = Deterministic** - Never touch production
2. **Research = Experimental** - Isolated sandbox only
3. **Pull-Only** - Read from exports, never from source
4. **Write-Only to Proposals** - Output goes to review queue
5. **Human Approval Required** - No auto-promotion ever

## Related Documentation

- `docs/EXPORT_CONTRACT.md` - Data bridge specification
- `README.md` - Quick start guide
- `IBKR_Algo_BOT_V2/.claude/CLAUDE.md` - Production bot reference
