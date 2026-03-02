# Export Contract: Production → Research Data Bridge

## Overview

This document defines the one-way trust boundary between production trading bots
and the SuperBot Research Lab.

**Direction:** Production → Research (PULL-ONLY)
**Access:** READ-ONLY for Research Lab
**Modification:** NEVER - source files are immutable to Research

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ENVIRONMENT                               │
│  ┌─────────────────┐     ┌─────────────────┐                            │
│  │  Morpheus AI    │     │     Max AI      │                            │
│  │ (IBKR Trading)  │     │   (Trading)     │                            │
│  └────────┬────────┘     └────────┬────────┘                            │
│           │                       │                                      │
│           └───────────┬───────────┘                                      │
│                       │                                                  │
│                       ▼ EXPORT (append-only)                             │
│           ┌───────────────────────────┐                                  │
│           │   \\AI_SHARE\exports\     │                                  │
│           │   (Production writes)     │                                  │
│           └───────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
                        │
                        │ READ-ONLY
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     RESEARCH ENVIRONMENT                                 │
│           ┌───────────────────────────┐                                  │
│           │  SuperBot Research Lab    │                                  │
│           │  (C:\AI_Bot_Research)     │                                  │
│           └───────────┬───────────────┘                                  │
│                       │                                                  │
│                       ▼ WRITE (proposals only)                           │
│           ┌───────────────────────────┐                                  │
│           │ \\AI_SHARE\proposals\     │                                  │
│           │ (Research writes)         │                                  │
│           └───────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Export Directory Structure

Production bots export to: `\\AI_SHARE\morpheus_exports\`

```
morpheus_exports/
├── morpheus/                    # Morpheus (IBKR Bot)
│   └── YYYY-MM-DD/
│       ├── signals.jsonl        # Signal events (append-only)
│       ├── trades.jsonl         # Trade executions (append-only)
│       ├── regimes.jsonl        # Regime classifications
│       ├── vetos.jsonl          # Veto events
│       ├── pnl.json             # Daily PnL summary
│       └── manifest.json        # Export metadata
│
├── morpheus_ai/                 # Morpheus AI (C:\Morpheus\Morpheus_AI)
│   └── YYYY-MM-DD/
│       ├── signals.jsonl
│       ├── trades.jsonl
│       ├── regimes.jsonl
│       ├── pnl.json
│       └── manifest.json
│
├── max_ai/                      # Max AI
│   └── YYYY-MM-DD/
│       ├── signals.jsonl
│       ├── trades.jsonl
│       ├── pnl.json
│       └── manifest.json
│
└── latest/                      # Symlinks to most recent
    ├── morpheus -> ../morpheus/2026-03-01
    ├── morpheus_ai -> ../morpheus_ai/2026-03-01
    └── max_ai -> ../max_ai/2026-03-01
```

## File Schemas

### signals.jsonl

One JSON object per line (append-only):

```json
{
  "timestamp": "2026-03-01T09:30:15.123Z",
  "signal_id": "sig_20260301_093015_AAPL",
  "symbol": "AAPL",
  "signal_type": "MOMENTUM_SPIKE",
  "direction": "LONG",
  "strength": 0.85,
  "source": "morpheus_ai",
  "metadata": {
    "spike_percent": 5.2,
    "volume_surge": 3.8,
    "regime": "TRENDING_UP"
  }
}
```

### trades.jsonl

One JSON object per line (append-only):

```json
{
  "timestamp": "2026-03-01T09:30:45.456Z",
  "trade_id": "trade_20260301_093045_AAPL",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100,
  "entry_price": 185.50,
  "exit_price": 191.20,
  "exit_timestamp": "2026-03-01T09:33:12.789Z",
  "pnl": 570.00,
  "pnl_percent": 3.07,
  "hold_seconds": 147,
  "exit_reason": "TRAILING_STOP",
  "signal_id": "sig_20260301_093015_AAPL",
  "source": "morpheus_ai"
}
```

### regimes.jsonl

```json
{
  "timestamp": "2026-03-01T09:30:00.000Z",
  "regime": "TRENDING_UP",
  "confidence": 0.82,
  "market_conditions": {
    "vix": 18.5,
    "spy_change": 0.45,
    "volume_ratio": 1.2
  }
}
```

### vetos.jsonl

```json
{
  "timestamp": "2026-03-01T09:31:00.000Z",
  "veto_id": "veto_20260301_093100",
  "signal_id": "sig_20260301_093015_TSLA",
  "reason": "CIRCUIT_BREAKER",
  "details": "Daily loss limit reached"
}
```

### pnl.json

Daily summary (single object per day):

```json
{
  "date": "2026-03-01",
  "source": "morpheus_ai",
  "total_pnl": 1234.56,
  "trade_count": 15,
  "win_count": 9,
  "loss_count": 6,
  "win_rate": 0.60,
  "largest_win": 450.00,
  "largest_loss": -180.00,
  "max_drawdown": -320.00
}
```

### manifest.json

Export metadata (required for validation):

```json
{
  "export_timestamp": "2026-03-01T16:00:00.000Z",
  "source": "morpheus_ai",
  "date": "2026-03-01",
  "git_sha": "abc123def456",
  "config_hash": "cfg_a3f8c2d1",
  "files": {
    "signals.jsonl": {
      "record_count": 247,
      "sha256": "abc123..."
    },
    "trades.jsonl": {
      "record_count": 15,
      "sha256": "def456..."
    }
  },
  "integrity_hash": "combined_hash_of_all_files"
}
```

## Proposal Directory Structure

Research Lab writes to: `\\AI_SHARE\superbot_proposals\`

```
superbot_proposals/
├── prop_20260301_001/
│   ├── proposal.json            # Proposal summary
│   ├── evidence.json            # Supporting data
│   ├── config_diff.json         # Parameter changes
│   ├── replay_results.json      # Shadow replay metrics
│   └── walkforward.json         # Validation results
│
└── pending/
    └── prop_20260301_001 -> ../prop_20260301_001
```

## Access Control

### Production Bots

| Bot | Location | Export Access |
|-----|----------|---------------|
| **Morpheus** | `D:\Trading\IBKR_Algo_BOT_V2` | WRITE to own export folder |
| **Morpheus AI** | `C:\Morpheus\Morpheus_AI` | WRITE to own export folder |
| **Max AI** | `D:\Trading\Max_AI` | WRITE to own export folder |

All production bots:
- WRITE to `\\AI_SHARE\morpheus_exports\{bot_name}\`
- READ-ONLY from `\\AI_SHARE\superbot_proposals\`
- NO ACCESS to `C:\AI_Bot_Research\`

### SuperBot Research Lab

| Path | Access |
|------|--------|
| `\\AI_SHARE\morpheus_exports\` | READ-ONLY |
| `\\AI_SHARE\superbot_proposals\` | READ + WRITE |
| `D:\Trading\IBKR_Algo_BOT_V2` | NO ACCESS |
| `C:\Morpheus\Morpheus_AI` | NO ACCESS |
| `D:\Trading\Max_AI` | NO ACCESS |

## Validation Requirements

### Hash Validation

Research Lab MUST validate all imported data:

1. Read `manifest.json`
2. Compute SHA256 of each data file
3. Compare against manifest hashes
4. Reject if mismatch

### Reproducibility

Every research run MUST record:

- `git_sha` - Code version
- `config_hash` - Configuration hash
- `data_hash` - Input data hash
- `run_id` - Unique run identifier

### Integrity Check

Before any analysis:

```python
def validate_export(export_path):
    manifest = load_json(export_path / "manifest.json")

    for filename, meta in manifest["files"].items():
        file_path = export_path / filename
        actual_hash = compute_sha256(file_path)

        if actual_hash != meta["sha256"]:
            raise IntegrityError(f"Hash mismatch: {filename}")

    return True
```

## Non-Negotiable Rules

1. **Production NEVER imports from Research**
2. **Research NEVER writes to Production**
3. **All exports are append-only**
4. **All imports are validated**
5. **Proposals require human approval**
6. **Zero shared mutable state**

## Trust Boundary

```
PRODUCTION                    RESEARCH
    │                             │
    │  ──── EXPORTS ────────►     │  (read-only)
    │                             │
    │  ◄──── PROPOSALS ──────     │  (requires approval)
    │                             │
    │         NEVER               │
    │  ◄────────────────────►     │  (no direct communication)
    │                             │
```

This boundary is **permanent and non-negotiable**.
