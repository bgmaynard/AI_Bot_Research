# Morpheus Lab
## Research & Promotion Framework

**Version:** 1.0  
**Author:** Bob Maynard  
**Status:** Governing Framework for all Morpheus Systems

---

## Overview

Morpheus Lab is a quantitative research laboratory for validating trading strategy changes before they enter live trading. It enforces hypothesis-driven development, multi-regime backtesting, execution sensitivity modeling, and structured promotion gates.

**No configuration change enters live trading without passing this framework.**

### Systems Governed
- **Morpheus_AI** — Trading bot (signal logic, gates, exits, regime switching)
- **IBKR_Morpheus** — Trading bot (order routing, slippage, trailing stops)
- **Max_AI** — Scanner server (ranking logic, float weighting, catalyst scoring)

Each system is independent with its own codebase. Research is conducted here. Validated changes are implemented via Claude Code in each bot's native repo.

---

## Directory Structure

```
Morpheus_Lab/
├── data/                    # Raw and processed data
├── hypotheses/              # YAML hypothesis definitions
│   └── sample_hypothesis.yaml
├── scenarios/               # Pre-built test scenarios
├── strategies/              # Baseline configs and strategy definitions
│   └── runtime_config_baseline.json
├── execution_models/        # Slippage and latency simulation
│   └── slippage_model.py
├── engine/                  # Core test engine
│   ├── cli.py               # CLI entrypoint
│   ├── grid_runner.py       # Grid test executor
│   ├── hypothesis_loader.py # YAML loader + grid generator
│   ├── metrics.py           # Full metrics calculator
│   └── regime_segmenter.py  # Regime classification
├── scoring/                 # Comparison and promotion scoring
│   ├── baseline_comparator.py
│   └── promotion_score.py
├── results/                 # All test outputs
│   ├── candidates/          # In-progress and unscored results
│   ├── validated/           # Passed promotion gate (≥ 0.75)
│   └── rejected/            # Failed promotion gate (< 0.65)
├── reports/                 # Generated analysis reports
└── promotion/               # Promotion artifacts and logs
    └── promote.py           # Promotion pipeline
```

---

## Quick Start

### 1. Define a Hypothesis

Copy `hypotheses/sample_hypothesis.yaml` and modify:

```yaml
hypothesis_id: H-2026-02-22-A
description: Increase signal_confirm_hold_seconds to reduce false breakouts
baseline_config: ../strategies/runtime_config_baseline.json
parameter_grid:
  signal_confirm_hold_seconds: [1.75, 2.0, 2.25, 2.5]
  spread_stability_ratio: [1.35, 1.45]
evaluation_window:
  start: "2026-01-01"
  end: "2026-02-20"
execution_models:
  slippage: [0.01, 0.03, 0.05]
  latency_ms: [0, 150, 300]
target_system: morpheus_ai
```

### 2. Validate the Hypothesis

```bash
cd Morpheus_Lab
python -m engine.cli load hypotheses/sample_hypothesis.yaml
```

### 3. Run Grid Test

```bash
python -m engine.cli run hypotheses/sample_hypothesis.yaml --workers 4
```

### 4. Score Against Baseline

```bash
python -m engine.cli score H-2026-02-22-A
```

### 5. Promote if Validated

```bash
python -m engine.cli promote H-2026-02-22-A --target morpheus_ai --approved
```

### 6. Shadow Validation

```bash
python -m engine.cli shadow H-2026-02-22-A
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `load <file>` | Load and validate a hypothesis YAML file |
| `run <file>` | Execute full grid test with execution modeling |
| `score <id>` | Score hypothesis against baseline |
| `promote <id>` | Promote validated hypothesis to runtime candidate |
| `shadow <id>` | Generate shadow validation checklist |
| `status <id>` | Check current status of a hypothesis |

**Flags:**
- `--workers N` — Parallel workers for grid testing (default: 1)
- `--seed N` — Random seed for reproducibility (default: 42)
- `--baseline <path>` — Override baseline config path
- `--target <system>` — Target system for promotion
- `--approved` — Supervisor approval flag
- `-v` — Verbose logging

---

## Validation Layers

All changes must pass five layers:

1. **Multi-Regime Backtesting** — Performance segmented by dead tape, momentum, catalyst, SSR, mixed
2. **Execution Sensitivity Modeling** — Tested under slippage (+0.01/+0.03/+0.05) and latency (0/150/300ms)
3. **Stability & Risk Analysis** — Worst-day, 95th percentile drawdown, variance, clustering
4. **Baseline Relative Comparison** — Absolute profit is irrelevant; relative improvement required
5. **Promotion Scoring Gate** — Weighted composite score must reach threshold

### Promotion Score Weights
| Component | Weight |
|-----------|--------|
| Expectancy improvement | 35% |
| Stability improvement | 25% |
| Drawdown reduction | 20% |
| Regime consistency | 10% |
| Execution robustness | 10% |

### Thresholds
| Score | Status |
|-------|--------|
| ≥ 0.75 | **Validated** — eligible for promotion |
| 0.65–0.75 | **Candidate** — needs improvement |
| < 0.65 | **Rejected** |

---

## Metrics Computed

Every backtest run produces:
- Total P&L
- Expectancy
- Win rate
- Profit factor
- Max drawdown
- 95th percentile worst day
- Daily variance
- Trade frequency
- Sharpe-like ratio

No incomplete metric sets allowed.

---

## Integration: Connecting Your Backtest

The grid runner requires a backtest function with this signature:

```python
def my_backtest(config: dict, eval_window: dict, seed: int) -> list[dict]:
    """
    Args:
        config: Merged parameter config
        eval_window: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
        seed: Deterministic seed

    Returns:
        List of trade dicts, each containing:
            - pnl: float
            - date: str
            - entry_price: float
            - exit_price: float
            - direction: str ("long" | "short")
            - shares: int
            - regime: str (optional, auto-classified if missing)
            - relative_volume: float (optional)
            - tape_speed: float (optional)
    """
```

Wire it into `engine/cli.py` replacing the placeholder.

---

## Non-Negotiable Rules

- No runtime bot logic may be modified directly
- No risk controls may be relaxed without lab validation + supervisor approval
- No config promotion without traceable hypothesis ID
- No assumption of zero slippage
- No skipping regime segmentation
- Manual runtime edits are prohibited

---

## Phase 2 (Future — Scaffolded, Not Implemented)

- Genetic parameter mutation
- Bayesian optimization
- AI-guided parameter search
- Automated quarterly evolution cycles
- Regime-specific config bundles

---

## Dependencies

```
pyyaml
```

Install: `pip install pyyaml`

---

## Governing Doctrine

This lab governs all Morpheus systems. No configuration change enters live trading unless it is statistically validated, regime segmented, execution stress tested, baseline improved, scored above threshold, and shadow confirmed.

**This is institutional discipline.**
