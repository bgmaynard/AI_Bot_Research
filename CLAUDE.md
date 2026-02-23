# CLAUDE.md -- Morpheus Lab Project Context
# Last Updated: 2026-02-23 (Session: Phase 8 + Strategy Price Matrix)
# Location: C:\AI_Bot_Research\Morpheus_Lab\CLAUDE.md

## PROJECT OVERVIEW

Morpheus_Lab is a research-grade backtesting and strategy validation laboratory
for the Morpheus Trading Platform ecosystem. It operates as a separate layer
from production bots, enforcing hypothesis-driven validation before any
configuration change enters live trading.

**Governing Doctrine**: No config change enters live trading unless statistically
validated, regime segmented, execution stress tested, and friction survivable.

## ARCHITECTURE

```
C:\AI_Bot_Research\Morpheus_Lab\     <- Research laboratory (this project)
C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\  <- IBKR trading bot (Bob1 PC)
C:\morpheus\morpheus_ai\             <- Morpheus_AI trading bot (Bob1 PC)
```

Three independent systems:
- **Max_AI**: Scanner (finds candidates)
- **Morpheus_AI**: Trading bot (independent codebase)
- **IBKR_Morpheus**: Trading bot (independent codebase, IBKR execution)

Bots are PEERS, not a pipeline. Each has its own codebase and reports.

**Workflow**: Research (Claude Project) -> Validate (Morpheus_Lab) -> Promote ->
Claude Code in each bot's native repo to implement. No manual runtime edits.

## DATA SOURCES

- **Databento Cache**: `Z:\AI_BOT_DATA\databento_cache` (~239 files, 146 symbols, XNAS.ITCH)
- **Bot EOD Reports**: 
  - IBKR: `C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\reports\`
  - Morpheus_AI: `C:\morpheus\morpheus_ai\reports\`
- **Test Universe**: CISS, BOXL, ELPW, BATL, ANL (Jan 30 - Feb 20, 2026)

## COMPLETED PHASES

### Phase 1: Deterministic Replay Engine
- Streaming Databento loader with vectorized numpy filtering
- Heap-merge replay across multiple symbol-days
- Benchmark: 1.06M events/second
- Files: `core/dbn_loader.py`, `core/market_replay.py`, `core/event_types.py`

### Phase 2: Batch Backtest Engine  
- BatchStrategy base class with on_batch() interface
- BatchBacktestEngine processes full symbol-day arrays
- MomentumBreakout (v1) strategy: vectorized signal + forward-scan exit
- Files: `strategies/batch_strategy.py`, `strategies/momentum_breakout.py`,
  `engine/batch_backtest.py`

### Phase 3: Edge Discovery Layer
- Parameter grid search engine (expand combinations, run per config)
- Walk-forward validation (IS/OOS split, stability scoring)
- Comprehensive metrics: PF, WR, Sharpe, max DD, expectancy
- Formal promotion gate with configurable thresholds
- Files: `engine/grid_engine.py`, `engine/walk_forward.py`,
  `engine/edge_metrics.py`, `core/promotion_gate.py`, `engine/report_generator.py`

### Phase 4: Regime Decomposition
- 6-regime classifier: TREND_UP, TREND_DOWN, HIGH_VOL_BREAKOUT,
  LOW_VOL_CHOP, MEAN_REVERT, VOLATILE_CHOP
- Vectorized classification using rolling volatility + trend slope
- Regime-conditioned metrics and breakdown reporting
- Files: `engine/regime_classifier.py`, updated `engine/cli.py`

### Phase 5: Strategy Evolution (flush_reclaim_v1)
- MomentumContinuationV2 (v2): regime-gated with entry type classification
- Entry type analysis revealed flush_reclaim as strongest pattern (PF 1.27)
- FlushReclaimV1 (fr1): standalone strategy isolated from v2 analysis
- Walk-forward validated: IS_PF=1.64, OOS_PF=1.63, stability=0.996
- Promoted params: lookback=100, flush_pct=0.3, reclaim_window=100,
  reward_multiple=1.5, allowed_regimes=LOW_VOL_CHOP
- All 5 walk-forward candidates qualified (first strategy to pass gate)
- Full backtest: 494 trades, 52% WR, PF 1.64, Sharpe 3.9
- Files: `strategies/flush_reclaim_v1.py`, `strategies/momentum_continuation_v2.py`

### Phase 6: Shadow Mode Deployment
- Shadow trading infrastructure: replay from cache, log signals without execution
- JSONL trade logging (per-trade with timestamps, entry/exit, regime tags)
- Daily summary comparison module (Shadow vs Production)
- Runtime config with promoted parameters
- Architectural fix: full symbol-day batches matching batch backtest exactly
- Validated: perfect alignment with backtest (494 trades, $600 PnL, PF 1.64)
- Files: `shadow/shadow_runner.py`, `shadow/shadow_logger.py`,
  `shadow/runtime_config.json`

### Phase 7: Friction Stress Testing
- FrictionConfig dataclass: slippage, latency, spread, commission, tick_size
- FrictionModel: post-processing layer, applies friction without modifying originals
- CLI flags: --slippage-ticks, --latency-ticks, --spread-cost, --commission
- **CRITICAL FINDING**: Baseline PF 1.64 collapsed to PF 0.55 under realistic
  friction (1 tick + $0.50 commission = $2.50/trade)
- Root cause: avg winner $6.05, avg loser -$3.94, edge $1.21 vs friction $2.50
- Price tier diagnostic: sub-$5 unviable, $5+ stocks survive friction
- Files: `engine/friction_model.py`

### Phase 8: Friction Tier Survivability Diagnostic
- Automated price-tier analysis tool (931 lines)
- Edge Buffer Ratio (EBR) = avg_winner_per_share / friction_per_share
- Verdicts: DEAD (<1.0 EBR), FRAGILE (1-2), VIABLE (2-3), ROBUST (>3)
- Recommendation engine: requires Net PF >= 1.2, Avg PnL > 0, EBR >= 2.0
- **RESULT ON REAL DATA**:
  - $1-$3 (359 trades): PF 0.25 net, EBR 1.4 -> FRAGILE (24% winners flipped)
  - $3-$5 (103 trades): PF 0.65 net, EBR 2.4 -> VIABLE (but fails PF criteria)
  - $5-$7 (5 trades): insufficient data
  - $7-$10 (21 trades): PF 1.57 net, EBR 9.2 -> ROBUST (only qualifying tier)
  - $10-$20 (6 trades): insufficient data
- **RECOMMENDATION**: $7.00 min-price filter, HIGH confidence
- ASCII-safe for Windows cp1252 (all Unicode replaced)
- Files: `analysis/friction_price_tier_analysis.py`, `analysis/__init__.py`

### Phase 8.5: Strategy x Price Tier Research Matrix
- Multi-strategy comparison across price tiers under identical conditions
- Runs all 3 strategies (v1, v2, fr1) through batch-backtest
- Slices trades into 6 tiers: $1-$3, $3-$5, $5-$7, $7-$10, $10-$15, $15+
- Applies standardized realistic friction ($0.03/share)
- Deployment criteria: Net PF >= 1.20, Avg PnL > 0, EBR >= 2.0, N >= 50
- Generates routing map: which strategy owns each price tier
- Output: reports/strategy_price_matrix.json, reports/strategy_price_matrix.md
- Files: `analysis/strategy_price_matrix.py`
- **STATUS**: Built, tested, awaiting first run on live data (scheduled Feb 24)

## CURRENT STRATEGIES

| Strategy | Code | WR | PF (gross) | Status |
|----------|------|----|------------|--------|
| momentum_breakout | v1 | 22% | 1.06-1.13 | Baseline, poor edge |
| momentum_continuation_v2 | v2 | ~35% | ~0.9 | Regime-gated, entry classification |
| flush_reclaim_v1 | fr1 | 52% | 1.64 | Validated, promoted, friction-tested |

## KEY FINDINGS

1. **flush_reclaim_v1 is the only strategy with validated edge** (PF 1.64, Sharpe 3.9)
2. **Friction destroys sub-$5 edge**: avg winner $0.04/share vs $0.03/share friction
3. **$7+ is the survivable price tier**: EBR 9.2, net PF 1.57
4. **Only 21 trades in qualifying tier**: Need expanded universe for deployment
5. **Price-aware routing is the path forward**: different tiers may need different strategies

## CLI COMMANDS

```bash
# Batch backtest (any strategy)
python -m engine.cli batch-backtest --cache Z:\AI_BOT_DATA\databento_cache --symbols CISS,BOXL,ELPW,BATL,ANL --start 2026-01-30 --end 2026-02-20 --strategy fr1

# Shadow replay
python -m engine.cli shadow-replay --cache Z:\AI_BOT_DATA\databento_cache --symbols CISS,BOXL,ELPW,BATL,ANL --start 2026-01-30 --end 2026-02-20

# Friction tier analysis
python -m engine.cli friction-tier-analysis --cache Z:\AI_BOT_DATA\databento_cache --symbols CISS,BOXL,ELPW,BATL,ANL --start 2026-01-30 --end 2026-02-20

# Strategy price matrix (NEW - run on live data Feb 24)
python -m engine.cli strategy-price-matrix --cache Z:\AI_BOT_DATA\databento_cache --symbols CISS,BOXL,ELPW,BATL,ANL --start 2026-01-30 --end 2026-02-20 --allowed-regimes "LOW_VOL_CHOP"

# Grid search
python -m engine.cli grid-search --cache Z:\AI_BOT_DATA\databento_cache --strategy fr1

# Walk-forward validation
python -m engine.cli walk-forward --cache Z:\AI_BOT_DATA\databento_cache --strategy fr1
```

## FILE MANIFEST

```
Morpheus_Lab/
  core/
    dbn_loader.py          # Databento cache loader + index
    market_replay.py       # Heap-merge replay engine
    event_types.py         # TradeEvent dataclass
    promotion_gate.py      # Formal promotion criteria
  engine/
    batch_backtest.py      # BatchBacktestEngine + results
    grid_engine.py         # Parameter grid search
    walk_forward.py        # Walk-forward validation
    edge_metrics.py        # Comprehensive metric computation
    regime_classifier.py   # 6-regime vectorized classifier
    friction_model.py      # Execution cost model (Phase 7)
    report_generator.py    # JSON/markdown reporting
    cli.py                 # CLI entrypoint (1297 lines)
  strategies/
    batch_strategy.py      # BatchStrategy base class
    momentum_breakout.py   # v1: vanilla breakout
    momentum_continuation_v2.py  # v2: regime-gated + entry types
    flush_reclaim_v1.py    # fr1: validated flush-reclaim
  analysis/
    __init__.py
    friction_price_tier_analysis.py  # Phase 8: tier survivability (931 lines)
    strategy_price_matrix.py         # Phase 8.5: multi-strategy matrix (706 lines)
  shadow/
    shadow_runner.py       # Shadow mode replay engine
    shadow_logger.py       # JSONL trade logger
    runtime_config.json    # Promoted fr1 parameters
    __main__.py
  reports/                 # Generated reports (JSON + MD)
  logs/                    # Shadow JSONL trade logs
```

## NEXT STEPS (Priority Order)

1. **Feb 24**: Run strategy-price-matrix on live Databento cache data
   - Compare lab friction predictions vs actual IBKR/Morpheus_AI EOD reports
   - Validate friction model accuracy against real execution
2. **Expand Databento cache**: More $5-$20 low-float symbols via live bot trading
3. **Re-run tier analysis**: Confirm $7+ edge holds with 80-100+ trades
4. **Build reconciliation tool**: Lab predictions vs live bot P&L by price tier
5. **If $7+ confirmed**: Add min-price filter to fr1 -> proceed to live deployment

## PRINCIPLES

- Validate and correlate data before implementing
- No manual runtime edits -- all changes through lab validation
- Hypothesis-driven: test -> validate -> promote -> deploy
- Statistical significance required (N >= 50 for deployment decisions)
- Friction-aware: no strategy deploys without execution cost testing
- Research PC (analysis) strictly separated from Trading PC (execution)
