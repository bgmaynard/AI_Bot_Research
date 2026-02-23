# ============================================================
# Morpheus_Lab — Databento Feed Integration Commit
# ============================================================
# Run from PowerShell:
#   cd C:\AI_Bot_Research
#   .\commit_databento_feed.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Set-Location "C:\AI_Bot_Research"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Committing Databento Feed Integration" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Stage all Morpheus_Lab changes
git add Morpheus_Lab/
Write-Host "[OK] Morpheus_Lab staged" -ForegroundColor Green

# Show staged files
Write-Host ""
Write-Host "Staged files:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Commit
git commit -m "feat: Databento feed integration + auto timeframe detection

NEW MODULES:
- core/events.py                  Unified event types (TradeEvent, BarEvent, QuoteEvent)
- core/market_replay.py           Deterministic replay engine with multi-symbol interleaving
- datafeeds/databento_inspector.py Cache inspector: schemas, symbols, date range, event volume
- datafeeds/databento_feed.py     Auto-detecting feed with streaming iteration

AUTO MODE PRIORITY:
  1. ohlcv-1s (bars_1s) if present - fastest usable
  2. trades + BarAggregator(1s) - most granular
  3. ohlcv-1m (bars_1m) - fallback

CLI COMMANDS ADDED:
  inspect-databento --cache <path>    Scan cache, write reports/dataset_profile.json
  backtest --cache <path> --mode auto Replay events through strategy callbacks

FEATURES:
- Chunked streaming iteration (no memory blowups)
- BarAggregator converts raw trades to 1s bars on the fly
- Multi-symbol heap-interleaved replay for precise cross-symbol timing
- Single-symbol fast loop mode for speed benchmarking
- TradeCollector callback for accumulating trade results
- Dataset profile JSON for machine-readable cache analysis

UPDATED:
- engine/cli.py now has 8 commands (added inspect-databento + backtest)
- __main__.py added for python -m Morpheus_Lab.cli support

USAGE:
  cd C:\AI_Bot_Research\Morpheus_Lab
  python -m engine.cli inspect-databento --cache Z:\AI_BOT_DATA\databento_cache
  python -m engine.cli backtest --cache Z:\AI_BOT_DATA\databento_cache --mode auto"

Write-Host "[OK] Commit created" -ForegroundColor Green

# Push
Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
git push origin main

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " PUSH COMPLETE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "    1. Run inspector to profile your cache:" -ForegroundColor White
Write-Host "       cd C:\AI_Bot_Research\Morpheus_Lab" -ForegroundColor Gray
Write-Host "       python -m engine.cli inspect-databento --cache Z:\AI_BOT_DATA\databento_cache" -ForegroundColor Gray
Write-Host ""
Write-Host "    2. Run observation backtest:" -ForegroundColor White
Write-Host "       python -m engine.cli backtest --cache Z:\AI_BOT_DATA\databento_cache --mode auto" -ForegroundColor Gray
Write-Host ""
