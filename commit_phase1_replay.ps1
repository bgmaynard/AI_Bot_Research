# ============================================================
# Morpheus_Lab — Phase 1: Trade-Level Replay Engine
# ============================================================
#   cd C:\AI_Bot_Research
#   .\commit_phase1_replay.ps1
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location "C:\AI_Bot_Research"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Phase 1: Trade-Level Replay Engine" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Pull first (fix remote divergence)
Write-Host ""
Write-Host "Pulling latest..." -ForegroundColor Yellow
git pull --rebase origin main

# Stage
git add Morpheus_Lab/
Write-Host "[OK] Staged" -ForegroundColor Green

Write-Host ""
git status --short
Write-Host ""

git commit -m "feat: Phase 1 — Trade-level market replay engine

NEW CORE MODULES (no pandas, pure streaming):
- core/event_types.py       Minimal frozen TradeEvent(ts, symbol, price, size)
- core/dbn_loader.py        Streaming .dbn.zst loader via store.replay()
                             File indexing by symbol + date from filename
                             Nanosecond timestamp filtering
                             Zero DataFrame conversions
- core/market_replay.py     Heap-merge multi-symbol replay (deterministic)
                             Single-symbol fast mode (no heap overhead)
                             ReplayStats with throughput metrics

CLI ADDED:
  benchmark-replay --cache <path> --symbols CISS --start 2026-02-05 --end 2026-02-05
  Reports: event count, throughput (evt/s), elapsed time, memory usage

DESIGN:
- Strategy-agnostic: yields TradeEvent, consumer decides
- Future: strategy.on_trade(event), slippage injection, order sim
- Deterministic: same inputs = identical event order (heap tiebreak by symbol)
- Streaming: never loads full file, chunks via store.replay()

CACHE PROFILE (post large-cap cleanup):
- XNAS.ITCH trades, nanosecond resolution
- ~160 low-float symbols, Jan 29 - Feb 20 2026
- Auto mode: trades (primary schema)"

Write-Host "[OK] Committed" -ForegroundColor Green

Write-Host ""
Write-Host "Pushing..." -ForegroundColor Cyan
git push origin main

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " PUSH COMPLETE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Test it:" -ForegroundColor Yellow
Write-Host "    cd C:\AI_Bot_Research\Morpheus_Lab" -ForegroundColor Gray
Write-Host "    python -m engine.cli benchmark-replay --cache Z:\AI_BOT_DATA\databento_cache --symbols CISS --start 2026-01-30 --end 2026-01-30" -ForegroundColor Gray
Write-Host ""
