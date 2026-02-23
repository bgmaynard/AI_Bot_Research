# ============================================================
# Morpheus_Lab — Git Commit & Push
# ============================================================
# Run from PowerShell:
#   cd C:\AI_Bot_Research
#   .\commit_morpheus_lab.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Set-Location "C:\AI_Bot_Research"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Committing Morpheus_Lab to GitHub" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Stage Morpheus_Lab
git add Morpheus_Lab/
Write-Host "[OK] Morpheus_Lab staged" -ForegroundColor Green

# Show staged files
Write-Host ""
Write-Host "Staged files:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Commit
git commit -m "feat: Deploy Morpheus Research & Promotion Framework v1.0

Modules:
- engine/hypothesis_loader.py    YAML loader + grid combination generator
- engine/grid_runner.py          Multiprocessing grid executor with resume
- engine/metrics.py              Full metric set (9 metrics, no incomplete sets)
- engine/regime_segmenter.py     5-regime classification (dead_tape/momentum/catalyst/SSR/mixed)
- engine/cli.py                  CLI: load, run, score, promote, shadow, status
- execution_models/slippage_model.py  Injectable slippage + latency + order type sim
- scoring/baseline_comparator.py      Relative improvement only (absolute profit forbidden)
- scoring/promotion_score.py          Weighted composite scoring (0.75 gate)
- promotion/promote.py                Promotion pipeline + shadow checklist + supervisor signoff
- hypotheses/sample_hypothesis.yaml   Copy-and-modify template
- strategies/runtime_config_baseline.json  Sample baseline config
- README.md                           Full documentation

Architecture: Max_AI (scanner) | Morpheus_AI (bot) | IBKR_Morpheus (bot)
Three independent systems - each evaluated on own terms
Workflow: Research (Claude Project) -> Validate -> Promote -> Claude Code per bot repo
Next: Wire real backtest function into engine/cli.py"

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
Write-Host "  Repo: https://github.com/bgmaynard/AI_Bot_Research" -ForegroundColor White
Write-Host "  Path: Morpheus_Lab/" -ForegroundColor White
Write-Host ""
