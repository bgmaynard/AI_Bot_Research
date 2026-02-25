# git_sync.ps1 - Run from anywhere on Bob1 PC
# Commits and pushes Phase 7-8.5 changes

Set-Location "C:\AI_Bot_Research"

# Copy CLAUDE.md to repo root if it exists in Morpheus_Lab
if (Test-Path "Morpheus_Lab\CLAUDE.md") {
    Copy-Item "Morpheus_Lab\CLAUDE.md" "CLAUDE.md" -Force
    Write-Host "Copied CLAUDE.md to repo root" -ForegroundColor Green
}

git add -A
Write-Host "`n=== GIT STATUS ===" -ForegroundColor Cyan
git status --short

$commitMsg = @"
Phase 7-8.5: Friction model, tier survivability, strategy price matrix

Phase 7: Friction stress testing (engine/friction_model.py)
- Realistic execution cost model: slippage, latency, spread, commission
- PF 1.64 -> 0.55 under friction on sub-`$5 stocks

Phase 8: Friction tier survivability (analysis/friction_price_tier_analysis.py)
- Edge Buffer Ratio (EBR) survivability metric
- Real data: `$7-`$10 only qualifying tier (PF 1.57 net, EBR 9.2)
- Windows cp1252 Unicode fix (all ASCII-safe)

Phase 8.5: Strategy x price tier matrix (analysis/strategy_price_matrix.py)
- Multi-strategy comparison: v1, v2, fr1 across 6 price tiers
- CLI: strategy-price-matrix command
- Awaiting first live data run Feb 24

Updated CLAUDE.md with complete project state.
"@

git commit -m $commitMsg
git push origin main

Write-Host "`n=== DONE ===" -ForegroundColor Green
