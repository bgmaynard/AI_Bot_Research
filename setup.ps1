Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AI_Bot_Research - Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "C:\AI_Bot_Research"
Write-Host "[OK] Working directory: C:\AI_Bot_Research" -ForegroundColor Green

# Create folders
$folders = "docs", "mrl", "results", "data"
foreach ($f in $folders)
{
    if (-not (Test-Path $f))
    {
        New-Item -ItemType Directory -Path $f | Out-Null
        Write-Host "[CREATED] $f" -ForegroundColor Yellow
    }
    else
    {
        Write-Host "[EXISTS]  $f" -ForegroundColor Gray
    }
}

# Move whitepaper to docs if in root
if (Test-Path "MPAI_whitepaper.md")
{
    Move-Item "MPAI_whitepaper.md" "docs\MPAI_whitepaper.md" -Force
    Write-Host "[MOVED]   MPAI_whitepaper.md -> docs\" -ForegroundColor Yellow
}
elseif (Test-Path "docs\MPAI_whitepaper.md")
{
    Write-Host "[EXISTS]  docs\MPAI_whitepaper.md" -ForegroundColor Gray
}
else
{
    Write-Host "[WARN]    MPAI_whitepaper.md not found" -ForegroundColor Red
}

# Move main.py to mrl if in root
if (Test-Path "main.py")
{
    Move-Item "main.py" "mrl\main.py" -Force
    Write-Host "[MOVED]   main.py -> mrl\" -ForegroundColor Yellow
}
elseif (Test-Path "mrl\main.py")
{
    Write-Host "[EXISTS]  mrl\main.py" -ForegroundColor Gray
}
else
{
    Write-Host "[WARN]    main.py not found" -ForegroundColor Red
}

# Create tickers.txt if missing
if (-not (Test-Path "tickers.txt"))
{
    "AAPL", "TSLA", "NVDA", "AMD", "BBAI" | Set-Content "tickers.txt"
    Write-Host "[CREATED] tickers.txt" -ForegroundColor Yellow
}
else
{
    Write-Host "[EXISTS]  tickers.txt" -ForegroundColor Gray
}

# Check README and gitignore
if (Test-Path "README.md") { Write-Host "[EXISTS]  README.md" -ForegroundColor Gray }
else { Write-Host "[WARN]    README.md not found" -ForegroundColor Red }

if (Test-Path ".gitignore") { Write-Host "[EXISTS]  .gitignore" -ForegroundColor Gray }
else { Write-Host "[WARN]    .gitignore not found" -ForegroundColor Red }

# Remove old bat file
if (Test-Path "setup_project.bat")
{
    Remove-Item "setup_project.bat" -Force
    Write-Host "[REMOVED] setup_project.bat" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Git Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check git
try
{
    $gv = git --version
    Write-Host "[OK] $gv" -ForegroundColor Green
}
catch
{
    Write-Host "[ERROR] Git not installed. Get it from https://git-scm.com" -ForegroundColor Red
    exit 1
}

# Init repo
if (-not (Test-Path ".git"))
{
    git init
    Write-Host "[OK] Git repo initialized" -ForegroundColor Green
}
else
{
    Write-Host "[EXISTS]  Git repo already initialized" -ForegroundColor Gray
}

# Stage and commit
git add .
Write-Host "[OK] Files staged" -ForegroundColor Green

Write-Host ""
Write-Host "Staged files:" -ForegroundColor Cyan
git status --short
Write-Host ""

git commit -m "Initial project setup - MPAI research engine and whitepaper V3.0"
Write-Host "[OK] Commit created" -ForegroundColor Green

git branch -M main
Write-Host "[OK] Branch set to main" -ForegroundColor Green

# Add remote
$remote = git remote -v 2>&1
if ($remote -match "origin")
{
    git remote set-url origin "https://github.com/bgmaynard/AI_Bot_Research.git"
    Write-Host "[UPDATED] Remote origin" -ForegroundColor Yellow
}
else
{
    git remote add origin "https://github.com/bgmaynard/AI_Bot_Research.git"
    Write-Host "[OK] Remote origin added" -ForegroundColor Green
}

# Push
Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
git push -u origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " SETUP COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Repo: https://github.com/bgmaynard/AI_Bot_Research"
Write-Host ""
Write-Host "  REMINDER: Set repo to PRIVATE on GitHub" -ForegroundColor Yellow
Write-Host ""
