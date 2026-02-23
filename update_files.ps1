# Copy optimized files to Morpheus_Lab
# Run from wherever you downloaded the files

$dl = "$env:USERPROFILE\Downloads"
$lab = "C:\AI_Bot_Research\Morpheus_Lab"

Copy-Item "$dl\dbn_loader.py"    "$lab\core\dbn_loader.py"    -Force
Copy-Item "$dl\market_replay.py" "$lab\core\market_replay.py" -Force
Copy-Item "$dl\event_types.py"   "$lab\core\event_types.py"   -Force
Copy-Item "$dl\cli.py"           "$lab\engine\cli.py"         -Force

Write-Host "4 files copied to Morpheus_Lab" -ForegroundColor Green
