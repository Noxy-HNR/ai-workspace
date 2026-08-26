$ErrorActionPreference="Continue"
Set-Location "C:\AI"

Write-Host "=== llama-server (native) ===" -ForegroundColor Cyan
$llamaConn = Get-NetTCPConnection -LocalPort 10000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($llamaConn) {
    Write-Host "Running: PID $($llamaConn.OwningProcess)"
} else {
    Write-Host "Not running." -ForegroundColor Yellow
}

Write-Host "`n=== Dashboard server ===" -ForegroundColor Cyan
$dashConn = Get-NetTCPConnection -LocalPort 9090 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dashConn) { Write-Host "Running (port 9090 listening)." } else { Write-Host "Not running." -ForegroundColor Yellow }

Write-Host "`n=== NVIDIA ===" -ForegroundColor Cyan
nvidia-smi

Write-Host "`n=== llama.cpp API ===" -ForegroundColor Cyan
try{Invoke-RestMethod "http://localhost:10000/v1/models" | ConvertTo-Json -Depth 6}
catch{Write-Host "llama.cpp API unavailable." -ForegroundColor Yellow}

Write-Host "`n=== cptr ===" -ForegroundColor Cyan
try{Invoke-RestMethod "http://localhost:8000/api/health" | ConvertTo-Json -Depth 5}
catch{Write-Host "cptr unavailable." -ForegroundColor Yellow}

Write-Host "`n=== Python/cptr ===" -ForegroundColor Cyan
python --version
python -m pip show cptr
