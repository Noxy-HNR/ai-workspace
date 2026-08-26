$ErrorActionPreference="Continue"
Set-Location "C:\AI"

Write-Host "Stopping llama-server..." -ForegroundColor Cyan
$llamaConn = Get-NetTCPConnection -LocalPort 10000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($llamaConn) { Stop-Process -Id $llamaConn.OwningProcess -Force -ErrorAction SilentlyContinue }
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Remove-Item "C:\AI\Config\llama-native.pid" -Force -ErrorAction SilentlyContinue

Write-Host "Stopping dashboard server..." -ForegroundColor Cyan
$dashConn = Get-NetTCPConnection -LocalPort 9090 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dashConn -and $dashConn.OwningProcess -ne 4) { Stop-Process -Id $dashConn.OwningProcess -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {$_.CommandLine -match 'dashboard-server\.ps1'} |
    ForEach-Object {Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue}

Write-Host "Stopping cptr..." -ForegroundColor Cyan
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {$_.CommandLine -match '\bcptr(\.exe)?\s+run\b'} |
    ForEach-Object {Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue}

Write-Host "Local AI stack stopped." -ForegroundColor Green
