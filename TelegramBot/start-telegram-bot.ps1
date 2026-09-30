<#
  Starts the Telegram bot dashboard (loopback-only, port 9095).
  Usage:  .\start-telegram-bot.ps1  [-Port 9095] [-PythonPath C:\path\python.exe]
#>
param([ValidateRange(1024,65535)][int]$Port = 9095, [switch]$Foreground, [string]$PythonPath,
      [switch]$OpenBrowser, [switch]$ValidateOnly)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $Root 'service-management.ps1')
$url = "http://127.0.0.1:$Port"
if ($ValidateOnly) {
  foreach ($required in @('app.py','service-management.ps1','stop-telegram-bot.ps1')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $required) -PathType Leaf)) { throw "Missing $required" }
  }
  Write-Output 'Relay Station launcher validated.'
  exit 0
}
$mutex = New-Object System.Threading.Mutex($false,"Local\RelayStation-Launcher-$Port")
$owned = $false
try {
try { $owned = $mutex.WaitOne(30000) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
if (-not $owned) { throw 'Another Relay Station launch is still in progress.' }
$existing = Get-RelayPortProcess -Port $Port
if ($existing) {
  $health = Get-RelayHealth -Port $Port
  $legacy = -not $health -and (Test-RelayProcessCommand -Process $existing -Port $Port -AllowLegacy) -and (Test-RelayLegacyConsole -Port $Port)
  if ($health -or $legacy) {
    Save-RelayInstance -Process $existing -Port $Port -AllowLegacy:$legacy
    Write-Host "Relay Station is already running: $url" -ForegroundColor Cyan
    if ($OpenBrowser) { Start-Process $url }
    exit 0
  }
  throw "Port $Port belongs to another service. Choose a different port."
}
if ($PythonPath) {
  if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "PythonPath must point to an existing Python executable."
  }
  $py = (Resolve-Path -LiteralPath $PythonPath).Path
} else {
  $py = (Get-Command py -ErrorAction SilentlyContinue).Source
  if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
  if (-not $py) {
    $installRoot = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    $candidate = Get-ChildItem -LiteralPath $installRoot -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match '^Python\d+$' } |
      Sort-Object { [int]($_.Name -replace '^Python', '') } -Descending |
      ForEach-Object { Join-Path $_.FullName 'python.exe' } |
      Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
      Select-Object -First 1
    if ($candidate) { $py = $candidate }
  }
}
if (-not $py) { throw "Python not found. Install Python 3 or pass -PythonPath with its executable path." }
if ((Split-Path -Leaf $py) -match '^py(?:\.exe)?$') {
  # Record the Python service itself, not the short-lived py.exe launcher.
  $resolvedPython = (& $py -3 -c 'import sys; print(sys.executable)' 2>$null | Select-Object -Last 1)
  if ($LASTEXITCODE -ne 0 -or -not $resolvedPython -or -not (Test-Path -LiteralPath $resolvedPython -PathType Leaf)) {
    throw 'The Python launcher could not resolve a Python 3 executable.'
  }
  $py = [string]$resolvedPython
}
if ($Foreground) {
  Push-Location $Root
  try { & $py -u (Join-Path $Root 'app.py') --port $Port } finally { Pop-Location }
  exit $LASTEXITCODE
}
$logRoot = Join-Path $Root 'logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$process = Start-Process -FilePath $py -ArgumentList @('-u', ('"'+(Join-Path $Root 'app.py')+'"'), '--port', "$Port") `
  -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $logRoot "$stamp-out.log") `
  -RedirectStandardError (Join-Path $logRoot "$stamp-error.log")
$managedProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)" -ErrorAction Stop
Save-RelayInstance -Process $managedProcess -Port $Port
for ($attempt = 0; $attempt -lt 80; $attempt++) {
  $process.Refresh()
  if ($process.HasExited) { throw "Relay Station exited during startup. See $logRoot\$stamp-error.log" }
  try {
    $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 1
    if ($health.service -eq 'relay-station') {
      Write-Host "Relay Station: $url (PID $($process.Id))" -ForegroundColor Cyan
      if ($OpenBrowser) { Start-Process $url }
      exit 0
    }
  } catch { }
  Start-Sleep -Milliseconds 250
}
throw "Relay Station did not become ready. See $logRoot\$stamp-error.log"
} finally {
  if ($owned) { $mutex.ReleaseMutex() }
  $mutex.Dispose()
}
