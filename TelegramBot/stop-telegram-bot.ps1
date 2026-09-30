<# Stops only a verified Relay Station process; never another local AI service. #>
param([ValidateRange(1024,65535)][int]$Port = 9095, [switch]$Force)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'service-management.ps1')
$process = Get-RelayPortProcess -Port $Port
if (-not $process) { Write-Output 'Relay Station is already stopped.'; exit 0 }
if (-not (Test-RelayOwnedProcess -Process $process -Port $Port)) {
    throw "Refusing to stop an unrecognized process on port $Port. Start Relay Station through its launcher first."
}
$relayProcessId = [int]$process.ProcessId
$health = Get-RelayHealth -Port $Port
if ($health) {
    try {
        $session = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/session" -TimeoutSec 2
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/shutdown" -Method Post -ContentType 'application/json' `
            -Headers @{'X-Relay-CSRF'=[string]$session.csrf} -Body '{}' -TimeoutSec 3
        $deadline = (Get-Date).AddSeconds(10)
        while (Get-Process -Id $relayProcessId -ErrorAction SilentlyContinue) {
            if ((Get-Date) -ge $deadline) { break }
            Start-Sleep -Milliseconds 150
        }
    } catch {}
}
$remaining = Get-CimInstance Win32_Process -Filter "ProcessId=$relayProcessId" -ErrorAction SilentlyContinue
if ($remaining) {
    if (-not $Force) { throw 'Relay Station is still running. Use -Force for a verified legacy or unresponsive instance.' }
    if (-not (Test-RelayOwnedProcess -Process $remaining -Port $Port)) { throw 'Process identity changed; stop refused.' }
    Stop-Process -Id $relayProcessId -Force -ErrorAction Stop
    Wait-Process -Id $relayProcessId -Timeout 5 -ErrorAction SilentlyContinue
}
if (Get-Process -Id $relayProcessId -ErrorAction SilentlyContinue) { throw 'Relay Station has not exited yet; no new instance was started.' }
Write-Output "Relay Station stopped (PID $relayProcessId)."
