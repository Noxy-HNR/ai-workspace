# ============================================================
# Local AI llama.cpp Auto Optimizer
#
# Usage:
#   .\model-optimize.ps1 -Mode quick
#   .\model-optimize.ps1 -Mode medium
#   .\model-optimize.ps1 -Mode full
#   .\model-optimize.ps1 -Mode quick -Json
#
# STATUS: not implemented for the native (Docker-free) stack.
#
# This optimizer used to benchmark configs by spinning up temporary
# Docker containers. The stack migrated to running llama-server
# natively (no Docker) for lower idle RAM usage, so that approach no
# longer applies -- it would need a native rewrite that spins up/tears
# down temporary llama-server.exe processes on an alternate port
# instead of containers. Until that exists, this fails cleanly with
# valid JSON (the dashboard's Auto Optimizer tab expects that) instead
# of crashing on missing docker commands. Tune parameters manually in
# the Models tab instead.
# ============================================================

param(
    [ValidateSet('quick','medium','full')]
    [string]$Mode = 'quick',

    [switch]$Json
)

$ErrorActionPreference = 'Stop'

$result = @{
    success = $false
    mode = $Mode
    message = 'Auto-Optimizer is not yet implemented for the native llama-server setup (it used to rely on Docker container benchmarking). Tune parameters manually in the Models tab instead.'
}

if ($Json) {
    $result | ConvertTo-Json -Depth 5
} else {
    Write-Host $result.message -ForegroundColor Yellow
}
exit 1
