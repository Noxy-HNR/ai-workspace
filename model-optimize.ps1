# ============================================================
# Model Tuner launcher (Tools/model-tuner)
#
# Usage:
#   .\model-optimize.ps1 -Model Qwen3.5-9B-UD-Q6_K_XL-mtp.gguf -Mode quick
#   .\model-optimize.ps1 -Model OxCoder-9B.i1-Q6_K.gguf -Mode medium -Detach
#   .\model-optimize.ps1 -Plan -Model medgemma-27b-it-Q4_K_M.gguf
#
# The old Docker-based optimizer is gone. Model Tuner reads the model file, plans placement and
# context on paper (llama-fit-params), then tests only the settings that matter, loading every
# trial through the dashboard without touching models.json. See Tools/model-tuner/README.md, or
# use the dashboard's Model Tuner tab.
# ============================================================

param(
    [string]$Model,
    [ValidateSet('quick','medium','full')]
    [string]$Mode = 'quick',
    [switch]$Plan,
    [switch]$Detach
)

$ErrorActionPreference = 'Stop'
$tuner = Join-Path $PSScriptRoot 'Tools\model-tuner'
if (-not $Model) {
    Push-Location $tuner
    try { python -m modeltuner profile } finally { Pop-Location }
    Write-Host "`nPass -Model <file> to plan or tune one of these."
    exit 0
}
$cliArgs = if ($Plan) { @('plan', $Model) } else { @('run', $Model, '--mode', $Mode) }
if ($Detach -and -not $Plan) { $cliArgs += '--detach' }
Push-Location $tuner
try { python -m modeltuner @cliArgs; exit $LASTEXITCODE } finally { Pop-Location }
