# Shared llama-server argument builder.
#
# Used by the dashboard's single model-launch path so startup and model
# switching never drift out of sync
# (they used to duplicate this logic separately, and start.ps1's copy
# was missing --cache-type-k/--cache-type-v -- a manual restart would
# silently drop any tuned KV-cache-quantization setting).

. (Join-Path $PSScriptRoot 'LlamaSafety.ps1')

function Build-LlamaArgs {
    param([string]$ModelPath, $Settings)
    foreach ($key in @('contextSize', 'parallel', 'batchSize', 'ubatchSize', 'threads')) {
        $value = 0
        if (-not [int]::TryParse([string]$Settings.$key, [ref]$value) -or $value -lt 1) {
            throw "The model profile requires a positive integer for $key."
        }
    }
    $gpuLayers = 0
    if (-not [int]::TryParse([string]$Settings.gpuLayers, [ref]$gpuLayers) -or $gpuLayers -lt -1) {
        throw 'The model profile requires a valid GPU layer count (-1 for all, or zero or more).'
    }
    $argList = @(
        '--model', $ModelPath,
        '--ctx-size', [string]$Settings.contextSize,
        '--n-gpu-layers', [string]$Settings.gpuLayers,
        '--host', '0.0.0.0',
        '--port', '10000',
        '--parallel', [string]$Settings.parallel,
        '--batch-size', [string]$Settings.batchSize,
        '--ubatch-size', [string]$Settings.ubatchSize,
        '--threads', [string]$Settings.threads,
        '--cache-type-k', [string]$Settings.cacheTypeK,
        '--cache-type-v', [string]$Settings.cacheTypeV
    )
    if ($Settings.flashAttn) { $argList += @('--flash-attn', 'on') } else { $argList += @('--flash-attn', 'off') }
    if ($Settings.jinja) { $argList += '--jinja' }
    if ($Settings.ropeFreqBase -and [double]$Settings.ropeFreqBase -ne 0) {
        $argList += @('--rope-freq-base', [string]$Settings.ropeFreqBase)
    }
    if ($Settings.ropeFreqScale -and [double]$Settings.ropeFreqScale -ne 0) {
        $argList += @('--rope-freq-scale', [string]$Settings.ropeFreqScale)
    }
    $effort = $null
    if ($Settings -is [System.Collections.IDictionary]) { $effort = [string]$Settings['reasoningEffort'] }
    elseif ($Settings.PSObject.Properties['reasoningEffort']) { $effort = [string]$Settings.reasoningEffort }
    if (-not [string]::IsNullOrWhiteSpace($effort) -and $effort -ne 'default') {
        $argList += @('--reasoning-effort', $effort)
    }
    $draft = Resolve-LlamaDraftFile -DraftFile $Settings.draftFile -ModelsDir (Split-Path -Parent $ModelPath)
    if ($draft) { $argList += @('--model-draft', $draft) }
    # Host placement (--n-cpu-moe, --cpu-moe, -ot) stays in extraArgs, where Model Tuner writes it; the
    # dashboard's admission check reads the same flags and charges them as host RAM.
    $argList += @(Get-LlamaSafeExtraArgs -ExtraArgs ([string]$Settings.extraArgs))
    return $argList
}
