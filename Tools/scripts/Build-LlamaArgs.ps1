# Shared llama-server argument builder.
#
# Dot-sourced by both dashboard-server.ps1 and start.ps1 so the two
# entry points that can launch llama-server never drift out of sync
# (they used to duplicate this logic separately, and start.ps1's copy
# was missing --cache-type-k/--cache-type-v -- a manual restart would
# silently drop any tuned KV-cache-quantization setting).

function Build-LlamaArgs {
    param([string]$ModelPath, $Settings)
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
    if (-not [string]::IsNullOrWhiteSpace($Settings.extraArgs)) {
        $tokenMatches = [regex]::Matches($Settings.extraArgs, '"[^"]*"|''[^'']*''|\S+')
        foreach ($m in $tokenMatches) { $argList += $m.Value.Trim('"').Trim("'") }
    }
    return $argList
}
