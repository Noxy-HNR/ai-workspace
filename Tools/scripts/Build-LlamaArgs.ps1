# Shared llama-server argument builder.
#
# Dot-sourced by both dashboard-server.ps1 and start.ps1 so the two
# entry points that can launch llama-server never drift out of sync
# (they used to duplicate this logic separately, and start.ps1's copy
# was missing --cache-type-k/--cache-type-v -- a manual restart would
# silently drop any tuned KV-cache-quantization setting).

. (Join-Path $PSScriptRoot 'ModelStorage.ps1')

function Resolve-LlamaProjectorArgs {
    # --mmproj may name a bare file or a full path. A bare name is looked up next to the model first
    # (so a server model takes its projector from the server), then on this PC, then the model server.
    # A full path that no longer exists (a model moved to the server) is looked up by its file name.
    param([string[]]$ArgList, [string]$ModelDir)
    $out = @($ArgList)
    for ($i = 0; $i -lt $out.Count - 1; $i++) {
        if ($out[$i] -notin @('--mmproj', '-mm')) { continue }
        $value = [string]$out[$i + 1]
        if ([IO.Path]::IsPathRooted($value) -and (Test-Path -LiteralPath $value -PathType Leaf)) { continue }
        $leaf = Split-Path -Leaf $value
        $resolved = $null
        if ($ModelDir -and (Test-Path -LiteralPath (Join-Path $ModelDir $leaf) -PathType Leaf)) { $resolved = Join-Path $ModelDir $leaf }
        if (-not $resolved) { $resolved = Resolve-ModelFilePath -FileName $leaf -Kind companion }
        if (-not $resolved) { throw "Projector not found next to the model, on this PC, or on the model server: $leaf" }
        $out[$i + 1] = $resolved
    }
    return $out
}

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
    $effort = $null
    if ($Settings -is [System.Collections.IDictionary]) { $effort = [string]$Settings['reasoningEffort'] }
    elseif ($Settings.PSObject.Properties['reasoningEffort']) { $effort = [string]$Settings.reasoningEffort }
    if (-not [string]::IsNullOrWhiteSpace($effort) -and $effort -ne 'default') {
        $argList += @('--reasoning-effort', $effort)
    }
    if (-not [string]::IsNullOrWhiteSpace($Settings.extraArgs)) {
        $tokenMatches = [regex]::Matches($Settings.extraArgs, '"[^"]*"|''[^'']*''|\S+')
        $extra = @($tokenMatches | ForEach-Object { $_.Value.Trim('"').Trim("'") })
        $argList += @(Resolve-LlamaProjectorArgs -ArgList $extra -ModelDir (Split-Path -Parent $ModelPath))
    }
    return $argList
}
