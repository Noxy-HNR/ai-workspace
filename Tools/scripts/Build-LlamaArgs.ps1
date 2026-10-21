# Shared llama-server argument builder.
#
# Used by the dashboard's single model-launch path so startup and model
# switching never drift out of sync
# (they used to duplicate this logic separately, and start.ps1's copy
# was missing --cache-type-k/--cache-type-v -- a manual restart would
# silently drop any tuned KV-cache-quantization setting).

. (Join-Path $PSScriptRoot 'LlamaSafety.ps1')
. (Join-Path $PSScriptRoot 'ModelStorage.ps1')

function Resolve-LlamaProjectorArgs {
    # --mmproj may name a bare file or a full path. A bare name is looked up next to the model first
    # (so a server model takes its projector from the server), then on this PC, then the model server.
    # A full path that no longer exists (a model moved to the server) is looked up by its file name.
    param([string[]]$ArgList, [string]$ModelDir)
    $out = @()
    for ($i = 0; $i -lt $ArgList.Count; $i++) {
        $flag, $inline = ([string]$ArgList[$i] -split '=', 2)
        if ($flag -notin @('--mmproj', '-mm')) { $out += $ArgList[$i]; continue }
        if ($null -ne $inline) { $value = $inline }
        # The next flag is never a projector name (the RAM-safety flags always follow extraArgs).
        elseif ($i + 1 -lt $ArgList.Count -and [string]$ArgList[$i + 1] -notlike '-*') { $i++; $value = [string]$ArgList[$i] }
        else { $value = '' }
        if ([string]::IsNullOrWhiteSpace($value)) { throw "$flag needs a projector file name." }
        # Server copies must have settled, like models, so a projector still being copied never loads.
        if ([IO.Path]::IsPathRooted($value) -and (Test-ModelFileUsable $value)) { $out += @($flag, $value); continue }
        $leaf = Split-Path -Leaf $value
        $resolved = $null
        if ($ModelDir -and (Test-ModelFileUsable (Join-Path $ModelDir $leaf))) { $resolved = Join-Path $ModelDir $leaf }
        if (-not $resolved) { $resolved = Resolve-ModelFilePath -FileName $leaf -Kind companion }
        if (-not $resolved) { throw "Projector not found next to the model, on this PC, or on the model server: $leaf" }
        $out += @($flag, $resolved)
    }
    return $out
}

function ConvertTo-LlamaArgumentLine {
    # Start-Process joins an argument array with spaces and no quoting, which splits a path such as
    # \\server\Model Files\m.gguf in two. Quote each argument the way Windows programs parse them.
    param([string[]]$ArgList)
    $quoted = foreach ($a in $ArgList) {
        if ($a -ne '' -and $a -notmatch '[\s"]') { $a; continue }
        $escaped = [regex]::Replace($a, '(\\*)"', { param($m) ($m.Groups[1].Value * 2) + '\"' })
        '"' + [regex]::Replace($escaped, '(\\+)$', '$1$1') + '"'
    }
    return ($quoted -join ' ')
}

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
    # dashboard's admission check reads the same flags and charges them as host RAM. The safe extraArgs
    # always end with --load-mode none --cache-ram 0, also for a model with no saved profile yet.
    $argList += @(Resolve-LlamaProjectorArgs -ArgList @(Get-LlamaSafeExtraArgs -ExtraArgs ([string]$Settings.extraArgs)) -ModelDir (Split-Path -Parent $ModelPath))
    return $argList
}
