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
    $out = @()
    for ($i = 0; $i -lt $ArgList.Count; $i++) {
        $flag, $inline = ([string]$ArgList[$i] -split '=', 2)
        if ($flag -notin @('--mmproj', '-mm')) { $out += $ArgList[$i]; continue }
        if ($null -ne $inline) { $value = $inline }
        elseif ($i + 1 -lt $ArgList.Count) { $i++; $value = [string]$ArgList[$i] }
        else { $value = '' }
        if ([string]::IsNullOrWhiteSpace($value)) { throw "$flag needs a projector file name." }
        if ([IO.Path]::IsPathRooted($value) -and (Test-Path -LiteralPath $value -PathType Leaf)) { $out += @($flag, $value); continue }
        $leaf = Split-Path -Leaf $value
        $resolved = $null
        if ($ModelDir -and (Test-Path -LiteralPath (Join-Path $ModelDir $leaf) -PathType Leaf)) { $resolved = Join-Path $ModelDir $leaf }
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
        # RAM safety (AGENTS.md): memory-mapped loading and the default prompt cache once exhausted RAM, so
        # any saved value for these flags is dropped and the safe values below always win.
        $safe = @()
        for ($i = 0; $i -lt $extra.Count; $i++) {
            $flag, $inline = ([string]$extra[$i] -split '=', 2)
            if ($flag -in @('--load-mode', '-lm', '--cache-ram', '-cram')) {
                if ($null -eq $inline -and $i + 1 -lt $extra.Count -and $extra[$i + 1] -notlike '-*') { $i++ }
                continue
            }
            $safe += $extra[$i]
        }
        $argList += @(Resolve-LlamaProjectorArgs -ArgList $safe -ModelDir (Split-Path -Parent $ModelPath))
    }
    # Also for a model with no saved profile yet (e.g. one just added to the model server).
    $argList += @('--load-mode', 'none', '--cache-ram', '0')
    return $argList
}
