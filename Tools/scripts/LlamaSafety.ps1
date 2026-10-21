# Shared launch checks. This file never starts or stops a model by itself.

function Get-LlamaDraftArchitecture {
    # Identity only, never tensor loading. Architecture must occur in bounded
    # leading metadata; ambiguous/unknown/truncated files fail closed.
    param([string]$Path)
    $stream = $null
    try {
        $stream = [IO.File]::OpenRead($Path)
        $reader = New-Object IO.BinaryReader($stream)
        if ($stream.Length -lt 24 -or [Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -cne 'GGUF') { throw 'Draft companion is not a GGUF file.' }
        $version = $reader.ReadUInt32()
        if ($version -notin @(2,3)) { throw 'Draft companion requires GGUF version 2 or 3.' }
        [void]$reader.ReadUInt64()
        $count = $reader.ReadUInt64()
        if ($count -lt 1 -or $count -gt 65536) { throw 'Draft companion has invalid GGUF metadata.' }
        $fixed = @{0=1;1=1;2=2;3=2;4=4;5=4;6=4;7=1;10=8;11=8;12=8}
        $utf8 = New-Object Text.UTF8Encoding($false,$true)
        function Read-DraftString {
            param([int]$Maximum)
            $length = $reader.ReadUInt64()
            if ($length -gt $Maximum -or $length -gt [uint64]($stream.Length-$stream.Position) -or
                $stream.Position -gt 4MB -or $length -gt [uint64](4MB-$stream.Position)) { throw 'Draft GGUF string is truncated or too large.' }
            return $utf8.GetString($reader.ReadBytes([int]$length))
        }
        function Skip-DraftBytes {
            param([long]$Length)
            if ($Length -lt 0 -or $Length -gt ($stream.Length-$stream.Position) -or
                $stream.Position -gt 4MB -or $Length -gt (4MB-$stream.Position)) { throw 'Draft GGUF metadata is truncated or exceeds its identity budget.' }
            [void]$stream.Seek($Length,[IO.SeekOrigin]::Current)
        }
        for ($index=0; $index -lt [Math]::Min([long]$count,128); $index++) {
            if ($stream.Position -gt 4MB) { throw 'Draft identity exceeds bounded leading metadata.' }
            $name = Read-DraftString 4096
            $type = $reader.ReadUInt32()
            if ($name -ceq 'general.architecture') {
                if ($type -ne 8) { throw 'Draft architecture must be a GGUF string.' }
                $architecture = Read-DraftString 64
                if ($architecture -cne 'dflash') { throw 'Only dflash companion architecture is supported.' }
                return $architecture
            }
            if ($type -eq 8) { [void](Read-DraftString 1MB) }
            elseif ($fixed.ContainsKey([int]$type)) { Skip-DraftBytes $fixed[[int]$type] }
            elseif ($type -eq 9) {
                $itemType = [int]$reader.ReadUInt32(); $items=$reader.ReadUInt64()
                if ($items -gt 4096) { throw 'Draft architecture must precede large metadata arrays.' }
                if ($fixed.ContainsKey($itemType)) { Skip-DraftBytes ([long]$items*$fixed[$itemType]) }
                elseif ($itemType -eq 8) { for ($item=0;$item -lt $items;$item++) { [void](Read-DraftString 1MB) } }
                else { throw 'Unsupported leading draft metadata array.' }
            }
            else { throw 'Unsupported leading draft GGUF metadata.' }
        }
        throw 'Draft companion has no supported leading architecture metadata.'
    } finally { if ($stream) { $stream.Dispose() } }
}

function Resolve-LlamaDraftFile {
    param($DraftFile,[string]$ModelsDir)
    if ($null -eq $DraftFile -or ($DraftFile -is [string] -and [string]::IsNullOrWhiteSpace($DraftFile))) { return $null }
    if ($DraftFile -isnot [string] -or $DraftFile -match '\.\.' -or
        $DraftFile -notmatch '^(?:drafts[\\/])?([A-Za-z0-9][A-Za-z0-9_.+-]{0,199}\.gguf)$') {
        throw 'draftFile must name a safe ASCII .gguf file under models/drafts; paths, spaces and quotes are not allowed.'
    }
    $name = [string]$Matches[1]
    $draftRoot = [IO.Path]::GetFullPath((Join-Path $ModelsDir 'drafts'))
    $path = [IO.Path]::GetFullPath((Join-Path $draftRoot $name))
    if (-not $path.StartsWith($draftRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Draft companion escaped models/drafts.'
    }
    $current = Get-Item -LiteralPath $path -Force -ErrorAction Stop
    if ($current.PSIsContainer -or $current.Extension -ine '.gguf') { throw 'Draft companion must be a regular GGUF file.' }
    $cursor = $current
    while ($cursor) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Draft companion path cannot contain symlinks or reparse points.' }
        $cursor = if ($cursor -is [IO.FileInfo]) { $cursor.Directory } else { $cursor.Parent }
    }
    if ($current.Length -lt 24) { throw 'Draft companion GGUF header is missing or truncated.' }
    [void](Get-LlamaDraftArchitecture -Path $path)
    return $path
}

function Get-LlamaPhysicalMemory {
    if (-not ('LocalLlama.MemoryApi' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace LocalLlama {
    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryStatus {
        public uint length, load;
        public ulong totalPhysical, availablePhysical, totalPageFile, availablePageFile;
        public ulong totalVirtual, availableVirtual, availableExtendedVirtual;
    }
    public static class MemoryApi {
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
    }
}
'@ -ErrorAction Stop
    }
    $status = New-Object LocalLlama.MemoryStatus
    $status.length = [Runtime.InteropServices.Marshal]::SizeOf($status)
    if (-not [LocalLlama.MemoryApi]::GlobalMemoryStatusEx([ref]$status)) {
        throw 'Windows available physical memory could not be read.'
    }
    return @{
        availableBytes = $status.availablePhysical
        totalBytes = $status.totalPhysical
        availableCommitBytes = $status.availablePageFile
    }
}

function Get-LlamaProfileRamMinimumBytes {
    param($Value)
    if ($null -eq $Value -or ($Value -is [string] -and [string]::IsNullOrWhiteSpace($Value))) { return [long]0 }
    if ($Value -is [bool] -or $Value -is [string]) { throw 'The measured RAM minimum must be a numeric GiB value.' }
    try { $gib = [double]$Value } catch { throw 'The measured RAM minimum must be a numeric GiB value.' }
    if ([double]::IsNaN($gib) -or [double]::IsInfinity($gib) -or $gib -lt 0 -or $gib -gt 1024) {
        throw 'The measured RAM minimum must be finite and between 0 and 1024 GiB.'
    }
    return [long][Math]::Ceiling($gib * 1GB)
}

function Get-LlamaCpuMoeLayers {
    param($Value, [int]$BlockCount = 0)
    if ($null -eq $Value -or ($Value -is [string] -and [string]::IsNullOrWhiteSpace($Value))) { return 0 }
    if ($Value -is [bool] -or $Value -is [string]) { throw 'CPU expert layers must be a numeric integer.' }
    try { $count = [double]$Value } catch { throw 'CPU expert layers must be a numeric integer.' }
    if ([double]::IsNaN($count) -or [double]::IsInfinity($count) -or $count -lt 0 -or $count -gt 1024 -or [Math]::Floor($count) -ne $count) {
        throw 'CPU expert layers must be an integer between 0 and 1024.'
    }
    if ($BlockCount -gt 0 -and $count -gt $BlockCount) { throw 'CPU expert layers exceed the model block count.' }
    return [int]$count
}

function Get-LlamaHostPlacement {
    # What the placement flags in extraArgs keep in system RAM, for the admission check. extraArgs must
    # already have passed Get-LlamaSafeExtraArgs.
    #   cpuMoeLayers  --n-cpu-moe N: the experts of the first N blocks.
    #   allCpuMoe     --cpu-moe: every block's experts.
    #   cpuFfnLayers  the tuner's own -ot rule: the FFN weights of the listed blocks.
    #   unknownCpuOverride  any other -ot rule that sends tensors to a CPU buffer. Its size cannot be read
    #                 from the pattern, so admission charges it as the whole model.
    param([string]$ExtraArgs)
    # The one -ot rule Model Tuner writes (Config.n_cpu_ffn, kept in step with _FFN_OT in modeltuner/knobs.py).
    $tunerFfnRule = '^blk\\\.\(([0-9|]+)\)\\\.ffn_\(up\|gate\|down\)\\\.weight$'
    $placement = @{ cpuMoeLayers = 0; allCpuMoe = $false; cpuFfnLayers = @(); unknownCpuOverride = $false }
    $tokens = @([regex]::Matches($ExtraArgs, '"[^"]*"|''[^'']*''|\S+') | ForEach-Object { $_.Value.Trim('"').Trim("'") })
    $ffn = @{}
    for ($i = 0; $i -lt $tokens.Count; $i++) {
        $parts = $tokens[$i] -split '=', 2
        $flag = $parts[0]
        if ($flag -in @('-cmoe', '--cpu-moe')) { $placement.allCpuMoe = $true; continue }
        if ($flag -notin @('-ncmoe', '--n-cpu-moe', '-ot', '--override-tensor')) { continue }
        $value = if ($parts.Count -eq 2) { $parts[1] } elseif ($i + 1 -lt $tokens.Count) { $i++; $tokens[$i] } else { '' }
        if ($flag -in @('-ncmoe', '--n-cpu-moe')) {
            $placement.cpuMoeLayers = [Math]::Max($placement.cpuMoeLayers, [int]$value)
            continue
        }
        foreach ($rule in ($value -split ',')) {
            $pattern, $buffer = $rule -split '=', 2
            if ($buffer -notmatch '^CPU') { continue }
            if ($pattern -match $tunerFfnRule) {
                foreach ($block in ($Matches[1] -split '\|')) { $ffn[[int]$block] = $true }
            } else { $placement.unknownCpuOverride = $true }
        }
    }
    $placement.cpuFfnLayers = @($ffn.Keys | Sort-Object)
    return $placement
}

function Get-LlamaMemoryAdmission {
    param(
        [long]$ModelBytes,
        [int]$BlockCount,
        [int]$GpuLayers,
        [long]$AvailableBytes,
        [long]$AdapterBytes = 0,
        [long]$MinimumAvailableBytes = 0,
        [int]$CpuMoeLayers = 0,
        [int[]]$CpuFfnLayers = @(),
        [switch]$AllCpuMoe,
        [switch]$UnknownCpuOverride
    )
    # File-size proportion is an estimate, not a measured allocation. Unknown
    # partial-offload metadata is charged as a full CPU model. Keep 1 GiB for
    # the user's apps plus 0.75 GiB for runtime/embedding buffers, with a 3 GiB floor.
    if ($GpuLayers -ge 999 -or $GpuLayers -eq -1) { $cpuFraction = 0.0 }
    elseif ($BlockCount -gt 0) { $cpuFraction = 1.0 - [Math]::Min([Math]::Max($GpuLayers, 0), $BlockCount) / [double]$BlockCount }
    else { $cpuFraction = 1.0 }
    [void](Get-LlamaCpuMoeLayers -Value $CpuMoeLayers -BlockCount $BlockCount)
    $CpuFfnLayers = @($CpuFfnLayers | Sort-Object -Unique)
    if ($CpuMoeLayers -gt 0 -or $CpuFfnLayers.Count -gt 0) {
        if ($BlockCount -le 0) { throw 'CPU expert or FFN placement needs known model block metadata.' }
        if (@($CpuFfnLayers | Where-Object { $_ -lt 0 -or $_ -ge $BlockCount }).Count) { throw 'The -ot FFN rule names a block the model does not have.' }
        # Charge whole layer proportions for host experts and FFN weights. Adding the fractions
        # deliberately overestimates any overlap with already-CPU blocks or with each other.
        $cpuFraction = [Math]::Min(1.0, $cpuFraction + ($CpuMoeLayers + $CpuFfnLayers.Count) / [double]$BlockCount)
    }
    # Every expert, or an -ot rule whose size the pattern does not show, is charged as the whole model.
    if ($AllCpuMoe -or $UnknownCpuOverride) { $cpuFraction = 1.0 }
    $cpuBytes = [long][Math]::Ceiling($ModelBytes * $cpuFraction) + $AdapterBytes
    if ($MinimumAvailableBytes -lt 0) { throw 'The saved RAM minimum cannot be negative.' }
    $requiredBytes = [long][Math]::Max([Math]::Max(3GB, $cpuBytes + 1.75GB), $MinimumAvailableBytes)
    $allowed = $ModelBytes -gt 0 -and $AvailableBytes -ge $requiredBytes
    return @{
        allowed = $allowed
        availableGiB = [Math]::Round($AvailableBytes / 1GB, 2)
        requiredGiB = [Math]::Round($requiredBytes / 1GB, 2)
        estimatedCpuWeightsGiB = [Math]::Round($cpuBytes / 1GB, 2)
        blockCount = $BlockCount
        gpuLayers = $GpuLayers
        cpuMoeLayers = $CpuMoeLayers
        cpuFfnLayers = $CpuFfnLayers.Count
        allCpuMoe = [bool]$AllCpuMoe
        unknownCpuOverride = [bool]$UnknownCpuOverride
        requiredBytes = $requiredBytes
        profileMinimumGiB = [Math]::Round($MinimumAvailableBytes / 1GB, 2)
        basis = 'Estimated CPU weights (including -ot and CPU expert placement) + 0.75 GiB runtime + 1 GiB reserve; at least 3 GiB and any saved measured RAM minimum.'
    }
}

function Resolve-LlamaServerBinary {
    param([string]$Root, [string]$DefaultBinary, $Settings)
    $custom = [string]$Settings.serverBinary
    $bin = $DefaultBinary
    if (-not [string]::IsNullOrWhiteSpace($custom)) {
        $bin = if ([IO.Path]::IsPathRooted($custom)) { $custom } else { Join-Path $Root $custom }
    }
    if (-not (Test-Path -LiteralPath $bin -PathType Leaf)) { throw "llama-server.exe not found at $bin" }
    return [IO.Path]::GetFullPath($bin)
}

function Get-LlamaSafeExtraArgs {
    param([string]$ExtraArgs)
    $result = @()
    $tokens = @([regex]::Matches($ExtraArgs, '"[^"]*"|''[^'']*''|\S+') | ForEach-Object { $_.Value.Trim('"').Trim("'") })
    for ($i = 0; $i -lt $tokens.Count; $i++) {
        $parts = $tokens[$i] -split '=', 2
        $flag = $parts[0]
        if ($flag -in @('--load-mode', '-lm', '--cache-ram', '-cram')) {
            $value = if ($parts.Count -eq 2) { $parts[1] } elseif ($i + 1 -lt $tokens.Count) { $i++; $tokens[$i] } else { '' }
            if (($flag -in @('--load-mode', '-lm') -and $value -cne 'none') -or
                ($flag -in @('--cache-ram', '-cram') -and $value -ne '0')) {
                throw 'Model loading requires --load-mode none and --cache-ram 0.'
            }
            continue
        }
        # Host placement is allowed: Model Tuner writes these, and the admission check charges what they
        # keep in system RAM (Get-LlamaHostPlacement). Only the syntax is checked here.
        if ($flag -in @('-ncmoe', '--n-cpu-moe', '-ot', '--override-tensor')) {
            $value = if ($parts.Count -eq 2) { $parts[1] } elseif ($i + 1 -lt $tokens.Count) { $i++; $tokens[$i] } else { '' }
            if ($flag -in @('-ncmoe', '--n-cpu-moe')) {
                if ($value -notmatch '^\d{1,4}$') { throw "$flag needs a whole number of layers." }
                [void](Get-LlamaCpuMoeLayers -Value ([int]$value))
            } else {
                foreach ($rule in ($value -split ',')) {
                    $pattern, $buffer = $rule -split '=', 2
                    if ([string]::IsNullOrWhiteSpace($pattern) -or [string]::IsNullOrWhiteSpace($buffer)) {
                        throw "$flag needs pattern=buffer rules, such as blk\.(0|1)\.ffn_(up|gate|down)\.weight=CPU."
                    }
                    try { [void][regex]::new($pattern) } catch { throw "$flag has an invalid tensor pattern: $pattern" }
                }
            }
            $result += @($flag, $value)
            continue
        }
        if ($flag -in @('-m', '--model', '-md', '--model-draft', '--spec-draft-model', '-hf', '-hfr', '--hf-repo', '--hf-file', '--host', '--port', '-c', '--ctx-size', '-ngl', '--n-gpu-layers', '--gpu-layers', '-np', '--parallel', '-nkvo', '--no-kv-offload', '-dev', '--device')) {
            throw "Set model, context, GPU layers and host/port through the profile fields, not extraArgs ($flag)."
        }
        $result += $tokens[$i]
    }
    return @($result) + @('--load-mode', 'none', '--cache-ram', '0')
}
