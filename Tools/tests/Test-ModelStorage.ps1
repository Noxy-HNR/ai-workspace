<#
  Checks for Tools\scripts\ModelStorage.ps1: config validation, the error text for an unreachable model
  server, and which copy of a model loads (server first, pinned models from this PC, local fallback).
  Uses temporary folders and a reserved test address only; it never touches C:\AI\models or the real share.
  Run: powershell -NoProfile -ExecutionPolicy Bypass -File Tools\tests\Test-ModelStorage.ps1
#>
$ErrorActionPreference = 'Stop'
$failures = 0
function Assert-Condition([bool]$Condition, [string]$Message) {
    if ($Condition) { Write-Output "ok   $Message" } else { Write-Output "FAIL $Message"; $script:failures++ }
}

$work = Join-Path ([IO.Path]::GetTempPath()) ("model-storage-test-" + [guid]::NewGuid().ToString('N'))
$fakeRoot = Join-Path $work 'AI'
$scripts = Join-Path (Join-Path $fakeRoot 'Tools') 'scripts'
New-Item -ItemType Directory -Force -Path $scripts, (Join-Path $fakeRoot 'Config') | Out-Null
foreach ($name in @('ModelStorage.ps1', 'Build-LlamaArgs.ps1')) {
    Copy-Item -LiteralPath (Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts') $name) -Destination (Join-Path $scripts $name)
}
$configFile = Join-Path $fakeRoot 'Config\model-storage.json'
$ModelsDir = Join-Path $work 'local'
$serverDir = Join-Path $work 'server'
New-Item -ItemType Directory -Force -Path $ModelsDir, $serverDir, (Join-Path $ModelsDir 'lora'), (Join-Path $serverDir 'lora') | Out-Null

try {
    # Build-LlamaArgs.ps1 dot-sources ModelStorage.ps1, the same way the dashboard and start.ps1 load it.
    . (Join-Path $scripts 'Build-LlamaArgs.ps1')

    # --- configuration ---
    $cfg = Get-ModelStorageConfig
    Assert-Condition (-not $cfg.networkDir -and -not $cfg.error) 'no config file means local models only, without an error'
    Assert-Condition ($cfg.localDir -eq $ModelsDir) 'the caller''s $ModelsDir is the local folder'
    Assert-Condition (-not (Get-ModelShareStatus).configured) 'no config file reports no model server'

    Set-Content -LiteralPath $configFile -Value '{ not json' -Encoding UTF8
    $cfg = Get-ModelStorageConfig
    Assert-Condition ($cfg.error -match 'could not be read' -and -not $cfg.networkDir) 'unreadable config falls back to local with an error'

    Set-Content -LiteralPath $configFile -Value '{"networkDir":"M:\\Models"}' -Encoding UTF8
    $cfg = Get-ModelStorageConfig
    Assert-Condition ($cfg.error -match 'must be a UNC path' -and -not $cfg.networkDir) 'a mapped drive letter is refused with a clear error'

    Set-Content -LiteralPath $configFile -Value '{"networkDir":"\\\\192.168.1.241\\ModelFiles\\Models\\"}' -Encoding UTF8
    $cfg = Get-ModelStorageConfig
    Assert-Condition ($cfg.networkDir -eq '\\192.168.1.241\ModelFiles\Models' -and $cfg.networkHost -eq '192.168.1.241') 'UNC path is accepted, trailing slash trimmed, host derived'

    # --- unreachable server: fails within the probe limit and says why ---
    # 192.0.2.0/24 is reserved for documentation and never routes.
    Set-Content -LiteralPath $configFile -Value '{"networkDir":"\\\\192.0.2.1\\ModelFiles\\Models"}' -Encoding UTF8
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $status = Get-ModelShareStatus -Refresh
    $timer.Stop()
    Assert-Condition ($status.configured -and -not $status.up) 'an unreachable server is reported as configured but offline'
    Assert-Condition ($status.message -match '192\.0\.2\.1' -and $status.message -match '445') "the offline message names the host and SMB port: $($status.message)"
    Assert-Condition ($timer.Elapsed.TotalSeconds -lt 5) "the probe gives up quickly ($([math]::Round($timer.Elapsed.TotalSeconds,1)) s)"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    [void](Get-ModelShareStatus)
    Assert-Condition ($timer.Elapsed.TotalMilliseconds -lt 200) 'a second check within the probe interval does not probe again'

    Set-Content -LiteralPath (Join-Path $ModelsDir 'local-only.gguf') -Value 'x'
    Assert-Condition ((Resolve-ModelFilePath -FileName 'local-only.gguf') -eq (Join-Path $ModelsDir 'local-only.gguf')) 'local models still resolve while the server is offline'
    $msg = Get-ModelNotFoundMessage -FileName 'server-only.gguf'
    Assert-Condition ($msg -match "can't be reached" -and $msg -match 'server-only\.gguf' -and $msg -match 'still load') "missing model while offline explains the server is down: $msg"

    Set-Content -LiteralPath $configFile -Value '{"networkDir":"\\\\192.0.2.1\\ModelFiles\\Models","pinnedLocal":["local-only.gguf"]}' -Encoding UTF8
    $script:ModelShareProbe = @{ at = [datetime]::MinValue; status = $null }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $p = Resolve-ModelFilePath -FileName 'local-only.gguf'
    $timer.Stop()
    Assert-Condition ($p -eq (Join-Path $ModelsDir 'local-only.gguf') -and $timer.Elapsed.TotalMilliseconds -lt 500 -and $null -eq $script:ModelShareProbe.status) 'a pinned local model resolves without probing the share'

    Set-Content -LiteralPath $configFile -Value '{"networkDir":"\\\\no-such-host.invalid\\ModelFiles"}' -Encoding UTF8
    $status = Get-ModelShareStatus -Refresh
    Assert-Condition (-not $status.up -and $status.message -match 'no-such-host\.invalid') "an unknown server name is reported by name: $($status.message)"

    # --- load order with the server online (the share is a temp folder here) ---
    function Get-ModelStorageConfig { [ordered]@{ localDir = $ModelsDir; networkDir = $serverDir; networkHost = 'test'; pinnedLocal = @('pinned.gguf'); error = '' } }
    function Get-ModelShareStatus { param([switch]$Refresh) @{ configured = $true; up = $true; networkDir = $serverDir; message = '' } }
    foreach ($name in @('both.gguf', 'pinned.gguf')) {
        Set-Content -LiteralPath (Join-Path $ModelsDir $name) -Value 'x'
        Set-Content -LiteralPath (Join-Path $serverDir $name) -Value 'x'
    }
    Set-Content -LiteralPath (Join-Path $serverDir 'server-only.gguf') -Value 'x'
    Set-Content -LiteralPath (Join-Path $serverDir 'lora\adapter.gguf') -Value 'x'
    foreach ($f in Get-ChildItem -LiteralPath $serverDir -Recurse -File) { $f.LastWriteTime = (Get-Date).AddHours(-1) }

    Assert-Condition ((Resolve-ModelFilePath -FileName 'both.gguf') -eq (Join-Path $serverDir 'both.gguf')) 'an unpinned model loads from the server'
    Assert-Condition ((Resolve-ModelFilePath -FileName 'pinned.gguf') -eq (Join-Path $ModelsDir 'pinned.gguf')) 'a pinned model loads from this PC'
    Assert-Condition ((Resolve-ModelFilePath -FileName 'local-only.gguf') -eq (Join-Path $ModelsDir 'local-only.gguf')) 'a model only on this PC still loads from this PC'
    Assert-Condition ((Resolve-ModelFilePath -FileName 'server-only.gguf') -eq (Join-Path $serverDir 'server-only.gguf')) 'a model only on the server loads from the server'
    Assert-Condition ((Resolve-ModelFilePath -FileName 'adapter.gguf' -Kind lora) -eq (Join-Path $serverDir 'lora\adapter.gguf')) 'a LoRA on the server is found in its lora folder'
    Assert-Condition ($null -eq (Resolve-ModelFilePath -FileName '..\both.gguf')) 'a path instead of a file name is refused'
    Assert-Condition ((Get-ModelNotFoundMessage -FileName 'nowhere.gguf') -match 'this PC .* or the model server') 'a model in neither place names both folders'

    # A bare --mmproj name follows its model: next to a server model first, then this PC, then the server.
    Set-Content -LiteralPath (Join-Path $serverDir 'mmproj-a.gguf') -Value 'x'
    Set-Content -LiteralPath (Join-Path $ModelsDir 'mmproj-b.gguf') -Value 'x'
    $settings = [pscustomobject]@{ contextSize = 4096; gpuLayers = 99; parallel = 1; batchSize = 512; ubatchSize = 256; threads = 4
        cacheTypeK = 'f16'; cacheTypeV = 'f16'; flashAttn = $true; jinja = $true; extraArgs = '--mmproj mmproj-a.gguf' }
    $argList = @(Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings)
    Assert-Condition ($argList[[array]::IndexOf($argList, '--mmproj') + 1] -eq (Join-Path $serverDir 'mmproj-a.gguf')) 'a projector next to a server model is taken from the server'
    $settings.extraArgs = '--mmproj C:\old\place\mmproj-b.gguf'
    $argList = @(Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings)
    Assert-Condition ($argList[[array]::IndexOf($argList, '--mmproj') + 1] -eq (Join-Path $ModelsDir 'mmproj-b.gguf')) 'a projector path that moved is found by its file name'
    $settings.extraArgs = '--mmproj missing.gguf'
    $threw = $false
    try { [void](Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings) } catch { $threw = $_.Exception.Message -match 'missing\.gguf' }
    Assert-Condition $threw 'a missing projector stops the load with its name'
    $settings.extraArgs = ''
    $argList = @(Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings)
    Assert-Condition (($argList -join ' ') -match '--load-mode none' -and ($argList -join ' ') -match '--cache-ram 0') 'a model with no saved profile still gets the RAM-safety flags'
    $settings.extraArgs = '--load-mode mmap --cache-ram=8192 -lm=mmap --jinja'
    $argList = @(Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings)
    $line = $argList -join ' '
    Assert-Condition ($line -notmatch 'mmap|8192' -and @($argList | Where-Object { $_ -in '--load-mode', '--cache-ram' }).Count -eq 2 -and $line -match '--load-mode none --cache-ram 0') "unsafe saved RAM flags are replaced by the safe ones: $line"
    $settings.extraArgs = '--mmproj=mmproj-a.gguf'
    $argList = @(Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings)
    Assert-Condition ($argList[[array]::IndexOf($argList, '--mmproj') + 1] -eq (Join-Path $serverDir 'mmproj-a.gguf')) 'the --mmproj=name form is resolved too'
    $settings.extraArgs = '--mmproj=nowhere.gguf'
    $threw = $false
    try { [void](Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings) } catch { $threw = $_.Exception.Message -match 'nowhere\.gguf' }
    Assert-Condition $threw 'a missing --mmproj=name projector stops the load'
    $settings.extraArgs = '--jinja --mmproj'
    $threw = $false
    try { [void](Build-LlamaArgs -ModelPath (Join-Path $serverDir 'both.gguf') -Settings $settings) } catch { $threw = $_.Exception.Message -match 'needs a projector' }
    Assert-Condition $threw 'a --mmproj with no value stops the load'
    $line = ConvertTo-LlamaArgumentLine @('--model', '\\server\Model Files\m.gguf', '-ot', 'blk\.(0|1)\.ffn=CPU', '--alias', 'say "hi"', 'C:\dir with space\')
    Assert-Condition ($line -eq '--model "\\server\Model Files\m.gguf" -ot blk\.(0|1)\.ffn=CPU --alias "say \"hi\"" "C:\dir with space\\"') "arguments with spaces and quotes are quoted for the child process: $line"

    Set-Content -LiteralPath (Join-Path $serverDir 'still-copying.gguf') -Value 'x'
    Set-Content -LiteralPath (Join-Path $ModelsDir 'still-copying.gguf') -Value 'x'
    Assert-Condition ((Resolve-ModelFilePath -FileName 'still-copying.gguf') -eq (Join-Path $ModelsDir 'still-copying.gguf')) 'a server copy still being written is skipped in favour of the local copy'
    Remove-Item -LiteralPath (Join-Path $ModelsDir 'still-copying.gguf')
    Assert-Condition ($null -eq (Resolve-ModelFilePath -FileName 'still-copying.gguf')) 'a server copy still being written does not load'
    $files = @(Get-ModelStorageFiles)
    $byName = @{}; foreach ($f in $files) { $byName[$f.Name] = $f }
    Assert-Condition ($byName['both.gguf'].Location -eq 'network' -and $byName['both.gguf'].HasLocal) 'the listing shows one entry per name, from the copy that would load'
    Assert-Condition ($byName['pinned.gguf'].Location -eq 'local' -and $byName['pinned.gguf'].Pinned) 'the listing marks pinned models as local'
    Assert-Condition (-not $byName.ContainsKey('still-copying.gguf')) 'a server file written in the last minute is not listed yet'
}
finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures) { Write-Output "$failures check(s) failed."; exit 1 }
Write-Output 'All model storage checks passed.'
