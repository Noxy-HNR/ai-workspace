# Offline regression checks: model headers are tiny fixtures and every process
# launch/termination below is replaced with an in-process test double.
$ErrorActionPreference = 'Stop'
$stackRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $stackRoot 'Tools\scripts\Build-LlamaArgs.ps1')
$script:Checks = 0
function Assert-Check { param([bool]$Condition, [string]$Label) if (-not $Condition) { throw "FAILED: $Label" }; $script:Checks++ }
function Assert-Throws { param([scriptblock]$Action, [string]$Label) $thrown=$false; try { & $Action | Out-Null } catch { $thrown=$true }; Assert-Check $thrown $Label }

$full = Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 3.5GB
Assert-Check ($full.allowed -and $full.requiredBytes -eq 3GB) 'fully GPU model has a physical RAM floor'
$measuredBytes = Get-LlamaProfileRamMinimumBytes -Value 4.25
Assert-Check ($measuredBytes -eq 4.25GB) 'measured profile floor converts GiB to bytes'
$measured = Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 4GB -MinimumAvailableBytes $measuredBytes
Assert-Check (-not $measured.allowed -and $measured.requiredBytes -eq 4.25GB) 'measured checkpoint RAM is included in admission'
Assert-Check ((Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 3.5GB -MinimumAvailableBytes 0).requiredBytes -eq 3GB) 'zero measured floor cannot lower baseline safety'
Assert-Check ((Get-LlamaProfileRamMinimumBytes -Value $null) -eq 0) 'old profiles do not require a new field'
foreach ($invalidMinimum in @($true,'4.25',-1,[double]::NaN,[double]::PositiveInfinity,2048)) {
    Assert-Throws { Get-LlamaProfileRamMinimumBytes -Value $invalidMinimum } 'invalid measured RAM minimum is refused'
}
Assert-Throws { Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 3.5GB -MinimumAvailableBytes -1 } 'negative byte minimum is refused'
$partial = Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 62 -GpuLayers 37 -AvailableBytes 4GB
Assert-Check (-not $partial.allowed -and $partial.estimatedCpuWeightsGiB -gt 6) 'partial offload includes CPU weights'
$unknown = Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 0 -GpuLayers 37 -AvailableBytes 10GB
Assert-Check (-not $unknown.allowed -and $unknown.estimatedCpuWeightsGiB -eq 16) 'missing partial metadata is conservative'
Assert-Check (-not (Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 0).allowed) 'unknown or zero available RAM refuses loading'
Assert-Check (Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 62 -GpuLayers 37 -AvailableBytes 10GB).allowed 'sufficient RAM admits a partial offload'
Assert-Check (-not (Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 3GB -AdapterBytes 2GB).allowed) 'adapter host allocation is included'
Assert-Check ((Get-LlamaCpuMoeLayers -Value $null) -eq 0) 'old profiles keep normal expert placement'
foreach ($invalidCpuMoe in @($true,'8',-1,1.5,[double]::NaN,[double]::PositiveInfinity,2048)) {
    Assert-Throws { Get-LlamaCpuMoeLayers -Value $invalidCpuMoe } 'invalid CPU expert count is refused'
}
Assert-Throws { Get-LlamaCpuMoeLayers -Value 65 -BlockCount 64 } 'CPU expert count is bounded by model blocks'
Assert-Throws { Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 0 -GpuLayers 999 -AvailableBytes 8GB -CpuMoeLayers 8 } 'unknown CPU expert allocation fails closed'
$experts=Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 3.5GB -CpuMoeLayers 8
Assert-Check (-not $experts.allowed -and $experts.requiredBytes -eq 3.75GB) 'CPU experts are charged host RAM despite fully GPU attention'
Assert-Check ((Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 64 -GpuLayers 32 -AvailableBytes 16GB -CpuMoeLayers 16).estimatedCpuWeightsGiB -eq 12) 'partial blocks plus CPU experts use a conservative combined bound'

$safe = @(Get-LlamaSafeExtraArgs '--load-mode=none --cache-ram 0 --load-mode none')
Assert-Check (($safe -join ' ') -eq '--load-mode none --cache-ram 0') 'duplicate safe arguments normalize once'
foreach ($bad in @('--cache-ram 8192', '--cache-ram=-1', '-cram 8', '--load-mode mmap', '-lm auto', '--load-mode mmap --load-mode none', '--cache-ram', '--ctx-size 1000000', '--model other.gguf', '--no-kv-offload', '--device none',
        '--n-cpu-moe', '--n-cpu-moe eight', '--n-cpu-moe -1', '--n-cpu-moe 1.5', '-ncmoe=99999', '-ot', '--override-tensor .*', '--override-tensor =CPU', '-ot blk\.([0-9]\.ffn=CPU')) {
    Assert-Throws { Get-LlamaSafeExtraArgs $bad } "unsafe or malformed override refused: $bad"
}

# Host placement that Model Tuner writes (modeltuner/knobs.py build_extra_args) is allowed through and charged.
$tunerExtra = '--n-cpu-moe 8 -ot blk\.(0|1|2)\.ffn_(up|gate|down)\.weight=CPU --load-mode none --cache-ram 0'
$tunerArgs = @(Get-LlamaSafeExtraArgs $tunerExtra)
Assert-Check (($tunerArgs -join ' ') -eq '--n-cpu-moe 8 -ot blk\.(0|1|2)\.ffn_(up|gate|down)\.weight=CPU --load-mode none --cache-ram 0') 'tuner -ot and --n-cpu-moe pass through unchanged'
Assert-Check ((@(Get-LlamaSafeExtraArgs '-ncmoe=4 --cpu-moe --override-tensor=exps=CPU') -join ' ') -eq '-ncmoe 4 --cpu-moe --override-tensor exps=CPU --load-mode none --cache-ram 0') 'aliases and = forms are accepted'
$tunerPlacement = Get-LlamaHostPlacement $tunerExtra
Assert-Check ($tunerPlacement.cpuMoeLayers -eq 8 -and (($tunerPlacement.cpuFfnLayers -join ',') -eq '0,1,2') -and -not $tunerPlacement.unknownCpuOverride -and -not $tunerPlacement.allCpuMoe) 'tuner placement is read back exactly'
$none = Get-LlamaHostPlacement '--spec-type ngram-mod --load-mode none --cache-ram 0'
Assert-Check ($none.cpuMoeLayers -eq 0 -and $none.cpuFfnLayers.Count -eq 0 -and -not $none.unknownCpuOverride -and -not $none.allCpuMoe) 'no placement flags charge nothing extra'
Assert-Check ((Get-LlamaHostPlacement '-ot blk\.1[0-9]\.ffn_.*=CPU').unknownCpuOverride) 'hand-written CPU -ot rule is flagged as unknown size'
Assert-Check ((Get-LlamaHostPlacement '-ot blk\.(0|1)\.ffn_(up|gate|down)\.weight=CPU,token_embd=CPU_REPACK').unknownCpuOverride) 'any extra CPU rule in a list is flagged'
Assert-Check (-not (Get-LlamaHostPlacement '-ot blk\.1[0-9]\.attn.*=CUDA0').unknownCpuOverride) 'rules that target a GPU buffer are not charged as host RAM'
Assert-Check ((Get-LlamaHostPlacement '--cpu-moe').allCpuMoe) '--cpu-moe is flagged as every expert on the CPU'

$ffn = Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 16GB -CpuFfnLayers @(0,1,2,3,4,5,6,7)
Assert-Check ($ffn.allowed -and $ffn.estimatedCpuWeightsGiB -eq 2 -and $ffn.cpuFfnLayers -eq 8) 'tuner FFN blocks are charged as host RAM'
Assert-Check (-not (Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 3.5GB -CpuFfnLayers @(0,1,2,3,4,5,6,7)).allowed) 'FFN placement that does not fit is refused'
Assert-Check ((Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 16GB -CpuFfnLayers @(0,0,1,1)).cpuFfnLayers -eq 2) 'repeated FFN blocks are counted once'
Assert-Throws { Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 16GB -CpuFfnLayers @(64) } 'FFN block outside the model is refused'
Assert-Throws { Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 0 -GpuLayers 999 -AvailableBytes 32GB -CpuFfnLayers @(0) } 'unknown block count with FFN placement fails closed'
$combined = Get-LlamaMemoryAdmission -ModelBytes 16GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 32GB -CpuMoeLayers 8 -CpuFfnLayers @(0,1,2,3,4,5,6,7)
Assert-Check ($combined.estimatedCpuWeightsGiB -eq 4) 'CPU experts and FFN blocks add up conservatively'
$unknownOt = Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 8GB -UnknownCpuOverride
Assert-Check (-not $unknownOt.allowed -and $unknownOt.estimatedCpuWeightsGiB -eq 8) 'unknown CPU -ot rule is charged as the whole model'
Assert-Check (Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 10GB -UnknownCpuOverride).allowed 'unknown CPU -ot rule still loads when the whole model fits'
Assert-Check ((Get-LlamaMemoryAdmission -ModelBytes 8GB -BlockCount 64 -GpuLayers 999 -AvailableBytes 8GB -AllCpuMoe).estimatedCpuWeightsGiB -eq 8) '--cpu-moe is charged as the whole model'
$quoted = @(Get-LlamaSafeExtraArgs '--mmproj "C:\AI\models\a projector.gguf"')
Assert-Check ($quoted[1] -eq 'C:\AI\models\a projector.gguf') 'quoted projector path survives tokenization'
# The native command line must split back into exactly the tokens that were checked.
$line = @(ConvertTo-LlamaCommandLine @('--mmproj', 'C:\AI\models\a projector.gguf', '-ot', 'blk\.(0|1)\.ffn_(up|gate|down)\.weight=CPU', 'C:\dir with space\', ''))
Assert-Check ($line[1] -eq '"C:\AI\models\a projector.gguf"' -and $line[3] -eq 'blk\.(0|1)\.ffn_(up|gate|down)\.weight=CPU' -and $line[4] -eq '"C:\dir with space\\"' -and $line[5] -eq '""') 'arguments with spaces are quoted for the native command line'
foreach ($smuggled in @('"--n-gpu-layers 0"', "'--model other.gguf'", '"--mmproj a.gguf --cache-ram 8192"', 'a"b', '--lora big.gguf', '--lora-scaled big.gguf 0.5', '--lora=big.gguf')) {
    Assert-Throws { Get-LlamaSafeExtraArgs $smuggled } "token that would split or bypass a profile field is refused: $smuggled"
}

# Define dashboard functions without executing its top-level server loop or services.
$tokens=$null; $parseErrors=$null
$ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $stackRoot 'dashboard-server.ps1'), [ref]$tokens, [ref]$parseErrors)
Assert-Check ($parseErrors.Count -eq 0) 'dashboard parses'
$wanted = @('Stop-LlamaNative','Start-LlamaNative','Get-GgufHeader','Get-LlamaModelAdmission','Get-AvailableModels','Get-ModelDefaults','Get-ModelsConfig','Get-ModelSettingsFor','Save-ModelsConfig','Save-ModelConfigOnly','Apply-ModelSettings','Get-ModelPath','Resolve-LoraPath')
foreach ($node in $ast.FindAll({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    if ($node.Name -in $wanted) { . ([scriptblock]::Create($node.Extent.Text)) }
}
$optionalKeysNode = $ast.Find({ param($n) $n -is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$OptionalModelKeys' }, $false)
. ([scriptblock]::Create($optionalKeysNode.Extent.Text))
$scratchRoot = [IO.Path]::GetFullPath((Join-Path $stackRoot 'Scratch'))
$Root = Join-Path $scratchRoot ('llama-safety-' + [guid]::NewGuid().ToString('N'))
$ModelsDir = Join-Path $Root 'models'
$ModelsConfigFile = Join-Path $Root 'models.json'
$LlamaBin = Join-Path $Root 'default\llama-server.exe'
$LlamaPidFile = Join-Path $Root 'llama.pid'
$LogDir = $Root
$script:GgufHeaderCache = @{}
New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null

function Write-TinyGguf {
    param([string]$Path, [string]$Architecture, [int]$ExpertCount = 0)
    $stream = [IO.File]::Create($Path); $writer = New-Object IO.BinaryWriter($stream)
    try {
        $writer.Write([Text.Encoding]::ASCII.GetBytes('GGUF')); $writer.Write([uint32]3); $writer.Write([uint64]0); $writer.Write([uint64](3 + [int]($ExpertCount -gt 0)))
        $tagKey=[Text.Encoding]::UTF8.GetBytes('general.tags'); $writer.Write([uint64]$tagKey.Length); $writer.Write($tagKey); $writer.Write([uint32]9)
        $writer.Write([uint32]8); $writer.Write([uint64]1); $tag=[Text.Encoding]::UTF8.GetBytes('fixture'); $writer.Write([uint64]$tag.Length); $writer.Write($tag)
        foreach ($entry in @(@('general.architecture',8,$Architecture), @("$Architecture.block_count",4,62))) {
            $keyBytes=[Text.Encoding]::UTF8.GetBytes($entry[0]); $writer.Write([uint64]$keyBytes.Length); $writer.Write($keyBytes); $writer.Write([uint32]$entry[1])
            if ($entry[1] -eq 8) { $valueBytes=[Text.Encoding]::UTF8.GetBytes($entry[2]); $writer.Write([uint64]$valueBytes.Length); $writer.Write($valueBytes) } else { $writer.Write([uint32]$entry[2]) }
        }
        if ($ExpertCount -gt 0) {
            $keyBytes=[Text.Encoding]::UTF8.GetBytes("$Architecture.expert_count")
            $writer.Write([uint64]$keyBytes.Length);$writer.Write($keyBytes);$writer.Write([uint32]4);$writer.Write([uint32]$ExpertCount)
        }
    } finally { $writer.Dispose(); $stream.Dispose() }
}

try {
    foreach ($dir in 'default','custom') { New-Item -ItemType Directory -Path (Join-Path $Root $dir) -Force | Out-Null }
    [IO.File]::WriteAllText($LlamaBin, 'fixture, never executed')
    $customBin = Join-Path $Root 'custom\llama-server.exe'; [IO.File]::WriteAllText($customBin, 'fixture, never executed')
    $renamedBin = Join-Path $Root 'custom\fork-server.exe'; [IO.File]::WriteAllText($renamedBin, 'fixture, never executed')
    Write-TinyGguf (Join-Path $ModelsDir 'model.gguf') 'gemma3'
    Write-TinyGguf (Join-Path $ModelsDir 'other.gguf') 'gemma3'
    Write-TinyGguf (Join-Path $ModelsDir 'projector.gguf') 'clip'
    Write-TinyGguf (Join-Path $ModelsDir 'moe.gguf') 'qwen35moe' 128
    $header = Get-GgufHeader (Get-Item (Join-Path $ModelsDir 'model.gguf'))
    Assert-Check ($header['gemma3.block_count'] -eq 62) 'GGUF numeric layer metadata is read'
    $settings = [ordered]@{contextSize=8192;gpuLayers=999;flashAttn=$true;parallel=1;jinja=$true;batchSize=512;ubatchSize=256;threads=8;cacheTypeK='q8_0';cacheTypeV='q4_0';extraArgs='';serverBinary='custom\llama-server.exe'}
    @{activeModel='model.gguf';models=@{'model.gguf'=$settings}} | ConvertTo-Json -Depth 8 | Set-Content $ModelsConfigFile -Encoding UTF8
    $savedInput = [pscustomobject]$settings
    $savedInput | Add-Member -NotePropertyName minimumAvailableRamGiB -NotePropertyValue 4.25
    Assert-Check ((Save-ModelConfigOnly 'model.gguf' $savedInput).success -and (Get-ModelsConfig).models.'model.gguf'.minimumAvailableRamGiB -eq 4.25) 'measured minimum is saved with the complete profile'
    $savedInput.PSObject.Properties.Remove('minimumAvailableRamGiB')
    Assert-Check ((Save-ModelConfigOnly 'model.gguf' $savedInput).success -and (Get-ModelsConfig).models.'model.gguf'.minimumAvailableRamGiB -eq 4.25) 'ordinary form saves preserve an omitted measured minimum'
    $beforeMinimumHash = (Get-FileHash $ModelsConfigFile).Hash
    $savedInput | Add-Member -NotePropertyName minimumAvailableRamGiB -NotePropertyValue 'invalid'
    Assert-Check (-not (Save-ModelConfigOnly 'model.gguf' $savedInput).success -and (Get-FileHash $ModelsConfigFile).Hash -eq $beforeMinimumHash) 'invalid minimum never overwrites saved settings'
    $savedInput.PSObject.Properties.Remove('minimumAvailableRamGiB')
    $measuredProfile = Get-ModelSettingsFor -Config (Get-ModelsConfig) -FileName 'model.gguf'
    Assert-Check ((Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $measuredProfile).requiredBytes -eq 4.25GB) 'dashboard admission enforces the saved measured minimum'
    # Host placement lives in extraArgs, where Model Tuner writes it.
    $settings.extraArgs='--n-cpu-moe 8 --load-mode none --cache-ram 0'
    $built=@(Build-LlamaArgs 'model.gguf' $settings)
    Assert-Check ((($built -join ' ') -split '--n-cpu-moe').Count -eq 2 -and ($built -join ' ') -match '--n-cpu-moe 8') 'tuner --n-cpu-moe reaches llama-server once'
    Assert-Throws { Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $settings } 'dense models reject CPU expert placement'
    Assert-Check ((Get-GgufHeader (Get-Item (Join-Path $ModelsDir 'moe.gguf')))['qwen35moe.expert_count'] -eq 128) 'expert metadata after block count is read'
    $moeAdmission = Get-LlamaModelAdmission -FileName 'moe.gguf' -Settings $settings
    Assert-Check ($moeAdmission.cpuMoeLayers -eq 8) 'real dashboard admission reads --n-cpu-moe from extraArgs'
    $settings.extraArgs='--n-cpu-moe 99 --load-mode none --cache-ram 0'
    Assert-Throws { Get-LlamaModelAdmission -FileName 'moe.gguf' -Settings $settings } 'more CPU expert layers than blocks is refused'
    $settings.extraArgs='-ot blk\.(0|1|2|3)\.ffn_(up|gate|down)\.weight=CPU --load-mode none --cache-ram 0'
    $built=@(Build-LlamaArgs 'model.gguf' $settings)
    Assert-Check ([Array]::IndexOf($built,'-ot') -ge 0 -and $built[[Array]::IndexOf($built,'-ot')+1] -eq 'blk\.(0|1|2|3)\.ffn_(up|gate|down)\.weight=CPU') 'tuner -ot rule reaches llama-server intact'
    Assert-Check ((Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $settings).cpuFfnLayers -eq 4) 'dense model admission charges the tuner FFN blocks'
    $settings.extraArgs='-ot blk\.(70)\.ffn_(up|gate|down)\.weight=CPU --load-mode none --cache-ram 0'
    Assert-Throws { Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $settings } 'FFN rule naming a missing block is refused'
    $settings.extraArgs='-ot exps=CPU --load-mode none --cache-ram 0'
    Assert-Check ((Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $settings).unknownCpuOverride) 'hand-written -ot is charged as the whole model'
    $settings.extraArgs=''
    $savedInput.extraArgs='--n-cpu-moe 8 -ot blk\.(0|1)\.ffn_(up|gate|down)\.weight=CPU'
    Assert-Check ((Save-ModelConfigOnly 'model.gguf' $savedInput).success -and (Get-ModelsConfig).models.'model.gguf'.extraArgs -eq '--n-cpu-moe 8 -ot blk\.(0|1)\.ffn_(up|gate|down)\.weight=CPU --load-mode none --cache-ram 0') 'tuner placement flags are saved with the enforced RAM flags'
    $savedInput.extraArgs='--n-cpu-moe -1'
    $beforeCpuHash=(Get-FileHash $ModelsConfigFile).Hash
    Assert-Check (-not (Save-ModelConfigOnly 'model.gguf' $savedInput).success -and (Get-FileHash $ModelsConfigFile).Hash -eq $beforeCpuHash) 'invalid expert count cannot overwrite settings'
    $savedInput.extraArgs=''
    Assert-Check (Save-ModelConfigOnly 'model.gguf' $savedInput).success 'clearing extraArgs restores normal placement'
    Assert-Check ((Resolve-LlamaServerBinary $Root $LlamaBin $settings) -eq $customBin) 'profile-specific binary resolves'
    Assert-Throws { Resolve-LlamaServerBinary $Root $LlamaBin @{serverBinary='missing.exe'} } 'missing binary fails before launch'
    Assert-Throws { Resolve-LlamaServerBinary $Root $LlamaBin @{serverBinary='custom\fork-server.exe'} } 'a renamed build is refused (the one-server checks find servers by name)'

    # Split GGUF: llama.cpp loads every shard from the first, so admission charges all of them.
    foreach ($n in 1..3) { [IO.File]::WriteAllBytes((Join-Path $ModelsDir ('big-{0:D5}-of-00003.gguf' -f $n)), (New-Object byte[] 1000)) }
    Assert-Check ((Get-LlamaModelFileBytes (Get-Item (Join-Path $ModelsDir 'big-00001-of-00003.gguf'))) -eq 3000) 'all split shards are charged'
    Assert-Throws { Get-LlamaModelFileBytes (Get-Item (Join-Path $ModelsDir 'big-00002-of-00003.gguf')) } 'a split model loads only from its first shard'
    Remove-Item -LiteralPath (Join-Path $ModelsDir 'big-00003-of-00003.gguf')
    Assert-Throws { Get-LlamaModelFileBytes (Get-Item (Join-Path $ModelsDir 'big-00001-of-00003.gguf')) } 'a missing shard is refused'
    Assert-Check ((Get-LlamaModelFileBytes (Get-Item (Join-Path $ModelsDir 'model.gguf'))) -eq (Get-Item (Join-Path $ModelsDir 'model.gguf')).Length) 'single-file model is charged its own size'

    # --no-mmproj-offload keeps the projector in system RAM, so its size is charged.
    [IO.File]::WriteAllBytes((Join-Path $ModelsDir 'big-projector.gguf'), (New-Object byte[] 5000))
    $settings.extraArgs = '--mmproj big-projector.gguf --load-mode none --cache-ram 0'
    $gpuProjector = (Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $settings).companionBytes
    $settings.extraArgs = '--mmproj big-projector.gguf --no-mmproj-offload --load-mode none --cache-ram 0'
    $cpuProjector = (Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $settings).companionBytes
    Assert-Check ($cpuProjector -eq 5000 -and $gpuProjector -eq 0) 'CPU projector size is charged; offloaded projector is not'
    $settings.extraArgs = '--mmproj missing-projector.gguf --no-mmproj-offload'
    Assert-Throws { Get-LlamaModelAdmission -FileName 'model.gguf' -Settings $settings } 'a missing CPU projector is refused'
    $settings.extraArgs = ''
    $listed = Get-AvailableModels
    Assert-Check ('projector.gguf' -notin $listed.models.filename) 'dashboard hides non-chat projector'
    $settings.contextSize=0; Assert-Throws { Build-LlamaArgs 'model.gguf' $settings } 'zero context is refused'; $settings.contextSize=8192

    # Test doubles never touch real processes or the actual dashboard.
    $script:StopCalls=0; $script:StartCalls=0
    function Get-LlamaServerState { return $script:FakeState }
    function Stop-Process { param($Id,[switch]$Force,$ErrorAction) $script:StopCalls++; if ($script:Vanish) { $script:FakeState=@{processIds=@();portOwner=$null;blockedPort=$false} } }
    function Start-Process { param($FilePath,$ArgumentList,$WorkingDirectory,$RedirectStandardOutput,$RedirectStandardError,$WindowStyle,[switch]$PassThru) $script:StartCalls++;$script:LastBinary=$FilePath; return [pscustomobject]@{Id=7777;PriorityClass='Normal'} }
    function Sync-CptrContext { param($NCtx) }
    $script:FakeState=@{processIds=@(123);portOwner=123;blockedPort=$false}; $script:Vanish=$false
    Set-Content $LlamaPidFile '123'
    $stuck=Stop-LlamaNative -TimeoutSeconds 0
    Assert-Check (-not $stuck.success -and (Test-Path $LlamaPidFile)) 'stuck process refuses stop and retains tracking'
    $script:FakeState=@{processIds=@();portOwner=900;blockedPort=$true};$script:StopCalls=0
    Assert-Check (-not (Stop-LlamaNative -TimeoutSeconds 0).success -and $script:StopCalls -eq 0) 'unrelated port owner is never killed'
    $script:FakeState=@{processIds=@(456);portOwner=456;blockedPort=$false}
    Assert-Check (-not (Stop-LlamaNative -TimeoutSeconds 0 -ExpectedPid 123).success -and $script:StopCalls -eq 0) 'a stale watchdog cannot stop a replacement model'
    $script:FakeState=@{processIds=@(123);portOwner=123;blockedPort=$false};$script:Vanish=$true
    Assert-Check ((Stop-LlamaNative -TimeoutSeconds 0).success -and -not (Test-Path $LlamaPidFile)) 'confirmed exit clears tracking'

    function Get-LlamaModelAdmission { param($FileName,$Settings) $script:AdmissionCalls++;if ($script:DropRam -and $script:AdmissionCalls -gt 1) { return @{allowed=$false;availableGiB=1;requiredGiB=3} };return @{allowed=$true;availableGiB=5;requiredGiB=3} }
    function Stop-LlamaNative { $script:StopCalls++;return @{success=$false;message='fixture: still exiting'} }
    $script:AdmissionCalls=0;$script:DropRam=$false
    $refused=Start-LlamaNative 'model.gguf' $settings
    Assert-Check (-not $refused.success -and $script:StartCalls -eq 0) 'failed shutdown prevents replacement launch'
    function Stop-LlamaNative { $script:StopCalls++;return @{success=$true} }
    $script:AdmissionCalls=0;$script:DropRam=$true
    Assert-Check (-not (Start-LlamaNative 'model.gguf' $settings).success -and $script:StartCalls -eq 0) 'RAM is checked again after shutdown'
    $settings.extraArgs='--cache-ram 8';$script:StopCalls=0
    Assert-Check (-not (Start-LlamaNative 'model.gguf' $settings).success -and $script:StopCalls -eq 0) 'invalid arguments preserve the prior server'
    $settings.extraArgs='';$script:DropRam=$false;$script:AdmissionCalls=0
    Assert-Check ((Start-LlamaNative 'model.gguf' $settings).success -and $script:StartCalls -eq 1 -and $script:LastBinary -eq $customBin) 'admitted launch uses the selected binary'

    function Start-LlamaNative { param($FileName,$Settings) return @{success=$false;message='fixture refusal'} }
    $beforeHash=(Get-FileHash $ModelsConfigFile).Hash
    Assert-Check (-not (Apply-ModelSettings 'other.gguf').success -and (Get-FileHash $ModelsConfigFile).Hash -eq $beforeHash) 'failed model load preserves saved selection and config'
    function Start-LlamaNative { param($FileName,$Settings) return @{success=$true;message='fixture launch'} }
    Assert-Check ((Apply-ModelSettings 'other.gguf').success -and (Get-ModelsConfig).activeModel -eq 'other.gguf') 'accepted model launch saves the selection'

    $startAst=[Management.Automation.Language.Parser]::ParseFile((Join-Path $stackRoot 'start.ps1'),[ref]$tokens,[ref]$parseErrors)
    Assert-Check ($parseErrors.Count -eq 0) 'startup script parses'
    $waitNode=$startAst.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Wait-Endpoint'},$false)|Select-Object -First 1
    . ([scriptblock]::Create($waitNode.Extent.Text))
    $DashboardBaseUrl='http://fixture'
    $script:StopRequest=$null
    function Test-Endpoint { param($Url,$TimeoutSec) return $script:EndpointReady }
    function Get-LlamaPhysicalMemory { return @{availableBytes=0.5GB} }
    function Start-Sleep { param($Seconds,$Milliseconds) }
    function Warn { param($Text) }
    function Invoke-RestMethod { param($Uri,$Method,$ContentType,$Body,$TimeoutSec,$ErrorAction) $script:StopRequest=$Body|ConvertFrom-Json; return @{success=$true} }
    $script:EndpointReady=$false
    $wait=Wait-Endpoint -Url 'http://fixture/health' -TimeoutSeconds 10 -Name 'fixture' -ExpectedLlamaPid 7777
    Assert-Check (-not $wait -and $script:StopRequest.expectedPid -eq 7777) 'startup watchdog requests a PID-specific stop after sustained low RAM'
    function Get-LlamaPhysicalMemory { return @{availableBytes=5GB} }
    $script:EndpointReady=$true;$script:StopRequest=$null
    Assert-Check ((Wait-Endpoint -Url 'http://fixture/health' -TimeoutSeconds 1 -Name 'fixture' -ExpectedLlamaPid 7777) -and -not $script:StopRequest) 'healthy startup does not issue a stop'
    Write-Output "PASS: $script:Checks offline safety assertions; no model processes launched."
} finally {
    $resolved=[IO.Path]::GetFullPath($Root)
    if ($resolved.StartsWith($scratchRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
