# Exercise the actual manager without performing inference or a viable model load.
param([string]$OutputPath)
$ErrorActionPreference='Stop'
$stackRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $stackRoot 'Tools\scripts\LlamaSafety.ps1')
if ((Get-Process llama-server -ErrorAction SilentlyContinue) -or (Get-NetTCPConnection -LocalPort 10000 -State Listen -ErrorAction SilentlyContinue)) {
    throw 'Run this check with the model server stopped; active work must be preserved.'
}
$base='http://127.0.0.1:9090'
function Post-Manager { param([string]$Path,$Body) return Invoke-RestMethod ($base+$Path) -Method Post -ContentType 'application/json' -Body ($Body|ConvertTo-Json -Depth 8 -Compress) -TimeoutSec 15 }
$configPath=Join-Path $stackRoot 'Config\models.json'
$before=(Get-FileHash $configPath -Algorithm SHA256).Hash
$catalog=Invoke-RestMethod "$base/api/models" -TimeoutSec 10
$preflights=@($catalog.models|ForEach-Object { Post-Manager '/api/models/preflight' @{filename=$_.filename} })
if (@($catalog.models|Where-Object { $_.filename -match '^mmproj' -or $_.architecture -eq 'clip' }).Count) { throw 'A projector is listed as a chat model.' }
$missingName='missing-stack-regression-'+[guid]::NewGuid().ToString('N')+'.gguf'
$missing=Post-Manager '/api/models/load' @{filename=$missingName}
$unsafe=Post-Manager '/api/models/load' @{filename=$catalog.activeModel;persist=$false;settings=@{extraArgs='--cache-ram 8192'}}
$unsafeSave=Post-Manager '/api/models/config' @{filename=$catalog.activeModel;settings=@{extraArgs='--load-mode auto'}}
$staleStop=Post-Manager '/api/stop' @{service='llama-server';expectedPid=2147483647}
$cpuRefusal=$null
$largeFile=Join-Path $stackRoot 'models\medgemma-27b-it-Q4_K_M.gguf'
if (Test-Path -LiteralPath $largeFile) {
    $memory=Get-LlamaPhysicalMemory
    $cpuRequired=(Get-Item -LiteralPath $largeFile).Length+1.75GB
    # This test can only run if the request cannot fit even with all physical
    # RAM available. No change in another app's memory can make it launch.
    if ($cpuRequired -gt $memory.totalBytes) {
        $cpuRefusal=Post-Manager '/api/models/load' @{filename='medgemma-27b-it-Q4_K_M.gguf';persist=$false;settings=@{gpuLayers=0}}
        if ($cpuRefusal.success) { throw 'An impossible CPU allocation was accepted.' }
    }
}
$after=(Get-FileHash $configPath -Algorithm SHA256).Hash
if ($missing.success -or $unsafe.success -or $unsafeSave.success -or $staleStop.success -or $before -ne $after -or (Get-Process llama-server -ErrorAction SilentlyContinue)) { throw 'Manager refusal or config preservation failed.' }
$result=[ordered]@{
    modelsListed=@($catalog.models).Count; projectorListed=0; activeModel=$catalog.activeModel
    preflight=$preflights; deniedMissingModel=$missing; deniedUnsafeOverride=$unsafe; deniedUnsafeSave=$unsafeSave
    deniedImpossibleCpuLoad=$cpuRefusal; configByteIdentical=$true; modelProcessesStarted=0
    ignoredStaleWatchdog=$staleStop
}
if ($OutputPath) { $result|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $OutputPath -Encoding UTF8 }
Write-Output "PASS: $(@($catalog.models).Count) model preflights, invalid model/settings refusals, configuration unchanged, zero model processes started."
