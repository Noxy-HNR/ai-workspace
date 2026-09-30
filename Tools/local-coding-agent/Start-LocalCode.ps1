[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Repository,
    [int]$TimeoutMinutes = 90,
    [string]$AiRoot = 'C:\AI'
)

$ErrorActionPreference = 'Stop'
if ($TimeoutMinutes -lt 1 -or $TimeoutMinutes -gt 240) { throw 'TimeoutMinutes must be between 1 and 240.' }
$repositoryPath = [System.IO.Path]::GetFullPath($Repository)
$configPath = Join-Path $repositoryPath '.opencode\opencode.jsonc'
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "No local profile is installed in '$repositoryPath'. Run Install-LocalCodeProfile.ps1 first while its model is already served."
}
$profile = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($profile.model -ne 'local-llama/coding') { throw 'The repository profile does not select the expected local model.' }
if ($profile.providers.'local-llama'.settings.baseURL -ne 'http://127.0.0.1:10000/v1') { throw 'The profile endpoint is not the local dashboard-managed server.' }
$config = Get-Content -LiteralPath (Join-Path $AiRoot 'Config\models.json') -Raw | ConvertFrom-Json
$modelFile = [string]$config.activeModel
if (-not $config.models.PSObject.Properties[$modelFile]) { throw "The dashboard config does not contain activeModel '$modelFile'." }
if ([string]$profile.providers.'local-llama'.models.coding.name -cne $modelFile) {
    throw 'The installed local profile was created for a different configured model. Reinstall it while the desired model is already served.'
}

$preflight = & (Join-Path $PSScriptRoot 'Test-LocalCodePreflight.ps1') -AiRoot $AiRoot
if (-not $preflight.AcPower) { throw 'Connect AC power before starting a coding session.' }
if ($preflight.AvailablePhysicalGiB -lt 1.0) { throw "Only $($preflight.AvailablePhysicalGiB) GiB physical RAM is available; session startup needs at least 1.0 GiB headroom." }
if ($preflight.LlamaServerProcessCount -ne 1 -or $preflight.Port10000ListenerCount -lt 1) { throw 'A single llama-server must already be serving on port 10000.' }

$live = Invoke-RestMethod -Uri 'http://127.0.0.1:10000/v1/models' -TimeoutSec 5
$liveModels = @($live.data)
$configuredId = [string]$profile.providers.'local-llama'.models.coding.modelID
if ($liveModels.Count -ne 1 -or [string]$liveModels[0].id -cne $configuredId) {
    throw "The local server model does not match this profile. Reinstall the profile using the currently served model; no model switch was attempted."
}

$openCode = (Get-Command opencode -ErrorAction Stop).Source
$dashboard = 'http://127.0.0.1:9090/api/stop'
$watchdog = Start-Job -ArgumentList $TimeoutMinutes, $dashboard -ScriptBlock {
    param($Minutes, $StopUrl)
    Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class LocalCodeWatchMemory {
 [StructLayout(LayoutKind.Sequential)] public struct M { public uint n; public uint load; public ulong total; public ulong avail; public ulong page; public ulong availPage; public ulong virt; public ulong availVirt; public ulong ext; }
 [DllImport("kernel32.dll")] public static extern bool GlobalMemoryStatusEx(ref M m);
}
'@ -ErrorAction SilentlyContinue
    $start = [DateTime]::UtcNow
    $lowCount = 0
    while ((([DateTime]::UtcNow - $start).TotalMinutes -lt $Minutes)) {
        $m = New-Object LocalCodeWatchMemory+M
        $m.n = [Runtime.InteropServices.Marshal]::SizeOf([type]'LocalCodeWatchMemory+M')
        if (-not [LocalCodeWatchMemory]::GlobalMemoryStatusEx([ref]$m)) { Start-Sleep -Seconds 2; continue }
        if ($m.avail -lt 858993459) { $lowCount++ } else { $lowCount = 0 }
        if ($lowCount -ge 3) {
            Write-Output 'RAM WATCHDOG: available physical RAM stayed below 0.8 GiB for about six seconds.'
            try {
                $body = @{ service = 'llama-server' } | ConvertTo-Json -Compress
                $result = Invoke-RestMethod -Uri $StopUrl -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 8
                if (-not $result.success) { throw [string]$result.message }
                Write-Output 'Requested llama-server stop through the dashboard.'
            } catch { Write-Output "Dashboard stop request failed: $($_.Exception.Message)" }
            break
        }
        Start-Sleep -Seconds 2
    }
    if (([DateTime]::UtcNow - $start).TotalMinutes -ge $Minutes) {
        Write-Output "SESSION WATCHDOG: hard session limit of $Minutes minutes reached."
        try {
            $body = @{ service = 'llama-server' } | ConvertTo-Json -Compress
            $result = Invoke-RestMethod -Uri $StopUrl -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 8
            if (-not $result.success) { throw [string]$result.message }
            Write-Output 'Requested llama-server stop through the dashboard.'
        } catch { Write-Output "Dashboard stop request failed: $($_.Exception.Message)" }
    }
}

try {
    Write-Host "Starting OpenCode with local model $modelFile in $repositoryPath"
    Write-Host "Session limit: $TimeoutMinutes minutes. RAM watchdog stops the server below 0.8 GiB for ~6 seconds."
    $client = Start-Process -FilePath $openCode -WorkingDirectory $repositoryPath -NoNewWindow -PassThru
    while (-not $client.HasExited) {
        Start-Sleep -Seconds 1
        $client.Refresh()
        $watchOutput = Receive-Job -Job $watchdog -Keep 2>$null
        if ($watchOutput -match 'RAM WATCHDOG|SESSION WATCHDOG') {
            $watchOutput | ForEach-Object { Write-Warning $_ }
            Stop-Process -Id $client.Id -Force -ErrorAction SilentlyContinue
            break
        }
    }
    if ($client.HasExited) { exit $client.ExitCode }
} finally {
    Stop-Job -Job $watchdog -ErrorAction SilentlyContinue
    Receive-Job -Job $watchdog -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
    Remove-Job -Job $watchdog -Force -ErrorAction SilentlyContinue
}
