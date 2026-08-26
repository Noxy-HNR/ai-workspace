# Model management for the native llama-server dashboard stack.
# Usage: .\model-manage.ps1 [command] [options]
#
# Commands:
#   list        List available models and their configured settings
#   status      Check the currently loaded model
#   load <name> Save+restart llama-server with <name> (matches on filename, .gguf optional)
#   unload      Stop llama-server
#   swap <name> Alias for load <name>
#
# This talks to the same dashboard API (:9090) the web UI uses to
# actually save config and restart llama-server -- it does not assume
# a multi-model hot-swap router (llama-server only ever runs one model
# at a time on this stack).

$ErrorActionPreference = "Stop"
$DashboardApi = "http://127.0.0.1:9090"
$LlamaApi = "http://127.0.0.1:10000"
$ModelDir = "C:\AI\models"

function Get-Models {
    try {
        $resp = Invoke-RestMethod "$DashboardApi/api/models" -TimeoutSec 5 -ErrorAction Stop
    } catch {
        Write-Host "Could not reach dashboard API at $DashboardApi (is dashboard-server.ps1 running?)" -ForegroundColor Red
        return
    }
    if (-not $resp.models -or $resp.models.Count -eq 0) {
        Write-Host "No GGUF files found in $ModelDir." -ForegroundColor Yellow
        return
    }
    Write-Host "Models in $ModelDir`:" -ForegroundColor Cyan
    foreach ($m in $resp.models) {
        $activeTag = if ($m.isActive) { " [ACTIVE]" } else { "" }
        Write-Host ("  {0}  ({1} GB)  ctx={2} ngl={3} kv={4}/{5}{6}" -f `
            $m.filename, $m.sizeGB, $m.settings.contextSize, $m.settings.gpuLayers, `
            $m.settings.cacheTypeK, $m.settings.cacheTypeV, $activeTag) -ForegroundColor White
    }
}

function Get-LoadedModel {
    try {
        $resp = Invoke-RestMethod "$LlamaApi/v1/models" -TimeoutSec 5 -ErrorAction Stop
        if ($resp.data -and $resp.data.Count -gt 0) {
            Write-Host "Currently loaded:" -ForegroundColor Cyan
            foreach ($m in $resp.data) { Write-Host "  $($m.id)" -ForegroundColor Green }
        } else {
            Write-Host "llama-server is up but reports no model." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "llama-server is not responding on $LlamaApi (not running, or still loading a model)." -ForegroundColor Yellow
    }
}

function Resolve-ModelName {
    param([string]$Name)
    if (Test-Path (Join-Path $ModelDir $Name)) { return $Name }
    if (Test-Path (Join-Path $ModelDir "$Name.gguf")) { return "$Name.gguf" }
    return $null
}

function Load-Model {
    param([string]$Name)
    $resolved = Resolve-ModelName $Name
    if (-not $resolved) {
        Write-Host "Model not found: $Name" -ForegroundColor Red
        Write-Host "  Run '.\model-manage.ps1 list' to see available models." -ForegroundColor Yellow
        return
    }

    Write-Host "Loading $resolved via dashboard (this restarts llama-server and can take a while for large models)..." -ForegroundColor Cyan
    try {
        $body = @{ filename = $resolved } | ConvertTo-Json
        $resp = Invoke-RestMethod "$DashboardApi/api/models/load" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 30 -ErrorAction Stop
        if (-not $resp.success) {
            Write-Host "Failed to start loading: $($resp.message)" -ForegroundColor Red
            return
        }
        Write-Host $resp.message -ForegroundColor Green
    } catch {
        Write-Host "Failed to reach dashboard API: $($_.Exception.Message)" -ForegroundColor Red
        return
    }

    Write-Host "Waiting for llama-server to report the model as loaded..." -ForegroundColor DarkGray
    $started = Get-Date
    while (((Get-Date) - $started).TotalSeconds -lt 600) {
        Start-Sleep -Seconds 3
        try {
            $models = Invoke-RestMethod "$LlamaApi/v1/models" -TimeoutSec 3 -ErrorAction Stop
            if ($models.data -and ($models.data[0].id -like "*$resolved*")) {
                Write-Host "$resolved is loaded and ready." -ForegroundColor Green
                return
            }
        } catch {}
    }
    Write-Host "Still waiting after 10 minutes -- check Logs\llama.stderr.log." -ForegroundColor Yellow
}

function Unload-Model {
    Write-Host "Stopping llama-server..." -ForegroundColor Cyan
    try {
        $body = @{ service = "llama-server" } | ConvertTo-Json
        $resp = Invoke-RestMethod "$DashboardApi/api/stop" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 15 -ErrorAction Stop
        Write-Host $resp.message -ForegroundColor Green
    } catch {
        Write-Host "Failed to reach dashboard API: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Main
$cmd = $args[0]

switch ($cmd) {
    "list" { Get-Models }
    "status" { Get-LoadedModel }
    "load" {
        if ($args.Count -lt 2) {
            Write-Host "Usage: .\model-manage.ps1 load <model-name>" -ForegroundColor Yellow
        } else {
            Load-Model ($args[1])
        }
    }
    "unload" { Unload-Model }
    "swap" {
        if ($args.Count -lt 2) {
            Write-Host "Usage: .\model-manage.ps1 swap <model-name>" -ForegroundColor Yellow
        } else {
            Load-Model ($args[1])
        }
    }
    default {
        Write-Host "AI Model Manager" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: .\model-manage.ps1 <command>" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Cyan
        Write-Host "  list        - List available models and their configured settings" -ForegroundColor White
        Write-Host "  status      - Check the currently loaded model" -ForegroundColor White
        Write-Host "  load <name> - Save+restart llama-server with <name>" -ForegroundColor White
        Write-Host "  unload      - Stop llama-server" -ForegroundColor White
        Write-Host "  swap <name> - Alias for load <name>" -ForegroundColor White
    }
}
