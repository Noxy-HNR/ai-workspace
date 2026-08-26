# Model comparison tool for the native llama-server dashboard stack.
# Usage: .\model-compare.ps1 [command] [args]
#
# Commands:
#   list                    Show models found in C:\AI\models and the active one
#   compare "<prompt>"      Load each model in turn, run the prompt, report tok/s + reply
#   quick "<prompt>"        Same as compare, but a short (50-token) reply for a fast speed check
#
# llama-server only ever runs one model at a time on this stack, so
# "compare" loads each model sequentially via the dashboard API (:9090),
# waits for it to come up, benchmarks it via the OpenAI-compatible API
# (:10000), and restores whichever model was active before this ran.

$ErrorActionPreference = "Continue"
$DashboardApi = "http://127.0.0.1:9090"
$LlamaApi = "http://127.0.0.1:10000/v1"

function List-Models {
    Write-Host "=== Models ===" -ForegroundColor Cyan
    try {
        $resp = Invoke-RestMethod "$DashboardApi/api/models" -TimeoutSec 5 -ErrorAction Stop
        if (-not $resp.models -or $resp.models.Count -eq 0) {
            Write-Host "  No GGUF files found in C:\AI\models." -ForegroundColor Yellow
            return
        }
        foreach ($m in $resp.models) {
            $tag = if ($m.isActive) { " [ACTIVE]" } else { "" }
            Write-Host "  $($m.filename) ($($m.sizeGB) GB)$tag"
        }
    } catch {
        Write-Host "  Could not reach dashboard API at $DashboardApi." -ForegroundColor Red
    }
}

function Wait-ModelReady {
    param([string]$Filename, [int]$TimeoutSec = 600)
    $started = Get-Date
    while (((Get-Date) - $started).TotalSeconds -lt $TimeoutSec) {
        Start-Sleep -Seconds 3
        try {
            $models = Invoke-RestMethod "$LlamaApi/models" -TimeoutSec 3 -ErrorAction Stop
            if ($models.data -and ($models.data[0].id -like "*$Filename*")) { return $true }
        } catch {}
    }
    return $false
}

function Run-Comparison {
    param([string]$Prompt, [int]$MaxTokens)

    if (-not $Prompt) {
        Write-Host "Usage: .\model-compare.ps1 compare '<prompt>'" -ForegroundColor Yellow
        return
    }

    try {
        $modelsResp = Invoke-RestMethod "$DashboardApi/api/models" -TimeoutSec 5 -ErrorAction Stop
    } catch {
        Write-Host "Could not reach dashboard API at $DashboardApi (is dashboard-server.ps1 running?)" -ForegroundColor Red
        return
    }
    if (-not $modelsResp.models -or $modelsResp.models.Count -eq 0) {
        Write-Host "No GGUF files found in C:\AI\models." -ForegroundColor Yellow
        return
    }

    $originalActive = $modelsResp.activeModel
    Write-Host "=== Comparing $($modelsResp.models.Count) model(s) ===" -ForegroundColor Cyan
    Write-Host "Prompt: $Prompt" -ForegroundColor White
    Write-Host "(this restarts llama-server once per model -- can take a while for large models)" -ForegroundColor DarkGray
    Write-Host ""

    foreach ($m in $modelsResp.models) {
        Write-Host "--- $($m.filename) ---" -ForegroundColor Green

        try {
            $body = @{ filename = $m.filename } | ConvertTo-Json
            $loadResp = Invoke-RestMethod "$DashboardApi/api/models/load" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 30 -ErrorAction Stop
            if (-not $loadResp.success) {
                Write-Host "  Failed to start loading: $($loadResp.message)" -ForegroundColor Red
                continue
            }
        } catch {
            Write-Host "  Failed to reach dashboard API: $($_.Exception.Message)" -ForegroundColor Red
            continue
        }

        if (-not (Wait-ModelReady -Filename $m.filename)) {
            Write-Host "  Timed out waiting for llama-server to load this model." -ForegroundColor Red
            continue
        }

        try {
            $chatBody = @{
                model = "local"
                messages = @(@{ role = "user"; content = $Prompt })
                max_tokens = $MaxTokens
                temperature = 0.7
            } | ConvertTo-Json -Depth 10

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $chatResp = Invoke-RestMethod "$LlamaApi/chat/completions" -Method POST -ContentType "application/json" `
                -Body ([System.Text.Encoding]::UTF8.GetBytes($chatBody)) -TimeoutSec 300 -ErrorAction Stop
            $sw.Stop()

            $tokens = if ($chatResp.usage) { $chatResp.usage.completion_tokens } else { 0 }
            $tps = if ($sw.Elapsed.TotalSeconds -gt 0 -and $tokens -gt 0) { [math]::Round($tokens / $sw.Elapsed.TotalSeconds, 2) } else { 0 }
            Write-Host "  $tps tok/s ($tokens tokens in $($sw.ElapsedMilliseconds)ms)" -ForegroundColor Gray

            $content = $chatResp.choices[0].message.content
            if ($content.Length -gt 400) { $content = $content.Substring(0, 400) + "..." }
            Write-Host "  $content"
        } catch {
            Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        }
        Write-Host ""
    }

    if ($originalActive -and $originalActive -ne $modelsResp.models[-1].filename) {
        Write-Host "Restoring originally active model ($originalActive)..." -ForegroundColor Cyan
        try {
            $body = @{ filename = $originalActive } | ConvertTo-Json
            Invoke-RestMethod "$DashboardApi/api/models/load" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 30 -ErrorAction Stop | Out-Null
            Wait-ModelReady -Filename $originalActive | Out-Null
            Write-Host "Restored." -ForegroundColor Green
        } catch {
            Write-Host "Could not restore original model automatically: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# Main
$cmd = $args[0]

switch ($cmd) {
    "list" { List-Models }
    "compare" { Run-Comparison -Prompt ($args[1]) -MaxTokens 200 }
    "quick" { Run-Comparison -Prompt ($args[1]) -MaxTokens 50 }
    default {
        Write-Host "Model Comparison Tool" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: .\model-compare.ps1 <command>" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Cyan
        Write-Host "  list                  - Show models found in C:\AI\models and the active one" -ForegroundColor White
        Write-Host "  compare '<prompt>'    - Load each model in turn and compare (200-token replies)" -ForegroundColor White
        Write-Host "  quick '<prompt>'      - Same, but a quick 50-token speed check" -ForegroundColor White
    }
}
