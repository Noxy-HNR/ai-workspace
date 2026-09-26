<#
=======================================================================
 C:\AI\start.ps1
 MASTER STARTUP SCRIPT - LOCAL AI STACK

 Starts, verifies, and opens:

   llama.cpp (native Windows binary, CUDA)
       +-- OpenAI-compatible API :10000

   cptr
       +-- Windows filesystem
       +-- Windows command execution
       +-- Browser automation
       +-- SQLite / databases
       +-- MCP / additional tools

 Normal usage:

   cd C:\AI
   .\start.ps1

 This script intentionally keeps both llama-server and cptr native
 on Windows -- no Docker/WSL layer, which meaningfully reduces idle
 RAM usage and startup latency compared to the previous Docker-based
 setup.

 IMPORTANT:
   cptr's actual tools/configuration are controlled by cptr itself.
   This script starts and verifies cptr; it does not invent unsupported
   cptr CLI arguments or fake tool configuration.
=======================================================================
#>

[CmdletBinding()]
param(
    # Start and check the services without opening browser tabs or console windows.
    [switch]$NoUi,
    # Return to the caller after startup (also used by automated validation).
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

$Root = "C:\AI"
$LlamaBin = Join-Path $Root "Tools\llama-native\bin\llama-server.exe"
$ModelsDir = Join-Path $Root "models"
$ModelsConfigFile = Join-Path $Root "Config\models.json"
$LogDir = Join-Path $Root "Logs"

# Network endpoints
$LlamaBaseUrl = "http://127.0.0.1:10000"
$LlamaHealthUrl = "$LlamaBaseUrl/health"
$LlamaModelsUrl = "$LlamaBaseUrl/v1/models"

$CptrUrl = "http://127.0.0.1:8000"

# Timeouts
$LlamaTimeout = 900
$CptrTimeout = 180

# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

function Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
}

function Step {
    param([string]$Text)
    Write-Host ""
    Write-Host "[*] $Text" -ForegroundColor Cyan
}

function OK {
    param([string]$Text)
    Write-Host "[OK] $Text" -ForegroundColor Green
}

function Warn {
    param([string]$Text)
    Write-Host "[!] $Text" -ForegroundColor Yellow
}

function Fail {
    param([string]$Text)
    Write-Host "[X] $Text" -ForegroundColor Red
}

function Die {
    param([string]$Text)
    Fail $Text
    Write-Host ""
    if (-not $NoPause) { Read-Host "Press Enter to close" }
    exit 1
}

# ---------------------------------------------------------------------
# HTTP HELPERS
# ---------------------------------------------------------------------

function Test-Endpoint {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [int]$TimeoutSec = 5
    )
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Wait-Endpoint {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][int]$TimeoutSeconds,
        [Parameter(Mandatory=$true)][string]$Name
    )
    $started = Get-Date
    $lastPrint = -15
    while ($true) {
        if (Test-Endpoint -Url $Url -TimeoutSec 5) { return $true }
        $elapsed = [int]((Get-Date) - $started).TotalSeconds
        if ($elapsed -ge $TimeoutSeconds) { return $false }
        if (($elapsed - $lastPrint) -ge 15) {
            Write-Host "    Waiting for $Name... $elapsed/$TimeoutSeconds seconds" -ForegroundColor DarkGray
            $lastPrint = $elapsed
        }
        Start-Sleep -Seconds 3
    }
}

function Get-PortPid {
    param([int]$Port)
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
        if ($c) { return [int]$c.OwningProcess }
    } catch {}
    return $null
}

# ---------------------------------------------------------------------
# VERIFY INSTALLATION
# ---------------------------------------------------------------------

Banner "LOCAL AI STACK (native)"

Step "Checking C:\AI"

if (-not (Test-Path $Root)) {
    Die "C:\AI does not exist. Run setup.ps1 first."
}

Set-Location $Root

if (-not (Test-Path $LlamaBin)) {
    Die "llama-server.exe not found at $LlamaBin. Run setup.ps1 first."
}

OK "C:\AI found."

# ---------------------------------------------------------------------
# CREATE / VERIFY DIRECTORY STRUCTURE
# ---------------------------------------------------------------------

Step "Checking AI workspace"

$Directories = @(
    "models",
    "Downloads",
    "Workspace",
    "Projects",
    "Projects\Scratch",
    "Projects\Research",
    "Projects\Code",
    "Databases",
    "Tools",
    "MCP",
    "Config",
    "Logs",
    "Cache"
)

foreach ($dir in $Directories) {
    $path = Join-Path $Root $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

OK "Workspace directories verified."

# ---------------------------------------------------------------------
# LOAD ACTIVE MODEL CONFIG
# ---------------------------------------------------------------------

Step "Reading active model configuration"

if (-not (Test-Path $ModelsConfigFile)) {
    Die "No models configured yet. Open the dashboard and load a model at least once first."
}

$modelsConfig = Get-Content $ModelsConfigFile -Raw -Encoding UTF8 | ConvertFrom-Json
$activeModel = $modelsConfig.activeModel

if ([string]::IsNullOrWhiteSpace($activeModel)) {
    Die "No active model set in Config\models.json. Load a model from the dashboard first."
}

$modelPath = Join-Path $ModelsDir $activeModel
if (-not (Test-Path $modelPath)) {
    Die "Active model file not found: $modelPath"
}

$settings = $modelsConfig.models.$activeModel
OK "Active model: $activeModel"

# ---------------------------------------------------------------------
# START LLAMA-SERVER NATIVELY
# ---------------------------------------------------------------------

Step "Starting llama-server natively"

$existingLlamaPid = Get-PortPid 10000
if ($existingLlamaPid) {
    OK "llama-server is already running (PID $existingLlamaPid)."
} else {
    . (Join-Path $Root "Tools\scripts\Build-LlamaArgs.ps1")
    $argList = Build-LlamaArgs -ModelPath $modelPath -Settings $settings

    $stdOut = Join-Path $LogDir "llama.stdout.log"
    $stdErr = Join-Path $LogDir "llama.stderr.log"

    $p = Start-Process -FilePath $LlamaBin -ArgumentList $argList -WorkingDirectory (Split-Path $LlamaBin) `
        -RedirectStandardOutput $stdOut -RedirectStandardError $stdErr -WindowStyle Hidden -PassThru

    OK "llama-server process launched (PID $($p.Id))."
}

# ---------------------------------------------------------------------
# WAIT FOR LLAMA-SERVER
# ---------------------------------------------------------------------

Step "Waiting for llama-server"

Write-Host "    API : $LlamaBaseUrl" -ForegroundColor DarkGray
Write-Host ""

if (Wait-Endpoint -Url $LlamaHealthUrl -TimeoutSeconds $LlamaTimeout -Name "llama-server") {
    OK "llama-server API is ready."
} else {
    Fail "llama-server did not become ready."
    Write-Host ""
    Write-Host "Recent llama-server logs:" -ForegroundColor Yellow
    Get-Content (Join-Path $LogDir "llama.stderr.log") -Tail 40 -ErrorAction SilentlyContinue
    Die "llama-server startup failed or timed out."
}

# ---------------------------------------------------------------------
# VERIFY MODEL
# ---------------------------------------------------------------------

Step "Checking loaded model"

try {
    $modelResponse = Invoke-RestMethod -Uri $LlamaModelsUrl -TimeoutSec 15 -ErrorAction Stop
    if ($null -eq $modelResponse.data) {
        Warn "llama-server responded but did not report a model."
    } else {
        foreach ($model in $modelResponse.data) {
            OK "Loaded model: $($model.id)"
        }
    }
} catch {
    Warn "Could not query llama-server /v1/models."
    Warn $_.Exception.Message
}

# ---------------------------------------------------------------------
# DASHBOARD SERVER
# ---------------------------------------------------------------------

Step "Starting dashboard server"

$dashboardPid = Get-PortPid 9090
if ($dashboardPid) {
    OK "Dashboard server is already running (PID $dashboardPid)."
} else {
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "dashboard-server.ps1")
    ) -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 2
    OK "Dashboard server launched on http://127.0.0.1:9090"
}

# ---------------------------------------------------------------------
# CPTR
# ---------------------------------------------------------------------

Step "Checking cptr"

$python = Get-Command python -ErrorAction SilentlyContinue

if (-not $python) {
    Warn "Python was not found. cptr cannot be started."
    $CptrAvailable = $false
} else {
    $cptr = Get-Command cptr -ErrorAction SilentlyContinue
    if (-not $cptr) {
        Warn "cptr command was not found."
        Warn "Open a new PowerShell window after installing cptr."
        $CptrAvailable = $false
    } else {
        $CptrAvailable = $true
        OK "cptr command found: $($cptr.Source)"
    }
}

# ---------------------------------------------------------------------
# START CPTR
# ---------------------------------------------------------------------

if ($CptrAvailable) {
    Step "Starting native cptr"

    $existingCptr = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match '\bcptr(\.exe)?\s+run\b' }

    if ($existingCptr) {
        OK "cptr is already running."
        $existingPids = ($existingCptr.ProcessId -join ", ")
        Write-Host "    PID: $existingPids" -ForegroundColor DarkGray
    } else {
        Write-Host "    Starting cptr on $CptrUrl" -ForegroundColor DarkGray

        # cptr gets native Windows permissions.
        #
        # IMPORTANT:
        # Current cptr releases use configuration/UI for capabilities
        # such as workspace, gateway keys, browser automation, etc.
        # Do not add unsupported CLI flags here.
        $cptrCommand = @"
Set-Location 'C:\AI'
cptr run --host 127.0.0.1 --port 8000
"@

        $cptrWindowStyle = if ($NoUi) { 'Hidden' } else { 'Normal' }
        Start-Process -FilePath "powershell.exe" -WindowStyle $cptrWindowStyle -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $cptrCommand
        ) -WorkingDirectory $Root | Out-Null

        OK "cptr process launched."
    }

    Step "Waiting for cptr"

    if (Wait-Endpoint -Url $CptrUrl -TimeoutSeconds $CptrTimeout -Name "cptr") {
        OK "cptr is ready."
    } else {
        Warn "cptr did not respond within $CptrTimeout seconds."
        Write-Host ""
        Write-Host "The cptr PowerShell window should still be open." -ForegroundColor Yellow
        Write-Host "Check that window for the startup error." -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------
# FINAL HEALTH CHECK
# ---------------------------------------------------------------------

# --- Switchboard (added by C:\AI\Switchboard\scripts\integrate-start.ps1) ---
# Optional: routes chat requests to the best local model. Remove this block, or
# run C:\AI\Switchboard\scripts\integrate-start.ps1 -Remove, to take it out.
# Nothing below can stop the rest of the stack: it is wrapped in try/catch.
$SwitchboardStart = "C:\AI\Switchboard\scripts\start.ps1"
if (Test-Path $SwitchboardStart) {
    Step "Starting Switchboard"
    try {
        # A child service can retain a native pipeline handle after its launcher
        # exits. Wait on the launcher process, with file-backed output instead.
        $sbLauncher = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -PassThru `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $SwitchboardStart) `
            -RedirectStandardOutput (Join-Path $LogDir 'switchboard-start.stdout.log') `
            -RedirectStandardError (Join-Path $LogDir 'switchboard-start.stderr.log')
        if (-not $sbLauncher.WaitForExit(60000)) {
            Warn 'Switchboard launcher is still running after 60 seconds; checking service health.'
        }
        if (Test-Endpoint -Url "http://127.0.0.1:8002/health" -TimeoutSec 5) {
            OK "Switchboard         READY  :8002"
        } else {
            Warn "Switchboard did not answer on :8002 (the rest of the stack is unaffected)."
        }
    } catch {
        Warn "Switchboard failed to start: $($_.Exception.Message)"
    }
}
# --- end Switchboard ---
Banner "FINAL HEALTH CHECK"

if (Test-Endpoint -Url $LlamaHealthUrl -TimeoutSec 5) {
    OK "llama-server        READY  :10000"
} else {
    Warn "llama-server        NOT READY"
}

if (Test-Endpoint -Url $CptrUrl -TimeoutSec 5) {
    OK "cptr                READY  :8000"
} elseif ($CptrAvailable) {
    Warn "cptr                NOT READY"
} else {
    Warn "cptr                NOT INSTALLED / NOT FOUND"
}

# ---------------------------------------------------------------------
# OPEN USER INTERFACES
# ---------------------------------------------------------------------

Step "Opening interfaces"

if (-not $NoUi) {
    Start-Process "http://127.0.0.1:9090"
    OK "Opened dashboard."
}

if ((-not $NoUi) -and (Test-Endpoint -Url $CptrUrl -TimeoutSec 5)) {
    Start-Process $CptrUrl
    OK "Opened cptr."
}

# ---------------------------------------------------------------------
# FINAL INFORMATION
# ---------------------------------------------------------------------

Banner "LOCAL AI STACK READY"

Write-Host ""
Write-Host " Dashboard" -ForegroundColor White
Write-Host "   http://localhost:9090" -ForegroundColor Cyan

Write-Host ""
Write-Host " cptr / Computer Agent" -ForegroundColor White
Write-Host "   http://localhost:8000" -ForegroundColor Cyan

Write-Host ""
Write-Host " llama-server API" -ForegroundColor White
Write-Host "   http://localhost:10000/v1" -ForegroundColor Cyan

Write-Host ""
Write-Host " AI workspace:" -ForegroundColor White
Write-Host "   C:\AI" -ForegroundColor Cyan

Write-Host ""
Write-Host " Important:" -ForegroundColor Yellow
Write-Host "   llama-server and cptr both run natively -- no Docker/WSL."
Write-Host "   The dashboard is the graphical interface / control layer."
Write-Host ""

Write-Host "Startup complete." -ForegroundColor Green
Write-Host ""

if (-not $NoPause) { Read-Host "Press Enter to close" }
