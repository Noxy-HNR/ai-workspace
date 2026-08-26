<#
  start-computer.ps1 - launches cptr natively on Windows.

  Single job: make sure cptr is running, and wait until it responds.
  llama-server startup lives in start.ps1, which calls this
  script - this file intentionally does not duplicate that.

  cptr runs as the current Windows user, so it has that account's real
  filesystem and (once you enable it in cptr's own Settings > Admin > Web)
  real browser access. Its current CLI does not accept --cwd or --api-key -
  workspace selection and the gateway API key are set inside its UI.
#>

$ErrorActionPreference = "SilentlyContinue"
$Root = "C:\AI"
$CptrUrl = "http://127.0.0.1:8000"
$CptrTimeoutSeconds = 180

function OK   ($m) { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "[!] $m" -ForegroundColor Yellow }
function Info ($m) { Write-Host "    $m" -ForegroundColor DarkGray }

function Test-Endpoint($Url, $TimeoutSec = 5) {
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
    } catch { return $false }
}

if (-not (Get-Command cptr -ErrorAction SilentlyContinue)) {
    Warn "cptr command not found. Run setup.ps1, or open a new PowerShell window"
    Warn "if you just installed it (PATH needs to refresh)."
    exit 1
}

# Already running?
$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match '\bcptr(\.exe)?\s+run\b' }

if ($existing) {
    OK "cptr is already running (PID $($existing.ProcessId -join ', '))."
} else {
    Info "Starting cptr on $CptrUrl ..."
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit",
                        "-Command", "Set-Location '$Root'; cptr run --host 127.0.0.1 --port 8000") `
        -WorkingDirectory $Root | Out-Null

    $started = Get-Date
    $ready = $false
    while (((Get-Date) - $started).TotalSeconds -lt $CptrTimeoutSeconds) {
        if (Test-Endpoint -Url $CptrUrl) { $ready = $true; break }
        Start-Sleep -Seconds 3
    }

    if ($ready) {
        OK "cptr is ready."
    } else {
        Warn "cptr did not respond within $CptrTimeoutSeconds seconds."
        Warn "Its PowerShell window should still be open - check that for errors."
    }
}
