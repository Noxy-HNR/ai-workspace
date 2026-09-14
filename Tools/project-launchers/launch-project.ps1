[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('chemistry','research','spectra','evaluation','lecture','ir')]
    [string]$Project,
    [switch]$ValidateOnly,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$logRoot = 'C:\AI\Logs\project-launchers'
$science = $Project -in @('chemistry','research','spectra','evaluation')
if ($science) {
    $group = 'science-studio'; $root = 'C:\AI\Projects\science-studio'
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $entry = Join-Path $root 'run.py'; $arguments = @('"'+$entry+'"')
    $port = 8780; $probe = 'http://127.0.0.1:8780/api/health'; $marker = '"chemistry"'
    $url = 'http://127.0.0.1:8780/#' + $Project
} elseif ($Project -eq 'lecture') {
    $group = 'lecture-notes'; $root = 'C:\AI\Projects\lecture-notes'
    $python = Join-Path $root 'venv\Scripts\python.exe'
    $entry = Join-Path $root 'src\dashboard.py'; $arguments = @('"'+$entry+'"','--no-browser')
    $port = 8770; $probe = 'http://127.0.0.1:8770/'; $marker = '<title>Lecture Notes</title>'
    $url = $probe
} else {
    $group = 'ir-viewer'; $root = 'C:\AI\Projects\IRViewer'
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $entry = Join-Path $root 'irview.py'; $arguments = @('"'+$entry+'"')
    $port = 0
}
function Test-Ready {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $probe -TimeoutSec 2
        return $response.StatusCode -eq 200 -and $response.Content -match $marker
    } catch { return $false }
}
$mutex = $null; $owned = $false
try {
    foreach ($path in @($python,$entry)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required project file missing: $path" }
    }
    if ($ValidateOnly) { Write-Output "Validated $Project : $entry"; exit 0 }
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $mutex = New-Object System.Threading.Mutex($false,('Local\AI-Project-Launcher-'+$group))
    try { $owned = $mutex.WaitOne(60000) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
    if (-not $owned) { throw 'Another launch is still in progress. Try again shortly.' }
    if ($port -gt 0 -and (Test-Ready)) {
        Write-Output "Reusing $group at $url"
    } else {
        if ($port -gt 0) {
            $client = New-Object System.Net.Sockets.TcpClient
            try { $client.Connect('127.0.0.1',$port); $occupied = $true } catch { $occupied = $false } finally { $client.Dispose() }
            if ($occupied) { throw "Port $port is occupied by an unrecognized or unready service. No existing process was stopped." }
        }
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $outLog = Join-Path $logRoot "$group-$stamp.log"
        $errLog = Join-Path $logRoot "$group-$stamp.error.log"
        $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
        if ($port -gt 0) {
            $deadline = (Get-Date).AddSeconds(45)
            while (-not (Test-Ready)) {
                $process.Refresh()
                if ($process.HasExited) { throw "The project exited during startup. See $errLog" }
                if ((Get-Date) -gt $deadline) { throw "Startup is taking longer than expected. See $errLog. The process was left running." }
                Start-Sleep -Milliseconds 250
            }
        }
        Write-Output "Started $group (PID $($process.Id)). Logs: $logRoot"
    }
    if ($port -gt 0 -and -not $NoBrowser) { Start-Process $url }
} catch {
    if ($ValidateOnly -or $NoBrowser) { Write-Error $_; exit 1 }
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message,'Project could not start','OK','Error') | Out-Null
    exit 1
} finally {
    if ($owned) { $mutex.ReleaseMutex() }
    if ($mutex) { $mutex.Dispose() }
}
