# Shared process ownership checks. This file never reads the bot database.
$RelayRoot = $PSScriptRoot
$RelayEntry = Join-Path $RelayRoot 'app.py'

function Get-RelayPortProcess {
    param([int]$Port = 9095)
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $connection) { return $null }
    return Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$connection.OwningProcess)" -ErrorAction SilentlyContinue
}

function Test-RelayProcessCommand {
    param($Process, [int]$Port = 9095, [switch]$AllowLegacy)
    if (-not $Process -or $Process.Name -notmatch '^pythonw?\.exe$') { return $false }
    if ($Process.CommandLine -notmatch '(?i)(?:^|\s)-u\s+("[^"]+"|\S+)\s+--port\s+(\d+)\s*$') { return $false }
    $entry = $Matches[1].Trim('"')
    if ([int]$Matches[2] -ne $Port) { return $false }
    if ($entry -eq $RelayEntry) { return $true }
    return $AllowLegacy -and $entry -eq 'app.py'
}

function Get-RelayHealth {
    param([int]$Port = 9095)
    # A short socket probe avoids slow HTTP connection refusal on Windows.
    $socket = New-Object Net.Sockets.TcpClient
    try {
        $pending = $socket.BeginConnect('127.0.0.1',$Port,$null,$null)
        if (-not ($pending.AsyncWaitHandle.WaitOne(200) -and $socket.Connected)) { return $null }
        $socket.EndConnect($pending)
    } catch { return $null }
    finally {
        if ($pending) { $pending.AsyncWaitHandle.Close() }
        $socket.Dispose()
    }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2 -ErrorAction Stop
        if ($health.service -eq 'relay-station') { return $health }
    } catch {}
    return $null
}

function Test-RelayLegacyConsole {
    param([int]$Port = 9095)
    try {
        $page = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        return $page.Content -match '<title>Relay Station[^<]*</title>'
    } catch { return $false }
}

function Save-RelayInstance {
    param($Process, [int]$Port = 9095, [switch]$AllowLegacy)
    if (-not (Test-RelayProcessCommand -Process $Process -Port $Port -AllowLegacy:$AllowLegacy)) {
        throw 'The listening process does not match the Relay Station launcher.'
    }
    $runRoot = Join-Path $RelayRoot 'run'
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
    $record = @{ process_id=[int]$Process.ProcessId; created=$Process.CreationDate.ToUniversalTime().ToString('o');
                 port=$Port; entry=$RelayEntry; command=[string]$Process.CommandLine }
    $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot "instance-$Port.json") -Encoding UTF8
}

function Test-RelayOwnedProcess {
    param($Process, [int]$Port = 9095)
    if (-not (Test-RelayProcessCommand -Process $Process -Port $Port -AllowLegacy)) { return $false }
    $recordPath = Join-Path $RelayRoot "run\instance-$Port.json"
    try {
        $record = Get-Content -LiteralPath $recordPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        # PowerShell 7 converts ISO JSON dates to DateTime; Windows PowerShell
        # 5 keeps strings. Compare their UTC instant rather than formatting.
        $recordCreated = if ($record.created -is [datetime]) { $record.created.ToUniversalTime() }
                         else { [DateTimeOffset]::Parse([string]$record.created,[Globalization.CultureInfo]::InvariantCulture).UtcDateTime }
        return $record.process_id -eq $Process.ProcessId -and $record.port -eq $Port -and
            $record.entry -eq $RelayEntry -and $record.command -eq $Process.CommandLine -and
            $recordCreated.Ticks -eq $Process.CreationDate.ToUniversalTime().Ticks
    } catch { return $false }
}

function Get-RelayServiceStatus {
    param([int]$Port = 9095)
    $health = Get-RelayHealth -Port $Port
    $process = Get-RelayPortProcess -Port $Port
    $legacy = $null -eq $health -and (Test-RelayProcessCommand -Process $process -Port $Port -AllowLegacy) -and (Test-RelayLegacyConsole -Port $Port)
    $ready = $null -ne $health -or $legacy
    $poller = $null
    if ($health) {
        try { $poller = (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/poller" -TimeoutSec 2 -ErrorAction Stop).poller } catch {}
    }
    return @{ running=$ready; ready=$ready; pid=if ($process) { [int]$process.ProcessId } else { $null };
              port=$Port; url="http://127.0.0.1:$Port"; version=if ($health) { $health.version } else { '' };
              legacy=$legacy; poller_running=if ($poller) { [bool]$poller.running } else { $null };
              error=if ($legacy) { 'Older backend running. Restart this bot to load the completed update.' }
                    elseif (-not $ready -and $process) { "Port $Port is occupied, but Relay Station is not ready." }
                    elseif (-not $ready) { '' } else { '' } }
}
