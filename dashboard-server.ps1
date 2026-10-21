# ============================================================
# Local AI Dashboard Server
# Compatible with the current C:\AI stack:
#   llama.cpp: native Windows binary -> host :10000
#   cptr native Windows server: host :8000
# Dashboard: http://127.0.0.1:9090
# ============================================================

$ErrorActionPreference = 'Continue'
$Root = 'C:\AI'
$Port = 9090
$LlamaHost = 'http://127.0.0.1:10000'
$CptrHost = 'http://127.0.0.1:8000'
$SwitchboardHost = 'http://127.0.0.1:8002'
$SwitchboardRoot = Join-Path $Root 'Switchboard'
$SwitchboardStart = Join-Path $SwitchboardRoot 'scripts\start.ps1'
$SwitchboardStop = Join-Path $SwitchboardRoot 'scripts\stop.ps1'
$AssistantRoot = Join-Path $Root 'Assistant'
$AssistantPython = Join-Path $AssistantRoot '.venv\Scripts\python.exe'
$TelegramRoot = Join-Path $Root 'TelegramBot'
$TelegramStart = Join-Path $TelegramRoot 'start-telegram-bot.ps1'
$TelegramStop = Join-Path $TelegramRoot 'stop-telegram-bot.ps1'
. (Join-Path $TelegramRoot 'service-management.ps1')
$DashboardFile = Join-Path $Root 'dashboard.html'
$LogDir = Join-Path $Root 'Logs'
$ModelsDir = Join-Path $Root 'models'
$ModelsConfigFile = Join-Path $Root 'Config\models.json'
$LlamaBin = Join-Path $Root 'Tools\llama-native\bin\llama-server.exe'
$LlamaPidFile = Join-Path $Root 'Config\llama-native.pid'
$HealthRoot = Join-Path $Root 'Personal-health-data'
$HealthLauncher = Join-Path $HealthRoot 'start-health.ps1'
$HealthEnvFile = Join-Path $HealthRoot 'oura.env'
$ScriptsLogDir = Join-Path $LogDir 'scripts'
$ScreenshotDir = Join-Path $Root 'Workspace\Screenshots'
$CptrSyncScript = Join-Path $Root 'Tools\scripts\sync-cptr-context.py'
# cptr compacts a chat when it passes this fraction of llama-server's real context. The rest is
# headroom for the model's reply and for cptr's rough (len/4) token estimate.
$CptrCompactRatio = 0.70
$script:CptrSync = @{ nctx = 0; cptrPid = 0; at = [datetime]::MinValue }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $ScriptsLogDir | Out-Null

# In-memory registry of background script runs (name -> {process, outLog, errLog, startedAt}).
# Reset when the dashboard server itself restarts, which is fine -- run history is only
# meant to be live-tailed, not kept forever.
$script:ScriptRuns = @{}

function Test-PortOpen {
    # A closed localhost port does NOT fail fast through Invoke-WebRequest/RestMethod on
    # this system -- it was observed to sit out nearly the entire -TimeoutSec instead of
    # refusing instantly, which is what made /api/status take 6+ seconds once a third
    # (usually-stopped) service check was added. A raw async TCP connect attempt resolves
    # in a few hundred ms either way, so every "is X listening" check probes with this
    # first and only pays for a full HTTP round-trip once a socket is confirmed open.
    param([string]$HostName = '127.0.0.1', [int]$Port, [int]$TimeoutMs = 300)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $iar = $tcp.BeginConnect($HostName, $Port, $null, $null)
        $connected = $iar.AsyncWaitHandle.WaitOne($TimeoutMs) -and $tcp.Connected
        if ($connected) { $tcp.EndConnect($iar) }
        return $connected
    } catch { return $false }
    finally { if ($tcp) { $tcp.Close() } }
}

function Test-Http {
    param([string]$Url, [int]$TimeoutSec = 2)
    try {
        $uri = [Uri]$Url
        if (-not (Test-PortOpen -HostName $uri.Host -Port $uri.Port)) { return $false }
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
        return $true
    } catch { return $false }
}

function Get-PortPid {
    param([int]$Port)
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
        if ($c) { return [int]$c.OwningProcess }
    } catch {}
    return $null
}

function Get-AssistantPort {
    # The Assistant's service port lives in its own settings file (default 18911).
    $port = 18911
    $file = Join-Path $AssistantRoot 'runtime\settings.json'
    if (Test-Path -LiteralPath $file) {
        try { $value = [int](Get-Content -Raw -LiteralPath $file | ConvertFrom-Json).service_port; if ($value -gt 0) { $port = $value } } catch {}
    }
    return $port
}

function Get-OuraPort {
    # Mirrors start-health.ps1's own Load-ServerConfig so status reporting matches
    # reality even if OURA_PORT was customized in oura.env.
    $port = 8765
    if (Test-Path $HealthEnvFile) {
        foreach ($line in Get-Content $HealthEnvFile) {
            if ($line -match '^\s*OURA_PORT\s*=\s*(\d+)\s*$') { $port = [int]$Matches[1] }
        }
    }
    return $port
}

function Get-GpuStats {
    try {
        $raw = & nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>$null
        if (-not $raw) { return $null }
        $parts = ($raw | Select-Object -First 1) -split ',\s*'
        return @{
            name = $parts[0]
            usedMiB = [int]$parts[1]
            totalMiB = [int]$parts[2]
            freeMiB = [int]$parts[2] - [int]$parts[1]
            tempC = [int]$parts[3]
            utilizationPercent = [int]$parts[4]
        }
    } catch { return $null }
}

function Sync-CptrContext {
    # cptr decides when to compact a chat from its own token threshold and never asks llama-server
    # what context it was started with, so that threshold went stale on every model switch
    # (24K-180K here). Tools\scripts\sync-cptr-context.py writes ratio x n_ctx into cptr's database,
    # where cptr re-reads it on every request (no restart). Re-sync when the context changes, when
    # cptr restarts (it re-seeds its database from config.toml at startup and could revert the value),
    # and every 5 minutes as a backstop. A CptrPid of -1 means "don't care".
    param([int]$NCtx = 0, [int]$CptrPid = -1)
    if (-not (Test-Path $CptrSyncScript)) { return }
    if ($NCtx -le 0) {
        try {
            $props = Invoke-RestMethod "$LlamaHost/props" -TimeoutSec 2 -ErrorAction Stop
            $NCtx = [int]$props.default_generation_settings.n_ctx
        } catch { return }
    }
    if ($NCtx -le 0) { return }

    $s = $script:CptrSync
    $stale = ((Get-Date) - $s.at).TotalMinutes -ge 5
    $pidChanged = ($CptrPid -ge 0) -and ($CptrPid -ne $s.cptrPid)
    if ($NCtx -eq $s.nctx -and -not $pidChanged -and -not $stale) { return }

    # Record first, so a slow or failing helper is retried at most every 5 minutes, not every poll.
    $s.nctx = $NCtx
    $s.at = Get-Date
    if ($CptrPid -ge 0) { $s.cptrPid = $CptrPid }
    $syncLog = Join-Path $LogDir 'cptr-sync.log'
    try {
        $py = (Get-Command python -ErrorAction Stop).Source
        $ratio = $CptrCompactRatio.ToString([Globalization.CultureInfo]::InvariantCulture)
        # Not waited on: the helper imports cptr (a few seconds) and this runs on the dashboard's
        # single request thread. It appends its own result line to the log.
        Start-Process -FilePath $py -ArgumentList @($CptrSyncScript, '--n-ctx', "$NCtx", '--ratio', $ratio, '--log', $syncLog) -WindowStyle Hidden | Out-Null
    } catch {
        "$(Get-Date -Format s) failed to launch sync: $($_.Exception.Message)" | Add-Content -Path $syncLog -Encoding UTF8
    }
}

function Get-ServiceStatus {
    $llamaUp = Test-Http "$LlamaHost/v1/models"
    $cptrUp = Test-Http "$CptrHost/openapi.json"
    $switchboardUp = Test-Http "$SwitchboardHost/health"
    $assistantPort = Get-AssistantPort
    $assistantUp = Test-Http "http://127.0.0.1:$assistantPort/"
    $ouraPort = Get-OuraPort
    $ouraUrl = "http://127.0.0.1:$ouraPort"
    $ouraHealth = $null
    if (Test-PortOpen -Port $ouraPort) {
        try { $ouraHealth = Invoke-RestMethod "$ouraUrl/health" -TimeoutSec 2 -ErrorAction Stop } catch {}
    }
    $ouraUp = $null -ne $ouraHealth

    $llamaPid = Get-PortPid 10000
    $cptrPid = Get-PortPid 8000
    $switchboardPid = Get-PortPid 8002
    $ouraPid = Get-PortPid $ouraPort
    # This is the dashboard's heartbeat: the cptr sync is a side feature and must never be able to break it.
    if ($llamaUp) { try { Sync-CptrContext -CptrPid ([int](@($cptrPid)[0])) } catch {} }

    $os = Get-CimInstance Win32_OperatingSystem
    $totalRam = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $usedRam = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
    # One CIM query reused for name/cores/threads -- this used to be three separate
    # Get-CimInstance Win32_Processor calls, each costing roughly a second.
    $cpuInfo = Get-CimInstance Win32_Processor

    return @{
        services = @{
            'llama-server' = @{
                running = $llamaUp
                pid = $llamaPid
                startTime = ''
                port = 10000
                url = $LlamaHost
                error = if (-not $llamaUp) { 'llama-server is not responding' } else { '' }
            }
            'cptr' = @{
                running = $cptrUp
                pid = $cptrPid
                startTime = ''
                port = 8000
                url = $CptrHost
                error = if (-not $cptrUp) { 'cptr OpenAPI endpoint is not responding' } else { '' }
            }
            'switchboard' = @{
                running = $switchboardUp
                pid = $switchboardPid
                startTime = ''
                port = 8002
                url = $SwitchboardHost
                error = if (-not $switchboardUp -and $switchboardPid) { 'Port 8002 is listening, but Switchboard health is not responding' } elseif (-not $switchboardUp) { 'Switchboard is not responding' } else { '' }
            }
            'assistant' = @{
                running = $assistantUp
                pid = Get-PortPid $assistantPort
                startTime = ''
                port = $assistantPort
                url = "http://127.0.0.1:$assistantPort"
                error = if (-not $assistantUp -and (Test-PortOpen -Port $assistantPort)) { "Port $assistantPort is listening, but the Assistant is not responding" } else { '' }
            }
            'telegram-bot' = (Get-RelayServiceStatus -Port 9095)
            'oura-mcp' = @{
                running = $ouraUp
                pid = $ouraPid
                startTime = ''
                port = $ouraPort
                url = $ouraUrl
                authorized = if ($ouraHealth) { [bool]$ouraHealth.authorized } else { $false }
                configured = if ($ouraHealth) { [bool]$ouraHealth.oauth_configured } else { $false }
                error = if (-not $ouraUp) { 'Personal Health MCP is not responding' } else { '' }
            }
        }
        gpu = Get-GpuStats
        system = @{
            cpu = ($cpuInfo | Select-Object -First 1).Name
            cores = ($cpuInfo | Measure-Object NumberOfCores -Sum).Sum
            threads = ($cpuInfo | Measure-Object NumberOfLogicalProcessors -Sum).Sum
            ramGB = $totalRam
            memoryUsedGB = $usedRam
            memoryPercent = if ($totalRam -gt 0) { [math]::Round(($usedRam / $totalRam) * 100, 1) } else { 0 }
            freeDiskGB = [math]::Round($disk.FreeSpace / 1GB, 1)
        }
    }
}

. (Join-Path $Root 'Tools\scripts\Build-LlamaArgs.ps1')

# ---- Remote model backend (the HP server's llama-server) ----
# The browser never sees the API key: the dashboard checks the server's health and relays chat
# requests itself. See Tools\scripts\RemoteBackend.ps1 for where the URL and key come from.
$RemoteBackendHelper = Join-Path $Root 'Tools\scripts\RemoteBackend.ps1'
$RemoteBackendConfigFile = Join-Path $Root 'Config\remote-backend.json'
. $RemoteBackendHelper
$script:RemoteStatusCache = @{ at = [datetime]::MinValue; value = $null }
$script:RemoteProxyJobs = New-Object System.Collections.ArrayList
# A CPU model on the HP can take minutes per reply, and this listener handles one request at a
# time, so chat requests are relayed from a small runspace pool instead of the listener loop.
$script:RemoteProxyPool = [runspacefactory]::CreateRunspacePool(1, 2)
$script:RemoteProxyPool.Open()
$RemoteProxyMaxPending = 4

function Get-RemoteBackendStatusCached {
    # /api/backends is polled with /api/status; a few seconds of caching keeps an offline HP
    # from adding a connect timeout to every refresh.
    param([switch]$Force)
    $age = ([datetime]::UtcNow - $script:RemoteStatusCache.at).TotalSeconds
    if (-not $Force -and $script:RemoteStatusCache.value -and $age -lt 5) { return $script:RemoteStatusCache.value }
    $config = Get-RemoteBackendConfig -ConfigPath $RemoteBackendConfigFile
    $value = Get-RemoteBackendStatus -Config $config
    $script:RemoteStatusCache = @{ at = [datetime]::UtcNow; value = $value }
    return $value
}

function Remove-FinishedRemoteProxyJobs {
    foreach ($job in @($script:RemoteProxyJobs)) {
        if ($job.handle.IsCompleted) {
            try { $job.ps.EndInvoke($job.handle) } catch {}
            $job.ps.Dispose()
            [void]$script:RemoteProxyJobs.Remove($job)
        }
    }
}

function Start-RemoteChatProxy {
    # Hands the request to the pool; the runspace writes and closes the response itself.
    param($Context, [string]$RequestBody)
    Remove-FinishedRemoteProxyJobs
    if ($script:RemoteProxyJobs.Count -ge $RemoteProxyMaxPending) { return $false }
    $config = Get-RemoteBackendConfig -ConfigPath $RemoteBackendConfigFile
    $ps = [powershell]::Create()
    $ps.RunspacePool = $script:RemoteProxyPool
    [void]$ps.AddScript({
        param($Context, $Helper, $Config, $RequestBody)
        $response = $Context.Response
        try {
            . $Helper
            $result = Invoke-RemoteChatProxy -Config $Config -RequestBody $RequestBody
        } catch {
            $result = @{ status = 500; body = (@{ error = @{ message = $_.Exception.Message } } | ConvertTo-Json -Compress) }
        }
        try {
            $bytes = [Text.Encoding]::UTF8.GetBytes([string]$result.body)
            $response.StatusCode = [int]$result.status
            $response.ContentType = 'application/json; charset=utf-8'
            $response.ContentLength64 = $bytes.Length
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
            $response.Close()
        } catch {
            try { $response.Abort() } catch {}
        }
    }).AddArgument($Context).AddArgument($RemoteBackendHelper).AddArgument($config).AddArgument($RequestBody)
    [void]$script:RemoteProxyJobs.Add(@{ ps = $ps; handle = $ps.BeginInvoke() })
    return $true
}

function Stop-LlamaNative {
    $stopped = $false
    $killed = @()
    $existingPid = Get-PortPid 10000
    if ($existingPid) {
        try { Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue; $stopped = $true; $killed += [int]$existingPid } catch {}
    }
    Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        $stopped = $true
        $killed += $_.Id
    }
    # Wait for the processes to actually exit so their VRAM is released before the next model loads;
    # starting early makes the new model spill into system RAM and load very slowly.
    foreach ($id in ($killed | Select-Object -Unique)) { Wait-Process -Id $id -Timeout 30 -ErrorAction SilentlyContinue }
    if ($stopped) { Start-Sleep -Seconds 1 }
    Remove-Item $LlamaPidFile -Force -ErrorAction SilentlyContinue
    return @{ success = $true; message = 'llama-server stopped.' }
}

function Start-LlamaNative {
    param([string]$FileName, $Settings)

    # A model can name its own llama-server build (serverBinary, relative to C:\AI or absolute), e.g. a fork
    # for quant types the normal build rejects. Without it, the normal build runs.
    $bin = $LlamaBin
    $customBin = if ($Settings -is [System.Collections.IDictionary]) { [string]$Settings['serverBinary'] } else { [string]$Settings.serverBinary }
    if (-not [string]::IsNullOrWhiteSpace($customBin)) {
        $bin = if ([System.IO.Path]::IsPathRooted($customBin)) { $customBin } else { Join-Path $Root $customBin }
    }
    if (-not (Test-Path $bin)) {
        return @{ success = $false; message = "llama-server.exe not found at $bin" }
    }
    $modelPath = Join-Path $ModelsDir $FileName
    if (-not (Test-Path $modelPath)) {
        return @{ success = $false; message = "Model file not found: $FileName" }
    }

    # Optional LoRA adapter. Validate before stopping the running server so a bad choice leaves it serving.
    $loraArgs = @()
    $loraFile = if ($Settings -is [System.Collections.IDictionary]) { [string]$Settings['loraFile'] } else { [string]$Settings.loraFile }
    if (-not [string]::IsNullOrWhiteSpace($loraFile)) {
        $loraPath = Resolve-LoraPath -FileName $loraFile
        if (-not $loraPath) {
            return @{ success = $false; message = "LoRA adapter not found: $loraFile (put adapters in C:\AI\models or C:\AI\models\lora)" }
        }
        $modelArch = [string](Get-GgufHeader -File (Get-Item -LiteralPath $modelPath))['general.architecture']
        $loraArch = [string](Get-GgufHeader -File (Get-Item -LiteralPath $loraPath))['general.architecture']
        if ($modelArch -and $loraArch -and $modelArch -ne $loraArch) {
            return @{ success = $false; message = "LoRA $loraFile is for '$loraArch' models, but $FileName is '$modelArch'." }
        }
        $scaleRaw = if ($Settings -is [System.Collections.IDictionary]) { $Settings['loraScale'] } else { $Settings.loraScale }
        $scale = 1.0
        if ($null -ne $scaleRaw -and "$scaleRaw" -ne '') { $scale = [double]::Parse("$scaleRaw", [System.Globalization.CultureInfo]::InvariantCulture) }
        if ($scale -eq 1.0) {
            $loraArgs = @('--lora', $loraPath)
        } else {
            # FNAME:SCALE can't take a drive-letter colon, so pass the path rooted on the current drive.
            $rooted = $loraPath -replace '^[A-Za-z]:', ''
            $loraArgs = @('--lora-scaled', ('{0}:{1}' -f $rooted, $scale.ToString([System.Globalization.CultureInfo]::InvariantCulture)))
        }
    }

    Stop-LlamaNative | Out-Null

    $argList = @(Build-LlamaArgs -ModelPath $modelPath -Settings $Settings) + $loraArgs
    $stdOut = Join-Path $LogDir 'llama.stdout.log'
    $stdErr = Join-Path $LogDir 'llama.stderr.log'

    try {
        $p = Start-Process -FilePath $bin -ArgumentList $argList -WorkingDirectory (Split-Path $bin) -RedirectStandardOutput $stdOut -RedirectStandardError $stdErr -WindowStyle Hidden -PassThru
        try { $p.PriorityClass = 'High' } catch {}
        Set-Content -Path $LlamaPidFile -Value $p.Id -Encoding UTF8
        # Tell cptr the new context right away (the model is still loading, so /props isn't up yet);
        # the status poll re-checks against the real n_ctx once it is.
        try {
            $ctxSize = if ($Settings -is [System.Collections.IDictionary]) { [int]$Settings['contextSize'] } else { [int]$Settings.contextSize }
            $slots = if ($Settings -is [System.Collections.IDictionary]) { [int]$Settings['parallel'] } else { [int]$Settings.parallel }
            if ($slots -lt 1) { $slots = 1 }
            if ($ctxSize -gt 0) { Sync-CptrContext -NCtx ([int]($ctxSize / $slots)) }
        } catch {}
        return @{ success = $true; message = "llama-server starting natively (PID $($p.Id)). Loading $FileName..." }
    } catch {
        return @{ success = $false; message = "Failed to start llama-server: $($_.Exception.Message)" }
    }
}

function Start-ServiceByName {
    param([string]$Name)
    switch ($Name) {
        'assistant' {
            $port = Get-AssistantPort
            if (Test-Http "http://127.0.0.1:$port/") { return @{ success=$true; message='Assistant is already running.' } }
            if (-not (Test-Path -LiteralPath $AssistantPython)) {
                return @{ success=$false; message='The Assistant Python environment is missing. Run C:\AI\Assistant\scripts\install.ps1 first.' }
            }
            # The Assistant's own launcher (the same one the desktop shortcut uses): it checks free memory, starts the
            # service detached and refuses a port owned by another program. It can take a while, so it runs detached.
            $outLog = Join-Path $ScriptsLogDir 'assistant-launcher.stdout.log'
            $errLog = Join-Path $ScriptsLogDir 'assistant-launcher.stderr.log'
            try {
                $p = Start-Process -FilePath $AssistantPython -ArgumentList @('-m','assistant.cli','--root',$AssistantRoot,'start') `
                    -WorkingDirectory $AssistantRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
                return @{ success=$true; message="Assistant starting (launcher PID $($p.Id)). If memory is tight it starts dashboard only; the Assistant's own page has Start the assistant." }
            } catch { return @{ success=$false; message="Failed to start Assistant: $($_.Exception.Message)" } }
        }
        'telegram-bot' {
            if (-not (Test-Path -LiteralPath $TelegramStart)) {
                return @{success=$false;message='Relay Station launcher is missing.'}
            }
            $current = Get-RelayServiceStatus -Port 9095
            if ($current.ready) {
                # Adopt only a verified bot command/console, so Stop can also
                # manage the older instance that predates the PID record.
                $process = Get-RelayPortProcess -Port 9095
                try { Save-RelayInstance -Process $process -Port 9095 -AllowLegacy }
                catch { return @{success=$false;message=$_.Exception.Message} }
                return @{success=$true;message='Relay Station is already running.'}
            }
            if ($current.pid) { return @{success=$false;message='Port 9095 belongs to an unrecognized or unready process.'} }
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
            try {
                $p = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -PassThru -WorkingDirectory $TelegramRoot `
                    -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$TelegramStart) `
                    -RedirectStandardOutput (Join-Path $ScriptsLogDir "telegram-$stamp-out.log") `
                    -RedirectStandardError (Join-Path $ScriptsLogDir "telegram-$stamp-error.log")
                return @{success=$true;message="Relay Station starting (launcher PID $($p.Id))."}
            } catch { return @{success=$false;message="Failed to start Relay Station: $($_.Exception.Message)"} }
        }
        'llama-server' {
            $config = Get-ModelsConfig
            if ([string]::IsNullOrWhiteSpace($config.activeModel)) {
                return @{ success = $false; message = 'No active model set. Load a model from the Models tab first.' }
            }
            $settings = Get-ModelSettingsFor -Config $config -FileName $config.activeModel
            return Start-LlamaNative -FileName $config.activeModel -Settings $settings
        }
        'cptr' {
            if (Test-Http "$CptrHost/openapi.json") { return @{ success=$true; message='cptr is already running.' } }
            try {
                $p = Start-Process -FilePath 'cptr' -ArgumentList @('run','--host','127.0.0.1','--port','8000','--headless') -WorkingDirectory $Root -WindowStyle Hidden -PassThru
                Start-Sleep -Seconds 2
                if (Test-Http "$CptrHost/openapi.json") { return @{ success=$true; message="cptr started (PID $($p.Id))." } }
                return @{ success=$false; message='cptr process started but its OpenAPI endpoint did not become ready.' }
            } catch { return @{ success=$false; message="Failed to start cptr: $($_.Exception.Message)" } }
        }
        'switchboard' {
            if (Test-Http "$SwitchboardHost/health") { return @{ success=$true; message='Switchboard is already running.' } }
            if (-not (Test-Path $SwitchboardStart)) {
                return @{ success=$false; message="Switchboard launcher not found at $SwitchboardStart" }
            }
            # Delegate to Switchboard's own guarded launcher. It refuses a foreign
            # port owner and never starts or restarts llama-server.
            $outLog = Join-Path $ScriptsLogDir 'switchboard-launcher.stdout.log'
            $errLog = Join-Path $ScriptsLogDir 'switchboard-launcher.stderr.log'
            try {
                $p = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$SwitchboardStart) `
                    -WorkingDirectory $SwitchboardRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
                return @{ success=$true; message="Switchboard starting (launcher PID $($p.Id)); llama-server will not be restarted." }
            } catch { return @{ success=$false; message="Failed to start Switchboard: $($_.Exception.Message)" } }
        }
        'oura-mcp' {
            if (-not (Test-Path $HealthLauncher)) {
                return @{ success=$false; message="Personal Health MCP launcher not found at $HealthLauncher" }
            }
            $ouraPort = Get-OuraPort
            if (Test-Http "http://127.0.0.1:$ouraPort/health") { return @{ success=$true; message='Personal Health MCP is already running.' } }
            # start-health.ps1's default mode blocks (git pull, dependency install, health poll,
            # up to ~45s) before it prints "running" -- but that happens inside ITS OWN process,
            # so launching it via Start-Process here does not block this dashboard's request loop.
            $outLog = Join-Path $ScriptsLogDir 'oura-mcp-launcher.stdout.log'
            $errLog = Join-Path $ScriptsLogDir 'oura-mcp-launcher.stderr.log'
            try {
                $p = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$HealthLauncher) `
                    -WorkingDirectory $HealthRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
                return @{ success=$true; message="Personal Health MCP starting (PID $($p.Id)). First start can take a minute (dependency check + Oura health poll)." }
            } catch { return @{ success=$false; message="Failed to start Personal Health MCP: $($_.Exception.Message)" } }
        }
        default { return @{ success=$false; message="Unknown service: $Name" } }
    }
}

function Stop-ServiceByName {
    param([string]$Name)
    switch ($Name) {
        'assistant' {
            $port = Get-AssistantPort
            if (-not (Test-PortOpen -Port $port)) { return @{ success=$true; message='Assistant is not running.' } }
            if (-not (Test-Path -LiteralPath $AssistantPython)) {
                return @{ success=$false; message='The Assistant Python environment is missing, so it cannot be stopped from here.' }
            }
            # `assistant stop` shuts down gracefully (up to 4 minutes while tasks settle) and only ever stops processes the
            # Assistant itself owns. Run it detached and wait briefly so this request loop is never held for minutes.
            $outLog = Join-Path $ScriptsLogDir 'assistant-stop.stdout.log'
            $errLog = Join-Path $ScriptsLogDir 'assistant-stop.stderr.log'
            try {
                $p = Start-Process -FilePath $AssistantPython -ArgumentList @('-m','assistant.cli','--root',$AssistantRoot,'stop') `
                    -WorkingDirectory $AssistantRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
                for ($i = 0; $i -lt 40 -and (Test-PortOpen -Port $port); $i++) { Start-Sleep -Milliseconds 500 }
                if (Test-PortOpen -Port $port) { return @{ success=$true; message='Assistant is shutting down (waiting for running tasks to settle).' } }
                return @{ success=$true; message='Assistant stopped.' }
            } catch { return @{ success=$false; message="Failed to stop Assistant: $($_.Exception.Message)" } }
        }
        'telegram-bot' {
            if (-not (Test-Path -LiteralPath $TelegramStop)) {
                return @{success=$false;message='Relay Station stop script is missing.'}
            }
            try {
                $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $TelegramStop -Force 2>&1
                if ($LASTEXITCODE -ne 0) { return @{success=$false;message=($output | Out-String).Trim()} }
                if (Get-RelayPortProcess -Port 9095) { return @{success=$false;message='Relay Station has not stopped; restart refused.'} }
                return @{success=$true;message=($output | Out-String).Trim()}
            } catch { return @{success=$false;message="Failed to stop Relay Station: $($_.Exception.Message)"} }
        }
        'llama-server' { return Stop-LlamaNative }
        'cptr' {
            $cptrPid = Get-PortPid 8000
            if ($cptrPid) { Stop-Process -Id $cptrPid -Force -ErrorAction SilentlyContinue }
            return @{ success=$true; message='cptr stopped.' }
        }
        'switchboard' {
            if (-not (Test-Path $SwitchboardStop)) {
                return @{ success=$false; message="Switchboard stopper not found at $SwitchboardStop" }
            }
            try {
                $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SwitchboardStop 2>&1
                if (Test-Http "$SwitchboardHost/health") {
                    return @{ success=$false; message='Switchboard is still responding after its guarded stop command.' }
                }
                $message = ($output | ForEach-Object { [string]$_ }) -join ' '
                if ([string]::IsNullOrWhiteSpace($message)) { $message = 'Switchboard stopped.' }
                return @{ success=$true; message=$message.Trim() }
            } catch { return @{ success=$false; message="Failed to stop Switchboard: $($_.Exception.Message)" } }
        }
        'oura-mcp' {
            if (-not (Test-Path $HealthLauncher)) {
                return @{ success=$false; message="Personal Health MCP launcher not found at $HealthLauncher" }
            }
            try {
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $HealthLauncher -Stop 2>&1 | Out-Null
                return @{ success=$true; message='Personal Health MCP stopped.' }
            } catch { return @{ success=$false; message="Failed to stop Personal Health MCP: $($_.Exception.Message)" } }
        }
        default { return @{ success=$false; message="Unknown service: $Name" } }
    }
}


# ---------------- Model Tuner integration ----------------
$TunerRoot = Join-Path $Root 'Tools\model-tuner'
$TunerState = Join-Path $TunerRoot 'state'
$TunerPython = (Get-Command python -ErrorAction SilentlyContinue).Source

$script:TunerLaunchPid = $null

function Test-TunerRunning {
    # The tuner writes current.json only once it has started (a second or two after launch), so the process this
    # dashboard launched counts as running too: a double-click on Start must not launch two tuners.
    if ($script:TunerLaunchPid -and (Get-Process -Id $script:TunerLaunchPid -ErrorAction SilentlyContinue)) { return $true }
    $cur = Join-Path $TunerState 'current.json'
    if (-not (Test-Path $cur)) { return $false }
    try { $s = Get-Content $cur -Raw | ConvertFrom-Json } catch { return $false }
    if ($s.status -notin @('running', 'starting')) { return $false }
    return [bool](Get-Process -Id ([int]$s.pid) -ErrorAction SilentlyContinue)
}

function Start-TunerProcess {
    param([string[]]$TunerArgs, [string]$LogName)
    if (-not $TunerPython) { throw 'python was not found on PATH' }
    New-Item -ItemType Directory -Force -Path $TunerState | Out-Null
    $log = Join-Path $TunerState $LogName
    $p = Start-Process -FilePath $TunerPython -ArgumentList (@('-m', 'modeltuner') + $TunerArgs) -WorkingDirectory $TunerRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    return $p
}

function Invoke-TunerAction {
    param($Request)
    $action = [string]$Request.action
    $model = [string]$Request.model
    if ($model -and ($model -match '[\/:]' -or $model -match '\.\.' -or -not (Test-Path (Join-Path $ModelsDir $model)))) {
        return @{ success = $false; message = "Unknown model: $model" }
    }
    switch ($action) {
        'start' {
            if (Test-TunerRunning) { return @{ success = $false; message = 'A tuning run is already in progress.' } }
            if (-not $model) { return @{ success = $false; message = 'Pick a model first.' } }
            $mode = [string]$Request.mode
            if ($mode -notin @('quick', 'medium', 'full')) { $mode = 'quick' }
            $a = @('run', $model, '--mode', $mode)
            if ($Request.ctx) { $a += @('--ctx', [string][int]$Request.ctx) }
            try { $p = Start-TunerProcess -TunerArgs $a -LogName 'last-run.log'; $script:TunerLaunchPid = $p.Id }
            catch { return @{ success = $false; message = "Could not start the tuner: $($_.Exception.Message)" } }
            return @{ success = $true; message = "Tuning $model ($mode) started (PID $($p.Id))."; pid = $p.Id }
        }
        'stop' {
            New-Item -ItemType Directory -Force -Path $TunerState | Out-Null
            Set-Content -Path (Join-Path $TunerState 'STOP') -Value (Get-Date -Format s) -Encoding UTF8
            return @{ success = $true; message = 'Stop requested. The run restores your model and writes its report.' }
        }
        'plan' {
            if (-not $model) { return @{ success = $false; message = 'Pick a model first.' } }
            if (Test-TunerRunning) { return @{ success = $false; message = 'A tuning run is in progress; plan when it finishes.' } }
            try { $p = Start-TunerProcess -TunerArgs @('plan', $model) -LogName 'last-plan.txt' }
            catch { return @{ success = $false; message = "Could not start the planner: $($_.Exception.Message)" } }
            return @{ success = $true; message = "Planning $model (about 30-40 seconds, nothing is loaded)." }
        }
        'apply' {
            $runId = [string]$Request.run_id
            if ($runId -notmatch '^[\w.\-]+$' -or -not (Test-Path (Join-Path $TunerState "runs\$runId\report.json"))) {
                return @{ success = $false; message = "Unknown run: $runId" }
            }
            if (-not $TunerPython) { return @{ success = $false; message = 'python was not found on PATH' } }
            Push-Location $TunerRoot
            $applyArgs = @('-m', 'modeltuner', 'apply', $runId, '--yes')
            if ($Request.switchboard) { $applyArgs += '--switchboard' }
            try { $out = & $TunerPython @applyArgs 2>&1 | Out-String } finally { Pop-Location }
            return @{ success = ($LASTEXITCODE -eq 0); message = $out.Trim() }
        }
        default { return @{ success = $false; message = "Unknown action: $action" } }
    }
}

function Get-TunerStatus {
    $result = [ordered]@{ running = (Test-TunerRunning); current = $null; events = @(); report = $null; runs = @(); plan = '' }
    $cur = Join-Path $TunerState 'current.json'
    if (Test-Path $cur) {
        try { $result.current = Get-Content $cur -Raw | ConvertFrom-Json } catch {}
    }
    if ($result.current -and $result.current.status -in @('running', 'starting') -and -not $result.running) {
        # current.json still says running, but no tuner process is alive: it crashed or was killed.
        $result.current.status = 'interrupted'
        $result.current.message = 'The tuning process is no longer running (it stopped without finishing).'
    }
    if ($result.current -and $result.current.run_id -and ([string]$result.current.run_id) -match '^[\w.\-]+$') {
        $runDir = Join-Path $TunerState ("runs\" + $result.current.run_id)
        $ev = Join-Path $runDir 'events.jsonl'
        if (Test-Path $ev) {
            $result.events = @(Get-Content $ev -Tail 60 | ForEach-Object { try { $_ | ConvertFrom-Json } catch {} } |
                Select-Object t, kind, msg)
        }
        $rj = Join-Path $runDir 'report.json'
        if (Test-Path $rj) { try { $result.report = Get-Content $rj -Raw | ConvertFrom-Json | Select-Object run_id, model, status, mode, improvement, baseline_metrics, best_metrics, changes, decisions, notes, seconds, loads } catch {} }
    }
    $runsDir = Join-Path $TunerState 'runs'
    if (Test-Path $runsDir) {
        $result.runs = @(Get-ChildItem $runsDir -Directory | Sort-Object Name -Descending | Select-Object -First 10 | ForEach-Object {
            $r = Join-Path $_.FullName 'report.json'
            if (Test-Path $r) { try { $j = Get-Content $r -Raw | ConvertFrom-Json; [ordered]@{ run_id = $_.Name; model = $j.model; status = $j.status; improvement = $j.improvement; mode = $j.mode; applied = (Test-Path (Join-Path $_.FullName 'applied.json')) } } catch {} }
            else { [ordered]@{ run_id = $_.Name; status = 'incomplete' } }
        })
    }
    $plan = Join-Path $TunerState 'last-plan.txt'
    if (Test-Path $plan) { $result.plan = (Get-Content $plan -Raw) }
    return $result
}

function Run-QuickBenchmark {
    $results = @()
    try {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $models = Invoke-RestMethod "$LlamaHost/v1/models" -TimeoutSec 5 -ErrorAction Stop
        $sw.Stop()
        $results += @{ test='health'; status='ok'; response_ms=$sw.ElapsedMilliseconds; model=if($models.data){$models.data[0].id}else{''} }
    } catch { $results += @{ test='health'; status='error'; error=$_.Exception.Message } }

    try {
        $body = @{ model='local'; messages=@(@{role='user';content='Reply with exactly: OK'}); max_tokens=8; temperature=0; stream=$false } | ConvertTo-Json -Depth 10
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod "$LlamaHost/v1/chat/completions" -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 60 -ErrorAction Stop
        $sw.Stop()
        $results += @{ test='generation'; status='ok'; response_ms=$sw.ElapsedMilliseconds; response=($r.choices[0].message.content) }
    } catch { $results += @{ test='generation'; status='error'; error=$_.Exception.Message } }

    $file = Join-Path $Root 'benchmark-results.json'
    $results | ConvertTo-Json -Depth 10 | Set-Content $file -Encoding UTF8
    return @{ benchmark=$results; saved=$file }
}

# Per-model settings that are stored only when set: a non-default llama-server build and a LoRA adapter.
$OptionalModelKeys = @('serverBinary', 'loraFile', 'loraScale')

function Get-ModelDefaults {
    return [ordered]@{
        contextSize   = 16384
        gpuLayers     = 999
        flashAttn     = $true
        parallel      = 1
        jinja         = $true
        batchSize     = 2048
        ubatchSize    = 512
        ropeFreqBase  = 0
        ropeFreqScale = 0
        threads       = 8
        cacheTypeK    = 'f16'
        cacheTypeV    = 'f16'
        reasoningEffort = 'default'
        extraArgs     = ''
    }
}

function Get-ModelsConfig {
    if (-not (Test-Path $ModelsConfigFile)) {
        $default = [ordered]@{ activeModel = ''; models = [ordered]@{} }
        $default | ConvertTo-Json -Depth 10 | Set-Content $ModelsConfigFile -Encoding UTF8
        return $default
    }
    try {
        $raw = Get-Content $ModelsConfigFile -Raw -Encoding UTF8
        if ([string]::IsNullOrWhiteSpace($raw)) { return [ordered]@{ activeModel = ''; models = [ordered]@{} } }
        return ($raw | ConvertFrom-Json)
    } catch {
        return [ordered]@{ activeModel = ''; models = [ordered]@{} }
    }
}

function Save-ModelsConfig {
    param($Config)
    $Config | ConvertTo-Json -Depth 10 | Set-Content $ModelsConfigFile -Encoding UTF8
}

function Get-ModelSettingsFor {
    param($Config, [string]$FileName)
    $defaults = Get-ModelDefaults
    if ($Config.models -and ($Config.models.PSObject.Properties.Name -contains $FileName)) {
        $stored = $Config.models.$FileName
        $merged = [ordered]@{}
        foreach ($key in $defaults.Keys) {
            if ($stored.PSObject.Properties.Name -contains $key) {
                $merged[$key] = $stored.$key
            } else {
                $merged[$key] = $defaults[$key]
            }
        }
        foreach ($opt in $OptionalModelKeys) {
            if (($stored.PSObject.Properties.Name -contains $opt) -and -not [string]::IsNullOrWhiteSpace([string]$stored.$opt)) {
                $merged[$opt] = $stored.$opt
            }
        }
        return $merged
    }
    return $defaults
}

$script:GgufHeaderCache = @{}

function Get-GgufHeader {
    # The leading GGUF metadata (general.architecture, general.type, general.name, adapter.*), read without
    # loading the file. llama.cpp writes the general.* keys first, so stopping at the first array is enough.
    param([System.IO.FileInfo]$File)
    $key = "$($File.FullName)|$($File.Length)|$($File.LastWriteTimeUtc.Ticks)"
    if ($script:GgufHeaderCache.ContainsKey($key)) { return $script:GgufHeaderCache[$key] }
    $meta = @{}
    $fs = $null
    try {
        $fs = [System.IO.File]::OpenRead($File.FullName)
        $br = New-Object System.IO.BinaryReader($fs)
        if ([System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4)) -eq 'GGUF') {
            [void]$br.ReadUInt32(); [void]$br.ReadUInt64(); $kvCount = $br.ReadUInt64()
            $fixed = @{ 0 = 1; 1 = 1; 2 = 2; 3 = 2; 4 = 4; 5 = 4; 7 = 1; 10 = 8; 11 = 8; 12 = 8 }
            for ($i = 0; $i -lt [Math]::Min([double]$kvCount, 24); $i++) {
                $name = [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$br.ReadUInt64()))
                $type = [int]$br.ReadUInt32()
                if ($type -eq 8) { $meta[$name] = [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$br.ReadUInt64())) }
                elseif ($type -eq 6) { $meta[$name] = $br.ReadSingle() }
                elseif ($fixed.ContainsKey($type)) { [void]$br.ReadBytes($fixed[$type]) }
                else { break }
            }
        }
    } catch { } finally { if ($fs) { $fs.Dispose() } }
    $script:GgufHeaderCache[$key] = $meta
    return $meta
}

function Resolve-LoraPath {
    # LoRA adapters live in C:\AI\models or C:\AI\models\lora; a bare file name only, never a path.
    param([string]$FileName)
    if ([string]::IsNullOrWhiteSpace($FileName) -or $FileName -match '[\\/:]' -or $FileName -match '\.\.') { return $null }
    foreach ($dir in @($ModelsDir, (Join-Path $ModelsDir 'lora'))) {
        $p = Join-Path $dir $FileName
        if (Test-Path -LiteralPath $p -PathType Leaf) { return $p }
    }
    return $null
}

function Get-LoraAdapters {
    $result = @()
    foreach ($dir in @($ModelsDir, (Join-Path $ModelsDir 'lora'))) {
        foreach ($f in (Get-ChildItem $dir -Filter '*.gguf' -File -ErrorAction SilentlyContinue | Sort-Object Name)) {
            $hdr = Get-GgufHeader -File $f
            if ($hdr['general.type'] -ne 'adapter') { continue }
            $result += [ordered]@{
                filename     = $f.Name
                sizeGB       = [math]::Round($f.Length / 1GB, 2)
                architecture = [string]$hdr['general.architecture']
                baseModel    = [string]$hdr['general.name']
                alpha        = $hdr['adapter.lora.alpha']
            }
        }
    }
    return @{ adapters = $result }
}

function Get-AvailableModels {
    $config = Get-ModelsConfig
    $files = Get-ChildItem $ModelsDir -Filter '*.gguf' -File -ErrorAction SilentlyContinue
    $result = @()
    foreach ($f in ($files | Sort-Object Name)) {
        $hdr = Get-GgufHeader -File $f
        if ($hdr['general.type'] -eq 'adapter') { continue }
        $settings = Get-ModelSettingsFor -Config $config -FileName $f.Name
        $hasConfig = $config.models -and ($config.models.PSObject.Properties.Name -contains $f.Name)
        $result += [ordered]@{
            filename     = $f.Name
            sizeGB       = [math]::Round($f.Length / 1GB, 2)
            lastModified = $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
            isActive     = ($f.Name -eq $config.activeModel)
            hasCustomConfig = [bool]$hasConfig
            architecture = [string]$hdr['general.architecture']
            settings     = $settings
        }
    }
    return @{ models = $result; activeModel = $config.activeModel }
}


function Apply-ModelSettings {
    param([string]$FileName)

    $modelPath = Join-Path $ModelsDir $FileName
    if (-not (Test-Path $modelPath)) {
        return @{ success = $false; message = "Model file not found: $FileName" }
    }

    $config = Get-ModelsConfig
    $settings = Get-ModelSettingsFor -Config $config -FileName $FileName

    if (-not $config.models) { $config | Add-Member -NotePropertyName models -NotePropertyValue ([ordered]@{}) -Force }
    $config.models | Add-Member -NotePropertyName $FileName -NotePropertyValue $settings -Force
    $config.activeModel = $FileName
    Save-ModelsConfig -Config $config

    return Start-LlamaNative -FileName $FileName -Settings $settings
}

function Start-ModelTrial {
    # Loads a model with per-request setting overrides without writing models.json or changing activeModel.
    # Model Tuner uses this for every trial so the user's saved settings are never touched by a test.
    param([string]$FileName, $Overrides)
    $modelPath = Join-Path $ModelsDir $FileName
    if (-not (Test-Path $modelPath)) {
        return @{ success = $false; message = "Model file not found: $FileName" }
    }
    $settings = Get-ModelSettingsFor -Config (Get-ModelsConfig) -FileName $FileName
    $overrideExtra = [string]$Overrides.extraArgs
    if ($overrideExtra -match '(^|\s)(-m|--model|-md|--model-draft|--spec-draft-model|--mmproj|--lora|--lora-scaled|-hf|-hfr|--hf-repo|--hf-file|--host|--port)(=|\s|$)') {
        return @{ success = $false; message = "Trial extraArgs may not choose the model, a LoRA, or the host/port: $overrideExtra" }
    }
    $allowed = @((Get-ModelDefaults).Keys) + $OptionalModelKeys
    foreach ($p in $Overrides.PSObject.Properties) {
        if ($allowed -contains $p.Name) { $settings[$p.Name] = $p.Value }
    }
    # The RAM-safety flags are not optional, whatever the overrides say.
    $extra = [string]$settings['extraArgs']
    if ($extra -notmatch '--load-mode') { $extra = "$extra --load-mode none".Trim() }
    if ($extra -notmatch '--cache-ram') { $extra = "$extra --cache-ram 0".Trim() }
    $settings['extraArgs'] = $extra
    $result = Start-LlamaNative -FileName $FileName -Settings $settings
    $result['trial'] = $true
    return $result
}

function Save-ModelConfigOnly {
    param([string]$FileName, $Settings)
    $config = Get-ModelsConfig
    $defaults = Get-ModelDefaults
    $merged = [ordered]@{}
    foreach ($key in $defaults.Keys) {
        if ($Settings.PSObject.Properties.Name -contains $key) {
            $merged[$key] = $Settings.$key
        } else {
            $merged[$key] = $defaults[$key]
        }
    }
    # Optional keys are stored only when set. A caller that doesn't send one (the Models tab never sends
    # serverBinary) keeps the saved value; sending an empty string clears it.
    $existing = $null
    if ($config.models -and ($config.models.PSObject.Properties.Name -contains $FileName)) { $existing = $config.models.$FileName }
    foreach ($opt in $OptionalModelKeys) {
        $value = $null
        if ($Settings.PSObject.Properties.Name -contains $opt) { $value = $Settings.$opt }
        elseif ($existing -and ($existing.PSObject.Properties.Name -contains $opt)) { $value = $existing.$opt }
        if (-not [string]::IsNullOrWhiteSpace([string]$value)) { $merged[$opt] = $value }
    }
    if (-not $merged.Contains('loraFile')) { $merged.Remove('loraScale') }
    if (-not $config.models) { $config | Add-Member -NotePropertyName models -NotePropertyValue ([ordered]@{}) -Force }
    $config.models | Add-Member -NotePropertyName $FileName -NotePropertyValue $merged -Force
    Save-ModelsConfig -Config $config
    return @{ success = $true; message = "Settings saved for $FileName." }
}


function Get-LogLines {
    param([string]$FileName)
    if ([string]::IsNullOrWhiteSpace($FileName)) { return @{ lines=@(); error='No log file specified.' } }
    $safe = Split-Path $FileName -Leaf
    $file = Join-Path $LogDir $safe
    if (Test-Path $file) {
        # Get-Content decorates each line with PSPath/PSDrive/PSProvider/ReadCount
        # NoteProperties -- invisible when printed (ToString() hides them) but
        # ConvertTo-Json faithfully serializes the whole decorated object, not
        # just the string. [string]$_ strips that back down to a plain string,
        # which is what the frontend's JS string methods actually expect.
        $lines = @(Get-Content $file -Tail 100 | ForEach-Object { [string]$_ })
        return @{ lines = $lines }
    }
    return @{ lines=@(); error="Log file not found: $safe" }
}

# ------------------------------------------------------------
# Scripts catalog -- a deliberately curated whitelist, not "every
# script in the repo." Excluded on purpose:
#   - setup.ps1: prompts interactively (Read-Host) and copies/relocates
#     the whole project; not safe to trigger from a running dashboard.
#   - start.ps1 / stop.ps1 / start-computer.ps1: redundant with the
#     per-service Start/Stop/Restart buttons already on the Services
#     tab, and start.ps1 ends with its own Read-Host prompt.
#   - model-manage.ps1: redundant with the Models tab's own load/save UI.
#   - windows-control.ps1 mouse/type/key: these inject real clicks and
#     keystrokes into whatever window has focus on the real desktop.
#     cptr already exposes this deliberately, one call at a time; a
#     dashboard button is an easy way to send a keystroke storm into
#     the wrong window by accident. apps/screenshot/launch/focus are
#     included since they carry much less blast radius.
# ------------------------------------------------------------

$ScriptCatalog = [ordered]@{
    'diagnostics' = @{
        label = 'Diagnostics'
        description = 'Quick health check of every service, the GPU, and the Python/cptr install.'
        path = (Join-Path $Root 'diagnostics.ps1')
        mode = 'sync'
        fixedArgs = @()
        argSpec = @()
    }
    'update' = @{
        label = 'Update Stack'
        description = 'Update llama.cpp and cptr, then restart llama-server. Can take a few minutes.'
        path = (Join-Path $Root 'update.ps1')
        mode = 'background'
        fixedArgs = @()
        argSpec = @()
    }
    'model-compare' = @{
        label = 'Model Compare'
        # Always background, even for "list": this script calls back into this very
        # dashboard's own /api/models -- running it synchronously would block the
        # single-threaded request loop on itself and that callback could never be
        # serviced, so every invocation deadlocked until this was backgrounded.
        description = '"list" shows models. "compare"/"quick" load every model in turn and compare replies -- slow, restarts llama-server repeatedly. All run in the background.'
        path = (Join-Path $Root 'model-compare.ps1')
        mode = 'background'
        fixedArgs = @()
        argSpec = @('command', 'prompt')
    }
    'windows-apps' = @{
        label = 'List Open Windows'
        description = 'List visible top-level windows and their process IDs.'
        path = (Join-Path $Root 'Tools\scripts\windows-control.ps1')
        mode = 'sync'
        fixedArgs = @('apps')
        argSpec = @()
    }
    'windows-screenshot' = @{
        label = 'Screenshot'
        description = 'Capture the current desktop.'
        path = (Join-Path $Root 'Tools\scripts\windows-control.ps1')
        mode = 'sync'
        fixedArgs = @('screenshot')
        argSpec = @()
    }
    'windows-launch' = @{
        label = 'Launch App'
        description = 'Start an application by name or path (e.g. notepad.exe).'
        path = (Join-Path $Root 'Tools\scripts\windows-control.ps1')
        mode = 'sync'
        fixedArgs = @('launch')
        argSpec = @('target')
    }
    'windows-focus' = @{
        label = 'Focus Window'
        description = 'Bring a window to the foreground by (partial) title match.'
        path = (Join-Path $Root 'Tools\scripts\windows-control.ps1')
        mode = 'sync'
        fixedArgs = @('focus')
        argSpec = @('target')
    }
}

function Invoke-SyncScript {
    param([string]$Path, [string[]]$Arguments, [int]$TimeoutSec = 30)
    $job = Start-Job -ScriptBlock {
        param($p, $a)
        # *>&1 merges EVERY stream (success, error, warning, verbose, debug, and
        # -- critically -- Information, which is what Write-Host writes to in
        # PowerShell 5.1+). These scripts use Write-Host for virtually all of
        # their user-facing output, so 2>&1 alone silently dropped almost
        # everything: only genuine external-process stdout (e.g. nvidia-smi)
        # or pipeline output (e.g. Format-Table) came through.
        & $p @a *>&1 | Out-String
    } -ArgumentList $Path, $Arguments
    try {
        if (Wait-Job $job -Timeout $TimeoutSec) {
            $out = Receive-Job $job -ErrorAction SilentlyContinue
            return @{ success = $true; output = [string]$out }
        }
        Stop-Job $job -ErrorAction SilentlyContinue
        return @{ success = $false; output = "Timed out after $TimeoutSec seconds." }
    } finally {
        Remove-Job $job -Force -ErrorAction SilentlyContinue
    }
}

function Start-BackgroundScript {
    param([string]$RunId, [string]$Path, [string[]]$Arguments)
    $outLog = Join-Path $ScriptsLogDir "$RunId.out.log"
    $errLog = Join-Path $ScriptsLogDir "$RunId.err.log"
    try {
        $p = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Path) + $Arguments) `
            -WorkingDirectory (Split-Path $Path) -WindowStyle Hidden `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
        $script:ScriptRuns[$RunId] = @{ process = $p; outLog = $outLog; errLog = $errLog; startedAt = Get-Date }
        return @{ success = $true; runId = $RunId; message = "Started (PID $($p.Id))." }
    } catch {
        return @{ success = $false; message = "Failed to start: $($_.Exception.Message)" }
    }
}

function Get-ScriptRunStatus {
    param([string]$RunId)
    if (-not $script:ScriptRuns.Contains($RunId)) {
        return @{ success = $false; message = 'Unknown runId (dashboard may have restarted since this run started).' }
    }
    $run = $script:ScriptRuns[$RunId]
    $p = $run.process
    $p.Refresh()
    $running = -not $p.HasExited

    # Every value below is fully computed into its own variable first. Windows
    # PowerShell 5.1 was observed to hang indefinitely (reproduced in isolation,
    # not a one-off) when an `if (...) {...} else {...}` was inlined directly as
    # a hashtable value -- e.g. `@{ x = if (...) {...} else {...} }` or
    # `@{ x = @(if (...) {...} else {...}) }`. Assigning the if/else result to a
    # plain variable first and referencing that variable in the hashtable avoids
    # it entirely; this bit both this function and the /api/gpu route earlier.
    $exitCode = $null
    if (-not $running) { $exitCode = $p.ExitCode }

    # [string]$_ strips Get-Content's PSPath/PSDrive/PSProvider decoration back
    # down to plain strings -- see the matching note in Get-LogLines.
    $outLines = @()
    if (Test-Path $run.outLog) { $outLines = @(Get-Content $run.outLog -Tail 300 | ForEach-Object { [string]$_ }) }

    $errLines = @()
    if (Test-Path $run.errLog) { $errLines = @(Get-Content $run.errLog -Tail 300 | ForEach-Object { [string]$_ }) }

    return @{
        success = $true
        running = $running
        exitCode = $exitCode
        output = $outLines
        errorOutput = $errLines
    }
}

function Invoke-CatalogScript {
    param([string]$Name, [string[]]$ScriptArgs)
    if (-not $ScriptCatalog.Contains($Name)) {
        return @{ success = $false; message = "Unknown script: $Name" }
    }
    $entry = $ScriptCatalog[$Name]
    if (-not (Test-Path $entry.path)) {
        return @{ success = $false; message = "Script not found: $($entry.path)" }
    }
    $fixed = @($entry.fixedArgs)
    $fullArgs = @($fixed + @($ScriptArgs))

    if ($entry.mode -eq 'sync') {
        $result = Invoke-SyncScript -Path $entry.path -Arguments $fullArgs -TimeoutSec 30
        return @{ success = $result.success; mode = 'sync'; output = $result.output }
    }
    $runId = "$Name-$([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())"
    $started = Start-BackgroundScript -RunId $runId -Path $entry.path -Arguments $fullArgs
    $started['mode'] = 'background'
    return $started
}

if (-not (Test-Path $DashboardFile)) {
    Write-Error "Dashboard not found: $DashboardFile"
    exit 1
}
$dashboardContent = Get-Content $DashboardFile -Raw -Encoding UTF8

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://127.0.0.1:$Port/")

try {
    $listener.Start()
    Write-Host "AI Dashboard: http://127.0.0.1:$Port" -ForegroundColor Cyan
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        $path = $request.Url.AbsolutePath
        $method = $request.HttpMethod
        $body = ''
        $contentType = 'application/json; charset=utf-8'
        $statusCode = 200
        # Set when a route hands the response to another runspace, which then owns writing it.
        $handedOff = $false

        # Everything for this one request -- including writing the response -- is
        # wrapped so a single bad request (e.g. a client disconnecting or timing out
        # mid-response, which throws from OutputStream.Write) can never escape this
        # while loop and kill the whole dashboard process. Previously only the route
        # dispatch was guarded; the write-out below it was not.
        try {
            switch -Regex ($path) {
                '^/api/status$' {
                    $body = (Get-ServiceStatus | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/start$' {
                    $reader = New-Object IO.StreamReader($request.InputStream)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    $obj = $json | ConvertFrom-Json
                    $body = (Start-ServiceByName ([string]$obj.service) | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/stop$' {
                    $reader = New-Object IO.StreamReader($request.InputStream)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    $obj = $json | ConvertFrom-Json
                    $body = (Stop-ServiceByName ([string]$obj.service) | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/restart$' {
                    $reader = New-Object IO.StreamReader($request.InputStream)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    $obj = $json | ConvertFrom-Json
                    $svcName = [string]$obj.service
                    $stopResult = Stop-ServiceByName $svcName
                    if ($svcName -eq 'telegram-bot' -and -not $stopResult.success) {
                        $body = ($stopResult | ConvertTo-Json -Depth 10)
                        break
                    }
                    Start-Sleep -Milliseconds 500
                    $startResult = Start-ServiceByName $svcName
                    $body = (@{ success = [bool]$startResult.success; message = "Stopped: $($stopResult.message) | $($startResult.message)" } | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/benchmark$' {
                    $body = (Run-QuickBenchmark | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/optimizer$' {
                    # Model Tuner (Tools/model-tuner). Actions: start | stop | plan | apply.
                    $reader = New-Object IO.StreamReader($request.InputStream)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    $obj = if ($json) { $json | ConvertFrom-Json } else { [pscustomobject]@{} }
                    $body = (Invoke-TunerAction -Request $obj | ConvertTo-Json -Depth 6)
                    break
                }
                '^/api/optimizer/status$' {
                    $body = (Get-TunerStatus | ConvertTo-Json -Depth 8)
                    break
                }
                '^/api/gpu$' {
                    $gpu = Get-GpuStats
                    $gpuResult = if ($gpu) { $gpu } else { @{ error = 'nvidia-smi unavailable' } }
                    $body = ($gpuResult | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/scripts/list$' {
                    $catalog = [ordered]@{}
                    foreach ($key in $ScriptCatalog.Keys) {
                        $entry = $ScriptCatalog[$key]
                        $catalog[$key] = [ordered]@{ label = $entry.label; description = $entry.description; argSpec = $entry.argSpec }
                    }
                    $body = ($catalog | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/scripts/run$' {
                    $reader = New-Object IO.StreamReader($request.InputStream)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    $obj = $json | ConvertFrom-Json
                    $scriptArgs = @()
                    if ($obj.args) { $scriptArgs = @($obj.args | ForEach-Object { [string]$_ }) }
                    $runResult = Invoke-CatalogScript -Name ([string]$obj.name) -ScriptArgs $scriptArgs
                    # -Depth 3, not 10 -- sync mode's "output" is a raw multi-line string; see
                    # the note on /api/scripts/status for why a high depth is dangerous here.
                    $body = ($runResult | ConvertTo-Json -Depth 3)
                    break
                }
                '^/api/scripts/status$' {
                    # -Depth 3 is deliberate, not copy-pasted from the other routes' -Depth 10:
                    # ConvertTo-Json in Windows PowerShell 5.1 has EXPONENTIAL-time serialization
                    # once -Depth exceeds ~3-4 on a value containing a string or a string array
                    # (strings implement IEnumerable<char>, and at higher depths the serializer
                    # recurses into that). Measured directly: a 7-line, ~180-char string took
                    # 0.03s at -Depth 3, 2.5s at -Depth 5, and effectively never finished at
                    # -Depth 10 (would be on the order of hours). This response is flat (no
                    # nesting deeper than 2 levels) so depth 3 loses nothing.
                    $statusResult = Get-ScriptRunStatus -RunId $request.QueryString['runId']
                    $body = ($statusResult | ConvertTo-Json -Depth 3)
                    break
                }
                '^/api/screenshot/latest$' {
                    $latest = Get-ChildItem $ScreenshotDir -Filter '*.png' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
                    if (-not $latest) {
                        $errBytes = [Text.Encoding]::UTF8.GetBytes((@{ success = $false; message = 'No screenshot captured yet.' } | ConvertTo-Json))
                        $response.StatusCode = 404
                        $response.ContentType = 'application/json; charset=utf-8'
                        $response.ContentLength64 = $errBytes.Length
                        $response.OutputStream.Write($errBytes, 0, $errBytes.Length)
                        $response.Close()
                        continue
                    }
                    $bytes = [IO.File]::ReadAllBytes($latest.FullName)
                    $response.StatusCode = 200
                    $response.ContentType = 'image/png'
                    $response.ContentLength64 = $bytes.Length
                    $response.OutputStream.Write($bytes, 0, $bytes.Length)
                    $response.Close()
                    continue
                }
                '^/api/logs$' {
                    $allowedLogs = @('llama.stdout.log', 'llama.stderr.log', 'dashboard-test-stdout.log')
                    if ($request.QueryString['file'] -notin $allowedLogs) {
                        $body = (@{success=$false; message='Invalid log file'} | ConvertTo-Json)
                        break
                    }
                    # -Depth 3, not 10: this was PRE-EXISTING at -Depth 10 and had likely been
                    # silently pathological for any log file with real multi-line content the
                    # whole time (Windows PowerShell 5.1's ConvertTo-Json has exponential-time
                    # serialization on string arrays past -Depth ~4 -- see the note on
                    # /api/scripts/status, where this was actually diagnosed). llama.stderr.log
                    # in particular routinely has hundreds of lines, so selecting it in the
                    # Logs tab would have hung the entire single-threaded dashboard.
                    $logLines = Get-LogLines $request.QueryString['file']
                    $body = ($logLines | ConvertTo-Json -Depth 3)
                    break
                }
                '^/api/results$' {
                    $file = Join-Path $Root 'benchmark-results.json'
                    if (Test-Path $file) { $body = Get-Content $file -Raw } else { $body = (@{success=$false;message='No benchmark results found'} | ConvertTo-Json) }
                    break
                }
                '^/api/loras$' {
                    $body = (Get-LoraAdapters | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/models$' {
                    $body = (Get-AvailableModels | ConvertTo-Json -Depth 10)
                    break
                }
                '^/api/models/config$' {
                    if ($method -eq 'POST') {
                        $reader = New-Object IO.StreamReader($request.InputStream)
                        $json = $reader.ReadToEnd(); $reader.Close()
                        $obj = $json | ConvertFrom-Json
                        $body = (Save-ModelConfigOnly -FileName ([string]$obj.filename) -Settings $obj.settings | ConvertTo-Json -Depth 10)
                    } else {
                        $file = $request.QueryString['file']
                        $config = Get-ModelsConfig
                        $settings = Get-ModelSettingsFor -Config $config -FileName $file
                        $body = ($settings | ConvertTo-Json -Depth 10)
                    }
                    break
                }
                '^/api/models/load$' {
                    $reader = New-Object IO.StreamReader($request.InputStream)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    $obj = $json | ConvertFrom-Json
                    $loadName = [string]$obj.filename
                    if ([string]::IsNullOrWhiteSpace($loadName) -or $loadName -match '[\\/:]' -or $loadName -match '\.\.') {
                        $body = (@{ success = $false; message = "Not a model file name: $loadName" } | ConvertTo-Json)
                        break
                    }
                    $persist = $true
                    if ($obj.PSObject.Properties.Name -contains 'persist') { $persist = [bool]$obj.persist }
                    if (-not $persist -and $obj.settings) {
                        # Trial load (Model Tuner): the model's saved settings with these overrides, nothing saved.
                        $body = (Start-ModelTrial -FileName ([string]$obj.filename) -Overrides $obj.settings | ConvertTo-Json -Depth 10)
                    } else {
                        $body = (Apply-ModelSettings -FileName ([string]$obj.filename) | ConvertTo-Json -Depth 10)
                    }
                    break
                }
                '^/api/backends$' {
                    # Local llama-server plus the remote (HP) one, for the backend picker.
                    $remote = Get-RemoteBackendStatusCached -Force:($request.QueryString['refresh'] -eq '1')
                    $localState = 'offline'
                    if (Test-Http "$LlamaHost/v1/models") { $localState = 'ready' }
                    $body = (@{
                        local = @{ id = 'local'; name = 'This PC'; url = $LlamaHost; configured = $true; state = $localState }
                        remote = $remote
                    } | ConvertTo-Json -Depth 4)
                    break
                }
                '^/api/remote/v1/chat/completions$' {
                    if ($method -ne 'POST') {
                        $statusCode = 405
                        $body = (@{ error = @{ message = 'POST only.' } } | ConvertTo-Json -Compress)
                        break
                    }
                    $reader = New-Object IO.StreamReader($request.InputStream, [Text.Encoding]::UTF8)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    if (Start-RemoteChatProxy -Context $context -RequestBody $json) {
                        $handedOff = $true
                    } else {
                        $statusCode = 429
                        $body = (@{ error = @{ message = 'Too many requests to the remote model are already waiting.' } } | ConvertTo-Json -Compress)
                    }
                    break
                }

                default {
                    $contentType = 'text/html; charset=utf-8'
                    $body = $dashboardContent
                    break
                }
            }
        } catch {
            $body = (@{success=$false;error=$_.Exception.Message} | ConvertTo-Json -Depth 10)
        }
        if ($handedOff) { continue }

        # Writing can fail on its own (client disconnected / timed out mid-response,
        # which throws from OutputStream.Write) -- that must not crash the listener
        # loop for every request after it, so it gets its own guard rather than
        # relying on the try/catch above (which has already run by this point).
        try {
            $bytes = [Text.Encoding]::UTF8.GetBytes($body)
            $response.StatusCode = $statusCode
            $response.ContentType = $contentType
            $response.ContentLength64 = $bytes.Length
            $response.OutputStream.Write($bytes,0,$bytes.Length)
            $response.Close()
        } catch {
            try { $response.Abort() } catch {}
        }
    }
} finally {
    if ($listener.IsListening) { $listener.Stop() }
    $listener.Close()
    $script:RemoteProxyPool.Close()
}
