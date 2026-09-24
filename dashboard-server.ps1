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

function Get-ServiceStatus {
    $llamaUp = Test-Http "$LlamaHost/v1/models"
    $cptrUp = Test-Http "$CptrHost/openapi.json"
    $ouraPort = Get-OuraPort
    $ouraUrl = "http://127.0.0.1:$ouraPort"
    $ouraHealth = $null
    if (Test-PortOpen -Port $ouraPort) {
        try { $ouraHealth = Invoke-RestMethod "$ouraUrl/health" -TimeoutSec 2 -ErrorAction Stop } catch {}
    }
    $ouraUp = $null -ne $ouraHealth

    $llamaPid = Get-PortPid 10000
    $cptrPid = Get-PortPid 8000
    $ouraPid = Get-PortPid $ouraPort

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

function Stop-LlamaNative {
    $stopped = $false
    $existingPid = Get-PortPid 10000
    if ($existingPid) {
        try { Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue; $stopped = $true } catch {}
    }
    Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        $stopped = $true
    }
    if ($stopped) { Start-Sleep -Seconds 1 }
    Remove-Item $LlamaPidFile -Force -ErrorAction SilentlyContinue
    return @{ success = $true; message = 'llama-server stopped.' }
}

function Start-LlamaNative {
    param([string]$FileName, $Settings)

    if (-not (Test-Path $LlamaBin)) {
        return @{ success = $false; message = "Native llama-server.exe not found at $LlamaBin" }
    }
    $modelPath = Join-Path $ModelsDir $FileName
    if (-not (Test-Path $modelPath)) {
        return @{ success = $false; message = "Model file not found: $FileName" }
    }

    Stop-LlamaNative | Out-Null

    $argList = Build-LlamaArgs -ModelPath $modelPath -Settings $Settings
    $stdOut = Join-Path $LogDir 'llama.stdout.log'
    $stdErr = Join-Path $LogDir 'llama.stderr.log'

    try {
        $p = Start-Process -FilePath $LlamaBin -ArgumentList $argList -WorkingDirectory (Split-Path $LlamaBin) -RedirectStandardOutput $stdOut -RedirectStandardError $stdErr -WindowStyle Hidden -PassThru
        try { $p.PriorityClass = 'High' } catch {}
        Set-Content -Path $LlamaPidFile -Value $p.Id -Encoding UTF8
        return @{ success = $true; message = "llama-server starting natively (PID $($p.Id)). Loading $FileName..." }
    } catch {
        return @{ success = $false; message = "Failed to start llama-server: $($_.Exception.Message)" }
    }
}

function Start-ServiceByName {
    param([string]$Name)
    switch ($Name) {
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
        'llama-server' { return Stop-LlamaNative }
        'cptr' {
            $cptrPid = Get-PortPid 8000
            if ($cptrPid) { Stop-Process -Id $cptrPid -Force -ErrorAction SilentlyContinue }
            return @{ success=$true; message='cptr stopped.' }
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
        return $merged
    }
    return $defaults
}

function Get-AvailableModels {
    $config = Get-ModelsConfig
    $files = Get-ChildItem $ModelsDir -Filter '*.gguf' -File -ErrorAction SilentlyContinue
    $result = @()
    foreach ($f in ($files | Sort-Object Name)) {
        $settings = Get-ModelSettingsFor -Config $config -FileName $f.Name
        $hasConfig = $config.models -and ($config.models.PSObject.Properties.Name -contains $f.Name)
        $result += [ordered]@{
            filename     = $f.Name
            sizeGB       = [math]::Round($f.Length / 1GB, 2)
            lastModified = $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
            isActive     = ($f.Name -eq $config.activeModel)
            hasCustomConfig = [bool]$hasConfig
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
                    $script = Join-Path $Root 'model-optimize.ps1'
                    if (-not (Test-Path $script)) {
                        $body = (@{success=$false;message='model-optimize.ps1 not found'} | ConvertTo-Json)
                        break
                    }
                    $reader = New-Object IO.StreamReader($request.InputStream)
                    $json = $reader.ReadToEnd(); $reader.Close()
                    $mode = 'quick'
                    if ($json) { try { $obj = $json | ConvertFrom-Json; if ($obj.mode) { $mode = [string]$obj.mode } } catch {} }
                    if ($mode -notin @('quick','medium','full')) { $mode = 'quick' }
                    try {
                        $out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script -Mode $mode -Json 2>&1 | Out-String
                        $code = $LASTEXITCODE
                        try {
                            $result = $out.Trim() | ConvertFrom-Json
                            $result.success = ($code -eq 0 -and [bool]$result.success)
                            $body = ($result | ConvertTo-Json -Depth 5)
                        } catch {
                            # -Depth 3: $out is raw script output (a string) -- see the
                            # ConvertTo-Json depth note on /api/scripts/status.
                            $body = (@{success=$false;message='Optimizer returned invalid output';output=$out} | ConvertTo-Json -Depth 3)
                        }
                    } catch {
                        $body = (@{success=$false;message=$_.Exception.Message} | ConvertTo-Json -Depth 10)
                    }
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
                    $body = (Apply-ModelSettings -FileName ([string]$obj.filename) | ConvertTo-Json -Depth 10)
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

        # Writing can fail on its own (client disconnected / timed out mid-response,
        # which throws from OutputStream.Write) -- that must not crash the listener
        # loop for every request after it, so it gets its own guard rather than
        # relying on the try/catch above (which has already run by this point).
        try {
            $bytes = [Text.Encoding]::UTF8.GetBytes($body)
            $response.StatusCode = 200
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
}