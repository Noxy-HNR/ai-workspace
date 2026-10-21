# Where model files live: a few pinned models on this PC, everything else on the home model server.
#
# Config\model-storage.json (copy Config\model-storage.example.json) sets the folders:
#   networkDir   the server's model folder as a UNC path, e.g. \\192.168.1.241\ModelFiles\Models.
#                Use the UNC path, not a mapped drive letter: drive letters belong to one logon session.
#   pinnedLocal  file names that should load from this PC even when the server also has them
# Without a networkDir, or while the server is unreachable, every model loads from C:\AI\models.
#
# A pinned model loads from this PC; every other model loads straight from networkDir. If the preferred
# copy is missing, the other location is used. Nothing is copied to this PC: llama-server reads the file
# over SMB once (--load-mode none), so the server's array is read directly every time.

$script:ModelStorageConfigFile = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Config\model-storage.json'
# A dead share can hang a directory listing for 30+ seconds, and one dashboard request resolves several
# paths, so the reachability probe runs at most once every 15 seconds.
$script:ModelShareProbe = @{ at = [datetime]::MinValue; status = $null }
$script:ModelShareProbeSeconds = 15

function Get-ModelStorageConfig {
    $cfg = [ordered]@{ localDir = 'C:\AI\models'; networkDir = ''; networkHost = ''; pinnedLocal = @(); error = '' }
    if (Test-Path -LiteralPath $script:ModelStorageConfigFile) {
        try {
            $raw = Get-Content -LiteralPath $script:ModelStorageConfigFile -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($k in @('networkDir', 'networkHost')) { if ($raw.$k) { $cfg[$k] = [string]$raw.$k } }
            if ($raw.pinnedLocal) { $cfg.pinnedLocal = @($raw.pinnedLocal | ForEach-Object { [string]$_ }) }
        } catch { $cfg.error = "Config\model-storage.json could not be read ($($_.Exception.Message)); using this PC's models only." }
    }
    # The local folder is the one the dashboard and start.ps1 define as $ModelsDir, so they always agree.
    $callerModelsDir = Get-Variable -Name ModelsDir -ValueOnly -ErrorAction SilentlyContinue
    if ($callerModelsDir) { $cfg.localDir = [string]$callerModelsDir }
    $cfg.networkDir = $cfg.networkDir.TrimEnd('\')
    if ($cfg.networkDir -and $cfg.networkDir -notmatch '^\\\\[^\\]+\\[^\\]+') {
        $cfg.error = "networkDir must be a UNC path such as \\192.168.1.241\ModelFiles\Models, not '$($cfg.networkDir)'; using this PC's models only."
        $cfg.networkDir = ''
    }
    if (-not $cfg.networkHost -and $cfg.networkDir -match '^\\\\([^\\]+)\\') { $cfg.networkHost = $Matches[1] }
    return $cfg
}

function Get-ModelShareStatus {
    # { configured, up, networkDir, message }. message says why the share is unavailable, in words the
    # dashboard can show as is.
    param([switch]$Refresh)
    $cfg = Get-ModelStorageConfig
    if (-not $cfg.networkDir) {
        return @{ configured = $false; up = $false; networkDir = ''; message = $(if ($cfg.error) { $cfg.error } else { 'No model server is configured (Config\model-storage.json).' }) }
    }
    $last = $script:ModelShareProbe
    if (-not $Refresh -and $last.status -and $last.status.networkDir -eq $cfg.networkDir -and
        ((Get-Date) - $last.at).TotalSeconds -lt $script:ModelShareProbeSeconds) { return $last.status }

    $status = @{ configured = $true; up = $false; networkDir = $cfg.networkDir; message = '' }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        # Probe SMB (TCP 445) with a 1 s limit first, so an unplugged server fails fast instead of hanging.
        $task = $client.ConnectAsync($cfg.networkHost, 445)
        if (-not $task.Wait(1000) -or -not $client.Connected) {
            $status.message = "The model server $($cfg.networkHost) did not answer on the SMB port (TCP 445) within 1 second. Check that it is powered on and that this PC is on the home network."
        } else {
            # Samba can accept the TCP connection and still hang on file requests, so the folder check
            # gets its own 3-second limit too.
            $opened = Test-ModelShareFolder -Path $cfg.networkDir -TimeoutMs 3000
            if ($null -eq $opened) {
                $status.message = "The model server $($cfg.networkHost) accepted the SMB connection but did not open $($cfg.networkDir) within 3 seconds. Samba may be hung; on the server, check: systemctl status smbd."
            } elseif (-not $opened) {
                $status.message = "The model server $($cfg.networkHost) is up, but $($cfg.networkDir) can't be opened. Check the share and folder name, and that this Windows account has saved credentials for it (open the path in Explorer once, or use cmdkey)."
            } else {
                $status.up = $true
            }
        }
    } catch {
        $inner = $_.Exception
        while ($inner.InnerException) { $inner = $inner.InnerException }
        $code = if ($inner -is [System.Net.Sockets.SocketException]) { [string]$inner.SocketErrorCode } else { '' }
        if ($code -in @('HostNotFound', 'NoData', 'TryAgain')) {
            $status.message = "The model server name $($cfg.networkHost) could not be resolved. Use its IP address in networkDir, or fix DNS."
        } elseif ($code -eq 'ConnectionRefused') {
            $status.message = "The model server $($cfg.networkHost) refused the SMB connection (TCP 445). Samba may be stopped; on the server, check: systemctl status smbd."
        } else {
            $status.message = "The model server $($cfg.networkHost) could not be reached on the SMB port (TCP 445): $($inner.Message.TrimEnd('.'))."
        }
    } finally { $client.Dispose() }
    $script:ModelShareProbe = @{ at = Get-Date; status = $status }
    return $status
}

function Invoke-ModelShareIO {
    # Runs one file operation on the share with a hard deadline, so a server that stops answering can't
    # hold the dashboard's request thread for Windows' SMB timeout. Returns @{ done; value }. On a timeout
    # the share is marked offline until the next probe, so later calls fall back to this PC at once.
    param([scriptblock]$Script, [object[]]$Arguments = @(), [int]$TimeoutMs = 3000)
    $ps = [powershell]::Create().AddScript($Script.ToString())
    foreach ($a in $Arguments) { [void]$ps.AddArgument($a) }
    $handle = $ps.BeginInvoke()
    if (-not $handle.AsyncWaitHandle.WaitOne($TimeoutMs)) {
        # Leave the stuck operation to finish on its own thread; waiting for it is what we are avoiding.
        [void]$ps.BeginStop($null, $null)
        $cfg = Get-ModelStorageConfig
        $script:ModelShareProbe = @{ at = Get-Date; status = @{ configured = $true; up = $false; networkDir = $cfg.networkDir
            message = "The model server $($cfg.networkHost) stopped answering file requests (nothing within $([int]($TimeoutMs / 1000)) seconds). Samba may be hung; on the server, check: systemctl status smbd." } }
        return @{ done = $false; value = $null }
    }
    try { return @{ done = $true; value = @($ps.EndInvoke($handle)) } }
    catch { return @{ done = $true; value = @() } }
    finally { $ps.Dispose() }
}

function Test-ModelShareFolder {
    # $true / $false for whether the folder opens, or $null if the check did not finish in time.
    param([string]$Path, [int]$TimeoutMs)
    $r = Invoke-ModelShareIO -Script { param($p) Test-Path -LiteralPath $p -PathType Container } -Arguments @($Path) -TimeoutMs $TimeoutMs
    if (-not $r.done) { return $null }
    return [bool]@($r.value)[0]
}

function Test-ModelNetworkPath {
    param([string]$Path)
    $net = (Get-ModelStorageConfig).networkDir
    return ($Path.StartsWith('\\') -or ($net -and $Path.StartsWith($net, [StringComparison]::OrdinalIgnoreCase)))
}

function Test-ModelFileUsable {
    # Whether a file exists and can be loaded now: on this PC it only has to exist; on the share it must
    # answer within the deadline and have settled (not changed in the last minute).
    param([string]$Path)
    if (-not (Test-ModelNetworkPath $Path)) { return (Test-Path -LiteralPath $Path -PathType Leaf) }
    $r = Invoke-ModelShareIO -Script { param($p) if (Test-Path -LiteralPath $p -PathType Leaf) { (Get-Item -LiteralPath $p).LastWriteTime } } -Arguments @($Path)
    if (-not $r.done -or @($r.value).Count -eq 0) { return $false }
    return ((Get-Date) - [datetime]@($r.value)[0]).TotalSeconds -ge 60
}

function Get-ModelDirGguf {
    # Top-level .gguf files of a folder; a folder on the share is listed with a deadline.
    param([string]$Dir)
    if (-not (Test-ModelNetworkPath $Dir)) { return @(Get-ChildItem -LiteralPath $Dir -Filter '*.gguf' -File -ErrorAction SilentlyContinue) }
    $r = Invoke-ModelShareIO -Script { param($d) Get-ChildItem -LiteralPath $d -Filter '*.gguf' -File -ErrorAction SilentlyContinue } -Arguments @($Dir) -TimeoutMs 5000
    if (-not $r.done) { return @() }
    return @($r.value | Where-Object { $_ -is [System.IO.FileInfo] } | ForEach-Object { [System.IO.FileInfo]$_ })
}

function Test-ModelShareReachable {
    return [bool](Get-ModelShareStatus).up
}

function Test-ModelPinnedLocal {
    param([string]$FileName)
    return @((Get-ModelStorageConfig).pinnedLocal) -contains $FileName
}

function Get-ModelSearchDirs {
    # Directories to try, in order, for a file of the given kind. Pinned models (and companions such as
    # LoRAs and projectors) prefer the local copy; other models prefer the network copy.
    param([string]$FileName, [ValidateSet('model', 'companion', 'lora')][string]$Kind = 'model')
    $cfg = Get-ModelStorageConfig
    $local = @($cfg.localDir)
    $net = @()
    if (Test-ModelShareReachable) { $net = @($cfg.networkDir) }
    if ($Kind -eq 'lora') {
        $local = @($cfg.localDir, (Join-Path $cfg.localDir 'lora'))
        $net = @($net | ForEach-Object { $_; Join-Path $_ 'lora' })
    }
    if ($Kind -eq 'model' -and -not (Test-ModelPinnedLocal $FileName)) { return @($net) + @($local) }
    return @($local) + @($net)
}

function Resolve-ModelFilePath {
    # Full path of a bare model/companion file name, or $null. Never accepts a path.
    param([string]$FileName, [ValidateSet('model', 'companion', 'lora')][string]$Kind = 'model')
    if ([string]::IsNullOrWhiteSpace($FileName) -or $FileName -match '[\\/:]' -or $FileName -match '\.\.') { return $null }
    $cfg = Get-ModelStorageConfig
    # A pinned model or companion that is on this PC resolves without touching the share at all.
    if ($Kind -ne 'model' -or (Test-ModelPinnedLocal $FileName)) {
        $localDirs = @($cfg.localDir)
        if ($Kind -eq 'lora') { $localDirs += Join-Path $cfg.localDir 'lora' }
        foreach ($dir in $localDirs) {
            $p = Join-Path $dir $FileName
            if (Test-Path -LiteralPath $p -PathType Leaf) { return $p }
        }
    }
    foreach ($dir in (Get-ModelSearchDirs -FileName $FileName -Kind $Kind)) {
        $p = Join-Path $dir $FileName
        # Same rule as the listing: a server file still being copied is skipped, so a partial GGUF never loads.
        if (Test-ModelFileUsable $p) { return $p }
    }
    return $null
}

function Test-ModelFileSettled {
    # A file on the share that changed in the last minute may still be arriving.
    param([System.IO.FileInfo]$File)
    return ((Get-Date) - $File.LastWriteTime).TotalSeconds -ge 60
}

function Get-ModelNotFoundMessage {
    # Why a model file name did not resolve, naming both places it was looked for.
    param([string]$FileName)
    $cfg = Get-ModelStorageConfig
    $share = Get-ModelShareStatus
    if (-not $share.configured) { return "Model file not found in $($cfg.localDir): $FileName" }
    if (-not $share.up) {
        return "$FileName is not on this PC ($($cfg.localDir)), and the model server can't be reached. $($share.message) Models on this PC still load."
    }
    return "Model file not found on this PC ($($cfg.localDir)) or the model server ($($cfg.networkDir)): $FileName"
}

function Get-ModelStorageFiles {
    # Every top-level .gguf on this PC and the share, one entry per file name (the copy that would load).
    $cfg = Get-ModelStorageConfig
    $byName = [ordered]@{}
    $sources = @(@{ dir = $cfg.localDir; location = 'local' })
    if (Test-ModelShareReachable) { $sources += @{ dir = $cfg.networkDir; location = 'network' } }
    foreach ($src in $sources) {
        foreach ($f in (Get-ModelDirGguf -Dir $src.dir)) {
            # A file still being copied to the share keeps changing; list it once it has settled.
            if ($src.location -eq 'network' -and -not (Test-ModelFileSettled $f)) { continue }
            if (-not $byName.Contains($f.Name)) { $byName[$f.Name] = @{ local = $null; network = $null } }
            $byName[$f.Name][$src.location] = $f
        }
    }
    $result = @()
    foreach ($name in $byName.Keys) {
        $entry = $byName[$name]
        $pinned = Test-ModelPinnedLocal $name
        $file = if ($pinned) { if ($entry.local) { $entry.local } else { $entry.network } }
                else { if ($entry.network) { $entry.network } else { $entry.local } }
        $result += [pscustomobject]@{
            Name = $name; File = $file; Pinned = $pinned
            Location = if ($file -eq $entry.local) { 'local' } else { 'network' }
            HasLocal = [bool]$entry.local; HasNetwork = [bool]$entry.network
        }
    }
    return $result
}

function Get-ModelStorageDirs {
    # Directories that can hold companions of the given kind (for listings such as the LoRA picker).
    param([ValidateSet('lora')][string]$Kind)
    return @(Get-ModelSearchDirs -FileName '' -Kind $Kind)
}
