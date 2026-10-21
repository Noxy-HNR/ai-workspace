# Remote model backend (the HP server's llama-server) for the dashboard.
#
# Dot-sourced by dashboard-server.ps1, and inside the runspaces that proxy chat requests, so it
# must stay free of dashboard globals. Works on Windows PowerShell 5.1 and PowerShell 7.
#
# Settings come from Config\remote-backend.json (gitignored; see remote-backend.example.json) and
# environment variables, which win over the file:
#   AI_REMOTE_LLM_URL      e.g. http://192.168.1.241:10000 or a Tailscale name
#   AI_REMOTE_LLM_API_KEY  the HP's llama-server API key
#   AI_REMOTE_LLM_NAME     display name (default "HP server")
# The file may hold the key itself ("apiKey") or name another variable to read it from
# ("apiKeyEnv"). The key is only ever sent to the remote server; nothing here returns or logs it.

function Test-RemoteBackendUrl {
    # Only plain http(s) base URLs with no credentials, query or fragment.
    param([string]$Url)
    if ([string]::IsNullOrWhiteSpace($Url)) { return $false }
    $uri = $null
    if (-not [Uri]::TryCreate($Url.Trim(), [UriKind]::Absolute, [ref]$uri)) { return $false }
    if ($uri.Scheme -ne 'http' -and $uri.Scheme -ne 'https') { return $false }
    if ($uri.UserInfo -or $uri.Query -or $uri.Fragment) { return $false }
    if ([string]::IsNullOrWhiteSpace($uri.Host)) { return $false }
    return $true
}

function Get-RemoteBackendConfig {
    param(
        [string]$ConfigPath,
        # Injectable for tests; defaults to the process environment.
        [hashtable]$Environment
    )
    if (-not $Environment) {
        $Environment = @{}
        foreach ($name in 'AI_REMOTE_LLM_URL', 'AI_REMOTE_LLM_API_KEY', 'AI_REMOTE_LLM_NAME') {
            $value = [Environment]::GetEnvironmentVariable($name)
            if ($value) { $Environment[$name] = $value }
        }
    }

    $file = $null
    $fileError = ''
    if ($ConfigPath -and (Test-Path -LiteralPath $ConfigPath)) {
        try { $file = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json }
        catch { $fileError = "Could not read $(Split-Path -Leaf $ConfigPath): $($_.Exception.Message)" }
    }

    $name = 'HP server'
    $url = ''
    $apiKey = ''
    $source = 'none'
    if ($file) {
        if ($file.name) { $name = [string]$file.name }
        if ($file.url) { $url = [string]$file.url; $source = 'file' }
        if ($file.apiKey) { $apiKey = [string]$file.apiKey }
        elseif ($file.apiKeyEnv) {
            $keyVar = [string]$file.apiKeyEnv
            $value = if ($Environment.ContainsKey($keyVar)) { $Environment[$keyVar] } else { [Environment]::GetEnvironmentVariable($keyVar) }
            if ($value) { $apiKey = [string]$value }
        }
    }
    if ($Environment['AI_REMOTE_LLM_NAME']) { $name = [string]$Environment['AI_REMOTE_LLM_NAME'] }
    if ($Environment['AI_REMOTE_LLM_URL']) { $url = [string]$Environment['AI_REMOTE_LLM_URL']; $source = 'env' }
    if ($Environment['AI_REMOTE_LLM_API_KEY']) { $apiKey = [string]$Environment['AI_REMOTE_LLM_API_KEY'] }

    $url = $url.Trim().TrimEnd('/')
    # People paste the OpenAI base URL; the dashboard adds /v1 itself.
    if ($url -match '/v1$') { $url = $url.Substring(0, $url.Length - 3) }
    $valid = Test-RemoteBackendUrl $url
    $problem = $fileError
    if (-not $problem -and $url -and -not $valid) { $problem = 'The remote URL must be http:// or https:// with no user name, query or fragment.' }

    return @{
        name = $name
        url = if ($valid) { $url } else { '' }
        apiKey = $apiKey.Trim()
        configured = $valid
        source = $source
        problem = $problem
    }
}

function Get-RemoteBackendPublicInfo {
    # What the browser may see: never the key itself.
    param($Config)
    return @{
        id = 'hp'
        name = $Config.name
        url = $Config.url
        configured = [bool]$Config.configured
        hasKey = -not [string]::IsNullOrEmpty($Config.apiKey)
        source = $Config.source
        problem = $Config.problem
    }
}

function Test-RemotePortOpen {
    param([string]$HostName, [int]$Port, [int]$TimeoutMs = 800)
    $tcp = $null
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $iar = $tcp.BeginConnect($HostName, $Port, $null, $null)
        $connected = $iar.AsyncWaitHandle.WaitOne($TimeoutMs) -and $tcp.Connected
        if ($connected) { $tcp.EndConnect($iar) }
        return $connected
    } catch { return $false }
    finally { if ($tcp) { $tcp.Close() } }
}

function Invoke-RemoteHttp {
    # One HTTP call that never throws: returns @{ status; body; error }. status 0 means no response
    # (refused, timed out, DNS failure). Non-2xx responses keep their status and body.
    param(
        [string]$Method = 'GET',
        [string]$Url,
        [string]$ApiKey,
        [string]$Body,
        [int]$TimeoutMs = 3000
    )
    $result = @{ status = 0; body = ''; error = '' }
    try {
        $req = [System.Net.HttpWebRequest]::Create($Url)
        $req.Method = $Method
        $req.Timeout = $TimeoutMs
        $req.ReadWriteTimeout = $TimeoutMs
        $req.Accept = 'application/json'
        # Only ever talk to the configured server: a redirect could carry the key elsewhere.
        $req.AllowAutoRedirect = $false
        if ($ApiKey) { $req.Headers.Add('Authorization', "Bearer $ApiKey") }
        if ($Method -ne 'GET' -and $null -ne $Body) {
            $bytes = [Text.Encoding]::UTF8.GetBytes($Body)
            $req.ContentType = 'application/json'
            $req.ContentLength = $bytes.Length
            $stream = $req.GetRequestStream()
            try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Close() }
        }
        $resp = $req.GetResponse()
    } catch [System.Net.WebException] {
        $resp = $_.Exception.Response
        if (-not $resp) {
            $result.error = $_.Exception.Message
            return $result
        }
    } catch {
        $msg = $_.Exception.Message
        if ($_.Exception.InnerException) { $msg = $_.Exception.InnerException.Message }
        $result.error = $msg
        return $result
    }
    try {
        $result.status = [int]$resp.StatusCode
        $reader = New-Object IO.StreamReader($resp.GetResponseStream(), [Text.Encoding]::UTF8)
        try { $result.body = $reader.ReadToEnd() } finally { $reader.Close() }
    } catch {
        $result.error = $_.Exception.Message
    } finally {
        $resp.Close()
    }
    return $result
}

function Get-RemoteModelName {
    # llama-server reports the model as its alias or file path; show the file's base name.
    param([string]$Id)
    if (-not $Id) { return '' }
    $leaf = ($Id -split '[\\/]')[-1]
    return ($leaf -replace '\.gguf$', '')
}

function Get-RemoteBackendStatus {
    # state: not-configured | offline | loading | unauthorized | error | ready
    param($Config, [int]$TimeoutMs = 3000)
    $status = Get-RemoteBackendPublicInfo $Config
    $status.state = 'not-configured'
    $status.model = ''
    $status.contextSize = $null
    $status.latencyMs = $null
    $status.message = ''
    $status.checkedAt = [DateTime]::UtcNow.ToString('o')
    if (-not $Config.configured) {
        $status.message = if ($Config.problem) { $Config.problem } else { 'No remote backend is set. Set AI_REMOTE_LLM_URL or create Config\remote-backend.json.' }
        return $status
    }

    $uri = [Uri]$Config.url
    if (-not (Test-RemotePortOpen -HostName $uri.Host -Port $uri.Port)) {
        $status.state = 'offline'
        $status.message = "$($Config.name) is not reachable at $($uri.Host):$($uri.Port)."
        return $status
    }

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $health = Invoke-RemoteHttp -Url "$($Config.url)/health" -TimeoutMs $TimeoutMs
    $sw.Stop()
    $status.latencyMs = [int]$sw.ElapsedMilliseconds
    if ($health.status -eq 0) {
        $status.state = 'offline'
        $status.message = "No answer from $($Config.name): $($health.error)"
        return $status
    }
    if ($health.status -eq 503) {
        $status.state = 'loading'
        $status.message = 'The model is still loading.'
        return $status
    }
    if ($health.status -ne 200) {
        $status.state = 'error'
        $status.message = "/health answered HTTP $($health.status)."
        return $status
    }

    $models = Invoke-RemoteHttp -Url "$($Config.url)/v1/models" -ApiKey $Config.apiKey -TimeoutMs $TimeoutMs
    if ($models.status -eq 401 -or $models.status -eq 403) {
        $status.state = 'unauthorized'
        $status.message = if ($status.hasKey) { 'The server rejected the API key.' } else { 'The server needs an API key. Set AI_REMOTE_LLM_API_KEY.' }
        return $status
    }
    if ($models.status -ne 200) {
        $status.state = 'error'
        $detail = if ($models.status) { "HTTP $($models.status)" } else { $models.error }
        $status.message = "Could not list models: $detail"
        return $status
    }
    try {
        $data = $models.body | ConvertFrom-Json
        $first = @($data.data)[0]
        if ($first) {
            $status.model = Get-RemoteModelName ([string]$first.id)
            if ($first.meta -and $first.meta.n_ctx_train) { $status.contextSize = [int]$first.meta.n_ctx_train }
        }
    } catch {}
    $status.state = 'ready'
    return $status
}

function Invoke-RemoteChatProxy {
    # Forwards one OpenAI chat request to the remote server and returns @{ status; body }, with the
    # body always JSON. Only chat completions are forwarded, so the key cannot be used for anything else.
    param($Config, [string]$RequestBody, [int]$TimeoutMs = 600000)
    $fail = {
        param([int]$code, [string]$state, [string]$message)
        @{ status = $code; body = (@{ error = @{ message = $message; state = $state } } | ConvertTo-Json -Depth 4 -Compress) }
    }
    if (-not $Config.configured) {
        return (& $fail 503 'not-configured' 'No remote backend is set. Set AI_REMOTE_LLM_URL or create Config\remote-backend.json.')
    }
    try { $parsed = $RequestBody | ConvertFrom-Json -ErrorAction Stop }
    catch { return (& $fail 400 'bad-request' 'The request body is not JSON.') }
    # The reply is relayed in one piece, so streaming would only arrive at the end as raw SSE.
    if ($parsed.stream -eq $true) { return (& $fail 400 'bad-request' 'Streaming is not supported through the dashboard; send stream: false.') }

    $uri = [Uri]$Config.url
    if (-not (Test-RemotePortOpen -HostName $uri.Host -Port $uri.Port)) {
        return (& $fail 503 'offline' "$($Config.name) is offline (nothing answering at $($uri.Host):$($uri.Port)).")
    }
    $r = Invoke-RemoteHttp -Method 'POST' -Url "$($Config.url)/v1/chat/completions" -ApiKey $Config.apiKey -Body $RequestBody -TimeoutMs $TimeoutMs
    if ($r.status -eq 0) {
        return (& $fail 504 'offline' "No answer from $($Config.name): $($r.error)")
    }
    if ($r.status -eq 401 -or $r.status -eq 403) {
        return (& $fail 502 'unauthorized' "$($Config.name) rejected the API key.")
    }
    if ($r.status -eq 503) {
        return (& $fail 503 'loading' "$($Config.name) is still loading its model.")
    }
    $body = $r.body
    if ([string]::IsNullOrWhiteSpace($body)) { $body = (@{ error = @{ message = "Empty reply (HTTP $($r.status))." } } | ConvertTo-Json -Compress) }
    return @{ status = $r.status; body = $body }
}
