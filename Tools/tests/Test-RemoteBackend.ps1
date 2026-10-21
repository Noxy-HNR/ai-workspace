<#
  Tests for the dashboard's remote model backend (Tools\scripts\RemoteBackend.ps1).
  A fake llama-server runs on a random 127.0.0.1 port, so no real model, network or key is used.
  Run: powershell -NoProfile -ExecutionPolicy Bypass -File Tools\tests\Test-RemoteBackend.ps1
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'Tools\scripts\RemoteBackend.ps1')

$assertions = 0
$failures = New-Object System.Collections.ArrayList
function Assert-Equal {
    param($Actual, $Expected, [string]$Message)
    $script:assertions++
    if ("$Actual" -ne "$Expected") { [void]$script:failures.Add("$Message (expected '$Expected', got '$Actual')") }
}
function Assert-True {
    param([bool]$Condition, [string]$Message)
    $script:assertions++
    if (-not $Condition) { [void]$script:failures.Add($Message) }
}

function Get-FreePort {
    $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
    $l.Start(); $port = $l.LocalEndpoint.Port; $l.Stop()
    return $port
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ('remote-backend-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
$secret = 'test-key-' + [guid]::NewGuid().ToString('N')

# ---- Config ----
$none = Get-RemoteBackendConfig -ConfigPath (Join-Path $tmp 'missing.json') -Environment @{}
Assert-Equal $none.configured $false 'no file and no env means not configured'
Assert-Equal $none.name 'HP server' 'default display name'

$cfgFile = Join-Path $tmp 'remote-backend.json'
Set-Content -LiteralPath $cfgFile -Encoding UTF8 -Value '{ "name": "HP", "url": "http://192.168.1.241:10000/v1/", "apiKeyEnv": "MY_HP_KEY" }'
$fromFile = Get-RemoteBackendConfig -ConfigPath $cfgFile -Environment @{ MY_HP_KEY = $secret }
Assert-Equal $fromFile.configured $true 'file URL is used'
Assert-Equal $fromFile.url 'http://192.168.1.241:10000' 'a pasted /v1/ suffix and trailing slash are dropped'
Assert-Equal $fromFile.apiKey $secret 'apiKeyEnv names the variable holding the key'
Assert-Equal $fromFile.name 'HP' 'name comes from the file'
Assert-Equal $fromFile.source 'file' 'source is the file'

$fromEnv = Get-RemoteBackendConfig -ConfigPath $cfgFile -Environment @{ AI_REMOTE_LLM_URL = 'https://hp.tailnet.ts.net'; AI_REMOTE_LLM_API_KEY = 'env-key'; MY_HP_KEY = $secret }
Assert-Equal $fromEnv.url 'https://hp.tailnet.ts.net' 'AI_REMOTE_LLM_URL wins over the file'
Assert-Equal $fromEnv.apiKey 'env-key' 'AI_REMOTE_LLM_API_KEY wins over the file'
Assert-Equal $fromEnv.source 'env' 'source is the environment'

foreach ($bad in 'ftp://hp:10000', 'http://user:pw@hp:10000', 'http://hp:10000/?x=1', 'not a url', 'file:///C:/x') {
    $c = Get-RemoteBackendConfig -ConfigPath $null -Environment @{ AI_REMOTE_LLM_URL = $bad }
    Assert-Equal $c.configured $false "rejects URL '$bad'"
    Assert-True ([bool]$c.problem) "explains why '$bad' was rejected"
}

$broken = Join-Path $tmp 'broken.json'
Set-Content -LiteralPath $broken -Value '{ not json' -Encoding UTF8
$c = Get-RemoteBackendConfig -ConfigPath $broken -Environment @{}
Assert-Equal $c.configured $false 'unreadable file means not configured'
Assert-True ($c.problem -match 'Could not read') 'unreadable file is reported'

$public = Get-RemoteBackendPublicInfo $fromFile
Assert-Equal $public.hasKey $true 'public info says a key is set'
Assert-True (-not $public.ContainsKey('apiKey')) 'public info has no apiKey field'
Assert-True (-not (($public | ConvertTo-Json -Compress).Contains($secret))) 'public info never contains the key'

Assert-Equal (Get-RemoteModelName '/srv/models/Cyber-Tiel-Coder-35B-A3B-Q4_K_M.gguf') 'Cyber-Tiel-Coder-35B-A3B-Q4_K_M' 'model name from a Linux path'
Assert-Equal (Get-RemoteModelName 'C:\AI\models\x.gguf') 'x' 'model name from a Windows path'
Assert-Equal (Get-RemoteModelName 'gpt-oss-20b') 'gpt-oss-20b' 'an alias is kept'

# ---- Fake llama-server ----
$port = Get-FreePort
$shared = [hashtable]::Synchronized(@{ health = 200; key = $secret; auth = ''; body = ''; redirect = $false })
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://127.0.0.1:$port/")
$listener.Start()
$serverPs = [powershell]::Create()
[void]$serverPs.AddScript({
    param($Listener, $Shared)
    while ($Listener.IsListening) {
        try { $ctx = $Listener.GetContext() } catch { break }
        $req = $ctx.Request; $res = $ctx.Response
        $auth = [string]$req.Headers['Authorization']
        $authorized = $auth -eq "Bearer $($Shared.key)"
        $code = 200; $out = ''
        switch ($req.Url.AbsolutePath) {
            '/health' {
                $code = $Shared.health
                $out = if ($code -eq 200) { '{"status":"ok"}' } else { '{"error":{"code":503,"message":"Loading model"}}' }
            }
            '/v1/models' {
                if (-not $authorized) { $code = 401; $out = '{"error":{"message":"Invalid API Key"}}' }
                else { $out = '{"object":"list","data":[{"id":"/srv/models/Cyber-Tiel-Coder-35B-A3B.gguf","object":"model","meta":{"n_ctx_train":262144}}]}' }
            }
            '/v1/chat/completions' {
                $Shared.auth = $auth
                $reader = New-Object IO.StreamReader($req.InputStream); $Shared.body = $reader.ReadToEnd(); $reader.Close()
                if ($Shared.redirect) { $code = 302; $res.RedirectLocation = 'http://example.invalid/' }
                elseif (-not $authorized) { $code = 401; $out = '{"error":{"message":"Invalid API Key"}}' }
                else { $out = '{"choices":[{"message":{"role":"assistant","content":"hello from hp"}}],"usage":{"completion_tokens":3}}' }
            }
            default { $code = 404; $out = '{"error":{"message":"not found"}}' }
        }
        $bytes = [Text.Encoding]::UTF8.GetBytes($out)
        $res.StatusCode = $code
        $res.ContentType = 'application/json'
        $res.ContentLength64 = $bytes.Length
        if ($bytes.Length) { $res.OutputStream.Write($bytes, 0, $bytes.Length) }
        $res.Close()
    }
}).AddArgument($listener).AddArgument($shared)
$serverHandle = $serverPs.BeginInvoke()

try {
    $good = Get-RemoteBackendConfig -ConfigPath $null -Environment @{ AI_REMOTE_LLM_URL = "http://127.0.0.1:$port"; AI_REMOTE_LLM_API_KEY = $secret }
    $wrongKey = Get-RemoteBackendConfig -ConfigPath $null -Environment @{ AI_REMOTE_LLM_URL = "http://127.0.0.1:$port"; AI_REMOTE_LLM_API_KEY = 'wrong' }
    $noKey = Get-RemoteBackendConfig -ConfigPath $null -Environment @{ AI_REMOTE_LLM_URL = "http://127.0.0.1:$port" }
    $offline = Get-RemoteBackendConfig -ConfigPath $null -Environment @{ AI_REMOTE_LLM_URL = "http://127.0.0.1:$(Get-FreePort)"; AI_REMOTE_LLM_API_KEY = $secret }

    # ---- Status ----
    $s = Get-RemoteBackendStatus -Config $good
    Assert-Equal $s.state 'ready' 'healthy server with the right key is ready'
    Assert-Equal $s.model 'Cyber-Tiel-Coder-35B-A3B' 'status shows the model name'
    Assert-Equal $s.contextSize 262144 'status shows the trained context size'
    Assert-True ($null -ne $s.latencyMs) 'status has a latency'
    Assert-True (-not (($s | ConvertTo-Json -Compress).Contains($secret))) 'status never contains the key'

    Assert-Equal (Get-RemoteBackendStatus -Config $wrongKey).state 'unauthorized' 'wrong key is reported'
    $nk = Get-RemoteBackendStatus -Config $noKey
    Assert-Equal $nk.state 'unauthorized' 'missing key is reported'
    Assert-True ($nk.message -match 'AI_REMOTE_LLM_API_KEY') 'missing key message names the variable'

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $off = Get-RemoteBackendStatus -Config $offline
    $sw.Stop()
    Assert-Equal $off.state 'offline' 'closed port is offline'
    Assert-True ($sw.ElapsedMilliseconds -lt 5000) "offline check is quick ($($sw.ElapsedMilliseconds) ms)"

    # A name that will not resolve is offline too, not an exception.
    $unresolvable = Get-RemoteBackendConfig -ConfigPath $null -Environment @{ AI_REMOTE_LLM_URL = 'http://hp-does-not-exist.invalid:10000' }
    Assert-Equal (Get-RemoteBackendStatus -Config $unresolvable).state 'offline' 'unresolvable host is offline'

    Assert-Equal (Get-RemoteBackendStatus -Config $none).state 'not-configured' 'no settings is not-configured'

    $shared.health = 503
    Assert-Equal (Get-RemoteBackendStatus -Config $good).state 'loading' '/health 503 means the model is loading'
    $shared.health = 500
    Assert-Equal (Get-RemoteBackendStatus -Config $good).state 'error' 'other /health failures are errors'
    $shared.health = 200

    # ---- Chat proxy ----
    $req = '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":false}'
    $r = Invoke-RemoteChatProxy -Config $good -RequestBody $req
    Assert-Equal $r.status 200 'chat is relayed'
    Assert-Equal (($r.body | ConvertFrom-Json).choices[0].message.content) 'hello from hp' 'the reply body is passed back'
    Assert-Equal $shared.auth "Bearer $secret" 'the key is sent as a bearer token'
    Assert-Equal $shared.body $req 'the request body is forwarded unchanged'

    $r = Invoke-RemoteChatProxy -Config $wrongKey -RequestBody $req
    Assert-Equal $r.status 502 'a rejected key is a 502'
    Assert-Equal (($r.body | ConvertFrom-Json).error.state) 'unauthorized' 'a rejected key says unauthorized'

    $r = Invoke-RemoteChatProxy -Config $offline -RequestBody $req
    Assert-Equal $r.status 503 'offline server is a 503'
    $err = ($r.body | ConvertFrom-Json).error
    Assert-Equal $err.state 'offline' 'offline is named in the error'
    Assert-True ($err.message -match '127\.0\.0\.1') 'offline error names the address'

    $r = Invoke-RemoteChatProxy -Config $none -RequestBody $req
    Assert-Equal (($r.body | ConvertFrom-Json).error.state) 'not-configured' 'no settings is not-configured'

    $r = Invoke-RemoteChatProxy -Config $good -RequestBody 'not json'
    Assert-Equal $r.status 400 'non-JSON bodies are refused'
    $r = Invoke-RemoteChatProxy -Config $good -RequestBody '{"messages":[],"stream":true}'
    Assert-Equal $r.status 400 'streaming requests are refused'

    $shared.redirect = $true
    $r = Invoke-RemoteChatProxy -Config $good -RequestBody $req
    Assert-Equal $r.status 302 'redirects are not followed'
    $shared.redirect = $false
} finally {
    $listener.Stop(); $listener.Close()
    try { $serverPs.EndInvoke($serverHandle) } catch {}
    $serverPs.Dispose()
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

# ---- Dashboard wiring (static checks; the dashboard itself needs Windows) ----
$serverFile = Join-Path $repo 'dashboard-server.ps1'
$tokens = $null; $parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($serverFile, [ref]$tokens, [ref]$parseErrors)
Assert-Equal @($parseErrors).Count 0 'dashboard-server.ps1 parses'
$serverText = Get-Content -Raw -LiteralPath $serverFile
Assert-True ($serverText -match "'\^/api/backends\$'") 'dashboard serves /api/backends'
Assert-True ($serverText -match "'\^/api/remote/v1/chat/completions\$'") 'dashboard relays remote chat'
$html = Get-Content -Raw -LiteralPath (Join-Path $repo 'dashboard.html')
Assert-True ($html -notmatch 'AI_REMOTE_LLM_API_KEY|Authorization') 'the page never handles the key'
Assert-True ($html -match '/api/remote/v1') 'the page sends HP chat through the dashboard'
$gitignore = Get-Content -Raw -LiteralPath (Join-Path $repo '.gitignore')
Assert-True ($gitignore -match '(?m)^Config/remote-backend\.json$') 'the settings file is gitignored'

if ($failures.Count) {
    $failures | ForEach-Object { Write-Host "FAIL: $_" -ForegroundColor Red }
    Write-Host "$($failures.Count) of $assertions assertions failed." -ForegroundColor Red
    exit 1
}
Write-Host "Remote backend tests passed ($assertions assertions)." -ForegroundColor Green
