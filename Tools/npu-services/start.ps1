$python = 'C:\AI\Projects\lecture-notes\venv\Scripts\python.exe'
$mutex = [System.Threading.Mutex]::new($false, 'Local\ProjectNpuService')
$owned = $false
try {
try { $owned = $mutex.WaitOne(20000) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
if (-not $owned) { throw 'Another NPU service startup is in progress.' }
try {
    $health = Invoke-RestMethod 'http://127.0.0.1:8788' -TimeoutSec 2
    if ($health.service -eq 'project-npu') { Write-Output 'NPU service is already running.'; exit }
    throw 'Port 8788 belongs to another service.'
} catch {
    if (Get-NetTCPConnection -LocalPort 8788 -State Listen -ErrorAction SilentlyContinue) { throw }
}
Start-Process -FilePath $python -ArgumentList '"C:\AI\Tools\npu-services\server.py"' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "$PSScriptRoot\server.log" -RedirectStandardError "$PSScriptRoot\server-error.log"
for ($attempt=0; $attempt -lt 40; $attempt++) {
    try { $health=Invoke-RestMethod 'http://127.0.0.1:8788' -TimeoutSec 1; if ($health.service -eq 'project-npu') { break } } catch {}
    Start-Sleep -Milliseconds 250
}
} finally { if ($owned) { $mutex.ReleaseMutex() }; $mutex.Dispose() }
