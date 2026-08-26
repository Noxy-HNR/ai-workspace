$ErrorActionPreference="Stop"
Set-Location "C:\AI"

Write-Host "Checking for a newer llama.cpp build..." -ForegroundColor Cyan
$llamaBinDir = "C:\AI\Tools\llama-native\bin"
$llamaExe = "$llamaBinDir\llama-server.exe"
$currentVersion = if (Test-Path $llamaExe) { (& $llamaExe --version 2>&1 | Select-String "version:").ToString() } else { "" }

$release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" -TimeoutSec 20 | Select-Object -First 1
$tag = $release.tag_name

if ($currentVersion -match [regex]::Escape($tag)) {
    Write-Host "llama.cpp is already up to date ($tag)." -ForegroundColor Green
} else {
    Write-Host "Updating llama.cpp to $tag..." -ForegroundColor Cyan
    $mainAsset = $release.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-13\.3-x64\.zip$' } | Select-Object -First 1
    $cudartAsset = $release.assets | Where-Object { $_.name -eq 'cudart-llama-bin-win-cuda-13.3-x64.zip' }
    if ($mainAsset -and $cudartAsset) {
        $llamaConn = Get-NetTCPConnection -LocalPort 10000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($llamaConn) { Stop-Process -Id $llamaConn.OwningProcess -Force -ErrorAction SilentlyContinue }
        $tmpDir = "C:\AI\Tools\llama-native"
        Invoke-WebRequest -Uri $mainAsset.browser_download_url -OutFile "$tmpDir\llama-cuda.zip" -TimeoutSec 180
        Invoke-WebRequest -Uri $cudartAsset.browser_download_url -OutFile "$tmpDir\cudart.zip" -TimeoutSec 180
        Expand-Archive -Path "$tmpDir\llama-cuda.zip" -DestinationPath "$tmpDir\bin" -Force
        Expand-Archive -Path "$tmpDir\cudart.zip" -DestinationPath "$tmpDir\bin" -Force
        Remove-Item "$tmpDir\llama-cuda.zip", "$tmpDir\cudart.zip" -Force
        Write-Host "llama.cpp updated." -ForegroundColor Green
    } else {
        Write-Host "Could not find a matching CUDA 13.3 build for $tag; skipping." -ForegroundColor Yellow
    }
}

Write-Host "Updating cptr..." -ForegroundColor Cyan
python -m pip install --upgrade "cptr[all]"

Write-Host "Restarting..." -ForegroundColor Cyan
& "C:\AI\start.ps1"

Write-Host "Update complete." -ForegroundColor Green
