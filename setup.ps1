<#
  Local AI Workstation - one-click installer
  Target: Windows 11 + RTX 5070 Ti Mobile 12GB + 16GB RAM

  Installs/configures:
    - llama.cpp CUDA (native Windows binary -- no Docker/WSL)
    - Python + cptr
    - Git
    - Visual C++ Redistributable
    - optional Brave/Exa/Tavily search
    - organized C:\AI workspace
    - local Windows-control helper scripts

  IMPORTANT:
    cptr's current CLI does NOT accept --cwd or --api-key. Its first-run
    setup token/account and gateway API key are created in the cptr UI.
    This installer therefore launches cptr correctly and opens its setup UI.

  cptr runs natively as the current Windows user and has the same
  filesystem/process access as that account. llama-server also now runs
  natively (previously ran in Docker); this removes the Docker Desktop/WSL2
  memory overhead entirely, which was measured at several GB of idle RAM.
#>

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$TargetRoot = "C:\AI"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Step($m){Write-Host "`n==> $m" -ForegroundColor Cyan}
function Ok($m){Write-Host "    $m" -ForegroundColor Green}
function Warn($m){Write-Host "    $m" -ForegroundColor Yellow}
function Fail($m){Write-Host "    $m" -ForegroundColor Red}

function Set-EnvValue([string]$path,[string]$name,[string]$value){
    $lines = @()
    if(Test-Path $path){$lines=@(Get-Content $path)}
    $esc=[regex]::Escape($name); $found=$false
    $out=foreach($line in $lines){
        if($line -match "^\s*$esc="){$found=$true;"$name=$value"}else{$line}
    }
    if(-not $found){$out += "$name=$value"}
    Set-Content $path $out -Encoding UTF8
}

function New-Secret([int]$bytes=32){
    $b=New-Object byte[] $bytes
    [Security.Cryptography.RandomNumberGenerator]::Fill($b)
    ([Convert]::ToBase64String($b)-replace '[^A-Za-z0-9]','')
}

Write-Host "============================================================" -ForegroundColor Magenta
Write-Host " C:\AI LOCAL AI WORKSTATION" -ForegroundColor Magenta
Write-Host " RTX 5070 Ti Mobile 12GB / 16GB RAM" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta

# Relocate project
if($Here.TrimEnd('\') -ne $TargetRoot.TrimEnd('\')){
    Step "Copying the project to $TargetRoot"
    New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
    Get-ChildItem $Here -Force | ForEach-Object {
        if($_.Name -notin @("models","Backups","Data",".venv","__pycache__")){
            Copy-Item $_.FullName $TargetRoot -Recurse -Force
        }
    }
    Ok "Copied."
    Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","$TargetRoot\setup.ps1"
    exit
}
Set-Location $TargetRoot

Step "Creating organized C:\AI directories"
foreach($d in @("models","models\incoming","models\archive","Downloads","Workspace","Projects","Projects\_templates","Databases","Data","Backups","Logs","Tools","Tools\llama-native","Tools\scripts","Config")){
    New-Item -ItemType Directory -Force -Path "$TargetRoot\$d" | Out-Null
}
Ok "Directory tree ready."

Step "Checking/installing prerequisites"
if(-not (Get-Command winget -ErrorAction SilentlyContinue)){
    throw "winget/App Installer is required."
}

if(-not (Get-Command python -ErrorAction SilentlyContinue)){
    Warn "Installing Python 3.12..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
}
if(-not (Get-Command git -ErrorAction SilentlyContinue)){
    Warn "Installing Git..."
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
}
# cptr's Windows terminal backend may need this runtime.
try{
    winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements
}catch{
    Warn "VC++ Redistributable installation was skipped/failed; cptr terminal may require it."
}

$env:Path=[Environment]::GetEnvironmentVariable("Path","Machine")+";"+[Environment]::GetEnvironmentVariable("Path","User")

Step "Checking NVIDIA"
$cudaVersion = $null
if(Get-Command nvidia-smi -ErrorAction SilentlyContinue){
    try{
        $gpu=& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        Ok "GPU: $gpu"
        $smiOutput = & nvidia-smi 2>&1 | Out-String
        if ($smiOutput -match 'CUDA UMD Version:\s*([\d.]+)') {
            $cudaVersion = $matches[1]
            Ok "CUDA driver version: $cudaVersion"
        }
    }catch{Warn "nvidia-smi returned an error."}
}else{
    Warn "nvidia-smi not found. Install/update NVIDIA drivers before GPU inference."
}

Step "Installing llama.cpp (native, no Docker)"
$llamaBinDir = "$TargetRoot\Tools\llama-native\bin"
$llamaExe = "$llamaBinDir\llama-server.exe"
if (Test-Path $llamaExe) {
    Ok "llama-server.exe already installed."
} else {
    $release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" -TimeoutSec 20 | Select-Object -First 1
    $tag = $release.tag_name

    # Match the CUDA build to the installed driver's CUDA version; known
    # published variants are 12.4 and 13.3 as of this writing. Fall back
    # to 12.4 (broadly backward compatible) if detection or an exact
    # match fails.
    $cudaAssetVersion = "12.4"
    if ($cudaVersion) {
        $available = $release.assets | Where-Object { $_.name -match '^llama-.*-bin-win-cuda-[\d.]+-x64\.zip$' } |
            ForEach-Object { if ($_.name -match 'cuda-([\d.]+)-x64') { $matches[1] } }
        if ($available -contains $cudaVersion) {
            $cudaAssetVersion = $cudaVersion
        } else {
            Warn "No exact llama.cpp build for CUDA $cudaVersion; using $cudaAssetVersion (should still work)."
        }
    }

    $mainAsset = $release.assets | Where-Object { $_.name -eq "llama-$tag-bin-win-cuda-$cudaAssetVersion-x64.zip" }
    $cudartAsset = $release.assets | Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-$cudaAssetVersion-x64.zip" }
    if (-not $mainAsset -or -not $cudartAsset) {
        throw "Could not find llama.cpp Windows CUDA $cudaAssetVersion release assets for $tag."
    }

    Ok "Downloading llama.cpp $tag (CUDA $cudaAssetVersion)..."
    $tmpDir = "$TargetRoot\Tools\llama-native"
    Invoke-WebRequest -Uri $mainAsset.browser_download_url -OutFile "$tmpDir\llama-cuda.zip" -TimeoutSec 180
    Invoke-WebRequest -Uri $cudartAsset.browser_download_url -OutFile "$tmpDir\cudart.zip" -TimeoutSec 180
    Expand-Archive -Path "$tmpDir\llama-cuda.zip" -DestinationPath "$tmpDir\bin" -Force
    Expand-Archive -Path "$tmpDir\cudart.zip" -DestinationPath "$tmpDir\bin" -Force
    Remove-Item "$tmpDir\llama-cuda.zip", "$tmpDir\cudart.zip" -Force

    if (-not (Test-Path $llamaExe)) { throw "llama-server.exe not found after extraction." }
    Ok "llama.cpp installed to $llamaBinDir"
}

Step "Writing .env"
$envFile="$TargetRoot\.env"
if(-not (Test-Path $envFile)){Copy-Item "$TargetRoot\.env.example" $envFile}

Step "Selecting model"
Write-Host "  1) Qwen3.5 9B Q4_K_M    - fastest"
Write-Host "  2) Qwen3.5 9B UD-Q6_K_XL - recommended, best quality for 12GB VRAM"
$choice=Read-Host "Enter 1 or 2 (Enter=2)"
if($choice -eq "1"){
    $repo="bartowski/Qwen_Qwen3.5-9B-GGUF"
    $file="Qwen3.5-9B-Q4_K_M.gguf"
}else{
    $repo="bartowski/Qwen_Qwen3.5-9B-GGUF"
    $file="Qwen3.5-9B-UD-Q6_K_XL.gguf"
}
Set-EnvValue $envFile "MODEL_REPO" $repo
Ok "Model: $file"

Step "Configuring web search"
Write-Host "  1) Brave"
Write-Host "  2) Exa"
Write-Host "  3) Tavily"
Write-Host "  4) Skip"
$sc=Read-Host "Choice (Enter=Skip)"
$provider=""; $key=""
switch($sc){
 "1" {$provider="brave";$key=Read-Host "Brave API key"}
 "2" {$provider="exa";$key=Read-Host "Exa API key"}
 "3" {$provider="tavily";$key=Read-Host "Tavily API key"}
}
Set-EnvValue $envFile "ENABLE_WEB_SEARCH" ($(if($key){"true"}else{"false"}))
Set-EnvValue $envFile "WEB_SEARCH_ENGINE" ($(if($provider){$provider}else{"brave"}))
if($key){
    if($provider -eq "brave"){Set-EnvValue $envFile "BRAVE_API_KEY" $key}
    if($provider -eq "exa"){Set-EnvValue $envFile "EXA_API_KEY" $key}
    if($provider -eq "tavily"){Set-EnvValue $envFile "TAVILY_API_KEY" $key}
    Ok "Search configured: $provider"
}else{Ok "Search can be configured later in cptr."}

Step "Downloading model"
$model="$TargetRoot\models\$file"
if(Test-Path $model){Ok "Already present."}
else{
    $url="https://huggingface.co/$repo/resolve/main/$file`?download=true"
    Ok "Download URL: $url"
    curl.exe -L --fail --retry 5 --retry-delay 3 -C - -o $model $url
    if($LASTEXITCODE -ne 0){throw "Model download failed."}
    Ok "Model downloaded."
}

Step "Writing model configuration"
$modelsConfigFile = "$TargetRoot\Config\models.json"
$defaultSettings = [ordered]@{
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
    cacheTypeK    = "f16"
    cacheTypeV    = "f16"
    extraArgs     = ""
}
$modelsConfig = [ordered]@{
    activeModel = $file
    models = [ordered]@{}
}
$modelsConfig.models | Add-Member -NotePropertyName $file -NotePropertyValue $defaultSettings -Force
$modelsConfig | ConvertTo-Json -Depth 10 | Set-Content $modelsConfigFile -Encoding UTF8
Ok "Config\models.json written with $file as the active model."

Step "Installing cptr"
python -m pip install --upgrade pip
python -m pip install --upgrade "cptr[all]"
if($LASTEXITCODE -ne 0){throw "cptr installation failed."}
Ok "cptr installed."

Step "Starting the stack"
& "$TargetRoot\start.ps1"
