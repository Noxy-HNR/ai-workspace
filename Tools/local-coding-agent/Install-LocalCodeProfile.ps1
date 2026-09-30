[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory)]
    [string]$Repository,

    [Parameter(Mandatory)]
    [string]$ModelFile,

    [string]$AiRoot = 'C:\AI',
    [switch]$Replace
)

$ErrorActionPreference = 'Stop'
$repositoryPath = [System.IO.Path]::GetFullPath($Repository)
$configPath = Join-Path $AiRoot 'Config\models.json'
$endpoint = 'http://127.0.0.1:10000'

if (-not (Test-Path -LiteralPath $repositoryPath -PathType Container)) {
    throw "Repository directory does not exist: $repositoryPath"
}
if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
    throw 'OpenCode is not on PATH. Install or repair it before using this profile.'
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Model configuration is missing: $configPath"
}

$settings = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($settings.activeModel -ne $ModelFile) {
    throw "The dashboard config selects '$($settings.activeModel)', not '$ModelFile'. Select the model through the dashboard first. This installer never switches models."
}
$entryProperty = $settings.models.PSObject.Properties[$ModelFile]
if (-not $entryProperty) {
    throw "Model file is not listed in the dashboard config: $ModelFile"
}
$entry = $entryProperty.Value

# Require exactly one live server and verify its binary matches the configured model runtime.
$serverProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" -ErrorAction SilentlyContinue)
if ($serverProcesses.Count -ne 1) {
    throw "Expected exactly one llama-server process; found $($serverProcesses.Count). Refusing to continue."
}
$expectedBinary = if ($entry.serverBinary) {
    if ([System.IO.Path]::IsPathRooted([string]$entry.serverBinary)) { [string]$entry.serverBinary }
    else { Join-Path $AiRoot ([string]$entry.serverBinary) }
} else {
    Join-Path $AiRoot 'Tools\llama-native\bin\llama-server.exe'
}
if ([System.IO.Path]::GetFullPath($serverProcesses[0].ExecutablePath) -ine [System.IO.Path]::GetFullPath($expectedBinary)) {
    throw "The live llama-server binary does not match this model's configured runtime. Expected '$expectedBinary'."
}
$listener = @(Get-NetTCPConnection -State Listen -LocalPort 10000 -ErrorAction SilentlyContinue)
if ($listener.Count -lt 1 -or @($listener | Where-Object { $_.OwningProcess -ne $serverProcesses[0].ProcessId }).Count -gt 0) {
    throw 'Port 10000 is not owned by the single configured llama-server process.'
}

# Use the real server-reported model ID; never invent an alias or silently follow another active model.
$modelResponse = Invoke-RestMethod -Uri "$endpoint/v1/models" -TimeoutSec 5
$liveModels = @($modelResponse.data)
if ($liveModels.Count -ne 1 -or [string]::IsNullOrWhiteSpace([string]$liveModels[0].id)) {
    throw 'The local server did not report exactly one usable model ID at /v1/models.'
}
$modelId = [string]$liveModels[0].id

$targetConfigDir = Join-Path $repositoryPath '.opencode'
$targetConfig = Join-Path $targetConfigDir 'opencode.jsonc'
$targetReviewer = Join-Path (Join-Path $targetConfigDir 'agents') 'reviewer.md'
if ((Test-Path -LiteralPath $targetConfig) -and -not $Replace) {
    throw "An OpenCode project config already exists at '$targetConfig'. Use -Replace to back it up before replacing it."
}

$context = [Math]::Min([int]$entry.contextSize, 32768)
$output = [Math]::Min(4096, [Math]::Max(2048, [int]($context / 4)))
$profile = [ordered]@{
    '$schema' = 'https://opencode.ai/config.json'
    model = 'local-llama/coding'
    snapshots = $true
    compaction = [ordered]@{ auto = $true; buffer = 4096; keep = [ordered]@{ tokens = 4096 } }
    tool_output = [ordered]@{ max_lines = 1200; max_bytes = 49152 }
    providers = [ordered]@{
        'local-llama' = [ordered]@{
            name = 'Dashboard-managed local llama.cpp'
            package = '@opencode/ai/providers/openai-compatible'
            settings = [ordered]@{ baseURL = "$endpoint/v1" }
            models = [ordered]@{
                coding = [ordered]@{
                    modelID = $modelId
                    name = $ModelFile
                    capabilities = [ordered]@{ tools = $true; input = @('text'); output = @('text') }
                    limit = [ordered]@{ context = $context; output = $output }
                }
            }
        }
    }
    permissions = @(
        [ordered]@{ action = 'read'; resource = '*'; effect = 'allow' },
        [ordered]@{ action = 'glob'; resource = '*'; effect = 'allow' },
        [ordered]@{ action = 'grep'; resource = '*'; effect = 'allow' },
        [ordered]@{ action = 'edit'; resource = '*'; effect = 'allow' },
        [ordered]@{ action = 'external_directory'; resource = '*'; effect = 'ask' },
        [ordered]@{ action = 'shell'; resource = '*'; effect = 'ask' },
        [ordered]@{ action = 'shell'; resource = 'git status *'; effect = 'allow' },
        [ordered]@{ action = 'shell'; resource = 'git diff *'; effect = 'allow' },
        [ordered]@{ action = 'shell'; resource = 'git log *'; effect = 'allow' },
        [ordered]@{ action = 'shell'; resource = 'git show *'; effect = 'allow' },
        [ordered]@{ action = 'shell'; resource = 'git push *'; effect = 'deny' },
        [ordered]@{ action = 'shell'; resource = 'git reset *'; effect = 'deny' },
        [ordered]@{ action = 'shell'; resource = 'git clean *'; effect = 'deny' },
        [ordered]@{ action = 'shell'; resource = 'git rebase *'; effect = 'deny' }
    )
    agents = [ordered]@{
        build = [ordered]@{
            steps = 24
            system = 'Work only in the active repository. Inspect project instructions and git status before editing. Prefer small changes, run the most relevant existing tests, inspect the final diff, and report commands and outcomes. If context is compacted, first re-read the task and current diff/status before continuing.'
        }
    }
    experimental = [ordered]@{
        policies = @(
            [ordered]@{ action = 'provider.use'; resource = '*'; effect = 'deny' },
            [ordered]@{ action = 'provider.use'; resource = 'local-llama'; effect = 'allow' },
            [ordered]@{ action = 'permission'; resource = 'shell:git push *'; effect = 'deny' },
            [ordered]@{ action = 'permission'; resource = 'shell:git reset *'; effect = 'deny' },
            [ordered]@{ action = 'permission'; resource = 'shell:git clean *'; effect = 'deny' },
            [ordered]@{ action = 'permission'; resource = 'shell:git rebase *'; effect = 'deny' }
        )
    }
}
$json = $profile | ConvertTo-Json -Depth 30
$reviewer = @'
---
description: Reviews repository changes for defects and missing tests
mode: subagent
permissions:
  - action: edit
    resource: "*"
    effect: deny
  - action: shell
    resource: "*"
    effect: ask
---

Review the current diff for correctness, regressions, security, and missing tests. Read relevant surrounding code and report only actionable findings with file and line references. Do not edit files.
'@

if ($PSCmdlet.ShouldProcess($targetConfig, 'Install local-only OpenCode project profile')) {
    New-Item -ItemType Directory -Path $targetConfigDir -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path $targetReviewer) -Force | Out-Null
    if (Test-Path -LiteralPath $targetConfig) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        Copy-Item -LiteralPath $targetConfig -Destination "$targetConfig.$stamp.bak"
    }
    Set-Content -LiteralPath $targetConfig -Value $json -Encoding utf8
    Set-Content -LiteralPath $targetReviewer -Value $reviewer -Encoding utf8
    Write-Output "Installed local-only profile: $targetConfig"
    Write-Output "Server model ID: $modelId"
    Write-Output "Working context budget: $context tokens"
}
