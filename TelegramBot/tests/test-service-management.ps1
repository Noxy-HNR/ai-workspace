<#
  Deterministic lifecycle ownership tests. No real processes, network requests,
  credentials, live bot data, or service start/stop operations are used.
  Run: powershell -NoProfile -ExecutionPolicy Bypass -File tests\test-service-management.ps1
#>
$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'service-management.ps1')

# Trap accidental service work if a pure ownership helper later grows a side effect.
$sideEffectAttempts = 0
function Block-LiveOperation {
    param([string]$Operation)
    $script:sideEffectAttempts++
    throw "Fixture forbids $Operation."
}
function Get-NetTCPConnection { Block-LiveOperation 'listener inspection' }
function Get-CimInstance { Block-LiveOperation 'live process inspection' }
function Get-Process { Block-LiveOperation 'live process inspection' }
function Invoke-RestMethod { Block-LiveOperation 'network requests' }
function Invoke-WebRequest { Block-LiveOperation 'network requests' }
function Start-Process { Block-LiveOperation 'starting a process' }
function Stop-Process { Block-LiveOperation 'stopping a process' }
function Wait-Process { Block-LiveOperation 'waiting on a live process' }

$fixtureBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixtureRoot = Join-Path $fixtureBase ('relay-management-test-' + [guid]::NewGuid().ToString('N'))
$assertions = 0

function Assert-Result {
    param([bool]$Actual, [bool]$Expected, [string]$Message)
    $script:assertions++
    if ($Actual -ne $Expected) { throw "$Message (expected $Expected, got $Actual)" }
}

function New-ProcessFixture {
    param([int]$ProcessId = 41234, [string]$Name = 'python.exe',
          [string]$CommandLine = ('"C:\Fixture Python\python.exe" -u "{0}" --port 9095' -f $RelayEntry),
          [datetime]$Created = ([datetimeoffset]'2026-09-29T12:00:00Z').UtcDateTime)
    return [pscustomobject]@{
        ProcessId = $ProcessId; Name = $Name; CommandLine = $CommandLine; CreationDate = $Created
    }
}

function Write-RecordFixture {
    param($Record)
    $Record | ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding UTF8
}

try {
    $null = New-Item -ItemType Directory -Path $fixtureRoot
    $RelayRoot = Join-Path $fixtureRoot 'Relay Station'
    $RelayEntry = Join-Path $RelayRoot 'app.py'
    $null = New-Item -ItemType Directory -Path $RelayRoot
    $recordPath = Join-Path $RelayRoot 'run\instance-9095.json'
    $process = New-ProcessFixture

    Assert-Result (Test-RelayProcessCommand -Process $process) $true 'Quoted absolute entry is recognized'
    Assert-Result (Test-RelayProcessCommand -Process (New-ProcessFixture -Name 'pythonw.exe')) $true 'Windowless Python is recognized'
    Assert-Result (Test-RelayProcessCommand -Process $null) $false 'Missing process is rejected'
    Assert-Result (Test-RelayProcessCommand -Process (New-ProcessFixture -Name 'cmd.exe')) $false 'Another executable is rejected'
    Assert-Result (Test-RelayProcessCommand -Process (New-ProcessFixture -Name 'python.exe.evil')) $false 'Python name suffix is rejected'
    Assert-Result (Test-RelayProcessCommand -Process $process -Port 9096) $false 'Wrong listening port is rejected'
    Assert-Result (Test-RelayProcessCommand -Process (New-ProcessFixture -CommandLine ('python.exe -u "{0}" --port 9096' -f $RelayEntry))) $false 'Wrong command port is rejected'
    Assert-Result (Test-RelayProcessCommand -Process (New-ProcessFixture -CommandLine 'python.exe -u C:\Other\app.py --port 9095')) $false 'Another app.py is rejected'
    Assert-Result (Test-RelayProcessCommand -Process (New-ProcessFixture -CommandLine ('python.exe "{0}" --port 9095' -f $RelayEntry))) $false 'Unrecognized launch shape is rejected'
    Assert-Result (Test-RelayProcessCommand -Process (New-ProcessFixture -CommandLine ('python.exe -u "{0}" --port 9095 --extra' -f $RelayEntry))) $false 'Extra trailing arguments are rejected'
    $legacy = New-ProcessFixture -CommandLine 'python.exe -u app.py --port 9095'
    Assert-Result (Test-RelayProcessCommand -Process $legacy) $false 'Legacy relative command requires explicit adoption'
    Assert-Result (Test-RelayProcessCommand -Process $legacy -AllowLegacy) $true 'Explicit legacy adoption recognizes the original launch command'

    Assert-Result (Test-RelayOwnedProcess -Process $process) $false 'A matching command alone does not prove ownership'
    Save-RelayInstance -Process $process
    Assert-Result (Test-RelayOwnedProcess -Process $process) $true 'Saved process identity is recognized'
    Assert-Result (Test-RelayOwnedProcess -Process (New-ProcessFixture -ProcessId 41235)) $false 'Another PID is rejected'
    Assert-Result (Test-RelayOwnedProcess -Process (New-ProcessFixture -Created ([datetimeoffset]'2026-09-29T12:00:01Z').UtcDateTime)) $false 'Reused PID with another creation time is rejected'
    Assert-Result (Test-RelayOwnedProcess -Process (New-ProcessFixture -CommandLine ('"C:\Another Python\python.exe" -u "{0}" --port 9095' -f $RelayEntry))) $false 'Changed command is rejected'
    Assert-Result (Test-RelayOwnedProcess -Process $process -Port 9096) $false 'Ownership cannot be used for another port'

    $original = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    foreach ($change in @(
        @{ Key = 'process_id'; Value = 99999 },
        @{ Key = 'port'; Value = 9096 },
        @{ Key = 'entry'; Value = 'C:\Other\app.py' },
        @{ Key = 'command'; Value = 'python.exe -u app.py --port 9095' },
        @{ Key = 'created'; Value = '2026-09-29T12:00:01.0000000Z' }
    )) {
        $record = $original | ConvertTo-Json | ConvertFrom-Json
        $record.($change.Key) = $change.Value
        Write-RecordFixture $record
        Assert-Result (Test-RelayOwnedProcess -Process $process) $false ('Changed ownership record is rejected: ' + $change.Key)
    }

    '{not valid JSON' | Set-Content -LiteralPath $recordPath -Encoding UTF8
    Assert-Result (Test-RelayOwnedProcess -Process $process) $false 'Malformed ownership record fails closed'
    '{}' | Set-Content -LiteralPath $recordPath -Encoding UTF8
    Assert-Result (Test-RelayOwnedProcess -Process $process) $false 'Incomplete ownership record fails closed'
    Save-RelayInstance -Process $legacy -AllowLegacy
    Assert-Result (Test-RelayOwnedProcess -Process $legacy) $true 'Adopted legacy identity can be managed'
    Assert-Result (Test-RelayOwnedProcess -Process $process) $false 'Legacy record does not authorize another command'

    $rejected = $false
    try { Save-RelayInstance -Process (New-ProcessFixture -Name 'cmd.exe') }
    catch { $rejected = $true }
    Assert-Result $rejected $true 'Saving an unrelated process is refused'
    Assert-Result (Test-RelayOwnedProcess -Process $legacy) $true 'Refused adoption preserves the previous ownership record'
    Assert-Result ($sideEffectAttempts -eq 0) $true 'Ownership validation performs no live process or network operations'
    Write-Output "$assertions ownership assertions passed; no live processes or network operations used."
} finally {
    # Resolve and check the exact generated fixture before recursive cleanup.
    $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
    $expectedPrefix = Join-Path $fixtureBase 'relay-management-test-'
    if (-not $resolvedFixture.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Fixture cleanup refused: resolved target is outside the generated temporary directory prefix.'
    }
    if (Test-Path -LiteralPath $resolvedFixture -PathType Container) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
}
