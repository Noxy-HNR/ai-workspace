<#
  Real launcher lifecycle against a disposable stdlib HTTP fixture only.
  Run through Windows PowerShell 5.1; no bot DB, Telegram, models, or browser.
#>
param([string]$PythonPath = (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'))
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$fixtureBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixtureRoot = Join-Path $fixtureBase ('relay-lifecycle-test-' + [guid]::NewGuid().ToString('N'))
$fixtureAppRoot = Join-Path $fixtureRoot 'Relay Station Fixture'
$ownedFixtureProcessId = $null
$port = $null
$checks = 0
$childShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    $script:checks++
    if (-not $Condition) { throw $Message }
}

function Invoke-FixtureScript {
    param([string]$ScriptName, [switch]$WithPython)
    $scriptPath = Join-Path $fixtureAppRoot $ScriptName
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $scriptPath + '"'), '-Port', [string]$port)
    if ($WithPython) { $arguments += @('-PythonPath', ('"' + $PythonPath + '"')) }
    $stamp = [guid]::NewGuid().ToString('N')
    $stdout = Join-Path $fixtureRoot ($stamp + '-script-out.log')
    $stderr = Join-Path $fixtureRoot ($stamp + '-script-error.log')
    # Capture via files: a background Python descendant can retain the pipe
    # used by PowerShell's native-command pipeline after its launcher exits.
    $child = Start-Process -FilePath $childShell -ArgumentList $arguments -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    # Retain the process handle before it exits; Windows PowerShell can
    # otherwise expose a null ExitCode after waiting on a detached process.
    $null = $child.Handle
    if (-not $child.WaitForExit(45000)) {
        $child.Refresh()
        if (-not $child.HasExited) { $child.Kill() }
        throw ('Disposable fixture script exceeded its deadline: ' + $ScriptName)
    }
    $exitCode = $child.ExitCode
    $output = @(Get-Content -LiteralPath $stdout -ErrorAction SilentlyContinue)
    $errors = @(Get-Content -LiteralPath $stderr -ErrorAction SilentlyContinue)
    if ($null -eq $exitCode -or $exitCode -ne 0) {
        throw ('Fixture script failed: ' + $ScriptName + [Environment]::NewLine + (($output + $errors) -join [Environment]::NewLine))
    }
    return $output
}

try {
    Assert-Condition (Test-Path -LiteralPath $PythonPath -PathType Leaf) 'Fixture requires a real Python executable.'
    $null = New-Item -ItemType Directory -Path $fixtureAppRoot
    foreach ($scriptName in @('start-telegram-bot.ps1', 'service-management.ps1', 'stop-telegram-bot.ps1')) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $scriptName) -Destination (Join-Path $fixtureAppRoot $scriptName)
        Assert-Condition ((Get-FileHash -LiteralPath (Join-Path $projectRoot $scriptName)).Hash -eq
                          (Get-FileHash -LiteralPath (Join-Path $fixtureAppRoot $scriptName)).Hash) ('Copied script differs from current source: ' + $scriptName)
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fake_managed_service.py') -Destination (Join-Path $fixtureAppRoot 'app.py')
    . (Join-Path $fixtureAppRoot 'service-management.ps1')

    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
    try {
        $listener.Start()
        $port = $listener.LocalEndpoint.Port
    } finally { $listener.Stop() }

    Write-Output 'Checking cold launch of the disposable loopback fixture.'
    $firstOutput = Invoke-FixtureScript 'start-telegram-bot.ps1' -WithPython
    $health = Get-RelayHealth -Port $port
    Assert-Condition ($health.version -eq 'fixture-lifecycle') 'Cold launch did not become the disposable fixture service.'
    $process = Get-RelayPortProcess -Port $port
    Assert-Condition ($null -ne $process) 'Cold launch has no listening fixture process.'
    $ownedFixtureProcessId = [int]$process.ProcessId
    Assert-Condition (Test-RelayOwnedProcess -Process $process -Port $port) 'Cold launch did not save a matching process identity.'
    Assert-Condition ($process.CommandLine -like ('*' + $fixtureAppRoot + '*')) 'Launcher did not use the exact temporary fixture path.'

    Write-Output ('Checking reuse of disposable fixture PID ' + $ownedFixtureProcessId + '.')
    $secondOutput = Invoke-FixtureScript 'start-telegram-bot.ps1' -WithPython
    $reused = Get-RelayPortProcess -Port $port
    Assert-Condition ($reused.ProcessId -eq $ownedFixtureProcessId) 'Second launch created a duplicate instead of reusing the existing fixture.'
    Assert-Condition (($secondOutput -join ' ') -match 'already running') 'Second launch did not report reuse.'
    Assert-Condition (Test-RelayOwnedProcess -Process $reused -Port $port) 'Reuse did not preserve the fixture ownership record.'

    # No -Force: success proves the authenticated HTTP shutdown path works.
    Write-Output 'Checking graceful fixture stop without -Force.'
    $stopOutput = Invoke-FixtureScript 'stop-telegram-bot.ps1'
    Assert-Condition (($stopOutput -join ' ') -match 'stopped') 'Graceful fixture stop did not acknowledge shutdown.'
    Assert-Condition ($null -eq (Get-RelayPortProcess -Port $port)) 'Graceful stop did not release the fixture listening port.'
    Assert-Condition ($null -eq (Get-CimInstance Win32_Process -Filter "ProcessId=$ownedFixtureProcessId" -ErrorAction SilentlyContinue)) 'Graceful stop did not exit the fixture process.'
    $ownedFixtureProcessId = $null
    Write-Output "$checks lifecycle checks passed: cold launch, PID reuse, owned identity, graceful stop, and released port."
} finally {
    # Failed launches may leave their own fixture process. Recover only the
    # identity saved by the copied helper; never enumerate or stop other apps.
    if ($port -and (Test-Path -LiteralPath (Join-Path $fixtureAppRoot "run\instance-$port.json"))) {
        try {
            $record = Get-Content -LiteralPath (Join-Path $fixtureAppRoot "run\instance-$port.json") -Raw | ConvertFrom-Json
            $remaining = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record.process_id)" -ErrorAction SilentlyContinue
            if ($remaining -and (Test-RelayOwnedProcess -Process $remaining -Port $port)) {
                Stop-Process -Id ([int]$remaining.ProcessId) -Force -ErrorAction Stop
                Wait-Process -Id ([int]$remaining.ProcessId) -Timeout 5 -ErrorAction SilentlyContinue
            }
        } catch { Write-Warning ('Could not clean up owned disposable fixture: ' + $_.Exception.Message) }
    }
    $resolvedFixture = [IO.Path]::GetFullPath($fixtureRoot)
    $expectedPrefix = Join-Path $fixtureBase 'relay-lifecycle-test-'
    if (-not $resolvedFixture.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Fixture cleanup refused: resolved path does not match its generated temporary directory.'
    }
    if (Test-Path -LiteralPath $resolvedFixture -PathType Container) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
}
