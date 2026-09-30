[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateSet('Audit', 'Apply', 'Rollback')]
    [string]$Action = 'Audit',
    [string]$ConfigPath = (Join-Path $env:USERPROFILE '.cptr\config.toml'),
    [string]$BackupPath
)

$ErrorActionPreference = 'Stop'
$targetedTools = @('delegate_task', 'update_memory', 'update_tasks')

function Get-TomlStringValue {
    param([string]$Text, [string]$Key)
    $escapedKey = [regex]::Escape($Key)
    $pattern = '(?m)^"' + $escapedKey + '"\s*=\s*"(?<raw>(?:\\.|[^"\\])*)"\s*$'
    $matches = [regex]::Matches($Text, $pattern)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one TOML string for '$Key'; found $($matches.Count). No change made."
    }
    $tomlQuoted = '"' + $matches[0].Groups['raw'].Value + '"'
    return (ConvertFrom-Json -InputObject $tomlQuoted)
}

function Set-TomlStringValue {
    param([string]$Text, [string]$Key, [string]$Value)
    $escapedKey = [regex]::Escape($Key)
    $pattern = '(?m)^"' + $escapedKey + '"\s*=\s*"(?:\\.|[^"\\])*"\s*$'
    $matches = [regex]::Matches($Text, $pattern)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one TOML string for '$Key'; found $($matches.Count). No change made."
    }
    $encoded = ConvertTo-Json -InputObject $Value -Compress
    $replacement = '"' + $Key + '" = ' + $encoded
    return [regex]::Replace($Text, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $replacement }, 1)
}

function Get-ApprovalState {
    param([string]$Text)
    $default = Get-TomlStringValue -Text $Text -Key 'tool_approval.default_builtin_approval'
    if ($default -notin @('allow', 'review')) {
        throw "Unsupported cptr approval policy '$default'. No change made."
    }
    $overridesJson = Get-TomlStringValue -Text $Text -Key 'tool_approval.builtin_tools'
    $overrides = ConvertFrom-Json -InputObject $overridesJson
    if ($null -eq $overrides) { $overrides = [pscustomobject]@{} }
    $browserProvider = Get-TomlStringValue -Text $Text -Key 'browser.provider'
    return [pscustomobject]@{ Default = $default; Overrides = $overrides; BrowserProvider = $browserProvider }
}

function Set-ApprovalState {
    param([string]$Text, [string]$Default, [psobject]$Overrides, [string]$BrowserProvider)
    $overridesJson = ConvertTo-Json -InputObject $Overrides -Depth 20 -Compress
    $Text = Set-TomlStringValue -Text $Text -Key 'tool_approval.default_builtin_approval' -Value $Default
    $Text = Set-TomlStringValue -Text $Text -Key 'tool_approval.builtin_tools' -Value $overridesJson
    return Set-TomlStringValue -Text $Text -Key 'browser.provider' -Value $BrowserProvider
}

function Assert-CptrStopped {
    param([string]$Text)
    if ($Text -match '(?m)^port\s*=\s*(?<port>\d+)\s*$') {
        $port = [int]$Matches['port']
        $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($listener) {
            throw "cptr appears to be listening on port $port (PID $($listener.OwningProcess)). Stop cptr before editing its config."
        }
    }
}

function Write-ConfigSafely {
    param([string]$Path, [string]$Text)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
    $encoding = [System.Text.UTF8Encoding]::new($hasBom)
    $tempPath = Join-Path (Split-Path -Parent $Path) ('.config.toml.' + [guid]::NewGuid().ToString('N') + '.tmp')
    $swapBackup = "$Path.replace-$([guid]::NewGuid().ToString('N')).bak"
    try {
        [System.IO.File]::WriteAllText($tempPath, $Text, $encoding)
        [System.IO.File]::Replace($tempPath, $Path, $swapBackup)
        Remove-Item -LiteralPath $swapBackup -Force
    } finally {
        if (Test-Path -LiteralPath $tempPath) { Remove-Item -LiteralPath $tempPath -Force }
        if (Test-Path -LiteralPath $swapBackup) { Remove-Item -LiteralPath $swapBackup -Force }
    }
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "cptr config was not found at '$ConfigPath'. No change made."
}
$configText = [System.IO.File]::ReadAllText($ConfigPath, [System.Text.Encoding]::UTF8)
$current = Get-ApprovalState -Text $configText

switch ($Action) {
    'Audit' {
        $overrides = @{}
        foreach ($property in $current.Overrides.PSObject.Properties) { $overrides[$property.Name] = $property.Value }
        [pscustomobject]@{
            ConfigPath = $ConfigPath
            DefaultBuiltinApproval = $current.Default
            BrowserProvider = $current.BrowserProvider
            TargetedToolOverrides = ($targetedTools | ForEach-Object {
                if ($overrides.ContainsKey($_)) { "$_=$($overrides[$_])" } else { "$_=<inherits default>" }
            }) -join '; '
            CptrListener = (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -eq 8000 | Measure-Object).Count -gt 0
        } | Format-List
        break
    }
    'Apply' {
        Assert-CptrStopped -Text $configText
        $changed = $current.Default -ne 'review' -or $current.BrowserProvider -ne 'local'
        foreach ($tool in $targetedTools) {
            $property = $current.Overrides.PSObject.Properties[$tool]
            if (-not $property -or $property.Value -ne 'review') { $changed = $true }
        }
        if (-not $changed) {
            Write-Output 'Safety profile is already applied; no file changed.'
            break
        }
        if (-not $PSCmdlet.ShouldProcess($ConfigPath, 'Back up and set cptr built-in approval defaults to review')) { break }

        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $createdBackup = "$ConfigPath.cptr-safety-$stamp.bak"
        Copy-Item -LiteralPath $ConfigPath -Destination $createdBackup
        Set-Acl -LiteralPath $createdBackup -AclObject (Get-Acl -LiteralPath $ConfigPath)

        $overrides = $current.Overrides
        foreach ($tool in $targetedTools) {
            $property = $overrides.PSObject.Properties[$tool]
            if ($property) { $property.Value = 'review' }
            else { $overrides | Add-Member -NotePropertyName $tool -NotePropertyValue 'review' }
        }
        $updated = Set-ApprovalState -Text $configText -Default 'review' -Overrides $overrides -BrowserProvider 'local'
        Write-ConfigSafely -Path $ConfigPath -Text $updated
        $verified = Get-ApprovalState -Text ([System.IO.File]::ReadAllText($ConfigPath, [System.Text.Encoding]::UTF8))
        if ($verified.Default -ne 'review') { throw 'Post-write verification failed for default approval.' }
        foreach ($tool in $targetedTools) {
            if ($verified.Overrides.$tool -ne 'review') { throw "Post-write verification failed for $tool." }
        }
        if ($verified.BrowserProvider -ne 'local') { throw 'Post-write verification failed for browser.provider.' }
        Write-Output "Applied cptr safety profile. Backup: $createdBackup"
        Write-Output 'Mutative tools inherit cptr reviewer policy, selected allow-listed tools remain unchanged, and browser automation uses local CDP.'
        break
    }
    'Rollback' {
        if (-not $BackupPath -or -not (Test-Path -LiteralPath $BackupPath -PathType Leaf)) {
            throw 'Rollback requires -BackupPath pointing to a backup made by this script. No change made.'
        }
        Assert-CptrStopped -Text $configText
        $baselineText = [System.IO.File]::ReadAllText($BackupPath, [System.Text.Encoding]::UTF8)
        $baseline = Get-ApprovalState -Text $baselineText
        if (-not $PSCmdlet.ShouldProcess($ConfigPath, 'Restore only the approval settings from the selected backup')) { break }

        $rollbackCopy = "$ConfigPath.before-rollback-$(Get-Date -Format 'yyyyMMdd-HHmmss').bak"
        Copy-Item -LiteralPath $ConfigPath -Destination $rollbackCopy
        Set-Acl -LiteralPath $rollbackCopy -AclObject (Get-Acl -LiteralPath $ConfigPath)
        $overrides = $current.Overrides
        foreach ($tool in $targetedTools) {
            $property = $baseline.Overrides.PSObject.Properties[$tool]
            if ($property) {
                $existing = $overrides.PSObject.Properties[$tool]
                if ($existing) { $existing.Value = $property.Value }
                else { $overrides | Add-Member -NotePropertyName $tool -NotePropertyValue $property.Value }
            } else {
                $overrides.PSObject.Properties.Remove($tool)
            }
        }
        $updated = Set-ApprovalState -Text $configText -Default $baseline.Default -Overrides $overrides -BrowserProvider $baseline.BrowserProvider
        Write-ConfigSafely -Path $ConfigPath -Text $updated
        Write-Output "Restored only approval settings from backup. Current config backup: $rollbackCopy"
        break
    }
}
