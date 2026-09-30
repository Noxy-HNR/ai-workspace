# Install only this project's shortcut; leave other desktop entries unchanged.
param([string]$DesktopPath = [Environment]::GetFolderPath('Desktop'))
$ErrorActionPreference = 'Stop'
if (-not $DesktopPath -or -not (Test-Path -LiteralPath $DesktopPath -PathType Container)) {
    throw 'The Windows Desktop folder could not be resolved.'
}
$target = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$launcher = Join-Path $PSScriptRoot 'open-telegram-bot.ps1'
$icon = Join-Path $PSScriptRoot 'static\relay-station.ico'
foreach ($path in @($target,$launcher,$icon)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Shortcut dependency is missing: $path" }
}
$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path $DesktopPath 'Relay Station - Telegram Bot.lnk'
$link = $shell.CreateShortcut($shortcutPath)
$link.TargetPath = $target
$link.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$launcher+'"'
$link.WorkingDirectory = $PSScriptRoot
$link.Description = 'Start Relay Station and open the local Telegram bot console'
$link.IconLocation = $icon+',0'
$link.WindowStyle = 7
$link.Save()
Write-Output "Installed shortcut: $shortcutPath"
