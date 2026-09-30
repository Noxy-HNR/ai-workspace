# Shortcut entry point: start/reuse the local service, then open its console.
$ErrorActionPreference = 'Stop'
try {
    & (Join-Path $PSScriptRoot 'start-telegram-bot.ps1') -OpenBrowser
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message,'Relay Station could not open','OK','Error') | Out-Null
    exit 1
}
