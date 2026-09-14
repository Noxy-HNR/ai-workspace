$ErrorActionPreference = 'Stop'
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
$icons = [ordered]@{
    'Chemistry Workbench' = 'chemistry'
    'Research Library' = 'research'
    'Spectroscopy Explorer' = 'spectra'
    'AI Evaluation Lab' = 'evaluation'
    'Lecture Notes Dashboard' = 'lecture'
    'Local AI Stack' = 'ai-stack'
    'Personal Health Service' = 'health'
    'TI-84 Chemistry - Transfer Files' = 'ti84'
    'IR Camera Viewer' = 'ir-camera'
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ProjectIconNotification {
 [DllImport("shell32.dll", CharSet=CharSet.Unicode)]
 public static extern void SHChangeNotify(uint eventId, uint flags, string path, IntPtr extra);
}
'@
foreach ($name in $icons.Keys) {
    $path = Join-Path $desktop ($name+'.lnk')
    $icon = Join-Path $PSScriptRoot ('icons\'+$icons[$name]+'.ico')
    if (-not (Test-Path -LiteralPath $path)) { throw "Shortcut is missing: $path" }
    if (-not (Test-Path -LiteralPath $icon)) { throw "Icon is missing: $icon" }
    $link = $shell.CreateShortcut($path)
    $previous = @($link.TargetPath,$link.Arguments,$link.WorkingDirectory,$link.WindowStyle)
    $link.IconLocation = $icon+',0'
    $link.Save()
    $verified = $shell.CreateShortcut($path)
    $current = @($verified.TargetPath,$verified.Arguments,$verified.WorkingDirectory,$verified.WindowStyle)
    for ($i=0;$i -lt $previous.Count;$i++) {
        if ($previous[$i] -cne $current[$i]) { throw "Launch configuration unexpectedly changed: $name" }
    }
    if ($verified.IconLocation -ne $icon+',0') { throw "Icon update failed: $name" }
    [ProjectIconNotification]::SHChangeNotify(0x2000,0x0005,$path,[IntPtr]::Zero)
    Write-Output "Updated icon: $name"
}
