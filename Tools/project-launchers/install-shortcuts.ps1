$ErrorActionPreference = 'Stop'
$desktop = [Environment]::GetFolderPath('Desktop')
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$shell = New-Object -ComObject WScript.Shell
$launcher = Join-Path $PSScriptRoot 'launch-project.ps1'
$entries = @(
    @{Name='Chemistry Workbench';Key='chemistry';Icon=23},
    @{Name='Research Library';Key='research';Icon=23},
    @{Name='Spectroscopy Explorer';Key='spectra';Icon=23},
    @{Name='AI Evaluation Lab';Key='evaluation';Icon=23},
    @{Name='Lecture Notes Dashboard';Key='lecture';Icon=1},
    @{Name='IR Camera Viewer';Key='ir';Icon=22}
)
foreach ($entry in $entries) {
    $link = $shell.CreateShortcut((Join-Path $desktop ($entry.Name+'.lnk')))
    $link.TargetPath = $powershell
    $link.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$launcher+'" -Project '+$entry.Key
    $link.WorkingDirectory = 'C:\AI'
    $link.Description = 'Start '+$entry.Name+' locally'
    $iconName = if ($entry.Key -eq 'ir') { 'ir-camera' } else { $entry.Key }
    $link.IconLocation = (Join-Path $PSScriptRoot ('icons\'+$iconName+'.ico'))+',0'
    $link.WindowStyle = 7
    $link.Save()
}
foreach ($entry in @(
    @{Name='Local AI Stack';Path='C:\AI\start.ps1';Root='C:\AI'},
    @{Name='Personal Health Service';Path='C:\AI\Personal-health-data\start-health.ps1';Root='C:\AI\Personal-health-data'}
)) {
    if (-not (Test-Path -LiteralPath $entry.Path)) { throw "Missing launcher: $($entry.Path)" }
    $link = $shell.CreateShortcut((Join-Path $desktop ($entry.Name+'.lnk')))
    $link.TargetPath = $powershell
    $link.Arguments = '-NoProfile -ExecutionPolicy Bypass -File "'+$entry.Path+'"'
    $link.WorkingDirectory = $entry.Root
    $link.Description = 'Start using the existing project launcher'
    $iconName = if ($entry.Name -eq 'Local AI Stack') { 'ai-stack' } else { 'health' }
    $link.IconLocation = (Join-Path $PSScriptRoot ('icons\'+$iconName+'.ico'))+',0'
    $link.WindowStyle = 1
    $link.Save()
}
$link = $shell.CreateShortcut((Join-Path $desktop 'TI-84 Chemistry - Transfer Files.lnk'))
$link.TargetPath = 'C:\AI\Projects\ti84evo-chem\release'
$link.Description = 'Open the chemistry programs to transfer to your TI-84 Evo'
$link.IconLocation = (Join-Path $PSScriptRoot 'icons\ti84.ico')+',0'
$link.Save()
Write-Output "Created 9 project shortcuts in $desktop"
