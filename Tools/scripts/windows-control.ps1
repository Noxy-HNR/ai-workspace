<#
Windows control helpers for the local agent.

These run as the current Windows user and therefore operate with that
user's normal permissions. They are intentionally simple PowerShell
commands so cptr's real terminal can invoke them.

Examples:
  .\windows-control.ps1 apps
  .\windows-control.ps1 launch notepad.exe
  .\windows-control.ps1 focus "Notepad"
  .\windows-control.ps1 screenshot
  .\windows-control.ps1 mouse 500 400 click
  .\windows-control.ps1 type "hello"
  .\windows-control.ps1 key ENTER
  .\windows-control.ps1 key CTRL+L
#>

param(
    [Parameter(Mandatory=$true,Position=0)]
    [ValidateSet("apps","launch","focus","screenshot","mouse","type","key")]
    [string]$Action,

    [Parameter(Position=1)]
    [string]$Value1,

    [Parameter(Position=2)]
    [string]$Value2,

    [Parameter(Position=3)]
    [string]$Value3
)

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Win32 {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint flags,uint dx,uint dy,uint data,UIntPtr extra);
}
'@

Add-Type -AssemblyName System.Windows.Forms

switch($Action){
  "apps" {
    Get-Process | Where-Object {$_.MainWindowTitle} |
      Select-Object Id,ProcessName,MainWindowTitle |
      Sort-Object ProcessName |
      Format-Table -AutoSize
  }

  "launch" {
    Start-Process $Value1
  }

  "focus" {
    $p=Get-Process | Where-Object {$_.MainWindowTitle -like "*$Value1*"} | Select-Object -First 1
    if(-not $p){throw "No window found matching '$Value1'."}
    [Win32]::ShowWindow($p.MainWindowHandle,9) | Out-Null
    [Win32]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
  }

  "screenshot" {
    $dir="C:\AI\Workspace\Screenshots"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $file=Join-Path $dir ("desktop-"+(Get-Date -Format "yyyyMMdd-HHmmss")+".png")
    $bounds=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp=New-Object System.Drawing.Bitmap $bounds.Width,$bounds.Height
    $g=[System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($bounds.Location,[System.Drawing.Point]::Empty,$bounds.Size)
    $bmp.Save($file,[System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose();$bmp.Dispose()
    Write-Output $file
  }

  "mouse" {
    $x=[int]$Value1;$y=[int]$Value2;$op=$Value3
    [Win32]::SetCursorPos($x,$y) | Out-Null
    if($op -eq "click"){
      [Win32]::mouse_event(0x0002,0,0,0,[UIntPtr]::Zero)
      [Win32]::mouse_event(0x0004,0,0,0,[UIntPtr]::Zero)
    }elseif($op -eq "double"){
      1..2 | ForEach-Object {
        [Win32]::mouse_event(0x0002,0,0,0,[UIntPtr]::Zero)
        [Win32]::mouse_event(0x0004,0,0,0,[UIntPtr]::Zero)
        Start-Sleep -Milliseconds 80
      }
    }
  }

  "type" {
    [System.Windows.Forms.SendKeys]::SendWait($Value1)
  }

  "key" {
    # Examples: ENTER, ESC, CTRL+L, ALT+F4
    [System.Windows.Forms.SendKeys]::SendWait($Value1)
  }
}
