[CmdletBinding()]
param([string]$AiRoot = 'C:\AI')

$ErrorActionPreference = 'Stop'
$source = @'
using System;
using System.Runtime.InteropServices;
public static class LocalCodeNative {
  [StructLayout(LayoutKind.Sequential)] public struct POWER_STATUS {
    public byte ACLineStatus; public byte BatteryFlag; public byte BatteryLifePercent;
    public byte Reserved1; public int BatteryLifeTime; public int BatteryFullLifeTime;
  }
  [StructLayout(LayoutKind.Sequential)] public struct MEMORY_STATUS_EX {
    public uint dwLength; public uint dwMemoryLoad; public ulong ullTotalPhys;
    public ulong ullAvailPhys; public ulong ullTotalPageFile; public ulong ullAvailPageFile;
    public ulong ullTotalVirtual; public ulong ullAvailVirtual; public ulong ullAvailExtendedVirtual;
  }
  [DllImport("kernel32.dll")] public static extern bool GetSystemPowerStatus(out POWER_STATUS value);
  [DllImport("kernel32.dll")] public static extern bool GlobalMemoryStatusEx(ref MEMORY_STATUS_EX value);
}
'@
if (-not ('LocalCodeNative' -as [type])) { Add-Type -TypeDefinition $source }
$power = New-Object LocalCodeNative+POWER_STATUS
$memory = New-Object LocalCodeNative+MEMORY_STATUS_EX
$memory.dwLength = [Runtime.InteropServices.Marshal]::SizeOf([type]'LocalCodeNative+MEMORY_STATUS_EX')
if (-not [LocalCodeNative]::GetSystemPowerStatus([ref]$power)) { throw 'GetSystemPowerStatus failed.' }
if (-not [LocalCodeNative]::GlobalMemoryStatusEx([ref]$memory)) { throw 'GlobalMemoryStatusEx failed.' }
$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
}
$serverProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'llama-server.exe'" -ErrorAction SilentlyContinue)
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort 10000 -ErrorAction SilentlyContinue)
[pscustomobject]@{
    AcPower = ($power.ACLineStatus -eq 1)
    BatteryPercent = $power.BatteryLifePercent
    AvailablePhysicalGiB = [Math]::Round($memory.ullAvailPhys / 1GB, 2)
    TotalPhysicalGiB = [Math]::Round($memory.ullTotalPhys / 1GB, 2)
    MemoryLoadPercent = $memory.dwMemoryLoad
    Gpu = $gpu
    LlamaServerProcessCount = $serverProcesses.Count
    Port10000ListenerCount = $listeners.Count
    Dashboard9090Listening = [bool](Get-NetTCPConnection -State Listen -LocalPort 9090 -ErrorAction SilentlyContinue | Select-Object -First 1)
}
