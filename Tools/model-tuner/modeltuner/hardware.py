"""Read-only hardware probes: RAM, GPU, CPU topology and power state.

Everything here only *reports facts*. Nothing in this module labels a value as a
fault or an anomaly, and nothing suggests a change to the machine. In particular a
GPU sitting in a low-power state such as ``P8`` while idle is normal and
intentional on this laptop, not a problem to fix.

The power state is recorded for one reason only: two measurements are only
comparable when they were taken in the same power state, so every sample carries
a :meth:`PowerState.key`.

Every probe degrades to ``None`` rather than raising, because a tuner that cannot
read one fact should still be able to run: a machine with no NVIDIA GPU, no
battery, or no power service still has RAM and a CPU worth reporting.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

__all__ = [
    "CpuTopology",
    "GpuState",
    "MemoryState",
    "PowerState",
    "cpu",
    "gpu",
    "memory",
    "parse_nvidia_smi_csv",
    "power",
    "snapshot",
]

IS_WINDOWS = sys.platform == "win32"

_GIB = 1024**3


# --------------------------------------------------------------------------- #
# RAM
# --------------------------------------------------------------------------- #


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class MemoryState:
    """Physical RAM, as reported by ``GlobalMemoryStatusEx``."""

    total_bytes: int
    #: ``ullAvailPhys``: memory available for allocation *without* paging. This is
    #: the number the tuner's RAM preflight uses. It is not "free" memory -- the
    #: standby and modified lists are already counted as available, which is
    #: exactly what makes it the right figure to budget against.
    avail_bytes: int
    #: ``dwMemoryLoad``: percent of physical memory in use.
    load_pct: int

    @property
    def avail_gb(self) -> float:
        return self.avail_bytes / _GIB

    @property
    def total_gb(self) -> float:
        return self.total_bytes / _GIB


def memory() -> MemoryState:
    """Current physical memory.

    Raises:
        OSError: if ``GlobalMemoryStatusEx`` is unavailable or fails. RAM is a
            hard safety input, so a silent zero here would be worse than an
            error.
    """
    if not IS_WINDOWS:
        return MemoryState(total_bytes=0, avail_bytes=0, load_pct=0)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    query.restype = wintypes.BOOL

    status = _MEMORYSTATUSEX()
    # dwLength is how the callee learns the caller's struct size; without it the
    # call fails on a 64-bit host.
    status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not query(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())

    return MemoryState(
        total_bytes=int(status.ullTotalPhys),
        avail_bytes=int(status.ullAvailPhys),
        load_pct=int(status.dwMemoryLoad),
    )


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #

_NVIDIA_SMI = "nvidia-smi"

#: Query order. Parsing depends on this exact order, so the two stay together.
_SMI_QUERY_FIELDS = (
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "pstate",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "driver_version",
)
_SMI_ARGS = (
    f"--query-gpu={','.join(_SMI_QUERY_FIELDS)}",
    "--format=csv,noheader,nounits",
)

#: Tokens nvidia-smi uses to mean "this field has no value right now". Which one
#: you get depends on the driver, the GPU and the query: ``power.limit`` on many
#: laptop parts is reported as ``[N/A]`` rather than a number. Compared against
#: after folding away whitespace, case and any surrounding brackets, so the
#: bracketed and bare spellings are both covered.
_NA_TOKENS = frozenset(
    {
        "",
        "n/a",
        "na",
        "not applicable",
        "not supported",
        "unknown",
    }
)


def _is_na(field: str) -> bool:
    """True when a CSV field is one of nvidia-smi's "no value" spellings."""
    text = field.strip().lower()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text in _NA_TOKENS


@dataclass(frozen=True)
class GpuState:
    """One NVIDIA GPU as reported by ``nvidia-smi``.

    Optional fields are ``None`` when the driver declines to report them; that is
    an absence of data, not a fault. ``pstate`` is the GPU's own power state --
    ``P8`` while idle is the expected resting state on this machine.
    """

    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    pstate: str | None
    util_pct: int | None
    temp_c: int | None
    power_w: float | None
    power_limit_w: float | None
    sm_clock_mhz: int | None
    mem_clock_mhz: int | None
    driver: str | None


def _as_int(field: str) -> int | None:
    """Parse an integer CSV field, or ``None`` if the driver gave no value."""
    if _is_na(field):
        return None
    try:
        # float() first: a few drivers emit "577.0" even with nounits.
        return int(float(field.strip()))
    except ValueError:
        return None


def _as_float(field: str) -> float | None:
    if _is_na(field):
        return None
    try:
        return float(field.strip())
    except ValueError:
        return None


def _as_text(field: str) -> str | None:
    if _is_na(field):
        return None
    return field.strip() or None


def _as_required_int(field: str, label: str) -> int:
    """Parse an integer the caller cannot do without; raise if it is missing."""
    value = _as_int(field)
    if value is None:
        raise ValueError(f"nvidia-smi field {label!r} has no numeric value: {field!r}")
    return value


def parse_nvidia_smi_csv(text: str) -> GpuState:
    """Parse the first GPU row of ``nvidia-smi --format=csv`` output.

    Pure and total over well-formed input, so it can be tested without a GPU.
    Extra leading/trailing whitespace on any field is tolerated. Fields the
    driver marks unavailable (``[N/A]``, ``N/A``, ``[Not Supported]``, ...) become
    ``None``; a row with the wrong number of fields, no name, or a non-numeric
    memory size is malformed and raises.

    nvidia-smi does not quote or escape CSV, so a device name containing a comma
    cannot be represented -- no real board name has one.
    """
    row = next((line for line in text.splitlines() if line.strip()), "")
    if not row:
        raise ValueError("nvidia-smi output contained no data row")

    fields = [part.strip() for part in row.split(",")]
    expected = len(_SMI_QUERY_FIELDS)
    if len(fields) != expected:
        raise ValueError(
            f"expected {expected} comma-separated fields, got {len(fields)}: {row!r}"
        )

    name = _as_text(fields[0])
    if not name:
        raise ValueError(f"nvidia-smi row has no GPU name: {row!r}")

    return GpuState(
        name=name,
        total_mib=_as_required_int(fields[1], "memory.total"),
        used_mib=_as_required_int(fields[2], "memory.used"),
        free_mib=_as_required_int(fields[3], "memory.free"),
        pstate=_as_text(fields[4]),
        util_pct=_as_int(fields[5]),
        temp_c=_as_int(fields[6]),
        power_w=_as_float(fields[7]),
        power_limit_w=_as_float(fields[8]),
        sm_clock_mhz=_as_int(fields[9]),
        mem_clock_mhz=_as_int(fields[10]),
        driver=_as_text(fields[11]),
    )


def gpu(timeout_s: float = 10.0) -> GpuState | None:
    """The first NVIDIA GPU, or ``None`` if ``nvidia-smi`` is absent or fails.

    A few seconds is a perfectly normal result on a laptop whose GPU is asleep;
    the call blocks until the driver answers rather than reporting a half-read
    row. ``timeout_s`` is a hard ceiling so a wedged driver cannot hang a run.
    """
    if not IS_WINDOWS:
        return None

    try:
        completed = subprocess.run(
            [_NVIDIA_SMI, *_SMI_ARGS],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        # No nvidia-smi on PATH, or it could not be started / did not finish.
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return parse_nvidia_smi_csv(completed.stdout)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# CPU topology
# --------------------------------------------------------------------------- #

_RELATION_PROCESSOR_CORE = 0


class _GROUP_AFFINITY(ctypes.Structure):
    """One processor group's core mask: ``KAFFINITY Mask`` + ``WORD Group``."""

    _fields_ = [
        ("Mask", ctypes.c_size_t),  # KAFFINITY == ULONG_PTR
        ("Group", ctypes.c_uint16),
        ("Reserved", ctypes.c_uint16 * 3),
    ]


class _PROCESSOR_RELATIONSHIP_HEAD(ctypes.Structure):
    """The fixed 24-byte prefix of ``PROCESSOR_RELATIONSHIP``.

    The real struct ends in a variable-length ``GroupMask[GroupCount]``, so the
    masks are read separately. The 20 reserved bytes before ``GroupCount`` are
    what gives the struct its documented x64 offsets: the masks start 32 bytes
    into the record.
    """

    _fields_ = [
        ("Flags", ctypes.c_ubyte),
        ("EfficiencyClass", ctypes.c_ubyte),
        ("Reserved", ctypes.c_ubyte * 20),
        ("GroupCount", ctypes.c_uint16),
    ]


class _LOGICAL_PROCESSOR_INFORMATION_EX_HEAD(ctypes.Structure):
    """``SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`` up to the end of the core data."""

    _fields_ = [
        ("Relationship", wintypes.DWORD),
        ("Size", wintypes.DWORD),
        ("Core", _PROCESSOR_RELATIONSHIP_HEAD),
    ]


def _physical_cores() -> list[tuple[int, int]]:
    """``(efficiency_class, logical_processors)`` for every physical core.

    Empty if the topology cannot be read.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GetLogicalProcessorInformationEx
    query.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL

    # First call with a null buffer only reports how much space is needed. Its
    # return value is meaningless; the size is what matters.
    size = wintypes.DWORD(0)
    query(_RELATION_PROCESSOR_CORE, None, ctypes.byref(size))
    if size.value == 0:
        return []

    buffer = ctypes.create_string_buffer(size.value)
    if not query(_RELATION_PROCESSOR_CORE, buffer, ctypes.byref(size)):
        return []

    head_size = ctypes.sizeof(_LOGICAL_PROCESSOR_INFORMATION_EX_HEAD)
    mask_size = ctypes.sizeof(_GROUP_AFFINITY)
    total = size.value
    data = buffer.raw

    cores: list[tuple[int, int]] = []
    offset = 0
    while offset + head_size <= total:
        record = _LOGICAL_PROCESSOR_INFORMATION_EX_HEAD.from_buffer_copy(
            data[offset : offset + head_size]
        )
        record_size = int(record.Size)
        if record_size < head_size:
            break  # malformed; stop rather than spin
        if int(record.Relationship) == _RELATION_PROCESSOR_CORE:
            logical = 0
            for index in range(int(record.Core.GroupCount)):
                start = offset + head_size + index * mask_size
                if start + mask_size > offset + record_size:
                    break
                affinity = _GROUP_AFFINITY.from_buffer_copy(data[start : start + mask_size])
                logical += int(affinity.Mask).bit_count()
            cores.append((int(record.Core.EfficiencyClass), logical))
        offset += record_size
    return cores


def _cpu_name() -> str:
    """Marketing name of CPU 0 from the registry, e.g. ``Intel(R) Core(TM) ...``."""
    if not IS_WINDOWS:
        return platform.processor() or ""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            raw, _value = winreg.QueryValueEx(key, "ProcessorNameString")
    except OSError:
        return platform.processor() or "unknown"
    return str(raw).strip()


@dataclass(frozen=True)
class CpuTopology:
    """Physical/logical core counts, and the P-core / E-core split if any.

    ``performance_cores`` and ``efficiency_cores`` are both ``None`` when every
    core reports the same efficiency class, i.e. the CPU has a single core class
    and the split does not exist rather than being 0.
    """

    name: str
    logical: int
    physical: int
    performance_cores: int | None
    efficiency_cores: int | None


def cpu() -> CpuTopology:
    """CPU topology, counting logical processors from the kernel's group masks.

    Cores in the highest efficiency class are performance cores; the rest are
    efficiency cores.
    """
    cores = _physical_cores() if IS_WINDOWS else []

    logical = sum(count for _cls, count in cores)
    if logical <= 0:
        # Topology unavailable; fall back to the only count we can still trust.
        logical = os.cpu_count() or 1
    physical = len(cores) or logical

    performance: int | None = None
    efficiency: int | None = None
    classes = {cls for cls, _count in cores}
    if len(classes) > 1:
        fastest = max(classes)
        performance = sum(1 for cls, _count in cores if cls == fastest)
        efficiency = sum(1 for cls, _count in cores if cls != fastest)

    return CpuTopology(
        name=_cpu_name(),
        logical=logical,
        physical=physical,
        performance_cores=performance,
        efficiency_cores=efficiency,
    )


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    """``SYSTEM_POWER_STATUS``. The 4th byte is the battery-saver status bit."""

    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


_BEST_PERFORMANCE_GUID = "{ded574b5-45a0-4f42-8737-46345c09c238}"
_BEST_EFFICIENCY_GUID = "{961cc777-2547-4f9d-8174-7d86181b8a7a}"
_BALANCED_GUID = "{00000000-0000-0000-0000-000000000000}"

#: The power-mode overlay schemes Windows knows by name. The all-zero GUID is
#: "balanced"; anything unrecognised is reported as its raw GUID string.
_OVERLAY_NAMES = {
    _BEST_PERFORMANCE_GUID: "best_performance",
    _BEST_EFFICIENCY_GUID: "best_efficiency",
    _BALANCED_GUID: "balanced",
}
_OVERLAY_GUIDS = (
    _BEST_PERFORMANCE_GUID,
    _BEST_EFFICIENCY_GUID,
    _BALANCED_GUID,
)


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid_text(pointer: Any) -> str | None:
    """Render a ``GUID*`` as a lower-case ``{...}`` string."""
    try:
        buffer = ctypes.create_unicode_buffer(64)
        # StringFromGUID2 returns the number of characters written, and 0 on failure.
        if ctypes.WinDLL("ole32").StringFromGUID2(pointer, buffer, 64) == 0:
            return None
    except (OSError, ValueError):
        return None
    text = buffer.value.strip().lower()
    return text or None


def _effective_overlay() -> str | None:
    """The active power-mode overlay: a friendly name, a raw GUID, or ``None``.

    ``None`` means the overlay could not be determined -- either the API is
    missing or it reports that no overlay scheme is in effect (the machine is on
    a plain power plan). That is reported as "unknown" rather than guessed at.
    """
    if not IS_WINDOWS:
        return None
    try:
        # DWORD PowerGetEffectiveOverlayScheme(GUID *EffectiveOverlayGuid): one out-parameter, 0 on success.
        query = ctypes.WinDLL("powrprof.dll", use_last_error=True).PowerGetEffectiveOverlayScheme
        query.argtypes = [ctypes.POINTER(_GUID)]
        query.restype = wintypes.DWORD
    except (OSError, AttributeError):
        return None
    guid = _GUID()
    try:
        if query(ctypes.byref(guid)) != 0:
            return None
    except OSError:
        return None
    found = _guid_text(ctypes.byref(guid))
    if not found:
        return None
    return _OVERLAY_NAMES.get(found, found)


def _power_status() -> _SYSTEM_POWER_STATUS | None:
    """``GetSystemPowerStatus``, or ``None`` if unavailable or it fails."""
    if not IS_WINDOWS:
        return None
    try:
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetSystemPowerStatus
        query.argtypes = [ctypes.POINTER(_SYSTEM_POWER_STATUS)]
        query.restype = wintypes.BOOL
        status = _SYSTEM_POWER_STATUS()
        if not query(ctypes.byref(status)):
            return None
        return status
    except Exception:  # never let a probe failure escape power()
        return None


@dataclass(frozen=True)
class PowerState:
    """Power source, battery charge, battery saver and the power-mode overlay.

    Every field is optional because every field can be genuinely unknown: a
    desktop has no battery, and the power service may be unavailable.
    """

    on_ac: bool | None
    battery_pct: int | None
    saver_on: bool | None
    overlay: str | None

    def key(self) -> str:
        """A stable string identifying the state a sample was taken in.

        Two samples are only comparable when their keys match. Battery percentage
        is deliberately excluded: charge drifts constantly and would make every
        sample look like a different power state.
        """
        if self.on_ac is None:
            source = "unknown"
        else:
            source = "ac" if self.on_ac else "battery"
        if self.saver_on is None:
            saver = "saver-unknown"
        else:
            saver = "saver-on" if self.saver_on else "saver-off"
        return f"{source}|{self.overlay or 'none'}|{saver}"


def power() -> PowerState:
    """Current power state. Never raises; unreadable facts come back ``None``."""
    status = _power_status()
    on_ac: bool | None = None
    battery_pct: int | None = None
    saver_on: bool | None = None
    if status is not None:
        # ACLineStatus: 1 on AC, 0 on battery, 255 unknown.
        on_ac = {0: False, 1: True}.get(int(status.ACLineStatus))
        percent = int(status.BatteryLifePercent)
        battery_pct = None if percent > 100 else percent
        saver_on = bool(int(status.SystemStatusFlag) & 1)
    return PowerState(
        on_ac=on_ac,
        battery_pct=battery_pct,
        saver_on=saver_on,
        overlay=_effective_overlay(),
    )


# --------------------------------------------------------------------------- #
# Everything at once
# --------------------------------------------------------------------------- #


def snapshot() -> dict[str, Any]:
    """A JSON-serialisable view of all four probes, timestamped.

    Power is read last, after the (possibly slow) GPU query, so the recorded
    state is the closest thing to the moment the snapshot was taken.
    """
    state = gpu()
    power_state = power()
    return {
        "taken_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "memory": asdict(memory()),
        "gpu": asdict(state) if state is not None else None,
        "cpu": asdict(cpu()),
        "power": asdict(power_state),
        "power_key": power_state.key(),
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(json.dumps(snapshot(), indent=2))
