"""Host resource inspection: RAM, commit charge, and top consumers.

GPU memory is not the only way concurrent heavy jobs kill each other. A data
loader that memory-maps large volumes exhausts *host* RAM long before it
troubles a 32 GiB GPU, and Windows starts failing allocations once the system
commit charge approaches its limit.

Everything here is best-effort and must never raise: `gpuq status`, `gpuq top`
and `gpuq doctor` all call it on every refresh.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from workerq.winproc import no_window_kwargs

IS_WINDOWS = os.name == "nt"

_MIB = 1024.0 * 1024.0
_GIB = _MIB * 1024.0


@dataclass
class HostMemory:
    total_mib: float | None = None
    available_mib: float | None = None
    #: Windows commit charge. Allocation fails when this reaches its limit,
    #: even while physical RAM still looks free, so it is tracked separately.
    commit_used_mib: float | None = None
    commit_limit_mib: float | None = None
    error: str | None = None

    @property
    def used_mib(self) -> float | None:
        if self.total_mib is None or self.available_mib is None:
            return None
        return max(0.0, self.total_mib - self.available_mib)

    @property
    def free_percent(self) -> float | None:
        if not self.total_mib or self.available_mib is None:
            return None
        return 100.0 * self.available_mib / self.total_mib

    @property
    def commit_percent(self) -> float | None:
        if not self.commit_limit_mib or self.commit_used_mib is None:
            return None
        return 100.0 * self.commit_used_mib / self.commit_limit_mib

    @property
    def available_gib(self) -> float | None:
        return self.available_mib / 1024.0 if self.available_mib is not None else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "used_mib": self.used_mib,
                "free_percent": self.free_percent,
                "commit_percent": self.commit_percent,
            }
        )
        return data


def memory() -> HostMemory:
    """Current host memory. Cheap enough to call every refresh (no subprocess)."""
    if IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return HostMemory(error="GlobalMemoryStatusEx failed")
            return HostMemory(
                total_mib=status.ullTotalPhys / _MIB,
                available_mib=status.ullAvailPhys / _MIB,
                commit_limit_mib=status.ullTotalPageFile / _MIB,
                commit_used_mib=(status.ullTotalPageFile - status.ullAvailPageFile) / _MIB,
            )
        except Exception as exc:  # pragma: no cover
            return HostMemory(error=f"{type(exc).__name__}: {exc}")

    try:  # pragma: no cover - POSIX
        info: dict[str, float] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key.strip()] = float(rest.strip().split()[0]) / 1024.0
        total = info.get("MemTotal")
        available = info.get("MemAvailable", info.get("MemFree"))
        return HostMemory(total_mib=total, available_mib=available)
    except Exception as exc:  # pragma: no cover
        return HostMemory(error=f"{type(exc).__name__}: {exc}")


@dataclass
class ProcessMemory:
    pid: int
    name: str
    memory_mib: float
    command: str = ""

    @property
    def memory_gib(self) -> float:
        return self.memory_mib / 1024.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Cache:
    at: float = 0.0
    value: list[ProcessMemory] = field(default_factory=list)
    tree: dict[int, int] | None = None


_TOP_CACHE = _Cache()
_TOP_TTL = 5.0


def top_processes(limit: int = 8, *, ttl: float = _TOP_TTL) -> list[ProcessMemory]:
    """Largest host-memory consumers, newest-first by size.

    Uses a subprocess, so results are cached briefly - a 1 Hz dashboard must
    not shell out on every frame.
    """
    now = time.monotonic()
    if _TOP_CACHE.value and now - _TOP_CACHE.at < ttl:
        return _TOP_CACHE.value[:limit]

    processes: list[ProcessMemory] = []
    if IS_WINDOWS:
        try:
            proc = subprocess.run(
                ["tasklist", "/fo", "csv", "/nh"],
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
                **no_window_kwargs(),
            )
            if proc.returncode == 0:
                import csv
                import io

                for row in csv.reader(io.StringIO(proc.stdout or "")):
                    if len(row) < 5:
                        continue
                    name, pid_text, mem_text = row[0], row[1], row[4]
                    digits = "".join(c for c in mem_text if c.isdigit())
                    if not digits or not pid_text.isdigit():
                        continue
                    processes.append(
                        ProcessMemory(
                            pid=int(pid_text),
                            name=name,
                            memory_mib=float(digits) / 1024.0,  # tasklist reports KiB
                        )
                    )
        except Exception:
            return _TOP_CACHE.value[:limit]
    else:  # pragma: no cover - POSIX
        try:
            proc = subprocess.run(
                ["ps", "-eo", "pid=,rss=,comm="],
                capture_output=True,
                text=True,
                timeout=30,
                **no_window_kwargs(),
            )
            for line in (proc.stdout or "").splitlines():
                parts = line.split(None, 2)
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                processes.append(
                    ProcessMemory(
                        pid=int(parts[0]),
                        name=parts[2].strip(),
                        memory_mib=float(parts[1]) / 1024.0,
                    )
                )
        except Exception:
            return _TOP_CACHE.value[:limit]

    processes.sort(key=lambda p: p.memory_mib, reverse=True)
    _TOP_CACHE.at = now
    _TOP_CACHE.value = processes
    return processes[:limit]


_TREE_CACHE = _Cache()
_TREE_TTL = 3.0


def parent_map(*, ttl: float = _TREE_TTL) -> dict[int, int]:
    """pid -> parent pid for every process.

    Uses the Toolhelp snapshot API rather than shelling out, so it is cheap
    enough for a live dashboard and adds no console flicker.
    """
    now = time.monotonic()
    cached = getattr(_TREE_CACHE, "tree", None)
    if cached is not None and now - _TREE_CACHE.at < ttl:
        return cached

    mapping: dict[int, int] = {}
    if IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes

            TH32CS_SNAPPROCESS = 0x00000002
            MAX_PATH = 260

            class PROCESSENTRY32(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * MAX_PATH),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            snapshot = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
            if snapshot and snapshot != wintypes.HANDLE(-1).value:
                try:
                    entry = PROCESSENTRY32()
                    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
                    if k32.Process32First(snapshot, ctypes.byref(entry)):
                        while True:
                            mapping[int(entry.th32ProcessID)] = int(
                                entry.th32ParentProcessID
                            )
                            if not k32.Process32Next(snapshot, ctypes.byref(entry)):
                                break
                finally:
                    k32.CloseHandle(snapshot)
        except Exception:  # pragma: no cover
            return cached or {}
    else:  # pragma: no cover - POSIX
        try:
            proc = subprocess.run(
                ["ps", "-eo", "pid=,ppid="],
                capture_output=True,
                text=True,
                timeout=30,
                **no_window_kwargs(),
            )
            for line in (proc.stdout or "").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    mapping[int(parts[0])] = int(parts[1])
        except Exception:
            return cached or {}

    _TREE_CACHE.at = now
    _TREE_CACHE.tree = mapping  # type: ignore[attr-defined]
    return mapping


def descendants_of(roots: set[int]) -> set[int]:
    """Every process descended from `roots`, plus the roots themselves.

    A job's real work is usually a grandchild of the wrapper gpuq launched (an
    interpreter re-exec, a dataloader pool), so attributing memory by the
    runner PID alone would report gpuq's own job as somebody else's.
    """
    if not roots:
        return set()
    mapping = parent_map()
    children: dict[int, list[int]] = {}
    for pid, ppid in mapping.items():
        children.setdefault(ppid, []).append(pid)

    seen = set(roots)
    stack = list(roots)
    while stack:
        current = stack.pop()
        for child in children.get(current, ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def command_line(pid: int) -> str:
    """Full command line for a PID, for attributing memory to a workload."""
    if not IS_WINDOWS:  # pragma: no cover - POSIX
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            return ""
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}')"
                ".CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
            **no_window_kwargs(),
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def summarize_pressure(
    mem: HostMemory | None = None, *, min_free_percent: float = 15.0
) -> tuple[bool, str | None]:
    """(under_pressure, reason). Used for gating and for warnings."""
    mem = mem or memory()
    if mem.error:
        return False, None

    commit = mem.commit_percent
    if commit is not None and commit >= 92.0:
        return True, (
            f"system commit charge is {commit:.0f}% of its limit "
            f"({(mem.commit_used_mib or 0) / 1024:.0f} / "
            f"{(mem.commit_limit_mib or 0) / 1024:.0f} GiB)"
        )

    free = mem.free_percent
    if free is not None and free < min_free_percent:
        return True, (
            f"host RAM is {free:.0f}% free "
            f"({(mem.available_mib or 0) / 1024:.1f} GiB), below the "
            f"{min_free_percent:.0f}% required"
        )
    return False, None
