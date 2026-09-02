"""NVIDIA GPU inspection via `nvidia-smi` (spec section 14).

Informational only: the dispatcher is the resource allocator. Every parser here
must tolerate `N/A`, absent processes, WSL/WDDM quirks and a missing driver
without raising, so `gpuq status` / `gpuq doctor` never crash on a metric.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from workerq.winproc import no_window_kwargs

_QUERY_GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
)

_QUERY_APP_FIELDS = ("pid", "process_name", "used_gpu_memory", "gpu_uuid")

_TIMEOUT = 15


@dataclass
class GpuProcess:
    pid: int | None
    process_name: str
    used_memory_mib: float | None
    gpu_uuid: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GpuDevice:
    index: int
    uuid: str
    name: str
    memory_total_mib: float | None
    memory_used_mib: float | None
    memory_free_mib: float | None
    utilization_percent: float | None
    processes: list[GpuProcess] = field(default_factory=list)

    @property
    def free_percent(self) -> float | None:
        if not self.memory_total_mib:
            return None
        if self.memory_free_mib is not None:
            return 100.0 * self.memory_free_mib / self.memory_total_mib
        if self.memory_used_mib is not None:
            return 100.0 * (self.memory_total_mib - self.memory_used_mib) / self.memory_total_mib
        return None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["processes"] = [p.to_dict() for p in self.processes]
        data["free_percent"] = self.free_percent
        return data


@dataclass
class GpuInfo:
    available: bool
    devices: list[GpuDevice] = field(default_factory=list)
    driver_version: str | None = None
    cuda_version: str | None = None
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.devices)

    def max_free_percent(self) -> float | None:
        values = [d.free_percent for d in self.devices if d.free_percent is not None]
        return max(values) if values else None

    def min_free_percent(self) -> float | None:
        values = [d.free_percent for d in self.devices if d.free_percent is not None]
        return min(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "error": self.error,
            "devices": [d.to_dict() for d in self.devices],
        }


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _to_float(text: str) -> float | None:
    """Parse an nvidia-smi numeric cell. Returns None for N/A / junk."""
    value = (text or "").strip()
    if not value or value.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]", "NOT SUPPORTED"}:
        return None
    for suffix in (" MiB", "MiB", " GiB", "GiB", " %", "%", " W", "W"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
            break
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(text: str) -> int | None:
    value = _to_float(text)
    return int(value) if value is not None else None


def nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


def _run(args: list[str]) -> tuple[int, str, str]:
    exe = nvidia_smi_path()
    if exe is None:
        return 127, "", "nvidia-smi not found on PATH"
    try:
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            encoding="utf-8",
            errors="replace",
            **no_window_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return 124, "", "nvidia-smi timed out"
    except OSError as exc:  # pragma: no cover
        return 126, "", f"nvidia-smi failed to start: {exc}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _query(fields: tuple[str, ...], mode: str) -> tuple[list[list[str]], str | None]:
    code, out, err = _run(
        [f"--query-{mode}={','.join(fields)}", "--format=csv,noheader,nounits"]
    )
    if code != 0:
        return [], (err.strip() or out.strip() or f"nvidia-smi exited {code}")
    rows: list[list[str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        # Right-pad short rows so a missing optional column cannot IndexError.
        if len(cells) < len(fields):
            cells += [""] * (len(fields) - len(cells))
        rows.append(cells)
    return rows, None


def _driver_and_cuda() -> tuple[str | None, str | None]:
    rows, err = _query(("driver_version",), "gpu")
    driver = rows[0][0] if rows and not err else None
    # CUDA runtime version is only in the human header; parse it defensively.
    code, out, _ = _run([])
    cuda = None
    if code == 0:
        for line in out.splitlines():
            if "CUDA Version" in line:
                part = line.split("CUDA Version", 1)[1]
                cuda = part.strip(" :|").split()[0] if part.strip(" :|") else None
                break
    return driver, cuda


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def query_gpus(*, include_processes: bool = True) -> GpuInfo:
    """Collect GPU inventory. Never raises."""
    if nvidia_smi_path() is None:
        return GpuInfo(available=False, error="nvidia-smi not found on PATH")

    rows, err = _query(_QUERY_GPU_FIELDS, "gpu")
    if err:
        return GpuInfo(available=False, error=err)
    if not rows:
        return GpuInfo(available=False, error="nvidia-smi reported no GPUs")

    devices: list[GpuDevice] = []
    for cells in rows:
        index = _to_int(cells[0])
        devices.append(
            GpuDevice(
                index=index if index is not None else len(devices),
                uuid=cells[1] or f"gpu-{len(devices)}",
                name=cells[2] or "unknown NVIDIA GPU",
                memory_total_mib=_to_float(cells[3]),
                memory_used_mib=_to_float(cells[4]),
                memory_free_mib=_to_float(cells[5]),
                utilization_percent=_to_float(cells[6]),
            )
        )

    if include_processes:
        proc_rows, proc_err = _query(_QUERY_APP_FIELDS, "compute-apps")
        if not proc_err:
            by_uuid = {d.uuid: d for d in devices}
            for cells in proc_rows:
                gpu_uuid = cells[3] or None
                process = GpuProcess(
                    pid=_to_int(cells[0]),
                    process_name=cells[1] or "?",
                    used_memory_mib=_to_float(cells[2]),
                    gpu_uuid=gpu_uuid,
                )
                target = by_uuid.get(gpu_uuid or "")
                if target is None and len(devices) == 1:
                    target = devices[0]
                if target is not None:
                    target.processes.append(process)

    driver, cuda = _driver_and_cuda()
    return GpuInfo(available=True, devices=devices, driver_version=driver, cuda_version=cuda)


def gpu_free_percent() -> float | None:
    """Best free-memory percentage across devices, or None when unknown."""
    info = query_gpus(include_processes=False)
    return info.max_free_percent() if info.available else None


def foreign_processes(info: GpuInfo, *, own_pids: set[int] | None = None) -> list[GpuProcess]:
    """Compute processes on the GPU that worker-q did not launch."""
    own_pids = own_pids or set()
    out: list[GpuProcess] = []
    for device in info.devices:
        for proc in device.processes:
            if proc.pid is not None and proc.pid not in own_pids:
                out.append(proc)
    return out


def cuda_toolkit_version() -> str | None:
    """Version reported by `nvcc`, if a CUDA toolkit is installed."""
    exe = shutil.which("nvcc")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        if "release" in line:
            for token in line.split():
                token = token.strip(",")
                if token[:1].isdigit():
                    return token
    return None
