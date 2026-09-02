"""Admission control: decide whether a job can safely start *right now*.

This is what turns gpuq from a GPU queue into a broker for any heavy workload.
A job declares what it needs (RAM, VRAM, CPUs) and is admitted only when that
request fits, judged against two independent limits:

* **Measured headroom** - what the OS says is free this instant. This is the
  only thing that accounts for work gpuq did not start (another agent's
  training run, a browser, a WSL VM), and it is what a pure slot-count queue
  is blind to.
* **Reserved headroom** - the sum of what already-running gpuq jobs asked for.
  A job that started ten seconds ago may not have allocated its peak yet, so
  measured free memory alone would happily admit a second job that cannot
  possibly fit once both reach full size.

A request must satisfy both. That is what lets several small jobs run in
parallel while two large ones serialise, without a human choosing a slot count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from gpuq import host
from gpuq.config import Config
from gpuq.gpu import GpuInfo

_GIB_MIB = 1024.0


@dataclass(frozen=True)
class ResourceRequest:
    """What one job needs. `None` means "use the configured default"."""

    ram_mib: float = 0.0
    vram_mib: float = 0.0
    cpus: int = 0
    gpu_count: int = 0

    @classmethod
    def from_job(
        cls,
        config: Config,
        *,
        ram_mib: float | None,
        vram_mib: float | None,
        cpus: int | None,
        gpu_count: int,
    ) -> ResourceRequest:
        r = config.resources
        return cls(
            ram_mib=r.default_ram_gb * _GIB_MIB if ram_mib is None else float(ram_mib),
            vram_mib=r.default_vram_gb * _GIB_MIB if vram_mib is None else float(vram_mib),
            cpus=r.default_cpus if cpus is None else int(cpus),
            gpu_count=int(gpu_count),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ram_mib": self.ram_mib,
            "vram_mib": self.vram_mib,
            "cpus": self.cpus,
            "gpu_count": self.gpu_count,
        }


@dataclass
class Capacity:
    """Total machine capacity, minus the headroom we never hand out."""

    total_ram_mib: float
    usable_ram_mib: float
    total_cpus: int
    usable_cpus: int
    total_vram_mib: float
    usable_vram_mib: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ram_mib": self.total_ram_mib,
            "usable_ram_mib": self.usable_ram_mib,
            "total_cpus": self.total_cpus,
            "usable_cpus": self.usable_cpus,
            "total_vram_mib": self.total_vram_mib,
            "usable_vram_mib": self.usable_vram_mib,
        }


@dataclass
class Decision:
    """Outcome of an admission check."""

    admit: bool
    reason: str | None = None
    #: Machine-readable detail, recorded in telemetry so `gpuq report` can
    #: explain later why a job sat blocked.
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.admit


def cpu_count() -> int:
    return os.cpu_count() or 1


def capacity(config: Config, gpu: GpuInfo | None = None, mem: host.HostMemory | None = None) -> Capacity:
    mem = mem or host.memory()
    r = config.resources

    total_ram = mem.total_mib or 0.0
    usable_ram = max(0.0, total_ram - r.reserve_ram_gb * _GIB_MIB)

    total_cpus = cpu_count()
    usable_cpus = max(1, total_cpus - r.reserve_cpus)

    total_vram = 0.0
    if gpu is not None and gpu.available:
        total_vram = sum(d.memory_total_mib or 0.0 for d in gpu.devices)
    usable_vram = max(0.0, total_vram - r.reserve_vram_gb * _GIB_MIB)

    return Capacity(
        total_ram_mib=total_ram,
        usable_ram_mib=usable_ram,
        total_cpus=total_cpus,
        usable_cpus=usable_cpus,
        total_vram_mib=total_vram,
        usable_vram_mib=usable_vram,
    )


def sum_reservations(running: list[ResourceRequest]) -> ResourceRequest:
    return ResourceRequest(
        ram_mib=sum(r.ram_mib for r in running),
        vram_mib=sum(r.vram_mib for r in running),
        cpus=sum(r.cpus for r in running),
        gpu_count=sum(r.gpu_count for r in running),
    )


def _gib(mib: float | None) -> str:
    return f"{(mib or 0.0) / _GIB_MIB:.1f} GiB"


def admit(
    config: Config,
    request: ResourceRequest,
    running: list[ResourceRequest],
    *,
    gpu: GpuInfo | None = None,
    mem: host.HostMemory | None = None,
) -> Decision:
    """Can this request start right now?

    Returns a `Decision` whose `reason` is written verbatim into the job's
    wait reason, so `gpuq status` always explains why something is not running
    instead of appearing mysteriously stuck.
    """
    r = config.resources
    if not r.enforce:
        return Decision(True, detail={"enforced": False})

    mem = mem or host.memory()
    cap = capacity(config, gpu=gpu, mem=mem)
    reserved = sum_reservations(running)

    detail: dict[str, Any] = {
        "request": request.to_dict(),
        "reserved": reserved.to_dict(),
        "capacity": cap.to_dict(),
        "host_free_mib": mem.available_mib,
        "host_free_percent": mem.free_percent,
        "commit_percent": mem.commit_percent,
        "running_jobs": len(running),
    }

    # ---- hard stops -----------------------------------------------------
    # These apply even to a job that asks for nothing: if the machine is
    # already in trouble, starting more work is how a dev box falls over.
    commit = mem.commit_percent
    if commit is not None and commit >= r.max_commit_percent:
        return Decision(
            False,
            f"system commit charge is {commit:.0f}% (limit {r.max_commit_percent}%) - "
            f"{_gib(mem.commit_used_mib)} of {_gib(mem.commit_limit_mib)} committed",
            detail,
        )

    free_percent = mem.free_percent
    if free_percent is not None and free_percent < r.min_host_free_percent:
        return Decision(
            False,
            f"host RAM is {free_percent:.0f}% free ({_gib(mem.available_mib)}), "
            f"below the {r.min_host_free_percent}% floor",
            detail,
        )

    # ---- RAM ------------------------------------------------------------
    if request.ram_mib > 0:
        # 1. Would it fit in what is measurably free? This is the check that
        #    sees foreign workloads.
        floor = cap.total_ram_mib * (r.min_host_free_percent / 100.0)
        spare_now = (mem.available_mib or 0.0) - floor
        if request.ram_mib > spare_now:
            return Decision(
                False,
                f"needs {_gib(request.ram_mib)} RAM but only {_gib(max(0.0, spare_now))} "
                f"is free after the {r.min_host_free_percent}% floor "
                f"({_gib(mem.available_mib)} free of {_gib(cap.total_ram_mib)})",
                detail,
            )
        # 2. Would it fit once every running job reaches its declared size?
        if reserved.ram_mib + request.ram_mib > cap.usable_ram_mib:
            return Decision(
                False,
                f"needs {_gib(request.ram_mib)} RAM; {len(running)} running job(s) "
                f"already reserve {_gib(reserved.ram_mib)} of {_gib(cap.usable_ram_mib)} usable",
                detail,
            )

    # ---- CPUs -----------------------------------------------------------
    if request.cpus > 0 and reserved.cpus + request.cpus > cap.usable_cpus:
        return Decision(
            False,
            f"needs {request.cpus} CPU(s); {reserved.cpus} of {cap.usable_cpus} "
            f"usable are already reserved",
            detail,
        )

    # ---- VRAM -----------------------------------------------------------
    # Only an explicit VRAM request is size-checked here. Whole-device
    # allocation and the free-memory threshold stay in the dispatcher, which
    # owns per-device placement.
    if request.vram_mib > 0:
        if gpu is None or not gpu.available:
            return Decision(
                False,
                f"needs {_gib(request.vram_mib)} VRAM but no NVIDIA GPU is available",
                detail,
            )
        best_free = max((d.memory_free_mib or 0.0) for d in gpu.devices) if gpu.devices else 0.0
        if request.vram_mib > best_free:
            return Decision(
                False,
                f"needs {_gib(request.vram_mib)} VRAM but the freest GPU has "
                f"{_gib(best_free)}",
                detail,
            )
        if reserved.vram_mib + request.vram_mib > cap.usable_vram_mib:
            return Decision(
                False,
                f"needs {_gib(request.vram_mib)} VRAM; running job(s) reserve "
                f"{_gib(reserved.vram_mib)} of {_gib(cap.usable_vram_mib)} usable",
                detail,
            )

    return Decision(True, None, detail)


def describe_capacity(config: Config) -> dict[str, Any]:
    """Human/machine summary for `gpuq resources` and the dashboard."""
    from gpuq.gpu import query_gpus

    mem = host.memory()
    gpu = query_gpus(include_processes=False)
    cap = capacity(config, gpu=gpu, mem=mem)
    return {
        "enforced": config.resources.enforce,
        "capacity": cap.to_dict(),
        "host": mem.to_dict(),
        "reserve": {
            "ram_gb": config.resources.reserve_ram_gb,
            "vram_gb": config.resources.reserve_vram_gb,
            "cpus": config.resources.reserve_cpus,
        },
        "limits": {
            "max_commit_percent": config.resources.max_commit_percent,
            "min_host_free_percent": config.resources.min_host_free_percent,
        },
        "defaults": {
            "ram_gb": config.resources.default_ram_gb,
            "vram_gb": config.resources.default_vram_gb,
            "cpus": config.resources.default_cpus,
        },
    }
