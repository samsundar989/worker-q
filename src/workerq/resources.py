"""Admission control: decide whether a job can safely start *right now*.

This is what turns worker-q from a GPU queue into a broker for any heavy workload.
A job declares what it needs (RAM, VRAM, CPUs) and is admitted only when that
request fits, judged against two independent limits:

* **Measured headroom** - what the OS says is free this instant. This is the
  only thing that accounts for work worker-q did not start (another agent's
  training run, a browser, a WSL VM), and it is what a pure slot-count queue
  is blind to.
* **Reserved headroom** - the sum of what already-running worker-q jobs asked for.
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

from workerq import host
from workerq.config import Config
from workerq.gpu import GpuInfo

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
    #: Machine-readable detail, recorded in telemetry so `workerq report` can
    #: explain later why a job sat blocked.
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.admit


@dataclass(frozen=True)
class Reserve:
    """Headroom worker-q promises never to hand out.

    Config supplies the standing value, but the machine's owner has to be able
    to reclaim RAM, VRAM or CPU *now* - to play a game, join a call, or just
    get the desktop back - without stopping the queue or restarting the
    daemon. So the live value is kept in the dispatcher's meta table and read
    every tick, the same way the slot count and GPU threshold already are.
    """

    ram_mib: float
    vram_mib: float
    cpus: int
    #: Name of the preset that set this, for explaining a wait reason.
    label: str | None = None
    #: ISO timestamp after which the reserve reverts, or None to hold it.
    expires_at: str | None = None

    @classmethod
    def from_config(cls, config: Config) -> Reserve:
        r = config.resources
        return cls(
            ram_mib=r.reserve_ram_gb * _GIB_MIB,
            vram_mib=r.reserve_vram_gb * _GIB_MIB,
            cpus=r.reserve_cpus,
        )

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        # Compared as instants, not via age_seconds(), which clamps at zero -
        # that would report a deadline an hour away as already reached and
        # release every timed reserve on the next tick.
        from workerq.util import parse_iso, utcnow

        deadline = parse_iso(self.expires_at)
        return deadline is not None and utcnow() >= deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "ram_mib": self.ram_mib,
            "vram_mib": self.vram_mib,
            "cpus": self.cpus,
            "label": self.label,
            "expires_at": self.expires_at,
        }


def cpu_count() -> int:
    return os.cpu_count() or 1


def capacity(
    config: Config,
    gpu: GpuInfo | None = None,
    mem: host.HostMemory | None = None,
    reserve: Reserve | None = None,
) -> Capacity:
    mem = mem or host.memory()
    reserve = reserve or Reserve.from_config(config)

    total_ram = mem.total_mib or 0.0
    usable_ram = max(0.0, total_ram - reserve.ram_mib)

    total_cpus = cpu_count()
    usable_cpus = max(1, total_cpus - reserve.cpus)

    total_vram = 0.0
    if gpu is not None and gpu.available:
        total_vram = sum(d.memory_total_mib or 0.0 for d in gpu.devices)
    usable_vram = max(0.0, total_vram - reserve.vram_mib)

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
    reserve: Reserve | None = None,
) -> Decision:
    """Can this request start right now?

    Returns a `Decision` whose `reason` is written verbatim into the job's
    wait reason, so `workerq status` always explains why something is not running
    instead of appearing mysteriously stuck.
    """
    r = config.resources
    if not r.enforce:
        return Decision(True, detail={"enforced": False})

    mem = mem or host.memory()
    reserve = reserve or Reserve.from_config(config)
    cap = capacity(config, gpu=gpu, mem=mem, reserve=reserve)
    reserved = sum_reservations(running)
    # A non-default reserve is the owner reclaiming the machine. When that is
    # why a job cannot start, the wait reason has to say so - otherwise the
    # only symptom is "nothing is starting" with no way to find out why.
    held = f" (held back by reserve '{reserve.label}')" if reserve.label else ""

    detail: dict[str, Any] = {
        "request": request.to_dict(),
        "reserved": reserved.to_dict(),
        "capacity": cap.to_dict(),
        "host_free_mib": mem.available_mib,
        "host_free_percent": mem.free_percent,
        "commit_percent": mem.commit_percent,
        "running_jobs": len(running),
        "reserve": reserve.to_dict(),
    }

    # ---- hard stops -----------------------------------------------------
    # These apply even to a job that asks for nothing: if the machine is
    # already in trouble, starting more work is how a dev box falls over.
    commit = mem.commit_percent
    free_percent = mem.free_percent
    if commit is not None:
        if commit >= r.max_commit_percent:
            return Decision(
                False,
                f"system commit charge is {commit:.0f}% (limit {r.max_commit_percent}%) - "
                f"{_gib(mem.commit_used_mib)} of {_gib(mem.commit_limit_mib)} committed",
                detail,
            )
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
                f"already reserve {_gib(reserved.ram_mib)} of "
                f"{_gib(cap.usable_ram_mib)} usable{held}",
                detail,
            )

    # ---- CPUs -----------------------------------------------------------
    if request.cpus > 0 and reserved.cpus + request.cpus > cap.usable_cpus:
        return Decision(
            False,
            f"needs {request.cpus} CPU(s); {reserved.cpus} of {cap.usable_cpus} "
            f"usable are already reserved{held}",
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
                f"{_gib(reserved.vram_mib)} of {_gib(cap.usable_vram_mib)} usable{held}",
                detail,
            )

    # Will this job's own commit fit in what is left? This is the check
    # that matters on a GPU box, and percentages cannot express it.
    #
    # Under WDDM the driver backs video allocations with system commit, so
    # a job's commit cost is roughly its RAM *plus its VRAM* - measured on
    # one workstation, going from 4 to 17 GiB of VRAM in use raised commit
    # charge by 24 GiB while physical RAM moved by 1.6 GiB. The commit
    # limit is RAM plus pagefile, and a pagefile with a fixed maximum stops
    # growing, so this ceiling is real and can be hit while most of RAM
    # sits free. A job died that way here: 100% commit, 40% of RAM free.
    #
    # Judging that on "commit % is high AND physical RAM is low" never
    # fires, because on this failure physical RAM is never low.
    headroom = None
    if mem.commit_limit_mib is not None and mem.commit_used_mib is not None:
        headroom = mem.commit_limit_mib - mem.commit_used_mib
    if headroom is not None:
        wants = request.ram_mib + request.vram_mib
        margin = cap.total_ram_mib * (r.commit_headroom_percent / 100.0)
        if wants > max(0.0, headroom - margin):
            return Decision(
                False,
                f"needs about {_gib(wants)} of commit (RAM + VRAM) but only "
                f"{_gib(headroom)} is left before the system commit limit "
                f"({_gib(mem.commit_used_mib)} of {_gib(mem.commit_limit_mib)} "
                "used). On Windows, GPU memory is backed by commit, so VRAM "
                "counts here even though physical RAM looks free",
                detail,
            )

    return Decision(True, None, detail)


def describe_capacity(config: Config, reserve: Reserve | None = None) -> dict[str, Any]:
    """Human/machine summary for `workerq resources` and the dashboard.

    Pass the live reserve so this reflects what is actually enforced; without
    it the report shows the configured value while the dispatcher is using
    something else.
    """
    from workerq.gpu import query_gpus

    mem = host.memory()
    reserve = reserve or Reserve.from_config(config)
    gpu = query_gpus(include_processes=False)
    cap = capacity(config, gpu=gpu, mem=mem, reserve=reserve)
    return {
        "enforced": config.resources.enforce,
        "capacity": cap.to_dict(),
        "host": mem.to_dict(),
        "reserve": {
            "ram_gb": reserve.ram_mib / _GIB_MIB,
            "vram_gb": reserve.vram_mib / _GIB_MIB,
            "cpus": reserve.cpus,
            "label": reserve.label,
            "expires_at": reserve.expires_at,
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
