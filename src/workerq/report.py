"""`workerq report` - why jobs are failing, and who is responsible.

Turns a pile of failed jobs into an answer. Each failure is classified from its
exit code and log, attributed to a project and the agent that submitted it, and
annotated with what the machine looked like at the time - so a job killed by
another workload's memory use is distinguishable from one that had a bug.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from workerq.core import GPUQService
from workerq.models import Job, JobState
from workerq.telemetry import open_telemetry
from workerq.util import age_seconds, parse_iso

_LOG_TAIL_BYTES = 20_000


@dataclass(frozen=True)
class Cause:
    key: str
    label: str
    #: Whether the *machine* caused this, as opposed to the job's own code.
    resource: bool = False
    advice: str = ""


CAUSES = {
    "cuda_oom": Cause(
        "cuda_oom",
        "CUDA out of memory",
        resource=True,
        advice="declare the job's VRAM with --vram so worker-q stops overlapping it",
    ),
    "host_oom": Cause(
        "host_oom",
        "host out of memory",
        resource=True,
        advice="declare --ram so worker-q holds the job until that much is actually free",
    ),
    "killed": Cause(
        "killed",
        "killed by the OS or another process",
        resource=True,
        advice="usually memory pressure; check the pressure column at that timestamp",
    ),
    "missing_file": Cause(
        "missing_file",
        "file or command not found",
        advice="gitignored paths are absent from the snapshot; use an absolute "
        "interpreter path or --passthrough",
    ),
    "import_error": Cause(
        "import_error",
        "import/module error",
        advice="name the project interpreter explicitly rather than bare 'python'",
    ),
    "cancelled": Cause("cancelled", "cancelled"),
    "app_error": Cause("app_error", "job raised an exception"),
    "nonzero": Cause("nonzero", "exited non-zero"),
    "unknown": Cause("unknown", "unknown"),
}

#: Ordered - the first pattern that matches wins, so specific beats generic.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cuda_oom", re.compile(r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation", re.I)),
    ("host_oom", re.compile(r"\bMemoryError\b|out of memory|Cannot allocate memory|bad_alloc|paging file is too small", re.I)),
    ("killed", re.compile(r"\bKilled\b|terminated by signal|Terminated\b|exit code -9|WinError 1455", re.I)),
    ("missing_file", re.compile(r"No such file or directory|cannot find the file|FileNotFoundError|is not recognized as an internal", re.I)),
    ("import_error", re.compile(r"ModuleNotFoundError|ImportError|DLL load failed", re.I)),
]


@dataclass
class Failure:
    job: Job
    cause: Cause
    excerpt: str = ""
    resource_state: dict[str, Any] | None = None
    peak: dict[str, Any] | None = None
    foreign: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job.id,
            "project": self.job.project,
            "agent": self.job.submitter_agent,
            "priority": self.job.priority,
            "state": self.job.state,
            "exit_code": self.job.exit_code,
            "finished_at": self.job.finished_at,
            "runtime_seconds": self.job.runtime_seconds,
            "cause": self.cause.key,
            "cause_label": self.cause.label,
            "resource_caused": self.cause.resource,
            "advice": self.cause.advice,
            "excerpt": self.excerpt,
            "resource_state": self.resource_state,
            "peak_during_run": self.peak,
            "foreign_consumers": self.foreign,
        }


def _log_tail(service: GPUQService, job: Job) -> str:
    path = service.resolve_log_path(job)
    if path is None or not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            if size > _LOG_TAIL_BYTES:
                fh.seek(size - _LOG_TAIL_BYTES)
            return fh.read()
    except OSError:
        return ""


def classify_failure(service: GPUQService, job: Job, log: str | None = None) -> Cause:
    """Best-effort cause for one failed job."""
    if job.state == JobState.CANCELLED.value:
        return CAUSES["cancelled"]
    if job.state == JobState.LOST.value:
        return CAUSES["killed"]

    text = log if log is not None else _log_tail(service, job)
    haystack = f"{text}\n{job.error or ''}"
    for key, pattern in _PATTERNS:
        if pattern.search(haystack):
            return CAUSES[key]

    if job.exit_code == 127:
        return CAUSES["missing_file"]
    if re.search(r"Traceback \(most recent call last\)", haystack):
        return CAUSES["app_error"]
    if job.exit_code not in (None, 0):
        return CAUSES["nonzero"]
    return CAUSES["unknown"]


def _excerpt(text: str, cause: Cause) -> str:
    """The most informative line or two, not the whole log."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("worker-q: ")]
    if not lines:
        return ""
    for key, pattern in _PATTERNS:
        if key != cause.key:
            continue
        for line in reversed(lines):
            if pattern.search(line):
                return line.strip()[:400]
    for line in reversed(lines):
        if re.match(r"^\w*(Error|Exception)\b|^\w+Error:", line.strip()):
            return line.strip()[:400]
    return lines[-1][:400]


def analyse(
    service: GPUQService, *, hours: float = 24.0, limit: int = 50
) -> dict[str, Any]:
    """Classify recent failures and summarise them by cause, project and agent."""
    service.ensure_ready()
    telemetry = open_telemetry(service.config.state_dir)

    jobs = service.db.list_jobs()
    cutoff = hours * 3600.0
    failures: list[Failure] = []
    considered = 0

    for job in jobs:
        if not job.is_terminal:
            continue
        age = age_seconds(job.finished_at or job.updated_at)
        if age is not None and age > cutoff:
            continue
        considered += 1
        if job.state not in (
            JobState.FAILED.value,
            JobState.LOST.value,
            JobState.CANCELLED.value,
        ):
            continue

        text = _log_tail(service, job)
        cause = classify_failure(service, job, text)

        state = telemetry.sample_near(job.finished_at) if job.finished_at else None
        peak = (
            telemetry.peak_between(job.started_at, job.finished_at)
            if job.started_at and job.finished_at
            else None
        )
        foreign: list[dict[str, Any]] = []
        if state and state.get("top_consumers_json"):
            import json

            try:
                foreign = json.loads(state["top_consumers_json"])[:5]
            except Exception:
                foreign = []

        failures.append(
            Failure(
                job=job,
                cause=cause,
                excerpt=_excerpt(text, cause),
                resource_state=state,
                peak=peak,
                foreign=foreign,
            )
        )
        if len(failures) >= limit:
            break

    telemetry.close()

    by_cause = Counter(f.cause.label for f in failures)
    by_project = Counter(f.job.project for f in failures)
    by_agent = Counter(f.job.submitter_agent or "unknown" for f in failures)
    resource_caused = [f for f in failures if f.cause.resource]

    return {
        "window_hours": hours,
        "jobs_considered": considered,
        "failures": [f.to_dict() for f in failures],
        "counts": {
            "total_failures": len(failures),
            "resource_caused": len(resource_caused),
            "by_cause": dict(by_cause.most_common()),
            "by_project": dict(by_project.most_common()),
            "by_agent": dict(by_agent.most_common()),
        },
        "throughput": service.throughput(hours=hours),
        "verdict": _verdict(failures, resource_caused),
    }


def _verdict(failures: list[Failure], resource_caused: list[Failure]) -> str:
    if not failures:
        return "No failures in this window."
    if not resource_caused:
        return (
            f"{len(failures)} failure(s), none caused by resource exhaustion - "
            "these look like problems in the jobs themselves."
        )
    projects = Counter(f.job.project for f in resource_caused)
    worst = projects.most_common(1)[0]
    return (
        f"{len(resource_caused)} of {len(failures)} failure(s) were caused by the machine "
        f"running out of memory, worst in '{worst[0]}' ({worst[1]}). "
        "Declare --ram/--vram on those jobs so worker-q can hold them until it is safe, "
        "and make sure every heavy workload is submitted rather than run directly."
    )


def foreign_pressure_report(service: GPUQService, *, hours: float = 6.0) -> dict[str, Any]:
    """Processes holding memory that worker-q did not start.

    This is the direct answer to "who is crashing my box": work that bypassed
    the queue is invisible to slot counting but very visible in the samples.
    """
    telemetry = open_telemetry(service.config.state_dir)
    from workerq.util import utcnow, utcnow_iso
    from datetime import timedelta

    start = (utcnow() - timedelta(hours=hours)).isoformat(timespec="microseconds")
    samples = telemetry.samples_between(start, utcnow_iso())
    telemetry.close()

    import json

    totals: dict[str, float] = {}
    peaks: dict[str, float] = {}
    for sample in samples:
        raw = sample.get("top_consumers_json")
        if not raw:
            continue
        try:
            consumers = json.loads(raw)
        except Exception:
            continue
        for entry in consumers:
            name = str(entry.get("name") or "?")
            mib = float(entry.get("mib") or 0.0)
            totals[name] = totals.get(name, 0.0) + mib
            peaks[name] = max(peaks.get(name, 0.0), mib)

    ranked = sorted(peaks.items(), key=lambda kv: kv[1], reverse=True)[:12]
    worst_free = min(
        (s["host_free_percent"] for s in samples if s.get("host_free_percent") is not None),
        default=None,
    )
    worst_commit = max(
        (s["commit_percent"] for s in samples if s.get("commit_percent") is not None),
        default=None,
    )
    return {
        "window_hours": hours,
        "samples": len(samples),
        "worst_host_free_percent": worst_free,
        "worst_commit_percent": worst_commit,
        "peak_consumers": [
            {"name": name, "peak_gib": round(mib / 1024, 2)} for name, mib in ranked
        ],
    }


def declared_vs_observed(service: GPUQService, *, limit: int = 200) -> dict[str, Any]:
    """How close each job's declared footprint was to what it actually used.

    Admission control hands out capacity against declarations, so a queue full
    of jobs that ask for four times what they need packs four times worse than
    it could. This is the evidence for correcting them - and for trusting the
    ledger enough to run jobs in parallel at all.

    Jobs sampled before observed-usage recording existed are skipped rather
    than counted as zero.
    """
    rows: list[dict[str, Any]] = []
    for job in service.db.list_jobs(limit=limit):
        if not job.usage_samples:
            continue
        declared_ram = job.requested_ram_mib
        declared_vram = job.requested_vram_mib
        rows.append(
            {
                "id": job.id,
                "project": job.project,
                "command_signature": job.command_signature,
                "declared_ram_mib": declared_ram,
                "peak_ram_mib": job.peak_ram_mib,
                "ram_ratio": (
                    job.peak_ram_mib / declared_ram
                    if declared_ram and job.peak_ram_mib is not None
                    else None
                ),
                "declared_vram_mib": declared_vram,
                "peak_vram_mib": job.peak_vram_mib,
                "vram_ratio": (
                    job.peak_vram_mib / declared_vram
                    if declared_vram and job.peak_vram_mib is not None
                    else None
                ),
                "samples": job.usage_samples,
            }
        )

    ratios = [r["ram_ratio"] for r in rows if r["ram_ratio"] is not None]
    ratios.sort()
    median = ratios[len(ratios) // 2] if ratios else None
    # Reclaimable headroom is what the ledger is holding back for RAM that the
    # jobs demonstrably never touched.
    waste = [
        r["declared_ram_mib"] - r["peak_ram_mib"]
        for r in rows
        if r["declared_ram_mib"] and r["peak_ram_mib"] is not None
    ]
    return {
        "jobs": rows,
        "measured": len(rows),
        "median_ram_ratio": median,
        "mean_overdeclared_ram_mib": (sum(waste) / len(waste)) if waste else None,
        "vram_measurable": any(r["peak_vram_mib"] is not None for r in rows),
    }
