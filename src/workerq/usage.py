"""Declared footprint against what a job actually used.

The subtlety this module exists for: **`peak_ram_mib` is not RAM.**

`host._windows_processes` records `PrivateUsage`, which is *commit charge* for
the whole process tree, and it does so deliberately - working set reports a
training job as tiny while it is the largest thing on the machine. But `--ram`
is admitted against free *physical* memory. The two numbers measure different
things, and under WDDM they diverge hard: the driver backs video allocations
with system commit, so a GPU job's commit is roughly its RAM plus its VRAM.
`resources.admit` already knows this and gates on it.

Compare a GPU job's peak commit against its declared RAM alone and it looks
wildly under-declared when it is nothing of the sort. Measured on this
machine's own history: of 21 jobs whose peak exceeded declared RAM, 18 had
declared VRAM, and 11 of those sat comfortably inside declared RAM + VRAM.
Advising those to raise their RAM declaration would park tens of GiB that
nothing needs; advising the reverse would take the box down.

So every comparison here is against the *commit budget* - declared RAM plus
declared VRAM for a job that asked for a GPU, declared RAM alone otherwise -
and the measured column is named peak commit everywhere it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: Below this many samples a peak is whatever the tree happened to hold at a
#: couple of arbitrary instants. The runner samples fast for the first half
#: minute (see `runner._USAGE_POLL_SECONDS`) so most jobs clear this easily;
#: the ones that do not are short, and a short job's peak is the least
#: interesting number in the table anyway.
MIN_CONFIDENT_SAMPLES = 3

#: A job using less than half its budget is holding capacity nobody can use.
#: Not an error - headroom is deliberate - but it is what the queue pays for.
OVER_DECLARED_RATIO = 0.5

#: And below this it is not headroom, it is a guess nobody revisited.
SEVERELY_OVER_DECLARED_RATIO = 0.25

VERDICT_OK = "ok"
VERDICT_OVER = "over-declared"
VERDICT_SEVERELY_OVER = "severely-over-declared"
VERDICT_UNDER = "under-declared"
VERDICT_UNMEASURED = "unmeasured"
VERDICT_LOW_CONFIDENCE = "low-confidence"

#: Verdicts that should never drive advice to change a declaration.
INCONCLUSIVE = frozenset({VERDICT_UNMEASURED, VERDICT_LOW_CONFIDENCE})


@dataclass
class Usage:
    """One job's declaration measured against what it actually held."""

    job_id: int
    project: str
    node: str
    state: str
    verdict: str

    declared_ram_mib: float | None = None
    declared_vram_mib: float | None = None
    #: Declared RAM + declared VRAM when the job asked for a GPU. This is the
    #: quantity `peak_commit_mib` is actually comparable against.
    commit_budget_mib: float | None = None
    #: Stored as `jobs.peak_ram_mib`; it is commit charge, not resident RAM.
    peak_commit_mib: float | None = None
    peak_vram_mib: float | None = None

    commit_ratio: float | None = None
    vram_ratio: float | None = None
    #: Budget the job never touched. Negative means it went over.
    unused_mib: float | None = None
    #: `unused_mib` weighted by how long the job held it - the honest unit for
    #: "what did this cost everyone else", since a huge over-declaration for
    #: ten seconds costs the queue nothing.
    unused_gib_hours: float | None = None

    samples: int = 0
    peak_source: str | None = None
    vram_source: str | None = None
    #: True when the VRAM figure came from watching the whole card rather than
    #: the process. It is the only attribution WDDM allows, and the display has
    #: to say so rather than implying a per-process reading.
    vram_is_device_delta: bool = False

    runtime_seconds: float | None = None
    wait_seconds: float | None = None
    eta_seconds: float | None = None
    #: Actual runtime over declared ETA. Under 1.0 means the ETA was
    #: pessimistic, which is the common case and still costs the queue nothing;
    #: over 1.0 means everything downstream was told the wrong finish time.
    eta_ratio: float | None = None

    suggested_ram_gb: float | None = None
    suggested_vram_gb: float | None = None
    description: str | None = None
    command_signature: str | None = None

    @property
    def conclusive(self) -> bool:
        return self.verdict not in INCONCLUSIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project": self.project,
            "node": self.node,
            "state": self.state,
            "verdict": self.verdict,
            "declared_ram_mib": self.declared_ram_mib,
            "declared_vram_mib": self.declared_vram_mib,
            "commit_budget_mib": self.commit_budget_mib,
            "peak_commit_mib": self.peak_commit_mib,
            "peak_vram_mib": self.peak_vram_mib,
            "commit_ratio": self.commit_ratio,
            "vram_ratio": self.vram_ratio,
            "unused_mib": self.unused_mib,
            "unused_gib_hours": self.unused_gib_hours,
            "samples": self.samples,
            "peak_source": self.peak_source,
            "vram_source": self.vram_source,
            "vram_is_device_delta": self.vram_is_device_delta,
            "runtime_seconds": self.runtime_seconds,
            "wait_seconds": self.wait_seconds,
            "eta_seconds": self.eta_seconds,
            "eta_ratio": self.eta_ratio,
            "suggested_ram_gb": self.suggested_ram_gb,
            "suggested_vram_gb": self.suggested_vram_gb,
            "description": self.description,
            "command_signature": self.command_signature,
        }


def node_label(node: str | None) -> str:
    """`jobs.node` is NULL for local jobs, and must stay that way.

    `eta._durations_for` matches `node IS NULL` to find same-machine history,
    so rewriting the column would quietly break duration learning. The name is
    a read-layer concern only.
    """
    return node or "local"


def commit_budget_mib(
    declared_ram_mib: float | None, declared_vram_mib: float | None
) -> float | None:
    """What the job's measured commit peak is fairly compared against.

    A job that declared no VRAM is judged on its RAM declaration alone; adding
    a zero would be the same number but would imply a GPU claim it never made.
    """
    if declared_ram_mib is None:
        return None
    if declared_vram_mib:
        return float(declared_ram_mib) + float(declared_vram_mib)
    return float(declared_ram_mib)


def classify(
    *,
    peak_commit_mib: float | None,
    budget_mib: float | None,
    samples: int,
) -> str:
    """The verdict for one job, in the order the cases actually matter."""
    if peak_commit_mib is None or not budget_mib:
        return VERDICT_UNMEASURED
    if samples < MIN_CONFIDENT_SAMPLES:
        # Say nothing rather than something wrong. A two-sample peak has been
        # seen both to miss the real peak entirely and to land on a transient.
        return VERDICT_LOW_CONFIDENCE
    ratio = peak_commit_mib / budget_mib
    if ratio > 1.0:
        return VERDICT_UNDER
    if ratio < SEVERELY_OVER_DECLARED_RATIO:
        return VERDICT_SEVERELY_OVER
    if ratio < OVER_DECLARED_RATIO:
        return VERDICT_OVER
    return VERDICT_OK


def for_job(job: Any, *, ram_ceiling: float | None = None,
            vram_ceiling: float | None = None) -> Usage:
    """Measure one `Job` against its own declaration."""
    from workerq.eta import suggested_ram_gb, suggested_vram_gb

    declared_ram = job.requested_ram_mib
    declared_vram = job.requested_vram_mib
    budget = commit_budget_mib(declared_ram, declared_vram)
    peak = job.peak_ram_mib
    samples = int(job.usage_samples or 0)
    verdict = classify(peak_commit_mib=peak, budget_mib=budget, samples=samples)

    runtime = job.runtime_seconds
    unused = (budget - peak) if (budget and peak is not None) else None
    unused_gib_hours = None
    if unused is not None and runtime:
        unused_gib_hours = (unused / 1024.0) * (runtime / 3600.0)

    eta_ratio = None
    if job.eta_seconds and runtime:
        eta_ratio = runtime / job.eta_seconds

    return Usage(
        job_id=job.id,
        project=job.project,
        node=node_label(job.node),
        state=job.state,
        verdict=verdict,
        declared_ram_mib=declared_ram,
        declared_vram_mib=declared_vram,
        commit_budget_mib=budget,
        peak_commit_mib=peak,
        peak_vram_mib=job.peak_vram_mib,
        commit_ratio=(peak / budget) if (budget and peak is not None) else None,
        vram_ratio=(
            job.peak_vram_mib / declared_vram
            if declared_vram and job.peak_vram_mib is not None
            else None
        ),
        unused_mib=unused,
        unused_gib_hours=unused_gib_hours,
        samples=samples,
        peak_source=job.peak_source,
        vram_source=job.vram_source,
        vram_is_device_delta=job.vram_source == "device_delta",
        runtime_seconds=runtime,
        wait_seconds=job.wait_seconds,
        eta_seconds=job.eta_seconds,
        eta_ratio=eta_ratio,
        suggested_ram_gb=(
            suggested_ram_gb(peak, ram_ceiling) if peak is not None else None
        ),
        suggested_vram_gb=(
            suggested_vram_gb(job.peak_vram_mib, vram_ceiling)
            if job.peak_vram_mib is not None
            else None
        ),
        description=job.description,
        command_signature=job.command_signature,
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class Summary:
    """What a set of jobs says about how well this machine is estimated."""

    total: int = 0
    conclusive: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    median_commit_ratio: float | None = None
    median_eta_ratio: float | None = None
    #: The headline cost. Budget held and never touched, weighted by how long
    #: it was held - this is queue time other people spent waiting for nothing.
    unused_gib_hours: float = 0.0
    worst_over: list[Usage] = field(default_factory=list)
    worst_under: list[Usage] = field(default_factory=list)
    vram_from_device_delta: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "conclusive": self.conclusive,
            "counts": self.counts,
            "median_commit_ratio": self.median_commit_ratio,
            "median_eta_ratio": self.median_eta_ratio,
            "unused_gib_hours": self.unused_gib_hours,
            "worst_over": [u.to_dict() for u in self.worst_over],
            "worst_under": [u.to_dict() for u in self.worst_under],
            "vram_from_device_delta": self.vram_from_device_delta,
        }


def summarise(rows: Iterable[Usage], *, worst: int = 10) -> Summary:
    """Aggregate, keeping over- and under-declaration on separate rankings.

    They are not two ends of one axis. Over-declaring makes other people wait;
    under-declaring takes the machine down. Sorting them into one list buries
    the second kind under the first, and the second kind is the one that has
    to be read.
    """
    rows = list(rows)
    summary = Summary(total=len(rows))
    counts: dict[str, int] = {}
    ratios: list[float] = []
    eta_ratios: list[float] = []
    any_delta = False
    any_measured_vram = False

    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
        if row.eta_ratio is not None:
            eta_ratios.append(row.eta_ratio)
        if row.vram_source == "device_delta":
            any_delta = True
        elif row.vram_source == "measured":
            any_measured_vram = True
        if not row.conclusive:
            continue
        summary.conclusive += 1
        if row.commit_ratio is not None:
            ratios.append(row.commit_ratio)
        if row.unused_gib_hours and row.unused_gib_hours > 0:
            summary.unused_gib_hours += row.unused_gib_hours

    over = [r for r in rows if r.verdict in (VERDICT_OVER, VERDICT_SEVERELY_OVER)]
    under = [r for r in rows if r.verdict == VERDICT_UNDER]
    over.sort(key=lambda r: r.unused_gib_hours or 0.0, reverse=True)
    under.sort(key=lambda r: r.commit_ratio or 0.0, reverse=True)

    summary.counts = counts
    summary.median_commit_ratio = _median(ratios)
    summary.median_eta_ratio = _median(eta_ratios)
    summary.worst_over = over[:worst]
    summary.worst_under = under[:worst]
    summary.vram_from_device_delta = any_delta and not any_measured_vram
    return summary


def group_by(rows: Iterable[Usage], key: str) -> list[dict[str, Any]]:
    """Per-project or per-signature calibration.

    A signature is what makes this actionable: one job being wrong is a typo,
    the same command being wrong twenty times running is a default worth
    changing at the source.
    """
    buckets: dict[str, list[Usage]] = {}
    for row in rows:
        value = getattr(row, key, None) or "(none)"
        buckets.setdefault(str(value), []).append(row)
    out = []
    for name, group in buckets.items():
        summary = summarise(group, worst=3)
        # A signature is a hash, which is unreadable as a row label. Carry the
        # most recent description alongside it so the grouping can be acted on
        # without opening a job to find out what the command was.
        described = next((g.description for g in group if g.description), None)
        out.append(
            {
                "key": name,
                "label": described or name,
                "projects": sorted({g.project for g in group}),
                "runs": len(group),
                **summary.to_dict(),
            }
        )
    out.sort(key=lambda g: g["unused_gib_hours"], reverse=True)
    return out
