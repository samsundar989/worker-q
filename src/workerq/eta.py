"""How long a job will take, and when a queued one will start.

Three sources, and the honest ranking between them matters more than the
arithmetic:

* **progress** - the job itself reported a completion fraction. Only this knows
  that epoch 3 of 120 is slower than epoch 90.
* **declared** - a human or agent said `--eta 90m` at submit, or corrected it
  at runtime once the job knew better.
* **learned** - the median wall time of previous runs of the same command in the
  same project. Useful, but it is a guess from a small sample.

Every estimate carries the source and the sample size, and "unknown" is a
first-class answer. A queue view that prints a confident wrong finish time is
worse than one that admits it cannot say.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from workerq.models import Job, JobState
from workerq.util import age_seconds, parse_iso, utcnow

if TYPE_CHECKING:  # pragma: no cover
    from workerq.core import GPUQService

#: A learned estimate needs at least this many past runs to be worth showing.
MIN_SAMPLES = 2
#: Past runs older than this are ignored: a command's cost changes as the code does.
LEARNED_WINDOW_DAYS = 30

SOURCE_PROGRESS = "progress"
SOURCE_DECLARED = "declared"
SOURCE_LEARNED = "learned"
SOURCE_UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Command signature
# --------------------------------------------------------------------------

_VALUE_LIKE = re.compile(r"^[-+]?\d[\d.,:_-]*$")


def command_signature(command: list[str], shell_mode: bool = False) -> str:
    """A stable identity for "the same kind of command".

    Two runs of `train --fold A --epochs 120` and `train --fold B --epochs 120`
    should share a signature, because they cost about the same. So flag *values*
    and anything path- or number-shaped are dropped, and what remains is the
    shape of the command: the program, its subcommands, and which options were
    given.
    """
    if not command:
        return ""

    words = (
        [w for w in re.split(r"\s+", command[0].strip()) if w]
        if shell_mode
        else [str(c) for c in command]
    )

    tokens: list[str] = []
    expecting_value = False
    for index, token in enumerate(words):
        if index == 0:
            # Interpreters live at machine-specific absolute paths, and the same
            # interpreter is `python` on one OS and `python.exe` on another.
            tokens.append(_program_name(token))
            expecting_value = False
            continue

        if token.startswith("-"):
            name, _, inline = token.partition("=")
            tokens.append(name)
            # `--flag=value` carries its value; a bare `--flag` may take the
            # next token, which is a value and must not enter the signature.
            expecting_value = not inline
            continue

        if expecting_value:
            expecting_value = False
            continue  # this is a flag's value: fold, seed, path, epoch count

        if _looks_like_value(token):
            continue
        tokens.append(_basename(token))

    digest = hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()
    return digest[:16]


_EXECUTABLE_SUFFIXES = (".exe", ".bat", ".cmd", ".com")


def _program_name(token: str) -> str:
    """Interpreter identity, independent of where it lives or which OS."""
    name = _basename(token).lower()
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # pythonw and python are the same interpreter for costing purposes.
    if name.startswith("pythonw"):
        name = "python" + name[len("pythonw") :]
    return name


def _basename(token: str) -> str:
    cleaned = token.replace("\\", "/").rstrip("/")
    return cleaned.rsplit("/", 1)[-1] if "/" in cleaned else cleaned


def _looks_like_value(token: str) -> bool:
    """True for things that vary between otherwise identical runs."""
    if _VALUE_LIKE.match(token):
        return True
    if "/" in token or "\\" in token:
        return True  # a path: an output directory, a config, a checkpoint
    if len(token) > 40:
        return True  # an inline script or a long literal
    return False


# --------------------------------------------------------------------------
# Estimate
# --------------------------------------------------------------------------


@dataclass
class Estimate:
    """A duration guess, always carrying where it came from."""

    #: Total expected wall time, or None when nothing can be said.
    total_seconds: float | None
    #: Expected time still to run. Equals total for a job that has not started.
    remaining_seconds: float | None
    source: str
    samples: int = 0

    @property
    def known(self) -> bool:
        return self.remaining_seconds is not None

    def label(self) -> str:
        """Short provenance tag, so nobody mistakes a guess for a measurement."""
        if self.source == SOURCE_LEARNED:
            return f"learned n={self.samples}"
        return self.source

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "remaining_seconds": self.remaining_seconds,
            "source": self.source,
            "samples": self.samples,
        }


UNKNOWN = Estimate(None, None, SOURCE_UNKNOWN)


def learned_duration(
    service: GPUQService, job: Job
) -> tuple[float | None, int]:
    """Median wall time of past successful runs of this same command shape."""
    signature = job.command_signature or command_signature(
        job.command, bool(job.shell_mode)
    )
    if not signature:
        return None, 0

    cutoff = (utcnow() - timedelta(days=LEARNED_WINDOW_DAYS)).isoformat(
        timespec="microseconds"
    )
    try:
        rows = service.db.conn.execute(
            "SELECT started_at, finished_at FROM jobs "
            "WHERE project = ? AND command_signature = ? AND state = ? "
            "AND id != ? AND started_at IS NOT NULL AND finished_at IS NOT NULL "
            "AND finished_at >= ? ORDER BY id DESC LIMIT 20",
            (job.project, signature, JobState.SUCCEEDED.value, job.id, cutoff),
        ).fetchall()
    except Exception:
        return None, 0

    durations: list[float] = []
    for row in rows:
        start, end = parse_iso(row["started_at"]), parse_iso(row["finished_at"])
        if start and end:
            seconds = (end - start).total_seconds()
            if seconds > 0:
                durations.append(seconds)

    if len(durations) < MIN_SAMPLES:
        return None, len(durations)

    durations.sort()
    middle = len(durations) // 2
    median = (
        durations[middle]
        if len(durations) % 2
        else (durations[middle - 1] + durations[middle]) / 2.0
    )
    return median, len(durations)


def estimate_job(service: GPUQService, job: Job) -> Estimate:
    """Best available estimate for one job, preferring the most informed source."""
    if job.is_terminal:
        return Estimate(job.runtime_seconds, 0.0, SOURCE_PROGRESS)

    elapsed = job.runtime_seconds or 0.0
    running = job.state == JobState.RUNNING.value

    # 1. The job's own progress report. Only trustworthy once it has actually
    #    made some, and it goes stale if the job stops reporting.
    fraction = job.progress_fraction
    if running and fraction is not None and 0.02 <= fraction < 1.0 and elapsed > 0:
        total = elapsed / fraction
        return Estimate(total, max(0.0, total - elapsed), SOURCE_PROGRESS)

    # 2. What the worker declared.
    if job.eta_seconds:
        total = float(job.eta_seconds)
        return Estimate(
            total, max(0.0, total - elapsed) if running else total, SOURCE_DECLARED
        )

    # 3. What this command has historically cost.
    median, samples = learned_duration(service, job)
    if median is not None:
        return Estimate(
            median,
            max(0.0, median - elapsed) if running else median,
            SOURCE_LEARNED,
            samples,
        )

    return UNKNOWN


def forecast_queue(service: GPUQService, jobs: list[Job]) -> dict[int, dict[str, Any]]:
    """Estimated start and finish for everything not yet finished.

    Queued jobs are laid out over the configured slots in dispatch order. This
    is a projection, not a promise: it inherits the uncertainty of every
    estimate ahead of it, and a job with an unknown duration makes everything
    behind it unknown too rather than silently optimistic.
    """
    slots = max(1, service.config.core.max_concurrent_jobs)
    now = utcnow()
    result: dict[int, dict[str, Any]] = {}

    # Slots become free as running jobs finish. An unknown finish time is
    # represented as None and poisons the slot it occupies.
    free_at: list[float | None] = []
    for job in jobs:
        if job.state != JobState.RUNNING.value:
            continue
        est = estimate_job(service, job)
        result[job.id] = {
            "estimate": est.to_dict(),
            "eta_source": est.label(),
            "remaining_seconds": est.remaining_seconds,
            "finish_at": (
                (now + timedelta(seconds=est.remaining_seconds)).isoformat(
                    timespec="seconds"
                )
                if est.remaining_seconds is not None
                else None
            ),
            "start_at": job.started_at,
        }
        free_at.append(est.remaining_seconds)

    while len(free_at) < slots:
        free_at.append(0.0)  # an idle slot is free immediately

    for job in jobs:
        if job.state not in (JobState.QUEUED.value, JobState.PREPARING.value):
            continue
        est = estimate_job(service, job)

        # Take the slot that frees soonest; an unknown slot is worst.
        known = [(index, value) for index, value in enumerate(free_at) if value is not None]
        if known:
            index, starts_in = min(known, key=lambda pair: pair[1])
        else:
            index, starts_in = 0, None

        finish_in = (
            None
            if starts_in is None or est.total_seconds is None
            else starts_in + est.total_seconds
        )
        result[job.id] = {
            "estimate": est.to_dict(),
            "eta_source": est.label(),
            "remaining_seconds": est.total_seconds,
            "starts_in_seconds": starts_in,
            "start_at": (
                (now + timedelta(seconds=starts_in)).isoformat(timespec="seconds")
                if starts_in is not None
                else None
            ),
            "finish_at": (
                (now + timedelta(seconds=finish_in)).isoformat(timespec="seconds")
                if finish_in is not None
                else None
            ),
        }
        free_at[index] = finish_in

    return result


def read_progress(path: str) -> tuple[float | None, str | None]:
    """Read a job's self-reported progress file.

    Accepts either a bare fraction (`0.42`), a percentage (`42%`), or JSON
    (`{"frac": 0.42, "note": "epoch 50/120"}`), because a job should be able to
    report progress with one line of shell or one line of Python.
    """
    import json
    import os

    try:
        if not os.path.exists(path):
            return None, None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read().strip()
    except OSError:
        return None, None
    if not raw:
        return None, None

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
        value = data.get("frac", data.get("fraction", data.get("progress")))
        note = data.get("note") or data.get("message")
        fraction = _coerce_fraction(value)
        return fraction, (str(note)[:200] if note else None)

    return _coerce_fraction(raw), None


def _coerce_fraction(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if percent or number > 1.0:
        number = number / 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return number
