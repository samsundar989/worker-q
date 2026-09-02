"""Core domain models: job states, priorities and the Job record."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from workerq.util import age_seconds, display_command


class JobState(str, Enum):
    """Normalized internal job states (spec section 9.2)."""

    PREPARING = "PREPARING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    LOST = "LOST"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return self in ACTIVE_STATES


TERMINAL_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.LOST}
)
ACTIVE_STATES = frozenset({JobState.PREPARING, JobState.QUEUED, JobState.RUNNING})

#: Allowed transitions. A terminal state never returns to an active one
#: (spec section 23: "terminal state cannot accidentally revert to QUEUED").
ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PREPARING: frozenset(
        {JobState.QUEUED, JobState.FAILED, JobState.CANCELLED, JobState.LOST}
    ),
    JobState.QUEUED: frozenset(
        {
            JobState.RUNNING,
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.LOST,
        }
    ),
    # RUNNING -> QUEUED is legal *only* for preemption: a higher-priority job
    # displaced this one, so it goes back to the queue rather than failing. It
    # is the one backwards edge in this machine; everything else still moves
    # forward only, and a terminal state is still permanent.
    JobState.RUNNING: frozenset(
        {
            JobState.QUEUED,
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.LOST,
        }
    ),
    JobState.SUCCEEDED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
    JobState.LOST: frozenset(),
}


def can_transition(current: JobState, target: JobState) -> bool:
    if current == target:
        return True
    return target in ALLOWED_TRANSITIONS[current]


class InvalidTransition(RuntimeError):
    def __init__(self, current: JobState, target: JobState) -> None:
        super().__init__(f"illegal job state transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Lower rank dispatches first."""
        return PRIORITY_RANKS[self]


PRIORITY_RANKS: dict[Priority, int] = {
    Priority.CRITICAL: 0,
    Priority.HIGH: 50,
    Priority.NORMAL: 100,
    Priority.LOW: 200,
}

PRIORITY_ORDER = (Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW)


def priority_rank(value: str | Priority) -> int:
    return PRIORITY_RANKS[Priority(value)]


class SnapshotMode(str, Enum):
    GIT = "git"
    COPY = "copy"
    NONE = "none"
    LIVE = "live"


@dataclass
class Job:
    """One worker-q job row."""

    id: int
    backend: str
    backend_job_id: int | None

    project: str
    label: str | None
    priority: str

    repo_root: str | None
    submitted_cwd: str
    execution_cwd: str | None

    command_json: str
    shell_mode: int

    requested_gpu_count: int
    gpu_mode: str

    snapshot_mode: str
    snapshot_commit: str | None
    snapshot_path: str | None

    host: str
    submitter_pid: int | None
    submitter_agent: str | None

    state: str
    exit_code: int | None
    runner_pid: int | None

    queued_at: str
    started_at: str | None
    finished_at: str | None

    error: str | None
    created_at: str
    updated_at: str

    # Derived / late-bound fields.
    log_path: str | None = None
    cuda_visible_devices: str | None = None
    requested_ram_mib: float | None = None
    requested_vram_mib: float | None = None
    requested_cpus: int | None = None
    preemptible: int = 0
    preemption_count: int = 0
    preempted_at: str | None = None
    preempted_by: int | None = None
    preempted_reason: str | None = None
    description: str | None = None
    blocks: str | None = None
    eta_seconds: float | None = None
    command_signature: str | None = None
    progress_fraction: float | None = None
    progress_note: str | None = None
    progress_updated_at: str | None = None
    passthrough_json: str | None = None
    env_json: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def command(self) -> list[str]:
        try:
            value = json.loads(self.command_json)
        except (TypeError, json.JSONDecodeError):
            return []
        return [str(x) for x in value] if isinstance(value, list) else []

    @property
    def env(self) -> dict[str, str]:
        if not self.env_json:
            return {}
        try:
            value = json.loads(self.env_json)
        except json.JSONDecodeError:
            return {}
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}

    @property
    def passthrough(self) -> list[str]:
        if not self.passthrough_json:
            return []
        try:
            value = json.loads(self.passthrough_json)
        except json.JSONDecodeError:
            return []
        return [str(x) for x in value] if isinstance(value, list) else []

    @property
    def state_enum(self) -> JobState:
        return JobState(self.state)

    @property
    def is_terminal(self) -> bool:
        return self.state_enum.is_terminal

    @property
    def display_command(self) -> str:
        return display_command(self.command, bool(self.shell_mode))

    @property
    def runtime_seconds(self) -> float | None:
        """Wall time running, or waiting time when still queued."""
        if self.started_at:
            return age_seconds(self.started_at, until=self.finished_at)
        return None

    @property
    def wait_seconds(self) -> float | None:
        if self.started_at:
            return age_seconds(self.queued_at, until=self.started_at)
        return age_seconds(self.queued_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend,
            "backend_job_id": self.backend_job_id,
            "project": self.project,
            "label": self.label,
            "priority": self.priority,
            "repo_root": self.repo_root,
            "submitted_cwd": self.submitted_cwd,
            "execution_cwd": self.execution_cwd,
            "command": self.command,
            "shell_mode": bool(self.shell_mode),
            "requested_gpu_count": self.requested_gpu_count,
            "requested_ram_mib": self.requested_ram_mib,
            "requested_vram_mib": self.requested_vram_mib,
            "requested_cpus": self.requested_cpus,
            "preemptible": bool(self.preemptible),
            "preemption_count": self.preemption_count,
            "preempted_at": self.preempted_at,
            "preempted_by": self.preempted_by,
            "preempted_reason": self.preempted_reason,
            "description": self.description,
            "blocks": self.blocks,
            "eta_seconds": self.eta_seconds,
            "command_signature": self.command_signature,
            "progress_fraction": self.progress_fraction,
            "progress_note": self.progress_note,
            "progress_updated_at": self.progress_updated_at,
            "gpu_mode": self.gpu_mode,
            "snapshot_mode": self.snapshot_mode,
            "snapshot_commit": self.snapshot_commit,
            "snapshot_path": self.snapshot_path,
            "snapshot_passthrough": self.passthrough,
            "host": self.host,
            "submitter_pid": self.submitter_pid,
            "submitter_agent": self.submitter_agent,
            "state": self.state,
            "exit_code": self.exit_code,
            "runner_pid": self.runner_pid,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "log_path": self.log_path,
            "cuda_visible_devices": self.cuda_visible_devices,
            "env": self.env,
            "runtime_seconds": self.runtime_seconds,
            "wait_seconds": self.wait_seconds,
        }
