"""Execution backend abstraction (spec section 7).

worker-q never constructs backend-specific commands outside a backend module. The
V1 implementation is `LocalDispatcherBackend`; `RemoteBackend` / `SlurmBackend`
can be added without touching core, CLI or MCP code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Backend could not carry out an operation."""


class BackendUnavailable(BackendError):
    """The backend is not reachable.

    Spec section 4.4: submission must fail loudly rather than run the command
    directly. Callers translate this into a non-zero exit with doctor advice.
    """


# Backend-level states. The mapping to worker-q `JobState` lives in one place only:
# `BACKEND_STATE_MAP` below.
BACKEND_QUEUED = "QUEUED"
BACKEND_RUNNING = "RUNNING"
BACKEND_FINISHED = "FINISHED"
BACKEND_REMOVED = "REMOVED"
BACKEND_MISSING = "MISSING"


@dataclass(frozen=True)
class BackendJob:
    backend_id: int
    state: str
    label: str | None = None
    output_path: Path | None = None
    pid: int | None = None
    exit_code: int | None = None
    enqueued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    wait_reason: str | None = None
    gpu_count: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SchedulerBackend(Protocol):
    """Everything worker-q needs from an execution backend."""

    name: str

    def health(self) -> dict: ...

    def initialize(self) -> None: ...

    def submit(
        self,
        argv: Sequence[str],
        *,
        label: str,
        gpu_count: int,
        slots: int = 1,
        log_name: str | None = None,
        priority_rank: int = 100,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        ram_mib: float | None = None,
        vram_mib: float | None = None,
        cpus: int | None = None,
        preemptible: bool = False,
        gpu_mode: str = "exclusive",
    ) -> int: ...

    def list_jobs(self) -> list[BackendJob]: ...

    def get_job(self, backend_id: int) -> BackendJob: ...

    def get_state(self, backend_id: int) -> str: ...

    def output_path(self, backend_id: int) -> Path | None: ...

    def remove_queued(self, backend_id: int) -> None: ...

    def terminate_running(self, backend_id: int, *, force: bool = False) -> None: ...

    def set_priority(self, backend_id: int, priority_rank: int) -> None: ...

    def promote(self, backend_id: int) -> None: ...

    def set_slots(self, count: int) -> None: ...

    def get_slots(self) -> int: ...

    def find_by_label(self, label: str) -> BackendJob | None: ...

    def shutdown(self) -> None: ...
