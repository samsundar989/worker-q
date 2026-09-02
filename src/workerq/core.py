"""`GPUQService` - the one place job business logic lives.

Both the CLI and the MCP adapter call these methods; neither reimplements
scheduling, snapshotting or state handling (spec sections 3 and 20).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workerq import BACKEND_NAME, __version__
from workerq import host
from workerq.backends.base import (
    BACKEND_FINISHED,
    BACKEND_MISSING,
    BACKEND_QUEUED,
    BACKEND_REMOVED,
    BACKEND_RUNNING,
    BackendJob,
    BackendUnavailable,
)
from workerq.backends.local_dispatcher import LocalDispatcherBackend, build_backend
from workerq.config import Config, load_config
from workerq.db import Database, json_dumps
from workerq.eta import command_signature
from workerq.gpu import GpuInfo, query_gpus
from workerq.models import (
    ACTIVE_STATES,
    Job,
    JobState,
    Priority,
    SnapshotMode,
    priority_rank,
)
from workerq.snapshot import (
    Snapshot,
    SnapshotError,
    create_copy_snapshot,
    create_git_snapshot,
    find_repo_root,
    load_project_defaults,
    load_project_passthrough,
    remove_snapshot,
)
from workerq.util import (
    atomic_write_text,
    ensure_dir,
    expand_path,
    hostname,
    parse_env_assignment,
    resolve_path,
    utcnow_iso,
)


class GPUQError(RuntimeError):
    """User-facing error. The CLI prints the message and exits non-zero."""


class JobNotFound(GPUQError):
    pass


#: Backend state -> worker-q state. The single mapping point (spec section 9.2).
def map_backend_state(backend_state: str, exit_code: int | None) -> JobState | None:
    if backend_state == BACKEND_QUEUED:
        return JobState.QUEUED
    if backend_state == BACKEND_RUNNING:
        return JobState.RUNNING
    if backend_state == BACKEND_REMOVED:
        return JobState.CANCELLED
    if backend_state == BACKEND_FINISHED:
        if exit_code is None:
            return JobState.FAILED
        return JobState.SUCCEEDED if exit_code == 0 else JobState.FAILED
    if backend_state == BACKEND_MISSING:
        return JobState.LOST
    return None


@dataclass
class SubmitRequest:
    command: list[str]
    project: str | None = None
    #: None means "not specified" - the project's own policy then decides.
    priority: str | None = None
    gpus: int | None = None
    label: str | None = None
    cwd: str | None = None
    snapshot: bool = True
    live_worktree: bool = False
    shell: str | None = None
    env: dict[str, str] | None = None
    passthrough: list[str] | None = None
    #: Declared resource footprint. None means "use the configured default",
    #: so undeclared work is still accounted for rather than treated as free.
    ram_gb: float | None = None
    vram_gb: float | None = None
    cpus: int | None = None
    #: Safe to stop and re-run, so a higher-priority job may displace it.
    preemptible: bool | None = None
    #: Willing to share a GPU with another job that also opted in. Requires an
    #: explicit --vram, since packing is judged on the declaration alone.
    share_gpu: bool = False
    #: What this job is doing, and what is waiting on it. Supplied by the
    #: worker: worker-q cannot infer intent from a command line.
    describe: str | None = None
    blocks: str | None = None
    #: Expected wall time in seconds, if the worker knows it.
    eta_seconds: float | None = None


@dataclass
class SubmitResult:
    job: Job
    snapshot: Snapshot
    backend_job_id: int
    queue_position: int | None = None
    #: Non-fatal warnings about this submission, e.g. a declaration unlikely
    #: ever to be admitted on this machine.
    advisories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job.id,
            "state": self.job.state,
            "project": self.job.project,
            "priority": self.job.priority,
            "backend_job_id": self.backend_job_id,
            "snapshot_commit": self.job.snapshot_commit,
            "snapshot_mode": self.job.snapshot_mode,
            "execution_cwd": self.job.execution_cwd,
            "log_path": self.job.log_path,
            "queue_position": self.queue_position,
            "advisories": self.advisories,
        }


class GPUQService:
    def __init__(
        self,
        config: Config | None = None,
        *,
        backend: LocalDispatcherBackend | None = None,
    ) -> None:
        self.config = config or load_config()
        self.db = Database(self.config.db_path)
        self._backend = backend
        self._db_ready = False

    # -- lifecycle --------------------------------------------------------
    @property
    def backend(self) -> LocalDispatcherBackend:
        if self._backend is None:
            self._backend = build_backend(self.config)
        return self._backend

    def ensure_ready(self) -> None:
        if not self._db_ready:
            self.config.ensure_dirs()
            self.db.initialize()
            self._db_ready = True

    def initialize(self) -> dict[str, Any]:
        """`workerq init`. Idempotent."""
        self.config.ensure_dirs()
        if self.config.source_path and not self.config.source_path.exists():
            self.config.save()
        schema = self.db.initialize()
        self._db_ready = True
        self.backend.initialize()
        return {
            "state_dir": str(self.config.state_dir),
            "config_path": str(self.config.source_path) if self.config.source_path else None,
            "schema_version": schema,
            "backend": self.backend.health(),
        }

    def close(self) -> None:
        self.db.close()
        if self._backend is not None:
            self._backend.close()

    def __enter__(self) -> GPUQService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(self, request: SubmitRequest) -> SubmitResult:
        """Validate, snapshot, enqueue. Never runs the command itself."""
        self.ensure_ready()

        # ---- validate ------------------------------------------------
        shell_mode = request.shell is not None
        if shell_mode:
            command = [request.shell or ""]
            if not command[0].strip():
                raise GPUQError("--shell requires a non-empty command string")
        else:
            command = [str(c) for c in (request.command or [])]
            if not command:
                raise GPUQError(
                    "no command given.\n"
                    "Usage: workerq submit --project NAME -- <command> [args...]"
                )

        if request.priority is not None:
            try:
                Priority(request.priority)
            except ValueError:
                raise GPUQError(
                    f"invalid priority {request.priority!r}; "
                    "choose from critical, high, normal, low"
                ) from None

        gpus = self.config.gpu.default_gpu_count if request.gpus is None else request.gpus
        if gpus < 0:
            raise GPUQError("--gpus must be >= 0")

        for name, value in (("--ram", request.ram_gb), ("--vram", request.vram_gb)):
            if value is not None and value < 0:
                raise GPUQError(f"{name} must be >= 0")
        if request.cpus is not None and request.cpus < 0:
            raise GPUQError("--cpus must be >= 0")

        ram_mib = None if request.ram_gb is None else float(request.ram_gb) * 1024.0
        vram_mib = None if request.vram_gb is None else float(request.vram_gb) * 1024.0
        cpus = None if request.cpus is None else int(request.cpus)

        # Refuse a request the machine could never satisfy, rather than
        # letting it sit QUEUED forever.
        self._reject_impossible(ram_mib, vram_mib, cpus)
        advisories = self._admission_advisories(ram_mib)

        # Sharing a GPU is judged entirely on declared VRAM: there is no
        # per-process VRAM accounting to check it against on consumer cards,
        # and VRAM has no swap to absorb a mistake. So an undeclared job is
        # never packed onto an occupied device.
        if request.share_gpu and gpus > 0 and not vram_mib:
            raise GPUQError(
                "--share-gpu needs --vram: a job may only share a device when it "
                "has said how much VRAM it will use."
            )
        if request.share_gpu:
            gpu_mode = "shared"
        else:
            gpu_mode = "exclusive" if self.config.gpu.exclusive_by_default else "shared"

        submitted_cwd = resolve_path(request.cwd or Path.cwd())
        if not submitted_cwd.is_dir():
            raise GPUQError(f"working directory does not exist: {submitted_cwd}")

        env: dict[str, str] = {}
        for key, value in (request.env or {}).items():
            k, v = parse_env_assignment(f"{key}={value}")
            env[k] = v

        repo_root = find_repo_root(submitted_cwd)
        project = request.project or self._infer_project(repo_root, submitted_cwd)
        priority = self.resolve_priority(project, request.priority, repo_root)
        preemptible = self.resolve_preemptible(request.preemptible, repo_root)

        passthrough = list(request.passthrough or [])
        passthrough += [p for p in load_project_passthrough(repo_root) if p not in passthrough]

        # ---- decide snapshot mode ------------------------------------
        if request.live_worktree or not request.snapshot:
            mode = SnapshotMode.LIVE if request.live_worktree else SnapshotMode.NONE
        elif self.config.core.snapshot_mode == "none":
            mode = SnapshotMode.NONE
        elif repo_root is not None:
            mode = SnapshotMode.GIT
        elif self.config.core.snapshot_mode == "copy":
            mode = SnapshotMode.COPY
        else:
            raise GPUQError(
                f"{submitted_cwd} is not inside a git repository, so worker-q cannot freeze "
                "the source for a queued job.\n"
                "Re-run with --live-worktree to accept running against the live "
                "directory (it may change before the job starts), or run 'git init'."
            )

        # ---- 1. insert PREPARING row ---------------------------------
        now = utcnow_iso()
        job_id = self.db.insert_job(
            backend=BACKEND_NAME,
            backend_job_id=None,
            project=project,
            label=request.label,
            priority=priority.value,
            repo_root=str(repo_root) if repo_root else None,
            submitted_cwd=str(submitted_cwd),
            execution_cwd=None,
            command_json=json_dumps(command),
            shell_mode=1 if shell_mode else 0,
            requested_gpu_count=gpus,
            gpu_mode=gpu_mode,
            snapshot_mode=mode.value,
            host=hostname(),
            submitter_pid=os.getpid(),
            submitter_agent=detect_agent(),
            state=JobState.PREPARING.value,
            queued_at=now,
            env_json=json_dumps(env) if env else None,
            requested_ram_mib=ram_mib,
            requested_vram_mib=vram_mib,
            requested_cpus=cpus,
            preemptible=1 if preemptible else 0,
            description=(request.describe or None),
            blocks=(request.blocks or None),
            eta_seconds=request.eta_seconds,
            command_signature=command_signature(command, shell_mode),
            log_path=str(self.config.log_path(0)),  # placeholder, fixed below
        )

        log_path = self.config.log_path(job_id)
        snapshot: Snapshot | None = None
        try:
            # ---- 2. snapshot -----------------------------------------
            snapshot = self._create_snapshot(
                job_id, mode, repo_root, submitted_cwd, passthrough
            )
            execution_cwd = self._resolve_execution_cwd(
                snapshot, repo_root, submitted_cwd
            )

            self.db.update_job(
                job_id,
                execution_cwd=str(execution_cwd),
                snapshot_mode=snapshot.mode,
                snapshot_commit=snapshot.commit,
                snapshot_path=str(snapshot.path) if snapshot.path else None,
                passthrough_json=json_dumps(snapshot.passthrough) if snapshot.passthrough else None,
                log_path=str(log_path),
            )

            # ---- 3. write the manifest before enqueue ----------------
            self._write_manifest(job_id)

            # ---- 4. enqueue ------------------------------------------
            label = self.backend_label(job_id, project, priority.value)
            backend_job_id = self.backend.submit(
                self.runner_argv(job_id),
                label=label,
                gpu_count=gpus,
                slots=1,
                log_name=self.config.log_name(job_id),
                priority_rank=priority_rank(priority),
                cwd=str(execution_cwd),
                env=None,  # job env is applied by the runner, from the DB
                ram_mib=ram_mib,
                vram_mib=vram_mib,
                cpus=cpus,
                preemptible=preemptible,
                gpu_mode=gpu_mode,
            )

            # ---- 5/6. record backend id, mark QUEUED -----------------
            job = self.db.update_job(
                job_id, backend_job_id=backend_job_id, state=JobState.QUEUED.value
            )

            if priority is Priority.CRITICAL:
                # Spec 11.7: critical goes to the front of the queue. Whether
                # it also displaces running work is the dispatcher's call, and
                # only for jobs that opted in with --preemptible.
                try:
                    self.backend.promote(backend_job_id)
                except BackendUnavailable:
                    pass

            self._write_manifest(job_id)
            return SubmitResult(
                job=job,
                snapshot=snapshot,
                backend_job_id=backend_job_id,
                queue_position=self._queue_position(job_id),
                advisories=advisories,
            )

        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            try:
                self.db.try_update_state(
                    job_id, JobState.FAILED, error=message, finished_at=utcnow_iso()
                )
            except Exception:
                pass
            if snapshot is not None and snapshot.path is not None:
                try:
                    remove_snapshot(
                        snapshot.path,
                        repo_root=snapshot.repo_root,
                        state_root=self.config.state_dir,
                        ref=snapshot.ref,
                    )
                except Exception:
                    pass
            if isinstance(exc, BackendUnavailable):
                raise GPUQError(
                    f"job not submitted: {exc}\n"
                    "worker-q will not run the command directly - the queue is the only "
                    "safe path for heavy GPU work."
                ) from exc
            if isinstance(exc, (SnapshotError, GPUQError)):
                raise GPUQError(f"job not submitted: {exc}") from exc
            raise

    def resolve_priority(
        self,
        project: str,
        explicit: str | None,
        repo_root: Path | None = None,
    ) -> Priority:
        """Decide a job's priority.

        Precedence, most specific first:

        1. `--priority` on the submission
        2. the project's policy (`workerq priority <project> high`) - set once,
           machine-wide, so every worker on that project inherits it without
           editing anything
        3. `[project] priority` in the repo's `.gpuq.toml`
        4. `core.default_priority`
        """
        if explicit:
            return Priority(explicit)

        self.ensure_ready()
        policy = self.db.get_project_priority(project)
        if policy:
            try:
                return Priority(policy)
            except ValueError:
                pass  # a hand-edited row must not break submission

        repo_default = load_project_defaults(repo_root).get("priority")
        if isinstance(repo_default, str):
            try:
                return Priority(repo_default.strip())
            except ValueError:
                pass

        try:
            return Priority(self.config.core.default_priority)
        except ValueError:
            return Priority.NORMAL

    @staticmethod
    def resolve_preemptible(explicit: bool | None, repo_root: Path | None) -> bool:
        """Whether this job may be displaced.

        Explicit `--preemptible/--no-preemptible` wins; otherwise a repository
        may opt in for all its jobs via `[project] preemptible` in `.gpuq.toml`.
        Defaults to False, because requeuing re-runs the command from the start
        and that is destructive for anything not resumable.
        """
        if explicit is not None:
            return bool(explicit)
        value = load_project_defaults(repo_root).get("preemptible")
        return bool(value) if isinstance(value, bool) else False

    # ------------------------------------------------------------------
    # Project policy
    # ------------------------------------------------------------------
    def set_project_priority(
        self, project: str, priority: str | None, *, note: str | None = None
    ) -> dict[str, Any]:
        """Set a project's default priority and re-rank its queued work.

        Re-ranking is the point: marking a project urgent should affect the
        jobs already waiting, not only the next one submitted.
        """
        self.ensure_ready()
        if priority is not None:
            try:
                Priority(priority)
            except ValueError:
                raise GPUQError(
                    f"invalid priority {priority!r}; "
                    "choose from critical, high, normal, low"
                ) from None

        self.db.set_project_priority(project, priority, note=note)

        requeued = 0
        if priority is not None:
            rank = priority_rank(priority)
            for job in self.db.list_jobs(states=[JobState.QUEUED.value], project=project):
                if job.backend_job_id is None:
                    continue
                try:
                    self.backend.set_priority(job.backend_job_id, rank)
                    self.db.update_job(job.id, priority=priority)
                    requeued += 1
                except Exception:
                    continue

        return {
            "project": project,
            "priority": priority,
            "requeued": requeued,
            "message": (
                f"project '{project}' priority cleared"
                if priority is None
                else f"project '{project}' now submits at '{priority}'"
                + (f"; re-ranked {requeued} queued job(s)" if requeued else "")
            ),
        }

    def bump_job(self, job_id: int, level: str) -> dict[str, Any]:
        """Raise (or lower) one job's priority.

        A queued job is re-ranked immediately. A *running* job is re-ranked too,
        which matters because the dispatcher compares a waiter's rank against
        what is running when it decides whether anything may be displaced.
        """
        try:
            priority = Priority(level)
        except ValueError:
            raise GPUQError(
                f"invalid priority {level!r}; choose from critical, high, normal, low"
            ) from None

        job = self.get_job(job_id)
        if job.is_terminal:
            raise GPUQError(
                f"job #{job_id} already finished in state {job.state}; nothing to raise"
            )
        if job.backend_job_id is None:
            self.db.update_job(job_id, priority=priority.value)
            return {
                "job_id": job_id,
                "priority": priority.value,
                "message": f"job #{job_id} set to '{priority.value}' (not yet queued)",
            }

        rank = priority_rank(priority)
        self.db.update_job(job_id, priority=priority.value)
        try:
            self.backend.set_priority(job.backend_job_id, rank)
        except BackendUnavailable:
            # A running job has no queue position to re-rank; the recorded
            # priority is what the dispatcher reads, so this is not an error.
            self.backend.store.update(job.backend_job_id, priority_rank=rank)

        return {
            "job_id": job_id,
            "priority": priority.value,
            "state": job.state,
            "message": (
                f"job #{job_id} raised to '{priority.value}'"
                + (
                    "; it may now displace lower-priority preemptible work"
                    if job.state == JobState.QUEUED.value
                    else ""
                )
            ),
        }

    def annotate_job(
        self,
        job_id: int,
        *,
        description: str | None = None,
        blocks: str | None = None,
        eta_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Update what a job says about itself, while it is queued or running.

        Jobs often only learn their own duration after a few epochs, so a
        declared ETA must be correctable at runtime rather than fixed at submit.
        """
        job = self.get_job(job_id)
        if job.is_terminal:
            raise GPUQError(
                f"job #{job_id} already finished in state {job.state}; nothing to annotate"
            )
        updates: dict[str, Any] = {}
        if description is not None:
            updates["description"] = description.strip() or None
        if blocks is not None:
            updates["blocks"] = blocks.strip() or None
        if eta_seconds is not None:
            if eta_seconds < 0:
                raise GPUQError("eta must be >= 0")
            updates["eta_seconds"] = float(eta_seconds)
        if not updates:
            raise GPUQError("nothing to update; pass a description, --blocks or an eta")

        self.db.update_job(job_id, **updates)
        self._write_manifest(job_id)
        return {"job_id": job_id, **updates}

    def estimate(self, job: Job) -> dict[str, Any]:
        """Duration estimate for one job, with its provenance."""
        from workerq.eta import estimate_job

        return estimate_job(self, job).to_dict()

    def forecast(self, jobs: list[Job] | None = None) -> dict[int, dict[str, Any]]:
        """Projected start/finish for everything not yet finished."""
        from workerq.eta import forecast_queue

        self.ensure_ready()
        if jobs is None:
            jobs = self.sort_for_display(self.db.active_jobs())
        return forecast_queue(self, jobs)

    def preemption_report(self, job_id: int) -> dict[str, Any]:
        """Why a job stopped and where it now sits, for the worker that owns it."""
        job = self.get_job(job_id)
        position = self._queue_position(job_id)
        displacer = self.db.get_job(job.preempted_by) if job.preempted_by else None
        return {
            "job_id": job.id,
            "state": job.state,
            "preemptible": bool(job.preemptible),
            "preemption_count": job.preemption_count,
            "preempted_at": job.preempted_at,
            "preempted_by": job.preempted_by,
            "preempted_by_project": displacer.project if displacer else None,
            "preempted_reason": job.preempted_reason,
            "queue_position": position,
            "wait_reason": self.queue_wait_reason(job),
        }

    def wait_for(
        self, job_id: int, *, timeout: float | None = None, poll: float = 2.0
    ) -> Job:
        """Block until a job reaches a terminal state.

        This is the notification primitive: a worker whose job was displaced can
        wait on the same id rather than polling `status` and guessing.
        """
        import time as _time

        deadline = None if timeout is None else _time.monotonic() + timeout
        while True:
            job = self.get_job(job_id)
            if job.is_terminal:
                return job
            if deadline is not None and _time.monotonic() >= deadline:
                return job
            _time.sleep(max(0.2, poll))

    def list_project_priorities(self) -> list[dict[str, Any]]:
        self.ensure_ready()
        return self.db.list_project_priorities()

    def _reject_impossible(
        self, ram_mib: float | None, vram_mib: float | None, cpus: int | None
    ) -> None:
        """Fail fast on a request no amount of waiting could ever satisfy."""
        from workerq import resources as res

        if not self.config.resources.enforce:
            return
        capacity = res.capacity(self.config, gpu=self.gpu_info())
        if ram_mib is not None and ram_mib > capacity.usable_ram_mib:
            raise GPUQError(
                f"requested {ram_mib / 1024:.1f} GiB RAM but only "
                f"{capacity.usable_ram_mib / 1024:.1f} GiB is usable "
                f"({capacity.total_ram_mib / 1024:.1f} GiB total minus "
                f"{self.config.resources.reserve_ram_gb:.1f} GiB reserved). "
                "This job would never start."
            )
        if cpus is not None and cpus > capacity.usable_cpus:
            raise GPUQError(
                f"requested {cpus} CPUs but only {capacity.usable_cpus} are usable "
                f"of {capacity.total_cpus}. This job would never start."
            )
        if vram_mib is not None and vram_mib > capacity.usable_vram_mib:
            raise GPUQError(
                f"requested {vram_mib / 1024:.1f} GiB VRAM but only "
                f"{capacity.usable_vram_mib / 1024:.1f} GiB is usable. "
                "This job would never start."
            )

    def _admission_advisories(self, ram_mib: float | None) -> list[str]:
        """Warn about a declaration that will pass submit and then never start.

        `_reject_impossible` compares against installed capacity, but admission
        compares against what is *free*. On a machine with a large steady
        baseline - editors, browsers, a row of agents - those are very
        different numbers, so a job can be accepted and then wait for headroom
        that never arrives. Advisory rather than fatal: the submitter may know
        the machine is about to free up, and refusing outright would be worse
        than saying so.
        """
        if ram_mib is None or not self.config.resources.enforce:
            return []
        try:
            from workerq.telemetry import open_telemetry

            telemetry = open_telemetry(self.config.state_dir)
            floor_mib = (host.memory().total_mib or 0.0) * (
                self.config.resources.min_host_free_percent / 100.0
            )
            ok, total = telemetry.admission_likelihood(ram_mib, floor_mib)
        except Exception:
            return []
        if total < 100 or ok * 100 >= total * 25:
            return []
        percent = 100.0 * ok / total
        if ok == 0:
            return [
                f"{ram_mib / 1024:.0f} GiB RAM has never been free on this machine in "
                f"{total} samples. This job will queue but is unlikely to ever start. "
                "Declare what it really needs, or free memory first."
            ]
        return [
            f"{ram_mib / 1024:.0f} GiB RAM has been available only {percent:.0f}% of the "
            f"time here ({ok} of {total} samples), so this may wait a long time. "
            "If that is more than the job really needs, declaring less will start it sooner."
        ]

    def _create_snapshot(
        self,
        job_id: int,
        mode: SnapshotMode,
        repo_root: Path | None,
        submitted_cwd: Path,
        passthrough: list[str],
    ) -> Snapshot:
        # `git worktree add` derives the worktree's registry name from the
        # destination's basename, so every snapshot must have a *distinct*
        # basename. Naming them all "repo" made two agents submitting at the
        # same moment collide inside .git/worktrees/repo.
        if mode is SnapshotMode.GIT:
            assert repo_root is not None
            destination = self.config.snapshots_dir / str(job_id) / f"job-{job_id}"
            return create_git_snapshot(
                repo_root, destination, job_id=job_id, passthrough=passthrough
            )
        if mode is SnapshotMode.COPY:
            destination = self.config.snapshots_dir / str(job_id) / f"job-{job_id}"
            return create_copy_snapshot(submitted_cwd, destination, passthrough=passthrough)
        return Snapshot(mode=mode.value, path=None, repo_root=repo_root, passthrough=[])

    @staticmethod
    def _resolve_execution_cwd(
        snapshot: Snapshot, repo_root: Path | None, submitted_cwd: Path
    ) -> Path:
        """Run in the snapshot at the same relative depth the user submitted from."""
        if snapshot.path is None:
            return submitted_cwd
        if repo_root is None:
            return snapshot.path
        try:
            relative = submitted_cwd.relative_to(resolve_path(repo_root))
        except ValueError:
            return snapshot.path
        target = snapshot.path / relative
        return target if target.is_dir() else snapshot.path

    def runner_argv(self, job_id: int) -> list[str]:
        """Command the backend executes.

        Only the job id crosses the process boundary: the runner reads the
        exact argv back from the database as JSON. That is a strictly stronger
        guarantee than re-quoting user arguments through a Windows command
        line, which is what spec section 8.4 is protecting against.
        """
        from workerq.winproc import windowless_python

        return [windowless_python(sys.executable), "-m", "workerq", "_run", str(job_id)]

    @staticmethod
    def backend_label(job_id: int, project: str, priority: str) -> str:
        """Unique marker used to recover a backend id after a crash (spec 9.3)."""
        safe_project = project.replace(":", "_")
        return f"worker-q:{job_id}:{safe_project}:{priority}"

    @staticmethod
    def _infer_project(repo_root: Path | None, cwd: Path) -> str:
        defaults = load_project_defaults(repo_root)
        name = defaults.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return (repo_root or cwd).name or "default"

    # ------------------------------------------------------------------
    # Manifest / provenance (spec section 19)
    # ------------------------------------------------------------------
    def _write_manifest(self, job_id: int) -> None:
        job = self.db.get_job(job_id)
        if job is None:
            return
        job_dir = ensure_dir(self.config.job_dir(job_id))
        manifest_path = job_dir / "manifest.json"

        # Several processes may refresh a manifest (the runner at exit, any CLI
        # doing reconciliation). A writer that read the database *before* the
        # job finished could otherwise land after the runner and leave stale,
        # non-terminal provenance on disk. State only ever moves forward.
        if not job.state_enum.is_terminal and manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if JobState(existing.get("state", "")).is_terminal:
                    return
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        manifest = {
            "gpuq_job_id": job.id,
            "gpuq_version": __version__,
            "backend": job.backend,
            "backend_job_id": job.backend_job_id,
            "project": job.project,
            "label": job.label,
            "priority": job.priority,
            "command": job.command,
            "shell_mode": bool(job.shell_mode),
            "submitted_cwd": job.submitted_cwd,
            "execution_cwd": job.execution_cwd,
            "repo_root": job.repo_root,
            "snapshot_mode": job.snapshot_mode,
            "snapshot_commit": job.snapshot_commit,
            "snapshot_path": job.snapshot_path,
            "snapshot_passthrough": job.passthrough,
            "gpu_count": job.requested_gpu_count,
            "gpu_mode": job.gpu_mode,
            "requested_ram_mib": job.requested_ram_mib,
            "requested_vram_mib": job.requested_vram_mib,
            "requested_cpus": job.requested_cpus,
            "env": job.env,
            "submitted_at": job.queued_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "state": job.state,
            "exit_code": job.exit_code,
            "log_path": job.log_path,
            "host": job.host,
            "submitter_agent": job.submitter_agent,
        }
        atomic_write_text(
            manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_job(self, job_id: int, *, refresh: bool = True) -> Job:
        self.ensure_ready()
        job = self.db.get_job(job_id)
        if job is None:
            raise JobNotFound(f"no such worker-q job: {job_id}")
        if refresh and not job.is_terminal:
            self.reconcile_job(job)
            job = self.db.get_job(job_id) or job
        return job

    def list_jobs(
        self,
        *,
        all_jobs: bool = False,
        project: str | None = None,
        state: str | None = None,
        limit: int = 40,
        refresh: bool = True,
    ) -> list[Job]:
        self.ensure_ready()
        if refresh:
            self.reconcile(mutate=True)

        states: list[str] | None = None
        if state:
            try:
                states = [JobState(state.upper()).value]
            except ValueError:
                raise GPUQError(
                    f"invalid state {state!r}; choose from "
                    + ", ".join(s.value for s in JobState)
                ) from None

        jobs = self.db.list_jobs(states=states, project=project, limit=None)
        if not all_jobs and states is None:
            active = [j for j in jobs if not j.is_terminal]
            finished = [j for j in jobs if j.is_terminal][: max(0, limit - len(active))]
            jobs = active + finished
        elif limit:
            jobs = jobs[:limit]
        return self.sort_for_display(jobs)

    def sort_for_display(self, jobs: list[Job]) -> list[Job]:
        """RUNNING, then QUEUED in dispatch order, then newest finished."""
        order = self._queue_order()

        def key(job: Job) -> tuple:
            state = job.state_enum
            if state is JobState.RUNNING:
                return (0, 0, job.id)
            if state in (JobState.QUEUED, JobState.PREPARING):
                return (1, order.get(job.id, 10**9), job.id)
            return (2, -_sort_ts(job.finished_at or job.updated_at), -job.id)

        return sorted(jobs, key=key)

    def _queue_order(self) -> dict[int, int]:
        """Map worker-q job id -> position the dispatcher will actually run it in."""
        try:
            backend_jobs = self.backend.list_jobs()
        except Exception:
            return {}
        order: dict[int, int] = {}
        position = 0
        for bjob in backend_jobs:
            if bjob.state != BACKEND_QUEUED:
                continue
            job_id = _job_id_from_label(bjob.label)
            if job_id is not None:
                order[job_id] = position
                position += 1
        return order

    def _queue_position(self, job_id: int) -> int | None:
        return self._queue_order().get(job_id)

    def queue_wait_reason(self, job: Job) -> str | None:
        if job.backend_job_id is None:
            return None
        try:
            return self.backend.get_job(job.backend_job_id).wait_reason
        except Exception:
            return None

    def job_detail(self, job_id: int) -> dict[str, Any]:
        job = self.get_job(job_id)
        data = job.to_dict()
        backend_state = None
        wait_reason = None
        if job.backend_job_id is not None:
            try:
                bjob = self.backend.get_job(job.backend_job_id)
                backend_state = bjob.state
                wait_reason = bjob.wait_reason
                if bjob.output_path and not data.get("log_path"):
                    data["log_path"] = str(bjob.output_path)
            except Exception:
                backend_state = "UNKNOWN"
        data["backend_state"] = backend_state
        forecast = self.forecast()
        entry = forecast.get(job_id, {})
        data["estimate"] = entry.get("estimate") or self.estimate(job)
        data["eta_source"] = entry.get("eta_source") or data["estimate"].get("source")
        data["start_at_estimate"] = entry.get("start_at")
        data["finish_at_estimate"] = entry.get("finish_at")
        data["wait_reason"] = wait_reason
        data["queue_position"] = self._queue_position(job_id)
        data["manifest_path"] = str(self.config.job_dir(job_id) / "manifest.json")
        result_path = self.config.job_dir(job_id) / "result.json"
        if result_path.exists():
            try:
                data["result"] = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        return data

    def resolve_log_path(self, job: Job) -> Path | None:
        if job.log_path:
            path = expand_path(job.log_path)
            if path.exists():
                return path
        if job.backend_job_id is not None:
            try:
                candidate = self.backend.output_path(job.backend_job_id)
            except Exception:
                candidate = None
            if candidate and Path(candidate).exists():
                return Path(candidate)
        fallback = self.config.log_path(job.id)
        return fallback if fallback.exists() else None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def cancel(self, job_id: int, *, force: bool = False) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.is_terminal:
            return {
                "job_id": job.id,
                "state": job.state,
                "action": "none",
                "message": f"job #{job.id} already finished in state {job.state}",
            }
        if job.backend_job_id is None:
            self.db.try_update_state(
                job_id, JobState.CANCELLED, finished_at=utcnow_iso(), error="cancelled before enqueue"
            )
            return {
                "job_id": job.id,
                "state": JobState.CANCELLED.value,
                "action": "removed",
                "message": f"job #{job.id} cancelled before it reached the queue",
            }

        action = self.backend.cancel(job.backend_job_id, force=force)
        if action == "removed":
            self.db.try_update_state(
                job_id, JobState.CANCELLED, finished_at=utcnow_iso(), error="cancelled while queued"
            )
            message = f"job #{job.id} removed from the queue; it will not run"
        elif action == "terminating":
            self.db.update_job(job_id, error="cancellation requested")
            confirmed = self.backend.wait_until_finished(
                job.backend_job_id,
                timeout=(2.0 if force else self.config.core.cancel_grace_seconds + 10.0),
            )
            if confirmed:
                self.db.try_update_state(
                    job_id,
                    JobState.CANCELLED,
                    finished_at=utcnow_iso(),
                    error="cancelled while running",
                )
                message = f"job #{job.id} was running and has been terminated"
            else:
                message = (
                    f"job #{job.id} termination requested; the process tree has not "
                    "exited yet. Re-run with --force to kill it immediately."
                )
        else:
            self.reconcile_job(job)
            refreshed = self.db.get_job(job_id) or job
            return {
                "job_id": job.id,
                "state": refreshed.state,
                "action": "none",
                "message": f"job #{job.id} already finished in state {refreshed.state}",
            }

        self._write_manifest(job_id)
        final = self.db.get_job(job_id) or job
        return {
            "job_id": job.id,
            "state": final.state,
            "action": action,
            "message": message,
        }

    def promote(self, job_id: int) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.state_enum is not JobState.QUEUED:
            raise GPUQError(
                f"job #{job.id} is {job.state}; only QUEUED jobs can be promoted. "
                "To displace running work, raise its priority with "
                "'workerq bump' - which stops a running job only if that job "
                "was submitted --preemptible."
            )
        if job.backend_job_id is None:
            raise GPUQError(f"job #{job.id} has no backend job to promote")
        self.backend.promote(job.backend_job_id)
        return {
            "job_id": job.id,
            "state": job.state,
            "queue_position": self._queue_position(job.id),
            "message": f"job #{job.id} moved to the front of the queue",
        }

    def get_reserve(self) -> dict[str, Any]:
        """The reserve in force, plus what it leaves for jobs."""
        from workerq.gpu import query_gpus
        from workerq.resources import Reserve, capacity

        reserve = self.backend.get_reserve()
        configured = Reserve.from_config(self.config)
        gpu = query_gpus(include_processes=False)
        cap = capacity(self.config, gpu=gpu, mem=host.memory(), reserve=reserve)
        return {
            "reserve": reserve.to_dict(),
            "configured": configured.to_dict(),
            "is_default": (
                reserve.ram_mib == configured.ram_mib
                and reserve.vram_mib == configured.vram_mib
                and reserve.cpus == configured.cpus
            ),
            "capacity": cap.to_dict(),
            "gpu_free_threshold_percent": self.backend.get_gpu_free_percent(),
        }

    def set_reserve(
        self,
        *,
        ram_gb: float | None = None,
        vram_gb: float | None = None,
        cpus: int | None = None,
        label: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Claim headroom back from the queue, effective on the next tick.

        Anything not named keeps its configured value, so reclaiming RAM does
        not silently hand out the CPUs as well.
        """
        from workerq.gpu import query_gpus
        from workerq.resources import Reserve, capacity, cpu_count

        base = Reserve.from_config(self.config)
        gib = 1024.0
        reserve = Reserve(
            ram_mib=base.ram_mib if ram_gb is None else float(ram_gb) * gib,
            vram_mib=base.vram_mib if vram_gb is None else float(vram_gb) * gib,
            cpus=base.cpus if cpus is None else int(cpus),
            label=label,
            expires_at=expires_at,
        )

        # A reserve at or beyond the machine's size stops the queue forever.
        # Refuse it rather than clamp it: silently doing something other than
        # what was asked is worse than saying no.
        mem = host.memory()
        total_ram = mem.total_mib or 0.0
        gpu = query_gpus(include_processes=False)
        total_vram = sum(d.memory_total_mib or 0.0 for d in gpu.devices) if gpu.available else 0.0
        total_cpus = cpu_count()
        if total_ram and reserve.ram_mib >= total_ram:
            raise GPUQError(
                f"reserving {reserve.ram_mib / gib:.1f} GiB RAM leaves nothing of the "
                f"machine's {total_ram / gib:.1f} GiB; nothing would ever start"
            )
        if total_vram and reserve.vram_mib >= total_vram:
            raise GPUQError(
                f"reserving {reserve.vram_mib / gib:.1f} GiB VRAM leaves nothing of the "
                f"machine's {total_vram / gib:.1f} GiB; no GPU job would ever start"
            )
        if reserve.cpus >= total_cpus:
            raise GPUQError(
                f"reserving {reserve.cpus} CPUs leaves nothing of the machine's "
                f"{total_cpus}; nothing would ever start"
            )

        self.backend.set_reserve(reserve)
        cap = capacity(self.config, gpu=gpu, mem=mem, reserve=reserve)

        # Report honestly: running work is not touched, and a tighter reserve
        # can make queued jobs impossible. Both are things the caller has to
        # be told rather than discover.
        holding: list[dict[str, Any]] = []
        for job in self.db.list_jobs(states=[JobState.RUNNING.value]):
            holding.append(
                {
                    "id": job.id,
                    "project": job.project,
                    "ram_mib": job.requested_ram_mib,
                    "vram_mib": job.requested_vram_mib,
                    "preemptible": bool(job.preemptible),
                }
            )
        stranded: list[dict[str, Any]] = []
        for job in self.db.list_jobs(states=[JobState.QUEUED.value]):
            reasons = []
            if (job.requested_ram_mib or 0) > cap.usable_ram_mib:
                reasons.append(f"needs {(job.requested_ram_mib or 0) / gib:.1f} GiB RAM")
            if (job.requested_vram_mib or 0) > cap.usable_vram_mib:
                reasons.append(f"needs {(job.requested_vram_mib or 0) / gib:.1f} GiB VRAM")
            if (job.requested_cpus or 0) > cap.usable_cpus:
                reasons.append(f"needs {job.requested_cpus} CPUs")
            if reasons:
                stranded.append({"id": job.id, "project": job.project, "why": "; ".join(reasons)})

        return {
            "reserve": reserve.to_dict(),
            "capacity": cap.to_dict(),
            "running": holding,
            "stranded": stranded,
        }

    def clear_reserve(self) -> dict[str, Any]:
        """Give the queue back the headroom, restoring the configured reserve."""
        self.backend.set_reserve(None)
        return self.get_reserve()

    def set_concurrency(self, count: int) -> dict[str, Any]:
        if count < 1:
            raise GPUQError("concurrency must be >= 1")
        from workerq.config import set_dotted_and_save

        self.config = set_dotted_and_save(self.config, "core.max_concurrent_jobs", count)
        self.backend.config = self.config
        self.backend.set_slots(count)
        return {"max_concurrent_jobs": count, "backend_slots": self.backend.get_slots()}

    def get_concurrency(self) -> dict[str, Any]:
        return {
            "config": self.config.core.max_concurrent_jobs,
            "backend_slots": self.backend.get_slots(),
        }

    def set_gpu_threshold(self, percent: int) -> dict[str, Any]:
        if not 0 <= percent <= 100:
            raise GPUQError("gpu threshold must be between 0 and 100")
        from workerq.config import set_dotted_and_save

        self.config = set_dotted_and_save(
            self.config, "gpu.free_memory_threshold_percent", percent
        )
        self.backend.config = self.config
        self.backend.set_gpu_free_percent(percent)
        return {
            "free_memory_threshold_percent": percent,
            "backend": self.backend.get_gpu_free_percent(),
        }

    # ------------------------------------------------------------------
    # Reconciliation (spec section 11.9)
    # ------------------------------------------------------------------
    def reconcile_job(self, job: Job, *, mutate: bool = True) -> str | None:
        """Align one DB row with backend truth. Returns a change description."""
        if job.is_terminal:
            return None

        if job.backend_job_id is None:
            # PREPARING rows from a crashed submit never reached the queue.
            if job.state_enum is JobState.PREPARING:
                age = _age_or_zero(job.created_at)
                if age > 300:
                    if mutate:
                        self.db.try_update_state(
                            job.id,
                            JobState.LOST,
                            error="submission did not complete (process exited during preparation)",
                            finished_at=utcnow_iso(),
                        )
                    return f"job #{job.id}: stale PREPARING -> LOST"
            return None

        try:
            bjob: BackendJob = self.backend.get_job(job.backend_job_id)
        except Exception:
            return None

        target = map_backend_state(bjob.state, bjob.exit_code)
        if target is None or target is job.state_enum:
            return None

        # Never resurrect a cancelled job, and never overwrite a runner-recorded
        # terminal state with a coarser backend guess.
        if bjob.state == BACKEND_REMOVED:
            target = JobState.CANCELLED

        updates: dict[str, Any] = {}
        if target is JobState.RUNNING:
            if not job.started_at and bjob.started_at:
                updates["started_at"] = bjob.started_at
            if bjob.pid:
                updates["runner_pid"] = bjob.pid
        elif target.is_terminal:
            updates["finished_at"] = job.finished_at or bjob.finished_at or utcnow_iso()
            if bjob.exit_code is not None and job.exit_code is None:
                updates["exit_code"] = bjob.exit_code
            if target is JobState.FAILED and bjob.exit_code is None and not job.error:
                updates["error"] = "job ended without recording an exit code"

        if not mutate:
            return f"job #{job.id}: {job.state} -> {target.value}"

        changed = self.db.try_update_state(job.id, target, **updates)
        if changed is None:
            return None
        self._write_manifest(job.id)
        return f"job #{job.id}: {job.state} -> {target.value}"

    def reconcile(self, *, mutate: bool = True) -> list[str]:
        """Repair metadata after crashes/restarts."""
        self.ensure_ready()
        changes: list[str] = []
        for job in self.db.list_jobs(states=[s.value for s in ACTIVE_STATES]):
            try:
                change = self.reconcile_job(job, mutate=mutate)
            except Exception as exc:  # never let one bad row break status
                change = f"job #{job.id}: reconcile error: {exc}"
            if change:
                changes.append(change)
        changes.extend(self._recover_lost_backend_ids(mutate=mutate))
        return changes

    def _recover_lost_backend_ids(self, *, mutate: bool) -> list[str]:
        """Re-attach rows whose backend id was never written (crash mid-submit)."""
        changes: list[str] = []
        orphans = [
            j
            for j in self.db.list_jobs(states=[JobState.PREPARING.value])
            if j.backend_job_id is None
        ]
        if not orphans:
            return changes
        try:
            backend_jobs = self.backend.list_jobs()
        except Exception:
            return changes
        by_job_id = {}
        for bjob in backend_jobs:
            jid = _job_id_from_label(bjob.label)
            if jid is not None:
                by_job_id[jid] = bjob
        for job in orphans:
            bjob = by_job_id.get(job.id)
            if bjob is None:
                continue
            if mutate:
                self.db.update_job(job.id, backend_job_id=bjob.backend_id)
                self.db.try_update_state(job.id, JobState.QUEUED)
            changes.append(f"job #{job.id}: recovered backend job {bjob.backend_id} from label")
        return changes

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------
    def gpu_info(self) -> GpuInfo:
        return query_gpus()

    def own_gpu_pids(self) -> set[int]:
        """PIDs belonging to jobs worker-q launched, for foreign-process detection.

        Two PIDs are collected per job. On Windows a virtualenv `python.exe`
        can be a trampoline that re-execs the real interpreter as a child, so
        the PID the dispatcher launched and the PID the runner reports are
        both legitimate and frequently differ. Missing either would make
        `doctor` report worker-q's own job as a foreign GPU process.
        """
        pids: set[int] = set()
        for job in self.db.list_jobs(states=[JobState.RUNNING.value]):
            if job.runner_pid:
                pids.add(int(job.runner_pid))
            if job.backend_job_id is not None:
                try:
                    backend_pid = self.backend.get_job(job.backend_job_id).pid
                except Exception:
                    backend_pid = None
                if backend_pid:
                    pids.add(int(backend_pid))
        return pids

    def own_pids(self) -> set[int]:
        """Every process belonging to a running worker-q job, including children.

        `own_gpu_pids` only knows the wrapper PIDs worker-q launched; the real
        workload is typically a descendant (an interpreter re-exec, a
        dataloader pool), so memory attribution must walk the process tree or
        worker-q reports its own job as somebody else's.
        """
        from workerq import host as _host

        return _host.descendants_of(self.own_gpu_pids())

    def throughput(self, *, hours: float = 24.0) -> dict[str, Any]:
        """Queue performance over a window: outcomes, waits and utilisation."""
        from workerq.util import age_seconds as _age

        cutoff = hours * 3600.0
        succeeded = failed = cancelled = lost = 0
        waits: list[float] = []
        runtimes: list[float] = []
        busy_seconds = 0.0

        for job in self.db.list_jobs():
            if not job.is_terminal:
                continue
            age = _age(job.finished_at or job.updated_at)
            if age is not None and age > cutoff:
                continue
            state = job.state
            if state == JobState.SUCCEEDED.value:
                succeeded += 1
            elif state == JobState.FAILED.value:
                failed += 1
            elif state == JobState.CANCELLED.value:
                cancelled += 1
            elif state == JobState.LOST.value:
                lost += 1
            if job.wait_seconds is not None:
                waits.append(job.wait_seconds)
            if job.runtime_seconds is not None:
                runtimes.append(job.runtime_seconds)
                busy_seconds += job.runtime_seconds

        finished = succeeded + failed + cancelled + lost
        completed = succeeded + failed

        def _median(values: list[float]) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[middle]
            return (ordered[middle - 1] + ordered[middle]) / 2.0

        return {
            "window_hours": hours,
            "finished": finished,
            "succeeded": succeeded,
            "failed": failed,
            "cancelled": cancelled,
            "lost": lost,
            "success_rate": (100.0 * succeeded / completed) if completed else 100.0,
            "median_wait_seconds": _median(waits),
            "median_runtime_seconds": _median(runtimes),
            "busy_seconds": busy_seconds,
            "utilisation_percent": min(100.0, 100.0 * busy_seconds / (hours * 3600.0))
            if hours
            else 0.0,
        }

    def status_summary(self) -> dict[str, Any]:
        from workerq import host as _host

        counts = self.db.count_by_state()
        memory = _host.memory()
        return {
            "host": memory.to_dict(),
            "resources_enforced": self.config.resources.enforce,
            "concurrency": self.config.core.max_concurrent_jobs,
            "backend_slots": self.backend.get_slots(),
            "gpu_free_threshold_percent": self.backend.get_gpu_free_percent(),
            "daemon_running": self.backend.daemon_running(),
            "state_dir": str(self.config.state_dir),
            "counts": counts,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _job_id_from_label(label: str | None) -> int | None:
    if not label or not label.startswith("worker-q:"):
        return None
    parts = label.split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _sort_ts(value: str | None) -> float:
    from workerq.util import parse_iso

    dt = parse_iso(value)
    return dt.timestamp() if dt else 0.0


def _age_or_zero(value: str | None) -> float:
    from workerq.util import age_seconds

    return age_seconds(value) or 0.0


def detect_agent() -> str | None:
    """Best-effort identification of the agent/terminal that submitted a job."""
    for key in ("CLAUDECODE", "CLAUDE_CODE"):
        if os.environ.get(key):
            return "claude-code"
    if os.environ.get("CURSOR_TRACE_ID"):
        return "cursor"
    if os.environ.get("CODEX_SANDBOX") or os.environ.get("CODEX_HOME"):
        return "codex"
    term = os.environ.get("TERM_PROGRAM")
    return term or None
