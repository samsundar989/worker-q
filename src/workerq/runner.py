"""`workerq _run <job_id>` - the wrapper every queued command executes inside.

Internal, but directly testable (spec section 13). It owns the transition into
RUNNING, captures provenance, runs the user command, forwards cancellation to
the child's whole process tree, and persists the final state before exiting
with the user's own exit code.

The user's argv is read back from the database as JSON rather than re-parsed
from a command line, so arguments containing spaces, quotes, globs, `=`,
Unicode or shell metacharacters reach the process byte-identical.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from workerq import __version__
from workerq.config import Config, load_config
from workerq.db import Database
from workerq.gpu import query_gpus
from workerq.models import Job, JobState
from workerq.util import atomic_write_text, ensure_dir, hostname, utcnow_iso
from workerq.winproc import ProcessGroup, child_creationflags, posix_child_kwargs

_CANCEL_POLL_SECONDS = 0.5


class RunnerError(RuntimeError):
    pass


def _build_argv(job: Job, override: list[str] | None) -> list[str]:
    command = override if override else job.command
    if not command:
        raise RunnerError(f"job #{job.id} has no command recorded")
    if job.shell_mode:
        shell_string = command[0] if len(command) == 1 else subprocess.list2cmdline(command)
        if os.name == "nt":
            comspec = os.environ.get("COMSPEC") or "cmd.exe"
            return [comspec, "/d", "/c", shell_string]
        return ["/bin/sh", "-c", shell_string]
    return list(command)


def _capture_environment(config: Config, job: Job, argv: list[str]) -> dict[str, Any]:
    """Record what this run actually saw, for reproducing a result later."""
    gpu = query_gpus(include_processes=False)
    return {
        "gpuq_version": __version__,
        "gpuq_job_id": job.id,
        "host": hostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "runner_pid": os.getpid(),
        "argv": argv,
        "execution_cwd": job.execution_cwd,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "path": os.environ.get("PATH"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "started_at": utcnow_iso(),
        "gpu_inventory": gpu.to_dict(),
        "job_env_overrides": job.env,
    }


def _stop_intent(config: Config, job: Job) -> tuple[bool, bool, int | None]:
    """(stop_requested, force, preempted_by) as recorded by the backend.

    Cancellation and preemption both stop the child, but they mean opposite
    things afterwards: a cancelled job is finished, a preempted one returns to
    the queue and runs again later.
    """
    if job.backend_job_id is None:
        return False, False, None
    try:
        from workerq.backends.queue_store import QueueStore

        store = QueueStore(config.backend_dir / "queue.sqlite3")
        row = store.get(job.backend_job_id)
        store.close()
        if not row:
            return False, False, None
        if row.get("preempt_requested"):
            return True, False, row.get("preempt_by")
        return bool(row.get("cancel_requested")), bool(row.get("cancel_force")), None
    except Exception:
        return False, False, None


def run_job(
    job_id: int,
    argv_override: list[str] | None = None,
    *,
    config: Config | None = None,
) -> int:
    # Job logs are read back as UTF-8, and the banner below echoes the user's
    # own argv. On Windows a file-backed stdout defaults to the locale codec,
    # so a Unicode argument would otherwise kill the job before it started.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass

    config = config or load_config()
    config.ensure_dirs()
    db = Database(config.db_path)
    db.initialize()

    job = db.get_job(job_id)
    if job is None:
        print(f"worker-q runner: no such job {job_id}", file=sys.stderr, flush=True)
        return 127

    # Ownership check: a runner must never take over a job that has already
    # finished or been cancelled.
    if job.is_terminal:
        print(
            f"worker-q runner: job #{job_id} is already {job.state}; refusing to run",
            file=sys.stderr,
            flush=True,
        )
        return 0 if job.state == JobState.SUCCEEDED.value else 1

    job_dir = ensure_dir(config.job_dir(job_id))

    try:
        argv = _build_argv(job, argv_override)
    except RunnerError as exc:
        db.set_error(job_id, str(exc))
        print(f"worker-q runner: {exc}", file=sys.stderr, flush=True)
        return 127

    execution_cwd = Path(job.execution_cwd or job.submitted_cwd)
    if not execution_cwd.is_dir():
        message = f"execution directory is missing: {execution_cwd}"
        db.set_error(job_id, message)
        print(f"worker-q runner: {message}", file=sys.stderr, flush=True)
        return 127

    # ---- 4. record RUNNING + provenance -------------------------------
    started_at = utcnow_iso()
    updated = db.try_update_state(
        job_id,
        JobState.RUNNING,
        started_at=started_at,
        runner_pid=os.getpid(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if updated is None:
        current = db.get_job(job_id)
        print(
            f"worker-q runner: job #{job_id} is {current.state if current else 'unknown'}; "
            "not starting",
            file=sys.stderr,
            flush=True,
        )
        return 1

    environment = _capture_environment(config, updated, argv)
    atomic_write_text(
        job_dir / "environment.json", json.dumps(environment, indent=2, ensure_ascii=False) + "\n"
    )

    print(
        f"worker-q: job #{job_id} starting at {started_at}\n"
        f"worker-q: cwd={execution_cwd}\n"
        f"worker-q: snapshot={job.snapshot_mode} {job.snapshot_commit or '-'}\n"
        f"worker-q: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n"
        f"worker-q: command={argv}\n"
        "worker-q: " + "-" * 60,
        flush=True,
    )

    # ---- 5/6/7. chdir, apply env, execute ------------------------------
    child_env = dict(os.environ)
    for key, value in job.env.items():
        child_env[key] = value
    child_env["GPUQ_JOB_ID"] = str(job_id)
    child_env["GPUQ_PROJECT"] = job.project
    child_env["GPUQ_STATE_DIR"] = str(config.state_dir)

    os.chdir(execution_cwd)

    # The runner is launched by a console-less interpreter, so sys.stdout is a
    # real file only because the dispatcher redirected it to the job log. If a
    # job was queued without a log path there is nothing to inherit.
    child_stdout: Any = sys.stdout if sys.stdout is not None else subprocess.DEVNULL

    kwargs: dict[str, Any] = {
        "cwd": str(execution_cwd),
        "env": child_env,
        "stdin": subprocess.DEVNULL,
        "stdout": child_stdout,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
        **posix_child_kwargs(),
    }
    flags = child_creationflags()
    if flags:
        kwargs["creationflags"] = flags

    try:
        proc = subprocess.Popen(argv, **kwargs)
    except (OSError, ValueError) as exc:
        message = f"failed to start command: {exc}"
        # By far the most common cause: a relative path to something that is
        # gitignored (a virtualenv interpreter), so it exists in the live repo
        # but not in the snapshot the job actually runs in.
        executable = argv[0]
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 2 or (
            isinstance(exc, FileNotFoundError)
        ):
            if not Path(executable).is_absolute() and job.snapshot_mode in ("git", "copy"):
                message += (
                    f"\nworker-q: {executable!r} was not found in the snapshot at "
                    f"{execution_cwd}.\n"
                    "worker-q: A queued job runs a frozen copy of the repository, which "
                    "excludes gitignored paths such as .venv.\n"
                    "worker-q: Fix by using an absolute path to the interpreter, or "
                    "re-submit with --passthrough .venv"
                )
        db.set_error(job_id, message)
        _write_result(job_dir, job_id, None, JobState.FAILED, started_at, message)
        print(f"worker-q: {message}", file=sys.stderr, flush=True)
        return 127

    group = ProcessGroup(f"gpuq-run-{job_id}")
    group.assign(proc.pid)

    stop_watch = threading.Event()
    cancelled = threading.Event()
    #: Backend id of the job that displaced this one, if any. Non-empty means
    #: the job was preempted rather than cancelled or finished.
    preempted_by: list[int] = []

    def _forward(signum: int, _frame: Any) -> None:
        """A signal aimed at the runner must reach the whole child tree."""
        cancelled.set()
        group.signal_break()
        group.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT) + (
        (signal.SIGBREAK,) if hasattr(signal, "SIGBREAK") else ()
    ):
        try:
            signal.signal(sig, _forward)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass

    def _watch_cancel() -> None:
        """Poll for `workerq cancel`, then stop the child tree.

        Windows cannot deliver a POSIX-style SIGTERM to a console-less child,
        so a polite CTRL_BREAK is attempted first and a hard tree kill is the
        guaranteed backstop after the configured grace period.
        """
        grace = max(0, config.core.cancel_grace_seconds)
        signalled_at: float | None = None
        while not stop_watch.wait(_CANCEL_POLL_SECONDS):
            requested, force, by = _stop_intent(config, job)
            if not requested:
                continue
            if by is not None:
                # Preemption, not cancellation: do not mark the job cancelled,
                # and allow the job's own grace period to stop cleanly.
                if not preempted_by:
                    preempted_by.append(int(by))
                grace = max(0, config.preemption.grace_seconds)
            else:
                cancelled.set()
            if signalled_at is None:
                signalled_at = time.monotonic()
                if by is not None:
                    print(
                        f"\nworker-q: PREEMPTED by a higher-priority job "
                        f"(backend job #{by}). Stopping cleanly; this job goes "
                        "back to the queue and will run again from the start.",
                        flush=True,
                    )
                else:
                    print("\nworker-q: cancellation requested; stopping job", flush=True)
                group.signal_break()
                if not force:
                    continue
            if force or (time.monotonic() - signalled_at) >= grace:
                group.terminate()
                return

    watcher = threading.Thread(target=_watch_cancel, name="gpuq-cancel-watch", daemon=True)
    watcher.start()

    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:  # pragma: no cover
        cancelled.set()
        group.terminate()
        exit_code = proc.wait()
    finally:
        stop_watch.set()
        watcher.join(timeout=2.0)
        group.close()

    finished_at = utcnow_iso()

    # ---- 9/10. persist final state -------------------------------------
    current = db.get_job(job_id)

    # A preempted job did not finish: it goes back to QUEUED with its
    # provenance recorded, so the worker can see why it stopped and follow it.
    if preempted_by and (current is None or not current.is_terminal):
        by = preempted_by[0]
        displacer = db.get_job_by_backend_id(job.backend, by)
        who = (
            f"job #{displacer.id} ({displacer.project}, {displacer.priority})"
            if displacer
            else f"backend job #{by}"
        )
        reason = f"preempted by {who}"
        db.try_update_state(
            job_id,
            JobState.QUEUED,
            started_at=None,
            runner_pid=None,
            exit_code=None,
            preemption_count=(current.preemption_count if current else 0) + 1,
            preempted_at=finished_at,
            preempted_by=displacer.id if displacer else None,
            preempted_reason=reason,
            error=reason,
        )
        _write_result(job_dir, job_id, exit_code, JobState.QUEUED, started_at, reason)
        print(
            f"worker-q: {'-' * 60}\n"
            f"worker-q: job #{job_id} PREEMPTED at {finished_at} - {reason}.\n"
            f"worker-q: It is QUEUED again, keeps id #{job_id}, and will re-run "
            "its command from the start.\n"
            f"worker-q: Track it with 'workerq show {job_id}' or block on it "
            f"with 'workerq wait {job_id}'.",
            flush=True,
        )
        db.close()
        return exit_code

    if current is not None and current.is_terminal:
        final_state = current.state_enum  # already CANCELLED by the CLI
    elif cancelled.is_set():
        final_state = JobState.CANCELLED
    elif exit_code == 0:
        final_state = JobState.SUCCEEDED
    else:
        final_state = JobState.FAILED

    db.try_update_state(
        job_id,
        final_state,
        exit_code=exit_code,
        finished_at=finished_at,
        error=("cancelled while running" if final_state is JobState.CANCELLED else None),
    )
    _write_result(job_dir, job_id, exit_code, final_state, started_at, None)

    # Refresh the manifest so provenance on disk reflects the final outcome.
    # A failure here must not change the job's result, but it must be visible
    # rather than silently swallowed.
    try:
        from workerq.core import GPUQService

        service = GPUQService(config)
        service.ensure_ready()
        service._write_manifest(job_id)
        service.close()
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(
            f"worker-q: warning: could not update manifest.json: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"worker-q: " + "-" * 60 + f"\nworker-q: job #{job_id} {final_state.value} "
        f"(exit code {exit_code}) at {finished_at}",
        flush=True,
    )
    db.close()

    # ---- 11. exit with the user's exit code ----------------------------
    return exit_code


def _write_result(
    job_dir: Path,
    job_id: int,
    exit_code: int | None,
    state: JobState,
    started_at: str,
    error: str | None,
) -> None:
    payload = {
        "gpuq_job_id": job_id,
        "state": state.value,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": utcnow_iso(),
        "error": error,
    }
    try:
        atomic_write_text(
            job_dir / "result.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
    except OSError:  # pragma: no cover
        pass
