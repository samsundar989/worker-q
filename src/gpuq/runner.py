"""`gpuq _run <job_id>` - the wrapper every queued command executes inside.

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

from gpuq import __version__
from gpuq.config import Config, load_config
from gpuq.db import Database
from gpuq.gpu import query_gpus
from gpuq.models import Job, JobState
from gpuq.util import atomic_write_text, ensure_dir, hostname, utcnow_iso
from gpuq.winproc import ProcessGroup, child_creationflags, posix_child_kwargs

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


def _cancel_requested(config: Config, job: Job) -> tuple[bool, bool]:
    """(cancel_requested, force) as recorded by the backend for this job."""
    if job.backend_job_id is None:
        return False, False
    try:
        from gpuq.backends.queue_store import QueueStore

        store = QueueStore(config.backend_dir / "queue.sqlite3")
        row = store.get(job.backend_job_id)
        store.close()
        if not row:
            return False, False
        return bool(row.get("cancel_requested")), bool(row.get("cancel_force"))
    except Exception:
        return False, False


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
        print(f"gpuq runner: no such job {job_id}", file=sys.stderr, flush=True)
        return 127

    # Ownership check: a runner must never take over a job that has already
    # finished or been cancelled.
    if job.is_terminal:
        print(
            f"gpuq runner: job #{job_id} is already {job.state}; refusing to run",
            file=sys.stderr,
            flush=True,
        )
        return 0 if job.state == JobState.SUCCEEDED.value else 1

    job_dir = ensure_dir(config.job_dir(job_id))

    try:
        argv = _build_argv(job, argv_override)
    except RunnerError as exc:
        db.set_error(job_id, str(exc))
        print(f"gpuq runner: {exc}", file=sys.stderr, flush=True)
        return 127

    execution_cwd = Path(job.execution_cwd or job.submitted_cwd)
    if not execution_cwd.is_dir():
        message = f"execution directory is missing: {execution_cwd}"
        db.set_error(job_id, message)
        print(f"gpuq runner: {message}", file=sys.stderr, flush=True)
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
            f"gpuq runner: job #{job_id} is {current.state if current else 'unknown'}; "
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
        f"gpuq: job #{job_id} starting at {started_at}\n"
        f"gpuq: cwd={execution_cwd}\n"
        f"gpuq: snapshot={job.snapshot_mode} {job.snapshot_commit or '-'}\n"
        f"gpuq: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n"
        f"gpuq: command={argv}\n"
        "gpuq: " + "-" * 60,
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
                    f"\ngpuq: {executable!r} was not found in the snapshot at "
                    f"{execution_cwd}.\n"
                    "gpuq: A queued job runs a frozen copy of the repository, which "
                    "excludes gitignored paths such as .venv.\n"
                    "gpuq: Fix by using an absolute path to the interpreter, or "
                    "re-submit with --passthrough .venv"
                )
        db.set_error(job_id, message)
        _write_result(job_dir, job_id, None, JobState.FAILED, started_at, message)
        print(f"gpuq: {message}", file=sys.stderr, flush=True)
        return 127

    group = ProcessGroup(f"gpuq-run-{job_id}")
    group.assign(proc.pid)

    stop_watch = threading.Event()
    cancelled = threading.Event()

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
        """Poll for `gpuq cancel`, then stop the child tree.

        Windows cannot deliver a POSIX-style SIGTERM to a console-less child,
        so a polite CTRL_BREAK is attempted first and a hard tree kill is the
        guaranteed backstop after the configured grace period.
        """
        grace = max(0, config.core.cancel_grace_seconds)
        signalled_at: float | None = None
        while not stop_watch.wait(_CANCEL_POLL_SECONDS):
            requested, force = _cancel_requested(config, job)
            if not requested:
                continue
            cancelled.set()
            if signalled_at is None:
                signalled_at = time.monotonic()
                print("\ngpuq: cancellation requested; stopping job", flush=True)
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
        from gpuq.core import GPUQService

        service = GPUQService(config)
        service.ensure_ready()
        service._write_manifest(job_id)
        service.close()
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(
            f"gpuq: warning: could not update manifest.json: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"gpuq: " + "-" * 60 + f"\ngpuq: job #{job_id} {final_state.value} "
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
