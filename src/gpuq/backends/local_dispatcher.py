"""`LocalDispatcherBackend` - the V1 execution backend.

On Linux this role is filled by GPU Task Spooler (spec section 2). Windows has
no equivalent, so GPUQ ships an equivalent-scope dispatcher of its own that
provides exactly the backend contract in spec section 7: a persistent,
terminal-independent queue with slot limits, GPU-free-memory gating, per-job
logs, labels, reordering and process-group termination.

All of it sits behind `SchedulerBackend`, so nothing above this module knows
which one is in use.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from gpuq import BACKEND_NAME, BACKEND_VERSION
from gpuq.backends.base import (
    BACKEND_MISSING,
    BACKEND_QUEUED,
    BACKEND_RUNNING,
    BackendJob,
    BackendUnavailable,
)
from gpuq.backends import dispatcher as dispatcher_mod
from gpuq.backends.queue_store import QueueStore, row_to_backend_job
from gpuq.config import Config
from gpuq.util import age_seconds, ensure_dir
from gpuq.winproc import (
    ExclusiveLock,
    detached_creationflags,
    is_locked,
    pid_matches,
    windowless_python,
)

_START_TIMEOUT_SECONDS = 20.0


class LocalDispatcherBackend:
    """Talks to the dispatcher daemon through the shared queue database."""

    name = BACKEND_NAME
    version = BACKEND_VERSION

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = QueueStore(config.backend_dir / "queue.sqlite3")
        self._initialized = False

    # -- lifecycle --------------------------------------------------------
    @property
    def lock_path(self) -> Path:
        return self.config.run_dir / "dispatcher.lock"

    def initialize(self) -> None:
        """Create state, apply config to the queue, and ensure a live daemon.

        Idempotent (spec section 8.3).
        """
        self.config.ensure_dirs()
        self.store.initialize()
        self.store.set_meta(dispatcher_mod.META_SHUTDOWN, "0")
        self.set_slots(self.config.core.max_concurrent_jobs)
        self.set_gpu_free_percent(self.config.gpu.free_memory_threshold_percent)
        self.store.set_meta(dispatcher_mod.META_LOGDIR, str(self.config.logs_dir))
        self.ensure_daemon()
        self._initialized = True

    def _ensure_store(self) -> None:
        if not self._initialized:
            ensure_dir(self.config.backend_dir)
            self.store.initialize()
            self._initialized = True

    def daemon_running(self) -> bool:
        """A daemon is live iff it holds the lock and its heartbeat is fresh."""
        if not is_locked(self.lock_path):
            return False
        return self.heartbeat_age() is not None and not self.heartbeat_stale()

    def heartbeat_age(self) -> float | None:
        self._ensure_store()
        return age_seconds(self.store.get_meta(dispatcher_mod.META_HEARTBEAT))

    def heartbeat_stale(self) -> bool:
        age = self.heartbeat_age()
        if age is None:
            return True
        return age > self.config.backend.daemon_heartbeat_stale_seconds

    def daemon_pid(self) -> int | None:
        self._ensure_store()
        pid = self.store.get_meta_int(dispatcher_mod.META_DAEMON_PID, 0)
        return pid or None

    def ensure_daemon(self, *, timeout: float = _START_TIMEOUT_SECONDS) -> bool:
        """Start the dispatcher if it is not already running. Returns liveness."""
        self._ensure_store()
        if self.daemon_running():
            return True
        self._spawn_daemon()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.daemon_running():
                return True
            time.sleep(0.1)
        return self.daemon_running()

    def _spawn_daemon(self) -> None:
        """Launch `gpuq _daemon` fully detached from the calling shell.

        Nothing is inherited from the submitting terminal, so closing that
        terminal cannot take the queue - or a running job - down with it.
        """
        self.config.ensure_dirs()
        # pythonw.exe leaves no console behind; see winproc.windowless_python.
        argv = [windowless_python(sys.executable), "-m", "gpuq", "_daemon"]
        stdout_path = self.config.run_dir / "dispatcher.out"
        env = {}
        import os

        env.update(os.environ)
        env["GPUQ_STATE_DIR"] = str(self.config.state_dir)
        if self.config.profile:
            env["GPUQ_PROFILE"] = self.config.profile
        # Pin the daemon to the exact config this process used, so a test
        # profile can never pick up the user's real configuration.
        if self.config.source_path:
            env["GPUQ_CONFIG_FILE"] = str(self.config.source_path)
        # The daemon must never inherit a blanked-out CUDA device list.
        if env.get("CUDA_VISIBLE_DEVICES", None) == "":
            env.pop("CUDA_VISIBLE_DEVICES", None)

        try:
            handle = open(stdout_path, "a", encoding="utf-8", errors="replace")
        except OSError:
            handle = None
        try:
            kwargs = {
                "cwd": str(self.config.state_dir),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": handle or subprocess.DEVNULL,
                "stderr": subprocess.STDOUT if handle else subprocess.DEVNULL,
                "close_fds": True,
            }
            flags = detached_creationflags()
            if flags:
                kwargs["creationflags"] = flags
            else:  # pragma: no cover - POSIX
                kwargs["start_new_session"] = True
            subprocess.Popen(argv, **kwargs)
        except OSError as exc:  # pragma: no cover
            raise BackendUnavailable(f"could not start the gpuq dispatcher: {exc}") from exc
        finally:
            if handle:
                handle.close()

    def shutdown(self, *, wait: bool = True, timeout: float = 15.0) -> bool:
        """Ask the dispatcher to stop. Running jobs are left alone."""
        self._ensure_store()
        self.store.set_meta(dispatcher_mod.META_SHUTDOWN, "1")
        if not wait:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not is_locked(self.lock_path):
                return True
            time.sleep(0.1)
        return not is_locked(self.lock_path)

    # -- health -----------------------------------------------------------
    def health(self) -> dict:
        self._ensure_store()
        pid = self.daemon_pid()
        meta = self.store.all_meta()
        alive = self.daemon_running()
        counts: dict[str, int] = {}
        try:
            for row in self.store.conn.execute(
                "SELECT state, COUNT(*) AS n FROM bjobs GROUP BY state"
            ):
                counts[row["state"]] = int(row["n"])
        except Exception:  # pragma: no cover
            pass
        return {
            "backend": self.name,
            "version": self.version,
            "daemon_running": alive,
            "daemon_pid": pid,
            "heartbeat_age_seconds": self.heartbeat_age(),
            "heartbeat_stale": self.heartbeat_stale(),
            "slots": self.get_slots(),
            "gpu_free_percent_threshold": self.get_gpu_free_percent(),
            "log_dir": meta.get(dispatcher_mod.META_LOGDIR),
            "queue_db": str(self.store.path),
            "lock": str(self.lock_path),
            "interpreter": meta.get(dispatcher_mod.META_INTERPRETER),
            "counts": counts,
            "supports_gpu_allocation": True,
            "supports_reorder": True,
            "supports_serialization": True,
        }

    def require_available(self) -> None:
        """Raise `BackendUnavailable` unless a dispatcher is live.

        Spec section 4.4: never silently run the command directly.
        """
        if self.ensure_daemon():
            return
        raise BackendUnavailable(
            "the gpuq dispatcher is not running and could not be started.\n"
            "Run 'gpuq doctor' for diagnostics, then 'gpuq init' to repair."
        )

    # -- settings ---------------------------------------------------------
    def set_slots(self, count: int) -> None:
        if count < 1:
            raise ValueError("slot count must be >= 1")
        self._ensure_store()
        self.store.set_meta(dispatcher_mod.META_SLOTS, int(count))

    def get_slots(self) -> int:
        self._ensure_store()
        return self.store.get_meta_int(
            dispatcher_mod.META_SLOTS, self.config.core.max_concurrent_jobs
        )

    def set_gpu_free_percent(self, percent: int) -> None:
        if not 0 <= percent <= 100:
            raise ValueError("gpu free percent must be between 0 and 100")
        self._ensure_store()
        self.store.set_meta(dispatcher_mod.META_GPU_FREE_PERC, int(percent))

    def get_gpu_free_percent(self) -> int:
        self._ensure_store()
        return self.store.get_meta_int(
            dispatcher_mod.META_GPU_FREE_PERC, self.config.gpu.free_memory_threshold_percent
        )

    def get_log_dir(self) -> str | None:
        self._ensure_store()
        return self.store.get_meta(dispatcher_mod.META_LOGDIR)

    # -- job operations ---------------------------------------------------
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
    ) -> int:
        """Enqueue an argv vector. No shell, no string concatenation."""
        argv = [str(a) for a in argv]
        if not argv:
            raise ValueError("cannot submit an empty command")
        self.require_available()
        log_path = str(self.config.logs_dir / log_name) if log_name else None
        if log_path:
            ensure_dir(Path(log_path).parent)
        return self.store.enqueue(
            argv,
            label=label,
            gpu_count=gpu_count,
            slots=slots,
            priority_rank=priority_rank,
            log_path=log_path,
            cwd=cwd,
            env=env,
            ram_mib=ram_mib,
            vram_mib=vram_mib,
            cpus=cpus,
        )

    def list_jobs(self, *, limit: int | None = None) -> list[BackendJob]:
        self._ensure_store()
        return [row_to_backend_job(r) for r in self.store.list_all(limit=limit)]

    def get_job(self, backend_id: int) -> BackendJob:
        self._ensure_store()
        row = self.store.get(backend_id)
        if row is None:
            return BackendJob(backend_id=backend_id, state=BACKEND_MISSING)
        return row_to_backend_job(row)

    def get_state(self, backend_id: int) -> str:
        return self.get_job(backend_id).state

    def output_path(self, backend_id: int) -> Path | None:
        return self.get_job(backend_id).output_path

    def find_by_label(self, label: str) -> BackendJob | None:
        self._ensure_store()
        row = self.store.get_by_label(label)
        return row_to_backend_job(row) if row else None

    def remove_queued(self, backend_id: int) -> None:
        self._ensure_store()
        if not self.store.remove_queued(backend_id):
            raise BackendUnavailable(
                f"backend job {backend_id} is not queued; it may already be running"
            )

    def terminate_running(self, backend_id: int, *, force: bool = False) -> None:
        """Request cancellation. The dispatcher performs the actual kill."""
        self._ensure_store()
        if not self.store.request_cancel(backend_id, force=force):
            raise BackendUnavailable(
                f"backend job {backend_id} is not active; nothing to terminate"
            )
        self.ensure_daemon()

    def cancel(self, backend_id: int, *, force: bool = False) -> str:
        """Cancel whatever state the job is in. Returns what was done."""
        self._ensure_store()
        row = self.store.get(backend_id)
        if row is None:
            return "missing"
        if row["state"] == BACKEND_QUEUED:
            if self.store.remove_queued(backend_id):
                return "removed"
            row = self.store.get(backend_id) or row
        if row["state"] == BACKEND_RUNNING:
            self.store.request_cancel(backend_id, force=force)
            self.ensure_daemon()
            return "terminating"
        return "finished"

    def wait_until_finished(self, backend_id: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.get_state(backend_id)
            if state not in (BACKEND_QUEUED, BACKEND_RUNNING):
                return True
            time.sleep(0.1)
        return False

    def set_priority(self, backend_id: int, priority_rank: int) -> None:
        """Change a queued job's dispatch rank (used by project policy)."""
        self._ensure_store()
        if not self.store.set_priority(backend_id, priority_rank):
            raise BackendUnavailable(
                f"backend job {backend_id} is not queued; only queued jobs can be re-ranked"
            )

    def promote(self, backend_id: int) -> None:
        self._ensure_store()
        if not self.store.promote(backend_id):
            raise BackendUnavailable(
                f"backend job {backend_id} is not queued; only queued jobs can be promoted"
            )

    def verify_job_pid(self, backend_id: int) -> bool:
        """True when the recorded PID is provably still this job's process."""
        row = self.store.get(backend_id)
        if not row or not row.get("pid"):
            return False
        return pid_matches(int(row["pid"]), row.get("pid_creation"))

    def close(self) -> None:
        self.store.close()


def build_backend(config: Config) -> LocalDispatcherBackend:
    """Backend factory. Add new backends here as they are implemented."""
    name = config.backend.name
    if name in (BACKEND_NAME, "local", "auto", "ts", "task_spooler"):
        # `task_spooler` is accepted so a spec-shaped config keeps working; on
        # this platform it resolves to the local dispatcher.
        return LocalDispatcherBackend(config)
    raise BackendUnavailable(f"unknown backend: {name!r}")


__all__ = ["LocalDispatcherBackend", "build_backend", "ExclusiveLock"]
