"""`workerq doctor` - health diagnostics (spec section 11.8).

Exit codes: 0 healthy, 1 warnings/degraded but usable, 2 broken/unsafe.
A failed check must never be silently downgraded (spec section 29.10).
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any, Callable

from workerq import __version__
from workerq.config import Config
from workerq.core import GPUQService
from workerq.gpu import cuda_toolkit_version, foreign_processes, nvidia_smi_path, query_gpus
from workerq.util import human_duration

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


class Doctor:
    def __init__(self, service: GPUQService) -> None:
        self.service = service
        self.config: Config = service.config
        self.checks: list[Check] = []

    # -- helpers ----------------------------------------------------------
    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> Check:
        check = Check(name, status, detail, hint)
        self.checks.append(check)
        return check

    def guard(self, name: str, fn: Callable[[], Check | None]) -> None:
        """Run one check; an unexpected exception becomes a FAIL, not a crash."""
        try:
            result = fn()
        except Exception as exc:
            self.add(name, FAIL, f"{type(exc).__name__}: {exc}")
            return
        if result is not None and result not in self.checks:
            self.checks.append(result)

    # -- checks -----------------------------------------------------------
    def check_python(self) -> None:
        version = sys.version_info
        detail = f"{platform.python_version()} ({sys.executable})"
        if version >= (3, 11):
            self.add("Python >= 3.11", PASS, detail)
        else:
            self.add(
                "Python >= 3.11",
                FAIL,
                detail,
                "worker-q requires Python 3.11 or newer for tomllib and typing features",
            )

    def check_platform(self) -> None:
        self.add(
            "Platform",
            PASS,
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        )

    def check_config(self) -> None:
        path = self.config.source_path
        if path is None:
            self.add("Config parse", WARN, "using built-in defaults (no config file)")
            return
        if not path.exists():
            self.add(
                "Config parse",
                WARN,
                f"{path} does not exist; built-in defaults in use",
                "run 'workerq init' to write a config file",
            )
            return
        try:
            self.config.validate()
        except Exception as exc:
            self.add("Config parse", FAIL, f"{path}: {exc}")
            return
        self.add("Config parse", PASS, str(path))

    def check_directories(self) -> None:
        failures: list[str] = []
        for directory in self.config.all_dirs():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                probe = directory / ".gpuq-write-probe"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                failures.append(f"{directory}: {exc}")
        if failures:
            self.add("State directories writable", FAIL, "; ".join(failures))
        else:
            self.add("State directories writable", PASS, str(self.config.state_dir))

    def check_sqlite(self) -> None:
        try:
            self.service.ensure_ready()
            version = self.service.db.schema_version()
            with self.service.db.transaction() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS _doctor_probe (id INTEGER PRIMARY KEY)"
                )
                conn.execute("INSERT INTO _doctor_probe(id) VALUES (1)")
                conn.execute("DELETE FROM _doctor_probe")
                conn.execute("DROP TABLE _doctor_probe")
            journal = self.service.db.conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.add(
                "SQLite",
                PASS,
                f"sqlite {sqlite3.sqlite_version}, schema v{version}, journal={journal}",
            )
        except Exception as exc:
            self.add(
                "SQLite",
                FAIL,
                f"{type(exc).__name__}: {exc}",
                f"database: {self.config.db_path}",
            )

    def check_backend(self) -> None:
        backend = self.service.backend
        try:
            health = backend.health()
        except Exception as exc:
            self.add("Dispatcher backend", FAIL, f"{type(exc).__name__}: {exc}")
            return

        self.add(
            f"Backend {health['backend']} v{health['version']}",
            PASS,
            f"queue db {health['queue_db']}",
        )

        if health["daemon_running"]:
            age = health.get("heartbeat_age_seconds")
            self.add(
                "Dispatcher daemon",
                PASS,
                f"pid {health['daemon_pid']}, heartbeat {age:.1f}s ago"
                if age is not None
                else f"pid {health['daemon_pid']}",
            )
        else:
            started = backend.ensure_daemon(timeout=15.0)
            if started:
                self.add("Dispatcher daemon", PASS, f"started (pid {backend.daemon_pid()})")
            else:
                self.add(
                    "Dispatcher daemon",
                    FAIL,
                    "not running and could not be started",
                    f"inspect {self.config.run_dir / 'dispatcher.out'} and run 'workerq init'",
                )
                return

        # Capability parity with the features the spec requires of a backend.
        capabilities = [
            ("GPU allocation support", health.get("supports_gpu_allocation")),
            ("Queue reordering support", health.get("supports_reorder")),
            ("Machine-readable job state", health.get("supports_serialization")),
        ]
        for name, ok in capabilities:
            self.add(name, PASS if ok else FAIL, "available" if ok else "missing")

        slots = health["slots"]
        configured = self.config.core.max_concurrent_jobs
        if slots == configured:
            self.add(
                f"Queue concurrency = {slots}",
                PASS if slots == 1 else WARN,
                "exclusive heavy-job execution"
                if slots == 1
                else f"{slots} concurrent jobs may exhaust VRAM",
            )
        else:
            self.add(
                "Queue concurrency",
                WARN,
                f"backend has {slots} slots, config says {configured}",
                "run 'workerq init' to re-apply the configured concurrency",
            )

        threshold = health["gpu_free_percent_threshold"]
        if threshold == self.config.gpu.free_memory_threshold_percent:
            self.add(f"GPU free threshold = {threshold}%", PASS, "applied to the dispatcher")
        else:
            self.add(
                "GPU free threshold",
                WARN,
                f"backend {threshold}%, config "
                f"{self.config.gpu.free_memory_threshold_percent}%",
                "run 'workerq init' to re-apply",
            )

        log_dir = health.get("log_dir")
        if log_dir and os.path.normcase(log_dir) == os.path.normcase(str(self.config.logs_dir)):
            self.add("Log directory", PASS, log_dir)
        else:
            self.add(
                "Log directory",
                WARN,
                f"backend reports {log_dir!r}, expected {self.config.logs_dir}",
                "run 'workerq init'",
            )

    def check_stale_daemon(self) -> None:
        """A lock held with no fresh heartbeat means a wedged dispatcher."""
        backend = self.service.backend
        from workerq.winproc import is_locked

        locked = is_locked(backend.lock_path)
        stale = backend.heartbeat_stale()
        if locked and stale:
            self.add(
                "No stale dispatcher",
                FAIL,
                f"lock held but heartbeat is {human_duration(backend.heartbeat_age())} old",
                f"kill pid {backend.daemon_pid()} then run 'workerq init'",
            )
        else:
            self.add("No stale dispatcher", PASS, "no conflicting dispatcher detected")

    def check_nvidia(self) -> None:
        smi = nvidia_smi_path()
        if smi is None:
            self.add(
                "nvidia-smi",
                WARN,
                "not found on PATH",
                "GPU jobs will still be serialised, but GPU gating and inventory are off",
            )
            return
        self.add("nvidia-smi", PASS, smi)

        info = query_gpus()
        if not info.available:
            self.add(
                "NVIDIA driver",
                FAIL,
                info.error or "driver did not respond",
                "GPU jobs cannot be gated safely until the driver responds",
            )
            return
        self.add(
            "NVIDIA driver",
            PASS,
            f"driver {info.driver_version or '?'}, CUDA {info.cuda_version or '?'}",
        )

        for device in info.devices:
            free = device.free_percent
            self.add(
                f"GPU {device.index} {device.name}",
                PASS,
                f"{(device.memory_used_mib or 0) / 1024:.1f} / "
                f"{(device.memory_total_mib or 0) / 1024:.1f} GiB used, "
                f"{free:.0f}% free" if free is not None else "memory unknown",
            )

        # Would the configured threshold currently block dispatch? A worker-q job
        # that is legitimately using the GPU is not a problem to report - the
        # next job was going to wait for the slot regardless.
        threshold = self.config.gpu.free_memory_threshold_percent
        best = info.max_free_percent()
        running = [
            job
            for job in self.service.db.list_jobs(states=["RUNNING"])
            if job.requested_gpu_count
        ]
        if best is None:
            pass
        elif running:
            self.add(
                "GPU meets free-memory threshold",
                PASS,
                f"{best:.0f}% free; the GPU is in use by worker-q job "
                f"#{running[0].id} ({running[0].project})",
            )
        elif best + 1e-9 < threshold:
            self.add(
                "GPU meets free-memory threshold",
                WARN,
                f"best GPU is {best:.0f}% free but the threshold is {threshold}%; "
                "GPU jobs will wait",
                f"lower it with 'workerq gpu-threshold {max(0, int(best) - 5)}' "
                "if this baseline usage is normal for this desktop",
            )
        else:
            self.add(
                "GPU meets free-memory threshold",
                PASS,
                f"{best:.0f}% free >= {threshold}% required",
            )

        # A desktop permanently has dozens of GUI processes touching the GPU
        # (explorer, browsers, Slack). Warning about those every single run
        # would make DEGRADED the normal state and teach the user to ignore
        # doctor. What is actually actionable is foreign work consuming enough
        # VRAM to block or endanger a job - so escalate on the memory, not on
        # the mere existence of other processes. `workerq gpu` always lists them.
        try:
            own_pids = self.service.own_pids()
        except Exception:
            own_pids = self.service.own_gpu_pids()
        foreign = foreign_processes(info, own_pids=own_pids)
        listed = ", ".join(
            f"{p.pid} {os.path.basename(p.process_name)}"
            f"{f' {p.used_memory_mib / 1024:.1f} GiB' if p.used_memory_mib else ''}"
            for p in foreign[:5]
        )
        if not foreign:
            self.add("Foreign GPU processes", PASS, "none detected")
        elif running:
            self.add(
                "Foreign GPU processes",
                PASS,
                f"{len(foreign)} other process(es) alongside worker-q job "
                f"#{running[0].id}",
            )
        elif best is not None and best + 1e-9 < threshold:
            self.add(
                "Foreign GPU processes",
                WARN,
                f"{len(foreign)} process(es) worker-q did not launch are holding the GPU "
                f"({best:.0f}% free): {listed}",
                "GPU jobs will wait until this frees up; 'workerq gpu' lists every process",
            )
        else:
            self.add(
                "Foreign GPU processes",
                PASS,
                f"{len(foreign)} other process(es) on the GPU, "
                f"but {best:.0f}% is free" if best is not None else f"{len(foreign)} other process(es)",
                "worker-q cannot control these; the free-memory threshold is the guard",
            )

    def check_host_resources(self) -> None:
        """Host RAM and commit charge - the limits that actually crash a box."""
        from workerq import host
        from workerq.resources import capacity

        mem = host.memory()
        if mem.error:
            self.add("Host memory", WARN, mem.error)
            return

        cap = capacity(self.config, gpu=None, mem=mem)
        self.add(
            "Host memory",
            PASS,
            f"{(mem.available_mib or 0) / 1024:.1f} GiB free of "
            f"{(mem.total_mib or 0) / 1024:.1f} GiB "
            f"({mem.free_percent or 0:.0f}%), {cap.usable_ram_mib / 1024:.0f} GiB usable",
        )

        r = self.config.resources
        if not r.enforce:
            self.add(
                "Resource admission control",
                WARN,
                "disabled (resources.enforce = false)",
                "jobs are gated only by slot count, so a heavy job can still "
                "start on an exhausted machine",
            )
        else:
            self.add(
                "Resource admission control",
                PASS,
                f"enforced - reserve {r.reserve_ram_gb:.0f} GiB RAM / {r.reserve_cpus} CPU, "
                f"floor {r.min_host_free_percent}% free, commit stop {r.max_commit_percent}%",
            )

        commit = mem.commit_percent
        if commit is None:
            return
        if commit >= r.max_commit_percent:
            # Degraded, not broken. worker-q is doing exactly its job by holding
            # work back, submitting is still safe (the job queues), and the
            # condition clears itself when the pressure does. Reserving FAIL
            # for "worker-q is broken" keeps the signal worth reading.
            self.add(
                "Commit charge",
                WARN,
                f"{commit:.0f}% of the limit, at or above the {r.max_commit_percent}% "
                "stop - new jobs will wait rather than start",
                "run 'workerq top' to see what is holding memory; jobs resume "
                "automatically once it frees up",
            )
        elif commit >= r.max_commit_percent - 8:
            self.add(
                "Commit charge",
                WARN,
                f"{commit:.0f}% of the limit, close to the {r.max_commit_percent}% stop",
                "run 'workerq top' to see what is holding memory",
            )
        else:
            self.add("Commit charge", PASS, f"{commit:.0f}% of the limit")

    def check_unqueued_heavy_work(self) -> None:
        """Large processes worker-q did not start are the usual cause of a crash."""
        from workerq import host

        try:
            own = self.service.own_pids()
        except Exception:
            own = set()
        offenders = [
            proc
            for proc in host.top_processes(12)
            if proc.pid not in own and proc.memory_gib >= 6.0
        ]
        if not offenders:
            self.add("Unqueued heavy workloads", PASS, "none holding 6+ GiB")
            return
        listed = ", ".join(f"{p.name} ({p.memory_gib:.1f} GiB, pid {p.pid})" for p in offenders[:4])
        self.add(
            "Unqueued heavy workloads",
            WARN,
            f"{len(offenders)} process(es) worker-q did not start: {listed}",
            "worker-q cannot schedule around these; submit that work through the "
            "queue so it is accounted for",
        )

    def check_cuda_toolkit(self) -> None:
        """The toolkit is informational here.

        It is only *required* when building a backend from source (the Linux
        Task Spooler path). This backend needs nothing but the NVIDIA runtime,
        so a missing nvcc is not a degradation and must not be reported as one.
        """
        version = cuda_toolkit_version()
        if version:
            self.add("CUDA toolkit (nvcc)", PASS, f"release {version}")
        else:
            self.add(
                "CUDA toolkit (nvcc)",
                PASS,
                "not installed (not required - worker-q uses only the NVIDIA runtime)",
            )

    def check_git(self) -> None:
        git = shutil.which("git")
        if git is None:
            self.add(
                "git (snapshot support)",
                FAIL,
                "not found on PATH",
                "source snapshots require git; submit with --live-worktree to bypass",
            )
            return
        import subprocess

        try:
            from workerq.winproc import no_window_kwargs

            proc = subprocess.run(
                [git, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                **no_window_kwargs(),
            )
            self.add("git (snapshot support)", PASS, (proc.stdout or "").strip() or git)
        except Exception as exc:  # pragma: no cover
            self.add("git (snapshot support)", WARN, f"{git}: {exc}")

    def check_claude_policy(self) -> None:
        from workerq.claude_policy import policy_status

        if not self.config.claude.install_user_policy:
            self.add("Claude policy", PASS, "disabled in config (claude.install_user_policy)")
            return
        status = policy_status()
        if status["installed"] and status["current"]:
            self.add("Claude policy installed", PASS, status["path"])
        elif status["installed"]:
            self.add(
                "Claude policy installed",
                WARN,
                f"{status['path']} contains an outdated worker-q block",
                "run 'workerq claude-policy install' to refresh it",
            )
        else:
            self.add(
                "Claude policy installed",
                WARN,
                f"not present in {status['path']}",
                "run 'workerq claude-policy install'",
            )

    def check_reconcile(self) -> None:
        """Read-only reconciliation: report drift without mutating anything."""
        try:
            drift = self.service.reconcile(mutate=False)
        except Exception as exc:
            self.add("Metadata consistent", WARN, f"{type(exc).__name__}: {exc}")
            return
        if drift:
            self.add(
                "Metadata consistent",
                WARN,
                f"{len(drift)} job(s) out of sync: " + "; ".join(drift[:3]),
                "run 'workerq reconcile' to repair",
            )
        else:
            self.add("Metadata consistent", PASS, "database agrees with the queue")

    def check_mcp(self) -> None:
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.add(
                "MCP adapter (optional)",
                PASS,
                "SDK not installed; the CLI is the supported interface",
                "install with: uv tool install --with 'mcp[cli]' worker-q",
            )
            return
        try:
            from workerq.mcp.server import build_server

            build_server(self.service.config)
            self.add("MCP adapter (optional)", PASS, "server imports and builds")
        except Exception as exc:
            self.add("MCP adapter (optional)", WARN, f"{type(exc).__name__}: {exc}")

    # -- run --------------------------------------------------------------
    def run(self) -> list[Check]:
        self.guard("Python", self.check_python)
        self.guard("Platform", self.check_platform)
        self.guard("Config", self.check_config)
        self.guard("Directories", self.check_directories)
        self.guard("SQLite", self.check_sqlite)
        self.guard("Backend", self.check_backend)
        self.guard("Stale daemon", self.check_stale_daemon)
        self.guard("NVIDIA", self.check_nvidia)
        self.guard("Host resources", self.check_host_resources)
        self.guard("Unqueued work", self.check_unqueued_heavy_work)
        self.guard("CUDA toolkit", self.check_cuda_toolkit)
        self.guard("git", self.check_git)
        self.guard("Claude policy", self.check_claude_policy)
        self.guard("Reconcile", self.check_reconcile)
        self.guard("MCP", self.check_mcp)
        return self.checks


def overall_status(checks: list[Check]) -> tuple[str, int]:
    worst = max((_SEVERITY[c.status] for c in checks), default=0)
    if worst >= 2:
        return "BROKEN", 2
    if worst == 1:
        return "DEGRADED", 1
    return "HEALTHY", 0


def run_doctor(service: GPUQService) -> tuple[list[Check], str, int]:
    checks = Doctor(service).run()
    label, code = overall_status(checks)
    return checks, label, code


def doctor_report(service: GPUQService) -> dict[str, Any]:
    checks, label, code = run_doctor(service)
    return {
        "gpuq_version": __version__,
        "overall": label,
        "exit_code": code,
        "checks": [c.to_dict() for c in checks],
    }
