"""The worker-q dispatcher daemon.

A single detached process per worker-q profile. It is the only component that
launches user work, which is what makes the "one heavy job at a time"
invariant hold across unrelated terminals and agents.

Loop, once per tick:

1. publish a heartbeat so `doctor` can tell a live daemon from a stale one;
2. reap finished children and record exit codes;
3. service cancellation requests (polite signal, then a hard tree kill);
4. start queued jobs while a slot is free and the GPU is free enough.

The daemon holds an exclusive lock for its whole life, so a second one can
never start and double-dispatch the queue.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from workerq.backends.base import BACKEND_QUEUED, BACKEND_RUNNING
from workerq.backends.queue_store import QueueStore
from workerq.config import Config
from workerq import host, resources as res
from workerq.gpu import query_gpus
from workerq.telemetry import (
    EVENT_BLOCKED,
    EVENT_DAEMON,
    EVENT_FINISHED,
    EVENT_PREEMPTED,
    EVENT_STARTED,
    open_telemetry,
)
from workerq.util import age_seconds, ensure_dir, utcnow_iso
from workerq.winproc import (
    ProcessGroup,
    ExclusiveLock,
    child_creationflags,
    posix_child_kwargs,
    process_creation_time,
    terminate_tree,
)

# meta keys
META_SLOTS = "slots"
META_GPU_FREE_PERC = "gpu_free_perc"
META_LOGDIR = "logdir"
META_DAEMON_PID = "daemon_pid"
META_DAEMON_PID_CREATION = "daemon_pid_creation"
META_HEARTBEAT = "heartbeat"
META_STARTED_AT = "daemon_started_at"
META_SHUTDOWN = "shutdown_requested"
META_VERSION = "daemon_version"
META_INTERPRETER = "interpreter"

_GPU_CACHE_SECONDS = 3.0
_SAMPLE_INTERVAL_SECONDS = 10.0
#: How long after the runner's own grace period the dispatcher waits
#: before killing it. The runner needs this window to record why the job
#: stopped; killing it sooner loses the worker's only trace.
_PREEMPT_BACKSTOP_MARGIN_SECONDS = 20.0


@dataclass
class _RunningJob:
    backend_id: int
    proc: subprocess.Popen
    group: ProcessGroup
    log_handle: TextIO | None
    devices: list[int] = field(default_factory=list)
    cancel_signalled_at: float | None = None


class Dispatcher:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = QueueStore(config.backend_dir / "queue.sqlite3")
        self.running: dict[int, _RunningJob] = {}
        #: Jobs inherited from a previous dispatcher: id -> (pid, creation).
        self.adopted: dict[int, tuple[int, int]] = {}
        self._gpu_cache: tuple[float, Any] | None = None
        self.telemetry = open_telemetry(config.state_dir)
        self._last_sample = 0.0
        #: backend_id -> (first_blocked_monotonic, last_reason), so a job that
        #: cannot be admitted is reported once rather than every tick.
        self._blocked: dict[int, tuple[float, str]] = {}
        self._log_path = config.state_dir / "run" / "dispatcher.log"
        self._stop = False

    # -- logging ----------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{utcnow_iso()} [dispatcher] {message}"
        try:
            ensure_dir(self._log_path.parent)
            with open(self._log_path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        print(line, flush=True)

    # -- gpu --------------------------------------------------------------
    def _gpu_info(self) -> Any:
        now = time.monotonic()
        if self._gpu_cache and now - self._gpu_cache[0] < _GPU_CACHE_SECONDS:
            return self._gpu_cache[1]
        info = query_gpus(include_processes=False)
        self._gpu_cache = (now, info)
        return info

    def _devices_in_use(self) -> set[int]:
        used: set[int] = set()
        for job in self.running.values():
            used.update(job.devices)
        return used

    def _allocate_devices(self, gpu_count: int) -> tuple[list[int] | None, str | None]:
        """Pick devices for a job, honouring the free-memory threshold.

        Returns (devices, wait_reason). `devices` is None when the job must
        wait; an empty list means the job needs no GPU.
        """
        if gpu_count <= 0:
            return [], None

        info = self._gpu_info()
        if not info.available:
            # No usable NVIDIA stack. The queue still serialises work, so the
            # job runs; it simply gets no CUDA_VISIBLE_DEVICES assignment.
            return [], None

        threshold = self.store.get_meta_int(
            META_GPU_FREE_PERC, self.config.gpu.free_memory_threshold_percent
        )
        in_use = self._devices_in_use()
        candidates: list[tuple[float, int]] = []
        blocked: list[str] = []
        for device in info.devices:
            if device.index in in_use:
                continue
            free = device.free_percent
            if free is None:
                # Unknown free memory: allow, but never prefer.
                candidates.append((-1.0, device.index))
                continue
            if free + 1e-9 < threshold:
                blocked.append(f"GPU{device.index} {free:.0f}% free < {threshold}% required")
                continue
            candidates.append((free, device.index))

        if len(candidates) < gpu_count:
            if blocked:
                return None, "waiting for GPU memory: " + "; ".join(blocked)
            return None, f"waiting for {gpu_count} free GPU(s)"

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return [index for _, index in candidates[:gpu_count]], None

    # -- child environment -------------------------------------------------
    def _build_env(self, row: dict[str, Any], devices: list[int]) -> dict[str, str]:
        env = dict(os.environ)

        # Defence in depth (spec 15.3): the safe launcher may set
        # CUDA_VISIBLE_DEVICES="" for the agent's own shell. The dispatcher must
        # not pass that emptiness on to real GPU work.
        if env.get("CUDA_VISIBLE_DEVICES", None) == "":
            env.pop("CUDA_VISIBLE_DEVICES", None)

        # Logs are written and read as UTF-8. Without this a Python job that
        # prints non-ASCII dies with UnicodeEncodeError, because a file-backed
        # stdout on Windows defaults to the locale codec. Never overrides a
        # value the user set themselves.
        env.setdefault("PYTHONIOENCODING", "utf-8")

        try:
            job_env = json.loads(row.get("env_json") or "{}")
        except json.JSONDecodeError:
            job_env = {}
        for key, value in job_env.items():
            env[str(key)] = str(value)

        if devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        env["GPUQ_BACKEND_JOB_ID"] = str(row["id"])
        env["GPUQ_STATE_DIR"] = str(self.config.state_dir)
        if self.config.profile:
            env["GPUQ_PROFILE"] = self.config.profile
        return env


    # -- resource accounting -----------------------------------------------
    def _request_for(self, row: dict[str, Any]) -> res.ResourceRequest:
        """What this queued row is asking for, filling in configured defaults."""
        return res.ResourceRequest.from_job(
            self.config,
            ram_mib=row.get("ram_mib"),
            vram_mib=row.get("vram_mib"),
            cpus=row.get("cpus"),
            gpu_count=int(row.get("gpu_count") or 0),
        )

    def _running_requests(self) -> list[res.ResourceRequest]:
        """Reservations held by everything currently executing."""
        requests: list[res.ResourceRequest] = []
        active = set(self.running) | set(self.adopted)
        for row in self.store.running():
            if int(row["id"]) in active:
                requests.append(self._request_for(row))
        return requests

    def _admit(self, row: dict[str, Any]) -> res.Decision:
        return res.admit(
            self.config,
            self._request_for(row),
            self._running_requests(),
            gpu=self._gpu_info(),
            mem=host.memory(),
        )

    def _note_blocked(self, backend_id: int, reason: str) -> None:
        """Record a blocked job once, and escalate if it stays blocked."""
        now = time.monotonic()
        first, previous = self._blocked.get(backend_id, (now, ""))
        self._blocked[backend_id] = (first, reason)
        if previous == reason:
            waited = now - first
            threshold = self.config.resources.blocked_warning_seconds
            if threshold and waited >= threshold and int(waited) % 300 < 1:
                self.log(f"job {backend_id}: still blocked after {waited / 60:.0f}m - {reason}")
            return
        self.log(f"job {backend_id}: waiting - {reason}")
        self.telemetry.record_event(
            EVENT_BLOCKED, backend_job_id=backend_id, detail=reason
        )

    # -- preemption ---------------------------------------------------------
    def _preemption_candidates(self, waiter: dict[str, Any]) -> list[dict[str, Any]]:
        """Running jobs this waiter is allowed to displace, cheapest first.

        Every guard here exists to stop preemption destroying work for nothing:
        it must outrank the victim, the victim must have opted in, must have run
        long enough to be worth interrupting, and must not already have been
        displaced so often that it would starve.
        """
        cfg = self.config.preemption
        if not cfg.enabled:
            return []

        waiter_rank = int(waiter.get("priority_rank") or 100)
        candidates: list[dict[str, Any]] = []
        for row in self.store.running():
            backend_id = int(row["id"])
            if row.get("preempt_requested"):
                continue  # already stopping
            if int(row.get("priority_rank") or 100) <= waiter_rank:
                continue  # equal or higher priority is never displaced
            if cfg.require_opt_in and not row.get("preemptible"):
                continue
            started = row.get("started_at")
            ran_for = age_seconds(started) or 0.0
            if ran_for < cfg.min_runtime_seconds:
                continue
            if backend_id not in self.running and backend_id not in self.adopted:
                continue  # not ours to stop
            candidates.append(row)

        # Displace the least work: lowest priority first, then shortest running.
        candidates.sort(
            key=lambda r: (
                -int(r.get("priority_rank") or 100),
                -(age_seconds(r.get("started_at")) or 0.0),
            )
        )
        return candidates

    def _consider_preemption(self, waiter: dict[str, Any], reason: str | None) -> None:
        """Displace running work only if doing so actually unblocks `waiter`.

        Killing a job that does not free enough to let the waiter start would
        lose the victim's progress and leave the waiter blocked anyway, so the
        admission check is re-run against the reduced set of reservations before
        anything is stopped.
        """
        cfg = self.config.preemption
        if not cfg.enabled:
            return

        candidates = self._preemption_candidates(waiter)
        if not candidates:
            return

        slots = max(1, self.store.get_meta_int(META_SLOTS, self.config.core.max_concurrent_jobs))
        want = self._request_for(waiter)
        running_rows = {int(r["id"]): r for r in self.store.running()}
        in_flight = len(self.running) + len(self.adopted)

        chosen: list[dict[str, Any]] = []
        for victim in candidates:
            chosen.append(victim)
            remaining = [
                self._request_for(r)
                for bid, r in running_rows.items()
                if bid not in {int(c["id"]) for c in chosen}
            ]
            frees_a_slot = (in_flight - len(chosen)) < slots
            fits = res.admit(
                self.config, want, remaining, gpu=self._gpu_info(), mem=host.memory()
            ).admit
            if frees_a_slot and fits:
                break
        else:
            # Even displacing every candidate would not let the waiter run.
            return

        for victim in chosen:
            victim_id = int(victim["id"])
            if self.store.request_preempt(victim_id, by_backend_id=int(waiter["id"])):
                self.log(
                    f"job {victim_id}: preempted by job {waiter['id']} "
                    f"(rank {waiter.get('priority_rank')} beats {victim.get('priority_rank')}); "
                    f"{reason or 'higher priority'}"
                )
                self.telemetry.record_event(
                    EVENT_PREEMPTED,
                    backend_job_id=victim_id,
                    detail=f"displaced by backend job {waiter['id']}",
                    data={
                        "by_backend_job_id": int(waiter["id"]),
                        "waiter_rank": waiter.get("priority_rank"),
                        "victim_rank": victim.get("priority_rank"),
                        "reason": reason,
                    },
                )

    def _service_preemptions(self) -> None:
        """Backstop for a requested preemption.

        The runner owns the stop: it sees the flag, stops its child within the
        configured grace period, and records *why* it stopped so the worker can
        find its job again. Killing the runner would destroy exactly that
        record, so this only fires well after the runner's own deadline has
        passed - it is for a wedged runner, not the normal path.
        """
        cfg = self.config.preemption
        backstop = cfg.grace_seconds + _PREEMPT_BACKSTOP_MARGIN_SECONDS
        rows = self.store.conn.execute(
            "SELECT id, preempt_at, pid, pid_creation FROM bjobs "
            "WHERE COALESCE(preempt_requested, 0) = 1 AND state = ?",
            (BACKEND_RUNNING,),
        ).fetchall()
        for row in rows:
            backend_id = int(row["id"])
            waited = age_seconds(row["preempt_at"]) or 0.0
            if waited < backstop:
                continue  # let the runner stop cleanly and record the reason

            job = self.running.get(backend_id)
            if job is None:
                # Adopted from a previous dispatcher: no handle, verified kill.
                pid, creation = row["pid"], row["pid_creation"]
                if pid:
                    self.log(f"job {backend_id}: preemption backstop killing pid {pid}")
                    terminate_tree(int(pid), expected_creation=creation)
                continue

            self.log(
                f"job {backend_id}: runner did not stop within {backstop:.0f}s of "
                "preemption; killing its process tree"
            )
            job.group.terminate()
            terminate_tree(job.proc.pid)

    # -- telemetry ----------------------------------------------------------
    def _sample_resources(self) -> None:
        """Cheap periodic snapshot so failures can be explained afterwards."""
        now = time.monotonic()
        if now - self._last_sample < _SAMPLE_INTERVAL_SECONDS:
            return
        self._last_sample = now
        try:
            gpu = self._gpu_info()
            mem = host.memory()
            device = gpu.devices[0] if (gpu.available and gpu.devices) else None
            running_ids = sorted(set(self.running) | set(self.adopted))
            top = [
                {"pid": p.pid, "name": p.name, "mib": round(p.memory_mib, 1)}
                for p in host.top_processes(6)
            ]
            self.telemetry.record_sample(
                gpu_used_mib=device.memory_used_mib if device else None,
                gpu_total_mib=device.memory_total_mib if device else None,
                gpu_free_percent=device.free_percent if device else None,
                gpu_utilization=device.utilization_percent if device else None,
                host_total_mib=mem.total_mib,
                host_available_mib=mem.available_mib,
                host_free_percent=mem.free_percent,
                commit_used_mib=mem.commit_used_mib,
                commit_limit_mib=mem.commit_limit_mib,
                commit_percent=mem.commit_percent,
                running_job_id=running_ids[0] if running_ids else None,
                queued_count=len(self.store.queued()),
                top_consumers=top,
            )
        except Exception:
            pass

    # -- start ------------------------------------------------------------
    def _start_job(self, row: dict[str, Any], devices: list[int]) -> bool:
        backend_id = int(row["id"])
        try:
            argv = json.loads(row["argv_json"])
        except json.JSONDecodeError:
            self.log(f"job {backend_id}: corrupt argv, removing")
            self.store.finish(backend_id, exit_code=127)
            return False
        if not argv:
            self.store.finish(backend_id, exit_code=127)
            return False

        log_path = Path(row["log_path"]) if row.get("log_path") else None
        handle: TextIO | None = None
        if log_path is not None:
            ensure_dir(log_path.parent)
            handle = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)

        cwd = row.get("cwd") or None
        if cwd and not Path(cwd).is_dir():
            self.log(f"job {backend_id}: cwd missing ({cwd}); falling back to state dir")
            cwd = str(self.config.state_dir)

        env = self._build_env(row, devices)

        if not self.store.claim_for_start(backend_id):
            if handle:
                handle.close()
            return False

        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "stdin": subprocess.DEVNULL,
            "stdout": handle if handle else subprocess.DEVNULL,
            "stderr": subprocess.STDOUT if handle else subprocess.DEVNULL,
            "close_fds": True,
            "creationflags": child_creationflags(),
            **posix_child_kwargs(),
        }
        if not kwargs["creationflags"]:
            kwargs.pop("creationflags")

        try:
            proc = subprocess.Popen(argv, **kwargs)
        except OSError as exc:
            self.log(f"job {backend_id}: failed to start: {exc}")
            if handle:
                handle.write(f"worker-q: failed to start job: {exc}\n")
                handle.close()
            self.store.finish(backend_id, exit_code=127)
            return False

        # Record the PID before anything else. If the dispatcher died in this
        # window the job would be running with no recorded owner - unkillable
        # by `workerq cancel` and invisible to orphan recovery.
        self.store.update(
            backend_id,
            pid=proc.pid,
            pid_creation=process_creation_time(proc.pid),
            assigned_devices=",".join(str(d) for d in devices) if devices else None,
            wait_reason=None,
        )

        group = ProcessGroup(f"gpuq-{backend_id}")
        group.assign(proc.pid)
        self.running[backend_id] = _RunningJob(
            backend_id=backend_id, proc=proc, group=group, log_handle=handle, devices=devices
        )
        request = self._request_for(row)
        self.log(
            f"job {backend_id}: started pid={proc.pid} devices={devices or '-'} "
            f"ram={request.ram_mib / 1024:.1f}GiB cpus={request.cpus} cmd={argv[:4]}"
        )
        self.telemetry.record_event(
            EVENT_STARTED,
            backend_job_id=backend_id,
            detail=f"pid {proc.pid}",
            data={"devices": devices, "request": request.to_dict()},
        )
        return True

    def _start_ready_jobs(self) -> None:
        slots = max(1, self.store.get_meta_int(META_SLOTS, self.config.core.max_concurrent_jobs))
        # Adopted jobs occupy slots exactly like jobs we launched.
        in_flight = len(self.running) + len(self.adopted)
        queued = self.store.queued()
        first_blocked_reason: str | None = None

        for row in queued:
            if row.get("cancel_requested"):
                continue  # handled by the cancellation pass

            backend_id = int(row["id"])

            if in_flight >= slots:
                # No slot free. The highest-priority waiter still gets a chance
                # to displace something it outranks; if it cannot, nobody
                # further down the queue could either.
                self._consider_preemption(row, "waiting for a free slot")
                break

            # Admission control: does this job's declared RAM/CPU/VRAM fit in
            # the headroom that is actually free, once running reservations and
            # foreign workloads are accounted for?
            decision = self._admit(row)
            if not decision.admit:
                if first_blocked_reason is None:
                    first_blocked_reason = decision.reason
                    self.store.update(backend_id, wait_reason=decision.reason)
                    self._note_blocked(backend_id, decision.reason or "blocked")
                    self._consider_preemption(row, decision.reason)
                break

            devices, reason = self._allocate_devices(int(row.get("gpu_count") or 0))
            if devices is None:
                # Head-of-line blocking is intentional: a queued critical job
                # must not be overtaken just because the GPU is busy.
                if first_blocked_reason is None:
                    first_blocked_reason = reason
                    self.store.update(backend_id, wait_reason=reason)
                    self._note_blocked(backend_id, reason or "blocked")
                    self._consider_preemption(row, reason)
                break

            if self._start_job(row, devices):
                self._blocked.pop(backend_id, None)
                in_flight += 1

    # -- reap -------------------------------------------------------------
    def _reap(self) -> None:
        for backend_id in list(self.running):
            job = self.running[backend_id]
            code = job.proc.poll()
            if code is None:
                continue
            del self.running[backend_id]
            if job.log_handle:
                try:
                    job.log_handle.flush()
                    job.log_handle.close()
                except OSError:
                    pass
            job.group.close()

            # A displaced job goes back to the queue rather than being recorded
            # as finished: it did not fail, it was interrupted.
            row = self.store.get(backend_id) or {}
            if row.get("preempt_requested"):
                self.store.requeue(backend_id)
                self.log(f"job {backend_id}: requeued after preemption (exit={code})")
                self.telemetry.record_event(
                    EVENT_PREEMPTED, backend_job_id=backend_id,
                    detail="requeued", data={"exit_code": code},
                )
                continue

            self.store.finish(backend_id, exit_code=code)
            self.log(f"job {backend_id}: finished exit={code}")
            self.telemetry.record_event(
                EVENT_FINISHED, backend_job_id=backend_id, detail=f"exit {code}",
                data={"exit_code": code},
            )

        self._reap_adopted()

    def _reap_adopted(self) -> None:
        """Reap jobs inherited from a previous dispatcher.

        We have no `Popen` handle for these, so completion is detected by the
        process disappearing. Without this an adopted job would stay RUNNING in
        the queue forever and permanently consume a slot.
        """
        for backend_id, (pid, creation) in list(self.adopted.items()):
            if process_creation_time(pid) == creation:
                continue
            del self.adopted[backend_id]
            # The exit code is unknown here; the runner records the real
            # outcome in the worker-q database, and a terminal state there is
            # immutable, so reconciliation will not overwrite it.
            self.store.finish(backend_id, exit_code=None)
            self.log(f"job {backend_id}: adopted process {pid} exited")

    # -- cancel -----------------------------------------------------------
    def _service_cancellations(self) -> None:
        rows = self.store.conn.execute(
            "SELECT id, state, pid, pid_creation, cancel_force, cancel_at FROM bjobs "
            "WHERE cancel_requested = 1 AND state IN (?, ?)",
            (BACKEND_QUEUED, BACKEND_RUNNING),
        ).fetchall()
        grace = max(0, self.config.core.cancel_grace_seconds)

        for row in rows:
            backend_id = int(row["id"])
            if row["state"] == BACKEND_QUEUED:
                if self.store.remove_queued(backend_id):
                    self.log(f"job {backend_id}: removed while queued")
                continue

            job = self.running.get(backend_id)
            if job is None:
                # Running per the DB but not ours: the daemon restarted while
                # the job kept going. Kill by verified PID identity only.
                pid, creation = row["pid"], row["pid_creation"]
                if pid and terminate_tree(int(pid), expected_creation=creation):
                    self.log(f"job {backend_id}: terminated orphaned pid {pid}")
                    self.store.finish(backend_id, exit_code=-1)
                elif pid and process_creation_time(int(pid)) is None:
                    self.store.finish(backend_id, exit_code=-1)
                else:
                    self.log(
                        f"job {backend_id}: cannot verify pid {pid}; refusing to kill"
                    )
                continue

            force = bool(row["cancel_force"])
            if job.cancel_signalled_at is None:
                job.cancel_signalled_at = time.monotonic()
                if not force:
                    job.group.signal_break()
                    self.log(f"job {backend_id}: cancellation requested (graceful)")
                    continue

            elapsed = time.monotonic() - job.cancel_signalled_at
            if force or elapsed >= grace:
                job.group.terminate()
                terminate_tree(job.proc.pid)
                self.log(
                    f"job {backend_id}: process tree terminated "
                    f"({'forced' if force else f'grace {grace}s elapsed'})"
                )

    # -- heartbeat --------------------------------------------------------
    def _heartbeat(self) -> None:
        self.store.set_meta(META_HEARTBEAT, utcnow_iso())

    # -- main loop --------------------------------------------------------
    def run(self) -> int:
        from workerq import BACKEND_VERSION

        self.store.initialize()
        pid = os.getpid()
        self.store.set_meta(META_DAEMON_PID, pid)
        self.store.set_meta(META_DAEMON_PID_CREATION, process_creation_time(pid) or 0)
        self.store.set_meta(META_STARTED_AT, utcnow_iso())
        self.store.set_meta(META_VERSION, BACKEND_VERSION)
        self.store.set_meta(META_INTERPRETER, sys.executable)
        self.store.set_meta(META_SHUTDOWN, "0")
        self.store.set_meta(
            META_SLOTS,
            self.store.get_meta_int(META_SLOTS, self.config.core.max_concurrent_jobs),
        )
        self.store.set_meta(
            META_GPU_FREE_PERC,
            self.store.get_meta_int(
                META_GPU_FREE_PERC, self.config.gpu.free_memory_threshold_percent
            ),
        )
        self.store.set_meta(META_LOGDIR, str(self.config.logs_dir))
        self._heartbeat()
        self.log(f"dispatcher started pid={pid} state_dir={self.config.state_dir}")
        self.telemetry.record_event(EVENT_DAEMON, detail=f"started pid {pid}")

        self._recover_orphans()

        interval = self.config.backend.poll_interval_seconds
        trim_counter = 0
        try:
            while not self._stop:
                try:
                    self._heartbeat()
                    self._sample_resources()
                    self._reap()
                    self._service_cancellations()
                    self._service_preemptions()
                    self._start_ready_jobs()
                    trim_counter += 1
                    if trim_counter >= int(60 / max(interval, 0.05)):
                        trim_counter = 0
                        self.store.trim_finished(self.config.backend.max_finished)
                        self.telemetry.prune()
                    if self.store.get_meta(META_SHUTDOWN, "0") == "1":
                        self.log("shutdown requested")
                        break
                except Exception:  # keep the daemon alive through transient faults
                    self.log("tick error:\n" + traceback.format_exc())
                time.sleep(interval)
        finally:
            self.log("dispatcher stopping")
            for job in self.running.values():
                if job.log_handle:
                    try:
                        job.log_handle.close()
                    except OSError:
                        pass
                # Deliberately not killed: a training run must survive a
                # dispatcher restart.
                job.group.close()
            self.telemetry.record_event(EVENT_DAEMON, detail="stopped")
            self.telemetry.close()
            self.store.set_meta(META_DAEMON_PID, 0)
            self.store.set_meta(META_SHUTDOWN, "0")
            self.store.close()
        return 0

    def _recover_orphans(self) -> None:
        """Reconcile RUNNING rows left behind by a previous daemon.

        A job whose process is still alive is adopted rather than killed - a
        training run must survive a dispatcher restart - and then reaped by
        `_reap_adopted` when it eventually exits.
        """
        for row in self.store.running():
            backend_id = int(row["id"])
            pid, creation = row.get("pid"), row.get("pid_creation")
            if pid:
                actual = process_creation_time(int(pid))
                if actual is not None and (creation is None or actual == creation):
                    self.adopted[backend_id] = (int(pid), actual)
                    self.log(f"job {backend_id}: still running as pid {pid} (adopted)")
                    continue
            self.log(f"job {backend_id}: process gone, marking finished")
            self.store.finish(backend_id, exit_code=None)


def run_daemon(config: Config) -> int:
    """Entry point for `workerq _daemon`. Exits quietly if one already runs."""
    config.ensure_dirs()
    lock = ExclusiveLock(config.run_dir / "dispatcher.lock")
    if not lock.acquire():
        return 0  # another dispatcher owns this profile
    try:
        return Dispatcher(config).run()
    finally:
        lock.release()
