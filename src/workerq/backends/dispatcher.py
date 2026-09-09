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
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from workerq.backends.base import BACKEND_QUEUED, BACKEND_RUNNING
from workerq.backends.queue_store import QueueStore
from workerq.config import Config, load_config
from workerq import host, resources as res
from workerq.gpu import query_gpus
from workerq.telemetry import (
    EVENT_BLOCKED,
    EVENT_DAEMON,
    EVENT_FINISHED,
    EVENT_PREEMPTED,
    EVENT_PRESSURE,
    EVENT_RESERVE,
    EVENT_STARTED,
    open_telemetry,
)
from workerq.util import age_seconds, ensure_dir, rotate_if_large, utcnow_iso
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

#: Prefix for the last report from each node, published into the queue meta
#: table. The dispatcher already polls every node every few seconds; anything
#: else that wants to know a node's state - `top`, `status`, `node list` -
#: reads it from here instead of opening its own SSH connection.
#:
#: This is the same mechanism the reserve and the slot count use, and for the
#: same reason: the queue database is the channel. A 1 Hz dashboard cannot
#: afford a 540 ms round trip on the render path, and two independent pollers
#: would double the traffic to say the same thing twice.
META_NODE_REPORT = "node_report:"
META_GPU_FREE_PERC = "gpu_free_perc"
#: Live reserve. Set through `workerq reserve` and re-read every tick, so the
#: owner can reclaim the machine without restarting the daemon.
META_RESERVE_RAM = "reserve_ram_mib"
META_RESERVE_VRAM = "reserve_vram_mib"
META_RESERVE_CPUS = "reserve_cpus"
META_RESERVE_LABEL = "reserve_label"
META_RESERVE_EXPIRES = "reserve_expires_at"
META_LOGDIR = "logdir"
META_DAEMON_PID = "daemon_pid"
META_DAEMON_PID_CREATION = "daemon_pid_creation"
META_HEARTBEAT = "heartbeat"
META_STARTED_AT = "daemon_started_at"
META_SHUTDOWN = "shutdown_requested"
META_VERSION = "daemon_version"
META_INTERPRETER = "interpreter"


def read_reserve(store: Any, config: Config) -> res.Reserve:
    """The live reserve, falling back to config for anything unset."""
    base = res.Reserve.from_config(config)

    def _num(key: str, fallback: float) -> float:
        raw = store.get_meta(key, "")
        if raw in (None, ""):
            return fallback
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    label = store.get_meta(META_RESERVE_LABEL, "") or None
    expires = store.get_meta(META_RESERVE_EXPIRES, "") or None
    return res.Reserve(
        ram_mib=_num(META_RESERVE_RAM, base.ram_mib),
        vram_mib=_num(META_RESERVE_VRAM, base.vram_mib),
        cpus=int(_num(META_RESERVE_CPUS, base.cpus)),
        label=label,
        expires_at=expires,
    )


def clear_reserve(store: Any) -> None:
    for key in (
        META_RESERVE_RAM,
        META_RESERVE_VRAM,
        META_RESERVE_CPUS,
        META_RESERVE_LABEL,
        META_RESERVE_EXPIRES,
    ):
        store.set_meta(key, "")


_GPU_CACHE_SECONDS = 3.0
_SAMPLE_INTERVAL_SECONDS = 10.0
#: How long after the runner's own grace period the dispatcher waits
#: before killing it. The runner needs this window to record why the job
#: stopped; killing it sooner loses the worker's only trace.
_PREEMPT_BACKSTOP_MARGIN_SECONDS = 20.0
#: Minimum gap between two log lines saying the same thing about the same job.
#: A blocked queue is polled several times a second; without this the log grows
#: by tens of megabytes a day and buries everything that matters.
_BLOCKED_REPEAT_SECONDS = 300.0
#: How long the main loop may go without completing a tick before the watchdog
#: concludes it is wedged and ends the process. Generous next to the 0.25s tick
#: and the 15s nvidia-smi timeout, so only a real hang trips it.
_WATCHDOG_GRACE_SECONDS = 180.0
#: Bytes written before the log size is re-checked, so rotation costs one stat
#: per megabyte rather than one per line.
_LOG_CHECK_BYTES = 1024 * 1024


def _reason_key(reason: str) -> str:
    """A wait reason with its numbers removed, for comparing one tick to the next.

    Reasons embed live measurements: "93.4 GiB of 93.6 GiB committed" differs
    from the same message one tick later purely because memory moved. Comparing
    the raw string therefore treats an unchanged condition as a new event every
    time, which is how a stalled queue produced 303,164 identical log lines.
    """
    return "".join("#" if c.isdigit() else c for c in reason)


@dataclass
class _RunningJob:
    backend_id: int
    proc: subprocess.Popen
    group: ProcessGroup
    log_handle: TextIO | None
    devices: list[int] = field(default_factory=list)
    cancel_signalled_at: float | None = None


#: Remote job states that mean "stop asking". Mirrors models.ALLOWED_TRANSITIONS
#: having no successors for these, and deliberately excludes QUEUED and RUNNING
#: - and anything unrecognised, so a state added by a newer worker-q is treated
#: as "still going" rather than silently finished.
_REMOTE_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "LOST"})


#: How often the node registry is re-read from the config file. Nodes are
#: added by hand, so this only has to be faster than a person gets impatient.
_NODE_RELOAD_SECONDS = 15.0

#: How often node reports are copied into the queue meta table for observers.
_NODE_PUBLISH_SECONDS = 2.0

#: How long a node's staging state is trusted. A whole SSH connection, on a
#: loop that ticks four times a second, to answer a question whose answer
#: changes only when a human runs `workerq node stage`.
_REPO_READY_SECONDS = 60.0

#: How far two machines' clocks may differ before dispatch stops. Generous,
#: because this is not about precision - it is about a clock that is wrong
#: enough to pick the wrong files when collecting a finished job's output.
_MAX_CLOCK_SKEW_SECONDS = 120.0


def _clock_skew_seconds(remote_time: str | None) -> float | None:
    """Remote clock minus ours, in seconds. None when it cannot be told."""
    if not remote_time:
        return None
    from workerq.util import parse_iso, utcnow

    parsed = parse_iso(remote_time)
    if parsed is None:
        return None
    return (parsed - utcnow()).total_seconds()


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
        #: backend_id -> (first_blocked_monotonic, reason_key, next_repeat_at),
        #: so a job that cannot be admitted is reported once rather than every
        #: tick. The key is the reason with its measurements stripped: the raw
        #: text carries a live figure that moves between ticks, and comparing
        #: that defeated the deduplication entirely.
        self._blocked: dict[int, tuple[float, str, float]] = {}
        #: head backend_id -> when the queue started being held for it, and when
        #: it may next be logged about. Bounding the hold is what stops one
        #: unstartable job stalling everything behind it indefinitely.
        self._hold_since: dict[int, float] = {}
        self._hold_logged_at: dict[int, float] = {}
        #: When the main loop last completed a tick, for the watchdog.
        self._last_tick = time.monotonic()
        self._log_path = config.state_dir / "run" / "dispatcher.log"
        self._log_written = _LOG_CHECK_BYTES  # force a size check on the first line
        #: Consecutive samples under the memory floor, for the pressure guard.
        self._pressure_strikes = 0
        self._stop = False
        #: Node reports, cached with their age. Built lazily: a single-machine
        #: install must not pay for multi-node machinery it never uses.
        self._reports: Any | None = None
        #: When the node registry was last re-read from the config file.
        self._nodes_loaded_at = 0.0
        self._reports_published_at = 0.0
        #: (node, repo) -> (checked_at, ready, reason). Staging state changes
        #: when somebody runs `node stage`, not on its own.
        self._repo_ready_cache: dict[tuple[str, str], tuple[float, bool, str | None]] = {}
        #: Last placement explanation per job, so the reason is logged when it
        #: changes rather than four times a second.
        self._placement_logged: dict[int, str] = {}

    # -- logging ----------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{utcnow_iso()} [dispatcher] {message}"
        try:
            ensure_dir(self._log_path.parent)
            self._log_written += len(line) + 1
            # Only stat occasionally: this is on the tick path.
            if self._log_written >= _LOG_CHECK_BYTES:
                self._log_written = 0
                rotate_if_large(self._log_path)
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

    def _device_occupancy(self) -> dict[int, dict[str, Any]]:
        """Per-device: who is on it, how much VRAM they declared, and the mode.

        Placement is per-device, so the accounting has to be too. The aggregate
        VRAM ledger in `resources.admit` sums every device's memory, which on a
        multi-GPU host would happily pass two jobs that cannot both fit on the
        one device they land on.
        """
        occupancy: dict[int, dict[str, Any]] = {}
        rows = {int(r["id"]): r for r in self.store.running()}
        for backend_id, job in self.running.items():
            row = rows.get(backend_id, {})
            mode = str(row.get("gpu_mode") or "exclusive")
            vram = float(row.get("vram_mib") or 0.0)
            for index in job.devices:
                entry = occupancy.setdefault(
                    index, {"jobs": [], "vram_mib": 0.0, "exclusive": False}
                )
                entry["jobs"].append(backend_id)
                entry["vram_mib"] += vram
                if mode != "shared":
                    entry["exclusive"] = True
        return occupancy

    def _allocate_devices(
        self, gpu_count: int, *, gpu_mode: str = "exclusive", vram_mib: float = 0.0
    ) -> tuple[list[int] | None, str | None]:
        """Pick devices for a job, honouring the free-memory threshold.

        Returns (devices, wait_reason). `devices` is None when the job must
        wait; an empty list means the job needs no GPU.

        A device already running a job is off limits unless both that job and
        this one asked to share it, and their declared VRAM fits the device
        together. VRAM has no swap and, on consumer cards in WDDM mode, no
        per-process accounting to check the guess against - so sharing is
        gated entirely on declarations, and a job that declares no VRAM is
        never packed onto an occupied device.
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
        reserve_vram = self._reserve().vram_mib
        occupancy = self._device_occupancy()
        wants_share = gpu_mode == "shared" and vram_mib > 0
        candidates: list[tuple[float, int]] = []
        blocked: list[str] = []

        for device in info.devices:
            held = occupancy.get(device.index)
            if held:
                # Occupied. Only a shared job may join a device whose current
                # occupants all agreed to share.
                if not wants_share or held["exclusive"]:
                    continue
                total = device.memory_total_mib or 0.0
                budget = max(0.0, total - reserve_vram)
                committed = held["vram_mib"]
                if committed + vram_mib > budget:
                    blocked.append(
                        f"GPU{device.index} shared: "
                        f"{(committed + vram_mib) / 1024:.1f} GiB declared exceeds "
                        f"{budget / 1024:.1f} GiB usable"
                    )
                    continue
                # Sharing is judged on declarations, not the live free-memory
                # threshold: the occupant's allocation is already counted.
                candidates.append((-2.0, device.index))
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

        # Freest first, and an empty device always beats sharing one.
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return [index for _, index in candidates[:gpu_count]], None

    # -- child environment -------------------------------------------------
    def _vram_baseline(self, row: dict[str, Any], devices: list[int]) -> float | None:
        """Card usage before this job starts, when it will own the card alone.

        Per-process VRAM is unreadable under WDDM, so the only honest way to
        attribute usage is to watch the device total rise while exactly one job
        is responsible for it. That holds when the job is exclusive-mode, has a
        single device, and nothing else worker-q started is on it. Any other
        arrangement would attribute somebody else's allocation to this job, so
        return None and leave the peak unrecorded rather than record a wrong one.
        """
        if len(devices) != 1:
            return None
        if str(row.get("gpu_mode") or "exclusive") != "exclusive":
            return None
        device = devices[0]
        occupancy = self._device_occupancy()
        if occupancy.get(device, {}).get("jobs"):
            return None
        info = self._gpu_info()
        if not getattr(info, "available", False):
            return None
        for dev in info.devices:
            if dev.index == device:
                return dev.memory_used_mib
        return None

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
            baseline = self._vram_baseline(row, devices)
            if baseline is not None:
                # What the card already held before this job existed. The runner
                # subtracts it to attribute the rise to this job, which is the
                # only way to get a per-job VRAM figure on a consumer card in
                # WDDM mode. Set only when the job owns the device alone, so the
                # rise cannot belong to somebody else.
                env["WORKERQ_VRAM_BASELINE_MIB"] = f"{baseline:.1f}"
                env["WORKERQ_VRAM_DEVICE"] = str(devices[0])
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

    def _reserve(self) -> res.Reserve:
        """The reserve in force this tick.

        Read fresh every time rather than cached at startup: the whole point
        is that `workerq reserve` takes effect while the daemon keeps running.
        An expired reserve is cleared here, so a temporary claim can never
        become a permanent one nobody remembers making.
        """
        reserve = read_reserve(self.store, self.config)
        if reserve.expires_at and reserve.is_expired:
            self.log(f"reserve '{reserve.label or 'custom'}' expired; restoring configured value")
            clear_reserve(self.store)
            self.telemetry.event(
                EVENT_RESERVE, detail=f"reserve '{reserve.label or 'custom'}' expired"
            )
            return res.Reserve.from_config(self.config)
        return reserve

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
            reserve=self._reserve(),
        )

    def _note_blocked(self, backend_id: int, reason: str) -> None:
        """Record a blocked job once, and escalate if it stays blocked."""
        now = time.monotonic()
        entry = self._blocked.get(backend_id)
        first = entry[0] if entry else now
        previous = entry[1] if entry else ""
        next_repeat = entry[2] if entry else 0.0
        key = _reason_key(reason)

        if previous == key:
            waited = now - first
            threshold = self.config.resources.blocked_warning_seconds
            if threshold and waited >= threshold and now >= next_repeat:
                self.log(
                    f"job {backend_id}: still blocked after {waited / 60:.0f}m - {reason}"
                )
                next_repeat = now + _BLOCKED_REPEAT_SECONDS
            self._blocked[backend_id] = (first, key, next_repeat)
            return

        self._blocked[backend_id] = (first, key, now + _BLOCKED_REPEAT_SECONDS)
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
                self.config,
                want,
                remaining,
                gpu=self._gpu_info(),
                mem=host.memory(),
                reserve=self._reserve(),
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
                running_job_ids=running_ids,
                queued_count=len(self.store.queued()),
                top_consumers=top,
            )
            self._relieve_pressure(mem)
        except Exception:
            pass

    # -- pressure guard -----------------------------------------------------
    def _relieve_pressure(self, mem: host.HostMemory) -> None:
        """Stop work when the machine is genuinely running out of memory.

        Admission control is a prediction made before a job starts. This is the
        backstop for when the prediction was wrong - a job that under-declared,
        a foreign workload, a game launched without claiming a reserve. Nothing
        else in worker-q can act on a machine that is already in trouble:
        preemption for memory pressure cannot help, because freeing a
        *reservation* does not change *measured* free memory until the victim
        actually exits.

        Two thresholds and a sample count, so a brief spike does not kill a
        long training run.
        """
        sched = self.config.scheduling
        free = mem.free_percent
        if free is None:
            return
        if free >= sched.pressure_recover_percent:
            if self._pressure_strikes:
                self.log(f"host memory recovered to {free:.0f}% free")
            self._pressure_strikes = 0
            return
        if free >= sched.pressure_free_percent:
            return  # between the two thresholds: hold, do not escalate or clear
        self._pressure_strikes += 1
        if self._pressure_strikes < sched.pressure_samples:
            return

        victim = self._pressure_victim()
        if victim is None:
            # Nothing safe to stop. Say so once per escalation rather than
            # silently doing nothing about a machine that is struggling.
            self.log(
                f"host memory at {free:.0f}% free and no preemptible job to stop; "
                "queue is holding but running work cannot be relieved"
            )
            self.telemetry.record_event(
                EVENT_PRESSURE,
                detail=f"{free:.0f}% RAM free, nothing preemptible to displace",
            )
            self._pressure_strikes = 0
            return

        victim_id = int(victim["id"])
        detail = (
            f"host memory at {free:.0f}% free "
            f"({(mem.available_mib or 0) / 1024:.1f} GiB); "
            f"stopping job {victim_id} to keep the machine usable"
        )
        if self.store.request_preempt(victim_id, by_backend_id=None):
            self.log(detail)
            self.telemetry.record_event(
                EVENT_PRESSURE, backend_job_id=victim_id, detail=detail
            )
        self._pressure_strikes = 0

    def _pressure_victim(self) -> dict[str, Any] | None:
        """The newest preemptible running job - the least work to lose.

        Only jobs that opted in are ever stopped. The contract is the same as
        priority preemption: being displaced means re-running from the start,
        so a job that did not declare itself safe to repeat is never chosen,
        however bad the pressure gets.
        """
        active = set(self.running) | set(self.adopted)
        candidates = [
            row
            for row in self.store.running()
            if int(row["id"]) in active
            and row.get("preemptible")
            and not row.get("preempt_requested")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda r: age_seconds(r.get("started_at")) or 0.0)
        return candidates[0]

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
            "creationflags": child_creationflags(
                background=self.config.scheduling.background_priority
            ),
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

    def _blocked_wait_seconds(self, backend_id: int) -> float:
        """How long this job has been continuously unable to start."""
        entry = self._blocked.get(backend_id)
        if entry is None:
            return 0.0
        return max(0.0, time.monotonic() - entry[0])

    def _record_wait(self, backend_id: int, reason: str | None) -> None:
        """Persist why a job is not running, so `status` can explain it.

        Every waiting job gets a reason, not just the one at the head. With
        several jobs running, "there is no reason recorded" is indistinguishable
        from "nobody has looked at it", and that is the difference between a
        queue you can reason about and one that looks stuck.
        """
        self.store.update(backend_id, wait_reason=reason)
        self._note_blocked(backend_id, reason or "blocked")

    def _start_ready_jobs(self) -> None:
        slots = max(1, self.store.get_meta_int(META_SLOTS, self.config.core.max_concurrent_jobs))
        # Adopted jobs occupy slots exactly like jobs we launched.
        in_flight = len(self.running) + len(self.adopted)
        queued = [row for row in self.store.queued() if not row.get("cancel_requested")]
        sched = self.config.scheduling
        now = time.monotonic()
        # A job that is no longer queued cannot be holding the queue.
        live = {int(row["id"]) for row in queued}
        for stale in [k for k in self._hold_since if k not in live]:
            self._hold_since.pop(stale, None)
            self._hold_logged_at.pop(stale, None)

        #: Set once a job has been passed over, so the job that caused it can be
        #: told apart from the ones merely behind it.
        head_blocked: dict[str, Any] | None = None
        skipped = 0

        for position, row in enumerate(queued):
            backend_id = int(row["id"])

            if in_flight >= slots:
                # Every slot is busy. Nothing further down can start either, so
                # say so for the rest of the queue rather than leaving it blank.
                reason = f"waiting for a free slot ({in_flight} of {slots} in use)"
                if head_blocked is None:
                    self._consider_preemption(row, reason)
                for rest in queued[position:]:
                    self._record_wait(int(rest["id"]), reason)
                return

            # A pinned job is judged against the machine it is pinned to,
            # not this one. Its resources are not ours to account for.
            pinned = row.get("pinned_node")
            if pinned:
                node = self.config.node(str(pinned))
                if node is None or not node.enabled:
                    self._record_wait(
                        backend_id,
                        f"pinned to {pinned}, which is not a registered, enabled node",
                    )
                    skipped += 1
                    if skipped > sched.backfill_max_skip:
                        return
                    continue
                ok, why = self._remote_admits(node, row)
                if ok and self._start_remote(row, node):
                    self._blocked.pop(backend_id, None)
                else:
                    self._record_wait(backend_id, why or f"waiting for {node.name}")
                    skipped += 1
                    if skipped > sched.backfill_max_skip:
                        return
                continue

            # Admission control: does this job's declared RAM/CPU/VRAM fit in
            # the headroom that is actually free, once running reservations and
            # foreign workloads are accounted for?
            devices: list[int] | None = []
            decision = self._admit(row)
            if not decision.admit:
                blocked_reason = decision.reason
            else:
                devices, blocked_reason = self._allocate_devices(
                    int(row.get("gpu_count") or 0),
                    gpu_mode=str(row.get("gpu_mode") or "exclusive"),
                    vram_mib=float(row.get("vram_mib") or 0.0),
                )
                if devices is not None:
                    blocked_reason = None

            # Placement. Asked whether or not this machine could take the job,
            # because "it fits here" is not the same as "here is the right
            # place for it" once a second machine exists.
            local_ok = blocked_reason is None
            chosen, remote_why = self._choose_node(row, queued, position, local_ok)
            if chosen is not None:
                if self._start_remote(row, chosen):
                    self._blocked.pop(backend_id, None)
                    continue
                # Placement failed after we decided to move it. Fall through
                # and let it run here if it can, rather than stalling.
                if local_ok:
                    pass
                else:
                    self._record_wait(backend_id, blocked_reason or "placement failed")
                    skipped += 1
                    if skipped > sched.backfill_max_skip:
                        return
                    continue

            if blocked_reason is not None:
                if remote_why:
                    blocked_reason = f"{blocked_reason} | {remote_why}"
                self._record_wait(backend_id, blocked_reason)
                if head_blocked is None:
                    head_blocked = row
                    self._consider_preemption(row, blocked_reason)

                # Backfill: a job this one cannot make room for may still fit
                # alongside what is running. Bounded two ways - how far we look,
                # and how long the blocked job has already waited - so a large
                # job cannot be deferred forever by a stream of small ones.
                if not sched.backfill:
                    return
                head_id = int(head_blocked["id"])
                waited = self._blocked_wait_seconds(head_id)
                # Holding is a bet that the machine will free up if we stop
                # adding to it. With nothing running there is nothing to drain,
                # so the bet cannot pay: the head is held out by the desktop,
                # the editors, or its own size, none of which the queue can
                # evict. Holding then buys the head nothing and costs every job
                # behind it - an idle machine with a 1 GiB job waiting on a
                # 32 GiB one that will not fit either way.
                if waited >= sched.backfill_head_wait_seconds and in_flight > 0:
                    # Holding drains the machine so the head job gets a clear
                    # run at it. That only works if queue pressure is what is
                    # keeping it out. When the blocker is a long-running job
                    # instead, holding forever stalls the whole queue and the
                    # head gains nothing: it cannot start until that job ends
                    # either way. So the hold is bounded, and once it expires
                    # work that fits is allowed through again.
                    started = self._hold_since.setdefault(head_id, now)
                    if now - started < sched.backfill_max_hold_seconds:
                        if now >= self._hold_logged_at.get(head_id, 0.0):
                            self.log(
                                f"job {head_id}: waited {waited:.0f}s; "
                                "holding the queue for it instead of backfilling"
                            )
                            self._hold_logged_at[head_id] = now + _BLOCKED_REPEAT_SECONDS
                        return
                    if now >= self._hold_logged_at.get(head_id, 0.0):
                        self.log(
                            f"job {head_id}: held the queue for "
                            f"{(now - started) / 60:.0f}m without starting; its "
                            "blocker is not queue pressure, so backfilling resumes"
                        )
                        self._hold_logged_at[head_id] = now + _BLOCKED_REPEAT_SECONDS
                elif in_flight == 0:
                    # Not holding, so the hold clock must not run. Otherwise an
                    # idle spell silently spends the head's one bounded hold and
                    # it never gets the drained machine the hold promises it.
                    self._hold_since.pop(head_id, None)
                skipped += 1
                if skipped > sched.backfill_max_skip:
                    return
                continue

            if self._start_job(row, devices or []):
                self._blocked.pop(backend_id, None)
                in_flight += 1



    # -- automatic placement ----------------------------------------------

    def _placement_note(self, backend_id: int, message: str) -> None:
        """Log a placement decision, once, and only when it changes.

        Every placement decision should be explainable after the fact - a job
        that ran on the slower machine, or did not, is otherwise impossible to
        argue about. Throttled by content, because this is on a loop that ticks
        four times a second and an unthrottled line here would reproduce the
        303,164-identical-lines incident that commit 16b2846 fixed.
        """
        if self._placement_logged.get(backend_id) == message:
            return
        self._placement_logged[backend_id] = message
        self.log(f"job {backend_id}: {message}")

    def _repo_ready(self, node: Any, spec: dict[str, Any]) -> tuple[bool, str | None]:
        """Does that node have this project's source and declared inputs?

        Cached: it is a whole SSH connection, staging state changes when
        somebody runs `node stage` and not otherwise, and this is on a loop
        that ticks four times a second.
        """
        from workerq import staging

        repo_root = spec.get("repo_root")
        if not repo_root:
            return False, "job has no repository"
        key = (node.name, str(repo_root))
        now = time.monotonic()
        cached = self._repo_ready_cache.get(key)
        if cached is not None and now - cached[0] < _REPO_READY_SECONDS:
            return cached[1], cached[2]

        try:
            status = staging.inspect_repo(
                node, Path(repo_root), list(spec.get("passthrough") or [])
            )
        except Exception as exc:
            answer = (False, f"could not check {node.name}: {exc}")
        else:
            if status.error:
                answer = (False, f"{node.name}: {status.error}")
            elif not status.exists:
                answer = (
                    False,
                    f"{node.name} has no clone of {status.project} "
                    f"(workerq node stage {node.name} --clone)",
                )
            elif status.missing:
                shown = ", ".join(status.missing[:3])
                more = f" and {len(status.missing) - 3} more" if len(status.missing) > 3 else ""
                answer = (False, f"{node.name} is missing {shown}{more}")
            else:
                answer = (True, None)

        self._repo_ready_cache[key] = (now, answer[0], answer[1])
        return answer

    def _would_block_the_queue(
        self, row: dict[str, Any], queued: list[dict[str, Any]], position: int
    ) -> bool:
        """Would running this job *here* keep a later one from starting?

        This is the whole of the placement rule, and it is a counterfactual
        rather than a queue-depth count. Depth is the wrong measure: a queue
        full of 30 GiB jobs is not a reason to exile a small one, because
        moving it frees nothing they can use.

        So the question asked is the one that matters - is there a job behind
        this one that cannot start now, but could if this one went elsewhere?
        If yes, moving this job buys an earlier start for that job. If no,
        moving it only makes this job slower on a slower machine.
        """
        mine = self._request_for(row)
        running = self._running_requests()
        with_me = running + [mine]
        for later in queued[position + 1:]:
            if later.get("pinned_node"):
                continue
            theirs = self._request_for(later)
            blocked_now = not res.admit(self.config, theirs, with_me, gpu=self._gpu_info()).admit
            free_if_moved = res.admit(self.config, theirs, running, gpu=self._gpu_info()).admit
            if blocked_now and free_if_moved:
                return True
        return False

    def _choose_node(
        self,
        row: dict[str, Any],
        queued: list[dict[str, Any]],
        position: int,
        local_ok: bool,
    ) -> tuple[Any | None, str | None]:
        """Pick a machine, or None to stay here.

        Contention, not fit. The worker runs the same job more slowly, so
        sending work there is a win only when it buys an earlier start - see
        docs/multi-node.md 5.4.
        """
        backend_id = int(row["id"])
        if not self.config.scheduling.auto_placement:
            return None, None
        raw = row.get("remote_spec_json")
        if not raw:
            self._placement_note(backend_id, "cannot travel: no remote spec")
            return None, None
        candidates = [n for n in self.config.nodes if n.enabled]
        if not candidates:
            return None, None
        try:
            spec = json.loads(raw)
        except (TypeError, ValueError):
            return None, None

        # If this job can start here and moving it frees nothing, keep it: the
        # local machine is faster and there is no transfer.
        if local_ok:
            if not self._would_block_the_queue(row, queued, position):
                self._placement_note(
                    backend_id, "keeping local: moving it would not free anything"
                )
                return None, None
            self._placement_note(
                backend_id, "it blocks a later job here, so looking for another machine"
            )

        reasons: list[str] = []
        for node in candidates:
            ready, why = self._repo_ready(node, spec)
            if not ready:
                reasons.append(why or f"{node.name} is not ready")
                continue
            admits, why = self._remote_admits(node, row)
            if not admits:
                reasons.append(why or f"{node.name} is busy")
                continue
            return node, None
        if reasons:
            self._placement_note(backend_id, "no node took it: " + "; ".join(reasons))
        return None, "; ".join(reasons) if reasons else None

    # -- remote placement -------------------------------------------------
    #
    # A remote job never enters `self.running` or `self.adopted`, so it is
    # already excluded from local slot counting and from `_running_requests`.
    # That is deliberate rather than incidental: its footprint is on the other
    # machine, and charging it against this one's headroom would idle the 5090
    # for work that is not here.

    def _publish_node_reports(self) -> None:
        """Write the latest node reports where anything else can read them.

        On the main loop, because the queue database connection belongs to this
        thread. The poller thread only refreshes the in-memory cache.
        """
        if self._reports is None or not self.config.nodes:
            return
        now = time.monotonic()
        if now - self._reports_published_at < _NODE_PUBLISH_SECONDS:
            return
        self._reports_published_at = now
        for node in self.config.nodes:
            cached = self._reports.peek(node.name)
            if cached is None:
                continue
            at, report = cached
            try:
                self.store.set_meta(
                    META_NODE_REPORT + node.name,
                    json.dumps({
                        "at": utcnow_iso(),
                        "age_at_publish": now - at,
                        **report.to_dict(),
                    }),
                )
            except Exception:
                pass

    def _refresh_nodes(self) -> None:
        """Re-read the node registry from disk.

        Nodes live in the config file, which the dispatcher reads once at
        start-up - so a node registered while it was running was invisible, and
        a job pinned to it waited forever against a message saying it was not
        registered. That is exactly the silent no-op the reserve and the slot
        count are re-read every tick to avoid.

        Cheap enough at this interval, and it only replaces the node list:
        everything else still needs a deliberate `workerq restart`, because
        changing it under a running scheduler is not obviously safe.
        """
        now = time.monotonic()
        if now - self._nodes_loaded_at < _NODE_RELOAD_SECONDS:
            return
        self._nodes_loaded_at = now
        try:
            fresh = load_config(self.config.source_path, profile=self.config.profile)
        except Exception as exc:
            self.log(f"could not re-read the node registry: {exc}")
            return
        before = {n.name for n in self.config.nodes}
        after = {n.name for n in fresh.nodes}
        if before != after:
            self.log(f"node registry changed: {sorted(before)} -> {sorted(after)}")
            if self._reports is not None:
                for gone in before - after:
                    self._reports.invalidate(gone)
        self.config.nodes = fresh.nodes

    def _node_report(self, node: Any) -> Any:
        from workerq import nodes as nodemod

        if self._reports is None:
            self._reports = nodemod.ReportCache()
        # Deliberately does not write to the store. This is called from the
        # node-poller thread, and a SQLite connection belongs to the thread
        # that created it - writing here raised, and the exception was
        # swallowed, so reports silently never appeared. Publishing happens on
        # the main loop instead, in `_publish_node_reports`.
        return self._reports.get(node)

    def _node_usable(self, node: Any, report: Any) -> tuple[bool, str | None]:
        """Is this node safe to send work to at all?

        Separate from "does the job fit there". A node can have ample room and
        still be the wrong place to send a job, and both of these are silent
        failures rather than loud ones:

        * **Protocol mismatch.** The wire format is one worker-q's JSON parsed
          by another. Sending a job across a version gap produces a parse error
          at the worst possible moment - after the work has been queued there.
          Only the protocol number may refuse; a differing `__version__` is
          reported and tolerated, because that check has already failed to
          notice a real skew.
        * **Clock skew.** Every timestamp that matters here crosses the link:
          a job's runtime, and the marker deciding which files a finished job
          wrote. A node whose clock is minutes off will silently collect the
          wrong files, so a large skew stops dispatch rather than corrupting
          results quietly.
        """
        from workerq import nodes as nodemod

        local = nodemod.local_report(self.config)
        ok, why = nodemod.compatibility(local, report)
        if not ok:
            return False, f"{node.name}: {why}"

        skew = _clock_skew_seconds(report.remote_time)
        if skew is not None and abs(skew) > _MAX_CLOCK_SKEW_SECONDS:
            return False, (
                f"{node.name}: its clock is {abs(skew):.0f}s "
                f"{'ahead of' if skew > 0 else 'behind'} this machine. "
                "Output collection compares file times against it, so results "
                "would be picked wrongly. Fix the clock (w32tm /resync)"
            )
        return True, None

    def _remote_admits(self, node: Any, row: dict[str, Any]) -> tuple[bool, str | None]:
        """Would that machine start this job right now?

        A prediction, made from a report up to `poll_interval_seconds` old and
        judged with the very same `resources.admit()` the node itself will
        re-run against live numbers before starting anything. So a stale
        prediction fails safe: the node refuses, and the job is placed again on
        a later tick rather than being wedged onto a machine that filled up.

        This is also what keeps the promise that nodes never hold a backlog -
        work is pushed only in the tick the node is expected to take it.
        """
        report = self._node_report(node)
        if not report.online:
            return False, f"node {node.name} is unreachable ({report.error})"
        usable, why = self._node_usable(node, report)
        if not usable:
            return False, why
        snapshot = report.snapshot(self.config)
        if snapshot is None:
            return False, f"node {node.name} reported no capacity"
        decision = res.admit(
            self.config,
            self._request_for(row),
            list(report.running),
            node=snapshot,
        )
        if decision.admit:
            return True, None
        return False, f"on {node.name}: {decision.reason}"

    def _start_remote(self, row: dict[str, Any], node: Any) -> bool:
        """Ship this job's source to `node` and queue it there."""
        from workerq import remote as remotemod
        from workerq import staging

        backend_id = int(row["id"])
        raw = row.get("remote_spec_json")
        if not raw:
            self._record_wait(
                backend_id,
                "cannot run on another machine: no git snapshot to ship "
                "(submitted --no-snapshot or --live-worktree)",
            )
            return False
        try:
            spec_data = json.loads(raw)
        except (TypeError, ValueError):
            self._record_wait(backend_id, "remote spec is unreadable")
            return False

        repo_root = Path(spec_data["repo_root"])
        origin_job_id = int(spec_data.get("origin_job_id") or backend_id)
        try:
            shipped = staging.ship_snapshot(
                node,
                repo_root,
                job_id=origin_job_id,
                commit=spec_data["snapshot_commit"],
                ref=spec_data.get("snapshot_ref"),
                passthrough=list(spec_data.get("passthrough") or []),
            )
            spec = remotemod.JobSpec(
                project=spec_data.get("project") or "unknown",
                argv=list(spec_data.get("argv") or []),
                cwd=shipped["worktree"],
                origin_job_id=origin_job_id,
                ram_gb=spec_data.get("ram_gb"),
                vram_gb=spec_data.get("vram_gb"),
                cpus=spec_data.get("cpus"),
                gpus=spec_data.get("gpus"),
                preemptible=spec_data.get("preemptible"),
                share_gpu=bool(spec_data.get("share_gpu")),
                priority=spec_data.get("priority"),
                shell=spec_data.get("shell"),
                describe=spec_data.get("describe"),
                blocks=spec_data.get("blocks"),
                eta_seconds=spec_data.get("eta_seconds"),
                env=dict(spec_data.get("env") or {}),
                passthrough=list(spec_data.get("passthrough") or []),
            )
            submitted = remotemod.submit(node, spec)
        except Exception as exc:
            # Placement failing is not the job failing. It goes back to the
            # queue with a reason and is tried again - possibly here.
            self._record_wait(backend_id, f"could not place on {node.name}: {exc}")
            self.log(f"job {backend_id}: placement on {node.name} failed: {exc}")
            return False

        remote_id = int(submitted["job_id"])
        if not self.store.claim_for_remote_start(backend_id, node.name, remote_id):
            # Something else claimed it first. Cancel what we just queued
            # rather than leaving an orphan running on the node.
            remotemod.cancel(node, remote_id, force=True)
            return False

        self.log(
            f"job {backend_id}: placed on {node.name} as its job {remote_id} "
            f"({shipped['bundle_bytes']} bytes shipped)"
        )
        self.telemetry.record_event(
            EVENT_STARTED,
            backend_job_id=backend_id,
            detail=f"remote:{node.name}",
            data={"node": node.name, "remote_id": remote_id,
                  "bundle_bytes": shipped["bundle_bytes"]},
        )
        return True

    def _reap_remote(self) -> None:
        """Ask each node how its jobs are doing.

        One call per node, not one per job: the connection is the cost. A node
        that cannot be reached leaves its jobs RUNNING - unreachable is not
        finished, and guessing otherwise is how a live training run gets
        recorded as failed and started again somewhere else.
        """
        from workerq import remote as remotemod

        rows = self.store.remote_running()
        if not rows:
            return
        by_node: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_node.setdefault(str(row["node"]), []).append(row)

        for node_name, node_rows in by_node.items():
            node = self.config.node(node_name)
            if node is None:
                continue
            try:
                listing = remotemod.list_jobs(node)
            except Exception as exc:
                self.log(f"node {node_name}: could not be polled ({exc})")
                continue
            for row in node_rows:
                backend_id = int(row["id"])
                remote_id = row.get("remote_id")
                entry = listing.get(int(remote_id)) if remote_id is not None else None
                if entry is None:
                    continue
                state = str(entry.get("state") or "")
                if state not in _REMOTE_TERMINAL:
                    continue
                code = entry.get("exit_code")
                if code is None:
                    code = 0 if state == "SUCCEEDED" else 1
                self._collect_remote_log(node, row, entry)
                collected = self._collect_remote_outputs(node, row)
                self.store.finish(backend_id, exit_code=int(code))
                if collected:
                    self.log(f"job {backend_id}: {collected}")
                self.log(f"job {backend_id}: finished on {node_name} exit={code}")
                self.telemetry.record_event(
                    EVENT_FINISHED,
                    backend_job_id=backend_id,
                    detail=f"remote:{node_name} exit {code}",
                    data={"exit_code": int(code), "node": node_name},
                )

    def _collect_remote_outputs(self, node: Any, row: dict[str, Any]) -> str | None:
        """Bring back what the job wrote, before it is recorded as finished.

        Order matters. A job marked SUCCEEDED whose results are still on the
        other machine is the failure this whole phase exists to prevent, so
        collection happens first and its outcome is logged either way - a
        silent partial success is worse than a loud one.
        """
        from workerq import staging

        raw = row.get("remote_spec_json")
        if not raw:
            return None
        try:
            spec = json.loads(raw)
        except (TypeError, ValueError):
            return None
        outputs = list(spec.get("outputs") or [])
        if not outputs:
            return None

        started = row.get("started_at")
        if not started:
            return None
        try:
            result = staging.collect_outputs(
                node,
                Path(spec["repo_root"]),
                outputs,
                job_id=int(spec.get("origin_job_id") or row["id"]),
                since_utc=str(started),
            )
        except Exception as exc:
            return f"could not collect results from {node.name}: {exc}"

        if result.get("error"):
            return (
                f"results are still on {node.name}: {result['error']}. "
                "They are not lost - fetch them by hand"
            )
        if not result.get("collected"):
            return f"declared outputs, but nothing was written on {node.name}"
        return (
            f"collected {result['collected']} output file(s) from {node.name} "
            f"({result['bytes']} bytes)"
        )

    def _collect_remote_log(self, node: Any, row: dict[str, Any], entry: dict[str, Any]) -> None:
        """Bring a finished remote job's log home.

        Best-effort by design: the log still exists on the node, so failing to
        fetch it must never stop the job being recorded as finished. But
        without this a job's output disappears the moment the worker is
        switched off, and `workerq logs` would have nothing to show.
        """
        from workerq import remote as remotemod

        local_path = row.get("log_path")
        remote_path = entry.get("log_path")
        if not local_path or not remote_path:
            return
        try:
            remotemod.fetch_log(node, str(remote_path), Path(str(local_path)))
        except Exception:
            pass

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
        self._reap_remote()

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
        self._start_watchdog()
        self._start_node_poller()
        try:
            while not self._stop:
                try:
                    self._last_tick = time.monotonic()
                    self._heartbeat()
                    self._sample_resources()
                    self._refresh_nodes()
                    self._publish_node_reports()
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


    def _start_node_poller(self) -> None:
        """Keep every node's report fresh, off the dispatch loop.

        On a thread rather than in the tick for two reasons. An SSH round trip
        costs about 540 ms on this pair and the tick runs four times a second,
        so polling inline would stall cancellation and reaping for a fifth of
        the time. And polling only when a job needs placing - which is what
        happened before this - meant an idle queue reported nothing at all, so
        `top` showed "no report yet" on a perfectly healthy machine, and the
        first job to arrive paid the cold poll itself.

        Failures are swallowed on purpose: a node that cannot be reached is a
        report with an error in it, not an exception that stops the loop.
        """
        import threading

        def poll() -> None:
            while not self._stop:
                try:
                    for node in list(self.config.nodes):
                        if self._stop:
                            break
                        if node.enabled:
                            self._node_report(node)
                except Exception:
                    pass
                # Sleep in short slices so shutdown is not held up by a node
                # with a long poll interval.
                waited = 0.0
                step = 0.25
                target = min(
                    (n.poll_interval_seconds for n in self.config.nodes if n.enabled),
                    default=5.0,
                )
                while waited < target and not self._stop:
                    time.sleep(step)
                    waited += step

        threading.Thread(target=poll, name="workerq-node-poll", daemon=True).start()

    def _start_watchdog(self) -> None:
        """End the process if the main loop stops making progress.

        The lock that keeps one dispatcher per profile is an OS file lock, so it
        is held for as long as this *process* lives, not for as long as the loop
        works. A wedged daemon therefore blocks its own replacement: `restart`
        and `init` both decline, and the queue stops scheduling until somebody
        kills the pid by hand. That is what happened when host commit ran out
        and a reader thread died mid-tick.

        Exiting hard is the right response. Jobs are adopted by the next
        dispatcher rather than killed, so the cost of being wrong is a restart,
        while the cost of hanging on is a queue that never recovers on its own.
        """

        def watch() -> None:
            while not self._stop:
                time.sleep(_WATCHDOG_GRACE_SECONDS / 6.0)
                stalled = time.monotonic() - self._last_tick
                if stalled < _WATCHDOG_GRACE_SECONDS:
                    continue
                try:
                    self.log(
                        f"watchdog: no tick for {stalled:.0f}s; exiting so a new "
                        "dispatcher can take the lock (running jobs are adopted)"
                    )
                except Exception:
                    pass  # under memory pressure logging is what fails first
                os._exit(1)

        threading.Thread(target=watch, name="workerq-watchdog", daemon=True).start()


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
