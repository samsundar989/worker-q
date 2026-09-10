"""Everything the web UI can ask for, as plain dictionaries.

Two rules hold this module together, and both come from mistakes the codebase
has already made once:

**A SQLite connection belongs to the thread that created it.** This broke the
dispatcher's node reporting and the runner's progress watcher, in both cases
silently. The server is threaded, so every request thread gets its own
`GPUQService` and its own telemetry handle, and none of them are shared.

**Reads go to the database; writes go through the service.** State transitions
are validated in `Database.update_job`, and cancel, promote and reserve are
meta-flag protocols the daemon polls for. Raw SQL from here could resurrect a
cancelled job. Writes are also serialised behind one lock, because a second
writer gets five seconds of `busy_timeout` and then `SQLITE_BUSY`.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from workerq import __version__, host, usage
from workerq.config import Config
from workerq.core import GPUQService
from workerq.telemetry import LIFECYCLE_EVENT_KINDS, open_telemetry

#: How many jobs one history page may ask for. A page is a scroll, not a dump.
MAX_PAGE = 500

#: How much of a log to hand over in one response.
MAX_LOG_BYTES = 512 * 1024


class Api:
    """Request-thread-safe access to one worker-q state directory."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._local = threading.local()
        # One writer at a time. Concurrent writers do not corrupt anything -
        # every mutation is a conditional UPDATE in an IMMEDIATE transaction -
        # but they do time out, and a timeout surfaces as a failed action.
        self._write_lock = threading.Lock()

    # -- per-thread handles ------------------------------------------------
    @property
    def service(self) -> GPUQService:
        svc = getattr(self._local, "service", None)
        if svc is None:
            svc = GPUQService(self.config)
            svc.ensure_ready()
            self._local.service = svc
        return svc

    @property
    def telemetry(self) -> Any:
        store = getattr(self._local, "telemetry", None)
        if store is None:
            store = open_telemetry(self.config.state_dir)
            self._local.telemetry = store
        return store

    def close_thread(self) -> None:
        """Drop this thread's handles. Called when a worker thread retires."""
        for name in ("service", "telemetry"):
            handle = getattr(self._local, name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self._local, name, None)

    # -- live view ---------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        """The frame the TUI draws, as data.

        `Dashboard.render` composes exactly this from five service calls, so
        the shape is not invented here - it is the one the TUI has been
        proving correct for as long as it has existed.
        """
        from workerq import nodes as nodes_mod

        service = self.service
        jobs = service.sort_for_display(service.list_jobs(limit=200))
        forecast = _safe(service.forecast, {}, jobs)
        summary = _safe(service.status_summary, {})
        reserve = _safe(service.backend.get_reserve, {})
        gpu = _safe(lambda: service.gpu_info().to_dict(), {})
        memory = _safe(lambda: host.memory().to_dict(), {})
        own = _safe(service.own_pids, set())

        # Never SSH on a request path: a round trip is about 540 ms, and the
        # dispatcher already publishes these into the queue meta table.
        node_reports = _safe(
            lambda: [r.to_dict() for r in
                     nodes_mod.published_reports(
                         self.config, service.backend.store).values()],
            [],
        )

        rows = []
        for job in jobs:
            entry = job.to_dict()
            entry["node"] = usage.node_label(job.node)
            entry["estimate"] = forecast.get(job.id)
            if job.state == "QUEUED":
                entry["wait_reason"] = _safe(service.queue_wait_reason, None, job)
            rows.append(entry)

        return {
            "version": __version__,
            "jobs": rows,
            "summary": summary,
            "gpu": gpu,
            "host": memory,
            "reserve": reserve,
            "nodes": node_reports,
            "throughput": _safe(service.throughput, {}),
            "processes": [
                {**p.to_dict(), "ours": p.pid in own}
                for p in _safe(host.top_processes, [], 10)
            ],
        }

    # -- history -----------------------------------------------------------
    def jobs(self, query: dict[str, Any]) -> dict[str, Any]:
        service = self.service
        limit = min(int(query.get("limit") or 50), MAX_PAGE)
        offset = max(int(query.get("offset") or 0), 0)
        rows, total = service.db.query_jobs(
            states=query.get("state"),
            projects=query.get("project"),
            nodes=query.get("node"),
            priorities=query.get("priority"),
            signature=query.get("signature"),
            since=query.get("since"),
            until=query.get("until"),
            search=query.get("search"),
            sort=str(query.get("sort") or "id"),
            descending=str(query.get("dir") or "desc") != "asc",
            limit=limit,
            offset=offset,
        )
        ram_ceiling, vram_ceiling = _safe(service._suggestion_ceiling, (None, None))
        out = []
        for job in rows:
            entry = job.to_dict()
            entry["node"] = usage.node_label(job.node)
            entry["usage"] = usage.for_job(
                job, ram_ceiling=ram_ceiling, vram_ceiling=vram_ceiling
            ).to_dict()
            out.append(entry)
        return {
            "jobs": out,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def facets(self) -> dict[str, Any]:
        db = self.service.db
        return {
            "project": db.distinct_values("project"),
            "state": db.distinct_values("state"),
            "priority": db.distinct_values("priority"),
            "node": db.distinct_values("node"),
        }

    def job(self, job_id: int) -> dict[str, Any]:
        service = self.service
        detail = service.job_detail(job_id)
        job = service.get_job(job_id)
        ram_ceiling, vram_ceiling = _safe(service._suggestion_ceiling, (None, None))
        detail["node"] = usage.node_label(job.node)
        detail["usage"] = usage.for_job(
            job, ram_ceiling=ram_ceiling, vram_ceiling=vram_ceiling
        ).to_dict()
        detail["events"] = [
            _decode_event(e)
            for e in self.telemetry.events_for_job(job.id, job.backend_job_id)
        ]
        detail["siblings"] = self._siblings(job)
        detail["suggestion"] = _safe(service.suggest_requests, None, job_id)
        return detail

    def _siblings(self, job: Any) -> list[dict[str, Any]]:
        """Other runs of the same command.

        One job being wrong is a typo. The same command being wrong twenty
        times running is a default worth changing at the source, and that is
        only visible next to its own history.
        """
        if not job.command_signature:
            return []
        rows, _ = self.service.db.query_jobs(
            signature=job.command_signature, limit=40
        )
        out = []
        for other in rows:
            u = usage.for_job(other)
            out.append(
                {
                    "id": other.id,
                    "state": other.state,
                    "node": usage.node_label(other.node),
                    "queued_at": other.queued_at,
                    "runtime_seconds": other.runtime_seconds,
                    "wait_seconds": other.wait_seconds,
                    "peak_commit_mib": u.peak_commit_mib,
                    "commit_budget_mib": u.commit_budget_mib,
                    "commit_ratio": u.commit_ratio,
                    "verdict": u.verdict,
                    "samples": u.samples,
                    "is_self": other.id == job.id,
                }
            )
        return out

    def job_series(self, job_id: int) -> dict[str, Any]:
        job = self.service.get_job(job_id)
        samples = self.telemetry.job_series(job.id)
        for row in samples:
            row.pop("top_consumers_json", None)
        return {
            "job_id": job.id,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "declared_ram_mib": job.requested_ram_mib,
            "declared_vram_mib": job.requested_vram_mib,
            "peak_commit_mib": job.peak_ram_mib,
            "peak_vram_mib": job.peak_vram_mib,
            "samples": samples,
            # Said out loud rather than implied: these are machine-wide
            # figures for the window the job ran in, not the job's own usage.
            # Per-job attribution is the peak columns, sampled by the runner.
            "scope": "machine",
        }

    def job_log(self, job_id: int, offset: int = 0) -> dict[str, Any]:
        job = self.service.get_job(job_id)
        path = self.service.resolve_log_path(job)
        if path is None or not Path(path).exists():
            return {"job_id": job.id, "offset": 0, "text": "", "eof": True,
                    "missing": True}
        size = Path(path).stat().st_size
        start = max(0, min(int(offset), size))
        if size - start > MAX_LOG_BYTES:
            # Jump to the tail rather than paging megabytes nobody reads.
            start = size - MAX_LOG_BYTES
        with open(path, "rb") as fh:
            fh.seek(start)
            chunk = fh.read(MAX_LOG_BYTES)
        return {
            "job_id": job.id,
            "offset": start + len(chunk),
            "size": size,
            "text": chunk.decode("utf-8", errors="replace"),
            "eof": job.is_terminal,
            "missing": False,
        }

    # -- accuracy ----------------------------------------------------------
    def accuracy(self, query: dict[str, Any]) -> dict[str, Any]:
        service = self.service
        limit = min(int(query.get("limit") or 400), 2000)
        rows, total = service.db.query_jobs(
            projects=query.get("project"),
            nodes=query.get("node"),
            signature=query.get("signature"),
            since=query.get("since"),
            limit=limit,
        )
        ram_ceiling, vram_ceiling = _safe(service._suggestion_ceiling, (None, None))
        measured = [
            usage.for_job(j, ram_ceiling=ram_ceiling, vram_ceiling=vram_ceiling)
            for j in rows
            if j.is_terminal
        ]
        summary = usage.summarise(measured)
        return {
            "considered": total,
            "rows": [u.to_dict() for u in measured],
            "summary": summary.to_dict(),
            "by_project": usage.group_by(measured, "project"),
            "by_signature": usage.group_by(measured, "command_signature"),
            "by_node": usage.group_by(measured, "node"),
        }

    # -- machines ----------------------------------------------------------
    def machines(self, query: dict[str, Any]) -> dict[str, Any]:
        """Per-machine throughput, and the honest test of placement.

        `_choose_node` only moves work when moving it lets a *different* job
        start sooner. Whether that is actually happening is not visible from
        any single job, only from how often one machine sat idle while the
        other had a queue.
        """
        from workerq import nodes as nodes_mod

        service = self.service
        limit = min(int(query.get("limit") or 1000), 5000)
        rows, _ = service.db.query_jobs(limit=limit)
        buckets: dict[str, dict[str, Any]] = {}
        for job in rows:
            name = usage.node_label(job.node)
            bucket = buckets.setdefault(
                name,
                {"node": name, "jobs": 0, "succeeded": 0, "failed": 0,
                 "cancelled": 0, "runtimes": [], "waits": [], "measured": 0,
                 "busy_seconds": 0.0},
            )
            bucket["jobs"] += 1
            if job.state == "SUCCEEDED":
                bucket["succeeded"] += 1
            elif job.state == "FAILED":
                bucket["failed"] += 1
            elif job.state == "CANCELLED":
                bucket["cancelled"] += 1
            if job.runtime_seconds:
                bucket["runtimes"].append(job.runtime_seconds)
                bucket["busy_seconds"] += job.runtime_seconds
            if job.wait_seconds is not None:
                bucket["waits"].append(job.wait_seconds)
            if job.peak_ram_mib is not None:
                bucket["measured"] += 1

        out = []
        for bucket in buckets.values():
            runtimes = sorted(bucket.pop("runtimes"))
            waits = sorted(bucket.pop("waits"))
            finished = bucket["succeeded"] + bucket["failed"]
            out.append(
                {
                    **bucket,
                    "median_runtime_seconds": _median(runtimes),
                    "median_wait_seconds": _median(waits),
                    "success_rate": (
                        bucket["succeeded"] / finished if finished else None
                    ),
                }
            )
        out.sort(key=lambda b: b["jobs"], reverse=True)
        for bucket in out:
            bucket["idle_while_queued"] = self._idle_while_queued([
                (j.started_at, j.finished_at)
                for j in rows
                if usage.node_label(j.node) == bucket["node"]
            ])
        reports = _safe(
            lambda: [r.to_dict() for r in
                     nodes_mod.published_reports(
                         self.config, service.backend.store).values()],
            [],
        )
        return {"machines": out, "reports": reports}

    def _idle_while_queued(self, intervals: list[tuple[str, str]]) -> dict[str, Any]:
        """How often this machine sat idle while the queue had work waiting.

        The honest test of `_choose_node`. Placement only pays when moving a
        job lets a *different* job start sooner, so a machine that is idle
        through the queue's busiest samples is not earning its place - and no
        single job's record can show that.

        Measured against the jobs' own start/finish times, deliberately not
        against `job_samples`. That table is written by the local dispatcher
        for jobs it is running, and a remote job never enters `self.running` -
        so a `job_samples` join would report every remote machine as idle 100%
        of the time, which is a number that looks like a finding and is only a
        bug.
        """
        conn = self.telemetry.conn
        try:
            queued = int(conn.execute(
                "SELECT COUNT(*) FROM samples WHERE queued_count > 0"
            ).fetchone()[0])
        except Exception:
            return {"queued_samples": 0, "idle_samples": 0, "fraction": None}
        if not queued:
            return {"queued_samples": 0, "idle_samples": 0, "fraction": None}
        # Newest first: those are the ones overlapping the retained telemetry
        # window, and SQLite takes at most 999 bound variables.
        windows = [w for w in intervals if w[0] and w[1]][:400]
        if not windows:
            return {"queued_samples": queued, "idle_samples": queued, "fraction": 1.0}
        clause = " OR ".join(["(at >= ? AND at <= ?)"] * len(windows))
        params: list[Any] = []
        for lo, hi in windows:
            params.extend([lo, hi])
        try:
            busy = int(conn.execute(
                f"SELECT COUNT(*) FROM samples WHERE queued_count > 0 AND ({clause})",
                params,
            ).fetchone()[0])
        except Exception:
            return {"queued_samples": queued, "idle_samples": 0, "fraction": None}
        idle = max(0, queued - busy)
        return {
            "queued_samples": queued,
            "idle_samples": idle,
            "fraction": idle / queued,
        }

    # -- efficiency --------------------------------------------------------
    def efficiency(self, query: dict[str, Any]) -> dict[str, Any]:
        """Where the queue's time actually goes.

        The interesting number is not how long jobs take, it is how long they
        wait - and what they were waiting for. A queue that waits longer than
        it computes is telling you the declarations are the bottleneck, not
        the code.
        """
        service = self.service
        limit = min(int(query.get("limit") or 1000), 5000)
        rows, _ = service.db.query_jobs(limit=limit)

        by_project: dict[str, dict[str, Any]] = {}
        for job in rows:
            bucket = by_project.setdefault(
                job.project, {"project": job.project, "runs": 0,
                              "wait_seconds": 0.0, "run_seconds": 0.0}
            )
            bucket["runs"] += 1
            bucket["wait_seconds"] += job.wait_seconds or 0.0
            bucket["run_seconds"] += job.runtime_seconds or 0.0
        for bucket in by_project.values():
            total = bucket["wait_seconds"] + bucket["run_seconds"]
            bucket["wait_fraction"] = (
                bucket["wait_seconds"] / total if total else None
            )

        blocked = self.telemetry.recent_events(limit=4000, kinds=["job_blocked"])
        reasons: dict[str, int] = {}
        for event in blocked:
            for segment in _block_reasons(event["detail"]):
                reasons[segment] = reasons.get(segment, 0) + 1

        lifecycle = [
            _decode_event(e)
            for e in self.telemetry.recent_events(
                limit=200, kinds=list(LIFECYCLE_EVENT_KINDS)
            )
        ]
        preemptions = [e for e in lifecycle if e["kind"] == "job_preempted"]
        return {
            "by_project": sorted(
                by_project.values(), key=lambda b: b["wait_seconds"], reverse=True
            ),
            "block_reasons": sorted(
                ({"reason": k, "count": v} for k, v in reasons.items()),
                key=lambda r: r["count"],
                reverse=True,
            )[:25],
            "preemptions": preemptions[:50],
            "events": lifecycle[:100],
            "throughput": _safe(lambda: service.throughput(hours=24.0), {}),
        }

    def events(self, query: dict[str, Any]) -> dict[str, Any]:
        kinds = query.get("kind") or None
        limit = min(int(query.get("limit") or 200), 2000)
        return {
            "events": [
                _decode_event(e)
                for e in self.telemetry.recent_events(limit=limit, kinds=kinds)
            ]
        }

    # -- actions -----------------------------------------------------------
    def act(self, job_id: int, action: str, body: dict[str, Any]) -> dict[str, Any]:
        """Every mutation the UI can perform, through the service, one at a time.

        The equivalent CLI command travels back with the result. A UI that
        does things a person cannot reproduce in a terminal is a UI nobody can
        debug, so every action says what it just did in the tool's own words.
        """
        service = self.service
        with self._write_lock:
            if action == "cancel":
                force = bool(body.get("force"))
                result = service.cancel(job_id, force=force)
                cmd = f"workerq cancel {job_id}" + (" --force" if force else "")
            elif action == "promote":
                result = service.promote(job_id)
                cmd = f"workerq promote {job_id}"
            elif action == "bump":
                level = str(body.get("level") or "critical")
                result = service.bump_job(job_id, level)
                cmd = f"workerq bump {job_id} {level}"
            elif action == "eta":
                seconds = float(body["eta_seconds"])
                result = service.annotate_job(job_id, eta_seconds=seconds)
                cmd = f"workerq eta {job_id} {int(seconds)}s"
            elif action == "describe":
                result = service.annotate_job(
                    job_id,
                    description=body.get("description"),
                    blocks=body.get("blocks"),
                )
                cmd = f'workerq describe {job_id} "{body.get("description") or ""}"'
            elif action == "requests":
                ram = body.get("ram_gb")
                vram = body.get("vram_gb")
                cpus = body.get("cpus")
                result = service.set_requests(
                    job_id,
                    ram_gb=float(ram) if ram is not None else None,
                    vram_gb=float(vram) if vram is not None else None,
                    cpus=int(cpus) if cpus is not None else None,
                )
                parts = [f"workerq requests {job_id}"]
                if ram is not None:
                    parts.append(f"--ram {ram}")
                if vram is not None:
                    parts.append(f"--vram {vram}")
                if cpus is not None:
                    parts.append(f"--cpus {cpus}")
                cmd = " ".join(parts)
            else:
                raise ValueError(f"unknown action: {action}")
        return {"result": result, "command": cmd}


#: A wait reason names the machine it applies to, and the dispatcher joins one
#: per machine with "|". Counted whole they would be one opaque row; counted
#: per segment each machine's own bottleneck is named.
_REASON_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")


def _block_reasons(detail: str | None) -> list[str]:
    """One normalised reason per machine the job was blocked on.

    Numbers are replaced with N because a reason embeds live measurements -
    "93.4 GiB of 93.6 GiB committed" differs from the same message one tick
    later purely because memory moved, and comparing raw strings treats an
    unchanged condition as a new one every time. `dispatcher._reason_key` does
    the same thing for the same reason.

    The lookahead matters: stripping every digit turns the node name `3080ti`
    into `Nti`, which collapsed a node's reasons into the local machine's and
    made two different bottlenecks read as one row counted twice. A number with
    letters against it is part of a name, not a measurement.
    """
    if not detail:
        return []
    out: list[str] = []
    for segment in detail.split("|"):
        text = segment.strip()
        if not text:
            continue
        out.append(_REASON_NUMBER.sub("N", text))
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _decode_event(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.pop("data_json", None)
    data: Any = None
    if raw:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = None
    event["data"] = data
    return event


def _safe(fn: Any, fallback: Any, *args: Any) -> Any:
    """Call `fn`, and fall back rather than failing the whole page.

    A dashboard where one unavailable number blanks every other number is
    worse than one that says a single field is unknown - and several of these
    reach nvidia-smi or the process table, both of which fail first under
    exactly the pressure the page exists to show.
    """
    try:
        return fn(*args)
    except Exception:
        return fallback
