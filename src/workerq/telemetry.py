"""Resource time-series and dispatcher events.

Without this, "why did my job die at 22:03?" is unanswerable: the evidence is
gone by the time anyone looks. The dispatcher writes a cheap sample every few
seconds and an event per state change, so `workerq report` can say what the
machine actually looked like at the moment a job failed - and, crucially, what
*else* was holding memory.

Deliberately a separate database from both the job metadata DB and the backend
queue DB: it is high-churn, disposable, and losing it must never affect job
state.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from workerq.util import ensure_dir, parse_iso, utcnow_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,

    gpu_used_mib REAL,
    gpu_total_mib REAL,
    gpu_free_percent REAL,
    gpu_utilization REAL,

    host_total_mib REAL,
    host_available_mib REAL,
    host_free_percent REAL,
    commit_used_mib REAL,
    commit_limit_mib REAL,
    commit_percent REAL,

    running_job_id INTEGER,
    queued_count INTEGER,
    top_consumers_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_samples_at ON samples(at);

CREATE TABLE IF NOT EXISTS job_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER NOT NULL,
    at TEXT NOT NULL,
    job_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_samples_job ON job_samples(job_id);
CREATE INDEX IF NOT EXISTS idx_job_samples_sample ON job_samples(sample_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    job_id INTEGER,
    backend_job_id INTEGER,
    detail TEXT,
    data_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id);
"""

# Event kinds
EVENT_STARTED = "job_started"
EVENT_FINISHED = "job_finished"
EVENT_BLOCKED = "job_blocked"
EVENT_CANCEL = "job_cancel"
EVENT_PREEMPTED = "job_preempted"
EVENT_DAEMON = "daemon"
EVENT_RESERVE = "reserve"
EVENT_PRESSURE = "resource_pressure"

#: Kinds that repeat while a condition persists rather than marking something
#: happening once. They are worth keeping - a stalled queue's reasons are how
#: you find out what it is stalled on - but on their own retention budget, or
#: they bury the events that describe individual jobs. See `Telemetry.prune`.
NOISY_EVENT_KINDS = (EVENT_BLOCKED, EVENT_PRESSURE)

#: The record of what happened to jobs. Never evicted by queue noise.
LIFECYCLE_EVENT_KINDS = (
    EVENT_STARTED,
    EVENT_FINISHED,
    EVENT_CANCEL,
    EVENT_PREEMPTED,
)

_SAMPLE_COLUMNS = (
    "id, at, gpu_used_mib, gpu_total_mib, gpu_free_percent, gpu_utilization, "
    "host_total_mib, host_available_mib, host_free_percent, commit_used_mib, "
    "commit_limit_mib, commit_percent, running_job_id, queued_count, top_consumers_json"
)


class Telemetry:
    """Append-only sample/event store. All methods swallow storage errors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle --------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            ensure_dir(self.path.parent)
            conn = sqlite3.connect(str(self.path), timeout=15.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def initialize(self) -> None:
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    @contextmanager
    def _safe(self) -> Iterator[sqlite3.Connection | None]:
        """Telemetry must never break the dispatcher or the CLI."""
        try:
            yield self.conn
        except Exception:
            yield None  # type: ignore[misc]

    # -- writes -----------------------------------------------------------
    def record_sample(
        self,
        *,
        gpu_used_mib: float | None = None,
        gpu_total_mib: float | None = None,
        gpu_free_percent: float | None = None,
        gpu_utilization: float | None = None,
        host_total_mib: float | None = None,
        host_available_mib: float | None = None,
        host_free_percent: float | None = None,
        commit_used_mib: float | None = None,
        commit_limit_mib: float | None = None,
        commit_percent: float | None = None,
        running_job_id: int | None = None,
        queued_count: int | None = None,
        top_consumers: list[dict[str, Any]] | None = None,
        running_job_ids: list[int] | None = None,
    ) -> None:
        """Record one machine-wide sample.

        `running_job_id` is a single column and stays for compatibility with
        everything that already reads it; with several jobs running it names an
        arbitrary one. `running_job_ids` records them all in `job_samples`, so
        pressure can be attributed to the whole set rather than a coin flip.
        """
        try:
            cursor = self.conn.execute(
                "INSERT INTO samples (at, gpu_used_mib, gpu_total_mib, gpu_free_percent, "
                "gpu_utilization, host_total_mib, host_available_mib, host_free_percent, "
                "commit_used_mib, commit_limit_mib, commit_percent, running_job_id, "
                "queued_count, top_consumers_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    utcnow_iso(),
                    gpu_used_mib,
                    gpu_total_mib,
                    gpu_free_percent,
                    gpu_utilization,
                    host_total_mib,
                    host_available_mib,
                    host_free_percent,
                    commit_used_mib,
                    commit_limit_mib,
                    commit_percent,
                    running_job_id,
                    queued_count,
                    json.dumps(top_consumers, ensure_ascii=False) if top_consumers else None,
                ),
            )
            if running_job_ids:
                sample_id = int(cursor.lastrowid or 0)
                at = utcnow_iso()
                self.conn.executemany(
                    "INSERT INTO job_samples (sample_id, at, job_id) VALUES (?,?,?)",
                    [(sample_id, at, int(j)) for j in running_job_ids],
                )
        except Exception:
            pass

    def record_event(
        self,
        kind: str,
        *,
        job_id: int | None = None,
        backend_job_id: int | None = None,
        detail: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.conn.execute(
                "INSERT INTO events (at, kind, job_id, backend_job_id, detail, data_json) "
                "VALUES (?,?,?,?,?,?)",
                (
                    utcnow_iso(),
                    kind,
                    job_id,
                    backend_job_id,
                    detail,
                    json.dumps(data, ensure_ascii=False) if data else None,
                ),
            )
        except Exception:
            pass

    # -- reads ------------------------------------------------------------
    def latest_sample(self) -> dict[str, Any] | None:
        try:
            row = self.conn.execute(
                f"SELECT {_SAMPLE_COLUMNS} FROM samples ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except Exception:
            return None
        return dict(row) if row else None

    def samples_between(self, start: str, end: str) -> list[dict[str, Any]]:
        try:
            rows = self.conn.execute(
                f"SELECT {_SAMPLE_COLUMNS} FROM samples WHERE at >= ? AND at <= ? ORDER BY at",
                (start, end),
            ).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    def sample_near(self, when: str, *, window_seconds: float = 180.0) -> dict[str, Any] | None:
        """The sample closest to `when`, for explaining a failure after the fact."""
        target = parse_iso(when)
        if target is None:
            return None
        try:
            rows = self.conn.execute(
                f"SELECT {_SAMPLE_COLUMNS} FROM samples ORDER BY ABS("
                "  (julianday(at) - julianday(?)) * 86400.0"
                ") LIMIT 1",
                (when,),
            ).fetchall()
        except Exception:
            return None
        if not rows:
            return None
        sample = dict(rows[0])
        at = parse_iso(sample.get("at"))
        if at is None or abs((at - target).total_seconds()) > window_seconds:
            return None
        return sample

    def peak_between(self, start: str, end: str) -> dict[str, Any] | None:
        """Worst-case resource pressure observed during a window."""
        samples = self.samples_between(start, end)
        if not samples:
            return None

        def _min(key: str) -> float | None:
            values = [s[key] for s in samples if s.get(key) is not None]
            return min(values) if values else None

        def _max(key: str) -> float | None:
            values = [s[key] for s in samples if s.get(key) is not None]
            return max(values) if values else None

        return {
            "samples": len(samples),
            "min_host_free_percent": _min("host_free_percent"),
            "max_commit_percent": _max("commit_percent"),
            "min_gpu_free_percent": _min("gpu_free_percent"),
            "max_gpu_used_mib": _max("gpu_used_mib"),
        }

    def recent_events(
        self, *, limit: int = 100, kinds: list[str] | None = None, since: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, at, kind, job_id, backend_job_id, detail, data_json FROM events"
        clauses: list[str] = []
        params: list[Any] = []
        if kinds:
            clauses.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            params.extend(kinds)
        if since:
            clauses.append("at >= ?")
            params.append(since)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    def job_series(self, job_id: int, *, limit: int = 2000) -> list[dict[str, Any]]:
        """Machine samples taken while this job was running.

        Via `job_samples`, not `samples.running_job_id`: that column names an
        arbitrary one of however many jobs were running, so a job sharing the
        machine would appear to have no series at all.

        These are whole-machine figures, not the job's own. On a box running
        several jobs plus a desktop they show the context a job ran in, which
        is what explains a slow run - they are not an attribution of its usage.
        """
        try:
            rows = self.conn.execute(
                f"SELECT s.{_SAMPLE_COLUMNS.replace(', ', ', s.')} FROM samples s "
                "JOIN job_samples js ON js.sample_id = s.id "
                "WHERE js.job_id = ? ORDER BY s.id ASC LIMIT ?",
                (int(job_id), int(limit)),
            ).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    def events_for_job(
        self, job_id: int | None, backend_job_id: int | None = None
    ) -> list[dict[str, Any]]:
        """One job's lifecycle, found by either id.

        Both, because `job_id` was NULL in every event written before it
        started being populated, and that history is worth keeping readable.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(int(job_id))
        if backend_job_id is not None:
            clauses.append("backend_job_id = ?")
            params.append(int(backend_job_id))
        if not clauses:
            return []
        try:
            rows = self.conn.execute(
                "SELECT id, at, kind, job_id, backend_job_id, detail, data_json "
                "FROM events WHERE " + " OR ".join(clauses) + " ORDER BY id ASC",
                params,
            ).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    def samples_since(self, since: str, *, limit: int = 5000) -> list[dict[str, Any]]:
        try:
            rows = self.conn.execute(
                f"SELECT {_SAMPLE_COLUMNS} FROM samples WHERE at >= ? "
                "ORDER BY id ASC LIMIT ?",
                (since, int(limit)),
            ).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    # -- retention --------------------------------------------------------
    def prune(
        self,
        *,
        keep_samples: int = 200_000,
        keep_events: int = 50_000,
        keep_noise_events: int = 10_000,
    ) -> None:
        """Trim the store, keeping each kind of event on its own budget.

        A single shared cap does not survive contact with a busy queue. One
        flat 50,000-row limit here left 49,454 `job_blocked` rows and 239
        `job_started` rows - for 486 jobs. A blocked job repeats its reason
        every few seconds for as long as it waits, so the routine noise
        evicted the lifecycle record of most jobs that ever ran, and the
        events table reached back six days while samples reached back eight.

        Lifecycle events are the history. They get their own budget, and the
        noisy kinds get a smaller one, so a stalled queue can no longer spend
        the record of what actually happened.
        """
        noise = list(NOISY_EVENT_KINDS)
        placeholders = ",".join("?" for _ in noise)
        try:
            self.conn.execute(
                "DELETE FROM samples WHERE id NOT IN "
                "(SELECT id FROM samples ORDER BY id DESC LIMIT ?)",
                (keep_samples,),
            )
            self.conn.execute(
                f"DELETE FROM events WHERE kind IN ({placeholders}) AND id NOT IN "
                f"(SELECT id FROM events WHERE kind IN ({placeholders}) "
                "ORDER BY id DESC LIMIT ?)",
                (*noise, *noise, keep_noise_events),
            )
            self.conn.execute(
                f"DELETE FROM events WHERE kind NOT IN ({placeholders}) AND id NOT IN "
                f"(SELECT id FROM events WHERE kind NOT IN ({placeholders}) "
                "ORDER BY id DESC LIMIT ?)",
                (*noise, *noise, keep_events),
            )
        except Exception:
            pass


    def admission_likelihood(self, needed_mib: float, floor_mib: float) -> tuple[int, int]:
        """(samples that would have admitted `needed_mib`, samples total).

        Answers "could this declaration ever actually start here?". A machine
        with a large steady baseline - editors, browsers, a row of agents - has
        far less free RAM than it has installed, so a request can pass the
        submit-time capacity check and then wait forever for headroom that
        never arrives. This is what makes that visible at submit time.
        """
        try:
            total = self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            if not total:
                return 0, 0
            ok = self.conn.execute(
                "SELECT COUNT(*) FROM samples WHERE host_available_mib >= ?",
                (needed_mib + floor_mib,),
            ).fetchone()[0]
        except Exception:
            return 0, 0
        return int(ok), int(total)


def open_telemetry(state_dir: Path) -> Telemetry:
    store = Telemetry(state_dir / "telemetry.sqlite3")
    store.initialize()
    return store
