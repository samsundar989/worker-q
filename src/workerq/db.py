"""SQLite metadata store for worker-q jobs.

One database, WAL journalling, a tiny schema-version table for migrations, and
transactional state changes (spec sections 9 and 29).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from workerq.models import (
    ACTIVE_STATES,
    InvalidTransition,
    Job,
    JobState,
    can_transition,
)
from workerq.util import ensure_dir, restrict_permissions, utcnow_iso

SCHEMA_VERSION = 6

_MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            backend TEXT NOT NULL,
            backend_job_id INTEGER,

            project TEXT NOT NULL,
            label TEXT,
            priority TEXT NOT NULL,

            repo_root TEXT,
            submitted_cwd TEXT NOT NULL,
            execution_cwd TEXT,

            command_json TEXT NOT NULL,
            shell_mode INTEGER NOT NULL DEFAULT 0,

            requested_gpu_count INTEGER NOT NULL DEFAULT 1,
            gpu_mode TEXT NOT NULL DEFAULT 'exclusive',

            snapshot_mode TEXT NOT NULL,
            snapshot_commit TEXT,
            snapshot_path TEXT,

            host TEXT NOT NULL,
            submitter_pid INTEGER,
            submitter_agent TEXT,

            state TEXT NOT NULL,
            exit_code INTEGER,
            runner_pid INTEGER,

            queued_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,

            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            log_path TEXT,
            cuda_visible_devices TEXT,
            passthrough_json TEXT,
            env_json TEXT,

            -- Reserved for the deferred multi-node design (spec section 34).
            node TEXT,
            minimum_vram_gb REAL,
            estimated_duration_seconds REAL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
        CREATE INDEX IF NOT EXISTS idx_jobs_backend_id ON jobs(backend, backend_job_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project);
        CREATE INDEX IF NOT EXISTS idx_jobs_queued_at ON jobs(queued_at);
        """,
    ),
    (
        2,
        """
        -- Resource requests. gpuq brokers any heavy workload, not just GPU
        -- work, so a job declares what it needs and the dispatcher admits it
        -- only when that fits. NULL means "use the configured default", which
        -- keeps rows written by an older gpuq working unchanged.
        ALTER TABLE jobs ADD COLUMN requested_ram_mib REAL;
        ALTER TABLE jobs ADD COLUMN requested_vram_mib REAL;
        ALTER TABLE jobs ADD COLUMN requested_cpus INTEGER;
        """,
    ),
    (
        3,
        """
        -- Per-project scheduling policy. Lets a whole project be marked more
        -- important once, machine-wide, instead of every worker remembering to
        -- pass --priority on every submission.
        CREATE TABLE IF NOT EXISTS project_policy (
            project TEXT PRIMARY KEY,
            priority TEXT,
            note TEXT,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    (
        4,
        """
        -- Preemption. A job is only ever displaced if it declared itself safe
        -- to stop and re-run, because requeuing means re-executing the command
        -- from the start. The counters exist so a displaced job cannot be
        -- starved by repeated preemption.
        ALTER TABLE jobs ADD COLUMN preemptible INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE jobs ADD COLUMN preemption_count INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE jobs ADD COLUMN preempted_at TEXT;
        ALTER TABLE jobs ADD COLUMN preempted_by INTEGER;
        ALTER TABLE jobs ADD COLUMN preempted_reason TEXT;
        """,
    ),
    (
        5,
        """
        -- What the job is doing and how long it should take. Descriptions come
        -- from the worker that submitted it; a duration may be declared, learned
        -- from this command's own history, or reported by the job as it runs.
        ALTER TABLE jobs ADD COLUMN description TEXT;
        ALTER TABLE jobs ADD COLUMN blocks TEXT;
        ALTER TABLE jobs ADD COLUMN eta_seconds REAL;
        ALTER TABLE jobs ADD COLUMN command_signature TEXT;
        ALTER TABLE jobs ADD COLUMN progress_fraction REAL;
        ALTER TABLE jobs ADD COLUMN progress_note TEXT;
        ALTER TABLE jobs ADD COLUMN progress_updated_at TEXT;

        CREATE INDEX IF NOT EXISTS idx_jobs_signature
            ON jobs(project, command_signature, state);
        """,
    ),
    (
        6,
        """
        -- What the job actually used, as opposed to what it declared. NULL
        -- means never measured, which is not the same as measured zero: jobs
        -- that ran before this column existed must not be read as free.
        ALTER TABLE jobs ADD COLUMN peak_ram_mib REAL;
        ALTER TABLE jobs ADD COLUMN peak_vram_mib REAL;
        ALTER TABLE jobs ADD COLUMN usage_samples INTEGER NOT NULL DEFAULT 0;
        """,
    ),
]

_JOB_COLUMNS = (
    "id, backend, backend_job_id, project, label, priority, repo_root, submitted_cwd, "
    "execution_cwd, command_json, shell_mode, requested_gpu_count, gpu_mode, snapshot_mode, "
    "snapshot_commit, snapshot_path, host, submitter_pid, submitter_agent, state, exit_code, "
    "runner_pid, queued_at, started_at, finished_at, error, created_at, updated_at, log_path, "
    "cuda_visible_devices, passthrough_json, env_json, requested_ram_mib, "
    "requested_vram_mib, requested_cpus, preemptible, preemption_count, "
    "preempted_at, preempted_by, preempted_reason, description, blocks, "
    "eta_seconds, command_signature, progress_fraction, progress_note, "
    "progress_updated_at, peak_ram_mib, peak_vram_mib, usage_samples"
)

#: Columns callers are allowed to update through `update_job`.
_UPDATABLE = frozenset(
    {
        "backend_job_id",
        "label",
        "priority",
        "execution_cwd",
        "snapshot_mode",
        "snapshot_commit",
        "snapshot_path",
        "state",
        "exit_code",
        "runner_pid",
        "started_at",
        "finished_at",
        "error",
        "log_path",
        "cuda_visible_devices",
        "passthrough_json",
        "env_json",
        "requested_ram_mib",
        "requested_vram_mib",
        "requested_cpus",
        "preemptible",
        "preemption_count",
        "preempted_at",
        "preempted_by",
        "preempted_reason",
        "description",
        "blocks",
        "eta_seconds",
        "command_signature",
        "progress_fraction",
        "progress_note",
        "progress_updated_at",
        "peak_ram_mib",
        "peak_vram_mib",
        "usage_samples",
    }
)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the pragmas required by the spec."""
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    restrict_permissions(db_path)
    return conn


class Database:
    """Thin transactional wrapper around the jobs table."""

    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle --------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = connect(self.path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # -- migrations -------------------------------------------------------
    def initialize(self) -> int:
        """Create/upgrade the schema. Idempotent."""
        conn = self.conn
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "  version INTEGER NOT NULL,"
            "  applied_at TEXT NOT NULL"
            ");"
        )
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] or 0
        for version, script in _MIGRATIONS:
            if version <= current:
                continue
            conn.executescript(script)
            conn.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                (version, utcnow_iso()),
            )
            current = version
        self._backfill_signatures()
        return current

    def _backfill_signatures(self) -> None:
        """Give pre-existing jobs a command signature.

        Duration learning reads finished jobs, so without this an upgrade would
        throw away every run already on disk and start estimating from zero.
        Signatures are computed in Python, which a SQL migration cannot do, so
        it happens here. Idempotent: after one pass nothing is left to fill.
        """
        from workerq.eta import command_signature

        try:
            rows = self.conn.execute(
                "SELECT id, command_json, shell_mode FROM jobs "
                "WHERE command_signature IS NULL"
            ).fetchall()
        except sqlite3.OperationalError:
            return  # the column does not exist yet: nothing to backfill

        for row in rows:
            try:
                command = json.loads(row["command_json"])
            except (TypeError, ValueError):
                command = []
            self.conn.execute(
                "UPDATE jobs SET command_signature = ? WHERE id = ?",
                (command_signature(command, bool(row["shell_mode"])), row["id"]),
            )

    def schema_version(self) -> int:
        try:
            row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["v"] or 0)

    # -- reads ------------------------------------------------------------
    def get_job(self, job_id: int) -> Job | None:
        row = self.conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return _row_to_job(row) if row else None

    def get_job_by_backend_id(self, backend: str, backend_job_id: int) -> Job | None:
        row = self.conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM jobs WHERE backend = ? AND backend_job_id = ?",
            (backend, backend_job_id),
        ).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(
        self,
        *,
        states: Iterable[str] | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[Job]:
        sql = f"SELECT {_JOB_COLUMNS} FROM jobs"
        clauses: list[str] = []
        params: list[Any] = []
        if states is not None:
            states = list(states)
            if not states:
                return []
            clauses.append("state IN (" + ",".join("?" for _ in states) + ")")
            params.extend(states)
        if project:
            clauses.append("project = ?")
            params.append(project)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [_row_to_job(r) for r in self.conn.execute(sql, params).fetchall()]

    def active_jobs(self) -> list[Job]:
        return self.list_jobs(states=[s.value for s in ACTIVE_STATES])

    def count_by_state(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
        return {r["state"]: int(r["n"]) for r in rows}

    # -- writes -----------------------------------------------------------
    def insert_job(self, **values: Any) -> int:
        now = utcnow_iso()
        values.setdefault("created_at", now)
        values.setdefault("updated_at", now)
        values.setdefault("queued_at", now)
        values.setdefault("state", JobState.PREPARING.value)
        cols = list(values)
        sql = (
            "INSERT INTO jobs (" + ", ".join(cols) + ") VALUES ("
            + ", ".join("?" for _ in cols)
            + ")"
        )
        with self.transaction() as conn:
            cur = conn.execute(sql, [values[c] for c in cols])
            return int(cur.lastrowid)

    def update_job(self, job_id: int, **values: Any) -> Job:
        """Update a job inside a transaction, validating any state change."""
        unknown = set(values) - _UPDATABLE
        if unknown:
            raise ValueError(f"non-updatable job columns: {sorted(unknown)}")
        with self.transaction() as conn:
            row = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"no such job: {job_id}")
            if "state" in values:
                current = JobState(row["state"])
                target = JobState(values["state"])
                if not can_transition(current, target):
                    raise InvalidTransition(current, target)
                values["state"] = target.value
            values["updated_at"] = utcnow_iso()
            cols = list(values)
            conn.execute(
                "UPDATE jobs SET " + ", ".join(f"{c} = ?" for c in cols) + " WHERE id = ?",
                [values[c] for c in cols] + [job_id],
            )
        job = self.get_job(job_id)
        assert job is not None
        return job

    def try_update_state(self, job_id: int, target: JobState, **values: Any) -> Job | None:
        """Update state, returning None when the transition is not allowed.

        Used by reconciliation and the runner, where a terminal state may have
        been set concurrently (for example by `gpuq cancel`).
        """
        try:
            return self.update_job(job_id, state=target.value, **values)
        except InvalidTransition:
            return None

    def set_error(self, job_id: int, message: str, *, state: JobState = JobState.FAILED) -> None:
        self.try_update_state(job_id, state, error=message, finished_at=utcnow_iso())


    # -- project policy ---------------------------------------------------
    def get_project_priority(self, project: str) -> str | None:
        try:
            row = self.conn.execute(
                "SELECT priority FROM project_policy WHERE project = ?", (project,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row["priority"] if row and row["priority"] else None

    def set_project_priority(
        self, project: str, priority: str | None, *, note: str | None = None
    ) -> None:
        """Set (or clear, with priority=None) a project's default priority."""
        with self.transaction() as conn:
            if priority is None:
                conn.execute("DELETE FROM project_policy WHERE project = ?", (project,))
                return
            conn.execute(
                "INSERT INTO project_policy(project, priority, note, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(project) DO UPDATE SET "
                "priority = excluded.priority, note = excluded.note, "
                "updated_at = excluded.updated_at",
                (project, priority, note, utcnow_iso()),
            )

    def list_project_priorities(self) -> list[dict[str, Any]]:
        try:
            rows = self.conn.execute(
                "SELECT project, priority, note, updated_at FROM project_policy "
                "ORDER BY project"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]


def _row_to_job(row: sqlite3.Row) -> Job:
    data = dict(row)
    return Job(**data)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
