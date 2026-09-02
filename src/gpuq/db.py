"""SQLite metadata store for GPUQ jobs.

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

from gpuq.models import (
    ACTIVE_STATES,
    InvalidTransition,
    Job,
    JobState,
    can_transition,
)
from gpuq.util import ensure_dir, restrict_permissions, utcnow_iso

SCHEMA_VERSION = 2

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
]

_JOB_COLUMNS = (
    "id, backend, backend_job_id, project, label, priority, repo_root, submitted_cwd, "
    "execution_cwd, command_json, shell_mode, requested_gpu_count, gpu_mode, snapshot_mode, "
    "snapshot_commit, snapshot_path, host, submitter_pid, submitter_agent, state, exit_code, "
    "runner_pid, queued_at, started_at, finished_at, error, created_at, updated_at, log_path, "
    "cuda_visible_devices, passthrough_json, env_json, requested_ram_mib, "
    "requested_vram_mib, requested_cpus"
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
        return current

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


def _row_to_job(row: sqlite3.Row) -> Job:
    data = dict(row)
    return Job(**data)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
