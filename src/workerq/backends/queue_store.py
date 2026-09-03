"""Durable queue state owned by the local dispatcher backend.

Kept in its own SQLite database, separate from the worker-q metadata DB, so the
backend remains genuinely swappable: core code only ever sees `BackendJob`.
Both the CLI (writer of intent) and the daemon (executor) reach the queue
through this class, and WAL + IMMEDIATE transactions make that safe.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from workerq.backends.base import (
    BACKEND_FINISHED,
    BACKEND_QUEUED,
    BACKEND_REMOVED,
    BACKEND_RUNNING,
    BackendJob,
)
from workerq.util import ensure_dir, utcnow_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS bjobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    argv_json TEXT NOT NULL,
    cwd TEXT,
    env_json TEXT,
    gpu_count INTEGER NOT NULL DEFAULT 1,
    slots INTEGER NOT NULL DEFAULT 1,
    priority_rank INTEGER NOT NULL DEFAULT 100,
    position INTEGER NOT NULL DEFAULT 0,
    log_path TEXT,
    state TEXT NOT NULL,
    exit_code INTEGER,
    pid INTEGER,
    pid_creation INTEGER,
    assigned_devices TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    cancel_force INTEGER NOT NULL DEFAULT 0,
    cancel_at TEXT,
    wait_reason TEXT,
    enqueued_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_bjobs_state ON bjobs(state);
CREATE INDEX IF NOT EXISTS idx_bjobs_order ON bjobs(state, priority_rank, position, id);
CREATE INDEX IF NOT EXISTS idx_bjobs_label ON bjobs(label);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

#: Columns added after v1. Applied with ALTER so an existing queue database -
#: possibly with jobs in flight - upgrades in place.
_ADDED_COLUMNS = (
    ("ram_mib", "REAL"),
    ("vram_mib", "REAL"),
    ("cpus", "INTEGER"),
    ("preemptible", "INTEGER NOT NULL DEFAULT 0"),
    ("preempt_requested", "INTEGER NOT NULL DEFAULT 0"),
    ("preempt_by", "INTEGER"),
    ("preempt_at", "TEXT"),
    # Whether this job will share a GPU with another. The dispatcher needs it
    # for placement, so it has to live here and not only in the jobs table.
    ("gpu_mode", "TEXT NOT NULL DEFAULT 'exclusive'"),
)

_COLUMNS = (
    "id, label, argv_json, cwd, env_json, gpu_count, slots, priority_rank, position, "
    "log_path, state, exit_code, pid, pid_creation, assigned_devices, cancel_requested, "
    "cancel_force, cancel_at, wait_reason, enqueued_at, started_at, finished_at, "
    "ram_mib, vram_mib, cpus, preemptible, preempt_requested, preempt_by, preempt_at, "
    "gpu_mode"
)


class QueueStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle --------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            ensure_dir(self.path.parent)
            conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def initialize(self) -> None:
        self.conn.executescript(SCHEMA)
        existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(bjobs)")}
        for column, sql_type in _ADDED_COLUMNS:
            if column not in existing:
                self.conn.execute(f"ALTER TABLE bjobs ADD COLUMN {column} {sql_type}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

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

    # -- meta -------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def get_meta_int(self, key: str, default: int) -> int:
        raw = self.get_meta(key)
        try:
            return int(raw) if raw is not None else default
        except ValueError:
            return default

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def all_meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}

    # -- enqueue ----------------------------------------------------------
    def enqueue(
        self,
        argv: list[str],
        *,
        label: str | None,
        gpu_count: int,
        slots: int,
        priority_rank: int,
        log_path: str | None,
        cwd: str | None,
        env: dict[str, str] | None,
        ram_mib: float | None = None,
        vram_mib: float | None = None,
        cpus: int | None = None,
        preemptible: bool = False,
        gpu_mode: str = "exclusive",
    ) -> int:
        with self.transaction() as conn:
            row = conn.execute("SELECT COALESCE(MAX(position), 0) AS p FROM bjobs").fetchone()
            position = int(row["p"]) + 1
            cur = conn.execute(
                "INSERT INTO bjobs (label, argv_json, cwd, env_json, gpu_count, slots, "
                "priority_rank, position, log_path, state, enqueued_at, ram_mib, "
                "vram_mib, cpus, preemptible, gpu_mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    label,
                    json.dumps(argv, ensure_ascii=False),
                    cwd,
                    json.dumps(env or {}, ensure_ascii=False),
                    int(gpu_count),
                    int(slots),
                    int(priority_rank),
                    position,
                    log_path,
                    BACKEND_QUEUED,
                    utcnow_iso(),
                    ram_mib,
                    vram_mib,
                    cpus,
                    1 if preemptible else 0,
                    gpu_mode,
                ),
            )
            return int(cur.lastrowid)

    # -- reads ------------------------------------------------------------
    def get(self, backend_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            f"SELECT {_COLUMNS} FROM bjobs WHERE id = ?", (backend_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_by_label(self, label: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            f"SELECT {_COLUMNS} FROM bjobs WHERE label = ? ORDER BY id DESC LIMIT 1",
            (label,),
        ).fetchone()
        return dict(row) if row else None

    def list_all(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = (
            f"SELECT {_COLUMNS} FROM bjobs ORDER BY "
            "CASE state WHEN 'RUNNING' THEN 0 WHEN 'QUEUED' THEN 1 ELSE 2 END, "
            "priority_rank, position, id"
        )
        params: list[Any] = []
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def list_by_state(self, state: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM bjobs WHERE state = ? "
            "ORDER BY priority_rank, position, id",
            (state,),
        ).fetchall()
        return [dict(r) for r in rows]

    def running(self) -> list[dict[str, Any]]:
        return self.list_by_state(BACKEND_RUNNING)

    def queued(self) -> list[dict[str, Any]]:
        return self.list_by_state(BACKEND_QUEUED)

    # -- mutations --------------------------------------------------------
    def update(self, backend_id: int, **values: Any) -> None:
        if not values:
            return
        cols = list(values)
        self.conn.execute(
            "UPDATE bjobs SET " + ", ".join(f"{c} = ?" for c in cols) + " WHERE id = ?",
            [values[c] for c in cols] + [backend_id],
        )

    def claim_for_start(self, backend_id: int) -> bool:
        """Atomically move QUEUED -> RUNNING. False if it was already taken."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE bjobs SET state = ?, started_at = ?, wait_reason = NULL "
                "WHERE id = ? AND state = ?",
                (BACKEND_RUNNING, utcnow_iso(), backend_id, BACKEND_QUEUED),
            )
            return cur.rowcount == 1

    def finish(self, backend_id: int, exit_code: int | None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE bjobs SET state = ?, exit_code = ?, finished_at = ?, pid = NULL, "
                "pid_creation = NULL WHERE id = ?",
                (BACKEND_FINISHED, exit_code, utcnow_iso(), backend_id),
            )

    def remove_queued(self, backend_id: int) -> bool:
        """Drop a job that has not started. False when it is no longer queued."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE bjobs SET state = ?, finished_at = ? WHERE id = ? AND state = ?",
                (BACKEND_REMOVED, utcnow_iso(), backend_id, BACKEND_QUEUED),
            )
            return cur.rowcount == 1

    def request_cancel(self, backend_id: int, *, force: bool) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE bjobs SET cancel_requested = 1, cancel_force = ?, cancel_at = ? "
                "WHERE id = ? AND state IN (?, ?)",
                (
                    1 if force else 0,
                    utcnow_iso(),
                    backend_id,
                    BACKEND_QUEUED,
                    BACKEND_RUNNING,
                ),
            )
            return cur.rowcount == 1

    def request_preempt(self, backend_id: int, *, by_backend_id: int | None) -> bool:
        """Ask a running job to stop so a higher-priority one can start."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE bjobs SET preempt_requested = 1, preempt_by = ?, preempt_at = ? "
                "WHERE id = ? AND state = ? AND COALESCE(preempt_requested, 0) = 0",
                (by_backend_id, utcnow_iso(), backend_id, BACKEND_RUNNING),
            )
            return cur.rowcount == 1

    def requeue(self, backend_id: int, *, priority_rank: int | None = None) -> bool:
        """Return a displaced job to the queue rather than finishing it.

        It goes to the head of its own priority tier: it already waited once and
        was interrupted, so it should not also go to the back of the line.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT priority_rank FROM bjobs WHERE id = ?", (backend_id,)
            ).fetchone()
            if row is None:
                return False
            rank = priority_rank if priority_rank is not None else int(row["priority_rank"])
            head = conn.execute(
                "SELECT COALESCE(MIN(position), 0) AS p FROM bjobs WHERE state = ?",
                (BACKEND_QUEUED,),
            ).fetchone()
            cur = conn.execute(
                "UPDATE bjobs SET state = ?, priority_rank = ?, position = ?, "
                "pid = NULL, pid_creation = NULL, started_at = NULL, "
                "assigned_devices = NULL, preempt_requested = 0, preempt_by = NULL, "
                "preempt_at = NULL, wait_reason = ? WHERE id = ?",
                (
                    BACKEND_QUEUED,
                    rank,
                    int(head["p"]) - 1,
                    "requeued after preemption",
                    backend_id,
                ),
            )
            return cur.rowcount == 1

    def set_preemptible(self, backend_id: int, preemptible: bool) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE bjobs SET preemptible = ? WHERE id = ?",
                (1 if preemptible else 0, backend_id),
            )
            return cur.rowcount == 1

    def set_priority(self, backend_id: int, priority_rank: int) -> bool:
        """Re-rank a queued job in place, keeping its arrival order within the tier."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE bjobs SET priority_rank = ? WHERE id = ? AND state = ?",
                (int(priority_rank), backend_id, BACKEND_QUEUED),
            )
            return cur.rowcount == 1

    def set_requests(
        self,
        backend_id: int,
        *,
        ram_mib: float | None = None,
        vram_mib: float | None = None,
        cpus: int | None = None,
    ) -> bool:
        """Correct a queued job's declared footprint in place.

        Only QUEUED: a running job's reservation is already counted against
        everything else, and changing it underneath the dispatcher would let
        the ledger disagree with what is actually allocated.
        """
        updates: list[str] = []
        values: list[Any] = []
        for column, value in (("ram_mib", ram_mib), ("vram_mib", vram_mib), ("cpus", cpus)):
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(value)
        if not updates:
            return False
        values.extend([backend_id, BACKEND_QUEUED])
        with self.transaction() as conn:
            cur = conn.execute(
                f"UPDATE bjobs SET {', '.join(updates)}, wait_reason = NULL "
                "WHERE id = ? AND state = ?",
                tuple(values),
            )
            return cur.rowcount == 1

    def promote(self, backend_id: int) -> bool:
        """Move a queued job to the head of the dispatch order."""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM bjobs WHERE id = ?", (backend_id,)
            ).fetchone()
            if row is None or row["state"] != BACKEND_QUEUED:
                return False
            head = conn.execute(
                "SELECT COALESCE(MIN(position), 0) AS p, COALESCE(MIN(priority_rank), 100) AS r "
                "FROM bjobs WHERE state = ?",
                (BACKEND_QUEUED,),
            ).fetchone()
            conn.execute(
                "UPDATE bjobs SET position = ?, priority_rank = ? WHERE id = ?",
                (int(head["p"]) - 1, min(0, int(head["r"])), backend_id),
            )
            return True

    def trim_finished(self, max_finished: int) -> int:
        """Keep the queue table bounded (`max_finished`)."""
        with self.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM bjobs WHERE state IN (?, ?) AND id NOT IN ("
                "  SELECT id FROM bjobs WHERE state IN (?, ?) ORDER BY id DESC LIMIT ?"
                ")",
                (
                    BACKEND_FINISHED,
                    BACKEND_REMOVED,
                    BACKEND_FINISHED,
                    BACKEND_REMOVED,
                    int(max_finished),
                ),
            )
            return cur.rowcount


def row_to_backend_job(row: dict[str, Any]) -> BackendJob:
    return BackendJob(
        backend_id=int(row["id"]),
        state=str(row["state"]),
        label=row.get("label"),
        output_path=Path(row["log_path"]) if row.get("log_path") else None,
        pid=row.get("pid"),
        exit_code=row.get("exit_code"),
        enqueued_at=row.get("enqueued_at"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        wait_reason=row.get("wait_reason"),
        gpu_count=int(row.get("gpu_count") or 0),
        extra={
            "assigned_devices": row.get("assigned_devices"),
            "cancel_requested": bool(row.get("cancel_requested")),
            "priority_rank": row.get("priority_rank"),
            "position": row.get("position"),
            "ram_mib": row.get("ram_mib"),
            "vram_mib": row.get("vram_mib"),
            "cpus": row.get("cpus"),
            "preemptible": bool(row.get("preemptible")),
            "preempt_requested": bool(row.get("preempt_requested")),
            "preempt_by": row.get("preempt_by"),
        },
    )
