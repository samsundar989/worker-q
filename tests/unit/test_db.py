"""Database schema, migrations and state-transition safety (spec section 23)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from gpuq.db import SCHEMA_VERSION, Database, json_dumps
from gpuq.models import InvalidTransition, JobState, can_transition
from gpuq.util import utcnow_iso


def _insert(db: Database, **overrides) -> int:
    values = {
        "backend": "local_dispatcher",
        "project": "demo",
        "priority": "normal",
        "submitted_cwd": "/tmp/demo",
        "command_json": json_dumps(["python", "train.py"]),
        "snapshot_mode": "git",
        "host": "testhost",
        "state": JobState.PREPARING.value,
    }
    values.update(overrides)
    return db.insert_job(**values)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "gpuq.sqlite3")
    database.initialize()
    yield database
    database.close()


def test_schema_initializes(db: Database):
    assert db.schema_version() == SCHEMA_VERSION
    tables = {
        r["name"]
        for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"jobs", "schema_version"} <= tables


def test_required_indexes_exist(db: Database):
    indexes = {
        r["name"] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    for expected in (
        "idx_jobs_state",
        "idx_jobs_backend_id",
        "idx_jobs_project",
        "idx_jobs_queued_at",
    ):
        assert expected in indexes


def test_pragmas_applied(db: Database):
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert db.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migration_is_idempotent(tmp_path: Path):
    path = tmp_path / "gpuq.sqlite3"
    first = Database(path)
    assert first.initialize() == SCHEMA_VERSION
    job_id = _insert(first)
    first.close()

    second = Database(path)
    assert second.initialize() == SCHEMA_VERSION  # re-running changes nothing
    assert second.get_job(job_id) is not None
    applied = second.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert applied == SCHEMA_VERSION  # one row per applied migration, none repeated
    second.close()


def test_insert_and_read_roundtrip(db: Database):
    job_id = _insert(db, command_json=json_dumps(["python", "-c", "print('hi there')"]))
    job = db.get_job(job_id)
    assert job is not None
    assert job.command == ["python", "-c", "print('hi there')"]
    assert job.project == "demo"
    assert job.state_enum is JobState.PREPARING


def test_command_json_survives_awkward_arguments(db: Database):
    argv = [
        "python",
        "train.py",
        "--name=a b",
        "--glob=*.py",
        '--quote="x"',
        "--unicode=café-日本",
        "--meta=a&b|c>d",
    ]
    job = db.get_job(_insert(db, command_json=json_dumps(argv)))
    assert job.command == argv


def test_update_job_rejects_unknown_columns(db: Database):
    job_id = _insert(db)
    with pytest.raises(ValueError, match="non-updatable"):
        db.update_job(job_id, project="somewhere-else")


def test_valid_transitions(db: Database):
    job_id = _insert(db)
    db.update_job(job_id, state=JobState.QUEUED.value)
    db.update_job(job_id, state=JobState.RUNNING.value)
    job = db.update_job(job_id, state=JobState.SUCCEEDED.value, exit_code=0)
    assert job.state_enum is JobState.SUCCEEDED


def test_terminal_state_cannot_revert_to_queued(db: Database):
    job_id = _insert(db, state=JobState.QUEUED.value)
    db.update_job(job_id, state=JobState.SUCCEEDED.value)
    with pytest.raises(InvalidTransition):
        db.update_job(job_id, state=JobState.QUEUED.value)
    assert db.get_job(job_id).state_enum is JobState.SUCCEEDED


@pytest.mark.parametrize(
    "terminal",
    [JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.LOST],
)
def test_no_terminal_state_reverts(db: Database, terminal: JobState):
    job_id = _insert(db, state=JobState.QUEUED.value)
    db.update_job(job_id, state=terminal.value)
    for target in (JobState.QUEUED, JobState.RUNNING, JobState.PREPARING):
        with pytest.raises(InvalidTransition):
            db.update_job(job_id, state=target.value)


def test_cancelled_is_never_resurrected(db: Database):
    """A cancelled job must not become RUNNING again."""
    job_id = _insert(db, state=JobState.QUEUED.value)
    db.update_job(job_id, state=JobState.CANCELLED.value)
    assert db.try_update_state(job_id, JobState.RUNNING) is None
    assert db.get_job(job_id).state_enum is JobState.CANCELLED


def test_try_update_state_returns_none_instead_of_raising(db: Database):
    job_id = _insert(db, state=JobState.QUEUED.value)
    db.update_job(job_id, state=JobState.FAILED.value)
    assert db.try_update_state(job_id, JobState.SUCCEEDED) is None


def test_same_state_is_allowed(db: Database):
    job_id = _insert(db, state=JobState.RUNNING.value)
    assert db.update_job(job_id, state=JobState.RUNNING.value) is not None


def test_can_transition_matrix():
    assert can_transition(JobState.QUEUED, JobState.RUNNING)
    assert can_transition(JobState.RUNNING, JobState.CANCELLED)
    assert not can_transition(JobState.SUCCEEDED, JobState.RUNNING)
    assert not can_transition(JobState.CANCELLED, JobState.QUEUED)


def test_updated_at_changes_on_update(db: Database):
    job_id = _insert(db)
    before = db.get_job(job_id).updated_at
    db.update_job(job_id, label="tagged")
    assert db.get_job(job_id).updated_at >= before


def test_concurrent_inserts_with_wal(tmp_path: Path):
    """Two writers must both succeed under WAL + busy_timeout."""
    path = tmp_path / "gpuq.sqlite3"
    Database(path).initialize()
    errors: list[Exception] = []
    created: list[int] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        database = Database(path)
        try:
            for n in range(10):
                job_id = _insert(database, project=f"proj-{index}", label=f"j{n}")
                with lock:
                    created.append(job_id)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
        finally:
            database.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert len(created) == 40
    assert len(set(created)) == 40  # ids are unique


def test_transaction_rolls_back_on_error(db: Database):
    job_id = _insert(db)
    with pytest.raises(sqlite3.OperationalError):
        with db.transaction() as conn:
            conn.execute("UPDATE jobs SET label = 'x' WHERE id = ?", (job_id,))
            conn.execute("SELECT * FROM table_that_does_not_exist")
    assert db.get_job(job_id).label is None


def test_list_and_filter(db: Database):
    a = _insert(db, project="alpha", state=JobState.QUEUED.value)
    b = _insert(db, project="beta", state=JobState.RUNNING.value)
    _insert(db, project="alpha", state=JobState.SUCCEEDED.value)

    assert {j.id for j in db.list_jobs(project="alpha")} == {a, db.list_jobs(project="alpha")[0].id}
    assert len(db.list_jobs(project="alpha")) == 2
    assert [j.id for j in db.list_jobs(states=[JobState.RUNNING.value])] == [b]
    assert len(db.active_jobs()) == 2
    assert db.count_by_state()[JobState.QUEUED.value] == 1


def test_set_error_marks_failed(db: Database):
    job_id = _insert(db, state=JobState.QUEUED.value)
    db.set_error(job_id, "snapshot exploded")
    job = db.get_job(job_id)
    assert job.state_enum is JobState.FAILED
    assert job.error == "snapshot exploded"
    assert job.finished_at is not None


def test_get_job_by_backend_id(db: Database):
    job_id = _insert(db, backend_job_id=99)
    found = db.get_job_by_backend_id("local_dispatcher", 99)
    assert found is not None and found.id == job_id


def test_update_missing_job_raises(db: Database):
    with pytest.raises(KeyError):
        db.update_job(4242, label="nope")


def test_timestamps_are_utc_iso(db: Database):
    job = db.get_job(_insert(db))
    assert job.created_at.endswith("+00:00")
    assert utcnow_iso().endswith("+00:00")


def test_migration_adds_resource_columns(db: Database):
    """v2 adds the resource request columns without touching existing rows."""
    columns = {row["name"] for row in db.conn.execute("PRAGMA table_info(jobs)")}
    assert {"requested_ram_mib", "requested_vram_mib", "requested_cpus"} <= columns


def test_resource_request_roundtrip(db: Database):
    job_id = _insert(db, requested_ram_mib=16384.0, requested_cpus=8)
    job = db.get_job(job_id)
    assert job.requested_ram_mib == 16384.0
    assert job.requested_cpus == 8
    assert job.requested_vram_mib is None  # undeclared -> configured default


def test_upgrade_from_v1_preserves_rows(tmp_path: Path):
    """A database written by the previous schema keeps its jobs on upgrade."""
    from gpuq.db import _MIGRATIONS

    path = tmp_path / "old.sqlite3"
    old = Database(path)
    old.conn.executescript(
        "CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL);"
    )
    old.conn.executescript(_MIGRATIONS[0][1])
    old.conn.execute(
        "INSERT INTO schema_version(version, applied_at) VALUES (1, '2020-01-01T00:00:00+00:00')"
    )
    job_id = _insert(old)
    old.close()

    upgraded = Database(path)
    assert upgraded.initialize() == SCHEMA_VERSION
    job = upgraded.get_job(job_id)
    assert job is not None and job.requested_ram_mib is None
    upgraded.close()
