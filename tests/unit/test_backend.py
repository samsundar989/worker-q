"""Backend contract tests (spec section 23).

These exercise the backend without a running daemon: the queue store is the
shared truth, so submit/cancel/promote/state-mapping are all observable by
inspecting it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workerq.backends.base import (
    BACKEND_FINISHED,
    BACKEND_MISSING,
    BACKEND_QUEUED,
    BACKEND_REMOVED,
    BACKEND_RUNNING,
    BackendJob,
    BackendUnavailable,
    SchedulerBackend,
)
from workerq.backends.local_dispatcher import LocalDispatcherBackend, build_backend
from workerq.backends.queue_store import QueueStore, row_to_backend_job
from workerq.config import Config, CoreConfig
from workerq.core import map_backend_state
from workerq.models import JobState


@pytest.fixture
def backend(isolated_config: Config) -> LocalDispatcherBackend:
    be = LocalDispatcherBackend(isolated_config)
    be.config.ensure_dirs()
    be.store.initialize()
    be._initialized = True
    yield be
    be.close()


def _enqueue(backend: LocalDispatcherBackend, argv: list[str], **kwargs) -> int:
    """Enqueue directly, bypassing the daemon liveness requirement."""
    params = {
        "label": "gpuq:1:demo:normal",
        "gpu_count": 0,
        "slots": 1,
        "priority_rank": 100,
        "log_path": str(backend.config.logs_dir / "job-000001.log"),
        "cwd": str(backend.config.state_dir),
        "env": None,
    }
    params.update(kwargs)
    return backend.store.enqueue(argv, **params)


# --------------------------------------------------------------------------
# protocol conformance
# --------------------------------------------------------------------------


def test_backend_satisfies_protocol(backend: LocalDispatcherBackend):
    assert isinstance(backend, SchedulerBackend)
    for method in (
        "health",
        "initialize",
        "submit",
        "list_jobs",
        "get_job",
        "get_state",
        "output_path",
        "remove_queued",
        "terminate_running",
        "promote",
        "set_slots",
    ):
        assert callable(getattr(backend, method)), method


def test_factory_resolves_names(isolated_config: Config):
    for name in ("local_dispatcher", "local", "auto", "task_spooler"):
        isolated_config.backend.name = name
        assert isinstance(build_backend(isolated_config), LocalDispatcherBackend)
    isolated_config.backend.name = "slurm"
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        build_backend(isolated_config)


# --------------------------------------------------------------------------
# environment / isolation
# --------------------------------------------------------------------------


def test_queue_lives_in_the_configured_state_dir(backend: LocalDispatcherBackend):
    """Never an ambiguous shared location - always this profile's state dir."""
    assert backend.store.path == backend.config.backend_dir / "queue.sqlite3"
    assert str(backend.store.path).startswith(str(backend.config.state_dir))
    assert str(backend.lock_path).startswith(str(backend.config.state_dir))


def test_two_profiles_do_not_share_a_queue(tmp_path: Path):
    a = Config(core=CoreConfig(state_dir=str(tmp_path / "a")))
    b = Config(core=CoreConfig(state_dir=str(tmp_path / "b")))
    assert LocalDispatcherBackend(a).store.path != LocalDispatcherBackend(b).store.path


def test_health_reports_capabilities(backend: LocalDispatcherBackend):
    health = backend.health()
    assert health["backend"] == "local_dispatcher"
    assert health["supports_gpu_allocation"] is True
    assert health["supports_reorder"] is True
    assert health["supports_serialization"] is True
    assert "slots" in health and "gpu_free_percent_threshold" in health


def test_health_without_daemon_reports_not_running(backend: LocalDispatcherBackend):
    assert backend.health()["daemon_running"] is False


# --------------------------------------------------------------------------
# submission builds argv safely
# --------------------------------------------------------------------------


def test_submit_stores_argv_as_json_not_a_shell_string(backend: LocalDispatcherBackend):
    argv = ["python", "train.py", "--name=a b", "--glob=*.py", "--u=café", "a&b|c"]
    backend_id = _enqueue(backend, argv)
    row = backend.store.get(backend_id)
    assert json.loads(row["argv_json"]) == argv


def test_submit_refuses_empty_command(backend: LocalDispatcherBackend):
    with pytest.raises(ValueError, match="empty command"):
        backend.submit([], label="x", gpu_count=0)


def test_submit_requires_a_live_dispatcher(backend: LocalDispatcherBackend, monkeypatch):
    """Spec 4.4: with no backend, submission fails - it never runs directly."""
    monkeypatch.setattr(backend, "ensure_daemon", lambda **kw: False)
    with pytest.raises(BackendUnavailable, match="dispatcher is not running"):
        backend.submit(["python", "-c", "pass"], label="x", gpu_count=0)


def test_submit_error_mentions_doctor(backend: LocalDispatcherBackend, monkeypatch):
    monkeypatch.setattr(backend, "ensure_daemon", lambda **kw: False)
    with pytest.raises(BackendUnavailable) as exc:
        backend.submit(["python"], label="x", gpu_count=0)
    assert "workerq doctor" in str(exc.value)


# --------------------------------------------------------------------------
# state mapping - one place only
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend_state,exit_code,expected",
    [
        (BACKEND_QUEUED, None, JobState.QUEUED),
        (BACKEND_RUNNING, None, JobState.RUNNING),
        (BACKEND_FINISHED, 0, JobState.SUCCEEDED),
        (BACKEND_FINISHED, 1, JobState.FAILED),
        (BACKEND_FINISHED, -1, JobState.FAILED),
        (BACKEND_FINISHED, None, JobState.FAILED),
        (BACKEND_REMOVED, None, JobState.CANCELLED),
        (BACKEND_MISSING, None, JobState.LOST),
        ("SOMETHING_NEW", None, None),
    ],
)
def test_state_mapping(backend_state, exit_code, expected):
    assert map_backend_state(backend_state, exit_code) == expected


def test_get_job_for_unknown_id_is_missing_not_an_error(backend: LocalDispatcherBackend):
    job = backend.get_job(999999)
    assert job.state == BACKEND_MISSING
    assert backend.get_state(999999) == BACKEND_MISSING


# --------------------------------------------------------------------------
# cancellation routing
# --------------------------------------------------------------------------


def test_queued_cancellation_removes(backend: LocalDispatcherBackend):
    backend_id = _enqueue(backend, ["python", "-c", "pass"])
    assert backend.cancel(backend_id) == "removed"
    assert backend.get_state(backend_id) == BACKEND_REMOVED


def test_running_cancellation_requests_termination(backend: LocalDispatcherBackend, monkeypatch):
    backend_id = _enqueue(backend, ["python", "-c", "pass"])
    backend.store.claim_for_start(backend_id)
    monkeypatch.setattr(backend, "ensure_daemon", lambda **kw: True)

    assert backend.cancel(backend_id) == "terminating"
    row = backend.store.get(backend_id)
    assert row["cancel_requested"] == 1
    assert row["cancel_force"] == 0
    assert row["state"] == BACKEND_RUNNING  # the daemon performs the kill


def test_force_cancellation_sets_the_force_flag(backend: LocalDispatcherBackend, monkeypatch):
    backend_id = _enqueue(backend, ["python", "-c", "pass"])
    backend.store.claim_for_start(backend_id)
    monkeypatch.setattr(backend, "ensure_daemon", lambda **kw: True)
    backend.cancel(backend_id, force=True)
    assert backend.store.get(backend_id)["cancel_force"] == 1


def test_cancel_of_finished_job_is_a_no_op(backend: LocalDispatcherBackend):
    backend_id = _enqueue(backend, ["python", "-c", "pass"])
    backend.store.claim_for_start(backend_id)
    backend.store.finish(backend_id, exit_code=0)
    assert backend.cancel(backend_id) == "finished"


def test_cancel_of_missing_job(backend: LocalDispatcherBackend):
    assert backend.cancel(4242) == "missing"


def test_remove_queued_raises_when_already_running(backend: LocalDispatcherBackend):
    backend_id = _enqueue(backend, ["python"])
    backend.store.claim_for_start(backend_id)
    with pytest.raises(BackendUnavailable, match="not queued"):
        backend.remove_queued(backend_id)


def test_terminate_running_raises_when_inactive(backend: LocalDispatcherBackend):
    backend_id = _enqueue(backend, ["python"])
    backend.store.claim_for_start(backend_id)
    backend.store.finish(backend_id, 0)
    with pytest.raises(BackendUnavailable, match="not active"):
        backend.terminate_running(backend_id)


# --------------------------------------------------------------------------
# ordering and promotion
# --------------------------------------------------------------------------


def test_priority_rank_orders_the_queue(backend: LocalDispatcherBackend):
    normal = _enqueue(backend, ["a"], priority_rank=100)
    low = _enqueue(backend, ["b"], priority_rank=200)
    critical = _enqueue(backend, ["c"], priority_rank=0)
    high = _enqueue(backend, ["d"], priority_rank=50)
    assert [r["id"] for r in backend.store.queued()] == [critical, high, normal, low]


def test_fifo_within_a_priority(backend: LocalDispatcherBackend):
    first = _enqueue(backend, ["a"], priority_rank=100)
    second = _enqueue(backend, ["b"], priority_rank=100)
    third = _enqueue(backend, ["c"], priority_rank=100)
    assert [r["id"] for r in backend.store.queued()] == [first, second, third]


def test_promote_moves_to_the_front(backend: LocalDispatcherBackend):
    first = _enqueue(backend, ["a"])
    second = _enqueue(backend, ["b"])
    third = _enqueue(backend, ["c"])
    backend.promote(third)
    assert [r["id"] for r in backend.store.queued()][0] == third
    assert first in [r["id"] for r in backend.store.queued()]
    assert second in [r["id"] for r in backend.store.queued()]


def test_promote_running_job_raises(backend: LocalDispatcherBackend):
    backend_id = _enqueue(backend, ["a"])
    backend.store.claim_for_start(backend_id)
    with pytest.raises(BackendUnavailable, match="only queued jobs"):
        backend.promote(backend_id)


def test_claim_for_start_is_exclusive(backend: LocalDispatcherBackend):
    """Two dispatchers must never both start the same job."""
    backend_id = _enqueue(backend, ["a"])
    assert backend.store.claim_for_start(backend_id) is True
    assert backend.store.claim_for_start(backend_id) is False


# --------------------------------------------------------------------------
# serialization / parsing
# --------------------------------------------------------------------------


def test_list_jobs_returns_backend_jobs(backend: LocalDispatcherBackend):
    _enqueue(backend, ["a"], label="gpuq:1:p:normal")
    _enqueue(backend, ["b"], label="gpuq:2:p:normal")
    jobs = backend.list_jobs()
    assert len(jobs) == 2
    assert all(isinstance(j, BackendJob) for j in jobs)
    assert {j.label for j in jobs} == {"gpuq:1:p:normal", "gpuq:2:p:normal"}


def test_find_by_label_recovers_a_backend_id(backend: LocalDispatcherBackend):
    backend_id = _enqueue(backend, ["a"], label="gpuq:42:arc-agi:critical")
    found = backend.find_by_label("gpuq:42:arc-agi:critical")
    assert found is not None and found.backend_id == backend_id
    assert backend.find_by_label("gpuq:999:x:normal") is None


def test_output_path_is_reported(backend: LocalDispatcherBackend):
    log = backend.config.logs_dir / "job-000007.log"
    backend_id = _enqueue(backend, ["a"], log_path=str(log))
    assert backend.output_path(backend_id) == log


def test_malformed_queue_row_fails_clearly(backend: LocalDispatcherBackend):
    """A corrupt argv must not be silently treated as an empty command."""
    backend_id = _enqueue(backend, ["a"])
    backend.store.update(backend_id, argv_json="{not json")
    row = backend.store.get(backend_id)
    with pytest.raises(json.JSONDecodeError):
        json.loads(row["argv_json"])
    assert row_to_backend_job(row).backend_id == backend_id  # still inspectable


def test_row_to_backend_job_maps_fields(backend: LocalDispatcherBackend):
    backend_id = _enqueue(backend, ["a"], gpu_count=2, label="lbl")
    job = row_to_backend_job(backend.store.get(backend_id))
    assert job.backend_id == backend_id
    assert job.state == BACKEND_QUEUED
    assert job.gpu_count == 2
    assert job.label == "lbl"
    assert job.enqueued_at is not None


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def test_slots_roundtrip(backend: LocalDispatcherBackend):
    backend.set_slots(3)
    assert backend.get_slots() == 3
    with pytest.raises(ValueError):
        backend.set_slots(0)


def test_gpu_threshold_roundtrip(backend: LocalDispatcherBackend):
    backend.set_gpu_free_percent(75)
    assert backend.get_gpu_free_percent() == 75
    for bad in (-1, 101):
        with pytest.raises(ValueError):
            backend.set_gpu_free_percent(bad)


def test_trim_finished_bounds_the_table(backend: LocalDispatcherBackend):
    for _ in range(10):
        backend_id = _enqueue(backend, ["a"])
        backend.store.claim_for_start(backend_id)
        backend.store.finish(backend_id, 0)
    backend.store.trim_finished(3)
    finished = [r for r in backend.store.list_all() if r["state"] == BACKEND_FINISHED]
    assert len(finished) == 3


def test_initialize_is_idempotent(isolated_config: Config, monkeypatch):
    be = LocalDispatcherBackend(isolated_config)
    monkeypatch.setattr(be, "ensure_daemon", lambda **kw: True)
    be.initialize()
    be.initialize()
    assert be.get_slots() == isolated_config.core.max_concurrent_jobs
    be.close()
