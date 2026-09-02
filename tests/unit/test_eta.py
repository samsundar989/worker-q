"""Duration estimates and queue forecasting.

The property that matters most here is restraint: an estimate must say where it
came from, and "unknown" must stay unknown rather than being dressed up as a
confident finish time.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from workerq.core import GPUQService
from workerq.db import json_dumps
from workerq.eta import (
    MIN_SAMPLES,
    SOURCE_DECLARED,
    SOURCE_LEARNED,
    SOURCE_PROGRESS,
    SOURCE_UNKNOWN,
    command_signature,
    estimate_job,
    forecast_queue,
    learned_duration,
    read_progress,
)
from workerq.models import JobState
from workerq.util import utcnow, utcnow_iso


# --------------------------------------------------------------------------
# command signature
# --------------------------------------------------------------------------


def test_same_command_shape_shares_a_signature():
    """Different fold, same cost: these must be treated as the same workload."""
    a = command_signature(["C:/x/.venv/Scripts/python.exe", "-m", "celltrack", "train",
                           "--fold", "holdout_44b6", "--epochs", "120"])
    b = command_signature(["C:/x/.venv/Scripts/python.exe", "-m", "celltrack", "train",
                           "--fold", "holdout_6bba", "--epochs", "120"])
    assert a == b and a


def test_different_subcommand_differs():
    train = command_signature(["python", "-m", "celltrack", "train"])
    evaluate = command_signature(["python", "-m", "celltrack", "evaluate"])
    assert train != evaluate


def test_different_flags_differ():
    plain = command_signature(["python", "train.py"])
    resumed = command_signature(["python", "train.py", "--resume"])
    assert plain != resumed


def test_interpreter_path_does_not_change_the_signature():
    """The same work submitted from two machines must group together."""
    a = command_signature(["C:/a/.venv/Scripts/python.exe", "train.py", "--fast"])
    b = command_signature(["/home/u/.venv/bin/python", "train.py", "--fast"])
    assert a == b


def test_empty_command_has_no_signature():
    assert command_signature([]) == ""


def test_shell_mode_uses_the_leading_words():
    a = command_signature(["python train.py --seed 1"], shell_mode=True)
    b = command_signature(["python train.py --seed 2"], shell_mode=True)
    assert a == b


# --------------------------------------------------------------------------
# progress parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content,expected",
    [
        ("0.42", 0.42),
        ("42%", 0.42),
        ("42", 0.42),
        ('{"frac": 0.5}', 0.5),
        ('{"fraction": 0.25}', 0.25),
        ('{"progress": "75%"}', 0.75),
        ("1.0", 1.0),
        ("0", 0.0),
    ],
)
def test_progress_formats(tmp_path, content, expected):
    path = tmp_path / "progress"
    path.write_text(content, encoding="utf-8")
    assert read_progress(str(path))[0] == pytest.approx(expected)


def test_progress_note_is_read(tmp_path):
    path = tmp_path / "progress"
    path.write_text('{"frac": 0.3, "note": "epoch 36/120"}', encoding="utf-8")
    fraction, note = read_progress(str(path))
    assert fraction == pytest.approx(0.3)
    assert note == "epoch 36/120"


@pytest.mark.parametrize("content", ["", "   ", "not a number", "{bad json", "-0.5", "500"])
def test_bad_progress_is_ignored_not_fatal(tmp_path, content):
    path = tmp_path / "progress"
    path.write_text(content, encoding="utf-8")
    assert read_progress(str(path))[0] is None


def test_missing_progress_file(tmp_path):
    assert read_progress(str(tmp_path / "nope"))[0] is None


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------


def _job(service: GPUQService, **overrides):
    values = {
        "backend": "local_dispatcher",
        "project": "demo",
        "priority": "normal",
        "submitted_cwd": str(service.config.state_dir),
        "command_json": json_dumps(["python", "train.py"]),
        "snapshot_mode": "none",
        "host": "testhost",
        "state": JobState.QUEUED.value,
        "command_signature": "sig-demo",
    }
    values.update(overrides)
    return service.db.get_job(service.db.insert_job(**values))


def _ago(seconds: float) -> str:
    return (utcnow() - timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def test_unknown_when_there_is_nothing_to_go_on(service: GPUQService):
    service.ensure_ready()
    est = estimate_job(service, _job(service))
    assert est.source == SOURCE_UNKNOWN
    assert est.remaining_seconds is None
    assert not est.known


def test_declared_eta_is_used(service: GPUQService):
    service.ensure_ready()
    est = estimate_job(service, _job(service, eta_seconds=600.0))
    assert est.source == SOURCE_DECLARED
    assert est.total_seconds == 600.0
    assert est.remaining_seconds == 600.0  # not started yet


def test_declared_eta_counts_down_while_running(service: GPUQService):
    service.ensure_ready()
    job = _job(
        service, state=JobState.RUNNING.value, started_at=_ago(200), eta_seconds=600.0
    )
    est = estimate_job(service, job)
    assert est.remaining_seconds == pytest.approx(400, abs=5)


def test_a_running_job_past_its_eta_reports_zero_not_negative(service: GPUQService):
    service.ensure_ready()
    job = _job(
        service, state=JobState.RUNNING.value, started_at=_ago(900), eta_seconds=600.0
    )
    assert estimate_job(service, job).remaining_seconds == 0.0


def test_progress_beats_a_declared_eta(service: GPUQService):
    """The job knows more about itself than whoever guessed at submit time."""
    service.ensure_ready()
    job = _job(
        service,
        state=JobState.RUNNING.value,
        started_at=_ago(100),
        eta_seconds=600.0,
        progress_fraction=0.5,
    )
    est = estimate_job(service, job)
    assert est.source == SOURCE_PROGRESS
    assert est.total_seconds == pytest.approx(200, abs=10)
    assert est.remaining_seconds == pytest.approx(100, abs=10)


def test_progress_is_ignored_before_it_means_anything(service: GPUQService):
    """1% done after two seconds would extrapolate to nonsense."""
    service.ensure_ready()
    job = _job(
        service,
        state=JobState.RUNNING.value,
        started_at=_ago(100),
        eta_seconds=600.0,
        progress_fraction=0.005,
    )
    assert estimate_job(service, job).source == SOURCE_DECLARED


def test_learned_from_history(service: GPUQService):
    service.ensure_ready()
    for seconds in (100, 120, 140):
        _job(
            service,
            state=JobState.SUCCEEDED.value,
            started_at=_ago(seconds + 10),
            finished_at=_ago(10),
        )
    est = estimate_job(service, _job(service))
    assert est.source == SOURCE_LEARNED
    assert est.samples == 3
    assert est.total_seconds == pytest.approx(120, abs=15)


def test_one_sample_is_not_enough_to_learn(service: GPUQService):
    service.ensure_ready()
    _job(service, state=JobState.SUCCEEDED.value, started_at=_ago(110), finished_at=_ago(10))
    assert MIN_SAMPLES > 1
    assert estimate_job(service, _job(service)).source == SOURCE_UNKNOWN


def test_failed_runs_are_not_learned_from(service: GPUQService):
    """A crash after 5 seconds says nothing about how long success takes."""
    service.ensure_ready()
    for _ in range(4):
        _job(service, state=JobState.FAILED.value, started_at=_ago(15), finished_at=_ago(10))
    assert estimate_job(service, _job(service)).source == SOURCE_UNKNOWN


def test_history_from_another_project_is_not_borrowed(service: GPUQService):
    service.ensure_ready()
    for _ in range(3):
        _job(
            service, project="other", state=JobState.SUCCEEDED.value,
            started_at=_ago(130), finished_at=_ago(10),
        )
    assert estimate_job(service, _job(service)).source == SOURCE_UNKNOWN


def test_stale_history_is_ignored(service: GPUQService):
    service.ensure_ready()
    old = (utcnow() - timedelta(days=90)).isoformat(timespec="microseconds")
    for _ in range(4):
        _job(
            service, state=JobState.SUCCEEDED.value,
            started_at=old, finished_at=old,
        )
    duration, _ = learned_duration(service, _job(service))
    assert duration is None


def test_a_finished_job_has_nothing_remaining(service: GPUQService):
    service.ensure_ready()
    job = _job(
        service, state=JobState.SUCCEEDED.value, started_at=_ago(60), finished_at=_ago(10)
    )
    assert estimate_job(service, job).remaining_seconds == 0.0


def test_estimate_records_its_source_label(service: GPUQService):
    service.ensure_ready()
    for seconds in (100, 110, 120):
        _job(service, state=JobState.SUCCEEDED.value,
             started_at=_ago(seconds + 5), finished_at=_ago(5))
    assert "n=3" in estimate_job(service, _job(service)).label()


# --------------------------------------------------------------------------
# queue forecast
# --------------------------------------------------------------------------


def test_queued_jobs_are_laid_out_behind_the_running_one(service: GPUQService):
    service.ensure_ready()
    service.config.core.max_concurrent_jobs = 1

    running = _job(
        service, state=JobState.RUNNING.value, started_at=_ago(60), eta_seconds=300.0
    )
    first = _job(service, eta_seconds=120.0)
    second = _job(service, eta_seconds=60.0)

    forecast = forecast_queue(service, [running, first, second])
    assert forecast[running.id]["remaining_seconds"] == pytest.approx(240, abs=5)
    # first waits for the running job, second waits for first
    assert forecast[first.id]["starts_in_seconds"] == pytest.approx(240, abs=5)
    assert forecast[second.id]["starts_in_seconds"] == pytest.approx(360, abs=5)


def test_more_slots_let_queued_jobs_start_sooner(service: GPUQService):
    service.ensure_ready()
    service.config.core.max_concurrent_jobs = 3

    running = _job(
        service, state=JobState.RUNNING.value, started_at=_ago(10), eta_seconds=300.0
    )
    queued = _job(service, eta_seconds=60.0)
    forecast = forecast_queue(service, [running, queued])
    assert forecast[queued.id]["starts_in_seconds"] == pytest.approx(0, abs=1)


def test_an_unknown_duration_does_not_fake_a_finish_time(service: GPUQService):
    """A job with no estimate must not silently imply one for those behind it."""
    service.ensure_ready()
    service.config.core.max_concurrent_jobs = 1

    unknown = _job(service, state=JobState.RUNNING.value, started_at=_ago(30))
    behind = _job(service, eta_seconds=60.0)

    forecast = forecast_queue(service, [unknown, behind])
    assert forecast[unknown.id]["remaining_seconds"] is None
    assert forecast[unknown.id]["finish_at"] is None
    assert forecast[behind.id]["starts_in_seconds"] is None
    assert forecast[behind.id]["start_at"] is None


def test_forecast_ignores_finished_jobs(service: GPUQService):
    service.ensure_ready()
    done = _job(
        service, state=JobState.SUCCEEDED.value, started_at=_ago(60), finished_at=_ago(10)
    )
    assert done.id not in forecast_queue(service, [done])


# --------------------------------------------------------------------------
# annotation
# --------------------------------------------------------------------------


def test_annotate_updates_description_blocks_and_eta(service: GPUQService):
    service.ensure_ready()
    job = _job(service)
    service.annotate_job(
        job.id, description="120-epoch train", blocks="slice 067", eta_seconds=900.0
    )
    updated = service.db.get_job(job.id)
    assert updated.description == "120-epoch train"
    assert updated.blocks == "slice 067"
    assert updated.eta_seconds == 900.0


def test_annotate_refuses_a_finished_job(service: GPUQService):
    from workerq.core import GPUQError

    service.ensure_ready()
    job = _job(service, state=JobState.SUCCEEDED.value, finished_at=utcnow_iso())
    with pytest.raises(GPUQError, match="already finished"):
        service.annotate_job(job.id, description="too late")


def test_annotate_requires_something_to_change(service: GPUQService):
    from workerq.core import GPUQError

    service.ensure_ready()
    job = _job(service)
    with pytest.raises(GPUQError, match="nothing to update"):
        service.annotate_job(job.id)


def test_negative_eta_is_rejected(service: GPUQService):
    from workerq.core import GPUQError

    service.ensure_ready()
    job = _job(service)
    with pytest.raises(GPUQError, match="eta must be"):
        service.annotate_job(job.id, eta_seconds=-5)


# --------------------------------------------------------------------------
# upgrading an existing queue
# --------------------------------------------------------------------------


def test_migration_backfills_signatures_for_existing_jobs(service: GPUQService):
    """An upgrade must not throw away the run history already on disk."""
    service.ensure_ready()
    argv = ["python", "-m", "train", "--fold", "a"]
    job = _job(service, command_json=json_dumps(argv))

    # A row as it would look before the column existed.
    service.db.conn.execute(
        "UPDATE jobs SET command_signature = NULL WHERE id = ?", (job.id,)
    )
    service.db.conn.commit()

    service.db.initialize()  # re-running migrations is what an upgrade does

    stored = service.db.conn.execute(
        "SELECT command_signature FROM jobs WHERE id = ?", (job.id,)
    ).fetchone()["command_signature"]
    assert stored == command_signature(argv)


def test_a_blocked_job_does_not_claim_it_starts_immediately(service: GPUQService):
    """"starts ~0s" beside "cannot start" is worse than admitting we cannot say.

    A free slot is not the same as an admissible job: the slot count says
    nothing about whether the machine has the RAM or VRAM the job declared.
    """
    from unittest.mock import patch

    service.ensure_ready()
    service.config.core.max_concurrent_jobs = 4
    queued = _job(service, eta_seconds=60.0)

    with patch.object(service, "queue_wait_reason", return_value=None):
        free = forecast_queue(service, [queued])
    assert free[queued.id]["starts_in_seconds"] == pytest.approx(0, abs=1)

    with patch.object(
        service,
        "queue_wait_reason",
        return_value="needs 24.0 GiB RAM but only 2.0 GiB is free",
    ):
        blocked = forecast_queue(service, [queued])
    assert blocked[queued.id]["starts_in_seconds"] is None
    assert blocked[queued.id]["start_at"] is None
