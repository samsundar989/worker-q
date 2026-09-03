"""Getting the declared footprint right, and fixing it when it is wrong.

A declaration that is too high does not error - the job simply waits for
headroom that may never arrive. That silence is the failure mode these cover:
learning what a command really needs, saying so, and correcting a queued job
without losing its snapshot or its place in the queue.
"""

from __future__ import annotations

import pytest

from workerq.core import GPUQError, GPUQService, SubmitRequest
from workerq.eta import learned_peak_ram, suggested_ram_gb
from workerq.models import JobState

GIB = 1024.0


def _submit(service: GPUQService, **kw) -> int:
    request = SubmitRequest(
        command=["python", "train.py", "--fold", "a"],
        project="decl-test",
        gpus=0,
        snapshot=False,
        **kw,
    )
    return service.submit(request).job.id


def _finish(service: GPUQService, job_id: int, *, peak_gb: float, state: JobState):
    service.db.update_job(
        job_id,
        started_at="2026-09-01T00:00:00+00:00",
        peak_ram_mib=peak_gb * GIB,
        peak_source="measured",
        usage_samples=5,
    )
    service.db.update_job(
        job_id, state=state.value, finished_at="2026-09-01T01:00:00+00:00", exit_code=0
    )


# -- suggestion arithmetic --------------------------------------------------


def test_a_suggestion_leaves_headroom_over_the_worst_run():
    assert suggested_ram_gb(10.0 * GIB) == 15.0
    assert suggested_ram_gb(10.2 * GIB) == 16.0


def test_a_suggestion_is_never_trivially_small():
    """Tiny numbers are noise; the configured default already covers them."""
    assert suggested_ram_gb(0.1 * GIB) == 2.0


# -- learning ---------------------------------------------------------------


def test_nothing_is_suggested_without_history(service: GPUQService):
    service.ensure_ready()
    job_id = _submit(service, ram_gb=28.0)
    peak, runs, _ = learned_peak_ram(service, service.db.get_job(job_id))
    assert peak is None and runs == 0


def test_the_worst_successful_run_is_what_has_to_fit(service: GPUQService):
    service.ensure_ready()
    for gb in (7.0, 10.2, 8.0):
        done = _submit(service, ram_gb=16.0)
        _finish(service, done, peak_gb=gb, state=JobState.SUCCEEDED)

    job_id = _submit(service, ram_gb=28.0)
    peak, runs, provenance = learned_peak_ram(service, service.db.get_job(job_id))
    assert peak == pytest.approx(10.2 * GIB)
    assert runs == 3
    assert provenance == "measured"
    assert suggested_ram_gb(peak) == 16.0


def test_a_cancelled_outlier_does_not_poison_the_suggestion(service: GPUQService):
    """A job cancelled mid-flight may have been doing something else entirely.

    Taking the raw maximum over every run let one such outlier suggest *more*
    than the over-declaration it was meant to correct.
    """
    service.ensure_ready()
    for gb in (7.0, 10.2):
        done = _submit(service, ram_gb=16.0)
        _finish(service, done, peak_gb=gb, state=JobState.SUCCEEDED)
    outlier = _submit(service, ram_gb=16.0)
    _finish(service, outlier, peak_gb=19.2, state=JobState.CANCELLED)

    job_id = _submit(service, ram_gb=28.0)
    peak, runs, _ = learned_peak_ram(service, service.db.get_job(job_id))
    assert peak == pytest.approx(10.2 * GIB)
    assert runs == 2


def test_unfinished_runs_are_ignored(service: GPUQService):
    """A running job's peak is provisional; it must not drive advice."""
    service.ensure_ready()
    running = _submit(service, ram_gb=16.0)
    service.db.update_job(
        running, state=JobState.RUNNING.value, peak_ram_mib=0.2 * GIB, usage_samples=1
    )
    job_id = _submit(service, ram_gb=28.0)
    peak, runs, _ = learned_peak_ram(service, service.db.get_job(job_id))
    assert peak is None and runs == 0


# -- amending a queued job --------------------------------------------------


def test_a_queued_job_declaration_can_be_corrected(service: GPUQService):
    service.ensure_ready()
    job_id = _submit(service, ram_gb=28.0, cpus=3)
    result = service.set_requests(job_id, ram_gb=16.0)
    assert result["ram_gb"] == pytest.approx(16.0)
    assert result["cpus"] == 3, "untouched fields must not be reset"
    assert service.db.get_job(job_id).requested_ram_mib == pytest.approx(16 * GIB)


def test_correcting_requires_at_least_one_value(service: GPUQService):
    service.ensure_ready()
    job_id = _submit(service, ram_gb=28.0)
    with pytest.raises(GPUQError, match="at least one"):
        service.set_requests(job_id)


def test_a_running_job_declaration_is_not_editable(service: GPUQService):
    """Its reservation is already counted against everything else.

    Guarded at the service for a clear message, and again in the queue store
    with `WHERE state = QUEUED` - that second one is what actually holds if a
    job starts between the check and the write.
    """
    from unittest.mock import patch

    service.ensure_ready()
    job_id = _submit(service, ram_gb=28.0)
    job = service.db.get_job(job_id)
    job.state = JobState.RUNNING.value
    with patch.object(service, "get_job", return_value=job):
        with pytest.raises(GPUQError, match="only a QUEUED job"):
            service.set_requests(job_id, ram_gb=16.0)


def test_the_store_refuses_to_edit_a_job_that_is_not_queued(isolated_config):
    """The atomic guard, for the case where a job starts mid-correction."""
    from workerq.backends.base import BACKEND_RUNNING
    from workerq.backends.queue_store import QueueStore

    store = QueueStore(isolated_config.backend_dir / "queue.sqlite3")
    store.initialize()
    try:
        backend_id = store.enqueue(
            ["python", "x.py"],
            label="t",
            gpu_count=0,
            slots=1,
            priority_rank=100,
            log_path=None,
            cwd=None,
            env=None,
            ram_mib=28 * GIB,
        )
        assert store.set_requests(backend_id, ram_mib=16 * GIB) is True
        assert store.get(backend_id)["ram_mib"] == pytest.approx(16 * GIB)

        store.update(backend_id, state=BACKEND_RUNNING)
        assert store.set_requests(backend_id, ram_mib=4 * GIB) is False
        assert store.get(backend_id)["ram_mib"] == pytest.approx(16 * GIB)
    finally:
        store.close()


def test_a_correction_may_not_swap_one_impossible_number_for_another(
    service: GPUQService,
):
    service.ensure_ready()
    service.config.resources.enforce = True
    job_id = _submit(service, ram_gb=8.0)
    try:
        with pytest.raises(GPUQError, match="would never start"):
            service.set_requests(job_id, ram_gb=100_000.0)
    finally:
        service.config.resources.enforce = False


def test_suggest_reports_the_declaration_and_the_evidence(service: GPUQService):
    service.ensure_ready()
    for gb in (7.0, 10.2):
        done = _submit(service, ram_gb=16.0)
        _finish(service, done, peak_gb=gb, state=JobState.SUCCEEDED)
    job_id = _submit(service, ram_gb=28.0)

    data = service.suggest_requests(job_id)
    assert data["declared_ram_gb"] == pytest.approx(28.0)
    assert data["peak_ram_gb"] == pytest.approx(10.2)
    assert data["suggested_ram_gb"] == 16.0
    assert data["runs"] == 2
