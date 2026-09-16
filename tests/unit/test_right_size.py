"""Reserving what a command uses rather than what it habitually declares.

The measured problem: the median job used half its declaration, and the most
common reason a job waited was other jobs' reservations rather than a shortage.
The danger in fixing it is starving a job whose size the signature cannot see -
`--workers 2` and `--workers 8` share one - so history is applied as a fraction
of each run's own declaration.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from workerq.core import GPUQService, SubmitRequest
from workerq.models import JobState
from workerq.util import utcnow

GIB = 1024.0


def _submit(service: GPUQService, ram_gb: float, **kw):
    return service.submit(
        SubmitRequest(
            command=["python", "arms.py", "--workers", "3"],
            project="rs-test",
            gpus=0,
            snapshot=False,
            ram_gb=ram_gb,
            **kw,
        )
    )


def _finish(service, job_id, *, peak_gb, state=JobState.SUCCEEDED, source="measured", samples=5):
    finished = utcnow() - timedelta(hours=1)
    service.db.update_job(
        job_id,
        started_at=(finished - timedelta(minutes=10)).isoformat(),
        peak_ram_mib=peak_gb * GIB,
        peak_source=source,
        usage_samples=samples,
    )
    service.db.update_job(
        job_id, state=state.value, finished_at=finished.isoformat(), exit_code=0
    )


def _history(service, n, *, declared=16.0, peak=4.0, **kw):
    for _ in range(n):
        _finish(service, _submit(service, declared, right_size=False).job.id, peak_gb=peak, **kw)


def test_a_habitually_padded_declaration_is_trimmed(service: GPUQService):
    _history(service, 5, declared=16.0, peak=4.0)  # uses 25% of what it declares
    result = _submit(service, 16.0)
    job = service.db.get_job(result.job.id)
    assert job.requested_ram_mib == pytest.approx(6.0 * GIB)  # 16 x 25% x 1.5
    assert job.declared_ram_mib == pytest.approx(16.0 * GIB)
    assert any("reserving 6.0 GiB" in a for a in result.advisories)


def test_a_bigger_declaration_keeps_its_scale(service: GPUQService):
    """Eight workers declared as 32 GiB are not squeezed into two workers' peak."""
    _history(service, 5, declared=16.0, peak=4.0)
    job = service.db.get_job(_submit(service, 32.0).job.id)
    assert job.requested_ram_mib == pytest.approx(12.0 * GIB)


def test_too_little_history_changes_nothing(service: GPUQService):
    _history(service, 4)
    job = service.db.get_job(_submit(service, 16.0).job.id)
    assert job.requested_ram_mib == pytest.approx(16.0 * GIB)
    assert job.declared_ram_mib is None


@pytest.mark.parametrize(
    "kw",
    [
        {"state": JobState.FAILED},
        {"source": "estimated"},
        {"samples": 1},
    ],
)
def test_only_trustworthy_runs_count(service: GPUQService, kw):
    _history(service, 5, **kw)
    job = service.db.get_job(_submit(service, 16.0).job.id)
    assert job.requested_ram_mib == pytest.approx(16.0 * GIB)


def test_the_worst_run_decides_and_small_savings_are_not_taken(service: GPUQService):
    _history(service, 4, peak=4.0)
    _history(service, 1, peak=9.0)  # 56% x 1.5 = 84% of 16: under a 20% saving
    job = service.db.get_job(_submit(service, 16.0).job.id)
    assert job.requested_ram_mib == pytest.approx(16.0 * GIB)


def test_trimmed_runs_do_not_undo_the_trimming(service: GPUQService):
    """History is read against the declaration, not the trimmed reservation.

    Otherwise a run reserved at 6 GiB that peaked at 4 looks 67% used, and the
    next submission would reserve the full 16 again.
    """
    _history(service, 5, declared=16.0, peak=4.0)
    for _ in range(5):
        _finish(service, _submit(service, 16.0).job.id, peak_gb=4.0)
    job = service.db.get_job(_submit(service, 16.0).job.id)
    assert job.requested_ram_mib == pytest.approx(6.0 * GIB)


def test_exact_resources_opts_out(service: GPUQService):
    _history(service, 5)
    job = service.db.get_job(_submit(service, 16.0, right_size=False).job.id)
    assert job.requested_ram_mib == pytest.approx(16.0 * GIB)


def test_it_can_be_switched_off(service: GPUQService):
    _history(service, 5)
    service.config.resources.auto_right_size = False
    job = service.db.get_job(_submit(service, 16.0).job.id)
    assert job.requested_ram_mib == pytest.approx(16.0 * GIB)


def test_never_below_the_floor(service: GPUQService):
    _history(service, 5, declared=16.0, peak=0.2)
    job = service.db.get_job(_submit(service, 16.0).job.id)
    assert job.requested_ram_mib == pytest.approx(2.0 * GIB)
