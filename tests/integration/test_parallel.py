"""Several jobs running at once, when their declared footprints fit.

The counterpart to `test_queue_exclusivity_no_overlap`: that one proves the
queue serialises at one slot, these prove it packs when allowed to. The slot
count is only a ceiling here - admission control is what decides whether a job
actually starts, so these tests turn enforcement on rather than relying on the
suite-wide default of off.
"""

from __future__ import annotations

import sys

import pytest

from workerq.config import ResourcesConfig
from workerq.core import GPUQService, SubmitRequest
from workerq.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _sleeper(seconds: float = 20.0) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def _submit(service: GPUQService, name: str, **kw) -> int:
    request = SubmitRequest(
        command=_sleeper(kw.pop("seconds", 20.0)),
        project="parallel-test",
        label=name,
        gpus=0,
        snapshot=False,
        **kw,
    )
    return service.submit(request).job.id


def _running_ids(service: GPUQService) -> set[int]:
    return {j.id for j in service.db.list_jobs(states=[JobState.RUNNING.value])}


def _cleanup(service: GPUQService, ids: list[int]) -> None:
    for job_id in ids:
        try:
            service.cancel(job_id, force=True)
        except Exception:
            pass


def test_two_small_jobs_run_at_the_same_time(live_service: GPUQService, waiter):
    """The whole point: a slot count no longer decides this, capacity does."""
    live_service.backend.set_slots(3)
    ids = [_submit(live_service, f"small-{i}", ram_gb=1.0, cpus=1) for i in range(2)]
    try:
        assert waiter(lambda: len(_running_ids(live_service)) >= 2, timeout=60), (
            f"expected 2 jobs running, saw {_running_ids(live_service)}"
        )
    finally:
        _cleanup(live_service, ids)
        live_service.backend.set_slots(1)


def test_the_slot_ceiling_is_still_obeyed(live_service: GPUQService, waiter):
    """Capacity decides what runs, but never more than the configured ceiling."""
    live_service.backend.set_slots(2)
    ids = [_submit(live_service, f"cap-{i}", ram_gb=1.0, cpus=1) for i in range(4)]
    try:
        assert waiter(lambda: len(_running_ids(live_service)) >= 2, timeout=60)
        # Give the dispatcher several more ticks to misbehave.
        assert not waiter(
            lambda: len(_running_ids(live_service)) > 2, timeout=8
        ), "slot ceiling exceeded"
    finally:
        _cleanup(live_service, ids)
        live_service.backend.set_slots(1)


def test_a_job_that_cannot_fit_does_not_block_one_that_can(
    live_service: GPUQService, waiter, tmp_path
):
    """Backfill. Without it the oversized job at the head parks the queue."""
    from workerq import host

    total_gb = (host.memory().total_mib or 0.0) / 1024.0
    live_service.config.resources = ResourcesConfig(
        enforce=True,
        reserve_ram_gb=1.0,
        reserve_cpus=1,
        min_host_free_percent=0,
        max_commit_percent=100,
        commit_soft_percent=100,
    )
    live_service.config.save()
    live_service.backend.shutdown(timeout=20)
    assert live_service.backend.ensure_daemon(timeout=30)
    live_service.backend.set_slots(3)

    # Asks for nearly the whole machine: it can never be admitted.
    huge = _submit(live_service, "huge", ram_gb=max(4.0, total_gb - 2.0), cpus=1)
    small = _submit(live_service, "small", ram_gb=1.0, cpus=1)
    try:
        assert waiter(lambda: small in _running_ids(live_service), timeout=60), (
            "the small job never started; the oversized one blocked the queue"
        )
        huge_job = live_service.db.get_job(huge)
        assert huge_job.state == JobState.QUEUED.value
        reason = live_service.queue_wait_reason(huge_job)
        assert reason, "a passed-over job must still say why it is waiting"
    finally:
        _cleanup(live_service, [huge, small])
        live_service.backend.set_slots(1)


def test_every_waiting_job_explains_itself(live_service: GPUQService, waiter):
    """With N slots, a blank reason is indistinguishable from 'nobody looked'."""
    live_service.backend.set_slots(1)
    ids = [_submit(live_service, f"queued-{i}", ram_gb=1.0, cpus=1) for i in range(3)]
    try:
        assert waiter(lambda: len(_running_ids(live_service)) == 1, timeout=60)

        def _all_explained() -> bool:
            queued = live_service.db.list_jobs(states=[JobState.QUEUED.value])
            queued = [j for j in queued if j.id in ids]
            return bool(queued) and all(
                live_service.queue_wait_reason(j) for j in queued
            )

        assert waiter(_all_explained, timeout=30), "a queued job had no wait reason"
    finally:
        _cleanup(live_service, ids)
