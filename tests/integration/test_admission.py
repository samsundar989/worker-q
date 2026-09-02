"""Resource-aware admission, end to end against a real dispatcher.

The behaviour that keeps the machine alive: a job whose declared footprint
cannot fit stays QUEUED with an explanation, and starts as soon as it fits.
"""

from __future__ import annotations

import sys

import pytest

from gpuq import host
from gpuq.config import ResourcesConfig
from gpuq.core import GPUQError, GPUQService, SubmitRequest
from gpuq.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _enable(service: GPUQService, **kwargs) -> None:
    """Turn admission control on with values pinned to this machine."""
    defaults = dict(
        enforce=True,
        default_ram_gb=0.1,
        default_vram_gb=0.0,
        default_cpus=0,
        reserve_ram_gb=0.5,
        reserve_vram_gb=0.0,
        reserve_cpus=0,
        max_commit_percent=100,
        min_host_free_percent=0,
    )
    defaults.update(kwargs)
    service.config.resources = ResourcesConfig(**defaults)
    service.config.save()


def _submit(service: GPUQService, repo, argv, **kwargs) -> int:
    return service.submit(
        SubmitRequest(command=argv, cwd=str(repo), gpus=0, **kwargs)
    ).job.id


def test_small_job_still_runs_with_enforcement_on(live_service, git_repo, waiter):
    _enable(live_service)
    live_service.backend.shutdown(timeout=20)
    assert live_service.backend.ensure_daemon(timeout=30)

    job_id = _submit(
        live_service, git_repo, [sys.executable, "-c", "print('ok')"], ram_gb=0.05
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    )


def test_impossible_request_is_rejected_at_submit(live_service, git_repo):
    """Better an immediate error than a job queued forever."""
    _enable(live_service)
    total_gb = (host.memory().total_mib or 0) / 1024.0
    with pytest.raises(GPUQError, match="would never start"):
        _submit(
            live_service,
            git_repo,
            [sys.executable, "-c", "pass"],
            ram_gb=total_gb * 4,
        )


def test_job_waits_when_ram_is_unavailable_and_says_why(
    live_service, git_repo, waiter
):
    """A job that cannot fit stays QUEUED with a readable reason."""
    free_gb = (host.memory().available_mib or 0) / 1024.0
    # Ask for more than is free, but less than the machine's usable total, so
    # the request is legal yet cannot be admitted right now.
    _enable(live_service, reserve_ram_gb=0.5, min_host_free_percent=0)
    live_service.backend.shutdown(timeout=20)
    assert live_service.backend.ensure_daemon(timeout=30)

    job_id = _submit(
        live_service,
        git_repo,
        [sys.executable, "-c", "print('should not run yet')"],
        ram_gb=free_gb + 8.0,
    )

    def blocked_with_reason() -> bool:
        job = live_service.get_job(job_id)
        return (
            job.state == JobState.QUEUED.value
            and bool(live_service.queue_wait_reason(job))
        )

    assert waiter(blocked_with_reason, timeout=60), (
        f"job never reported a wait reason (state "
        f"{live_service.get_job(job_id).state})"
    )
    reason = live_service.queue_wait_reason(live_service.get_job(job_id))
    assert "RAM" in reason
    assert live_service.get_job(job_id).state == JobState.QUEUED.value

    # Relaxing the limit must let it through, proving it was the gate.
    _enable(live_service, default_ram_gb=0.1, reserve_ram_gb=0.5, enforce=False)
    live_service.backend.shutdown(timeout=20)
    assert live_service.backend.ensure_daemon(timeout=30)
    assert waiter(
        lambda: live_service.get_job(job_id).is_terminal, timeout=180
    ), "job did not start once enforcement was lifted"


def test_declared_footprint_is_recorded_and_visible(live_service, git_repo, waiter):
    _enable(live_service)
    job_id = _submit(
        live_service,
        git_repo,
        [sys.executable, "-c", "print('hi')"],
        ram_gb=1.5,
        cpus=2,
    )
    job = live_service.get_job(job_id)
    assert job.requested_ram_mib == pytest.approx(1.5 * 1024)
    assert job.requested_cpus == 2

    detail = live_service.job_detail(job_id)
    assert detail["requested_ram_mib"] == pytest.approx(1.5 * 1024)
    assert detail["requested_cpus"] == 2


def test_telemetry_records_samples_while_running(live_service, git_repo, waiter):
    """`gpuq report` needs this history to explain a failure afterwards."""
    from gpuq.telemetry import open_telemetry

    job_id = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(2)"]
    )
    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=180)

    store = open_telemetry(live_service.config.state_dir)
    try:
        assert waiter(lambda: store.latest_sample() is not None, timeout=60), (
            "dispatcher recorded no resource samples"
        )
        sample = store.latest_sample()
        assert sample["host_free_percent"] is not None
        events = store.recent_events()
        assert any(e["kind"] == "job_started" for e in events)
    finally:
        store.close()
