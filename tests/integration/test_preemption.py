"""Preemption, end to end against a real dispatcher.

Preemption is the one place worker-q deliberately destroys work in progress, so
these tests are mostly about what it must *refuse* to do: never touch a job that
did not opt in, never displace equal or higher priority, never kill something
when doing so would not actually let the waiter run, and never lose the
displaced job.
"""

from __future__ import annotations

import sys

import pytest

from workerq.config import PreemptionConfig
from workerq.core import GPUQError, GPUQService, SubmitRequest
from workerq.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _configure(service: GPUQService, **kwargs) -> None:
    defaults = dict(
        enabled=True,
        require_opt_in=True,
        min_runtime_seconds=0,
        max_preemptions=3,
        grace_seconds=2,
    )
    defaults.update(kwargs)
    service.config.preemption = PreemptionConfig(**defaults)
    service.config.save()
    service.backend.shutdown(timeout=20)
    assert service.backend.ensure_daemon(timeout=30)


def _submit(service, repo, argv, **kwargs) -> int:
    return service.submit(
        SubmitRequest(command=argv, cwd=str(repo), gpus=0, **kwargs)
    ).job.id


def _long(seconds: int = 120) -> list[str]:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def _start(service, waiter, job_id: int) -> None:
    assert waiter(
        lambda: service.get_job(job_id).state == JobState.RUNNING.value, timeout=90
    ), f"job #{job_id} never started (state {service.get_job(job_id).state})"


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_higher_priority_displaces_a_preemptible_job(live_service, git_repo, waiter):
    """The feature: a raised job stops lower-priority work and runs."""
    _configure(live_service)

    victim = _submit(live_service, git_repo, _long(), preemptible=True, priority="low")
    _start(live_service, waiter, victim)

    urgent = _submit(
        live_service, git_repo, [sys.executable, "-c", "print('URGENT RAN')"],
        priority="critical",
    )

    assert waiter(
        lambda: live_service.get_job(urgent).state == JobState.SUCCEEDED.value,
        timeout=180,
    ), "the urgent job never ran"

    displaced = live_service.get_job(victim)
    assert displaced.state == JobState.QUEUED.value, (
        f"displaced job should be back in the queue, not {displaced.state}"
    )
    assert displaced.preemption_count == 1
    assert displaced.preempted_by == urgent
    assert "preempted by" in (displaced.preempted_reason or "")

    log = live_service.resolve_log_path(displaced).read_text(
        encoding="utf-8", errors="replace"
    )
    assert "PREEMPTED" in log, "the worker must be able to read why it stopped"

    live_service.cancel(victim, force=True)


def test_the_displaced_job_runs_again_afterwards(live_service, git_repo, waiter):
    """Requeued means requeued: it must actually get another turn."""
    _configure(live_service)

    victim = _submit(
        live_service, git_repo,
        [sys.executable, "-c", "import time; time.sleep(6); print('VICTIM FINISHED')"],
        preemptible=True, priority="low",
    )
    _start(live_service, waiter, victim)

    urgent = _submit(
        live_service, git_repo, [sys.executable, "-c", "print('urgent')"],
        priority="critical",
    )
    assert waiter(
        lambda: live_service.get_job(urgent).is_terminal, timeout=180
    )
    assert waiter(
        lambda: live_service.get_job(victim).state == JobState.SUCCEEDED.value,
        timeout=240,
    ), "the displaced job never got another turn"

    log = live_service.resolve_log_path(live_service.get_job(victim)).read_text(
        encoding="utf-8", errors="replace"
    )
    assert "VICTIM FINISHED" in log
    assert log.count("worker-q: job #") >= 2  # ran twice: once displaced, once through


# --------------------------------------------------------------------------
# what it must refuse to do
# --------------------------------------------------------------------------


def test_a_job_that_did_not_opt_in_is_never_displaced(live_service, git_repo, waiter):
    """The default must be safe: no --preemptible, no interruption."""
    _configure(live_service, require_opt_in=True)

    protected = _submit(live_service, git_repo, _long(), priority="low")
    _start(live_service, waiter, protected)

    _submit(
        live_service, git_repo, [sys.executable, "-c", "print('urgent')"],
        priority="critical",
    )

    # Give the dispatcher plenty of ticks to misbehave.
    assert not waiter(
        lambda: live_service.get_job(protected).state != JobState.RUNNING.value,
        timeout=15,
    ), "a job that did not opt in was displaced"
    assert live_service.get_job(protected).preemption_count == 0

    live_service.cancel(protected, force=True)


def test_equal_priority_never_displaces(live_service, git_repo, waiter):
    _configure(live_service)

    running = _submit(live_service, git_repo, _long(), preemptible=True, priority="normal")
    _start(live_service, waiter, running)

    _submit(live_service, git_repo, [sys.executable, "-c", "pass"], priority="normal")

    assert not waiter(
        lambda: live_service.get_job(running).state != JobState.RUNNING.value,
        timeout=12,
    ), "an equal-priority job displaced a running one"

    live_service.cancel(running, force=True)


def test_lower_priority_never_displaces(live_service, git_repo, waiter):
    _configure(live_service)

    running = _submit(live_service, git_repo, _long(), preemptible=True, priority="high")
    _start(live_service, waiter, running)

    _submit(live_service, git_repo, [sys.executable, "-c", "pass"], priority="low")

    assert not waiter(
        lambda: live_service.get_job(running).state != JobState.RUNNING.value,
        timeout=12,
    ), "a lower-priority job displaced a running one"

    live_service.cancel(running, force=True)


def test_min_runtime_protects_a_job_that_just_started(live_service, git_repo, waiter):
    """A burst of urgent work must not leave nothing making progress."""
    _configure(live_service, min_runtime_seconds=3600)

    fresh = _submit(live_service, git_repo, _long(), preemptible=True, priority="low")
    _start(live_service, waiter, fresh)

    _submit(live_service, git_repo, [sys.executable, "-c", "pass"], priority="critical")

    assert not waiter(
        lambda: live_service.get_job(fresh).state != JobState.RUNNING.value, timeout=12
    ), "a job was displaced before its minimum runtime"

    live_service.cancel(fresh, force=True)


def test_preemption_can_be_disabled_entirely(live_service, git_repo, waiter):
    _configure(live_service, enabled=False)

    running = _submit(live_service, git_repo, _long(), preemptible=True, priority="low")
    _start(live_service, waiter, running)
    _submit(live_service, git_repo, [sys.executable, "-c", "pass"], priority="critical")

    assert not waiter(
        lambda: live_service.get_job(running).state != JobState.RUNNING.value,
        timeout=12,
    ), "preemption fired while disabled"

    live_service.cancel(running, force=True)


# --------------------------------------------------------------------------
# raising priority after submission - the worker-facing lever
# --------------------------------------------------------------------------


def test_bumping_a_queued_job_makes_it_displace_running_work(
    live_service, git_repo, waiter
):
    """Submit normally, then raise it: the whole point of the feature."""
    _configure(live_service)

    victim = _submit(live_service, git_repo, _long(), preemptible=True, priority="low")
    _start(live_service, waiter, victim)

    later = _submit(
        live_service, git_repo, [sys.executable, "-c", "print('BUMPED RAN')"],
        priority="normal",
    )
    assert live_service.get_job(later).state == JobState.QUEUED.value

    result = live_service.bump_job(later, "critical")
    assert result["priority"] == "critical"

    assert waiter(
        lambda: live_service.get_job(later).state == JobState.SUCCEEDED.value,
        timeout=180,
    ), "the bumped job did not overtake and run"
    assert live_service.get_job(victim).preemption_count == 1

    live_service.cancel(victim, force=True)


def test_bump_rejects_a_finished_job(live_service, git_repo, waiter):
    job_id = _submit(live_service, git_repo, [sys.executable, "-c", "pass"])
    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=180)
    with pytest.raises(GPUQError, match="already finished"):
        live_service.bump_job(job_id, "critical")


def test_bump_rejects_an_invalid_level(live_service, git_repo, waiter):
    job_id = _submit(live_service, git_repo, _long())
    try:
        with pytest.raises(GPUQError, match="invalid priority"):
            live_service.bump_job(job_id, "immediately")
    finally:
        live_service.cancel(job_id, force=True)


# --------------------------------------------------------------------------
# tracking a displaced job
# --------------------------------------------------------------------------


def test_preemption_report_explains_the_stop(live_service, git_repo, waiter):
    _configure(live_service)

    victim = _submit(live_service, git_repo, _long(), preemptible=True, priority="low")
    _start(live_service, waiter, victim)
    urgent = _submit(
        live_service, git_repo, [sys.executable, "-c", "print('u')"], priority="critical"
    )
    assert waiter(lambda: live_service.get_job(urgent).is_terminal, timeout=180)
    assert waiter(
        lambda: live_service.get_job(victim).preemption_count == 1, timeout=60
    )

    report = live_service.preemption_report(victim)
    assert report["preemption_count"] == 1
    assert report["preempted_by"] == urgent
    assert report["preemptible"] is True
    assert report["preempted_reason"]

    live_service.cancel(victim, force=True)


def test_wait_returns_the_final_state_across_a_preemption(
    live_service, git_repo, waiter
):
    """A worker can follow one id through being displaced and resumed."""
    _configure(live_service)

    victim = _submit(
        live_service, git_repo,
        [sys.executable, "-c", "import time; time.sleep(5); print('done')"],
        preemptible=True, priority="low",
    )
    _start(live_service, waiter, victim)
    _submit(live_service, git_repo, [sys.executable, "-c", "pass"], priority="critical")

    job = live_service.wait_for(victim, timeout=240, poll=1.0)
    assert job.state == JobState.SUCCEEDED.value
    assert job.preemption_count >= 1
    assert job.id == victim  # the id never changes


def test_wait_times_out_cleanly(live_service, git_repo, waiter):
    job_id = _submit(live_service, git_repo, _long())
    try:
        _start(live_service, waiter, job_id)
        job = live_service.wait_for(job_id, timeout=3, poll=0.5)
        assert not job.is_terminal
        assert job.state == JobState.RUNNING.value
    finally:
        live_service.cancel(job_id, force=True)
