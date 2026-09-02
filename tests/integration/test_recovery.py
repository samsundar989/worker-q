"""Crash and restart recovery (spec sections 24.6, 11.9 and 31)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from workerq.core import GPUQService, SubmitRequest
from workerq.db import json_dumps
from workerq.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _submit(service, repo: Path, argv: list[str], **kwargs) -> int:
    return service.submit(
        SubmitRequest(command=argv, cwd=str(repo), gpus=0, **kwargs)
    ).job.id


def test_a_new_process_sees_jobs_submitted_by_another(
    live_service, git_repo, git_helper, waiter
):
    """24.6: a fresh GPUQ process must find the job, its backend id and its log."""
    job_id = _submit(live_service, git_repo, [sys.executable, "-c", "print('from A')"])

    fresh = GPUQService(live_service.config)
    fresh.ensure_ready()
    try:
        job = fresh.get_job(job_id)
        assert job.backend_job_id is not None
        assert job.project == git_repo.name

        assert waiter(lambda: fresh.get_job(job_id).is_terminal, timeout=180)
        assert fresh.get_job(job_id).state == JobState.SUCCEEDED.value

        log = fresh.resolve_log_path(fresh.get_job(job_id))
        assert log is not None and "from A" in log.read_text(encoding="utf-8", errors="replace")
    finally:
        fresh.close()


def test_dispatcher_restart_adopts_a_running_job(live_service, git_repo, git_helper, waiter):
    """A dispatcher restart must not kill or lose an in-flight job."""
    job_id = _submit(
        live_service,
        git_repo,
        [sys.executable, "-c", "import time; time.sleep(6); print('still here', flush=True)"],
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.RUNNING.value, timeout=60
    )
    # The PID the dispatcher launched. On Windows a venv python.exe may be a
    # trampoline that re-execs the interpreter, so this is not necessarily the
    # same PID the runner later reports for itself - both are legitimate.
    backend_job_id = live_service.get_job(job_id).backend_job_id
    assert waiter(
        lambda: live_service.backend.get_job(backend_job_id).pid is not None, timeout=30
    ), "dispatcher never recorded the launched pid"

    assert live_service.backend.shutdown(timeout=20.0), "dispatcher did not stop"
    assert live_service.backend.ensure_daemon(timeout=30.0), "dispatcher did not restart"

    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=180)
    job = live_service.get_job(job_id)
    assert job.state == JobState.SUCCEEDED.value, f"job did not survive the restart: {job.state}"

    # The job ran exactly once. The runner writes one banner per execution, so
    # counting banners is the reliable check - the command text itself is
    # echoed in that banner, which makes counting program output misleading.
    log = live_service.resolve_log_path(job)
    text = log.read_text(encoding="utf-8", errors="replace")
    assert text.count(f"worker-q: job #{job_id} starting") == 1, (
        f"job appears to have run more than once:\n{text}"
    )
    assert "still here" in text

    # The adopted job must be reaped, not left occupying a slot forever.
    assert waiter(
        lambda: live_service.backend.get_job(job.backend_job_id).state == "FINISHED",
        timeout=60,
    ), "adopted job was never reaped by the restarted dispatcher"


def test_queued_jobs_survive_a_dispatcher_restart(live_service, git_repo, git_helper, waiter):
    blocker = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(4)"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )
    queued = _submit(live_service, git_repo, [sys.executable, "-c", "print('ran after restart')"])
    assert live_service.get_job(queued).state == JobState.QUEUED.value

    live_service.backend.shutdown(timeout=20.0)
    assert live_service.backend.ensure_daemon(timeout=30.0)

    assert waiter(
        lambda: live_service.get_job(queued).state == JobState.SUCCEEDED.value, timeout=180
    ), "a queued job was lost across a dispatcher restart"


def test_reconcile_recovers_a_crashed_submission_via_its_label(live_service, git_repo):
    """Crash between backend enqueue and the DB write: the label recovers it."""
    service = live_service
    service.ensure_ready()

    # Simulate a submit that enqueued but never recorded the backend id.
    job_id = service.db.insert_job(
        backend="local_dispatcher",
        project="crashed",
        priority="normal",
        submitted_cwd=str(git_repo),
        command_json=json_dumps([sys.executable, "-c", "print('recovered')"]),
        snapshot_mode="none",
        host="testhost",
        state=JobState.PREPARING.value,
        execution_cwd=str(git_repo),
    )
    backend_job_id = service.backend.submit(
        service.runner_argv(job_id),
        label=service.backend_label(job_id, "crashed", "normal"),
        gpu_count=0,
        log_name=service.config.log_name(job_id),
        cwd=str(git_repo),
    )

    assert service.db.get_job(job_id).backend_job_id is None
    changes = service.reconcile(mutate=True)

    assert any(f"job #{job_id}" in c for c in changes), changes
    assert service.db.get_job(job_id).backend_job_id == backend_job_id


def test_reconcile_is_read_only_in_dry_run(live_service, git_repo, git_helper, waiter):
    job_id = _submit(live_service, git_repo, [sys.executable, "-c", "print('x')"])
    assert waiter(lambda: live_service.get_job(job_id, refresh=False).is_terminal is False)

    before = live_service.db.get_job(job_id).state
    live_service.reconcile(mutate=False)
    assert live_service.db.get_job(job_id).state == before


def test_finished_jobs_are_immutable_under_reconcile(
    live_service, git_repo, git_helper, waiter
):
    job_id = _submit(live_service, git_repo, [sys.executable, "-c", "import sys; sys.exit(7)"])
    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=180)

    job = live_service.get_job(job_id)
    assert job.state == JobState.FAILED.value and job.exit_code == 7
    fingerprint = (job.state, job.exit_code, job.finished_at, job.started_at)

    for _ in range(3):
        live_service.reconcile(mutate=True)
    after = live_service.get_job(job_id)
    assert (after.state, after.exit_code, after.finished_at, after.started_at) == fingerprint


def test_stale_preparing_row_becomes_lost_not_successful(live_service, git_repo):
    """A submission that died during preparation must never look complete."""
    from workerq.util import utcnow_iso
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(timespec="microseconds")
    job_id = live_service.db.insert_job(
        backend="local_dispatcher",
        project="stale",
        priority="normal",
        submitted_cwd=str(git_repo),
        command_json=json_dumps(["python", "-c", "pass"]),
        snapshot_mode="git",
        host="testhost",
        state=JobState.PREPARING.value,
        created_at=old,
        updated_at=old,
        queued_at=old,
    )
    live_service.reconcile(mutate=True)
    job = live_service.db.get_job(job_id)
    assert job.state == JobState.LOST.value
    assert job.error and "did not complete" in job.error
    assert utcnow_iso()


def test_status_stays_coherent_across_processes(live_service, git_repo, git_helper, waiter):
    ids = [
        _submit(live_service, git_repo, [sys.executable, "-c", f"print({n})"]) for n in range(3)
    ]
    for job_id in ids:
        assert waiter(lambda jid=job_id: live_service.get_job(jid).is_terminal, timeout=240)

    fresh = GPUQService(live_service.config)
    fresh.ensure_ready()
    try:
        jobs = {j.id: j for j in fresh.list_jobs(all_jobs=True)}
        for job_id in ids:
            assert jobs[job_id].state == JobState.SUCCEEDED.value
            assert jobs[job_id].exit_code == 0
    finally:
        fresh.close()
