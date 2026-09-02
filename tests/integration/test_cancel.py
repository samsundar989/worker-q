"""Cancellation behaviour (spec section 24.4).

Covers both halves: a queued job must never execute, and a running job's whole
process tree must actually die - including grandchildren, which is the case a
naive single-PID kill gets wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from workerq.core import SubmitRequest
from workerq.models import JobState
from workerq.winproc import process_creation_time

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _submit(service, repo: Path, argv: list[str], **kwargs) -> int:
    return service.submit(
        SubmitRequest(command=argv, cwd=str(repo), gpus=0, **kwargs)
    ).job.id


def test_cancelling_a_queued_job_prevents_execution(
    live_service, git_repo, git_helper, waiter, tmp_path
):
    marker = tmp_path / "should_not_exist.txt"

    blocker = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(3)"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )

    victim = _submit(
        live_service,
        git_repo,
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')"],
    )
    assert live_service.get_job(victim).state == JobState.QUEUED.value

    result = live_service.cancel(victim)
    assert result["action"] == "removed"
    assert live_service.get_job(victim).state == JobState.CANCELLED.value

    assert waiter(
        lambda: live_service.get_job(blocker).is_terminal, timeout=120
    ), "blocker never finished"
    # Give the dispatcher room to (incorrectly) pick the cancelled job up.
    assert not waiter(lambda: marker.exists(), timeout=8)
    assert live_service.get_job(victim).state == JobState.CANCELLED.value


def test_cancelling_a_running_job_terminates_it(live_service, git_repo, git_helper, waiter):
    job_id = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(300)"]
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.RUNNING.value, timeout=60
    )
    runner_pid = live_service.get_job(job_id).runner_pid
    assert runner_pid

    result = live_service.cancel(job_id, force=True)
    assert result["action"] == "terminating"

    assert waiter(
        lambda: live_service.get_job(job_id).is_terminal, timeout=90
    ), "cancelled job never reached a terminal state"
    assert live_service.get_job(job_id).state == JobState.CANCELLED.value
    assert waiter(lambda: process_creation_time(runner_pid) is None, timeout=30), (
        "runner process survived cancellation"
    )


def test_cancellation_kills_the_whole_process_tree(
    live_service, git_repo, git_helper, waiter, tmp_path
):
    """A detached grandchild must not outlive the cancelled job."""
    pid_file = tmp_path / "grandchild_pid.txt"
    spawner = git_repo / "spawner.py"
    spawner.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "print('spawned', child.pid, flush=True)\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    git_helper(["add", "-A"], git_repo)
    git_helper(["commit", "-qm", "spawner"], git_repo)

    job_id = _submit(live_service, git_repo, [sys.executable, "spawner.py"])
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.RUNNING.value, timeout=60
    )
    assert waiter(lambda: pid_file.exists() and pid_file.read_text().strip(), timeout=60)
    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    assert process_creation_time(grandchild_pid) is not None

    live_service.cancel(job_id, force=True)
    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=90)
    assert waiter(lambda: process_creation_time(grandchild_pid) is None, timeout=60), (
        f"grandchild {grandchild_pid} survived cancellation of its parent job"
    )


def test_cancel_is_idempotent_and_reports_final_state(
    live_service, git_repo, git_helper, waiter
):
    job_id = _submit(live_service, git_repo, [sys.executable, "-c", "print('quick')"])
    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=120)
    final = live_service.get_job(job_id).state

    first = live_service.cancel(job_id)
    second = live_service.cancel(job_id)
    assert first["action"] == "none" and second["action"] == "none"
    assert live_service.get_job(job_id).state == final


def test_cancelled_job_is_never_resurrected(live_service, git_repo, git_helper, waiter):
    """Reconciliation must not walk a CANCELLED job back to an active state."""
    job_id = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(300)"]
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.RUNNING.value, timeout=60
    )
    live_service.cancel(job_id, force=True)
    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=90)

    for _ in range(3):
        live_service.reconcile(mutate=True)
    assert live_service.get_job(job_id).state == JobState.CANCELLED.value


def test_queue_continues_after_a_cancellation(live_service, git_repo, git_helper, waiter):
    long_job = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(300)"]
    )
    assert waiter(
        lambda: live_service.get_job(long_job).state == JobState.RUNNING.value, timeout=60
    )
    follower = _submit(live_service, git_repo, [sys.executable, "-c", "print('next up')"])

    live_service.cancel(long_job, force=True)
    assert waiter(
        lambda: live_service.get_job(follower).state == JobState.SUCCEEDED.value, timeout=180
    ), "the queue stalled after a cancellation"


def test_graceful_cancel_falls_back_to_a_hard_kill(live_service, git_repo, git_helper, waiter):
    """Without --force the grace period elapses, then the tree is killed."""
    job_id = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(300)"]
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.RUNNING.value, timeout=60
    )
    live_service.cancel(job_id, force=False)
    assert waiter(lambda: live_service.get_job(job_id).is_terminal, timeout=120)
    assert live_service.get_job(job_id).state == JobState.CANCELLED.value
