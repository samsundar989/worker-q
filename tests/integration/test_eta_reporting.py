"""Descriptions and self-reported progress, end to end.

The progress protocol only earns its place if a job can use it in one line, and
if the estimate it produces actually reaches the queue view.
"""

from __future__ import annotations

import sys

import pytest

from workerq.core import SubmitRequest
from workerq.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# A job that reports progress with one line of Python, as a real job would.
REPORTER = """
import os, time
p = os.environ["WORKERQ_PROGRESS"]
for i in range(1, 11):
    time.sleep(0.6)
    open(p, "w").write('{"frac": %s, "note": "step %d/10"}' % (i / 10, i))
print("reporter done")
"""


def _submit(service, repo, argv, **kwargs) -> int:
    return service.submit(
        SubmitRequest(command=argv, cwd=str(repo), gpus=0, **kwargs)
    ).job.id


def test_a_job_can_report_its_own_progress(live_service, git_repo, waiter):
    job_id = _submit(
        live_service, git_repo, [sys.executable, "-c", REPORTER], eta_seconds=600.0
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.RUNNING.value, timeout=90
    )

    assert waiter(
        lambda: (live_service.get_job(job_id).progress_fraction or 0) > 0.1, timeout=90
    ), "the runner never picked up the job's progress file"

    job = live_service.get_job(job_id)
    assert job.progress_note and "step" in job.progress_note

    # Progress must now drive the estimate, overriding the declared 10 minutes.
    est = live_service.estimate(job)
    assert est["source"] == "progress"
    assert est["total_seconds"] < 600

    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value,
        timeout=180,
    )


def test_description_and_blocks_reach_the_queue_view(live_service, git_repo, waiter):
    job_id = _submit(
        live_service,
        git_repo,
        [sys.executable, "-c", "import time; time.sleep(3)"],
        describe="120-epoch celltrack train",
        blocks="slice 067 promotion gate",
        eta_seconds=180.0,
    )
    detail = live_service.job_detail(job_id)
    assert detail["description"] == "120-epoch celltrack train"
    assert detail["blocks"] == "slice 067 promotion gate"
    assert detail["estimate"]["source"] == "declared"

    live_service.cancel(job_id, force=True)


def test_eta_can_be_corrected_at_runtime(live_service, git_repo, waiter):
    """A job often only learns its own cost after it starts."""
    job_id = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(20)"]
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.RUNNING.value, timeout=90
    )
    assert live_service.estimate(live_service.get_job(job_id))["source"] == "unknown"

    live_service.annotate_job(job_id, eta_seconds=45.0, description="revised")
    job = live_service.get_job(job_id)
    est = live_service.estimate(job)
    assert est["source"] == "declared"
    assert est["total_seconds"] == 45.0
    assert job.description == "revised"

    live_service.cancel(job_id, force=True)


def test_history_produces_a_learned_estimate(live_service, git_repo, waiter):
    """After a couple of real runs, the same command estimates itself."""
    argv = [sys.executable, "-c", "import time; time.sleep(2); print('ok')"]
    for _ in range(2):
        done = _submit(live_service, git_repo, argv)
        assert waiter(
            lambda j=done: live_service.get_job(j).state == JobState.SUCCEEDED.value,
            timeout=180,
        )

    nxt = _submit(live_service, git_repo, argv)
    est = live_service.estimate(live_service.get_job(nxt))
    assert est["source"] == "learned"
    assert est["samples"] >= 2
    assert 0.5 < est["total_seconds"] < 60

    assert waiter(
        lambda: live_service.get_job(nxt).is_terminal, timeout=180
    )


def test_forecast_orders_queued_jobs_behind_the_running_one(
    live_service, git_repo, waiter
):
    blocker = _submit(
        live_service, git_repo,
        [sys.executable, "-c", "import time; time.sleep(30)"], eta_seconds=30.0,
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=90
    )
    queued = _submit(
        live_service, git_repo, [sys.executable, "-c", "pass"], eta_seconds=10.0
    )

    forecast = live_service.forecast()
    assert forecast[queued.__int__() if hasattr(queued, "__int__") else queued][
        "starts_in_seconds"
    ] is not None
    assert forecast[blocker]["remaining_seconds"] is not None

    live_service.cancel(queued, force=True)
    live_service.cancel(blocker, force=True)
