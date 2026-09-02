"""Project-level priority: set importance once, not on every submission.

The precedence chain is the delicate part - a project marked urgent must not
silently override a worker that asked for something specific, and clearing a
policy must fall back cleanly rather than leaving jobs stuck at the old level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuq.core import GPUQError, GPUQService, SubmitRequest
from gpuq.models import JobState, Priority


@pytest.fixture
def svc(service: GPUQService) -> GPUQService:
    service.ensure_ready()
    return service


# --------------------------------------------------------------------------
# precedence
# --------------------------------------------------------------------------


def test_falls_back_to_config_default(svc: GPUQService):
    svc.config.core.default_priority = "low"
    assert svc.resolve_priority("anything", None) is Priority.LOW


def test_project_policy_beats_the_config_default(svc: GPUQService):
    svc.config.core.default_priority = "normal"
    svc.set_project_priority("arc-agi", "high")
    assert svc.resolve_priority("arc-agi", None) is Priority.HIGH
    assert svc.resolve_priority("other", None) is Priority.NORMAL


def test_explicit_priority_beats_the_project_policy(svc: GPUQService):
    """A worker asking for something specific must win."""
    svc.set_project_priority("arc-agi", "low")
    assert svc.resolve_priority("arc-agi", "critical") is Priority.CRITICAL


def test_repo_toml_priority_is_honoured(svc: GPUQService, git_repo: Path):
    (git_repo / ".gpuq.toml").write_text(
        '[project]\npriority = "high"\n', encoding="utf-8"
    )
    assert svc.resolve_priority("demo", None, git_repo) is Priority.HIGH


def test_project_policy_beats_repo_toml(svc: GPUQService, git_repo: Path):
    """The machine-wide knob wins, so urgency can change without repo edits."""
    (git_repo / ".gpuq.toml").write_text(
        '[project]\npriority = "low"\n', encoding="utf-8"
    )
    svc.set_project_priority("demo", "critical")
    assert svc.resolve_priority("demo", None, git_repo) is Priority.CRITICAL


def test_a_corrupt_policy_never_breaks_submission(svc: GPUQService):
    """A hand-edited row must degrade to the default, not raise."""
    svc.db.conn.execute(
        "INSERT INTO project_policy(project, priority, updated_at) VALUES (?, ?, ?)",
        ("weird", "urgent-ish", "2020-01-01T00:00:00+00:00"),
    )
    assert svc.resolve_priority("weird", None) is Priority.NORMAL


def test_a_corrupt_repo_toml_priority_is_ignored(svc: GPUQService, git_repo: Path):
    (git_repo / ".gpuq.toml").write_text(
        '[project]\npriority = "very"\n', encoding="utf-8"
    )
    assert svc.resolve_priority("demo", None, git_repo) is Priority.NORMAL


# --------------------------------------------------------------------------
# setting and clearing
# --------------------------------------------------------------------------


def test_set_list_and_clear(svc: GPUQService):
    svc.set_project_priority("biohub", "high", note="deadline friday")
    rows = svc.list_project_priorities()
    assert len(rows) == 1
    assert rows[0]["project"] == "biohub"
    assert rows[0]["priority"] == "high"
    assert rows[0]["note"] == "deadline friday"

    svc.set_project_priority("biohub", None)
    assert svc.list_project_priorities() == []
    assert svc.resolve_priority("biohub", None) is Priority.NORMAL


def test_setting_twice_updates_rather_than_duplicates(svc: GPUQService):
    svc.set_project_priority("biohub", "high")
    svc.set_project_priority("biohub", "critical")
    rows = svc.list_project_priorities()
    assert len(rows) == 1 and rows[0]["priority"] == "critical"


def test_invalid_level_is_rejected(svc: GPUQService):
    with pytest.raises(GPUQError, match="invalid priority"):
        svc.set_project_priority("biohub", "urgent")


# --------------------------------------------------------------------------
# effect on real submissions
# --------------------------------------------------------------------------


def test_submission_inherits_the_project_policy(svc: GPUQService, git_repo: Path):
    svc.set_project_priority(git_repo.name, "high")
    result = svc.submit(
        SubmitRequest(command=["python", "-V"], cwd=str(git_repo), gpus=0)
    )
    assert result.job.priority == "high"


def test_submission_can_still_override(svc: GPUQService, git_repo: Path):
    svc.set_project_priority(git_repo.name, "low")
    result = svc.submit(
        SubmitRequest(
            command=["python", "-V"], cwd=str(git_repo), gpus=0, priority="critical"
        )
    )
    assert result.job.priority == "critical"


def test_invalid_priority_on_submit_is_rejected(svc: GPUQService, git_repo: Path):
    with pytest.raises(GPUQError, match="invalid priority"):
        svc.submit(
            SubmitRequest(
                command=["python", "-V"], cwd=str(git_repo), gpus=0, priority="soon"
            )
        )


def test_raising_a_project_reranks_its_queued_jobs(svc: GPUQService, git_repo: Path):
    """Marking a project urgent must affect work already waiting."""
    first = svc.submit(
        SubmitRequest(command=["python", "1"], cwd=str(git_repo), gpus=0)
    ).job
    second = svc.submit(
        SubmitRequest(command=["python", "2"], cwd=str(git_repo), gpus=0)
    ).job
    assert first.priority == "normal"

    result = svc.set_project_priority(git_repo.name, "critical")
    assert result["requeued"] == 2

    for job_id in (first.id, second.id):
        assert svc.db.get_job(job_id).priority == "critical"

    ranks = {
        row["id"]: row["priority_rank"]
        for row in svc.backend.store.list_all()
        if row["state"] == "QUEUED"
    }
    assert set(ranks.values()) == {0}  # critical


def test_clearing_does_not_rerank_existing_jobs(svc: GPUQService, git_repo: Path):
    """Clearing a policy changes what happens next, not settled decisions."""
    svc.set_project_priority(git_repo.name, "critical")
    job = svc.submit(
        SubmitRequest(command=["python", "-V"], cwd=str(git_repo), gpus=0)
    ).job
    assert job.priority == "critical"

    result = svc.set_project_priority(git_repo.name, None)
    assert result["requeued"] == 0
    assert svc.db.get_job(job.id).priority == "critical"


def test_queue_order_follows_project_priority(svc: GPUQService, git_repo: Path):
    """A raised project overtakes normal work already in the queue."""
    normal = svc.submit(
        SubmitRequest(command=["python", "a"], cwd=str(git_repo), gpus=0)
    ).job

    svc.set_project_priority("urgent-proj", "critical")
    urgent = svc.submit(
        SubmitRequest(
            command=["python", "b"],
            cwd=str(git_repo),
            gpus=0,
            project="urgent-proj",
        )
    ).job
    assert urgent.priority == "critical"

    queued = [r["id"] for r in svc.backend.store.queued()]
    backend_ids = {
        svc.db.get_job(normal.id).backend_job_id: "normal",
        svc.db.get_job(urgent.id).backend_job_id: "urgent",
    }
    assert backend_ids[queued[0]] == "urgent"
