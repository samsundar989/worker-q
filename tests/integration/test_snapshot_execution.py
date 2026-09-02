"""Mandatory snapshot-execution test (spec section 24.5).

The scenario the whole snapshot design exists for:

1. the repo contains a script printing VALUE = "A";
2. a job is submitted while the queue is blocked by a long-running job;
3. the live repo is then edited to print "B";
4. the queue is released;
5. the job's log must show "A".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from workerq.core import SubmitRequest
from workerq.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _submit(service, repo: Path, argv: list[str], **kwargs) -> int:
    return service.submit(
        SubmitRequest(command=argv, cwd=str(repo), gpus=0, **kwargs)
    ).job.id


def _log_text(service, job_id: int) -> str:
    path = service.resolve_log_path(service.get_job(job_id))
    assert path is not None, f"job #{job_id} produced no log"
    return path.read_text(encoding="utf-8", errors="replace")


def test_queued_job_runs_the_snapshot_not_later_edits(
    live_service, git_repo, git_helper, waiter
):
    (git_repo / "value.py").write_text('VALUE = "A"\n', encoding="utf-8")
    (git_repo / "show.py").write_text(
        "import value\nprint('RESULT:', value.VALUE, flush=True)\n", encoding="utf-8"
    )
    git_helper(["add", "-A"], git_repo)
    git_helper(["commit", "-qm", "value A"], git_repo)

    # 1. block the queue so the job under test cannot start yet.
    blocker = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(4)"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )

    # 2. submit while blocked - this freezes VALUE = "A".
    job_id = _submit(live_service, git_repo, [sys.executable, "show.py"])
    assert live_service.get_job(job_id).state == JobState.QUEUED.value

    # 3. edit the live repo to "B" before the job starts.
    (git_repo / "value.py").write_text('VALUE = "B"\n', encoding="utf-8")
    assert (git_repo / "value.py").read_text(encoding="utf-8") == 'VALUE = "B"\n'

    # 4/5. release the queue and check what actually ran.
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    ), f"job never succeeded (state {live_service.get_job(job_id).state})"

    text = _log_text(live_service, job_id)
    assert "RESULT: A" in text, f"job ran the edited source, not the snapshot:\n{text}"
    assert "RESULT: B" not in text


def test_uncommitted_work_in_progress_is_captured(live_service, git_repo, git_helper, waiter):
    """The user must not have to commit WIP for it to run."""
    (git_repo / "wip.py").write_text("print('WIP-ORIGINAL', flush=True)\n", encoding="utf-8")
    # deliberately never committed

    blocker = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(3)"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )

    job_id = _submit(live_service, git_repo, [sys.executable, "wip.py"])
    (git_repo / "wip.py").write_text("print('WIP-EDITED', flush=True)\n", encoding="utf-8")

    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    )
    text = _log_text(live_service, job_id)
    assert "WIP-ORIGINAL" in text
    assert "WIP-EDITED" not in text


def test_each_queued_job_gets_its_own_snapshot(live_service, git_repo, git_helper, waiter):
    (git_repo / "v.py").write_text("print('V1', flush=True)\n", encoding="utf-8")
    git_helper(["add", "-A"], git_repo)
    git_helper(["commit", "-qm", "v1"], git_repo)

    blocker = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(4)"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )

    first = _submit(live_service, git_repo, [sys.executable, "v.py"])
    (git_repo / "v.py").write_text("print('V2', flush=True)\n", encoding="utf-8")
    second = _submit(live_service, git_repo, [sys.executable, "v.py"])
    (git_repo / "v.py").write_text("print('V3', flush=True)\n", encoding="utf-8")

    for job_id in (first, second):
        assert waiter(
            lambda jid=job_id: live_service.get_job(jid).state == JobState.SUCCEEDED.value,
            timeout=180,
        )

    assert "V1" in _log_text(live_service, first)
    assert "V2" in _log_text(live_service, second)
    assert live_service.get_job(first).snapshot_commit != live_service.get_job(
        second
    ).snapshot_commit


def test_live_worktree_opt_out_sees_later_edits(live_service, git_repo, git_helper, waiter):
    """--live-worktree is an explicit opt-out and must behave as documented."""
    (git_repo / "live.py").write_text("print('BEFORE', flush=True)\n", encoding="utf-8")

    blocker = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(3)"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )

    job_id = _submit(
        live_service, git_repo, [sys.executable, "live.py"], live_worktree=True
    )
    assert live_service.get_job(job_id).snapshot_mode == "live"
    (git_repo / "live.py").write_text("print('AFTER', flush=True)\n", encoding="utf-8")

    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    )
    assert "AFTER" in _log_text(live_service, job_id)


def test_passthrough_data_is_visible_to_the_job(live_service, git_repo, git_helper, waiter):
    data = git_repo / "ignored"
    data.mkdir()
    (data / "dataset.txt").write_text("REAL-DATA\n", encoding="utf-8")
    (git_repo / "read_data.py").write_text(
        "print('DATA:', open('ignored/dataset.txt').read().strip(), flush=True)\n",
        encoding="utf-8",
    )

    job_id = _submit(
        live_service,
        git_repo,
        [sys.executable, "read_data.py"],
        passthrough=["ignored"],
    )
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    ), _log_text(live_service, job_id)
    assert "DATA: REAL-DATA" in _log_text(live_service, job_id)


def test_project_gpuq_toml_passthrough_is_honoured(live_service, git_repo, git_helper, waiter):
    (git_repo / ".gpuq.toml").write_text(
        '[snapshot]\npassthrough = ["ignored"]\n', encoding="utf-8"
    )
    data = git_repo / "ignored"
    data.mkdir()
    (data / "d.txt").write_text("FROM-TOML\n", encoding="utf-8")
    (git_repo / "r.py").write_text(
        "print('GOT:', open('ignored/d.txt').read().strip(), flush=True)\n", encoding="utf-8"
    )

    job_id = _submit(live_service, git_repo, [sys.executable, "r.py"])
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    ), _log_text(live_service, job_id)
    assert "GOT: FROM-TOML" in _log_text(live_service, job_id)


def test_execution_cwd_matches_the_submitted_subdirectory(
    live_service, git_repo, git_helper, waiter
):
    """Submitting from a subdirectory runs at the same depth in the snapshot."""
    sub = git_repo / "experiments" / "exp17"
    sub.mkdir(parents=True)
    (sub / "here.py").write_text("print('IN-SUBDIR', flush=True)\n", encoding="utf-8")

    job_id = _submit(live_service, sub, [sys.executable, "here.py"])
    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    ), _log_text(live_service, job_id)

    job = live_service.get_job(job_id)
    assert job.execution_cwd.endswith(str(Path("experiments") / "exp17"))
    assert "IN-SUBDIR" in _log_text(live_service, job_id)


def test_snapshot_survives_a_repo_level_git_operation(
    live_service, git_repo, git_helper, waiter
):
    """A branch switch after submission must not change what the job runs."""
    (git_repo / "b.py").write_text("print('ON-MAIN', flush=True)\n", encoding="utf-8")
    git_helper(["add", "-A"], git_repo)
    git_helper(["commit", "-qm", "on main"], git_repo)

    blocker = _submit(
        live_service, git_repo, [sys.executable, "-c", "import time; time.sleep(4)"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )

    job_id = _submit(live_service, git_repo, [sys.executable, "b.py"])

    git_helper(["checkout", "-q", "-b", "other"], git_repo)
    (git_repo / "b.py").write_text("print('ON-OTHER', flush=True)\n", encoding="utf-8")
    git_helper(["add", "-A"], git_repo)
    git_helper(["commit", "-qm", "on other"], git_repo)

    assert waiter(
        lambda: live_service.get_job(job_id).state == JobState.SUCCEEDED.value, timeout=180
    )
    assert "ON-MAIN" in _log_text(live_service, job_id)
