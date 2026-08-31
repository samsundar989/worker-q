"""End-to-end queue behaviour against a real dispatcher daemon.

`test_queue_exclusivity_no_overlap` is the most important non-GPU test in the
suite: it is the direct evidence for the product invariant that two heavy jobs
never run at the same time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gpuq.core import GPUQService, SubmitRequest
from gpuq.models import JobState

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


INTERVAL_SCRIPT = """
import json, os, sys, time
out_dir = sys.argv[1]
name = sys.argv[2]
start = time.time()
time.sleep(float(sys.argv[3]))
end = time.time()
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as fh:
    json.dump({"name": name, "start": start, "end": end}, fh)
print("interval", name, start, end, flush=True)
"""


def _prepare_repo(git_repo: Path, git_helper) -> Path:
    (git_repo / "interval.py").write_text(INTERVAL_SCRIPT, encoding="utf-8")
    git_helper(["add", "-A"], git_repo)
    git_helper(["commit", "-qm", "add interval script"], git_repo)
    return git_repo


def _submit(service: GPUQService, repo: Path, argv: list[str], **kwargs) -> int:
    request = SubmitRequest(command=argv, cwd=str(repo), gpus=0, **kwargs)
    return service.submit(request).job.id


def _await_state(service: GPUQService, job_id: int, states, waiter, timeout=180.0) -> str:
    """Wait for one of `states`, failing fast on any other terminal state."""
    targets = {s.value if hasattr(s, "value") else s for s in states}

    def settled() -> bool:
        job = service.get_job(job_id)
        return job.state in targets or job.is_terminal

    reached = waiter(settled, timeout=timeout)
    job = service.get_job(job_id)
    if not reached or job.state not in targets:
        log = service.resolve_log_path(job)
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:] if log else "(no log)"
        raise AssertionError(
            f"job #{job_id} reached {job.state}, expected one of {targets}\n"
            f"error: {job.error}\n--- log tail ---\n{tail}"
        )
    return job.state


# --------------------------------------------------------------------------
# 24.1 serialization
# --------------------------------------------------------------------------


def test_job_runs_and_transitions_to_succeeded(live_service, git_repo, git_helper, waiter):
    repo = _prepare_repo(git_repo, git_helper)
    job_id = _submit(
        live_service,
        repo,
        [
            sys.executable,
            "-c",
            "import time; print('hello', flush=True); time.sleep(1); print('done', flush=True)",
        ],
    )

    # With an empty queue the dispatcher may already have started it, so the
    # guarantee here is "accepted and active", not "still QUEUED".
    assert live_service.get_job(job_id).state in (
        JobState.QUEUED.value,
        JobState.RUNNING.value,
    )
    _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter)

    job = live_service.get_job(job_id)
    assert job.exit_code == 0
    assert job.started_at and job.finished_at
    assert job.runner_pid

    log = live_service.resolve_log_path(job)
    assert log is not None
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "hello" in text and "done" in text


def test_failing_job_records_exit_code(live_service, git_repo, git_helper, waiter):
    repo = _prepare_repo(git_repo, git_helper)
    job_id = _submit(live_service, repo, [sys.executable, "-c", "import sys; sys.exit(3)"])
    _await_state(live_service, job_id, [JobState.FAILED], waiter)
    assert live_service.get_job(job_id).exit_code == 3


def test_manifest_and_result_are_written(live_service, git_repo, git_helper, waiter):
    repo = _prepare_repo(git_repo, git_helper)
    job_id = _submit(live_service, repo, [sys.executable, "-c", "print('ok')"])
    _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter)

    job_dir = live_service.config.job_dir(job_id)
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gpuq_job_id"] == job_id
    assert manifest["state"] == JobState.SUCCEEDED.value
    assert manifest["snapshot_commit"]

    environment = json.loads((job_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["gpuq_job_id"] == job_id
    assert "gpu_inventory" in environment

    result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    assert result["exit_code"] == 0


def test_awkward_arguments_reach_the_process_intact(
    live_service, git_repo, git_helper, waiter
):
    """Spaces, quotes, globs, '=', unicode and shell metacharacters."""
    repo = _prepare_repo(git_repo, git_helper)
    payload = ["a b", "*.py", 'q"uote', "k=v", "café-日本", "a&b|c>d", "semi;colon"]
    job_id = _submit(
        live_service,
        repo,
        [sys.executable, "-c", "import sys, json; print(json.dumps(sys.argv[1:]))", *payload],
    )
    _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter)

    log = live_service.resolve_log_path(live_service.get_job(job_id))
    text = log.read_text(encoding="utf-8", errors="replace")
    echoed = next(
        json.loads(line)
        for line in text.splitlines()
        if line.startswith("[") and line.endswith("]")
    )
    assert echoed == payload


def test_shell_mode_runs_through_a_shell(live_service, git_repo, git_helper, waiter):
    repo = _prepare_repo(git_repo, git_helper)
    job_id = live_service.submit(
        SubmitRequest(command=[], cwd=str(repo), gpus=0, shell="echo alpha && echo beta")
    ).job.id
    _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter)
    text = live_service.resolve_log_path(live_service.get_job(job_id)).read_text(
        encoding="utf-8", errors="replace"
    )
    assert "alpha" in text and "beta" in text


def test_job_env_is_applied(live_service, git_repo, git_helper, waiter):
    repo = _prepare_repo(git_repo, git_helper)
    job_id = _submit(
        live_service,
        repo,
        [sys.executable, "-c", "import os; print('SEEN', os.environ.get('MY_VAR'))"],
        env={"MY_VAR": "hello world"},
    )
    _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter)
    text = live_service.resolve_log_path(live_service.get_job(job_id)).read_text(
        encoding="utf-8", errors="replace"
    )
    assert "SEEN hello world" in text


# --------------------------------------------------------------------------
# 24.2 queue exclusivity - the critical test
# --------------------------------------------------------------------------


def test_queue_exclusivity_no_overlap(live_service, git_repo, git_helper, waiter, tmp_path):
    """With concurrency 1, job execution intervals must never overlap."""
    repo = _prepare_repo(git_repo, git_helper)
    results = tmp_path / "intervals"
    results.mkdir()

    job_ids = [
        _submit(live_service, repo, [sys.executable, "interval.py", str(results), name, "2"])
        for name in ("first", "second", "third")
    ]

    for job_id in job_ids:
        _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter, timeout=240)

    intervals = [
        json.loads((results / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("first", "second", "third")
    ]
    intervals.sort(key=lambda i: i["start"])

    for earlier, later in zip(intervals, intervals[1:]):
        assert earlier["end"] <= later["start"] + 1e-6, (
            "job intervals overlapped - exclusivity is broken: "
            f"{earlier['name']} ended {earlier['end']}, "
            f"{later['name']} started {later['start']}"
        )


def test_only_one_job_is_ever_running(live_service, git_repo, git_helper, waiter):
    """Sample the queue while work is in flight; never two RUNNING at once."""
    repo = _prepare_repo(git_repo, git_helper)
    job_ids = [
        _submit(live_service, repo, [sys.executable, "-c", "import time; time.sleep(2)"])
        for _ in range(3)
    ]

    observed_running_at_once = 0
    seen_any_running = False

    def poll() -> bool:
        nonlocal observed_running_at_once, seen_any_running
        running = [
            j
            for j in live_service.db.list_jobs(states=[JobState.RUNNING.value])
            if j.id in job_ids
        ]
        observed_running_at_once = max(observed_running_at_once, len(running))
        if running:
            seen_any_running = True
        live_service.reconcile(mutate=True)
        return all(live_service.get_job(i).is_terminal for i in job_ids)

    assert waiter(poll, timeout=240, interval=0.05)
    assert seen_any_running, "never observed a running job"
    assert observed_running_at_once <= 1, (
        f"observed {observed_running_at_once} jobs running simultaneously"
    )


def test_priority_order_is_respected(live_service, git_repo, git_helper, waiter, tmp_path):
    """A critical job queued last must still run before queued normal jobs."""
    repo = _prepare_repo(git_repo, git_helper)
    results = tmp_path / "prio"
    results.mkdir()

    blocker = _submit(
        live_service, repo, [sys.executable, "interval.py", str(results), "blocker", "3"]
    )
    assert waiter(
        lambda: live_service.get_job(blocker).state == JobState.RUNNING.value, timeout=60
    )

    normal = _submit(
        live_service, repo, [sys.executable, "interval.py", str(results), "normal", "0.2"]
    )
    critical = _submit(
        live_service,
        repo,
        [sys.executable, "interval.py", str(results), "critical", "0.2"],
        priority="critical",
    )

    for job_id in (blocker, normal, critical):
        _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter, timeout=240)

    normal_start = json.loads((results / "normal.json").read_text(encoding="utf-8"))["start"]
    critical_start = json.loads((results / "critical.json").read_text(encoding="utf-8"))["start"]
    assert critical_start < normal_start, "critical job did not overtake the normal job"


def test_critical_never_preempts_a_running_job(live_service, git_repo, git_helper, waiter):
    """A new critical job must not interrupt work already in progress."""
    repo = _prepare_repo(git_repo, git_helper)
    running = _submit(live_service, repo, [sys.executable, "-c", "import time; time.sleep(3)"])
    assert waiter(
        lambda: live_service.get_job(running).state == JobState.RUNNING.value, timeout=60
    )

    _submit(live_service, repo, [sys.executable, "-c", "print('urgent')"], priority="critical")

    _await_state(live_service, running, [JobState.SUCCEEDED], waiter, timeout=120)
    assert live_service.get_job(running).exit_code == 0


# --------------------------------------------------------------------------
# 24.3 terminal independence
# --------------------------------------------------------------------------


def test_job_survives_the_submitting_process_exiting(
    live_service, git_repo, git_helper, waiter, tmp_path
):
    """Submit from a subprocess that exits immediately; the job must still run."""
    import subprocess

    repo = _prepare_repo(git_repo, git_helper)
    marker = tmp_path / "survived.txt"

    submitter = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parents[2] / "src")!r})
from gpuq.config import load_config
from gpuq.core import GPUQService, SubmitRequest
service = GPUQService(load_config())
service.ensure_ready()
result = service.submit(SubmitRequest(
    command=[{sys.executable!r}, "-c",
             "import sys, time; time.sleep(1); open(sys.argv[1], 'w').write('yes')",
             {str(marker)!r}],
    cwd={str(repo)!r},
    gpus=0,
))
print(result.job.id)
service.close()
"""
    env = dict(**__import__("os").environ)
    env["GPUQ_STATE_DIR"] = str(live_service.config.state_dir)
    env["GPUQ_CONFIG_FILE"] = str(live_service.config.source_path)

    proc = subprocess.run(
        [sys.executable, "-c", submitter],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    job_id = int(proc.stdout.strip().splitlines()[-1])

    # The submitting process is gone; the job must still complete.
    _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter, timeout=180)
    assert marker.read_text(encoding="utf-8") == "yes"


# --------------------------------------------------------------------------
# logs and status coherence
# --------------------------------------------------------------------------


def test_status_json_stays_coherent_through_a_job_lifecycle(
    live_service, git_repo, git_helper, waiter
):
    repo = _prepare_repo(git_repo, git_helper)
    job_id = _submit(live_service, repo, [sys.executable, "-c", "import time; time.sleep(1)"])

    # An idle dispatcher may already have picked the job up, so assert on the
    # set of states that mean "accepted and not yet finished".
    detail = live_service.job_detail(job_id)
    assert detail["state"] in (JobState.QUEUED.value, JobState.RUNNING.value)
    assert detail["backend_state"] in ("QUEUED", "RUNNING")

    _await_state(live_service, job_id, [JobState.SUCCEEDED], waiter)
    detail = live_service.job_detail(job_id)
    assert detail["state"] == JobState.SUCCEEDED.value
    assert detail["exit_code"] == 0
    assert detail["runtime_seconds"] is not None

    # The runner records the final state as soon as the command exits; the
    # dispatcher reaps the wrapper a moment later. Both converge.
    assert waiter(
        lambda: live_service.job_detail(job_id)["backend_state"] == "FINISHED", timeout=60
    ), "backend never reported the job as finished"
