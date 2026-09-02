"""Host inspection, telemetry and failure classification.

These back `gpuq top` and `gpuq report`: the tools that answer "is the queue
healthy?" and "what killed my job, and whose workload was it?".
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest

from gpuq import host
from gpuq.core import GPUQService
from gpuq.db import json_dumps
from gpuq.models import JobState
from gpuq.report import CAUSES, analyse, classify_failure
from gpuq.telemetry import EVENT_BLOCKED, EVENT_STARTED, Telemetry, open_telemetry
from gpuq.util import utcnow, utcnow_iso

IS_WINDOWS = os.name == "nt"


# --------------------------------------------------------------------------
# host inspection
# --------------------------------------------------------------------------


def test_memory_reports_plausible_numbers():
    mem = host.memory()
    assert mem.error is None
    assert mem.total_mib and mem.total_mib > 512
    assert mem.available_mib is not None
    assert 0 <= (mem.free_percent or 0) <= 100
    assert mem.used_mib is not None and mem.used_mib >= 0


@pytest.mark.skipif(not IS_WINDOWS, reason="commit charge is a Windows concept")
def test_commit_charge_is_reported():
    mem = host.memory()
    assert mem.commit_limit_mib and mem.commit_limit_mib > 0
    assert 0 <= (mem.commit_percent or 0) <= 100


def test_top_processes_are_sorted_and_cached():
    first = host.top_processes(5)
    assert first, "expected at least one process"
    assert all(
        first[i].memory_mib >= first[i + 1].memory_mib for i in range(len(first) - 1)
    )
    assert host.top_processes(5) == first  # served from cache


def test_parent_map_contains_this_process():
    mapping = host.parent_map()
    assert os.getpid() in mapping


def test_descendants_include_the_root_and_children():
    import subprocess
    import sys
    import time

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        time.sleep(1.0)
        host.parent_map(ttl=0.0)  # refresh past the cache
        found = host.descendants_of({proc.pid})
        assert proc.pid in found
        assert len(found) >= 1
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_descendants_of_nothing_is_empty():
    assert host.descendants_of(set()) == set()


def test_pressure_summary(tmp_path):
    healthy = host.HostMemory(
        total_mib=64000, available_mib=40000, commit_limit_mib=80000, commit_used_mib=30000
    )
    assert host.summarize_pressure(healthy)[0] is False

    tight = host.HostMemory(
        total_mib=64000, available_mib=1000, commit_limit_mib=80000, commit_used_mib=30000
    )
    under, reason = host.summarize_pressure(tight)
    assert under and "host RAM" in (reason or "")

    committed = host.HostMemory(
        total_mib=64000, available_mib=40000, commit_limit_mib=80000, commit_used_mib=78000
    )
    under, reason = host.summarize_pressure(committed)
    assert under and "commit" in (reason or "")


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------


@pytest.fixture
def telemetry(tmp_path: Path) -> Telemetry:
    store = open_telemetry(tmp_path)
    yield store
    store.close()


def test_sample_roundtrip(telemetry: Telemetry):
    telemetry.record_sample(
        gpu_used_mib=1024.0,
        gpu_free_percent=90.0,
        host_free_percent=55.0,
        commit_percent=61.0,
        running_job_id=7,
        queued_count=2,
        top_consumers=[{"pid": 1, "name": "python.exe", "mib": 4096}],
    )
    latest = telemetry.latest_sample()
    assert latest is not None
    assert latest["running_job_id"] == 7
    assert latest["commit_percent"] == 61.0
    assert "python.exe" in latest["top_consumers_json"]


def test_events_are_queryable(telemetry: Telemetry):
    telemetry.record_event(EVENT_STARTED, backend_job_id=1, detail="pid 5")
    telemetry.record_event(EVENT_BLOCKED, backend_job_id=2, detail="no RAM")
    blocked = telemetry.recent_events(kinds=[EVENT_BLOCKED])
    assert len(blocked) == 1
    assert blocked[0]["detail"] == "no RAM"
    assert len(telemetry.recent_events()) == 2


def test_sample_near_finds_the_closest_in_window(telemetry: Telemetry):
    telemetry.record_sample(host_free_percent=42.0)
    when = utcnow_iso()
    found = telemetry.sample_near(when, window_seconds=60)
    assert found is not None and found["host_free_percent"] == 42.0


def test_sample_near_rejects_distant_samples(telemetry: Telemetry):
    telemetry.record_sample(host_free_percent=42.0)
    long_ago = (utcnow() - timedelta(days=2)).isoformat(timespec="microseconds")
    assert telemetry.sample_near(long_ago, window_seconds=60) is None


def test_peak_between_reports_worst_case(telemetry: Telemetry):
    start = utcnow_iso()
    for free in (60.0, 15.0, 45.0):
        telemetry.record_sample(host_free_percent=free, commit_percent=100.0 - free)
    peak = telemetry.peak_between(start, utcnow_iso())
    assert peak is not None
    assert peak["min_host_free_percent"] == 15.0
    assert peak["max_commit_percent"] == 85.0


def test_prune_bounds_growth(telemetry: Telemetry):
    for _ in range(50):
        telemetry.record_sample(host_free_percent=50.0)
    telemetry.prune(keep_samples=10, keep_events=10)
    remaining = telemetry.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    assert remaining == 10


def test_telemetry_never_raises_on_a_broken_store(tmp_path: Path):
    """Losing telemetry must never take the dispatcher down with it."""
    store = Telemetry(tmp_path / "t.sqlite3")
    store.initialize()
    store.conn.execute("DROP TABLE samples")
    store.record_sample(host_free_percent=1.0)  # must not raise
    assert store.latest_sample() is None
    store.close()


# --------------------------------------------------------------------------
# failure classification
# --------------------------------------------------------------------------


def _job(service: GPUQService, **overrides):
    values = {
        "backend": "local_dispatcher",
        "project": "demo",
        "priority": "normal",
        "submitted_cwd": str(service.config.state_dir),
        "command_json": json_dumps(["python", "train.py"]),
        "snapshot_mode": "none",
        "host": "testhost",
        "state": JobState.FAILED.value,
        "exit_code": 1,
        "submitter_agent": "claude-code",
        "finished_at": utcnow_iso(),
        "started_at": utcnow_iso(),
    }
    values.update(overrides)
    return service.db.get_job(service.db.insert_job(**values))


@pytest.mark.parametrize(
    "log,expected",
    [
        ("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate", "cuda_oom"),
        ("MemoryError: could not read frame 78; the host is out of memory", "host_oom"),
        ("RuntimeError: Cannot allocate memory", "host_oom"),
        ("The paging file is too small for this operation to complete", "host_oom"),
        ("bash: line 1: C:/Program: No such file or directory", "missing_file"),
        ("ModuleNotFoundError: No module named 'torch'", "import_error"),
        ("ImportError: DLL load failed while importing", "import_error"),
        ("Traceback (most recent call last):\nValueError: bad config", "app_error"),
        ("everything was fine until it was not", "nonzero"),
    ],
)
def test_classification(service: GPUQService, log: str, expected: str):
    job = _job(service)
    assert classify_failure(service, job, log).key == expected


def test_cuda_oom_beats_generic_memory_text(service: GPUQService):
    """Specific causes must win over the generic ones."""
    job = _job(service)
    log = "MemoryError\nCUDA out of memory. Tried to allocate 2.00 GiB"
    assert classify_failure(service, job, log).key == "cuda_oom"


def test_exit_127_is_a_missing_command(service: GPUQService):
    job = _job(service, exit_code=127)
    assert classify_failure(service, job, "").key == "missing_file"


def test_cancelled_and_lost_are_not_bugs(service: GPUQService):
    cancelled = _job(service, state=JobState.CANCELLED.value)
    assert classify_failure(service, cancelled, "").key == "cancelled"
    lost = _job(service, state=JobState.LOST.value)
    assert classify_failure(service, lost, "").key == "killed"


def test_resource_causes_are_flagged_as_machine_problems():
    assert CAUSES["cuda_oom"].resource and CAUSES["host_oom"].resource
    assert not CAUSES["app_error"].resource
    assert not CAUSES["missing_file"].resource
    assert CAUSES["host_oom"].advice


def test_analyse_groups_and_reaches_a_verdict(service: GPUQService, tmp_path: Path):
    log_dir = service.config.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    oom = _job(service, project="biohub")
    service.config.log_path(oom.id).write_text(
        "MemoryError: the host is out of memory\n", encoding="utf-8"
    )
    service.db.update_job(oom.id, log_path=str(service.config.log_path(oom.id)))

    bug = _job(service, project="arc")
    service.config.log_path(bug.id).write_text(
        "Traceback (most recent call last):\nValueError: nope\n", encoding="utf-8"
    )
    service.db.update_job(bug.id, log_path=str(service.config.log_path(bug.id)))

    data = analyse(service, hours=24)
    assert data["counts"]["total_failures"] == 2
    assert data["counts"]["resource_caused"] == 1
    assert data["counts"]["by_project"]["biohub"] == 1
    assert data["counts"]["by_agent"]["claude-code"] == 2
    assert "biohub" in data["verdict"]

    causes = {f["job_id"]: f["cause"] for f in data["failures"]}
    assert causes[oom.id] == "host_oom"
    assert causes[bug.id] == "app_error"

    excerpts = {f["job_id"]: f["excerpt"] for f in data["failures"]}
    assert "MemoryError" in excerpts[oom.id]


def test_analyse_with_no_failures_is_clear(service: GPUQService):
    _job(service, state=JobState.SUCCEEDED.value, exit_code=0)
    data = analyse(service, hours=24)
    assert data["counts"]["total_failures"] == 0
    assert "No failures" in data["verdict"]


def test_analyse_respects_the_window(service: GPUQService):
    old = (utcnow() - timedelta(days=5)).isoformat(timespec="microseconds")
    _job(service, finished_at=old, updated_at=old)
    assert analyse(service, hours=1)["counts"]["total_failures"] == 0


# --------------------------------------------------------------------------
# throughput
# --------------------------------------------------------------------------


def test_throughput_counts_outcomes(service: GPUQService):
    _job(service, state=JobState.SUCCEEDED.value, exit_code=0)
    _job(service, state=JobState.SUCCEEDED.value, exit_code=0)
    _job(service, state=JobState.FAILED.value, exit_code=1)
    _job(service, state=JobState.CANCELLED.value)

    stats = service.throughput(hours=24)
    assert stats["succeeded"] == 2
    assert stats["failed"] == 1
    assert stats["cancelled"] == 1
    assert stats["finished"] == 4
    assert stats["success_rate"] == pytest.approx(200 / 3, rel=1e-3)


def test_throughput_on_an_empty_queue(service: GPUQService):
    stats = service.throughput(hours=24)
    assert stats["finished"] == 0
    assert stats["success_rate"] == 100.0
    assert stats["median_wait_seconds"] is None
