"""Choosing which machine a job runs on.

Phase 5 of docs/multi-node.md. The rule is about **contention, not fit**: the
worker runs the same job more slowly, so moving work there is a win only when
it buys an earlier start for something else.

The measure is a counterfactual, not queue depth. Depth is the wrong question -
a queue full of 30 GiB jobs is not a reason to exile a small one, because
moving it frees nothing they can use.
"""

from __future__ import annotations

import pytest

from workerq.backends.dispatcher import Dispatcher

GIB = 1024.0


@pytest.fixture
def dispatcher(isolated_config):
    isolated_config.resources.enforce = True
    isolated_config.resources.reserve_cpus = 2
    isolated_config.resources.reserve_ram_gb = 8.0
    isolated_config.gpu.free_memory_threshold_percent = 0
    d = Dispatcher(isolated_config)
    d.store.initialize()
    try:
        yield d
    finally:
        d.store.close()


def row(backend_id: int, *, cpus: int = 0, ram_gb: float = 0.0, pinned: str | None = None):
    return {
        "id": backend_id,
        "cpus": cpus,
        "ram_mib": ram_gb * GIB,
        "vram_mib": 0.0,
        "gpu_count": 0,
        "gpu_mode": "exclusive",
        "pinned_node": pinned,
    }


def test_a_job_that_blocks_a_later_one_should_move(dispatcher, monkeypatch):
    """The row that is the whole feature.

    Sixteen usable CPUs less a 2-CPU reserve is 14. An 8-CPU job and a 10-CPU
    job cannot both run here, but either can run alone - so moving the first
    lets the second start now instead of in half an hour.
    """
    monkeypatch.setattr(dispatcher, "_running_requests", lambda: [])
    queued = [row(1, cpus=8), row(2, cpus=10)]
    assert dispatcher._would_block_the_queue(queued[0], queued, 0)


def test_a_job_with_nothing_behind_it_stays(dispatcher, monkeypatch):
    """Moving it would only make it slower on a slower machine."""
    monkeypatch.setattr(dispatcher, "_running_requests", lambda: [])
    queued = [row(1, cpus=8)]
    assert not dispatcher._would_block_the_queue(queued[0], queued, 0)


def test_a_queue_of_jobs_that_could_not_run_anyway_is_not_contention(
    dispatcher, monkeypatch
):
    """Depth is the wrong measure, stated as a test.

    The job behind needs more CPUs than the machine has at all. Moving the one
    in front frees nothing it can use, so exiling it to the slower machine buys
    nobody anything.
    """
    monkeypatch.setattr(dispatcher, "_running_requests", lambda: [])
    queued = [row(1, cpus=4), row(2, cpus=99)]
    assert not dispatcher._would_block_the_queue(queued[0], queued, 0)


def test_a_job_that_fits_alongside_is_not_contention(dispatcher, monkeypatch):
    """Both run here regardless, so there is nothing to buy."""
    monkeypatch.setattr(dispatcher, "_running_requests", lambda: [])
    queued = [row(1, cpus=4), row(2, cpus=4)]
    assert not dispatcher._would_block_the_queue(queued[0], queued, 0)


def test_a_pinned_job_behind_does_not_argue_for_moving_this_one(
    dispatcher, monkeypatch
):
    """A job pinned elsewhere is not waiting on local headroom."""
    monkeypatch.setattr(dispatcher, "_running_requests", lambda: [])
    queued = [row(1, cpus=8), row(2, cpus=10, pinned="somewhere")]
    assert not dispatcher._would_block_the_queue(queued[0], queued, 0)


def test_placement_is_off_when_no_node_is_registered(dispatcher):
    """A single-machine install must behave exactly as it always did."""
    chosen, why = dispatcher._choose_node(row(1, cpus=8), [row(1, cpus=8)], 0, True)
    assert chosen is None and why is None


def test_placement_can_be_switched_off(dispatcher):
    """The escape hatch, if placement ever misbehaves."""
    dispatcher.config.scheduling.auto_placement = False
    chosen, _ = dispatcher._choose_node(row(1, cpus=8), [row(1, cpus=8)], 0, True)
    assert chosen is None


def test_a_job_with_no_snapshot_is_never_placed_elsewhere(dispatcher):
    """No commit to ship means nothing to reproduce there."""
    from workerq.config import NodeConfig

    dispatcher.config.nodes = [NodeConfig(name="w", address="h")]
    candidate = row(1, cpus=8)  # no remote_spec_json
    chosen, _ = dispatcher._choose_node(candidate, [candidate], 0, True)
    assert chosen is None


def test_placement_notes_are_logged_once_per_decision(dispatcher):
    """This loop ticks four times a second.

    An unthrottled explanation here would reproduce the 303,164 identical log
    lines that commit 16b2846 had to fix.
    """
    written: list[str] = []
    dispatcher.log = lambda message: written.append(message)  # type: ignore[method-assign]
    for _ in range(5):
        dispatcher._placement_note(7, "keeping local: moving it would not free anything")
    assert len(written) == 1
    dispatcher._placement_note(7, "something else happened")
    assert len(written) == 2


# --------------------------------------------------------------------------
# Refusing a node that is reachable but wrong (phase 6)
# --------------------------------------------------------------------------


def _report(**kw):
    from workerq import nodes as nodemod

    base = dict(name="w", protocol=nodemod.NODE_PROTOCOL_VERSION, version="1.3.0")
    base.update(kw)
    return nodemod.NodeReport(**base)


def test_a_protocol_mismatch_stops_dispatch(dispatcher):
    """Reachable and roomy is not the same as safe to send work to.

    The wire format is one worker-q's JSON parsed by another, so a version gap
    produces a parse error *after* the work has been queued on the far side.
    """
    from workerq import nodes as nodemod
    from workerq.config import NodeConfig

    node = NodeConfig(name="w", address="h")
    remote = _report(protocol=nodemod.NODE_PROTOCOL_VERSION + 1)
    ok, why = dispatcher._node_usable(node, remote)
    assert not ok and "protocol" in why


def test_a_differing_version_alone_does_not_stop_dispatch(dispatcher):
    """That check already failed to notice a real skew, so it may not refuse."""
    from workerq.config import NodeConfig

    ok, _ = dispatcher._node_usable(NodeConfig(name="w", address="h"), _report(version="9.9.9"))
    assert ok


def test_a_badly_skewed_clock_stops_dispatch(dispatcher):
    """Collecting a finished job's output compares file times against the node.

    A clock minutes out silently collects the wrong files, which is worse than
    refusing: the job reports success and the results are wrong.
    """
    from datetime import timedelta

    from workerq.config import NodeConfig
    from workerq.util import utcnow

    remote = _report(remote_time=(utcnow() + timedelta(minutes=10)).isoformat())
    ok, why = dispatcher._node_usable(NodeConfig(name="w", address="h"), remote)
    assert not ok and "clock" in why


def test_a_slightly_off_clock_is_tolerated(dispatcher):
    """This is not about precision, only about being wrong enough to matter."""
    from datetime import timedelta

    from workerq.config import NodeConfig
    from workerq.util import utcnow

    remote = _report(remote_time=(utcnow() + timedelta(seconds=5)).isoformat())
    ok, _ = dispatcher._node_usable(NodeConfig(name="w", address="h"), remote)
    assert ok


def test_a_node_that_reports_no_time_is_not_refused_for_it(dispatcher):
    """Unknown is not the same as wrong."""
    from workerq.config import NodeConfig

    ok, _ = dispatcher._node_usable(NodeConfig(name="w", address="h"), _report())
    assert ok


def test_a_drained_node_is_not_offered_to_placement(dispatcher):
    """Drain is for maintenance: stop sending work, do not kill any."""
    from workerq.config import NodeConfig

    dispatcher.config.nodes = [NodeConfig(name="w", address="h", enabled=False)]
    chosen, _ = dispatcher._choose_node(row(1, cpus=8), [row(1, cpus=8)], 0, True)
    assert chosen is None


# --------------------------------------------------------------------------
# A full machine is a fact about this machine, not about the queue
# --------------------------------------------------------------------------


def _enqueue_travelling(dispatcher, *, cpus: int = 1) -> int:
    """A queued job that has somewhere else it could run."""
    return dispatcher.store.enqueue(
        ["python", "-c", "pass"],
        label=None,
        gpu_count=0,
        slots=1,
        priority_rank=100,
        log_path=None,
        cwd=None,
        env=None,
        ram_mib=GIB,
        cpus=cpus,
        remote_spec={"repo_root": "C:/Users/samsu/Documents/x", "outputs": []},
    )


def test_a_full_machine_still_sends_work_to_another(dispatcher, monkeypatch):
    """The bug: local slot exhaustion ended the scan for the whole queue.

    A remote job consumes no local slot - the dispatcher says so itself - so
    returning here idles the second machine at exactly the moment the queue is
    fullest, which is when it is most needed.
    """
    dispatcher.config.core.max_concurrent_jobs = 1
    job = _enqueue_travelling(dispatcher)
    dispatcher.adopted[999] = object()  # the one slot is occupied

    offered: list[tuple[int, bool]] = []
    placed: list[int] = []

    def fake_choose(row, queued, position, local_ok):
        offered.append((int(row["id"]), local_ok))
        return object(), None

    def fake_start_remote(row, node):
        placed.append(int(row["id"]))
        return True

    monkeypatch.setattr(dispatcher, "_choose_node", fake_choose)
    monkeypatch.setattr(dispatcher, "_start_remote", fake_start_remote)
    dispatcher._start_ready_jobs()

    assert offered == [(job, False)], "offered for placement, and told it cannot run here"
    assert placed == [job], "and actually placed, rather than left queued"


def test_a_full_machine_never_starts_a_job_locally(dispatcher, monkeypatch):
    """The guard that must survive the fix: no slot still means no local start."""
    dispatcher.config.core.max_concurrent_jobs = 1
    job = _enqueue_travelling(dispatcher)
    dispatcher.adopted[999] = object()

    started: list[int] = []
    monkeypatch.setattr(dispatcher, "_choose_node", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        dispatcher, "_start_job", lambda row, devices: started.append(int(row["id"])) or True
    )
    dispatcher._start_ready_jobs()

    assert started == [], "a job must never start without a slot"
    row = dispatcher.store.get(job)
    assert "free slot" in str(row["wait_reason"]), row["wait_reason"]


# --------------------------------------------------------------------------
# Holding the queue here must not idle the other machine
# --------------------------------------------------------------------------


def test_jobs_past_the_backfill_limit_are_still_offered_elsewhere(dispatcher, monkeypatch):
    """The scan used to `return` once backfill had skipped its limit.

    That limit protects this machine's headroom, and a job placed on another
    machine uses none of it - but every job past it went unoffered, so the
    3080 Ti idled behind a full queue of work it could run.
    """
    dispatcher.config.core.max_concurrent_jobs = 1
    dispatcher.config.scheduling.backfill_max_skip = 1
    dispatcher.adopted[999] = object()
    jobs = [_enqueue_travelling(dispatcher) for _ in range(4)]

    offered: list[int] = []
    monkeypatch.setattr(
        dispatcher, "_choose_node",
        lambda row, queued, position, local_ok: offered.append(int(row["id"])) or (None, "busy"),
    )
    dispatcher._start_ready_jobs()
    assert offered == jobs


def test_a_held_queue_still_places_later_jobs_elsewhere(dispatcher, monkeypatch):
    """Holding drains this machine for the head; it is no reason to idle another."""
    dispatcher.config.core.max_concurrent_jobs = 1
    dispatcher.adopted[999] = object()
    head = _enqueue_travelling(dispatcher)
    later = _enqueue_travelling(dispatcher)
    # The head has waited past the threshold, so the hold engages.
    dispatcher._blocked[head] = (0.0, "blocked", float("inf"))
    dispatcher.config.scheduling.backfill_head_wait_seconds = 0

    placed: list[int] = []
    monkeypatch.setattr(
        dispatcher, "_choose_node",
        lambda row, queued, position, local_ok: (object(), None)
        if int(row["id"]) == later else (None, "no room"),
    )
    monkeypatch.setattr(
        dispatcher, "_start_remote", lambda row, node: placed.append(int(row["id"])) or True
    )
    started: list[int] = []
    monkeypatch.setattr(
        dispatcher, "_start_job", lambda row, devices: started.append(int(row["id"])) or True
    )
    dispatcher._start_ready_jobs()
    assert placed == [later]
    assert started == []


def test_a_failed_placement_backs_off_instead_of_retrying_every_tick(dispatcher):
    """Each attempt is several SSH round trips on the dispatch loop."""
    from workerq.config import NodeConfig

    node = NodeConfig(name="w", address="h")
    assert dispatcher._placement_backoff_reason(7, node) is None
    dispatcher._note_placement_failed(7, node)
    assert "retrying" in dispatcher._placement_backoff_reason(7, node)
    first = dispatcher._placement_failures[(7, "w")][1]
    dispatcher._note_placement_failed(7, node)
    assert dispatcher._placement_failures[(7, "w")][1] == first * 2
    assert dispatcher._placement_backoff_reason(8, node) is None, "per job, not per node"


def test_a_placed_job_is_charged_against_the_cached_node_report(dispatcher):
    """Otherwise one tick sends several jobs to a node with room for one."""
    import time

    from workerq import nodes as nodemod
    from workerq.config import NodeConfig

    node = NodeConfig(name="w", address="h")
    dispatcher._reports = nodemod.ReportCache()
    report = _report()
    dispatcher._reports._reports["w"] = (time.monotonic(), report)
    dispatcher._account_remote_start(node, row(1, cpus=3, ram_gb=5.0))
    assert len(report.running) == 1
    assert report.running[0].cpus == 3


def test_a_node_with_work_already_waiting_is_sent_no_more(dispatcher, monkeypatch):
    from workerq.config import NodeConfig

    monkeypatch.setattr(dispatcher, "_node_report", lambda node: _report(queued=2))
    ok, why = dispatcher._remote_admits(NodeConfig(name="w", address="h"), row(1, cpus=1))
    assert not ok and "waiting" in why


# --------------------------------------------------------------------------
# A remote job survives a restart and can be cancelled
# --------------------------------------------------------------------------


def _remote_running(dispatcher, *, cancel: bool = False) -> int:
    job = _enqueue_travelling(dispatcher)
    assert dispatcher.store.claim_for_remote_start(job, "w", 42)
    if cancel:
        dispatcher.store.conn.execute(
            "UPDATE bjobs SET cancel_requested = 1 WHERE id = ?", (job,)
        )
    return job


def test_a_restart_does_not_fail_a_job_running_on_another_machine(dispatcher):
    """There is no local process to find, and its absence means nothing."""
    job = _remote_running(dispatcher)
    dispatcher._recover_orphans()
    assert dispatcher.store.get(job)["state"] == "RUNNING"


def test_cancelling_a_remote_job_asks_the_node_once(dispatcher, monkeypatch):
    """It used to log "cannot verify pid None" forever while the job ran on."""
    from workerq import remote as remotemod
    from workerq.config import NodeConfig

    dispatcher.config.nodes = [NodeConfig(name="w", address="h")]
    job = _remote_running(dispatcher, cancel=True)
    calls: list[int] = []
    monkeypatch.setattr(
        remotemod, "cancel", lambda node, remote_id, force=False: calls.append(remote_id) or True
    )
    dispatcher._service_cancellations()
    dispatcher._service_cancellations()
    assert calls == [42]
    assert dispatcher.store.get(job)["state"] == "RUNNING", "finished by the reaper, not here"


def test_small_missing_passthrough_is_sent_rather_than_refusing_the_project(
    dispatcher, monkeypatch, tmp_path
):
    from workerq import staging
    from workerq.config import NodeConfig

    (tmp_path / "engine" / "bin").mkdir(parents=True)
    (tmp_path / "engine" / "bin" / "kagx.pyd").write_text("x", encoding="utf-8")
    pushed: list[list[str]] = []
    monkeypatch.setattr(
        staging, "push_passthrough", lambda node, repo, missing: pushed.append(missing) or 1
    )
    monkeypatch.setattr(
        staging, "inspect_repo",
        lambda node, repo, passthrough: staging.RepoStatus(
            node="w", project="p", remote_path="x", exists=True,
            passthrough={"engine/bin": True},
        ),
    )
    ok, why = dispatcher._push_missing(
        NodeConfig(name="w", address="h"), tmp_path, {"passthrough": ["engine/bin"]},
        ["engine/bin"],
    )
    assert ok and why is None and pushed == [["engine/bin"]]


def test_large_missing_passthrough_still_refuses(dispatcher, monkeypatch, tmp_path):
    from workerq import staging
    from workerq.config import NodeConfig

    monkeypatch.setattr(staging, "pushable_passthrough", lambda repo, missing: None)
    ok, why = dispatcher._push_missing(
        NodeConfig(name="w", address="h"), tmp_path, {}, ["weights"]
    )
    assert not ok and "missing weights" in why
