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
