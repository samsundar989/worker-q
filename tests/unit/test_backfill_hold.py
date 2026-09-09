"""Holding the queue for the job at its head, and when that is pointless.

Backfill lets a small job past a large blocked one. The starvation guard stops
that becoming permanent: once the head has waited long enough, the queue is
held so the machine drains and the head gets a clear run at it.

That trade only pays when worker-q's own jobs are what stand in the way. With
nothing running there is nothing to drain, and holding buys the head nothing
while stalling everything behind it.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from workerq.backends.dispatcher import Dispatcher
from workerq.host import HostMemory

GIB = 1024.0


@pytest.fixture
def dispatcher(isolated_config):
    isolated_config.resources.enforce = True
    isolated_config.core.max_concurrent_jobs = 6
    isolated_config.scheduling.backfill = True
    isolated_config.scheduling.backfill_head_wait_seconds = 900
    isolated_config.scheduling.backfill_max_hold_seconds = 1800
    d = Dispatcher(isolated_config)
    d.store.initialize()
    try:
        yield d
    finally:
        d.store.close()


def _enqueue(dispatcher, *, ram_gb: float, cpus: int = 1) -> int:
    return dispatcher.store.enqueue(
        ["python", "-c", "pass"],
        label=None,
        gpu_count=0,
        slots=1,
        priority_rank=100,
        log_path=None,
        cwd=None,
        env=None,
        ram_mib=ram_gb * GIB,
        cpus=cpus,
    )


def _memory(free_gb: float, total_gb: float = 61.6) -> HostMemory:
    """A host with plenty of commit headroom but limited free RAM."""
    return HostMemory(
        total_mib=total_gb * GIB,
        available_mib=free_gb * GIB,
        commit_used_mib=40.0 * GIB,
        commit_limit_mib=157.0 * GIB,
    )


def _run_dispatch(dispatcher, *, free_gb: float) -> list[int]:
    """One dispatch pass. Returns the ids it tried to start."""
    started: list[int] = []

    def _fake_start(self, row, devices):  # patched onto the class, so takes self
        started.append(int(row["id"]))
        return True

    with (
        patch("workerq.backends.dispatcher.host.memory", return_value=_memory(free_gb)),
        patch.object(Dispatcher, "_start_job", _fake_start),
    ):
        dispatcher._start_ready_jobs()
    return started


def _age_the_head(dispatcher, head_id: int, seconds: float) -> None:
    """Backdate the head's blocked-since stamp past the head-wait threshold."""
    dispatcher._blocked[head_id] = (time.monotonic() - seconds, "ram", 0.0)


def test_an_idle_machine_never_holds_the_queue(dispatcher):
    """The bug: a 1 GiB job stalled behind a 32 GiB one on an empty machine.

    Nothing was running, so no amount of draining would ever free the RAM the
    head needed - it was held out by the desktop. The hold still engaged and
    stopped every job behind it for the full 30 minute bound.
    """
    head = _enqueue(dispatcher, ram_gb=32.0)
    small = _enqueue(dispatcher, ram_gb=1.0)
    _age_the_head(dispatcher, head, 2000)

    started = _run_dispatch(dispatcher, free_gb=31.0)

    assert head not in started, "the head genuinely does not fit"
    assert started == [small], "the small job must be let through"


def test_the_hold_still_applies_while_jobs_are_running(dispatcher):
    """The guard itself must survive: draining works when there is work to drain."""
    head = _enqueue(dispatcher, ram_gb=32.0)
    small = _enqueue(dispatcher, ram_gb=1.0)
    _age_the_head(dispatcher, head, 2000)

    # One job in flight, so stopping now genuinely frees something later.
    dispatcher.adopted[999] = object()

    started = _run_dispatch(dispatcher, free_gb=31.0)

    assert started == [], "the queue is held so the head gets a clear run"
    assert head in dispatcher._hold_since


def test_an_idle_spell_does_not_spend_the_heads_hold(dispatcher):
    """The hold clock measures holding, not waiting.

    If it kept running while the machine sat idle, the head's one bounded hold
    would expire without it ever getting the drained machine it was promised.
    """
    head = _enqueue(dispatcher, ram_gb=32.0)
    _enqueue(dispatcher, ram_gb=1.0)
    _age_the_head(dispatcher, head, 2000)

    dispatcher.adopted[999] = object()
    _run_dispatch(dispatcher, free_gb=31.0)
    assert head in dispatcher._hold_since, "holding while a job runs"

    # Everything finishes; the machine goes idle.
    dispatcher.adopted.clear()
    _run_dispatch(dispatcher, free_gb=31.0)
    assert head not in dispatcher._hold_since, "the clock stops when not holding"
