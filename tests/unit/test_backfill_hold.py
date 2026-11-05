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
    # 16 CPUs less the default 2-CPU reserve: 14 usable, like the owner's box.
    with patch("workerq.resources.cpu_count", return_value=16):
        try:
            yield d
        finally:
            d.store.close()


def _enqueue(dispatcher, *, ram_gb: float, cpus: int = 1, gpus: int = 0) -> int:
    return dispatcher.store.enqueue(
        ["python", "-c", "pass"],
        label=None,
        gpu_count=gpus,
        slots=1,
        priority_rank=100,
        log_path=None,
        cwd=None,
        env=None,
        ram_mib=ram_gb * GIB,
        cpus=cpus,
    )


def _running(dispatcher, *, ram_gb: float, cpus: int = 1) -> int:
    """A job already executing here, with the reservation it declared."""
    backend_id = _enqueue(dispatcher, ram_gb=ram_gb, cpus=cpus)
    assert dispatcher.store.claim_for_start(backend_id)
    dispatcher.adopted[backend_id] = (0, 0)
    return backend_id


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
    """The guard itself must survive: draining works when there is work to drain.

    The running job holds 16 GiB, so once it ends the 32 GiB head fits. Any
    RAM a newcomer takes now is RAM the head is still short of.
    """
    _running(dispatcher, ram_gb=16.0)
    head = _enqueue(dispatcher, ram_gb=32.0)
    small = _enqueue(dispatcher, ram_gb=1.0)
    _age_the_head(dispatcher, head, 2000)

    started = _run_dispatch(dispatcher, free_gb=31.0)

    assert started == [], "the queue is held so the head gets a clear run"
    assert head in dispatcher._hold_since
    assert f"queue held for job {head}" in str(dispatcher.store.get(small)["wait_reason"])


def test_an_idle_spell_does_not_spend_the_heads_hold(dispatcher):
    """The hold clock measures holding, not waiting.

    If it kept running while the machine sat idle, the head's one bounded hold
    would expire without it ever getting the drained machine it was promised.
    """
    running = _running(dispatcher, ram_gb=16.0)
    head = _enqueue(dispatcher, ram_gb=32.0)
    _enqueue(dispatcher, ram_gb=1.0)
    _age_the_head(dispatcher, head, 2000)

    _run_dispatch(dispatcher, free_gb=31.0)
    assert head in dispatcher._hold_since, "holding while a job runs"

    # Everything finishes; the machine goes idle.
    dispatcher.adopted.clear()
    dispatcher.store.finish(running, 0)
    _run_dispatch(dispatcher, free_gb=31.0)
    assert head not in dispatcher._hold_since, "the clock stops when not holding"


# --------------------------------------------------------------------------
# Head-of-line blocking seen on the owner's box, Nov 3-4 2026
# --------------------------------------------------------------------------


def test_identical_blocked_jobs_do_not_hide_a_small_one(dispatcher):
    """The 24 GiB head-of-line case, as the log recorded it.

    61.6 GiB box, 14 usable CPUs, 6 slots. A 24 GiB job (3938) sat at the head
    while a batch of 12 GiB / 4 CPU jobs queued behind it, all blocked on
    CPUs. Each blocked job used one unit of `backfill_max_skip`, so after nine
    of them the scan stopped looking here and a 1 GiB / 1 CPU job behind them
    was logged "waiting for this machine" for hours - while a CPU and 18 GiB
    of RAM sat free. No hold was even in force: the look-ahead had run out.
    """
    for _ in range(4):
        _running(dispatcher, ram_gb=8.0, cpus=3)  # 12 of 14 CPUs reserved
    head = _enqueue(dispatcher, ram_gb=24.0, cpus=4)
    batch = [_enqueue(dispatcher, ram_gb=12.0, cpus=4) for _ in range(9)]
    small = _enqueue(dispatcher, ram_gb=1.0, cpus=1)

    started = _run_dispatch(dispatcher, free_gb=25.0)

    assert started == [small]
    assert head not in started and not set(batch) & set(started)
    assert "this machine" not in str(dispatcher.store.get(small)["wait_reason"] or "")


def test_the_look_ahead_still_bounds_distinct_refusals(dispatcher):
    """The cap survives for what it was for: jobs that each say something new."""
    dispatcher.config.scheduling.backfill_max_skip = 1
    for _ in range(4):
        _running(dispatcher, ram_gb=8.0, cpus=3)
    _enqueue(dispatcher, ram_gb=24.0, cpus=4)
    _enqueue(dispatcher, ram_gb=12.0, cpus=4)  # less RAM: a distinct refusal
    small = _enqueue(dispatcher, ram_gb=1.0, cpus=1)

    assert _run_dispatch(dispatcher, free_gb=25.0) == []
    assert "this machine" in str(dispatcher.store.get(small)["wait_reason"])


def test_a_hold_lets_through_work_that_takes_nothing_the_head_needs(dispatcher):
    """Holding reserves the head's request; it does not stop the machine.

    The head waits for the only GPU. A CPU job behind it takes no GPU and
    leaves the head all the RAM and CPUs it needs, so it starts. A second GPU
    job would take what the head is waiting for, so it does not.
    """
    from workerq.gpu import GpuDevice, GpuInfo

    gpu = GpuInfo(
        available=True,
        devices=[GpuDevice(0, "u0", "card", 12288.0, 11000.0, 1288.0, 90.0)],
    )
    _running(dispatcher, ram_gb=8.0, cpus=4)
    head = _enqueue(dispatcher, ram_gb=8.0, cpus=4, gpus=1)
    cpu_job = _enqueue(dispatcher, ram_gb=2.0, cpus=2)
    gpu_job = _enqueue(dispatcher, ram_gb=1.0, cpus=1, gpus=1)
    _age_the_head(dispatcher, head, 2000)

    def busy_gpu(gpu_count, **_):
        return ([], None) if gpu_count == 0 else (None, f"waiting for {gpu_count} free GPU(s)")

    with (
        patch.object(dispatcher, "_gpu_info", return_value=gpu),
        patch.object(dispatcher, "_devices_in_use", return_value={0}),
        patch.object(dispatcher, "_allocate_devices", side_effect=busy_gpu),
    ):
        started = _run_dispatch(dispatcher, free_gb=40.0)

    assert head in dispatcher._hold_since, "the hold is in force"
    assert started == [cpu_job]
    assert gpu_job not in started


def test_a_hold_keeps_a_slot_and_the_heads_resources(dispatcher):
    """Work beside a hold must leave the head a slot and its full request."""
    from workerq.gpu import GpuDevice, GpuInfo

    gpu = GpuInfo(available=True, devices=[GpuDevice(0, "u0", "card", 12288.0, 0, 12288.0, 0)])
    for _ in range(4):
        _running(dispatcher, ram_gb=2.0, cpus=1)
    head = _enqueue(dispatcher, ram_gb=8.0, cpus=4, gpus=1)
    too_hungry = _enqueue(dispatcher, ram_gb=2.0, cpus=8)  # would leave 2 CPUs < 4
    fits = _enqueue(dispatcher, ram_gb=2.0, cpus=2)  # leaves 8 CPUs, 1 slot
    last = _enqueue(dispatcher, ram_gb=1.0, cpus=1)  # would take the head's slot
    _age_the_head(dispatcher, head, 2000)

    def busy_gpu(gpu_count, **_):
        return ([], None) if gpu_count == 0 else (None, "waiting for 1 free GPU(s)")

    with (
        patch.object(dispatcher, "_gpu_info", return_value=gpu),
        patch.object(dispatcher, "_devices_in_use", return_value={0}),
        patch.object(dispatcher, "_allocate_devices", side_effect=busy_gpu),
    ):
        started = _run_dispatch(dispatcher, free_gb=40.0)

    assert started == [fits]
    assert "CPU" in str(dispatcher.store.get(too_hungry)["wait_reason"])
    assert "slot" in str(dispatcher.store.get(last)["wait_reason"])


def test_no_hold_for_a_head_that_draining_cannot_admit(dispatcher):
    """Its RAM is held by work outside the queue, so draining buys it nothing.

    Two small jobs run, declaring 2 GiB between them. Even if both ended, only
    about 16 GiB would be free - the rest belongs to the desktop and an editor.
    Holding for a 24 GiB job would park the queue for its whole budget.
    """
    _running(dispatcher, ram_gb=1.0)
    _running(dispatcher, ram_gb=1.0)
    head = _enqueue(dispatcher, ram_gb=24.0)
    small = _enqueue(dispatcher, ram_gb=1.0)
    _age_the_head(dispatcher, head, 2000)

    started = _run_dispatch(dispatcher, free_gb=20.0)

    assert started == [small]
    assert head not in dispatcher._hold_since, "not holding, so the clock does not run"


def test_the_hold_budget_is_not_renewed_by_an_idle_spell(dispatcher):
    """The bound is for the job's whole stay, not for each stretch of holding.

    It used to restart whenever the machine went idle: 3938 was given four
    separate 30 minute holds on Nov 4.
    """
    running = _running(dispatcher, ram_gb=16.0)
    head = _enqueue(dispatcher, ram_gb=32.0)
    small = _enqueue(dispatcher, ram_gb=1.0)
    _age_the_head(dispatcher, head, 5000)

    _run_dispatch(dispatcher, free_gb=31.0)
    dispatcher._hold_since[head] -= 1000  # held for 1000s ...
    dispatcher.adopted.pop(running)  # ... then the machine went idle
    dispatcher.store.finish(running, 0)
    _run_dispatch(dispatcher, free_gb=31.0)
    assert dispatcher._hold_spent[head] >= 1000

    # Busy again: the hold resumes with what is left of the budget, not anew.
    _running(dispatcher, ram_gb=16.0)
    assert _run_dispatch(dispatcher, free_gb=31.0) == []
    dispatcher._hold_since[head] -= 900  # 1900s in all > 1800s budget
    assert _run_dispatch(dispatcher, free_gb=31.0) == [small]
