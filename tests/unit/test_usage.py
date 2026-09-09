"""Per-job usage measurement.

Admission control is only as good as the numbers jobs declare, and nothing
measured whether those were right. These cover the attribution helpers and,
above all, the difference between "measured zero" and "could not measure" -
recording an unknown as zero would make an over-committed machine look free.
"""

from __future__ import annotations

import os

import pytest

from workerq import host
from workerq.backends.dispatcher import Dispatcher
from workerq.gpu import (
    GpuDevice,
    GpuInfo,
    GpuProcess,
    device_delta_mib,
    tree_vram_mib,
)


def _device(*procs: GpuProcess) -> GpuInfo:
    device = GpuDevice(
        index=0,
        uuid="GPU-test",
        name="Test",
        memory_total_mib=32000.0,
        memory_used_mib=1000.0,
        memory_free_mib=31000.0,
        utilization_percent=0.0,
    )
    device.processes = list(procs)
    return GpuInfo(available=True, devices=[device])


# -- VRAM attribution -------------------------------------------------------


def test_vram_sums_only_the_jobs_own_processes():
    info = _device(
        GpuProcess(pid=1, process_name="mine", used_memory_mib=1000.0, gpu_uuid="GPU-test"),
        GpuProcess(pid=2, process_name="mine", used_memory_mib=500.0, gpu_uuid="GPU-test"),
        GpuProcess(pid=99, process_name="theirs", used_memory_mib=8000.0, gpu_uuid="GPU-test"),
    )
    assert tree_vram_mib(info, {1, 2}) == 1500.0


def test_vram_is_unknown_when_the_driver_reports_na():
    """Consumer cards in WDDM mode report `[N/A]` for every process.

    Returning 0.0 here would record a 20 GiB training job as using no VRAM.
    """
    info = _device(
        GpuProcess(pid=1, process_name="mine", used_memory_mib=None, gpu_uuid="GPU-test"),
        GpuProcess(pid=2, process_name="mine", used_memory_mib=None, gpu_uuid="GPU-test"),
    )
    assert tree_vram_mib(info, {1, 2}) is None


def test_vram_is_unknown_with_no_gpu_or_no_pids():
    info = _device(GpuProcess(pid=1, process_name="m", used_memory_mib=10.0, gpu_uuid="u"))
    assert tree_vram_mib(GpuInfo(available=False), {1}) is None
    assert tree_vram_mib(info, set()) is None


def test_vram_partial_measurement_counts_what_it_can():
    info = _device(
        GpuProcess(pid=1, process_name="mine", used_memory_mib=1000.0, gpu_uuid="u"),
        GpuProcess(pid=2, process_name="mine", used_memory_mib=None, gpu_uuid="u"),
    )
    assert tree_vram_mib(info, {1, 2}) == 1000.0


# -- host RAM attribution ---------------------------------------------------


def test_tree_memory_includes_this_process():
    """The runner's own tree must be measurable, or nothing else is."""
    measured = host.tree_memory_mib({os.getpid()})
    assert measured is not None and measured > 0.0


def test_tree_memory_is_unknown_for_no_roots():
    assert host.tree_memory_mib(set()) is None


def test_tree_memory_ignores_unrelated_processes():
    """A pid that does not exist owns nothing, so the total is not the machine."""
    everything = sum(p.memory_mib for p in host.all_processes())
    mine = host.tree_memory_mib({os.getpid()})
    assert mine is not None and mine < everything


def test_top_processes_is_a_slice_of_all_processes():
    everything = host.all_processes()
    if not everything:  # pragma: no cover - process table unreadable
        pytest.skip("no process table available")
    assert host.top_processes(3) == everything[:3]
    assert len(host.all_processes()) >= len(host.top_processes(8))


# -- VRAM by whole-card delta -----------------------------------------------
#
# The fallback for consumer cards in WDDM mode, where per-process VRAM reads as
# [N/A] for every process and the only attributable signal is the rise in total
# card usage while exactly one job is responsible for it.


def _card(used_mib: float | None, index: int = 0) -> GpuInfo:
    return GpuInfo(
        available=True,
        devices=[
            GpuDevice(
                index=index,
                uuid="GPU-test",
                name="Test",
                memory_total_mib=32000.0,
                memory_used_mib=used_mib,
                memory_free_mib=None,
                utilization_percent=0.0,
            )
        ],
    )


def test_device_delta_is_the_rise_above_the_baseline():
    assert device_delta_mib(_card(19000.0), 0, 2200.0) == pytest.approx(16800.0)


def test_device_delta_clamps_at_zero_when_the_desktop_releases_memory():
    """A negative delta means the baseline moved, not that the job freed VRAM."""
    assert device_delta_mib(_card(1800.0), 0, 2200.0) == 0.0


def test_device_delta_is_unknown_rather_than_zero_when_the_card_cannot_be_read():
    assert device_delta_mib(_card(None), 0, 2200.0) is None
    assert device_delta_mib(GpuInfo(available=False, devices=[]), 0, 2200.0) is None


def test_device_delta_ignores_a_device_that_is_not_the_one_measured():
    """Attributing another card's usage to this job would be worse than silence."""
    assert device_delta_mib(_card(19000.0, index=1), 0, 2200.0) is None
    assert device_delta_mib(_card(19000.0), None, 2200.0) is None


# -- when a whole-card delta may be trusted ---------------------------------


class _FakeDispatcher:
    """Just enough Dispatcher to exercise the baseline rule in isolation."""

    def __init__(self, occupancy, used_mib=2200.0, available=True):
        self._occupancy = occupancy
        self._info = GpuInfo(
            available=available,
            devices=[
                GpuDevice(
                    index=0,
                    uuid="GPU-test",
                    name="Test",
                    memory_total_mib=32000.0,
                    memory_used_mib=used_mib,
                    memory_free_mib=None,
                    utilization_percent=0.0,
                )
            ],
        )

    _device_occupancy = lambda self: self._occupancy  # noqa: E731
    _gpu_info = lambda self: self._info  # noqa: E731

    _vram_baseline = Dispatcher._vram_baseline


def test_baseline_is_taken_when_the_job_owns_the_card_alone():
    d = _FakeDispatcher({})
    assert d._vram_baseline({"gpu_mode": "exclusive"}, [0]) == 2200.0


def test_no_baseline_when_another_job_is_already_on_the_card():
    """Its allocation would be attributed to this job."""
    d = _FakeDispatcher({0: {"jobs": [7], "vram_mib": 4000.0, "exclusive": False}})
    assert d._vram_baseline({"gpu_mode": "exclusive"}, [0]) is None


def test_no_baseline_for_a_shared_job():
    """Another job may join the card later, which would spoil the delta."""
    d = _FakeDispatcher({})
    assert d._vram_baseline({"gpu_mode": "shared"}, [0]) is None


def test_no_baseline_across_several_devices():
    d = _FakeDispatcher({})
    assert d._vram_baseline({"gpu_mode": "exclusive"}, [0, 1]) is None


def test_no_baseline_when_the_gpu_cannot_be_read():
    d = _FakeDispatcher({}, available=False)
    assert d._vram_baseline({"gpu_mode": "exclusive"}, [0]) is None
