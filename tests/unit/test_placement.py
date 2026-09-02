"""GPU placement, including sharing a device between jobs.

Whole-device exclusivity means two GPU jobs can never overlap on a
single-GPU machine however much VRAM is spare. Sharing lifts that, but only
between jobs that both opted in and both said how much VRAM they will use:
VRAM has no swap, and on consumer cards in WDDM mode there is no per-process
accounting to check a guess against.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from workerq.backends.dispatcher import Dispatcher
from workerq.gpu import GpuDevice, GpuInfo

GIB = 1024.0


def _gpu(count: int = 1, total_gb: float = 32.0, free_gb: float = 30.0) -> GpuInfo:
    devices = [
        GpuDevice(
            index=i,
            uuid=f"GPU-{i}",
            name="Test",
            memory_total_mib=total_gb * GIB,
            memory_used_mib=(total_gb - free_gb) * GIB,
            memory_free_mib=free_gb * GIB,
            utilization_percent=0.0,
        )
        for i in range(count)
    ]
    return GpuInfo(available=True, devices=devices)


@pytest.fixture
def dispatcher(isolated_config):
    isolated_config.gpu.free_memory_threshold_percent = 0
    d = Dispatcher(isolated_config)
    d.store.initialize()
    try:
        yield d
    finally:
        d.store.close()


def _occupy(dispatcher, backend_id: int, device: int, *, vram_gb: float, mode: str):
    """Pretend `backend_id` is running on `device` with a declared footprint."""

    class _Fake:
        devices = [device]

    dispatcher.running[backend_id] = _Fake()
    rows = [
        {
            "id": backend_id,
            "gpu_mode": mode,
            "vram_mib": vram_gb * GIB,
        }
    ]
    return patch.object(dispatcher.store, "running", lambda: rows)


def test_an_empty_device_is_allocated(dispatcher):
    with patch.object(dispatcher, "_gpu_info", return_value=_gpu()):
        devices, reason = dispatcher._allocate_devices(1)
    assert devices == [0] and reason is None


def test_a_job_needing_no_gpu_takes_none(dispatcher):
    with patch.object(dispatcher, "_gpu_info", return_value=_gpu()):
        devices, reason = dispatcher._allocate_devices(0)
    assert devices == [] and reason is None


def test_an_exclusive_occupant_blocks_the_device(dispatcher):
    """The current behaviour, and the reason two GPU jobs never overlap."""
    with _occupy(dispatcher, 1, 0, vram_gb=4, mode="exclusive"):
        with patch.object(dispatcher, "_gpu_info", return_value=_gpu()):
            devices, reason = dispatcher._allocate_devices(
                1, gpu_mode="shared", vram_mib=4 * GIB
            )
    assert devices is None
    assert reason is not None


def test_two_sharing_jobs_fit_on_one_device(dispatcher):
    with _occupy(dispatcher, 1, 0, vram_gb=18, mode="shared"):
        with patch.object(dispatcher, "_gpu_info", return_value=_gpu(total_gb=32)):
            devices, reason = dispatcher._allocate_devices(
                1, gpu_mode="shared", vram_mib=10 * GIB
            )
    assert devices == [0], reason


def test_sharing_is_refused_when_the_declarations_do_not_fit(dispatcher):
    """18 + 20 GiB does not fit a 32 GiB card, however free it looks now."""
    with _occupy(dispatcher, 1, 0, vram_gb=18, mode="shared"):
        with patch.object(dispatcher, "_gpu_info", return_value=_gpu(total_gb=32)):
            devices, reason = dispatcher._allocate_devices(
                1, gpu_mode="shared", vram_mib=20 * GIB
            )
    assert devices is None
    assert "shared" in (reason or "")


def test_an_exclusive_job_never_joins_a_shared_device(dispatcher):
    """Sharing has to be mutual, or the occupant did not consent to it."""
    with _occupy(dispatcher, 1, 0, vram_gb=4, mode="shared"):
        with patch.object(dispatcher, "_gpu_info", return_value=_gpu()):
            devices, reason = dispatcher._allocate_devices(
                1, gpu_mode="exclusive", vram_mib=4 * GIB
            )
    assert devices is None


def test_a_shared_job_declaring_no_vram_is_never_packed(dispatcher):
    """Nothing to pack against; treating it as zero would oversubscribe."""
    with _occupy(dispatcher, 1, 0, vram_gb=4, mode="shared"):
        with patch.object(dispatcher, "_gpu_info", return_value=_gpu()):
            devices, reason = dispatcher._allocate_devices(
                1, gpu_mode="shared", vram_mib=0.0
            )
    assert devices is None


def test_an_empty_device_is_preferred_over_sharing(dispatcher):
    with _occupy(dispatcher, 1, 0, vram_gb=4, mode="shared"):
        with patch.object(dispatcher, "_gpu_info", return_value=_gpu(count=2)):
            devices, reason = dispatcher._allocate_devices(
                1, gpu_mode="shared", vram_mib=4 * GIB
            )
    assert devices == [1], reason


def test_the_reserve_reduces_what_may_be_packed(dispatcher):
    """A VRAM reserve held for the owner is not available to share into."""
    from workerq.resources import Reserve

    big = Reserve(ram_mib=0, vram_mib=12 * GIB, cpus=0, label="gaming")
    with _occupy(dispatcher, 1, 0, vram_gb=18, mode="shared"):
        with patch.object(dispatcher, "_gpu_info", return_value=_gpu(total_gb=32)):
            with patch.object(dispatcher, "_reserve", return_value=big):
                devices, reason = dispatcher._allocate_devices(
                    1, gpu_mode="shared", vram_mib=10 * GIB
                )
    # 18 + 10 = 28 fits 32 GiB, but not 32 - 12 reserved.
    assert devices is None
