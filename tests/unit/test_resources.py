"""Admission control: the rules that decide when a job may start.

These encode the behaviour that stops the dev box falling over: small jobs may
run together, large ones serialise, and nothing starts when the machine is
already out of headroom - including headroom consumed by work gpuq did not
launch.
"""

from __future__ import annotations

import pytest

from workerq import host
from workerq.config import Config, CoreConfig, ResourcesConfig
from workerq.gpu import GpuDevice, GpuInfo
from workerq.resources import (
    Decision,
    ResourceRequest,
    admit,
    capacity,
    describe_capacity,
    sum_reservations,
)

GIB = 1024.0


def make_config(tmp_path, **resource_kwargs) -> Config:
    defaults = dict(
        enforce=True,
        default_ram_gb=4.0,
        default_vram_gb=0.0,
        default_cpus=1,
        reserve_ram_gb=8.0,
        reserve_vram_gb=1.0,
        reserve_cpus=2,
        max_commit_percent=88,
        min_host_free_percent=10,
    )
    defaults.update(resource_kwargs)
    return Config(
        core=CoreConfig(state_dir=str(tmp_path)),
        resources=ResourcesConfig(**defaults),
        source_path=tmp_path / "config.toml",
    )


def mem(
    *, total_gb: float = 64.0, free_gb: float = 40.0, commit_percent: float = 50.0
) -> host.HostMemory:
    limit = total_gb * 1.4 * GIB
    return host.HostMemory(
        total_mib=total_gb * GIB,
        available_mib=free_gb * GIB,
        commit_limit_mib=limit,
        commit_used_mib=limit * commit_percent / 100.0,
    )


def gpu(*, total_gb: float = 32.0, free_gb: float = 30.0) -> GpuInfo:
    return GpuInfo(
        available=True,
        devices=[
            GpuDevice(
                index=0,
                uuid="gpu-0",
                name="Test GPU",
                memory_total_mib=total_gb * GIB,
                memory_used_mib=(total_gb - free_gb) * GIB,
                memory_free_mib=free_gb * GIB,
                utilization_percent=0.0,
            )
        ],
    )


def request(ram_gb: float = 0.0, cpus: int = 0, vram_gb: float = 0.0) -> ResourceRequest:
    return ResourceRequest(ram_mib=ram_gb * GIB, cpus=cpus, vram_mib=vram_gb * GIB)


# --------------------------------------------------------------------------
# capacity
# --------------------------------------------------------------------------


def test_capacity_subtracts_reserved_headroom(tmp_path):
    config = make_config(tmp_path, reserve_ram_gb=8.0, reserve_cpus=2)
    cap = capacity(config, gpu=gpu(), mem=mem(total_gb=64.0))
    assert cap.total_ram_mib == pytest.approx(64 * GIB)
    assert cap.usable_ram_mib == pytest.approx(56 * GIB)
    assert cap.usable_cpus == cap.total_cpus - 2
    assert cap.usable_vram_mib == pytest.approx(31 * GIB)


def test_sum_reservations():
    total = sum_reservations([request(4, 2), request(8, 4)])
    assert total.ram_mib == pytest.approx(12 * GIB)
    assert total.cpus == 6


# --------------------------------------------------------------------------
# the basic decision
# --------------------------------------------------------------------------


def test_small_job_is_admitted_on_an_idle_machine(tmp_path):
    decision = admit(
        make_config(tmp_path), request(4, 2), [], gpu=gpu(), mem=mem(free_gb=40)
    )
    assert decision.admit
    assert decision.reason is None
    assert bool(decision) is True


def test_several_small_jobs_run_in_parallel(tmp_path):
    """This is the point of resource-aware admission, not a slot count."""
    config = make_config(tmp_path)
    running = [request(4, 2), request(4, 2), request(4, 2)]
    assert admit(config, request(4, 2), running, gpu=gpu(), mem=mem(free_gb=40)).admit


def test_large_jobs_serialise(tmp_path):
    """Two 40 GiB jobs cannot both fit in 56 GiB of usable RAM.

    RAM is reported as almost entirely free, because the running job has not
    grown to its declared size yet. Measured free memory alone would happily
    admit the second job; the reservation check is what prevents it.
    """
    config = make_config(tmp_path)
    decision = admit(
        config, request(40), [request(40)], gpu=gpu(), mem=mem(total_gb=64, free_gb=60)
    )
    assert not decision.admit
    assert "already reserve" in (decision.reason or "")


def test_enforcement_can_be_disabled(tmp_path):
    config = make_config(tmp_path, enforce=False)
    decision = admit(
        config, request(999), [request(999)], gpu=gpu(), mem=mem(free_gb=0.1)
    )
    assert decision.admit
    assert decision.detail["enforced"] is False


# --------------------------------------------------------------------------
# foreign pressure - what a slot count cannot see
# --------------------------------------------------------------------------


def test_foreign_memory_use_blocks_a_job(tmp_path):
    """Nothing is running under gpuq, yet the RAM is gone: still refuse."""
    config = make_config(tmp_path)
    decision = admit(config, request(16), [], gpu=gpu(), mem=mem(total_gb=64, free_gb=8))
    assert not decision.admit
    assert "free" in (decision.reason or "")
    assert decision.detail["running_jobs"] == 0


def test_commit_exhaustion_is_a_hard_stop(tmp_path):
    """Windows fails allocations near the commit limit even with RAM free."""
    config = make_config(tmp_path, max_commit_percent=88)
    decision = admit(
        config,
        request(1),
        [],
        gpu=gpu(),
        mem=mem(total_gb=64, free_gb=40, commit_percent=93),
    )
    assert not decision.admit
    assert "commit charge" in (decision.reason or "")


def test_commit_stop_applies_even_to_a_zero_request(tmp_path):
    config = make_config(tmp_path)
    decision = admit(
        config, ResourceRequest(), [], gpu=gpu(), mem=mem(commit_percent=95)
    )
    assert not decision.admit


def test_host_free_floor_is_respected(tmp_path):
    config = make_config(tmp_path, min_host_free_percent=20)
    decision = admit(config, request(1), [], gpu=gpu(), mem=mem(total_gb=64, free_gb=6))
    assert not decision.admit
    assert "floor" in (decision.reason or "")


def test_a_job_may_not_eat_into_the_free_floor(tmp_path):
    """32 GiB free, 10% floor on 64 GiB: a 30 GiB job must wait."""
    config = make_config(tmp_path, min_host_free_percent=10)
    decision = admit(
        config, request(30), [], gpu=gpu(), mem=mem(total_gb=64, free_gb=32)
    )
    assert not decision.admit
    assert "free after the" in (decision.reason or "")


# --------------------------------------------------------------------------
# CPUs and VRAM
# --------------------------------------------------------------------------


def test_cpu_oversubscription_is_refused(tmp_path):
    config = make_config(tmp_path, reserve_cpus=2)
    cap = capacity(config, gpu=gpu(), mem=mem())
    decision = admit(
        config,
        ResourceRequest(cpus=cap.usable_cpus),
        [ResourceRequest(cpus=cap.usable_cpus)],
        gpu=gpu(),
        mem=mem(),
    )
    assert not decision.admit
    assert "CPU" in (decision.reason or "")


def test_vram_request_larger_than_free_is_refused(tmp_path):
    config = make_config(tmp_path)
    decision = admit(
        config, request(vram_gb=24), [], gpu=gpu(total_gb=32, free_gb=10), mem=mem()
    )
    assert not decision.admit
    assert "VRAM" in (decision.reason or "")


def test_vram_reservations_accumulate(tmp_path):
    config = make_config(tmp_path)
    decision = admit(
        config,
        request(vram_gb=20),
        [request(vram_gb=20)],
        gpu=gpu(total_gb=32, free_gb=30),
        mem=mem(),
    )
    assert not decision.admit


def test_vram_request_without_a_gpu_is_refused(tmp_path):
    config = make_config(tmp_path)
    decision = admit(
        config,
        request(vram_gb=4),
        [],
        gpu=GpuInfo(available=False, error="no driver"),
        mem=mem(),
    )
    assert not decision.admit
    assert "no NVIDIA GPU" in (decision.reason or "")


# --------------------------------------------------------------------------
# defaults and reporting
# --------------------------------------------------------------------------


def test_undeclared_jobs_are_charged_the_default(tmp_path):
    """An undeclared job must not be treated as free."""
    config = make_config(tmp_path, default_ram_gb=6.0, default_cpus=3)
    req = ResourceRequest.from_job(config, ram_mib=None, vram_mib=None, cpus=None, gpu_count=1)
    assert req.ram_mib == pytest.approx(6 * GIB)
    assert req.cpus == 3
    assert req.gpu_count == 1


def test_explicit_request_overrides_the_default(tmp_path):
    config = make_config(tmp_path, default_ram_gb=6.0)
    req = ResourceRequest.from_job(
        config, ram_mib=2048.0, vram_mib=1024.0, cpus=2, gpu_count=0
    )
    assert req.ram_mib == 2048.0
    assert req.vram_mib == 1024.0
    assert req.cpus == 2


def test_reason_explains_the_numbers(tmp_path):
    """A blocked job must say why, in units a human can act on."""
    config = make_config(tmp_path)
    decision = admit(config, request(40), [], gpu=gpu(), mem=mem(total_gb=64, free_gb=12))
    assert not decision.admit
    reason = decision.reason or ""
    assert "GiB" in reason and "40.0" in reason


def test_decision_detail_is_recorded_for_forensics(tmp_path):
    config = make_config(tmp_path)
    decision = admit(config, request(4), [request(8)], gpu=gpu(), mem=mem())
    assert decision.detail["request"]["ram_mib"] == pytest.approx(4 * GIB)
    assert decision.detail["reserved"]["ram_mib"] == pytest.approx(8 * GIB)
    assert "capacity" in decision.detail
    assert decision.detail["commit_percent"] is not None


def test_describe_capacity_shape(tmp_path):
    data = describe_capacity(make_config(tmp_path))
    assert set(data) >= {"enforced", "capacity", "host", "reserve", "limits", "defaults"}
    assert data["capacity"]["total_cpus"] >= 1


def test_decision_is_falsey_when_refused(tmp_path):
    decision = Decision(False, "nope")
    assert not decision
