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
    Reserve,
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


# --------------------------------------------------------------------------
# The live reserve: reclaiming the machine without stopping the queue
# --------------------------------------------------------------------------


def test_reserve_defaults_to_the_configured_value(tmp_path):
    config = make_config(tmp_path, reserve_ram_gb=8.0, reserve_vram_gb=1.0, reserve_cpus=2)
    reserve = Reserve.from_config(config)
    assert reserve.ram_mib == 8.0 * GIB
    assert reserve.vram_mib == 1.0 * GIB
    assert reserve.cpus == 2
    assert reserve.label is None


def test_a_bigger_reserve_shrinks_what_jobs_may_use(tmp_path):
    config = make_config(tmp_path)
    gaming = Reserve(ram_mib=24 * GIB, vram_mib=22 * GIB, cpus=8, label="gaming")
    default_cap = capacity(config, gpu=gpu(), mem=mem())
    gaming_cap = capacity(config, gpu=gpu(), mem=mem(), reserve=gaming)
    assert gaming_cap.usable_ram_mib < default_cap.usable_ram_mib
    assert gaming_cap.usable_vram_mib < default_cap.usable_vram_mib
    assert gaming_cap.usable_cpus < default_cap.usable_cpus
    # Totals are a property of the machine and must not move.
    assert gaming_cap.total_ram_mib == default_cap.total_ram_mib


def test_reserving_vram_blocks_a_gpu_job_that_previously_fitted(tmp_path):
    """The point of the feature: claim the GPU back and heavy work waits."""
    config = make_config(tmp_path)
    request = ResourceRequest(ram_mib=8 * GIB, vram_mib=20 * GIB, cpus=2)
    assert admit(config, request, [], gpu=gpu(), mem=mem()).admit
    gaming = Reserve(ram_mib=24 * GIB, vram_mib=22 * GIB, cpus=8, label="gaming")
    decision = admit(config, request, [], gpu=gpu(), mem=mem(), reserve=gaming)
    assert not decision.admit


def test_a_blocked_job_says_the_reserve_is_why(tmp_path):
    """Otherwise the only symptom is "nothing starts" with no way to find out."""
    config = make_config(tmp_path)
    gaming = Reserve(ram_mib=24 * GIB, vram_mib=22 * GIB, cpus=8, label="gaming")
    request = ResourceRequest(ram_mib=8 * GIB, vram_mib=20 * GIB, cpus=2)
    decision = admit(config, request, [], gpu=gpu(), mem=mem(), reserve=gaming)
    assert decision.reason is not None
    assert "gaming" in decision.reason
    assert decision.detail["reserve"]["label"] == "gaming"


def test_small_work_still_runs_while_the_reserve_is_held(tmp_path):
    """Reclaiming the machine throttles the queue; it must not stop it."""
    config = make_config(tmp_path)
    gaming = Reserve(ram_mib=24 * GIB, vram_mib=22 * GIB, cpus=8, label="gaming")
    small = ResourceRequest(ram_mib=4 * GIB, vram_mib=0.0, cpus=2)
    assert admit(config, small, [], gpu=gpu(), mem=mem(), reserve=gaming).admit


def test_an_unlabelled_reserve_adds_no_note(tmp_path):
    config = make_config(tmp_path)
    quiet = Reserve(ram_mib=60 * GIB, vram_mib=1 * GIB, cpus=2)
    decision = admit(
        config, ResourceRequest(ram_mib=8 * GIB), [], gpu=gpu(), mem=mem(), reserve=quiet
    )
    assert not decision.admit
    assert "held back by reserve" not in (decision.reason or "")


def test_expiry_is_only_reached_once_the_deadline_passes():
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert not Reserve(ram_mib=0, vram_mib=0, cpus=0).is_expired
    assert not Reserve(ram_mib=0, vram_mib=0, cpus=0, expires_at=future).is_expired
    assert Reserve(ram_mib=0, vram_mib=0, cpus=0, expires_at=past).is_expired


# --------------------------------------------------------------------------
# Commit charge, recalibrated
# --------------------------------------------------------------------------


def _default_commit_config(tmp_path) -> Config:
    """A config using the shipped commit thresholds, not the pinned 88."""
    return Config(
        core=CoreConfig(state_dir=str(tmp_path)),
        resources=ResourcesConfig(),
        source_path=tmp_path / "config.toml",
    )


def test_high_commit_is_fine_while_there_is_room_left_to_commit(tmp_path):
    """A high percentage on its own says little; the absolute room does.

    Half the samples taken on a live workstation while a job ran sat above 88%
    commit with ~42% of RAM free, and nothing was wrong.
    """
    config = _default_commit_config(tmp_path)
    decision = admit(
        config,
        request(4),
        [],
        gpu=gpu(),
        # 64 GiB machine, limit 89.6, 70% used -> 26.9 GiB of commit left.
        mem=mem(total_gb=64, free_gb=27, commit_percent=70),
    )
    assert decision.admit, decision.reason


def test_a_job_may_not_exceed_the_remaining_commit(tmp_path):
    """The failure this actually catches, and percentages could not express.

    A job died on the live machine at 100% commit with 40% of RAM free. Under
    WDDM the GPU driver backs video memory with system commit, so VRAM counts
    against the commit limit even though physical RAM looks untouched.
    """
    config = _default_commit_config(tmp_path)
    decision = admit(
        config,
        ResourceRequest(ram_mib=8 * GIB, vram_mib=20 * GIB, cpus=1),
        [],
        gpu=gpu(),
        # limit 89.6, 90% used -> 9.0 GiB left; the job wants 28 GiB of commit.
        mem=mem(total_gb=64, free_gb=40, commit_percent=90),
    )
    assert not decision.admit
    assert "commit" in (decision.reason or "")
    assert "VRAM counts here" in (decision.reason or "")


def test_vram_counts_toward_commit_even_with_no_ram_declared(tmp_path):
    """The GPU-only job is exactly the one the old percentage rule missed."""
    config = _default_commit_config(tmp_path)
    decision = admit(
        config,
        ResourceRequest(ram_mib=0.0, vram_mib=24 * GIB, cpus=1),
        [],
        gpu=gpu(),
        mem=mem(total_gb=64, free_gb=40, commit_percent=90),
    )
    assert not decision.admit
    assert "commit" in (decision.reason or "")


def test_the_ram_ledger_still_explains_itself_first(tmp_path):
    """Commit headroom is the specialised guard; it must not mask the common
    case, where the useful thing to say is which job is holding the RAM."""
    config = _default_commit_config(tmp_path)
    decision = admit(
        config, request(40), [request(40)], gpu=gpu(), mem=mem(total_gb=64, free_gb=60)
    )
    assert not decision.admit
    assert "already reserve" in (decision.reason or "")


def test_commit_at_the_limit_is_still_a_hard_stop(tmp_path):
    """Close to the limit, pagefile growth may not keep up. Refuse regardless."""
    config = _default_commit_config(tmp_path)
    decision = admit(
        config,
        ResourceRequest(),
        [],
        gpu=gpu(),
        mem=mem(total_gb=64, free_gb=40, commit_percent=100),
    )
    assert not decision.admit
    assert "commit charge" in (decision.reason or "")


def test_the_physical_floor_still_governs_regardless_of_commit(tmp_path):
    """Physical exhaustion is the thing that freezes a desktop."""
    config = _default_commit_config(tmp_path)
    decision = admit(
        config,
        request(1),
        [],
        gpu=gpu(),
        mem=mem(total_gb=64, free_gb=3, commit_percent=50),
    )
    assert not decision.admit
    assert "floor" in (decision.reason or "")
