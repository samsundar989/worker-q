"""The declaration-accuracy classifier (workerq.usage).

Separate from test_usage.py, which covers how usage is *measured*. This covers
what the measurement is then compared against.

The test that matters most here is `test_a_gpu_job_inside_its_commit_budget_is_not_under_declared`.
Getting it wrong does not produce a wrong pixel, it produces confident advice
to raise a declaration that is already correct - which parks tens of GiB that
nothing needs - or to lower one that is not, which takes the machine down.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from workerq import usage


def make_job(**kwargs):
    """A Job-shaped object with only the fields `usage` reads."""
    base = dict(
        id=1,
        project="proj",
        state="SUCCEEDED",
        node=None,
        requested_ram_mib=8192.0,
        requested_vram_mib=0.0,
        peak_ram_mib=4096.0,
        peak_vram_mib=None,
        usage_samples=50,
        peak_source="measured",
        vram_source=None,
        runtime_seconds=3600.0,
        wait_seconds=10.0,
        eta_seconds=None,
        description=None,
        command_signature="sig",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestCommitBudget:
    def test_a_cpu_job_is_judged_on_its_ram_alone(self):
        assert usage.commit_budget_mib(8192.0, 0.0) == 8192.0
        assert usage.commit_budget_mib(8192.0, None) == 8192.0

    def test_a_gpu_job_is_judged_on_ram_plus_vram(self):
        # Under WDDM the driver backs video allocations with system commit, so
        # the peak the runner records includes the VRAM the job declared.
        assert usage.commit_budget_mib(8192.0, 24576.0) == 32768.0

    def test_no_declaration_means_no_budget(self):
        assert usage.commit_budget_mib(None, 4096.0) is None


class TestClassify:
    def test_a_gpu_job_inside_its_commit_budget_is_not_under_declared(self):
        """The regression this whole module exists to prevent.

        Real job #474: declared 8 GiB RAM and 10 GiB VRAM, peak commit 12116
        MiB. Against declared RAM alone that reads as 148% - dangerously over.
        Against the commit budget it is 66%, which is a well-sized job.
        """
        job = make_job(
            requested_ram_mib=8192.0,
            requested_vram_mib=10240.0,
            peak_ram_mib=12116.0,
            peak_vram_mib=9808.0,
            vram_source="device_delta",
            usage_samples=9,
        )
        result = usage.for_job(job)
        assert result.verdict == usage.VERDICT_OK
        assert result.commit_ratio == pytest.approx(12116.0 / 18432.0, rel=1e-6)
        # And the naive comparison really would have flagged it.
        assert job.peak_ram_mib / job.requested_ram_mib > 1.0

    def test_a_job_over_its_whole_budget_is_still_caught(self):
        """Correcting the units must not silence the real cases.

        Real job #453: 16 GiB RAM + 26 GiB VRAM declared, 75 GiB peak commit.
        That is over even the generous reading, and has to stay loud.
        """
        job = make_job(
            requested_ram_mib=16384.0,
            requested_vram_mib=26624.0,
            peak_ram_mib=76839.0,
            usage_samples=608,
        )
        assert usage.for_job(job).verdict == usage.VERDICT_UNDER

    def test_a_short_job_with_two_samples_draws_no_conclusion(self):
        job = make_job(peak_ram_mib=40.0, usage_samples=2)
        result = usage.for_job(job)
        assert result.verdict == usage.VERDICT_LOW_CONFIDENCE
        assert not result.conclusive
        # The ratio is still reported - it is shown, greyed - but the verdict
        # is what any advice keys off.
        assert result.commit_ratio is not None

    def test_an_unmeasured_job_is_never_read_as_using_nothing(self):
        result = usage.for_job(make_job(peak_ram_mib=None))
        assert result.verdict == usage.VERDICT_UNMEASURED
        assert result.commit_ratio is None
        assert result.unused_gib_hours is None

    def test_the_two_over_declared_tiers(self):
        assert usage.for_job(make_job(peak_ram_mib=8192.0 * 0.10)).verdict == (
            usage.VERDICT_SEVERELY_OVER
        )
        assert usage.for_job(make_job(peak_ram_mib=8192.0 * 0.40)).verdict == (
            usage.VERDICT_OVER
        )
        assert usage.for_job(make_job(peak_ram_mib=8192.0 * 0.80)).verdict == (
            usage.VERDICT_OK
        )


class TestCost:
    def test_waste_is_weighted_by_how_long_it_was_held(self):
        """A huge over-declaration for ten seconds costs the queue nothing.

        Ranking by ratio alone would put it above a modest over-declaration
        held for eleven hours, which is the one actually making people wait.
        """
        brief = usage.for_job(make_job(peak_ram_mib=256.0, runtime_seconds=10.0))
        long = usage.for_job(make_job(peak_ram_mib=6144.0, runtime_seconds=40000.0))
        assert brief.commit_ratio < long.commit_ratio
        assert brief.unused_gib_hours < long.unused_gib_hours

    def test_summary_ranks_over_and_under_separately(self):
        rows = [
            usage.for_job(make_job(id=1, peak_ram_mib=512.0)),
            usage.for_job(make_job(id=2, peak_ram_mib=9999.0)),
        ]
        summary = usage.summarise(rows)
        assert [u.job_id for u in summary.worst_under] == [2]
        assert [u.job_id for u in summary.worst_over] == [1]

    def test_going_over_never_counts_as_reclaimable_headroom(self):
        rows = [usage.for_job(make_job(peak_ram_mib=99999.0))]
        assert usage.summarise(rows).unused_gib_hours == 0.0


class TestNodeLabel:
    def test_null_means_local(self):
        # The column itself must keep meaning NULL-is-local: eta._durations_for
        # matches `node IS NULL` to find same-machine history.
        assert usage.node_label(None) == "local"
        assert usage.node_label("3080ti") == "3080ti"


class TestGrouping:
    def test_a_signature_group_carries_a_readable_label(self):
        rows = [
            usage.for_job(make_job(id=1, description="nightly harvest")),
            usage.for_job(make_job(id=2, description=None)),
        ]
        groups = usage.group_by(rows, "command_signature")
        assert len(groups) == 1
        assert groups[0]["key"] == "sig"
        assert groups[0]["label"] == "nightly harvest"
        assert groups[0]["runs"] == 2
