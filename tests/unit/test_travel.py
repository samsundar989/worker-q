"""Whether a job is safe to run on a different machine.

Phase 4b of docs/multi-node.md. Eligibility asks "is the data there"; this asks
the question eligibility cannot see, because its failure does not look like a
failure: a job that writes to an absolute path under the repository resolves
that path on either machine, so it succeeds and leaves its output on a machine
nobody is looking at.
"""

from __future__ import annotations

import pytest

from workerq import travel


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "biohub"
    (root / "outputs").mkdir(parents=True)
    return root


def test_the_biohub_pattern_is_refused(repo):
    """The real case, from a real `.gpuq.toml`.

    biohub instructs jobs to write submissions to an absolute path, and on one
    machine that is correct - a relative path would land inside the snapshot
    and be deleted when it expires. It is exactly that fix which makes the job
    unsafe to move.
    """
    argv = ["python", "predict.py", "--out", str(repo / "outputs" / "submission.csv")]
    verdict = travel.assess(argv, repo, snapshot_commit="abc123")
    assert not verdict.ok
    assert verdict.repo_paths
    assert "you are not looking at" in verdict.reasons[0]


def test_the_equals_form_is_caught_too(repo):
    argv = ["python", "run.py", f"--out={repo / 'outputs' / 'x.csv'}"]
    assert not travel.assess(argv, repo, snapshot_commit="abc").ok


def test_a_relative_output_path_travels_fine(repo):
    argv = ["python", "predict.py", "--out", "outputs/submission.csv"]
    assert travel.assess(argv, repo, snapshot_commit="abc").ok


def test_an_absolute_path_outside_the_repo_is_left_alone(repo, tmp_path):
    """A dataset elsewhere is a read. If it is missing on the node the job
    fails loudly, which is the safe direction - unlike a path that resolves."""
    argv = ["python", "train.py", "--data", str(tmp_path / "datasets" / "imagenet")]
    assert travel.assess(argv, repo, snapshot_commit="abc").ok


def test_a_job_with_no_snapshot_cannot_travel(repo):
    """There is no commit to ship, so there is nothing to reproduce there."""
    verdict = travel.assess(["python", "x.py"], repo, snapshot_commit=None)
    assert not verdict.ok
    assert "no git snapshot" in verdict.reasons[0]


def test_a_job_outside_any_repository_cannot_travel():
    assert not travel.assess(["python", "x.py"], None, snapshot_commit="abc").ok


def test_several_offending_paths_are_summarised_not_dumped(repo):
    argv = ["python", "x.py"] + [str(repo / "outputs" / f"f{i}.csv") for i in range(6)]
    verdict = travel.assess(argv, repo, snapshot_commit="abc")
    assert not verdict.ok
    assert len(verdict.repo_paths) == 6
    assert "and 3 more" in verdict.reasons[0]


def test_the_verdict_is_serialisable(repo):
    verdict = travel.assess(["python", "x.py"], repo, snapshot_commit="abc")
    assert verdict.to_dict()["ok"] is True
