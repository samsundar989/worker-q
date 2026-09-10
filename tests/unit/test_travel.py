"""Whether a job is safe to run on a different machine, and what comes back.

Phase 4b of docs/multi-node.md. Eligibility asks "is the data there"; this asks
the question eligibility cannot see, because its failure does not look like a
failure: a job that writes to an absolute path under the repository resolves
that path on either machine, so it succeeds and leaves its output on a machine
nobody is looking at.

Refusing such a job was the first answer and it was too strict - `arc-whest` and
`biohub` both mandate absolute output paths, so refusal locked them out of the
second machine entirely. The write target is now adopted as a declared output
instead, and `staging.collect_outputs` copies it home.
"""

from __future__ import annotations

import pytest

from workerq import travel


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "biohub"
    (root / "outputs").mkdir(parents=True)
    return root


def test_the_biohub_pattern_travels_and_is_collected(repo):
    """The real case, from a real `.gpuq.toml`.

    biohub instructs jobs to write submissions to an absolute path, because a
    relative path would land inside the snapshot and be deleted when it
    expires. That instruction is correct, so the job must travel - and its
    output must be declared, or it would be left on the node.
    """
    argv = ["python", "predict.py", "--out", str(repo / "outputs" / "submission.csv")]
    verdict = travel.assess(argv, repo, snapshot_commit="abc123")
    assert verdict.ok
    assert verdict.repo_paths
    assert verdict.adopt_outputs == ["outputs/submission.csv"]


def test_the_equals_form_is_adopted_too(repo):
    argv = ["python", "run.py", f"--out={repo / 'outputs' / 'x.csv'}"]
    verdict = travel.assess(argv, repo, snapshot_commit="abc")
    assert verdict.ok
    assert verdict.adopt_outputs == ["outputs/x.csv"]


def test_adopted_paths_use_forward_slashes(repo):
    """`.gpuq.toml` spells outputs this way and the collector converts itself."""
    argv = ["python", "x.py", "--out", str(repo / "outputs" / "deep" / "y.json")]
    adopted = travel.assess(argv, repo, snapshot_commit="abc").adopt_outputs
    assert adopted == ["outputs/deep/y.json"]
    assert "\\" not in adopted[0]


def test_an_already_declared_output_is_not_adopted_twice(repo):
    argv = ["python", "x.py", "--out", str(repo / "outputs" / "submission.csv")]
    verdict = travel.assess(
        argv, repo, outputs=["outputs/submission.csv"], snapshot_commit="abc"
    )
    assert verdict.ok
    assert verdict.adopt_outputs == []


def test_a_declared_parent_directory_already_covers_it(repo):
    """Declaring `outputs` collects everything under it, so naming the file
    again would only make the spec longer."""
    argv = ["python", "x.py", "--out", str(repo / "outputs" / "submission.csv")]
    verdict = travel.assess(argv, repo, outputs=["outputs"], snapshot_commit="abc")
    assert verdict.adopt_outputs == []


def test_a_relative_output_path_travels_and_adopts_nothing(repo):
    argv = ["python", "predict.py", "--out", "outputs/submission.csv"]
    verdict = travel.assess(argv, repo, snapshot_commit="abc")
    assert verdict.ok
    assert verdict.adopt_outputs == []


def test_an_absolute_path_outside_the_repo_is_left_alone(repo, tmp_path):
    """A dataset elsewhere is a read. If it is missing on the node the job
    fails loudly, which is the safe direction - unlike a path that resolves."""
    argv = ["python", "train.py", "--data", str(tmp_path / "datasets" / "imagenet")]
    verdict = travel.assess(argv, repo, snapshot_commit="abc")
    assert verdict.ok
    assert verdict.adopt_outputs == []


def test_the_interpreter_is_never_adopted(repo):
    """argv[0] resolves to the node's own venv, which is what should happen."""
    argv = [str(repo / ".venv" / "Scripts" / "python.exe"), "train.py"]
    verdict = travel.assess(argv, repo, snapshot_commit="abc")
    assert verdict.ok
    assert verdict.adopt_outputs == []


def test_a_job_with_no_snapshot_cannot_travel(repo):
    """There is no commit to ship, so there is nothing to reproduce there."""
    verdict = travel.assess(["python", "x.py"], repo, snapshot_commit=None)
    assert not verdict.ok
    assert "no git snapshot" in verdict.reasons[0]


def test_a_job_outside_any_repository_cannot_travel():
    assert not travel.assess(["python", "x.py"], None, snapshot_commit="abc").ok


def test_every_offending_path_is_adopted_not_just_the_first_three(repo):
    """The old code summarised these into a refusal message. Every one of them
    now has to be collected, so none may be dropped."""
    argv = ["python", "x.py"] + [str(repo / "outputs" / f"f{i}.csv") for i in range(6)]
    verdict = travel.assess(argv, repo, snapshot_commit="abc")
    assert verdict.ok
    assert len(verdict.repo_paths) == 6
    assert verdict.adopt_outputs == [f"outputs/f{i}.csv" for i in range(6)]


def test_a_duplicated_path_is_adopted_once(repo):
    target = str(repo / "outputs" / "same.csv")
    verdict = travel.assess(["python", "x.py", "--out", target, "--report", target],
                            repo, snapshot_commit="abc")
    assert verdict.adopt_outputs == ["outputs/same.csv"]


def test_the_verdict_is_serialisable(repo):
    verdict = travel.assess(["python", "x.py"], repo, snapshot_commit="abc")
    payload = verdict.to_dict()
    assert payload["ok"] is True
    assert payload["adopt_outputs"] == []


def test_repo_relative_refuses_a_path_it_cannot_express(repo):
    assert travel.repo_relative("C:/elsewhere/x.json", repo) is None


# --------------------------------------------------------------------------
# Pointing a job at the node's own copy of the repository
# --------------------------------------------------------------------------


def test_paths_are_rebased_when_the_node_spells_the_repo_differently(repo):
    """worker-q's own repo is `gpu-queue` here and `worker-q` on the node,
    because the far side is matched by git origin, not directory name.

    Without rebasing, an adopted absolute write lands outside the node's
    repository and the collector never sees it."""
    argv = [
        str(repo / ".venv" / "python.exe"),
        "--out",
        str(repo / "outputs" / "a.json"),
        f"--report={repo / 'outputs' / 'b.json'}",
        "--data",
        "D:/elsewhere/set.npz",
    ]
    out = travel.rebase_repo_paths(argv, repo, r"D:\repos\worker-q")

    assert out[0] == r"D:\repos\worker-q\.venv\python.exe"
    assert out[2] == r"D:\repos\worker-q\outputs\a.json"
    assert out[3] == r"--report=D:\repos\worker-q\outputs\b.json"
    assert out[5] == "D:/elsewhere/set.npz", "a path outside the repo is not ours to move"


def test_rebasing_is_a_no_op_when_both_machines_agree(repo):
    """The common case, and it must not churn the command line."""
    argv = ["python", "x.py", "--out", str(repo / "outputs" / "a.json")]
    assert travel.rebase_repo_paths(argv, repo, str(repo)) == argv


def test_a_path_merely_sharing_a_prefix_is_not_rebased(repo, tmp_path):
    """`biohub-old` must not be rewritten because `biohub` is the repo."""
    sibling = str(tmp_path / "biohub-old" / "x.json")
    out = travel.rebase_repo_paths(["python", sibling], repo, r"D:\repos\w")
    assert out[1] == sibling
