"""Getting a job's frozen source onto another machine.

Phase 3 of docs/multi-node.md. No SSH here - the remote is stubbed, so these
pin the decisions rather than the plumbing: which repository on the far side is
*this* repository, what a bundle may exclude, and what happens when the node
already has the commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from workerq import staging
from workerq.config import NodeConfig
from workerq.nodes import RemoteResult


def node() -> NodeConfig:
    return NodeConfig(name="w", address="host", repo_root=r"D:\repos")


class FakeRemote:
    """Records commands and replies from a scripted table."""

    def __init__(self, replies: dict[str, str] | None = None, default: str = ""):
        self.replies = replies or {}
        self.default = default
        self.calls: list[str] = []

    def __call__(self, node, command, *, timeout=None):
        self.calls.append(command)
        for fragment, out in self.replies.items():
            if fragment in command:
                return RemoteResult(True, 0, out, "")
        return RemoteResult(True, 0, self.default, "")


@pytest.fixture(autouse=True)
def clear_caches():
    staging._REPO_INDEX.clear()
    staging._EXPANDED.clear()
    yield
    staging._REPO_INDEX.clear()
    staging._EXPANDED.clear()


# --------------------------------------------------------------------------
# Identifying the repository on the far side
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        ("git@github.com:me/x.git", "https://github.com/me/x"),
        ("https://github.com/me/x/", "git@github.com:me/x"),
        ("ssh://git@github.com/me/X.git", "https://github.com/me/x"),
    ],
)
def test_the_same_repository_spelled_differently_is_the_same_repository(a, b):
    assert staging.normalise_origin(a) == staging.normalise_origin(b)


def test_different_repositories_do_not_collide():
    assert staging.normalise_origin("git@github.com:me/x") != staging.normalise_origin(
        "git@github.com:me/y"
    )


def test_a_repo_is_found_by_origin_when_the_directory_name_differs(monkeypatch, tmp_path):
    """The two machines really do disagree: `gpu-queue` here, `worker-q` there.

    Deriving the remote path from the local directory name looked obviously
    right and was wrong on the very first repository it was tried against.
    """
    repo = tmp_path / "gpu-queue"
    repo.mkdir()
    monkeypatch.setattr(staging, "origin_url", lambda _r: "git@github.com:me/worker-q.git")
    monkeypatch.setattr(
        staging.nodes,
        "run_remote",
        FakeRemote({"Get-ChildItem": "worker-q|https://github.com/me/worker-q\nother|git@github.com:me/other"}),
    )
    assert staging.remote_repo_path(node(), repo) == r"D:\repos\worker-q"


def test_the_matching_name_wins_when_two_clones_share_an_origin(monkeypatch, tmp_path):
    """A machine holding two clones of one repo must behave predictably."""
    repo = tmp_path / "kaggriculture"
    repo.mkdir()
    monkeypatch.setattr(staging, "origin_url", lambda _r: "git@github.com:me/kaggriculture")
    monkeypatch.setattr(
        staging.nodes,
        "run_remote",
        FakeRemote({
            "Get-ChildItem": (
                "kaggriculture-old|git@github.com:me/kaggriculture\n"
                "kaggriculture|git@github.com:me/kaggriculture"
            )
        }),
    )
    assert staging.remote_repo_path(node(), repo) == r"D:\repos\kaggriculture"


def test_a_repo_with_no_origin_falls_back_to_its_directory_name(monkeypatch, tmp_path):
    repo = tmp_path / "local-only"
    repo.mkdir()
    monkeypatch.setattr(staging, "origin_url", lambda _r: None)
    monkeypatch.setattr(staging.nodes, "run_remote", FakeRemote())
    assert staging.remote_repo_path(node(), repo) == r"D:\repos\local-only"


def test_the_repo_index_is_read_once(monkeypatch):
    """A scan is a whole SSH connection; repositories are not cloned often."""
    fake = FakeRemote({"Get-ChildItem": "a|git@github.com:me/a"})
    monkeypatch.setattr(staging.nodes, "run_remote", fake)
    staging.index_repos(node())
    staging.index_repos(node())
    assert len(fake.calls) == 1
    staging.index_repos(node(), refresh=True)
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------
# Remote path expansion
# --------------------------------------------------------------------------


def test_percent_variables_are_resolved_before_anything_reaches_scp(monkeypatch):
    """scp talks to sftp, which does not expand `%TEMP%`.

    A command sent over SSH runs through cmd.exe and expands it; scp took the
    literal string as a directory name and failed. Anything handed to scp must
    be a real path first.
    """
    fake = FakeRemote({"echo": r"C:\Users\me\AppData\Local\Temp"})
    monkeypatch.setattr(staging.nodes, "run_remote", fake)
    assert staging.expand_remote(node(), "%TEMP%") == r"C:\Users\me\AppData\Local\Temp"
    # cached
    staging.expand_remote(node(), "%TEMP%")
    assert len(fake.calls) == 1


def test_a_path_with_no_variable_costs_no_round_trip(monkeypatch):
    fake = FakeRemote()
    monkeypatch.setattr(staging.nodes, "run_remote", fake)
    assert staging.expand_remote(node(), r"D:\plain") == r"D:\plain"
    assert fake.calls == []


# --------------------------------------------------------------------------
# Bundle bases
# --------------------------------------------------------------------------


def test_a_bundle_may_only_exclude_objects_this_machine_also_has(monkeypatch, tmp_path):
    """The node being ahead of us is normal, not an error.

    `git bundle` can only exclude objects the sender holds, and asking it to
    exclude one it lacks fails the entire bundle rather than that one base.
    """
    repo = tmp_path / "r"
    repo.mkdir()
    mine = "a" * 40
    theirs = "b" * 40
    monkeypatch.setattr(
        staging.nodes, "run_remote", FakeRemote({"for-each-ref": f"{mine}\n{theirs}\n"})
    )
    monkeypatch.setattr(staging, "have_object", lambda _r, oid: oid == mine)
    assert staging.bundle_bases(node(), repo) == [mine]


def test_a_node_that_cannot_be_asked_yields_no_bases(monkeypatch, tmp_path):
    """No bases means a fat bundle, which is slow but correct. Never wrong."""
    repo = tmp_path / "r"
    repo.mkdir()

    def refuse(node, command, *, timeout=None):
        return RemoteResult(False, 1, "", "", "unreachable")

    monkeypatch.setattr(staging.nodes, "run_remote", refuse)
    assert staging.bundle_bases(node(), repo) == []


# --------------------------------------------------------------------------
# Bundles, for real, against a throwaway repository
# --------------------------------------------------------------------------


def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo_with_snapshot(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "t@example.com"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "a.txt").write_text("one", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "first"], repo)
    base = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    (repo / "a.txt").write_text("two", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "second"], repo)
    head = _git(["rev-parse", "HEAD"], repo).stdout.strip()
    _git(["update-ref", "refs/gpuq/snapshots/7", head], repo)
    return repo, base, head


def test_a_bundle_excluding_what_the_node_has_is_far_smaller(repo_with_snapshot, tmp_path):
    """The whole economic argument, on a link measured at 6.1 MB/s."""
    repo, base, _head = repo_with_snapshot
    fat = staging.build_bundle(repo, "refs/gpuq/snapshots/7", [], tmp_path / "fat.bundle")
    thin = staging.build_bundle(
        repo, "refs/gpuq/snapshots/7", [base], tmp_path / "thin.bundle"
    )
    assert thin.stat().st_size < fat.stat().st_size


def test_a_node_that_already_has_the_commit_is_a_result_not_a_failure(
    repo_with_snapshot, tmp_path
):
    """Re-placing a job onto a node that ran it before must not error."""
    repo, _base, head = repo_with_snapshot
    with pytest.raises(staging.StagingError, match="EMPTY_BUNDLE"):
        staging.build_bundle(
            repo, "refs/gpuq/snapshots/7", [head], tmp_path / "empty.bundle"
        )


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------


def test_declaring_nothing_costs_no_round_trip(monkeypatch, tmp_path):
    fake = FakeRemote()
    monkeypatch.setattr(staging.nodes, "run_remote", fake)
    assert staging.verify_passthrough(node(), tmp_path / "r", []) == {}
    assert fake.calls == []


def test_a_missing_repository_is_reported_rather_than_guessed(monkeypatch, tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    monkeypatch.setattr(staging, "origin_url", lambda _r: None)
    monkeypatch.setattr(staging.nodes, "run_remote", FakeRemote(default="__NOREPO__"))
    status = staging.inspect_repo(node(), repo, [".venv"])
    assert not status.exists
    assert not status.ready
    assert status.error is None  # absent is not an error, it is a state


def test_every_passthrough_path_is_checked_in_one_call(monkeypatch, tmp_path):
    """Twenty-nine entries in twenty-nine calls would be sixteen seconds."""
    repo = tmp_path / "proj"
    repo.mkdir()
    entries = [".venv", "data/train", "models/clean"]
    reply = (
        "__HEAD__\nabc123\n__BRANCH__\nmain\n__ORIGIN__\ngit@github.com:me/proj\n"
        "__PT__.venv\nYES\n__PT__data/train\nNO\n__PT__models/clean\nYES\n"
    )
    fake = FakeRemote({"__HEAD__": reply}, default=reply)
    monkeypatch.setattr(staging, "origin_url", lambda _r: None)
    monkeypatch.setattr(staging.nodes, "run_remote", fake)

    status = staging.inspect_repo(node(), repo, entries)
    assert len(fake.calls) == 1
    assert status.passthrough == {".venv": True, "data/train": False, "models/clean": True}
    assert status.missing == ["data/train"]
    assert not status.ready


# --------------------------------------------------------------------------
# Removing a worktree must not destroy what it links to
# --------------------------------------------------------------------------


def test_junctions_are_detached_before_anything_recursive_runs(monkeypatch, tmp_path):
    """The local rule, applied to the remote path, after it was learned twice.

    `git worktree remove --force` deletes recursively and follows reparse
    points. The first version of this went straight to it and emptied a staged
    `.cache` through its junction on a real node; had the tree been biohub's it
    would have taken 80 GB of dataset with it.

    Locally the same mistake was made once before, which is why
    `snapshot.unlink_reparse_points` exists and why
    `test_cleanup_does_not_delete_live_passthrough_data` guards it.
    """
    order: list[str] = []

    def record(node, command, *, timeout=None):
        if "unlink-" in command and "powershell" in command:
            order.append("unlink")
            return RemoteResult(True, 0, "UNLINKED 3", "")
        if "worktree remove" in command:
            order.append("worktree-remove")
        return RemoteResult(True, 0, "", "")

    monkeypatch.setattr(staging, "origin_url", lambda _r: None)
    monkeypatch.setattr(staging.nodes, "run_remote", record)
    monkeypatch.setattr(
        staging.nodes, "copy_to_node", lambda *a, **k: RemoteResult(True, 0, "", "")
    )

    assert staging.remove_worktree(node(), tmp_path / "proj", 42)
    assert order.index("unlink") < order.index("worktree-remove")


def test_a_junction_that_will_not_detach_stops_the_removal(monkeypatch, tmp_path):
    """Better to leak a worktree than to walk through a link into live data."""

    def refuse_unlink(node, command, *, timeout=None):
        if "unlink-" in command and "powershell" in command:
            return RemoteResult(True, 0, "UNLINK_FAILED some-path", "")
        if "worktree remove" in command:
            raise AssertionError("must not remove while a junction is still attached")
        return RemoteResult(True, 0, "", "")

    monkeypatch.setattr(staging, "origin_url", lambda _r: None)
    monkeypatch.setattr(staging.nodes, "run_remote", refuse_unlink)
    monkeypatch.setattr(
        staging.nodes, "copy_to_node", lambda *a, **k: RemoteResult(True, 0, "", "")
    )

    assert staging.remove_worktree(node(), tmp_path / "proj", 42) is False


def test_removal_is_abandoned_if_the_unlinker_cannot_be_sent(monkeypatch, tmp_path):
    """No unlink step means removal is not safe to attempt at all."""

    def fail_copy(*a, **k):
        return RemoteResult(False, 1, "", "", "no route to host")

    def guard(node, command, *, timeout=None):
        if "worktree remove" in command:
            raise AssertionError("must not remove without unlinking first")
        return RemoteResult(True, 0, "", "")

    monkeypatch.setattr(staging, "origin_url", lambda _r: None)
    monkeypatch.setattr(staging.nodes, "run_remote", guard)
    monkeypatch.setattr(staging.nodes, "copy_to_node", fail_copy)

    assert staging.remove_worktree(node(), tmp_path / "proj", 42) is False
