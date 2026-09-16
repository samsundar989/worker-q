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
    # The scripted remotes below answer every command generically, so the
    # staging directory is resolved up front, as a live node would resolve it.
    staging._EXPANDED["w"] = {staging.REMOTE_STAGE_DIR: r"C:\T\workerq-bundles"}
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


def test_a_node_that_cannot_answer_is_not_remembered_as_the_answer(monkeypatch):
    """The 3-day outage of 2026-09-13..16.

    The dispatcher started while the node was powered off. Expansion failed,
    fell back to the literal `%USERPROFILE%\\Documents`, and cached it - so
    every job afterwards shipped a working directory the node's Python could
    not resolve, and every placement was refused until the process restarted.
    """
    down = lambda node, command, *, timeout=None: RemoteResult(False, 255, "", "", "timed out")
    monkeypatch.setattr(staging.nodes, "run_remote", down)
    with pytest.raises(staging.StagingError, match="could not resolve"):
        staging.expand_remote(node(), "%USERPROFILE%")

    up = FakeRemote({"echo": r"C:\Users\me"})
    monkeypatch.setattr(staging.nodes, "run_remote", up)
    assert staging.expand_remote(node(), "%USERPROFILE%") == r"C:\Users\me"


def test_an_unexpanded_echo_is_not_an_answer(monkeypatch):
    """cmd.exe echoes an undefined variable back verbatim."""
    monkeypatch.setattr(staging.nodes, "run_remote", FakeRemote({"echo": "%NOPE%"}))
    with pytest.raises(staging.StagingError):
        staging.expand_remote(node(), "%NOPE%")


def test_a_failed_repo_scan_is_not_cached_as_an_empty_index(monkeypatch):
    """An empty index maps every repo to its directory name for good."""
    down = lambda node, command, *, timeout=None: RemoteResult(False, 255, "", "", "timed out")
    monkeypatch.setattr(staging.nodes, "run_remote", down)
    assert staging.index_repos(node()) == {}

    up = FakeRemote({"Get-ChildItem": "worker-q|git@github.com:me/worker-q"})
    monkeypatch.setattr(staging.nodes, "run_remote", up)
    assert staging.index_repos(node()) == {"worker-q": "github.com/me/worker-q"}


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


# --------------------------------------------------------------------------
# Where results are looked for
# --------------------------------------------------------------------------


@pytest.fixture
def staged(monkeypatch, tmp_path):
    """A repo whose far-side path needs no scan, and a scripted remote."""
    repo = tmp_path / "arc-whest"
    repo.mkdir()
    fake = FakeRemote(default="COLLECTED 0")
    monkeypatch.setattr(staging, "origin_url", lambda _r: None)
    monkeypatch.setattr(staging.nodes, "run_remote", fake)
    monkeypatch.setattr(
        staging.nodes, "copy_to_node", lambda *a, **k: RemoteResult(True, 0, "", "")
    )
    return repo, fake


def test_a_worktree_left_by_a_failed_placement_is_reused(staged, monkeypatch):
    """A bare `git worktree add` failed every retry with "already exists".

    The tree is left behind whenever placement fails after materialising it,
    and biohub job 1438 was retried 187 times against its own leftover.
    """
    repo, fake = staged
    monkeypatch.setattr(staging, "bundle_bases", lambda *_a: [])
    monkeypatch.setattr(
        staging, "build_bundle",
        lambda *_a: (_ for _ in ()).throw(staging.StagingError("EMPTY_BUNDLE")),
    )
    staging.ship_snapshot(node(), repo, job_id=450, commit="abc123")

    add = [c for c in fake.calls if "worktree add" in c]
    assert len(add) == 1
    worktree = staging.worktree_path(node(), repo, 450)
    assert f"if exist {worktree}\\.git" in add[0]
    assert "findstr /b abc123" in add[0], "reused only when it is the same commit"


def test_the_worktree_path_has_a_single_definition(staged):
    """Three callers have to agree on this, so it is worth pinning."""
    repo, _ = staged
    assert staging.worktree_path(node(), repo, 450) == (
        "D:" + chr(92) + "repos" + chr(92) + "arc-whest" + chr(92)
        + ".gpuq-work" + chr(92) + "job-000450"
    )


def test_the_collector_searches_the_worktree_before_the_live_tree(staged):
    """The bug this fixes: results written to a *relative* path live in the
    worktree, and searching only the live tree reported "nothing was written".

    Worktree first, so a job's own copy beats a stale one in the live tree.
    """
    import re

    repo, fake = staged
    staging.collect_outputs(
        node(),
        repo,
        ["experiments/x.json"],
        job_id=450,
        since_utc="2026-09-09T20:00:00+00:00",
    )

    collect = [c for c in fake.calls if "-Roots" in c]
    assert collect, f"no collector invocation in {fake.calls}"
    roots = re.search(r'-Roots "([^"]+)"', collect[0]).group(1).split("|")
    assert len(roots) == 2, roots
    assert roots[0] == staging.worktree_path(node(), repo, 450)
    assert roots[1] == staging.remote_repo_path(node(), repo)


def test_declaring_no_outputs_still_costs_no_round_trip(staged):
    repo, fake = staged
    result = staging.collect_outputs(
        node(), repo, [], job_id=1, since_utc="2026-09-09T20:00:00+00:00"
    )
    assert result["collected"] == 0
    assert not [c for c in fake.calls if "-Roots" in c]


# --------------------------------------------------------------------------
# Inputs named on the command line that live in passthrough data
# --------------------------------------------------------------------------


def _tree(root, files):
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def test_a_script_in_passthrough_data_travels_with_its_folder(tmp_path):
    """Four jobs failed on the 3080 Ti with "can't open file ...run_arms.py".

    The harness was written into `.cache/market_20260916/` minutes before it
    was submitted, and the node's `.cache` had never heard of it.
    """
    _tree(tmp_path, {
        ".cache/market_20260916/run_arms.py": "x",
        ".cache/market_20260916/patch_builder.py": "y",
        ".cache/other/huge.bin": "z",
        "benchmarks/panels/field.json": "{}",
        "benchmarks/panels/unrelated.json": "{}",
        "agents/tracked.py": "t",
    })
    argv = [
        ".venv/Scripts/python.exe", ".cache/market_20260916/run_arms.py",
        "--manifest", "benchmarks/panels/field.json", "agents/tracked.py",
        "--out=.cache/other/huge.bin",
    ]
    files = staging.passthrough_inputs(
        argv, tmp_path, [".venv", ".cache", "benchmarks/panels"]
    )
    assert sorted(files) == [
        ".cache/market_20260916/patch_builder.py",
        ".cache/market_20260916/run_arms.py",
        ".cache/other/huge.bin",
        "benchmarks/panels/field.json",
    ], "siblings in a small folder come along; a passthrough root is never sent whole"


def test_nothing_is_sent_for_a_command_naming_no_passthrough_files(tmp_path):
    _tree(tmp_path, {"tools/x.py": "x"})
    assert staging.passthrough_inputs(["python", "tools/x.py"], tmp_path, [".cache"]) == []


def test_inputs_too_large_to_send_keep_the_job_home(monkeypatch, tmp_path):
    _tree(tmp_path, {".cache/big.bin": "x"})
    monkeypatch.setattr(staging, "_SYNC_MAX_BYTES", 0)
    with pytest.raises(staging.StagingError, match="too large"):
        staging.sync_inputs(node(), tmp_path, ["python", ".cache/big.bin"], [".cache"])


def test_the_interpreter_and_its_venv_are_never_sent(tmp_path):
    """The first live run pushed a whole `.venv/Scripts` over the node's own."""
    _tree(tmp_path, {
        ".venv/pyvenv.cfg": "home = x",
        ".venv/Scripts/python.exe": "bin",
        ".venv/Scripts/tool.py": "t",
        ".venv/Lib/site-packages/mod.py": "m",
    })
    argv = [".venv/Scripts/python.exe", ".venv/Scripts/tool.py", ".venv/Lib/site-packages/mod.py"]
    assert staging.passthrough_inputs(argv, tmp_path, [".venv"]) == []


# --------------------------------------------------------------------------
# Missing passthrough data small enough to send
# --------------------------------------------------------------------------


def test_small_missing_passthrough_is_pushable(tmp_path):
    """300 KB of engine/bin kept every kaggriculture job off the 3080 Ti."""
    _tree(tmp_path, {"engine/bin/kagx.pyd": "x", "benchmarks/panels/a.json": "{}"})
    files = staging.pushable_passthrough(tmp_path, ["engine/bin", "benchmarks/panels"])
    assert sorted(files) == ["benchmarks/panels/a.json", "engine/bin/kagx.pyd"]


@pytest.mark.parametrize(
    "setup, missing",
    [
        ({".venv/pyvenv.cfg": "h", ".venv/Scripts/python.exe": "b"}, [".venv"]),
        ({}, ["not-here"]),
        ({"data/x.bin": "x"}, ["C:/abs/data"]),
    ],
)
def test_what_cannot_be_pushed_is_refused_whole(tmp_path, setup, missing):
    _tree(tmp_path, setup)
    assert staging.pushable_passthrough(tmp_path, missing) is None


def test_large_passthrough_is_not_pushed(tmp_path):
    _tree(tmp_path, {"weights/w.f32": "x" * 100})
    assert staging.pushable_passthrough(tmp_path, ["weights"], max_bytes=10) is None
