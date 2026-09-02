"""Git snapshot behaviour (spec sections 12 and 23).

The invariant these tests defend: a snapshot is a faithful, immutable copy of
the working tree at submission time, produced without disturbing the user's
repository in any way.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from workerq.snapshot import (
    Snapshot,
    SnapshotError,
    apply_passthrough,
    create_copy_snapshot,
    create_git_snapshot,
    find_repo_root,
    has_head,
    head_commit,
    load_project_defaults,
    load_project_passthrough,
    read_index_hash,
    remove_snapshot,
)


def _snap(repo: Path, dest: Path, job_id: int = 1, **kwargs) -> Snapshot:
    return create_git_snapshot(repo, dest, job_id=job_id, **kwargs)


# --------------------------------------------------------------------------
# content capture
# --------------------------------------------------------------------------


def test_clean_repo_snapshot(git_repo: Path, tmp_path: Path):
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert snap.commit
    assert (snap.path / "tracked.txt").read_text(encoding="utf-8") == "original\n"


def test_staged_changes_included(git_repo: Path, tmp_path: Path, git_helper):
    (git_repo / "tracked.txt").write_text("staged\n", encoding="utf-8")
    git_helper(["add", "tracked.txt"], git_repo)
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert (snap.path / "tracked.txt").read_text(encoding="utf-8") == "staged\n"


def test_unstaged_changes_included(git_repo: Path, tmp_path: Path):
    (git_repo / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert (snap.path / "tracked.txt").read_text(encoding="utf-8") == "unstaged\n"


def test_untracked_non_ignored_file_included(git_repo: Path, tmp_path: Path):
    (git_repo / "brand_new.py").write_text("print('new')\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert (snap.path / "brand_new.py").exists()


def test_ignored_file_excluded(git_repo: Path, tmp_path: Path):
    (git_repo / "ignored").mkdir()
    (git_repo / "ignored" / "huge.bin").write_text("x" * 100, encoding="utf-8")
    (git_repo / "debug.log").write_text("noise\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert not (snap.path / "ignored").exists()
    assert not (snap.path / "debug.log").exists()


def test_deletion_is_captured(git_repo: Path, tmp_path: Path):
    (git_repo / "tracked.txt").unlink()
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert not (snap.path / "tracked.txt").exists()


def test_nested_directories_captured(git_repo: Path, tmp_path: Path):
    nested = git_repo / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("deep\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert (snap.path / "a" / "b" / "c" / "deep.py").read_text(encoding="utf-8") == "deep\n"


# --------------------------------------------------------------------------
# the user's repository must be untouched
# --------------------------------------------------------------------------


def test_real_git_index_unchanged(git_repo: Path, tmp_path: Path):
    (git_repo / "unstaged.py").write_text("x = 1\n", encoding="utf-8")
    before = read_index_hash(git_repo)
    _snap(git_repo, tmp_path / "snap" / "repo")
    assert read_index_hash(git_repo) == before


def test_real_branch_head_unchanged(git_repo: Path, tmp_path: Path):
    before = head_commit(git_repo)
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    ).stdout.strip()
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert head_commit(git_repo) == before
    assert snap.commit != before  # the snapshot is its own commit
    after_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert after_branch == branch


def test_working_tree_status_unchanged(git_repo: Path, tmp_path: Path):
    (git_repo / "wip.py").write_text("wip\n", encoding="utf-8")

    def status() -> str:
        return subprocess.run(
            ["git", "status", "--porcelain"], cwd=git_repo, capture_output=True, text=True
        ).stdout

    before = status()
    _snap(git_repo, tmp_path / "snap" / "repo")
    assert status() == before


def test_snapshot_is_immutable_against_later_edits(git_repo: Path, tmp_path: Path):
    """The core promise: editing after submission cannot change a snapshot."""
    (git_repo / "value.py").write_text('VALUE = "A"\n', encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")

    (git_repo / "value.py").write_text('VALUE = "B"\n', encoding="utf-8")
    (git_repo / "tracked.txt").write_text("edited later\n", encoding="utf-8")
    (git_repo / "added_later.py").write_text("late\n", encoding="utf-8")

    assert (snap.path / "value.py").read_text(encoding="utf-8") == 'VALUE = "A"\n'
    assert (snap.path / "tracked.txt").read_text(encoding="utf-8") == "original\n"
    assert not (snap.path / "added_later.py").exists()


def test_two_snapshots_are_independent(git_repo: Path, tmp_path: Path):
    (git_repo / "v.py").write_text("1\n", encoding="utf-8")
    first = _snap(git_repo, tmp_path / "s1" / "repo", job_id=1)
    (git_repo / "v.py").write_text("2\n", encoding="utf-8")
    second = _snap(git_repo, tmp_path / "s2" / "repo", job_id=2)

    assert (first.path / "v.py").read_text(encoding="utf-8") == "1\n"
    assert (second.path / "v.py").read_text(encoding="utf-8") == "2\n"
    assert first.commit != second.commit


# --------------------------------------------------------------------------
# awkward filenames
# --------------------------------------------------------------------------


def test_filename_with_spaces(git_repo: Path, tmp_path: Path):
    (git_repo / "my data file.txt").write_text("spaced\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert (snap.path / "my data file.txt").read_text(encoding="utf-8") == "spaced\n"


def test_unicode_filename(git_repo: Path, tmp_path: Path):
    (git_repo / "café_日本.py").write_text("# unicode\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert (snap.path / "café_日本.py").exists()


def test_filename_with_equals_and_brackets(git_repo: Path, tmp_path: Path):
    (git_repo / "cfg=v1[a].yaml").write_text("k: v\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo")
    assert (snap.path / "cfg=v1[a].yaml").exists()


# --------------------------------------------------------------------------
# unborn repository
# --------------------------------------------------------------------------


def test_unborn_repository(tmp_path: Path, git_helper):
    """A repo with no commits yet must snapshot, not crash."""
    repo = tmp_path / "fresh"
    repo.mkdir()
    git_helper(["init", "-q", "-b", "main", "."], repo)
    git_helper(["config", "user.email", "t@example.invalid"], repo)
    git_helper(["config", "user.name", "t"], repo)
    (repo / "first.py").write_text("print('first')\n", encoding="utf-8")

    assert not has_head(repo)
    snap = create_git_snapshot(repo, tmp_path / "snap" / "repo", job_id=1)
    assert snap.commit
    assert (snap.path / "first.py").read_text(encoding="utf-8") == "print('first')\n"
    assert not has_head(repo)  # still unborn afterwards


# --------------------------------------------------------------------------
# passthrough
# --------------------------------------------------------------------------


def test_passthrough_link_created(git_repo: Path, tmp_path: Path):
    data = git_repo / "ignored"
    data.mkdir()
    (data / "dataset.bin").write_text("big data\n", encoding="utf-8")

    snap = _snap(git_repo, tmp_path / "snap" / "repo", passthrough=["ignored"])
    assert "ignored" in snap.passthrough
    linked = snap.path / "ignored" / "dataset.bin"
    assert linked.exists()
    assert linked.read_text(encoding="utf-8") == "big data\n"


def test_passthrough_reflects_live_data(git_repo: Path, tmp_path: Path):
    """Passthrough is a link, not a copy: live data stays live."""
    data = git_repo / "ignored"
    data.mkdir()
    (data / "d.txt").write_text("v1\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo", passthrough=["ignored"])

    (data / "d.txt").write_text("v2\n", encoding="utf-8")
    assert (snap.path / "ignored" / "d.txt").read_text(encoding="utf-8") == "v2\n"


def test_passthrough_missing_path_is_skipped(git_repo: Path, tmp_path: Path):
    snap = _snap(git_repo, tmp_path / "snap" / "repo", passthrough=["nope"])
    assert snap.passthrough == []


def test_passthrough_does_not_overwrite_snapshot_content(git_repo: Path, tmp_path: Path):
    snap = _snap(git_repo, tmp_path / "snap" / "repo", passthrough=["tracked.txt"])
    assert (snap.path / "tracked.txt").read_text(encoding="utf-8") == "original\n"


def test_relative_passthrough_cannot_escape_repo(git_repo: Path, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="escapes the repository"):
        _snap(git_repo, tmp_path / "snap" / "repo", passthrough=["../outside"])


def test_absolute_passthrough_is_allowed(git_repo: Path, tmp_path: Path):
    external = tmp_path / "external_data"
    external.mkdir()
    (external / "f.txt").write_text("ext\n", encoding="utf-8")
    snap = _snap(git_repo, tmp_path / "snap" / "repo", passthrough=[str(external)])
    assert (snap.path / "external_data" / "f.txt").read_text(encoding="utf-8") == "ext\n"


def test_apply_passthrough_is_standalone(tmp_path: Path):
    live = tmp_path / "live"
    (live / "data").mkdir(parents=True)
    (live / "data" / "x.txt").write_text("d\n", encoding="utf-8")
    snap = tmp_path / "snap"
    snap.mkdir()
    linked = apply_passthrough(live, snap, ["data"])
    assert linked == ["data"]
    assert (snap / "data" / "x.txt").exists()


# --------------------------------------------------------------------------
# cleanup
# --------------------------------------------------------------------------


def test_cleanup_removes_worktree(git_repo: Path, tmp_path: Path):
    state = tmp_path / "state"
    snap = _snap(git_repo, state / "snapshots" / "1" / "repo")
    assert snap.path.exists()
    assert remove_snapshot(snap.path, repo_root=git_repo, state_root=state, ref=snap.ref)
    assert not snap.path.exists()

    worktrees = subprocess.run(
        ["git", "worktree", "list"], cwd=git_repo, capture_output=True, text=True
    ).stdout
    assert str(snap.path) not in worktrees


def test_cleanup_does_not_delete_live_passthrough_data(git_repo: Path, tmp_path: Path):
    """Deleting a snapshot must not follow a junction into the real dataset."""
    data = git_repo / "ignored"
    data.mkdir()
    (data / "precious.bin").write_text("do not delete\n", encoding="utf-8")

    state = tmp_path / "state"
    snap = _snap(git_repo, state / "snapshots" / "1" / "repo", passthrough=["ignored"])
    assert (snap.path / "ignored" / "precious.bin").exists()

    remove_snapshot(snap.path, repo_root=git_repo, state_root=state, ref=snap.ref)
    assert (data / "precious.bin").read_text(encoding="utf-8") == "do not delete\n"


def test_cleanup_refuses_paths_outside_state_dir(git_repo: Path, tmp_path: Path):
    state = tmp_path / "state"
    state.mkdir()
    with pytest.raises(SnapshotError, match="outside the gpuq state directory"):
        remove_snapshot(git_repo, repo_root=git_repo, state_root=state)
    assert git_repo.exists()


def test_cleanup_of_missing_path_is_ok(tmp_path: Path):
    state = tmp_path / "state"
    state.mkdir()
    assert remove_snapshot(state / "gone", state_root=state)


def test_snapshot_ref_anchors_commit(git_repo: Path, tmp_path: Path):
    """gc must not be able to prune a queued job's source."""
    snap = _snap(git_repo, tmp_path / "snap" / "repo", job_id=7)
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    ).stdout
    assert "refs/gpuq/snapshots/7" in refs


def test_destination_must_not_exist(git_repo: Path, tmp_path: Path):
    dest = tmp_path / "snap" / "repo"
    dest.mkdir(parents=True)
    with pytest.raises(SnapshotError, match="already exists"):
        _snap(git_repo, dest)


def test_failure_leaves_no_partial_snapshot(tmp_path: Path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    dest = tmp_path / "snap" / "repo"
    with pytest.raises(SnapshotError, match="not a git repository"):
        create_git_snapshot(not_a_repo, dest, job_id=1)
    assert not dest.exists()


# --------------------------------------------------------------------------
# repo discovery, copy mode, project config
# --------------------------------------------------------------------------


def test_find_repo_root(git_repo: Path, tmp_path: Path):
    nested = git_repo / "src" / "pkg"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == git_repo.resolve()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert find_repo_root(plain) is None


def test_copy_snapshot_for_non_git_tree(tmp_path: Path):
    source = tmp_path / "plain"
    source.mkdir()
    (source / "run.py").write_text("print(1)\n", encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "junk").write_text("junk\n", encoding="utf-8")

    snap = create_copy_snapshot(source, tmp_path / "snap" / "repo")
    assert snap.mode == "copy"
    assert (snap.path / "run.py").exists()
    assert not (snap.path / ".venv").exists()

    (source / "run.py").write_text("print(2)\n", encoding="utf-8")
    assert (snap.path / "run.py").read_text(encoding="utf-8") == "print(1)\n"


def test_load_project_passthrough(git_repo: Path):
    (git_repo / ".gpuq.toml").write_text(
        '[snapshot]\npassthrough = ["data", "checkpoints"]\n', encoding="utf-8"
    )
    assert load_project_passthrough(git_repo) == ["data", "checkpoints"]


def test_load_project_defaults(git_repo: Path):
    (git_repo / ".gpuq.toml").write_text('[project]\nname = "custom-name"\n', encoding="utf-8")
    assert load_project_defaults(git_repo).get("name") == "custom-name"


def test_missing_or_malformed_project_config_is_safe(git_repo: Path, tmp_path: Path):
    assert load_project_passthrough(git_repo) == []
    assert load_project_passthrough(None) == []
    (git_repo / ".gpuq.toml").write_text("[snapshot\nbroken", encoding="utf-8")
    assert load_project_passthrough(git_repo) == []


def test_concurrent_snapshots_of_one_repo(git_repo: Path, tmp_path: Path):
    """Several agents submitting at once must not collide in .git/worktrees.

    Regression guard: `git worktree add` names the worktree after the
    destination's basename, so snapshots that all ended in "/repo" raced and
    failed with "failed to read .git/worktrees/repo/commondir".
    """
    import threading

    errors: list[Exception] = []
    created: list[Snapshot] = []
    lock = threading.Lock()

    def worker(job_id: int) -> None:
        try:
            snap = create_git_snapshot(
                git_repo,
                tmp_path / "state" / "snapshots" / str(job_id) / f"job-{job_id}",
                job_id=job_id,
            )
            with lock:
                created.append(snap)
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)

    assert not errors, errors
    assert len(created) == 5
    for snap in created:
        assert (snap.path / "tracked.txt").read_text(encoding="utf-8") == "original\n"
