"""Immutable source snapshots (spec section 12).

Agents keep editing while a job waits in the queue, so the source a job runs
must be frozen at submission time. We build an ephemeral commit through a
*temporary* Git index - the user's real index, working tree and branch are
never touched - and check it out into a detached worktree.

Guarantee under test (`tests/integration/test_snapshot_execution.py`): a job
submitted at source state S runs S, no matter what the repo looks like when
the dispatcher finally starts it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from workerq.util import ensure_dir, expand_path, is_within, resolve_path
from workerq.winproc import no_window_kwargs

GIT_TIMEOUT = 300

#: Ephemeral snapshot commits live under this ref namespace so Git's garbage
#: collector cannot prune a queued job's source out from under it.
SNAPSHOT_REF_PREFIX = "refs/gpuq/snapshots"


class SnapshotError(RuntimeError):
    """Snapshot creation failed. Submission must abort before enqueue."""


@dataclass
class Snapshot:
    mode: str  # git | copy | live | none
    path: Path | None
    commit: str | None = None
    repo_root: Path | None = None
    passthrough: list[str] = field(default_factory=list)
    ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "path": str(self.path) if self.path else None,
            "commit": self.commit,
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "passthrough": self.passthrough,
            "ref": self.ref,
        }


# --------------------------------------------------------------------------
# Git plumbing
# --------------------------------------------------------------------------


def git_available() -> bool:
    return shutil.which("git") is not None


def _git(
    args: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run git with an argv vector. Never a shell string."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    # Keep snapshot commits reproducible and independent of user hooks/config.
    full_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            env=full_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT,
            **no_window_kwargs(),
        )
    except FileNotFoundError as exc:
        raise SnapshotError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SnapshotError(f"git {' '.join(args[:2])} timed out") from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SnapshotError(f"git {' '.join(args[:3])} failed: {detail}")
    return proc


def find_repo_root(path: Path | str) -> Path | None:
    """Top level of the Git working tree containing `path`, if any."""
    start = expand_path(path)
    if not start.exists():
        return None
    proc = _git(["rev-parse", "--show-toplevel"], cwd=start, check=False)
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    return expand_path(text)


def has_head(repo_root: Path) -> bool:
    """False for a freshly `git init`-ed repository with no commits yet."""
    return _git(["rev-parse", "--verify", "-q", "HEAD"], cwd=repo_root, check=False).returncode == 0


def head_commit(repo_root: Path) -> str | None:
    proc = _git(["rev-parse", "HEAD"], cwd=repo_root, check=False)
    return (proc.stdout or "").strip() or None if proc.returncode == 0 else None


def read_index_hash(repo_root: Path) -> str | None:
    """Fingerprint of the real index, used by tests to prove we did not touch it."""
    index = repo_root / ".git" / "index"
    if not index.exists():
        return None
    import hashlib

    return hashlib.sha256(index.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Snapshot creation
# --------------------------------------------------------------------------


def create_git_snapshot(
    repo_root: Path,
    destination: Path,
    *,
    job_id: int,
    passthrough: list[str] | None = None,
    message: str | None = None,
) -> Snapshot:
    """Freeze the working tree into a detached worktree at `destination`.

    Captures tracked content, staged and unstaged edits, and untracked
    non-ignored files. Ignored paths stay out; bring them in with passthrough.
    """
    repo_root = resolve_path(repo_root)
    if not (repo_root / ".git").exists():
        raise SnapshotError(f"not a git repository: {repo_root}")

    destination = expand_path(destination)
    if destination.exists():
        raise SnapshotError(f"snapshot destination already exists: {destination}")
    ensure_dir(destination.parent)

    tmp_index_dir = Path(tempfile.mkdtemp(prefix=f"gpuq-index-{job_id}-"))
    tmp_index = tmp_index_dir / "index"
    commit: str | None = None
    ref = f"{SNAPSHOT_REF_PREFIX}/{job_id}"

    try:
        env = {"GIT_INDEX_FILE": str(tmp_index)}

        parent = head_commit(repo_root) if has_head(repo_root) else None
        if parent:
            # Seed the temp index from HEAD so unchanged tracked files are
            # cheap; then overlay the entire working tree.
            _git(["read-tree", parent], cwd=repo_root, env=env)

        # `add -A` from the repo root stages every tracked change plus untracked
        # non-ignored files, and records deletions.
        _git(["add", "-A", "--", "."], cwd=repo_root, env=env)

        tree = (_git(["write-tree"], cwd=repo_root, env=env).stdout or "").strip()
        if not tree:
            raise SnapshotError("git write-tree produced no tree object")

        commit_args = ["commit-tree", tree]
        if parent:
            commit_args += ["-p", parent]
        commit_args += ["-m", message or f"gpuq snapshot job {job_id}"]
        commit_env = {
            "GIT_AUTHOR_NAME": "gpuq",
            "GIT_AUTHOR_EMAIL": "gpuq@localhost",
            "GIT_COMMITTER_NAME": "gpuq",
            "GIT_COMMITTER_EMAIL": "gpuq@localhost",
            **env,
        }
        commit = (_git(commit_args, cwd=repo_root, env=commit_env).stdout or "").strip()
        if not commit:
            raise SnapshotError("git commit-tree produced no commit object")

        # Anchor the commit so `git gc` cannot collect a queued job's source.
        _git(["update-ref", ref, commit], cwd=repo_root)

        _git(
            ["worktree", "add", "--detach", str(destination), commit],
            cwd=repo_root,
        )
        if not destination.is_dir():
            raise SnapshotError("git worktree add did not create the snapshot directory")

        linked = apply_passthrough(repo_root, destination, passthrough or [])
        return Snapshot(
            mode="git",
            path=destination,
            commit=commit,
            repo_root=repo_root,
            passthrough=linked,
            ref=ref,
        )
    except BaseException:
        # Leave nothing half-built behind.
        try:
            if destination.exists():
                _git(
                    ["worktree", "remove", "--force", str(destination)],
                    cwd=repo_root,
                    check=False,
                )
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
        except Exception:
            pass
        try:
            _git(["update-ref", "-d", ref], cwd=repo_root, check=False)
        except Exception:
            pass
        raise
    finally:
        shutil.rmtree(tmp_index_dir, ignore_errors=True)


def create_copy_snapshot(
    source: Path, destination: Path, *, passthrough: list[str] | None = None
) -> Snapshot:
    """Plain copy snapshot for a small non-Git tree (spec section 12.5)."""
    source = resolve_path(source)
    destination = expand_path(destination)
    if destination.exists():
        raise SnapshotError(f"snapshot destination already exists: {destination}")
    ensure_dir(destination.parent)

    skip = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache"}
    passthrough = passthrough or []

    def _ignore(directory: str, names: list[str]) -> set[str]:
        rel = os.path.relpath(directory, str(source))
        out = {n for n in names if n in skip}
        for entry in passthrough:
            candidate = os.path.normpath(os.path.join(rel, ""))
            if candidate in (".", "") and entry in names:
                out.add(entry)
        return out

    try:
        shutil.copytree(source, destination, ignore=_ignore, symlinks=True)
    except OSError as exc:
        raise SnapshotError(f"copy snapshot failed: {exc}") from exc

    linked = apply_passthrough(source, destination, passthrough)
    return Snapshot(
        mode="copy", path=destination, commit=None, repo_root=source, passthrough=linked
    )


# --------------------------------------------------------------------------
# Passthrough (ignored runtime data - spec section 12.4)
# --------------------------------------------------------------------------


def _link_dir(target: Path, link: Path) -> bool:
    """Link a directory without needing administrator rights."""
    if os.name == "nt":
        try:
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
            return True
        except (OSError, AttributeError, NotImplementedError):
            pass
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


def _link_file(target: Path, link: Path) -> bool:
    try:
        os.link(str(target), str(link))
        return True
    except (OSError, NotImplementedError):
        pass
    try:
        os.symlink(str(target), str(link))
        return True
    except (OSError, NotImplementedError):
        pass
    try:
        shutil.copy2(str(target), str(link))
        return True
    except OSError:
        return False


def apply_passthrough(
    live_root: Path, snapshot_root: Path, entries: list[str]
) -> list[str]:
    """Expose large ignored paths (datasets, checkpoints) inside the snapshot.

    Never copies bulk data: directories become junctions/symlinks back to the
    live path. Relative entries are confined to the repository; only an
    explicit absolute path may point outside it.
    """
    live_root = resolve_path(live_root)
    snapshot_root = resolve_path(snapshot_root)
    linked: list[str] = []

    for entry in entries:
        raw = str(entry).strip()
        if not raw:
            continue
        candidate = Path(os.path.expandvars(raw)).expanduser()

        if candidate.is_absolute():
            target = resolve_path(candidate)
            link = snapshot_root / target.name
        else:
            target = resolve_path(live_root / raw)
            if not is_within(target, live_root):
                raise SnapshotError(
                    f"passthrough {raw!r} escapes the repository; "
                    "use an absolute --passthrough path if that is intended"
                )
            link = snapshot_root / raw

        if not target.exists():
            continue
        if not is_within(link, snapshot_root):
            raise SnapshotError(f"passthrough destination escapes the snapshot: {raw!r}")
        if link.exists() or link.is_symlink():
            continue  # already present from the snapshot itself

        ensure_dir(link.parent)
        ok = _link_dir(target, link) if target.is_dir() else _link_file(target, link)
        if ok:
            linked.append(raw)
    return linked


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------


def remove_snapshot(
    snapshot_path: Path,
    *,
    repo_root: Path | None = None,
    state_root: Path | None = None,
    ref: str | None = None,
) -> bool:
    """Delete a snapshot worktree.

    Refuses any path outside the worker-q state directory (spec section 30:
    resolve targets and verify descent before deleting).
    """
    snapshot_path = expand_path(snapshot_path)
    if state_root is not None and not is_within(snapshot_path, expand_path(state_root)):
        raise SnapshotError(
            f"refusing to delete {snapshot_path}: outside the gpuq state directory"
        )
    if not snapshot_path.exists():
        _drop_ref(repo_root, ref)
        return True

    # Passthrough junctions/symlinks point at live datasets. They must be
    # unlinked *before* anything recursive runs - including git's own removal -
    # or deleting a snapshot would delete the user's real data.
    unlink_reparse_points(snapshot_path)

    if repo_root is not None and (expand_path(repo_root) / ".git").exists():
        _git(
            ["worktree", "remove", "--force", str(snapshot_path)],
            cwd=repo_root,
            check=False,
        )
    if snapshot_path.exists():
        _rmtree_no_follow(snapshot_path)

    if repo_root is not None and (expand_path(repo_root) / ".git").exists():
        _git(["worktree", "prune"], cwd=repo_root, check=False)
    _drop_ref(repo_root, ref)
    return not snapshot_path.exists()


def _drop_ref(repo_root: Path | None, ref: str | None) -> None:
    if repo_root and ref and (expand_path(repo_root) / ".git").exists():
        _git(["update-ref", "-d", ref], cwd=repo_root, check=False)


def is_reparse_point(path: Path) -> bool:
    """True for a symlink or a Windows directory junction.

    `os.path.islink` is False for junctions and `os.walk` happily descends
    them, so relying on either would let a recursive delete escape the
    snapshot and destroy live data. `os.path.isjunction` only exists from
    Python 3.12, hence the attribute-bit fallback.
    """
    try:
        if os.path.islink(str(path)):
            return True
    except OSError:  # pragma: no cover
        return False
    if hasattr(os.path, "isjunction"):
        try:
            if os.path.isjunction(str(path)):
                return True
        except OSError:  # pragma: no cover
            return False
    if os.name == "nt":
        FILE_ATTRIBUTE_REPARSE_POINT = 0x400
        try:
            attributes = os.lstat(str(path)).st_file_attributes
        except (OSError, AttributeError):
            return False
        return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def unlink_reparse_points(root: Path) -> list[str]:
    """Detach every symlink/junction under `root` without following any.

    Returns the paths detached. Safe to call on a tree that has none.
    """
    detached: list[str] = []
    if not root.exists() and not is_reparse_point(root):
        return detached

    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            path = Path(entry.path)
            if is_reparse_point(path):
                try:
                    # rmdir/unlink removes the link itself, never the target.
                    if entry.is_dir(follow_symlinks=False):
                        os.rmdir(path)
                    else:
                        os.unlink(path)
                    detached.append(str(path))
                except OSError:
                    pass
                continue
            if entry.is_dir(follow_symlinks=False):
                stack.append(path)
    return detached


def _rmtree_no_follow(path: Path) -> None:
    """Remove a tree after detaching every reparse point inside it."""
    unlink_reparse_points(path)
    if is_reparse_point(path):
        try:
            os.rmdir(path) if path.is_dir() else os.unlink(path)
        except OSError:  # pragma: no cover
            pass
        return

    def _on_error(func, target, _exc) -> None:
        try:
            os.chmod(target, 0o600)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_on_error)


def prune_worktrees(repo_root: Path) -> None:
    _git(["worktree", "prune"], cwd=repo_root, check=False)


# --------------------------------------------------------------------------
# Project-level configuration (.gpuq.toml)
# --------------------------------------------------------------------------


def load_project_passthrough(repo_root: Path | None) -> list[str]:
    """Read `[snapshot] passthrough` from a repository's `.gpuq.toml`."""
    if repo_root is None:
        return []
    path = expand_path(repo_root) / ".gpuq.toml"
    if not path.exists():
        return []
    import tomllib

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    section = data.get("snapshot")
    if not isinstance(section, dict):
        return []
    values = section.get("passthrough")
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if str(v).strip()]


def load_project_defaults(repo_root: Path | None) -> dict:
    """Read optional `[project]` defaults from a repository's `.gpuq.toml`."""
    if repo_root is None:
        return {}
    path = expand_path(repo_root) / ".gpuq.toml"
    if not path.exists():
        return {}
    import tomllib

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    section = data.get("project")
    return section if isinstance(section, dict) else {}
