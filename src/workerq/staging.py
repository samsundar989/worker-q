"""Getting a job's frozen source onto another machine.

Phase 3 of docs/multi-node.md. Two machines, one Wi-Fi link measured at
6.1 MB/s, so the whole design is about *not* sending things.

The shape:

* The node clones each repo **once, from its own internet connection**, never
  across the link. A full clone of the repos here is 67-115 MB; the datasets
  they reference are 80 GB.
* Per job, only a **thin bundle** crosses - the objects the node does not
  already have. Measured against clones 130 and 159 commits behind, that was
  6.4 MB and 4.6 MB, about a second each. Against a current clone it is
  kilobytes.
* `--passthrough` data never crosses at all. It has to already be there, which
  is what `verify_passthrough` checks before a job is placed.

`snapshot.create_git_snapshot` already anchors each snapshot commit at
`refs/gpuq/snapshots/<id>` so `git gc` cannot prune a queued job's source. That
ref is exactly what a bundle needs to name, so shipping a snapshot needs no new
concept on the sending side - only a decision about what the far end already
has.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workerq import nodes
from workerq.config import NodeConfig
from workerq.snapshot import SNAPSHOT_REF_PREFIX
from workerq.winproc import no_window_kwargs

#: Where bundles land on the far side before being unpacked.
REMOTE_STAGE_DIR = "%TEMP%\\workerq-bundles"

#: A source delta above this is not a delta, it is an accident - a large
#: untracked file swept in by `git add -A`. At 6.1 MB/s this is ten seconds,
#: and far larger than any honest snapshot of source.
BUNDLE_WARN_BYTES = 64 * 1024 * 1024


class StagingError(RuntimeError):
    """Staging could not be completed. Never leaves a half-materialised tree."""


@dataclass
class RepoStatus:
    """What a node has for one project."""

    node: str
    project: str
    remote_path: str
    exists: bool = False
    head: str | None = None
    branch: str | None = None
    origin: str | None = None
    behind_by: int | None = None
    #: passthrough entry -> present on the node
    passthrough: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    @property
    def missing(self) -> list[str]:
        return [p for p, ok in self.passthrough.items() if not ok]

    @property
    def ready(self) -> bool:
        return self.exists and not self.missing and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "project": self.project,
            "remote_path": self.remote_path,
            "exists": self.exists,
            "head": self.head,
            "branch": self.branch,
            "origin": self.origin,
            "passthrough": self.passthrough,
            "missing": self.missing,
            "ready": self.ready,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# Local git
# --------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **no_window_kwargs(),
    )


def origin_url(repo_root: Path) -> str | None:
    proc = _git(["remote", "get-url", "origin"], repo_root)
    return proc.stdout.strip() if proc.returncode == 0 else None


def have_object(repo_root: Path, oid: str) -> bool:
    """Does this repo hold that object?

    A bundle may only exclude objects the *sender* has. When the node is ahead
    of us - or on a branch we never fetched - its refs are unusable as bundle
    bases, and asking git to exclude them fails the whole bundle rather than
    just that base.
    """
    return _git(["cat-file", "-e", f"{oid}^{{commit}}"], repo_root).returncode == 0


# --------------------------------------------------------------------------
# Remote repository
# --------------------------------------------------------------------------


def normalise_origin(url: str | None) -> str | None:
    """Compare remotes by identity, not by spelling.

    `git@github.com:me/x.git` and `https://github.com/me/x` are the same
    repository, and which one a clone happens to use is not something the
    scheduler should care about.
    """
    if not url:
        return None
    text = url.strip().rstrip("/")
    text = re.sub(r"\.git$", "", text)
    text = re.sub(r"^ssh://", "", text)
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^git@", "", text)
    text = text.replace(":", "/", 1) if "@" not in text and ":" in text else text
    return text.lower()


#: node name -> {directory name: normalised origin}. Repositories are not
#: cloned often, so one scan per process is plenty.
_REPO_INDEX: dict[str, dict[str, str]] = {}


def index_repos(node: NodeConfig, *, refresh: bool = False) -> dict[str, str]:
    """Every repository under the node's repo root, by its origin.

    One SSH call for the whole directory, because the connection is the cost.
    """
    if not refresh and node.name in _REPO_INDEX:
        return _REPO_INDEX[node.name]

    script = (
        "powershell -NoProfile -Command \""
        "Get-ChildItem -Path '{root}' -Directory -ErrorAction SilentlyContinue | "
        "ForEach-Object {{ $u = git -C $_.FullName remote get-url origin 2>$null; "
        "if ($u) {{ '{{0}}|{{1}}' -f $_.Name, $u.Trim() }} }}\""
    ).format(root=node.repo_root)
    result = nodes.run_remote(node, script, timeout=max(node.timeout_seconds, 60.0))

    index: dict[str, str] = {}
    if result.ok:
        for line in result.stdout.splitlines():
            if "|" not in line:
                continue
            name, _, url = line.partition("|")
            normalised = normalise_origin(url)
            if normalised:
                index[name.strip()] = normalised
    _REPO_INDEX[node.name] = index
    return index


def remote_repo_path(node: NodeConfig, repo_root: Path) -> str:
    """Where `repo_root` lives on `node`.

    Two machines need not agree on the directory name, and here they do not:
    this repository is `gpu-queue` on the primary and `worker-q` on the worker.
    So identity is the **origin URL**, with the directory name as a fast path
    and as the fallback for a repo that has no origin.

    Only the parent directory has to be agreed. worker-q materialises snapshots
    under its own state directory and derives `execution_cwd` relative to the
    repo root, so the absolute paths never have to match.
    """
    wanted = normalise_origin(origin_url(repo_root))
    if wanted:
        index = index_repos(node)
        # Prefer the same name when it is also the same repository, so a
        # machine holding two clones of one repo behaves predictably.
        if index.get(repo_root.name) == wanted:
            return f"{node.repo_root}\\{repo_root.name}"
        for name, origin in index.items():
            if origin == wanted:
                return f"{node.repo_root}\\{name}"
    return f"{node.repo_root}\\{repo_root.name}"


def _q(path: str) -> str:
    """Quote a Windows path for cmd.exe."""
    return f'"{path}"' if " " in path else path


#: node name -> {literal text: expanded text}
_EXPANDED: dict[str, dict[str, str]] = {}


def expand_remote(node: NodeConfig, text: str) -> str:
    """Resolve `%VAR%` against the node's environment.

    Needed because the two channels behave differently: a command sent over SSH
    runs through `cmd.exe`, which expands `%TEMP%`, but **scp talks to the sftp
    subsystem**, which does not - it took the literal string `%TEMP%\\...` as a
    directory name and failed to find it. Anything that will be handed to scp
    has to be a real path before it leaves here.
    """
    if "%" not in text:
        return text
    cache = _EXPANDED.setdefault(node.name, {})
    if text in cache:
        return cache[text]
    result = nodes.run_remote(node, f"echo {text}")
    expanded = result.out.splitlines()[0].strip() if result.ok and result.out else text
    cache[text] = expanded
    return expanded


def inspect_repo(
    node: NodeConfig, repo_root: Path, passthrough: list[str] | None = None
) -> RepoStatus:
    """One SSH call describing what the node has for this project.

    Everything is asked at once - existence, HEAD, branch, origin and every
    passthrough path - because the connection is the cost. Twenty-nine
    passthrough entries in twenty-nine calls would be 16 seconds; in one call
    it is under one.
    """
    remote = remote_repo_path(node, repo_root)
    status = RepoStatus(node=node.name, project=repo_root.name, remote_path=remote)
    entries = list(passthrough or [])

    parts = [
        f"cd /d {_q(remote)} 2>nul || (echo __NOREPO__ & exit /b 0)",
        "echo __HEAD__ & git rev-parse HEAD",
        "echo __BRANCH__ & git rev-parse --abbrev-ref HEAD",
        "echo __ORIGIN__ & git remote get-url origin",
    ]
    for entry in entries:
        win = entry.replace("/", "\\")
        parts.append(
            f'echo __PT__{entry} & if exist {_q(win)} (echo YES) else (echo NO)'
        )
    result = nodes.run_remote(node, " & ".join(parts))
    if not result.ok:
        status.error = result.error or "remote command failed"
        return status

    text = result.stdout
    if "__NOREPO__" in text:
        return status
    status.exists = True

    def section(marker: str) -> str | None:
        match = re.search(rf"__{marker}__\s*\r?\n(.*?)(?:\r?\n|$)", text)
        value = match.group(1).strip() if match else ""
        return value or None

    status.head = section("HEAD")
    status.branch = section("BRANCH")
    status.origin = section("ORIGIN")

    for entry in entries:
        pattern = rf"__PT__{re.escape(entry)}\s*\r?\n\s*(YES|NO)"
        found = re.search(pattern, text)
        status.passthrough[entry] = bool(found and found.group(1) == "YES")
    return status


def clone_repo(node: NodeConfig, repo_root: Path, *, url: str | None = None) -> RepoStatus:
    """Clone a project onto the node, from its origin - not from here.

    The node has the git credentials and its own internet connection; pulling a
    100 MB repo across the Wi-Fi link would take three times longer and
    accomplish the same thing.
    """
    url = url or origin_url(repo_root)
    if not url:
        raise StagingError(
            f"{repo_root.name} has no `origin` remote, so the node has nothing to "
            "clone from. Add one, or copy the repository across by hand."
        )
    remote = remote_repo_path(node, repo_root)
    result = nodes.run_remote(
        node,
        f"mkdir {_q(node.repo_root)} 2>nul & git clone {url} {_q(remote)}",
        timeout=max(node.timeout_seconds, 900.0),
    )
    if not result.ok:
        raise StagingError(f"clone failed on {node.name}: {result.error}")
    return inspect_repo(node, repo_root)


# --------------------------------------------------------------------------
# Shipping one snapshot
# --------------------------------------------------------------------------


def bundle_bases(node: NodeConfig, repo_root: Path) -> list[str]:
    """Commits the node already has that we can also see.

    Filtered against the local object store, because `git bundle` can only
    exclude objects the sender holds - and the node being ahead of us on some
    branch is normal, not an error.
    """
    remote = remote_repo_path(node, repo_root)
    result = nodes.run_remote(
        node,
        f"cd /d {_q(remote)} && git for-each-ref --format=%(objectname) "
        "refs/heads refs/remotes refs/tags",
    )
    if not result.ok:
        return []
    seen: list[str] = []
    for line in result.stdout.splitlines():
        oid = line.strip()
        if len(oid) == 40 and oid not in seen and have_object(repo_root, oid):
            seen.append(oid)
    return seen


def build_bundle(repo_root: Path, ref: str, bases: list[str], dest: Path) -> Path:
    """A bundle carrying `ref` and only the objects `bases` lack."""
    args = ["bundle", "create", str(dest), ref]
    for base in bases:
        args += ["--not", base]
    proc = _git(args, repo_root)
    if proc.returncode != 0 or not dest.exists():
        # A bundle with nothing to send is a real outcome, not a failure: the
        # node already has this exact commit.
        if "create empty bundle" in (proc.stderr or "").lower():
            raise StagingError("EMPTY_BUNDLE")
        raise StagingError(f"git bundle failed: {(proc.stderr or '').strip()[:200]}")
    return dest


def ship_snapshot(
    node: NodeConfig,
    repo_root: Path,
    *,
    job_id: int,
    commit: str,
    ref: str | None = None,
) -> dict[str, Any]:
    """Put one snapshot commit on the node and materialise a worktree for it.

    Returns the remote worktree path plus what the transfer cost, so a caller
    can report it and a slow link shows up as a number rather than a feeling.
    """
    ref = ref or f"{SNAPSHOT_REF_PREFIX}/{job_id}"
    remote_repo = remote_repo_path(node, repo_root)
    bases = bundle_bases(node, repo_root)

    already = False
    size = 0
    with tempfile.TemporaryDirectory(prefix="workerq-bundle-") as tmp:
        local_bundle = Path(tmp) / f"job-{job_id:06d}.bundle"
        try:
            build_bundle(repo_root, ref, bases, local_bundle)
            size = local_bundle.stat().st_size
        except StagingError as exc:
            if str(exc) != "EMPTY_BUNDLE":
                raise
            already = True

        stage_dir = expand_remote(node, REMOTE_STAGE_DIR)
        remote_bundle = f"{stage_dir}\\job-{job_id:06d}.bundle"
        if not already:
            prep = nodes.run_remote(node, f"mkdir {_q(stage_dir)} 2>nul & exit /b 0")
            if not prep.ok:
                raise StagingError(f"could not prepare staging dir: {prep.error}")
            sent = nodes.copy_to_node(node, local_bundle, remote_bundle)
            if not sent.ok:
                raise StagingError(f"bundle transfer failed: {sent.error}")

    worktree = f"{remote_repo}\\.gpuq-work\\job-{job_id:06d}"
    steps = [f"cd /d {_q(remote_repo)}"]
    if not already:
        steps.append(f"git fetch {_q(remote_bundle)} {ref}:{ref}")
    steps += [
        f"git worktree add --detach {_q(worktree)} {commit}",
    ]
    result = nodes.run_remote(
        node, " && ".join(steps), timeout=max(node.timeout_seconds, 300.0)
    )
    if not result.ok:
        raise StagingError(f"materialising the snapshot failed: {result.error}")

    if not already:
        nodes.run_remote(node, f"del {_q(remote_bundle)} 2>nul & exit /b 0")

    return {
        "node": node.name,
        "worktree": worktree,
        "commit": commit,
        "ref": ref,
        "bundle_bytes": size,
        "already_present": already,
        "bases_used": len(bases),
    }


def remove_worktree(node: NodeConfig, repo_root: Path, job_id: int) -> bool:
    """Drop a materialised snapshot. Best-effort; the node's own cleanup owns it.

    Uses `git worktree remove`, never a recursive delete: passthrough entries
    are junctions back to live data, and following one would destroy a dataset
    the primary cannot see and did not put there.
    """
    remote_repo = remote_repo_path(node, repo_root)
    worktree = f"{remote_repo}\\.gpuq-work\\job-{job_id:06d}"
    result = nodes.run_remote(
        node,
        f"cd /d {_q(remote_repo)} && git worktree remove --force {_q(worktree)}",
    )
    return result.ok


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------


def verify_passthrough(
    node: NodeConfig, repo_root: Path, entries: list[str]
) -> dict[str, bool]:
    """Which declared passthrough paths actually exist on that node.

    This is the check that decides whether a job may be placed there. It is
    deliberately about *existence* and nothing more - but see
    docs/multi-node.md 8.6: for a path the job writes to, existence is what
    makes placement dangerous rather than safe, which is why outputs must be
    declared separately.
    """
    if not entries:
        return {}
    return inspect_repo(node, repo_root, entries).passthrough
