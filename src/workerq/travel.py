"""Deciding whether a job is safe to run on a different machine.

Phase 4b of docs/multi-node.md. Eligibility (does the data exist there?) is a
separate question, handled in `staging`. This is about the failures eligibility
cannot see, because they do not look like failures at all.

The one that matters:

> A job that writes to an absolute path under the repository will resolve that
> path on either machine. Repos live under `C:\\Users\\samsu\\Documents\\<project>`
> on both, so a dispatched job runs correctly, reports `SUCCEEDED`, and leaves
> its output on a machine nobody is looking at.

biohub's own `.gpuq.toml` instructs jobs to do exactly this, for a good reason
on one machine: an artifact written to a *relative* path lands inside the
snapshot and is deleted when the snapshot expires. The absolute path is the fix
for that, and it is what makes the job unsafe to move.

A missing dataset fails loudly and is therefore not the dangerous case. This
one succeeds wrongly, so it is refused before a job is placed rather than
detected afterwards.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Looks like a Windows absolute path (`C:\...` or `C:/...`) or a POSIX one.
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")

#: Options whose *value* is a path even when it does not look like one to a
#: regex. Only used to make the explanation better, never to widen the check.
_OUTPUT_FLAGS = ("--out", "--output", "--out-dir", "--outdir", "--save", "--save-to")


@dataclass
class TravelVerdict:
    """Whether this job may run on another machine, and why not."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    #: Absolute paths found inside the repository, which is the sharp case.
    repo_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": self.reasons, "repo_paths": self.repo_paths}


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def absolute_repo_paths(argv: list[str], repo_root: Path) -> list[str]:
    """Absolute paths in `argv` that point inside `repo_root`.

    Absolute paths *outside* the repository are left alone: a dataset at
    `D:\\data` is a read, and if it is missing on the node the job fails loudly,
    which is the safe direction. It is the ones inside the repo that resolve on
    both machines and quietly diverge.
    """
    found: list[str] = []
    for token in argv:
        text = str(token)
        # `--out=C:\path` as well as `--out C:\path`
        candidate = text.split("=", 1)[1] if "=" in text and _ABSOLUTE.match(text.split("=", 1)[1]) else text
        if not _ABSOLUTE.match(candidate):
            continue
        try:
            path = Path(os.path.expandvars(candidate)).expanduser()
        except (OSError, ValueError):
            continue
        if _within(path, repo_root) and candidate not in found:
            found.append(candidate)
    return found


def assess(
    argv: list[str],
    repo_root: Path | None,
    *,
    outputs: list[str] | None = None,
    snapshot_commit: str | None = None,
) -> TravelVerdict:
    """May this job be run on another machine?

    Deliberately conservative. Being wrong in the permissive direction produces
    a job that reports success and puts its results somewhere else; being wrong
    in the restrictive direction just keeps a job on the machine it would have
    run on anyway.
    """
    reasons: list[str] = []

    if not snapshot_commit or repo_root is None:
        reasons.append(
            "no git snapshot to ship (submitted --no-snapshot or --live-worktree)"
        )
        return TravelVerdict(False, reasons)

    repo_paths = absolute_repo_paths(list(argv), repo_root)
    if repo_paths:
        shown = ", ".join(repo_paths[:3])
        more = f" (and {len(repo_paths) - 3} more)" if len(repo_paths) > 3 else ""
        reasons.append(
            f"the command writes to an absolute path inside the repository: {shown}{more}. "
            "That path exists on the other machine too, so the job would succeed "
            "there and leave its output on a machine you are not looking at. Use a "
            "path relative to the repository and declare it in [snapshot] outputs, "
            "or pin the job with --node local"
        )

    return TravelVerdict(not reasons, reasons, repo_paths)


def describe_outputs(outputs: list[str] | None) -> str | None:
    """A one-line note for `show`, or None when a job declares no outputs."""
    if not outputs:
        return None
    return f"{len(outputs)} declared output path(s), copied back when the job ends"
