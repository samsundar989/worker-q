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
one succeeds wrongly, so it must be handled before a job is placed rather than
detected afterwards.

**Handled, not refused.** Refusing was the first answer, and it was too strict:
`arc-whest` and `biohub` both *instruct* their jobs to write absolute paths,
for the good reason above, so refusing them meant those projects could never
use a second machine at all. But the path is already known precisely - it was
found in order to refuse it - and its repo-relative form is exactly what
`staging.collect_outputs` indexes by. So the write target is adopted as a
declared output instead, and the results are copied home.

Declaring a path costs nothing if the guess is wrong: collection only returns
files modified after the job started, so a path that turns out to be an input
is skipped. That asymmetry is what makes adoption safe where permissiveness
would not be.

One refusal remains, because it cannot be repaired this way: a job with no git
snapshot has no commit to ship.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Looks like a Windows absolute path (`C:\...` or `C:/...`) or a POSIX one.
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")

#: Options whose value is where results go. A path after one of these is a
#: write target whether or not it exists yet, which is what makes it dangerous
#: across two machines.
_OUTPUT_FLAGS = (
    "--out", "--output", "--out-dir", "--outdir", "--out-path", "--outfile",
    "--save", "--save-to", "--save-dir", "--dest", "--destination",
    "--log-dir", "--logdir", "--checkpoint-dir", "--ckpt-dir", "--report",
    "-o",
)


@dataclass
class TravelVerdict:
    """Whether this job may run on another machine, and why not."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    #: Absolute paths found inside the repository, which is the sharp case.
    repo_paths: list[str] = field(default_factory=list)
    #: Repo-relative forms of `repo_paths`, to be declared as outputs so the
    #: results are copied home. This is what turns the hazard into a handled
    #: case rather than a refusal.
    adopt_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": self.reasons,
            "repo_paths": self.repo_paths,
            "adopt_outputs": self.adopt_outputs,
        }


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def repo_relative(path_text: str, repo_root: Path) -> str | None:
    """`C:/.../arc-whest/experiments/x.json` -> `experiments/x.json`.

    Forward slashes, because that is how `.gpuq.toml` spells outputs and the
    collector converts to the node's separator itself. Returns None when the
    path cannot be expressed against the repository, which is the only case
    where a job still has to stay home.
    """
    try:
        resolved = Path(os.path.expandvars(path_text)).expanduser()
        return resolved.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def absolute_repo_paths(argv: list[str], repo_root: Path) -> list[str]:
    """Absolute paths under `repo_root` that the job appears to **write**.

    Reads are not the hazard, and treating them as one rejects every real job.
    An absolute path to the interpreter, a dataset or a checkpoint resolves to
    the node's own copy, which is exactly what should happen - each machine has
    its own venv and its own data. Only a *write* to such a path diverges: the
    job succeeds and leaves its output where nobody is looking.

    Three things are therefore not flagged:

    * **argv[0]**, which is the program being run, never an output.
    * Paths **outside** the repository. A dataset at `D:\\data` missing on the
      node fails loudly, which is the safe direction.
    * Paths that **already exist** and are not introduced by an output option -
      an input the job reads.

    That last rule is a heuristic and is deliberately biased toward reads: an
    output option is what actually marks a write, and existence only decides
    the ambiguous remainder. A bare, non-existent absolute path is treated as a
    write because that is what one usually is.
    """
    found: list[str] = []
    tokens = [str(a) for a in argv]
    for index, text in enumerate(tokens):
        # The program itself is never an output.
        if index == 0:
            continue

        after_output_flag = index > 0 and tokens[index - 1].lower() in _OUTPUT_FLAGS
        candidate = text
        if "=" in text:
            flag, _, value = text.partition("=")
            if flag.lower() in _OUTPUT_FLAGS and _ABSOLUTE.match(value):
                candidate, after_output_flag = value, True

        if not _ABSOLUTE.match(candidate):
            continue
        try:
            path = Path(os.path.expandvars(candidate)).expanduser()
        except (OSError, ValueError):
            continue
        if not _within(path, repo_root):
            continue
        if not after_output_flag and path.exists():
            continue  # an input the job reads
        if candidate not in found:
            found.append(candidate)
    return found


def assess(
    argv: list[str],
    repo_root: Path | None,
    *,
    outputs: list[str] | None = None,
    snapshot_commit: str | None = None,
) -> TravelVerdict:
    """May this job run on another machine, and what must be brought back?

    Returns `adopt_outputs`: repo-relative paths the caller should add to the
    job's declared outputs. Acting on them is not optional - a job placed
    remotely without them is the silent-success case this module exists to
    prevent.
    """
    reasons: list[str] = []

    if not snapshot_commit or repo_root is None:
        reasons.append(
            "no git snapshot to ship (submitted --no-snapshot or --live-worktree)"
        )
        return TravelVerdict(False, reasons)

    repo_paths = absolute_repo_paths(list(argv), repo_root)

    declared = [
        str(o).replace(chr(92), "/").strip("/") for o in (outputs or []) if str(o).strip()
    ]
    adopt: list[str] = []
    unreachable: list[str] = []
    for raw in repo_paths:
        rel = repo_relative(raw, repo_root)
        if rel is None:
            # Inside the repository by `_within`, yet not expressible against
            # it. Nothing could collect this, so the job stays home.
            unreachable.append(raw)
            continue
        if rel in adopt:
            continue
        # Already covered, either exactly or by a declared parent directory.
        if any(rel == d or rel.startswith(d + "/") for d in declared):
            continue
        adopt.append(rel)

    if unreachable:
        shown = ", ".join(unreachable[:3])
        reasons.append(
            f"the command writes to {shown}, which is under the repository but "
            "cannot be expressed relative to it, so results could not be copied "
            "back. Use a path relative to the repository instead"
        )
        return TravelVerdict(False, reasons, repo_paths)

    return TravelVerdict(True, [], repo_paths, adopt)


def describe_outputs(outputs: list[str] | None) -> str | None:
    """A one-line note for `show`, or None when a job declares no outputs."""
    if not outputs:
        return None
    return f"{len(outputs)} declared output path(s), copied back when the job ends"
