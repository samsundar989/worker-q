"""Install the shared GPU policy into the user's Claude Code memory file.

Claude Code reads persistent user instructions from `~/.claude/CLAUDE.md`.
This installer is idempotent, marker-delimited and strictly additive: existing
instructions are preserved, only the gpuq block is ever rewritten, and the file
is backed up before the first modification (spec section 15.2).

This is behavioural guidance, not an enforceable boundary - Claude's own docs
distinguish CLAUDE.md from hooks/permissions - so gpuq treats it as one layer
of defence, not the guarantee.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpuq.util import atomic_write_text, ensure_dir, expand_path

START_MARKER = "<!-- gpuq-policy:start -->"
END_MARKER = "<!-- gpuq-policy:end -->"

POLICY_BODY = """## GPU / Heavy Workload Policy

This machine uses `gpuq` to coordinate expensive workloads across concurrent agents.

NEVER directly launch a command expected to substantially use NVIDIA CUDA/VRAM
or large enough host resources that overlapping runs could cause OOM.

This includes, unless clearly tiny:
- model training or fine-tuning
- substantial inference/evaluation
- CUDA-heavy test suites
- torchrun / accelerate launches
- GPU benchmarks
- GPU simulators
- hyperparameter/ablation sweeps
- commands that load large models
- other long-running high-memory experiments

Submit them instead:

    gpuq submit --project <project> --priority normal -- <command> <args...>

Use `--priority critical` only for work that is genuinely blocking urgent progress.

After submitting, continue independent coding/research/testing instead of waiting
unless the result is required for the next action.

Inspect work with:

    gpuq status
    gpuq show <job_id>
    gpuq logs <job_id>
    gpuq logs <job_id> --follow

Cancel with:

    gpuq cancel <job_id>

Do not bypass `gpuq` just because `nvidia-smi` currently looks idle.
The shared queue is the source of truth for broker-managed heavy work.

Small CPU-only commands and genuinely lightweight tests may run directly.

Notes for this machine:
- A queued job runs the source exactly as it was at submission time, so you may
  keep editing the repository immediately after submitting.
- Jobs outlive the terminal that submitted them; closing a session does not
  cancel work.
- `gpuq status --json`, `gpuq show <id> --json` and `gpuq list --json` are the
  machine-readable forms to parse."""


def policy_block() -> str:
    return f"{START_MARKER}\n{POLICY_BODY}\n{END_MARKER}"


def default_policy_path() -> Path:
    return expand_path("~/.claude/CLAUDE.md")


# --------------------------------------------------------------------------
# Block manipulation (pure functions, easy to unit test)
# --------------------------------------------------------------------------


def find_block(text: str) -> tuple[int, int] | None:
    """Byte offsets of the gpuq block, or None. Tolerates duplicates."""
    start = text.find(START_MARKER)
    if start == -1:
        return None
    end = text.find(END_MARKER, start)
    if end == -1:
        return None
    return start, end + len(END_MARKER)


def count_blocks(text: str) -> int:
    return text.count(START_MARKER)


def upsert_block(text: str, block: str) -> str:
    """Insert or replace the gpuq block, preserving everything else.

    Any accidental duplicate blocks collapse into one.
    """
    while count_blocks(text) > 1:
        span = find_block(text)
        if span is None:
            break
        last_start = text.rfind(START_MARKER)
        last_end = text.find(END_MARKER, last_start)
        if last_end == -1:
            break
        removed = text[:last_start].rstrip("\n") + "\n" + text[last_end + len(END_MARKER) :].lstrip("\n")
        if removed == text:
            break
        text = removed

    span = find_block(text)
    if span is None:
        base = text.rstrip("\n")
        return (base + "\n\n" if base else "") + block + "\n"
    start, end = span
    return text[:start] + block + text[end:]


def remove_block(text: str) -> str:
    """Strip every gpuq block, leaving unrelated instructions untouched."""
    while True:
        span = find_block(text)
        if span is None:
            return text
        start, end = span
        before = text[:start].rstrip("\n")
        after = text[end:].lstrip("\n")
        if before and after:
            text = before + "\n\n" + after
        elif before:
            text = before + "\n"
        else:
            text = after


# --------------------------------------------------------------------------
# Filesystem operations
# --------------------------------------------------------------------------


def _backup(path: Path) -> Path | None:
    """Timestamped backup taken before the first modification of a real file."""
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.gpuq-backup-{stamp}")
    if backup.exists():
        return backup
    shutil.copy2(path, backup)
    return backup


def install_policy(path: Path | None = None, *, force: bool = False) -> dict[str, Any]:
    target = expand_path(path) if path else default_policy_path()
    ensure_dir(target.parent)

    existed = target.exists()
    original = target.read_text(encoding="utf-8") if existed else ""
    block = policy_block()

    if not force and find_block(original) is not None and _block_text(original) == block:
        return {
            "path": str(target),
            "changed": False,
            "created": False,
            "backup": None,
            "message": "policy already installed and up to date",
        }

    backup = _backup(target) if existed else None
    updated = upsert_block(original, block)
    atomic_write_text(target, updated)
    return {
        "path": str(target),
        "changed": True,
        "created": not existed,
        "backup": str(backup) if backup else None,
        "message": (
            "policy installed (new file)"
            if not existed
            else "policy block updated; existing instructions preserved"
        ),
    }


def _block_text(text: str) -> str | None:
    span = find_block(text)
    return text[span[0] : span[1]] if span else None


def policy_status(path: Path | None = None) -> dict[str, Any]:
    target = expand_path(path) if path else default_policy_path()
    if not target.exists():
        return {
            "path": str(target),
            "exists": False,
            "installed": False,
            "current": False,
            "blocks": 0,
        }
    text = target.read_text(encoding="utf-8")
    block = _block_text(text)
    return {
        "path": str(target),
        "exists": True,
        "installed": block is not None,
        "current": block == policy_block(),
        "blocks": count_blocks(text),
        "size_bytes": len(text.encode("utf-8")),
    }


def remove_policy(path: Path | None = None) -> dict[str, Any]:
    target = expand_path(path) if path else default_policy_path()
    if not target.exists():
        return {"path": str(target), "changed": False, "message": "no CLAUDE.md file present"}
    original = target.read_text(encoding="utf-8")
    if find_block(original) is None:
        return {"path": str(target), "changed": False, "message": "no gpuq policy block found"}
    backup = _backup(target)
    atomic_write_text(target, remove_block(original))
    return {
        "path": str(target),
        "changed": True,
        "backup": str(backup) if backup else None,
        "message": "gpuq policy block removed; all other instructions preserved",
    }


# --------------------------------------------------------------------------
# Optional safe launcher (spec section 15.3) - defence in depth, opt-in only
# --------------------------------------------------------------------------

_LAUNCHER_CMD = """@echo off
REM gpuq safe launcher - starts Claude Code with CUDA hidden from the agent's
REM own shell, so a heavy command run directly cannot reach the GPU.
REM Trade-off: legitimate lightweight GPU probes run directly by Claude will
REM also see no device. gpuq's dispatcher restores the real device list for
REM queued jobs.
set CUDA_VISIBLE_DEVICES=
claude %*
"""

_LAUNCHER_SH = """#!/usr/bin/env bash
# gpuq safe launcher - see the .cmd variant for the trade-off this makes.
export CUDA_VISIBLE_DEVICES=""
exec claude "$@"
"""


def install_safe_launcher(directory: Path | None = None) -> dict[str, Any]:
    target_dir = expand_path(directory) if directory else expand_path("~/.local/bin")
    ensure_dir(target_dir)
    written: list[str] = []

    sh_path = target_dir / "claude-gpu-safe"
    atomic_write_text(sh_path, _LAUNCHER_SH)
    try:
        sh_path.chmod(0o755)
    except OSError:
        pass
    written.append(str(sh_path))

    import os

    if os.name == "nt":
        cmd_path = target_dir / "claude-gpu-safe.cmd"
        atomic_write_text(cmd_path, _LAUNCHER_CMD)
        written.append(str(cmd_path))

    return {
        "paths": written,
        "directory": str(target_dir),
        "message": (
            "safe launcher installed. It is NOT enabled globally: start Claude Code "
            "with 'claude-gpu-safe' when you want CUDA hidden from the agent shell."
        ),
    }


def safe_launcher_status(directory: Path | None = None) -> dict[str, Any]:
    target_dir = expand_path(directory) if directory else expand_path("~/.local/bin")
    candidates = [target_dir / "claude-gpu-safe", target_dir / "claude-gpu-safe.cmd"]
    present = [str(p) for p in candidates if p.exists()]
    return {"installed": bool(present), "paths": present, "directory": str(target_dir)}
