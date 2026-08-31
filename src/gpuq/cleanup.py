"""Snapshot / log retention (spec section 11.10).

Deletion is deliberately conservative. Nothing outside the GPUQ state directory
is ever touched, active jobs are never disturbed, and failed-job evidence
survives its own longer retention window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from gpuq.core import GPUQService
from gpuq.models import JobState
from gpuq.snapshot import (
    SnapshotError,
    is_reparse_point,
    remove_snapshot,
    unlink_reparse_points,
)
from gpuq.util import age_seconds, expand_path, is_within, parse_duration


@dataclass
class CleanupPlan:
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    temp_files: list[str] = field(default_factory=list)
    orphan_dirs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    removed_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": self.snapshots,
            "temp_files": self.temp_files,
            "orphan_dirs": self.orphan_dirs,
            "skipped": self.skipped,
            "errors": self.errors,
            "removed_bytes": self.removed_bytes,
        }

    @property
    def total(self) -> int:
        return len(self.snapshots) + len(self.temp_files) + len(self.orphan_dirs)


def _retention(service: GPUQService, state: JobState) -> timedelta:
    core = service.config.core
    days = (
        core.cleanup_failed_snapshots_after_days
        if state in (JobState.FAILED, JobState.LOST, JobState.CANCELLED)
        else core.cleanup_successful_snapshots_after_days
    )
    return timedelta(days=days)


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def build_plan(service: GPUQService, *, older_than: str | None = None) -> CleanupPlan:
    """Decide what may be removed. Pure inspection - no deletion here."""
    plan = CleanupPlan()
    service.ensure_ready()
    state_dir = service.config.state_dir
    override = parse_duration(older_than) if older_than else None

    known_snapshot_dirs: set[str] = set()

    for job in service.db.list_jobs():
        if not job.snapshot_path:
            continue
        snapshot_path = expand_path(job.snapshot_path)
        job_snapshot_root = snapshot_path.parent
        known_snapshot_dirs.add(str(job_snapshot_root))

        if not job.is_terminal:
            plan.skipped.append(f"job #{job.id}: still active ({job.state})")
            continue
        if not snapshot_path.exists():
            continue

        retention = override if override is not None else _retention(service, job.state_enum)
        age = age_seconds(job.finished_at or job.updated_at) or 0.0
        if age < retention.total_seconds():
            remaining = retention.total_seconds() - age
            plan.skipped.append(
                f"job #{job.id}: within retention ({remaining / 86400:.1f}d left)"
            )
            continue

        plan.snapshots.append(
            {
                "job_id": job.id,
                "state": job.state,
                "path": str(snapshot_path),
                "repo_root": job.repo_root,
                "ref": f"refs/gpuq/snapshots/{job.id}",
                "size_bytes": _dir_size(snapshot_path),
                "age_days": round(age / 86400, 2),
            }
        )

    # Snapshot directories with no DB row at all (crashed submissions).
    snapshots_dir = service.config.snapshots_dir
    if snapshots_dir.is_dir():
        for child in snapshots_dir.iterdir():
            if not child.is_dir():
                continue
            if str(child) in known_snapshot_dirs or str(child / "repo") in {
                s["path"] for s in plan.snapshots
            }:
                continue
            try:
                job_id = int(child.name)
            except ValueError:
                continue
            if service.db.get_job(job_id) is not None:
                continue
            if (age_seconds_of(child) or 0) < 3600:
                continue  # a submission in flight
            plan.orphan_dirs.append(str(child))

    tmp_dir = service.config.tmp_dir
    if tmp_dir.is_dir():
        for child in tmp_dir.iterdir():
            age = age_seconds_of(child) or 0
            if age > 86400:
                plan.temp_files.append(str(child))

    # Never propose anything outside the state directory.
    for bucket in (plan.temp_files, plan.orphan_dirs):
        for item in list(bucket):
            if not is_within(Path(item), state_dir):
                bucket.remove(item)
                plan.errors.append(f"refused (outside state dir): {item}")
    for entry in list(plan.snapshots):
        if not is_within(Path(entry["path"]), state_dir):
            plan.snapshots.remove(entry)
            plan.errors.append(f"refused (outside state dir): {entry['path']}")

    return plan


def age_seconds_of(path: Path) -> float | None:
    try:
        import time

        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def execute_plan(service: GPUQService, plan: CleanupPlan) -> CleanupPlan:
    """Carry out a plan. A failure on one entry never aborts the rest."""
    import shutil

    state_dir = service.config.state_dir

    for entry in plan.snapshots:
        path = Path(entry["path"])
        try:
            remove_snapshot(
                path,
                repo_root=Path(entry["repo_root"]) if entry.get("repo_root") else None,
                state_root=state_dir,
                ref=entry.get("ref"),
            )
            plan.removed_bytes += int(entry.get("size_bytes") or 0)
            parent = path.parent
            if is_within(parent, state_dir) and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except (SnapshotError, OSError) as exc:
            # Spec section 31: cleanup failure never invalidates a job result.
            plan.errors.append(f"{path}: {exc}")

    for item in plan.temp_files + plan.orphan_dirs:
        target = Path(item)
        if not is_within(target, state_dir):
            plan.errors.append(f"refused (outside state dir): {item}")
            continue
        try:
            if is_reparse_point(target):
                # Detach the link; never touch what it points at.
                (target.rmdir if target.is_dir() else target.unlink)()
            elif target.is_dir():
                # An orphan snapshot may still hold passthrough junctions.
                unlink_reparse_points(target)
                shutil.rmtree(target, ignore_errors=False)
            else:
                target.unlink()
        except OSError as exc:
            plan.errors.append(f"{target}: {exc}")

    return plan


def run_cleanup(
    service: GPUQService, *, dry_run: bool = False, older_than: str | None = None
) -> CleanupPlan:
    plan = build_plan(service, older_than=older_than)
    if dry_run:
        return plan
    return execute_plan(service, plan)


# --------------------------------------------------------------------------
# Uninstall inventory (spec section 22)
# --------------------------------------------------------------------------


def uninstall_inventory(service: GPUQService) -> dict[str, Any]:
    """What an uninstall would touch, grouped so each part is opt-in."""
    from gpuq.claude_policy import policy_status, safe_launcher_status

    config = service.config
    state_size = _dir_size(config.state_dir) if config.state_dir.exists() else 0
    return {
        "package": {
            "name": "gpuq",
            "hint": "uv tool uninstall gpuq   (or: pipx uninstall gpuq)",
        },
        "state": {
            "path": str(config.state_dir),
            "exists": config.state_dir.exists(),
            "size_bytes": state_size,
            "database": str(config.db_path),
            "logs": str(config.logs_dir),
            "snapshots": str(config.snapshots_dir),
            "note": "preserved unless --purge is given",
        },
        "config": {
            "path": str(config.source_path) if config.source_path else None,
            "exists": bool(config.source_path and config.source_path.exists()),
        },
        "claude_policy": policy_status(),
        "safe_launcher": safe_launcher_status(),
        "never_touched": [
            "your source repositories",
            "any path outside the gpuq state directory",
        ],
    }
