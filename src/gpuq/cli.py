"""The `gpuq` command line - the only user- and agent-facing executable.

Human output is compact and scannable; `--json` emits machine-readable data on
stdout with no decoration, and every error goes to stderr (spec section 28).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from gpuq import __version__
from gpuq.config import (
    ConfigError,
    Config,
    get_dotted,
    load_config,
    set_dotted_and_save,
)
from gpuq.core import GPUQError, GPUQService, JobNotFound, SubmitRequest
from gpuq.models import JobState
from gpuq.util import human_duration, parse_env_assignment, truncate

app = typer.Typer(
    name="gpuq",
    help=(
        "Agent GPU workload broker. Submit heavy GPU work to one shared queue so "
        "concurrent agents cannot OOM each other."
    ),
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

console = Console()
err_console = Console(stderr=True)

STATE_STYLES = {
    "RUNNING": "bold green",
    "QUEUED": "yellow",
    "PREPARING": "cyan",
    "SUCCEEDED": "green",
    "FAILED": "bold red",
    "CANCELLED": "magenta",
    "LOST": "red",
}

PRIORITY_STYLES = {
    "critical": "bold red",
    "high": "yellow",
    "normal": "white",
    "low": "dim",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def fail(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code)


def emit_json(payload: Any) -> None:
    """JSON goes to stdout verbatim; nothing decorative may join it."""
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def get_service() -> GPUQService:
    try:
        config = load_config()
    except ConfigError as exc:
        fail(str(exc))
        raise
    return GPUQService(config)


def _state_text(state: str) -> Text:
    return Text(state, style=STATE_STYLES.get(state, ""))


def _job_age_column(job: Any) -> str:
    if job.state == JobState.RUNNING.value:
        return human_duration(job.runtime_seconds)
    if job.state in (JobState.QUEUED.value, JobState.PREPARING.value):
        return human_duration(job.wait_seconds) + " wait"
    return human_duration(job.runtime_seconds) if job.started_at else "-"


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Rewrite the config file from defaults."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Create state directories, the database and the dispatcher. Idempotent."""
    service = get_service()
    try:
        if force and service.config.source_path:
            service.config.save()
        info = service.initialize()
    except Exception as exc:
        fail(f"initialization failed: {exc}")
        return

    # A desktop GPU always has a few GiB taken by the compositor and browsers.
    # Warn (do not silently rewrite) when the threshold would stall the queue.
    advisory = _threshold_advisory(service)

    if json_output:
        info["threshold_advisory"] = advisory
        emit_json(info)
        service.close()
        return

    console.print(f"[bold green]gpuq initialized[/bold green] (v{__version__})")
    console.print(f"  state dir : {info['state_dir']}")
    console.print(f"  config    : {info['config_path']}")
    console.print(f"  database  : schema v{info['schema_version']}")
    backend = info["backend"]
    console.print(
        f"  dispatcher: {'running' if backend['daemon_running'] else 'NOT RUNNING'}"
        f" (pid {backend['daemon_pid']}), {backend['slots']} slot(s),"
        f" GPU free threshold {backend['gpu_free_percent_threshold']}%"
    )
    if advisory:
        console.print(f"\n[yellow]note:[/yellow] {advisory}")
    console.print(
        "\nNext:\n"
        "  gpuq doctor\n"
        "  gpuq submit --project my-project -- python train.py\n"
        "  gpuq status\n"
        "  gpuq logs <id> --follow"
    )
    service.close()


def _threshold_advisory(service: GPUQService) -> str | None:
    try:
        info = service.gpu_info()
    except Exception:
        return None
    if not info.available:
        return None
    best = info.max_free_percent()
    threshold = service.config.gpu.free_memory_threshold_percent
    if best is None or best + 1e-9 >= threshold:
        return None
    suggested = max(0, int(best) - 5)
    return (
        f"the GPU is currently {best:.0f}% free but the configured threshold is "
        f"{threshold}%, so GPU jobs would wait. If that baseline usage is normal for "
        f"this desktop, run: gpuq gpu-threshold {suggested}"
    )


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def submit(
    ctx: typer.Context,
    command: Optional[list[str]] = typer.Argument(
        None, help="Command to run, after a '--' separator."
    ),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name."),
    priority: str = typer.Option(
        "normal", "--priority", help="critical | high | normal | low"
    ),
    gpus: Optional[int] = typer.Option(None, "--gpus", help="GPUs to request (default 1)."),
    ram: Optional[float] = typer.Option(
        None, "--ram", help="Peak host RAM this job needs, in GiB."
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Peak VRAM this job needs, in GiB."
    ),
    cpus: Optional[int] = typer.Option(None, "--cpus", help="CPU cores this job needs."),
    label: Optional[str] = typer.Option(None, "--label", help="Free-text label."),
    cwd: Optional[str] = typer.Option(None, "--cwd", help="Directory to submit from."),
    snapshot: bool = typer.Option(
        True, "--snapshot/--no-snapshot", help="Freeze the source at submission time."
    ),
    live_worktree: bool = typer.Option(
        False,
        "--live-worktree",
        help="Run against the live directory (it may change before the job starts).",
    ),
    shell: Optional[str] = typer.Option(
        None, "--shell", help="Run a shell command string instead of an argv vector."
    ),
    env: Optional[list[str]] = typer.Option(
        None, "--env", help="KEY=VALUE passed to the job. Repeatable."
    ),
    passthrough: Optional[list[str]] = typer.Option(
        None,
        "--passthrough",
        help="Ignored path (dataset/checkpoints) to link into the snapshot. Repeatable.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Submit a job to the shared GPU queue and return immediately."""
    argv = list(command or []) + list(ctx.args)

    if shell and argv:
        fail("use either --shell 'command string' or -- <argv...>, not both")
    if not shell and not argv:
        fail(
            "no command given.\n"
            "  Usage: gpuq submit --project NAME -- <command> [args...]\n"
            "  Note the '--' separator before your command."
        )

    env_map: dict[str, str] = {}
    for item in env or []:
        try:
            key, value = parse_env_assignment(item)
        except ValueError as exc:
            fail(str(exc))
            return
        env_map[key] = value

    service = get_service()
    request = SubmitRequest(
        command=argv,
        project=project,
        priority=priority,
        gpus=gpus,
        label=label,
        cwd=cwd,
        snapshot=snapshot,
        live_worktree=live_worktree,
        shell=shell,
        env=env_map,
        passthrough=list(passthrough or []),
        ram_gb=ram,
        vram_gb=vram,
        cpus=cpus,
    )

    try:
        result = service.submit(request)
    except GPUQError as exc:
        service.close()
        fail(str(exc))
        return
    except Exception as exc:
        service.close()
        fail(f"unexpected error during submission: {type(exc).__name__}: {exc}")
        return

    job = result.job
    if json_output:
        emit_json(result.to_dict())
        service.close()
        return

    console.print(f"[bold green]GPUQ job #{job.id} submitted[/bold green]")
    console.print(f"Project:  {job.project}")
    console.print(
        f"Priority: [{PRIORITY_STYLES.get(job.priority, 'white')}]{job.priority}[/]"
    )
    console.print(f"State:    {job.state}")
    console.print(f"Backend job: {result.backend_job_id}")
    if job.snapshot_commit:
        console.print(f"Snapshot: {job.snapshot_commit[:9]} ({job.snapshot_mode})")
    else:
        console.print(f"Snapshot: {job.snapshot_mode}")
    if result.snapshot.passthrough:
        console.print(f"Passthrough: {', '.join(result.snapshot.passthrough)}")
    if result.queue_position is not None and result.queue_position > 0:
        console.print(f"Queue position: {result.queue_position + 1}")
    footprint: list[str] = []
    if job.requested_ram_mib:
        footprint.append(f"{job.requested_ram_mib / 1024:.1f} GiB RAM")
    if job.requested_vram_mib:
        footprint.append(f"{job.requested_vram_mib / 1024:.1f} GiB VRAM")
    if job.requested_cpus:
        footprint.append(f"{job.requested_cpus} CPU")
    if job.requested_gpu_count:
        footprint.append(f"{job.requested_gpu_count} GPU")
    if footprint:
        console.print("Requests: " + ", ".join(footprint))
    console.print(f"Logs:     gpuq logs {job.id} --follow")
    service.close()


# --------------------------------------------------------------------------
# status / list
# --------------------------------------------------------------------------


def _render_status(
    service: GPUQService,
    *,
    all_jobs: bool,
    project: str | None,
    state: str | None,
    limit: int,
    json_output: bool,
    header: bool,
) -> None:
    try:
        jobs = service.list_jobs(
            all_jobs=all_jobs, project=project, state=state, limit=limit
        )
    except GPUQError as exc:
        fail(str(exc))
        return

    if json_output:
        summary = service.status_summary()
        gpu = service.gpu_info()
        emit_json(
            {
                "gpuq_version": __version__,
                "summary": summary,
                "gpu": gpu.to_dict(),
                "jobs": [j.to_dict() for j in jobs],
            }
        )
        return

    if header:
        summary = service.status_summary()
        gpu = service.gpu_info()
        if gpu.available and gpu.devices:
            for device in gpu.devices:
                used = (device.memory_used_mib or 0) / 1024
                total = (device.memory_total_mib or 0) / 1024
                free = device.free_percent
                console.print(
                    f"GPU {device.index}: {device.name}  "
                    f"{used:.1f} / {total:.1f} GiB used"
                    + (f"  ({free:.0f}% free)" if free is not None else "")
                )
        else:
            console.print(f"[dim]GPU: unavailable ({gpu.error or 'no devices'})[/dim]")
        daemon = (
            "[green]running[/green]"
            if summary["daemon_running"]
            else "[bold red]NOT RUNNING[/bold red]"
        )
        console.print(
            f"Concurrency: {summary['backend_slots']}   "
            f"GPU free threshold: {summary['gpu_free_threshold_percent']}%   "
            f"Dispatcher: {daemon}"
        )
        console.print()

    if not jobs:
        console.print("[dim]No jobs. Submit one with:[/dim]")
        console.print("  gpuq submit --project my-project -- python train.py")
        return

    table = Table(box=None, pad_edge=False, show_edge=False)
    table.add_column("ID", justify="right", style="bold")
    table.add_column("STATE")
    table.add_column("PRI")
    table.add_column("PROJECT")
    table.add_column("AGE/RUNTIME", justify="right")
    table.add_column("GPU", justify="right")
    table.add_column("BE", justify="right")
    table.add_column("COMMAND")

    width = max(30, console.width - 74)
    for job in jobs:
        command = truncate(job.display_command, width)
        wait_reason = None
        if job.state == JobState.QUEUED.value:
            wait_reason = service.queue_wait_reason(job)
        table.add_row(
            str(job.id),
            _state_text(job.state),
            Text(job.priority, style=PRIORITY_STYLES.get(job.priority, "")),
            job.project,
            _job_age_column(job),
            str(job.requested_gpu_count),
            str(job.backend_job_id if job.backend_job_id is not None else "-"),
            command,
        )
        if wait_reason:
            table.add_row("", "", "", "", "", "", "", Text(f"  ^ {wait_reason}", style="dim"))

    console.print(table)

    running = [j for j in jobs if j.state == JobState.RUNNING.value]
    if running:
        console.print(f"\n[dim]Use: gpuq logs {running[0].id} --follow[/dim]")


@app.command()
def status(
    all_jobs: bool = typer.Option(False, "--all", "-a", help="Include all finished jobs."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project."),
    state: Optional[str] = typer.Option(None, "--state", help="Filter by state."),
    limit: int = typer.Option(40, "--limit", help="Maximum jobs to show."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show the queue: what is running, what is next, what just finished."""
    service = get_service()
    _render_status(
        service,
        all_jobs=all_jobs,
        project=project,
        state=state,
        limit=limit,
        json_output=json_output,
        header=True,
    )
    service.close()


@app.command("list")
def list_jobs(
    all_jobs: bool = typer.Option(False, "--all", "-a", help="Include all finished jobs."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project."),
    state: Optional[str] = typer.Option(None, "--state", help="Filter by state."),
    limit: int = typer.Option(40, "--limit", help="Maximum jobs to show."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List jobs (same data as `status`, without the GPU header)."""
    service = get_service()
    _render_status(
        service,
        all_jobs=all_jobs,
        project=project,
        state=state,
        limit=limit,
        json_output=json_output,
        header=False,
    )
    service.close()


# --------------------------------------------------------------------------
# show
# --------------------------------------------------------------------------


@app.command()
def show(
    job_id: int = typer.Argument(..., help="GPUQ job id."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show everything known about a job, including source provenance."""
    service = get_service()
    try:
        detail = service.job_detail(job_id)
    except JobNotFound as exc:
        service.close()
        fail(str(exc))
        return

    if json_output:
        emit_json(detail)
        service.close()
        return

    state = detail["state"]
    console.print(
        f"[bold]GPUQ job #{detail['id']}[/bold]  "
        f"[{STATE_STYLES.get(state, 'white')}]{state}[/]"
    )
    rows = [
        ("Project", detail["project"]),
        ("Label", detail["label"] or "-"),
        ("Priority", detail["priority"]),
        ("Command", detail_command(detail)),
        ("Shell mode", "yes" if detail["shell_mode"] else "no"),
        ("Backend", f"{detail['backend']} job {detail['backend_job_id']}"),
        ("Backend state", detail.get("backend_state") or "-"),
        ("Queue position", _fmt_position(detail.get("queue_position"))),
        ("Waiting on", detail.get("wait_reason") or "-"),
        ("GPUs requested", str(detail["requested_gpu_count"])),
        ("RAM requested", _fmt_gib(detail.get("requested_ram_mib"))),
        ("VRAM requested", _fmt_gib(detail.get("requested_vram_mib"))),
        ("CPUs requested", str(detail.get("requested_cpus") or "-")),
        ("GPU mode", detail["gpu_mode"]),
        ("CUDA_VISIBLE_DEVICES", detail.get("cuda_visible_devices") or "-"),
        ("Repo root", detail["repo_root"] or "-"),
        ("Submitted from", detail["submitted_cwd"]),
        ("Execution cwd", detail["execution_cwd"] or "-"),
        ("Snapshot mode", detail["snapshot_mode"]),
        ("Snapshot commit", detail["snapshot_commit"] or "-"),
        ("Snapshot path", detail["snapshot_path"] or "-"),
        ("Passthrough", ", ".join(detail["snapshot_passthrough"]) or "-"),
        ("Queued at", detail["queued_at"]),
        ("Started at", detail["started_at"] or "-"),
        ("Finished at", detail["finished_at"] or "-"),
        ("Wait time", human_duration(detail.get("wait_seconds"))),
        ("Runtime", human_duration(detail.get("runtime_seconds"))),
        ("Exit code", "-" if detail["exit_code"] is None else str(detail["exit_code"])),
        ("Runner pid", str(detail["runner_pid"] or "-")),
        ("Host", detail["host"]),
        ("Submitted by", detail["submitter_agent"] or "-"),
        ("Log", detail.get("log_path") or "-"),
        ("Manifest", detail.get("manifest_path") or "-"),
        ("Error", detail["error"] or "-"),
    ]
    if detail.get("env"):
        rows.append(("Env", ", ".join(f"{k}={v}" for k, v in detail["env"].items())))

    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="dim", justify="right")
    table.add_column(overflow="fold")
    for name, value in rows:
        table.add_row(name, str(value))
    console.print(table)
    console.print(f"\n[dim]Logs: gpuq logs {detail['id']}[/dim]")
    service.close()


def detail_command(detail: dict[str, Any]) -> str:
    command = detail.get("command") or []
    if detail.get("shell_mode"):
        return command[0] if command else "-"
    import shlex

    return shlex.join(command) if command else "-"


def _fmt_gib(mib: float | None) -> str:
    return "-" if not mib else f"{mib / 1024:.1f} GiB"


def _fmt_position(position: int | None) -> str:
    return "-" if position is None else str(position + 1)


# --------------------------------------------------------------------------
# logs
# --------------------------------------------------------------------------


@app.command()
def logs(
    job_id: int = typer.Argument(..., help="GPUQ job id."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new output."),
    tail: Optional[int] = typer.Option(None, "--tail", "-n", help="Show only the last N lines."),
) -> None:
    """Show a job's output. You never need to know backend job ids."""
    service = get_service()
    try:
        job = service.get_job(job_id)
    except JobNotFound as exc:
        service.close()
        fail(str(exc))
        return

    path = service.resolve_log_path(job)
    if path is None or not path.exists():
        if job.state in (JobState.QUEUED.value, JobState.PREPARING.value):
            console.print(
                f"Job #{job.id} is {job.state.lower()}; output file has not been created yet."
            )
            if follow:
                path = _await_log(service, job_id)
                if path is None:
                    service.close()
                    return
            else:
                service.close()
                return
        else:
            service.close()
            fail(f"no log file found for job #{job_id} (state {job.state})")
            return

    try:
        if tail is not None and not follow:
            _print_tail(path, tail)
        elif follow:
            _follow(service, job_id, path, tail)
        else:
            _print_all(path)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    service.close()


def _await_log(service: GPUQService, job_id: int, timeout: float = 86400.0) -> Path | None:
    console.print("[dim]waiting for the job to start...[/dim]")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        path = service.resolve_log_path(job)
        if path and path.exists():
            return path
        if job.is_terminal:
            console.print(f"Job #{job_id} finished in state {job.state} with no output.")
            return None
        time.sleep(0.5)
    return None


def _print_all(path: Path) -> None:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            sys.stdout.write(line)
    sys.stdout.flush()


def _print_tail(path: Path, count: int) -> None:
    from collections import deque

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = deque(fh, maxlen=max(0, count))
    sys.stdout.writelines(lines)
    sys.stdout.flush()


def _follow(service: GPUQService, job_id: int, path: Path, tail: int | None) -> None:
    if tail is not None:
        _print_tail(path, tail)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            offset = fh.tell()
    else:
        _print_all(path)
        offset = path.stat().st_size

    idle_after_finish = 0
    while True:
        try:
            size = path.stat().st_size
        except OSError:
            break
        if size > offset:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                chunk = fh.read()
                offset = fh.tell()
            sys.stdout.write(chunk)
            sys.stdout.flush()
            idle_after_finish = 0
            continue

        job = service.get_job(job_id, refresh=True)
        if job.is_terminal:
            idle_after_finish += 1
            if idle_after_finish >= 3:  # let the last writes land
                break
        time.sleep(0.4)


# --------------------------------------------------------------------------
# cancel / promote
# --------------------------------------------------------------------------


@app.command()
def cancel(
    job_id: int = typer.Argument(..., help="GPUQ job id."),
    force: bool = typer.Option(
        False, "--force", help="Skip the grace period and kill the process tree immediately."
    ),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Cancel a queued or running job."""
    service = get_service()
    try:
        result = service.cancel(job_id, force=force)
    except JobNotFound as exc:
        service.close()
        fail(str(exc))
        return
    except GPUQError as exc:
        service.close()
        fail(str(exc))
        return

    if json_output:
        emit_json(result)
    else:
        console.print(result["message"])
    service.close()


@app.command()
def promote(
    job_id: int = typer.Argument(..., help="GPUQ job id."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Move a queued job to the front. Never preempts a running job."""
    service = get_service()
    try:
        result = service.promote(job_id)
    except (JobNotFound, GPUQError) as exc:
        service.close()
        fail(str(exc))
        return
    if json_output:
        emit_json(result)
    else:
        console.print(result["message"])
    service.close()


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Run health checks. Exit 0 healthy, 1 degraded, 2 broken."""
    from gpuq.doctor import doctor_report, run_doctor

    service = get_service()
    if json_output:
        report = doctor_report(service)
        emit_json(report)
        service.close()
        raise typer.Exit(report["exit_code"])

    checks, label, code = run_doctor(service)
    width = max((len(c.name) for c in checks), default=20)
    for check in checks:
        colour = {"PASS": "green", "WARN": "yellow", "FAIL": "bold red"}[check.status]
        line = f"[{colour}]{check.status:<4}[/]  {check.name.ljust(width)}"
        if check.detail:
            line += f"  [dim]{check.detail}[/dim]"
        console.print(line)
        if check.hint and check.status != "PASS":
            console.print(f"      [dim]-> {check.hint}[/dim]")

    style = {"HEALTHY": "bold green", "DEGRADED": "bold yellow", "BROKEN": "bold red"}[label]
    console.print(f"\nOverall: [{style}]{label}[/]")
    if label == "BROKEN":
        console.print("[bold red]Do not submit jobs until the FAIL items are fixed.[/bold red]")
    service.close()
    raise typer.Exit(code)


# --------------------------------------------------------------------------
# reconcile / cleanup
# --------------------------------------------------------------------------


@app.command()
def reconcile(
    dry_run: bool = typer.Option(False, "--dry-run", help="Report drift without repairing."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Repair job metadata after a crash, reboot or dispatcher restart."""
    service = get_service()
    changes = service.reconcile(mutate=not dry_run)
    if json_output:
        emit_json({"dry_run": dry_run, "changes": changes})
    elif changes:
        verb = "would change" if dry_run else "repaired"
        console.print(f"[yellow]{verb} {len(changes)} job(s):[/yellow]")
        for change in changes:
            console.print(f"  {change}")
    else:
        console.print("[green]Everything already consistent.[/green]")
    service.close()


@app.command()
def cleanup(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed."),
    older_than: Optional[str] = typer.Option(
        None, "--older-than", help="Override retention, e.g. 7d, 12h."
    ),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Remove expired snapshots and orphan temp files. Never touches active jobs."""
    from gpuq.cleanup import run_cleanup

    service = get_service()
    try:
        plan = run_cleanup(service, dry_run=dry_run, older_than=older_than)
    except ValueError as exc:
        service.close()
        fail(str(exc))
        return

    if json_output:
        emit_json({"dry_run": dry_run, **plan.to_dict()})
        service.close()
        return

    if plan.total == 0:
        console.print("[green]Nothing to clean up.[/green]")
    else:
        verb = "Would remove" if dry_run else "Removed"
        console.print(f"[bold]{verb} {plan.total} item(s)[/bold]")
        for entry in plan.snapshots:
            size = entry.get("size_bytes", 0) / (1024 * 1024)
            console.print(
                f"  snapshot job #{entry['job_id']} ({entry['state']}, "
                f"{entry['age_days']}d, {size:.1f} MiB)"
            )
        for item in plan.orphan_dirs:
            console.print(f"  orphan  {item}")
        for item in plan.temp_files:
            console.print(f"  temp    {item}")
        if not dry_run and plan.removed_bytes:
            console.print(f"\nReclaimed {plan.removed_bytes / (1024 * 1024):.1f} MiB")

    if plan.skipped:
        console.print(f"\n[dim]Kept {len(plan.skipped)} item(s):[/dim]")
        for reason in plan.skipped[:10]:
            console.print(f"  [dim]{reason}[/dim]")
    if plan.errors:
        err_console.print(f"\n[yellow]{len(plan.errors)} problem(s):[/yellow]")
        for problem in plan.errors:
            err_console.print(f"  {problem}")
    service.close()


# --------------------------------------------------------------------------
# gpu
# --------------------------------------------------------------------------


@app.command()
def gpu(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show GPU inventory and which processes hold memory."""
    service = get_service()
    info = service.gpu_info()
    if json_output:
        emit_json(info.to_dict())
        service.close()
        return

    if not info.available:
        console.print(f"[yellow]No NVIDIA GPU available:[/yellow] {info.error}")
        service.close()
        return

    own = service.own_gpu_pids()
    for device in info.devices:
        used = (device.memory_used_mib or 0) / 1024
        total = (device.memory_total_mib or 0) / 1024
        console.print(f"[bold]GPU {device.index}  {device.name}[/bold]")
        console.print(f"Memory: {used:.1f} / {total:.1f} GiB")
        if device.free_percent is not None:
            console.print(f"Free:   {device.free_percent:.1f}%")
        if device.utilization_percent is not None:
            console.print(f"Util:   {device.utilization_percent:.0f}%")
        if device.processes:
            console.print("Processes:")
            for proc in device.processes:
                memory = (
                    f"{proc.used_memory_mib / 1024:.1f} GiB"
                    if proc.used_memory_mib is not None
                    else "unknown"
                )
                tag = " [dim](gpuq)[/dim]" if proc.pid in own else ""
                console.print(
                    f"  {proc.pid} {os.path.basename(proc.process_name)} {memory}{tag}"
                )
        else:
            console.print("Processes: none")
        console.print()
    console.print(
        f"[dim]gpuq gates jobs at {service.config.gpu.free_memory_threshold_percent}% "
        "free memory.[/dim]"
    )
    service.close()


# --------------------------------------------------------------------------
# top / report / resources
# --------------------------------------------------------------------------


@app.command()
def top(
    interval: float = typer.Option(2.0, "--interval", "-i", help="Refresh seconds."),
    once: bool = typer.Option(False, "--once", help="Render a single frame and exit."),
) -> None:
    """Live dashboard: queue, machine pressure, and who is holding memory."""
    from gpuq.dashboard import run_dashboard

    service = get_service()
    try:
        run_dashboard(service, interval=max(0.25, interval), once=once)
    finally:
        service.close()


@app.command()
def report(
    hours: float = typer.Option(24.0, "--hours", "-H", help="Window to analyse."),
    limit: int = typer.Option(50, "--limit", help="Maximum failures to list."),
    pressure: bool = typer.Option(
        False, "--pressure", help="Also show what held memory in the window."
    ),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Explain recent failures: what caused them, and whose workload it was."""
    from gpuq.report import analyse, foreign_pressure_report

    service = get_service()
    data = analyse(service, hours=hours, limit=limit)
    if pressure:
        data["pressure"] = foreign_pressure_report(service, hours=hours)

    if json_output:
        emit_json(data)
        service.close()
        return

    counts = data["counts"]
    stats = data["throughput"]
    console.print(
        f"[bold]Last {hours:g}h[/bold]  "
        f"{stats['finished']} finished - "
        f"[green]{stats['succeeded']} ok[/green] - "
        f"[red]{stats['failed']} failed[/red] - "
        f"{stats['cancelled']} cancelled - "
        f"success {stats['success_rate']:.0f}%"
    )
    console.print(
        f"[dim]median wait {human_duration(stats['median_wait_seconds'])}, "
        f"median runtime {human_duration(stats['median_runtime_seconds'])}, "
        f"queue busy {stats['utilisation_percent']:.0f}% of the window[/dim]"
    )
    console.print()

    if not data["failures"]:
        console.print("[green]No failures in this window.[/green]")
        service.close()
        return

    grouped = Table(box=None, pad_edge=False)
    grouped.add_column("CAUSE", width=34)
    grouped.add_column("N", justify="right", width=4)
    for label, count in counts["by_cause"].items():
        grouped.add_row(label, str(count))
    console.print(grouped)

    who = Table(box=None, pad_edge=False)
    who.add_column("PROJECT", width=24)
    who.add_column("N", justify="right", width=4)
    who.add_column("AGENT", width=18)
    who.add_column("N", justify="right", width=4)
    projects = list(counts["by_project"].items())
    agents = list(counts["by_agent"].items())
    for index in range(max(len(projects), len(agents))):
        p_name, p_count = projects[index] if index < len(projects) else ("", "")
        a_name, a_count = agents[index] if index < len(agents) else ("", "")
        who.add_row(str(p_name), str(p_count), str(a_name), str(a_count))
    console.print()
    console.print(who)
    console.print()
    console.print("[bold]Failures[/bold]")

    for failure in data["failures"][:limit]:
        style = "red" if failure["resource_caused"] else "yellow"
        console.print(
            f"  [bold]#{failure['job_id']}[/bold] {failure['project']} "
            f"[{style}]{failure['cause_label']}[/{style}]"
            f" [dim](exit {failure['exit_code']}, "
            f"{failure['agent'] or 'unknown agent'})[/dim]"
        )
        if failure["excerpt"]:
            console.print(
                f"      [dim]{truncate(failure['excerpt'], max(40, console.width - 10))}[/dim]"
            )
        state = failure.get("resource_state") or {}
        if failure["resource_caused"] and state:
            console.print(
                f"      [dim]at the time: host "
                f"{state.get('host_free_percent') or 0:.0f}% free, commit "
                f"{state.get('commit_percent') or 0:.0f}%[/dim]"
            )
        if failure["advice"]:
            console.print(f"      [dim]-> {failure['advice']}[/dim]")

    if pressure:
        p = data["pressure"]
        console.print()
        console.print("[bold]Memory pressure in this window[/bold]")
        console.print(
            f"  [dim]worst host free {p['worst_host_free_percent'] or 0:.0f}%, "
            f"worst commit {p['worst_commit_percent'] or 0:.0f}%, "
            f"{p['samples']} samples[/dim]"
        )
        for entry in p["peak_consumers"][:8]:
            console.print(f"  {entry['peak_gib']:>6.1f} GiB  {entry['name']}")

    console.print()
    console.print(f"[bold]{data['verdict']}[/bold]")
    service.close()


@app.command()
def resources(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show capacity, headroom and the limits admission control enforces."""
    from gpuq.resources import describe_capacity

    service = get_service()
    data = describe_capacity(service.config)
    if json_output:
        emit_json(data)
        service.close()
        return

    cap = data["capacity"]
    host_info = data["host"]
    state = "[green]enforced[/green]" if data["enforced"] else "[yellow]off[/yellow]"
    console.print(f"[bold]Admission control:[/bold] {state}")
    console.print(
        f"  RAM   {cap['usable_ram_mib'] / 1024:6.1f} GiB usable of "
        f"{cap['total_ram_mib'] / 1024:.1f} GiB "
        f"(now {(host_info.get('available_mib') or 0) / 1024:.1f} GiB free, "
        f"{host_info.get('free_percent') or 0:.0f}%)"
    )
    console.print(
        f"  VRAM  {cap['usable_vram_mib'] / 1024:6.1f} GiB usable of "
        f"{cap['total_vram_mib'] / 1024:.1f} GiB"
    )
    console.print(f"  CPU   {cap['usable_cpus']:6d} usable of {cap['total_cpus']}")
    console.print(
        f"  Commit charge now {host_info.get('commit_percent') or 0:.0f}% "
        f"(hard stop at {data['limits']['max_commit_percent']}%)"
    )
    console.print()
    console.print(
        f"[dim]Reserved for the OS and your editors: "
        f"{data['reserve']['ram_gb']:.0f} GiB RAM, {data['reserve']['cpus']} CPU, "
        f"{data['reserve']['vram_gb']:.0f} GiB VRAM[/dim]"
    )
    console.print(
        f"[dim]Jobs that declare nothing are charged "
        f"{data['defaults']['ram_gb']:.0f} GiB RAM / {data['defaults']['cpus']} CPU. "
        f"Declare real numbers with --ram/--cpus/--vram.[/dim]"
    )
    service.close()


# --------------------------------------------------------------------------
# config / concurrency / threshold
# --------------------------------------------------------------------------

config_app = typer.Typer(help="Inspect and change configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Print the effective configuration and where it came from."""
    config = load_config()
    data = config.to_dict()
    if json_output:
        emit_json(
            {
                "source_path": str(config.source_path) if config.source_path else None,
                "profile": config.profile,
                "exists": bool(config.source_path and config.source_path.exists()),
                "config": data,
                "paths": {
                    "state_dir": str(config.state_dir),
                    "database": str(config.db_path),
                    "logs": str(config.logs_dir),
                    "snapshots": str(config.snapshots_dir),
                    "jobs": str(config.jobs_dir),
                },
            }
        )
        return

    exists = bool(config.source_path and config.source_path.exists())
    console.print(
        f"[dim]source: {config.source_path} "
        f"({'loaded' if exists else 'not present, using defaults'})[/dim]"
    )
    if config.profile:
        console.print(f"[dim]profile: {config.profile}[/dim]")
    console.print()
    for section, values in data.items():
        console.print(f"[bold]\\[{section}][/bold]")
        for key, value in values.items():
            console.print(f"  {key} = {json.dumps(value)}")
        console.print()
    console.print("[dim]Precedence: CLI flag > GPUQ_* env var > config file > default[/dim]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="section.key")) -> None:
    """Print one configuration value."""
    try:
        value = get_dotted(load_config(), key)
    except ConfigError as exc:
        fail(str(exc))
        return
    console.print(json.dumps(value))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="section.key"),
    value: str = typer.Argument(..., help="New value."),
) -> None:
    """Set a configuration value and persist it."""
    try:
        config = set_dotted_and_save(load_config(), key, value)
    except ConfigError as exc:
        fail(str(exc))
        return
    console.print(f"{key} = {json.dumps(get_dotted(config, key))}")
    console.print(f"[dim]saved to {config.source_path}[/dim]")

    if key.endswith("max_concurrent_jobs") or key.endswith("free_memory_threshold_percent"):
        service = GPUQService(config)
        try:
            service.backend.set_slots(config.core.max_concurrent_jobs)
            service.backend.set_gpu_free_percent(config.gpu.free_memory_threshold_percent)
            console.print("[dim]applied to the running dispatcher[/dim]")
        except Exception:
            console.print("[yellow]run 'gpuq init' to apply this to the dispatcher[/yellow]")
        service.close()


@app.command()
def concurrency(
    count: Optional[int] = typer.Argument(None, help="New concurrent job limit."),
    yes: bool = typer.Option(False, "--yes", help="Confirm raising concurrency above 1."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show or set how many heavy jobs may run at once."""
    service = get_service()
    if count is None:
        info = service.get_concurrency()
        if json_output:
            emit_json(info)
        else:
            console.print(
                f"max_concurrent_jobs = {info['config']} "
                f"(dispatcher slots: {info['backend_slots']})"
            )
        service.close()
        return

    if count > 1 and not yes:
        service.close()
        err_console.print(
            "[bold yellow]WARNING: concurrent GPU jobs can cause VRAM OOM.[/bold yellow]\n"
            "GPUQ V1 assumes exclusive heavy-job execution.\n"
            f"Re-run with --yes to set concurrency to {count}."
        )
        raise typer.Exit(1)

    try:
        info = service.set_concurrency(count)
    except GPUQError as exc:
        service.close()
        fail(str(exc))
        return

    if json_output:
        emit_json(info)
    else:
        console.print(f"max_concurrent_jobs = {info['max_concurrent_jobs']}")
        if count > 1:
            console.print(
                "[yellow]Heavy jobs may now overlap and exhaust VRAM. "
                "Set it back with: gpuq concurrency 1[/yellow]"
            )
    service.close()


@app.command("gpu-threshold")
def gpu_threshold(
    percent: Optional[int] = typer.Argument(None, help="Required free VRAM percent."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show or set the free-VRAM percentage a GPU job waits for."""
    service = get_service()
    if percent is None:
        value = service.config.gpu.free_memory_threshold_percent
        if json_output:
            emit_json(
                {
                    "free_memory_threshold_percent": value,
                    "backend": service.backend.get_gpu_free_percent(),
                }
            )
        else:
            console.print(f"gpu.free_memory_threshold_percent = {value}")
        service.close()
        return

    try:
        info = service.set_gpu_threshold(percent)
    except GPUQError as exc:
        service.close()
        fail(str(exc))
        return
    if json_output:
        emit_json(info)
    else:
        console.print(
            f"gpu.free_memory_threshold_percent = {info['free_memory_threshold_percent']}"
        )
    service.close()


# --------------------------------------------------------------------------
# claude policy
# --------------------------------------------------------------------------

claude_app = typer.Typer(
    help="Install the shared GPU policy into Claude Code's user memory.",
    no_args_is_help=True,
)
app.add_typer(claude_app, name="claude-policy")


@claude_app.command("install")
def claude_policy_install(
    path: Optional[str] = typer.Option(None, "--path", help="Override ~/.claude/CLAUDE.md."),
    force: bool = typer.Option(False, "--force", help="Rewrite even if already current."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Add or refresh the gpuq policy block. Idempotent; preserves other content."""
    from gpuq.claude_policy import install_policy

    result = install_policy(Path(path) if path else None, force=force)
    if json_output:
        emit_json(result)
        return
    console.print(f"[green]{result['message']}[/green]")
    console.print(f"  file:   {result['path']}")
    if result.get("backup"):
        console.print(f"  backup: {result['backup']}")


@claude_app.command("status")
def claude_policy_status(
    path: Optional[str] = typer.Option(None, "--path", help="Override ~/.claude/CLAUDE.md."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Report whether the policy block is installed and current."""
    from gpuq.claude_policy import policy_status

    result = policy_status(Path(path) if path else None)
    if json_output:
        emit_json(result)
        return
    if not result["exists"]:
        console.print(f"[yellow]no file at {result['path']}[/yellow]")
    elif not result["installed"]:
        console.print(f"[yellow]gpuq policy not installed[/yellow] in {result['path']}")
    elif not result["current"]:
        console.print(f"[yellow]gpuq policy is outdated[/yellow] in {result['path']}")
    else:
        console.print(f"[green]gpuq policy installed and current[/green]: {result['path']}")
    if result.get("blocks", 0) > 1:
        console.print(f"[yellow]warning: {result['blocks']} policy blocks found[/yellow]")


@claude_app.command("remove")
def claude_policy_remove(
    path: Optional[str] = typer.Option(None, "--path", help="Override ~/.claude/CLAUDE.md."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Remove only the gpuq block, leaving other instructions untouched."""
    from gpuq.claude_policy import remove_policy

    result = remove_policy(Path(path) if path else None)
    if json_output:
        emit_json(result)
        return
    console.print(result["message"])
    if result.get("backup"):
        console.print(f"  backup: {result['backup']}")


launcher_app = typer.Typer(
    help="Optional defence-in-depth launcher that hides CUDA from the agent shell.",
    no_args_is_help=True,
)
app.add_typer(launcher_app, name="claude-safe-launcher")


@launcher_app.command("install")
def safe_launcher_install(
    directory: Optional[str] = typer.Option(None, "--dir", help="Install directory."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Create `claude-gpu-safe`. Not enabled globally; you invoke it explicitly."""
    from gpuq.claude_policy import install_safe_launcher

    result = install_safe_launcher(Path(directory) if directory else None)
    if json_output:
        emit_json(result)
        return
    console.print(f"[green]{result['message']}[/green]")
    for path in result["paths"]:
        console.print(f"  {path}")
    console.print(
        "\n[dim]Trade-off: commands Claude runs directly will see no CUDA device, "
        "including legitimate lightweight probes. Queued gpuq jobs are unaffected.[/dim]"
    )


@launcher_app.command("status")
def safe_launcher_status_cmd(
    directory: Optional[str] = typer.Option(None, "--dir", help="Install directory."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Report whether the safe launcher is installed."""
    from gpuq.claude_policy import safe_launcher_status

    result = safe_launcher_status(Path(directory) if directory else None)
    if json_output:
        emit_json(result)
        return
    if result["installed"]:
        console.print("[green]installed[/green]: " + ", ".join(result["paths"]))
    else:
        console.print(f"[yellow]not installed[/yellow] in {result['directory']}")


# --------------------------------------------------------------------------
# mcp
# --------------------------------------------------------------------------

mcp_app = typer.Typer(help="Optional MCP adapter over the same core API.", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("command")
def mcp_command(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Print the stdio server command to register with Claude Code."""
    argv = [sys.executable, "-m", "gpuq", "mcp", "serve"]
    payload = {
        "command": argv[0],
        "args": argv[1:],
        "transport": "stdio",
        "register": (
            "claude mcp add gpuq --scope user -- "
            + " ".join(_quote(a) for a in argv)
        ),
    }
    if json_output:
        emit_json(payload)
        return
    console.print("[bold]stdio server command[/bold]")
    console.print("  " + " ".join(_quote(a) for a in argv))
    console.print("\n[bold]register with Claude Code[/bold]")
    console.print("  " + payload["register"])
    console.print(
        "\n[dim]Verify the flags against your installed Claude Code with "
        "'claude mcp add --help'. The CLI remains fully supported without MCP.[/dim]"
    )


def _quote(text: str) -> str:
    return f'"{text}"' if " " in text else text


@mcp_app.command("test")
def mcp_test(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Build the MCP server in-process and list its tools."""
    try:
        from gpuq.mcp.server import self_test
    except ImportError as exc:
        fail(f"MCP adapter unavailable: {exc}")
        return
    result = self_test(load_config())
    if json_output:
        emit_json(result)
        raise typer.Exit(0 if result["ok"] else 1)
    if result["ok"]:
        console.print(f"[green]MCP server OK[/green] - {len(result['tools'])} tools")
        for tool in result["tools"]:
            console.print(f"  {tool}")
    else:
        err_console.print("[bold red]MCP server failed:[/bold red]")
        # Not markup: the message contains things like 'mcp[cli]'.
        err_console.print(result["error"], markup=False, highlight=False)
        raise typer.Exit(1)


@mcp_app.command("serve")
def mcp_serve() -> None:
    """Run the MCP stdio server (used by Claude Code, not by humans)."""
    try:
        from gpuq.mcp.server import serve
    except ImportError as exc:
        err_console.print(
            f"error: MCP SDK not installed ({exc}).\n"
            "Install with: uv tool install --with 'mcp[cli]' gpuq"
        )
        raise typer.Exit(2) from exc
    serve(load_config())


# --------------------------------------------------------------------------
# uninstall
# --------------------------------------------------------------------------


@app.command()
def uninstall(
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Preview by default."),
    purge: bool = typer.Option(False, "--purge", help="Also delete logs, database, snapshots."),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show (or perform) removal of gpuq state, policy and dispatcher."""
    from gpuq.cleanup import uninstall_inventory

    service = get_service()
    inventory = uninstall_inventory(service)
    inventory["dry_run"] = dry_run
    inventory["purge"] = purge

    if not dry_run:
        actions: list[str] = []
        try:
            service.backend.shutdown()
            actions.append("dispatcher stopped")
        except Exception as exc:
            actions.append(f"dispatcher stop failed: {exc}")
        from gpuq.claude_policy import remove_policy

        actions.append(remove_policy()["message"])
        if purge:
            import shutil

            from gpuq.util import is_within

            state_dir = service.config.state_dir
            service.close()
            if state_dir.exists() and is_within(state_dir, state_dir):
                shutil.rmtree(state_dir, ignore_errors=True)
                actions.append(f"state directory removed: {state_dir}")
        inventory["actions"] = actions

    if json_output:
        emit_json(inventory)
    else:
        console.print("[bold]gpuq uninstall[/bold]" + (" [dim](dry run)[/dim]" if dry_run else ""))
        console.print(f"  package     : {inventory['package']['hint']}")
        console.print(
            f"  state       : {inventory['state']['path']} "
            f"({inventory['state']['size_bytes'] / (1024 * 1024):.1f} MiB)"
            + ("" if purge else "  [dim]preserved (use --purge to delete)[/dim]")
        )
        console.print(f"  config      : {inventory['config']['path']}")
        console.print(
            f"  claude policy: "
            f"{'installed' if inventory['claude_policy']['installed'] else 'not installed'}"
        )
        for action in inventory.get("actions", []):
            console.print(f"  [green]done[/green]: {action}")
        if dry_run:
            console.print("\n[dim]Nothing was changed. Re-run with --execute.[/dim]")
        console.print("[dim]Source repositories are never touched.[/dim]")
    try:
        service.close()
    except Exception:
        pass


# --------------------------------------------------------------------------
# hidden internals
# --------------------------------------------------------------------------


@app.command("_run", hidden=True, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def internal_run(
    ctx: typer.Context,
    job_id: int = typer.Argument(..., help="GPUQ job id."),
) -> None:
    """Internal: execute a queued job. Invoked by the dispatcher."""
    from gpuq.runner import run_job

    override = list(ctx.args) or None
    code = run_job(job_id, override, config=load_config())
    raise typer.Exit(code)


@app.command("_daemon", hidden=True)
def internal_daemon() -> None:
    """Internal: run the dispatcher loop in the foreground."""
    from gpuq.backends.dispatcher import run_daemon

    raise typer.Exit(run_daemon(load_config()))


@app.command("_dispatcher-status", hidden=True)
def internal_dispatcher_status() -> None:
    """Internal: dump backend health as JSON."""
    service = get_service()
    emit_json(service.backend.health())
    service.close()


@app.command("_stop-daemon", hidden=True)
def internal_stop_daemon() -> None:
    """Internal: ask the dispatcher to exit (used by tests and uninstall)."""
    service = get_service()
    stopped = service.backend.shutdown()
    emit_json({"stopped": stopped})
    service.close()


@app.command()
def version(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Print version information."""
    from gpuq import BACKEND_NAME, BACKEND_VERSION

    payload = {
        "gpuq": __version__,
        "backend": BACKEND_NAME,
        "backend_version": BACKEND_VERSION,
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    if json_output:
        emit_json(payload)
        return
    console.print(f"gpuq {__version__} (backend {BACKEND_NAME} {BACKEND_VERSION})")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"gpuq {__version__}")
        raise typer.Exit(0)


@app.callback()
def main_callback(
    version_flag: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """Agent GPU workload broker."""


def main() -> None:
    # Output is UTF-8: meters, box drawing and the ellipsis in truncated
    # commands are all non-ASCII, and on Windows a redirected stdout defaults
    # to the locale codec, which turns `gpuq report | tee` into a crash.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        err_console.print("interrupted")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
