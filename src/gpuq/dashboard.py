"""`gpuq top` - a live view of the queue and the machine it is protecting.

Answers, at a glance, the three questions that matter when a box is falling
over: what is running, what is holding resources (including work gpuq did not
start), and why is the next job not starting yet.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from gpuq import __version__, host
from gpuq.core import GPUQService
from gpuq.models import JobState
from gpuq.resources import capacity
from gpuq.util import human_duration, truncate

STATE_STYLES = {
    "RUNNING": "bold green",
    "QUEUED": "yellow",
    "PREPARING": "cyan",
    "SUCCEEDED": "green",
    "FAILED": "bold red",
    "CANCELLED": "magenta",
    "LOST": "red",
}


def _glyphs() -> tuple[str, str, str, str]:
    """Meter characters the current output encoding can actually represent."""
    encoding = getattr(__import__("sys").stdout, "encoding", None) or "utf-8"
    for candidate in (("█", "░", "─", "↳"), ("#", "-", "-", ">")):
        try:
            "".join(candidate).encode(encoding)
            return candidate
        except (UnicodeEncodeError, LookupError):
            continue
    return ("#", "-", "-", ">")


def _bar(used: float | None, total: float | None, width: int = 24) -> Text:
    """A meter that turns colour as pressure rises."""
    full, empty, rule, _ = _glyphs()
    if not total or used is None:
        return Text(rule * width + "  n/a", style="dim")
    fraction = max(0.0, min(1.0, used / total))
    filled = int(round(fraction * width))
    if fraction >= 0.90:
        style = "bold red"
    elif fraction >= 0.75:
        style = "yellow"
    else:
        style = "green"
    bar = Text(full * filled, style=style)
    bar.append(empty * (width - filled), style="dim")
    bar.append(f"  {fraction * 100:5.1f}%", style=style)
    return bar


def _gib(mib: float | None) -> str:
    return "-" if mib is None else f"{mib / 1024:.1f}"


class Dashboard:
    def __init__(self, service: GPUQService) -> None:
        self.service = service
        self.started = time.monotonic()

    # -- panels -----------------------------------------------------------
    def machine_panel(self) -> Panel:
        gpu = self.service.gpu_info()
        mem = host.memory()
        cap = capacity(self.service.config, gpu=gpu, mem=mem)
        rows = Table.grid(padding=(0, 1))
        rows.add_column(style="dim", width=10)
        rows.add_column(width=34)
        rows.add_column()

        if gpu.available and gpu.devices:
            device = gpu.devices[0]
            rows.add_row(
                "VRAM",
                _bar(device.memory_used_mib, device.memory_total_mib),
                f"{_gib(device.memory_used_mib)} / {_gib(device.memory_total_mib)} GiB"
                + (
                    f"   util {device.utilization_percent:.0f}%"
                    if device.utilization_percent is not None
                    else ""
                ),
            )
        else:
            rows.add_row("VRAM", Text("no NVIDIA GPU", style="dim"), gpu.error or "")

        rows.add_row(
            "RAM",
            _bar(mem.used_mib, mem.total_mib),
            f"{_gib(mem.used_mib)} / {_gib(mem.total_mib)} GiB"
            f"   free {_gib(mem.available_mib)} GiB",
        )
        commit_style = "bold red" if (mem.commit_percent or 0) >= 88 else ""
        rows.add_row(
            "Commit",
            _bar(mem.commit_used_mib, mem.commit_limit_mib),
            Text(
                f"{_gib(mem.commit_used_mib)} / {_gib(mem.commit_limit_mib)} GiB",
                style=commit_style,
            ),
        )
        rows.add_row(
            "Usable",
            Text(
                f"{cap.usable_ram_mib / 1024:.0f} GiB RAM · {cap.usable_cpus} CPU · "
                f"{cap.usable_vram_mib / 1024:.0f} GiB VRAM",
                style="dim",
            ),
            Text("(after reserved headroom)", style="dim"),
        )
        return Panel(rows, title="machine", border_style="blue", padding=(0, 1))

    def queue_panel(self, jobs: list[Any]) -> Panel:
        summary = self.service.status_summary()
        table = Table(box=None, pad_edge=False, expand=True)
        table.add_column("ID", justify="right", style="bold", width=5)
        table.add_column("STATE", width=9)
        table.add_column("PRI", width=8)
        table.add_column("PROJECT", width=18, overflow="ellipsis")
        table.add_column("TIME", justify="right", width=9)
        table.add_column("REQ", width=15)
        table.add_column("COMMAND", overflow="ellipsis")

        shown = [j for j in jobs if not j.is_terminal][:12]
        for job in shown:
            request: list[str] = []
            if job.requested_ram_mib:
                request.append(f"{job.requested_ram_mib / 1024:.0f}G")
            if job.requested_cpus:
                request.append(f"{job.requested_cpus}c")
            if job.requested_gpu_count:
                request.append(f"{job.requested_gpu_count}gpu")
            age = (
                human_duration(job.runtime_seconds)
                if job.state == JobState.RUNNING.value
                else human_duration(job.wait_seconds) + "w"
            )
            table.add_row(
                str(job.id),
                Text(job.state, style=STATE_STYLES.get(job.state, "")),
                job.priority,
                job.project,
                age,
                " ".join(request) or "-",
                truncate(job.display_command, 60),
            )
            if job.state == JobState.QUEUED.value:
                reason = self.service.queue_wait_reason(job)
                if reason:
                    table.add_row(
                        "", "", "", "", "", "", Text(f"{_glyphs()[3]} {reason}", style="yellow")
                    )

        if not shown:
            table.add_row("", Text("idle", style="dim"), "", "", "", "", "")

        daemon = (
            Text("running", style="green")
            if summary["daemon_running"]
            else Text("NOT RUNNING", style="bold red")
        )
        header = Text.assemble(
            "slots ",
            (str(summary["backend_slots"]), "bold"),
            "   dispatcher ",
            daemon,
            "   ",
            (f"{summary['counts'].get('RUNNING', 0)} running", "green"),
            " · ",
            (f"{summary['counts'].get('QUEUED', 0)} queued", "yellow"),
        )
        return Panel(
            Group(header, table), title="queue", border_style="blue", padding=(0, 1)
        )

    def pressure_panel(self) -> Panel:
        """Who is actually holding memory - including work gpuq never started."""
        own = self.service.own_pids()
        table = Table(box=None, pad_edge=False, expand=True)
        table.add_column("PID", justify="right", width=7)
        table.add_column("RAM", justify="right", width=9)
        table.add_column("PROCESS", overflow="ellipsis")
        table.add_column("", width=8)

        for proc in host.top_processes(8):
            if proc.memory_mib < 200:
                continue
            tag = Text("gpuq", style="green") if proc.pid in own else Text("foreign", style="yellow")
            style = "bold red" if proc.memory_gib >= 8 else ""
            table.add_row(
                str(proc.pid),
                Text(f"{proc.memory_gib:.1f} GiB", style=style),
                proc.name,
                tag,
            )
        return Panel(
            table,
            title="top memory consumers",
            border_style="blue",
            padding=(0, 1),
        )

    def recent_panel(self, jobs: list[Any]) -> Panel:
        table = Table(box=None, pad_edge=False, expand=True)
        table.add_column("ID", justify="right", style="bold", width=5)
        table.add_column("STATE", width=10)
        table.add_column("PROJECT", width=18, overflow="ellipsis")
        table.add_column("RUNTIME", justify="right", width=8)
        table.add_column("EXIT", justify="right", width=5)
        table.add_column("WHY", overflow="ellipsis")

        finished = [j for j in jobs if j.is_terminal][:8]
        for job in finished:
            why = ""
            if job.state in (JobState.FAILED.value, JobState.LOST.value):
                from gpuq.report import classify_failure

                why = classify_failure(self.service, job).label
            table.add_row(
                str(job.id),
                Text(job.state, style=STATE_STYLES.get(job.state, "")),
                job.project,
                human_duration(job.runtime_seconds),
                "-" if job.exit_code is None else str(job.exit_code),
                Text(why, style="red" if why else ""),
            )
        if not finished:
            table.add_row("", Text("nothing finished yet", style="dim"), "", "", "", "")
        return Panel(table, title="recently finished", border_style="blue", padding=(0, 1))

    def footer(self) -> Text:
        stats = self.service.throughput(hours=24)
        return Text.assemble(
            (f"gpuq {__version__}", "dim"),
            ("   24h: ", "dim"),
            (f"{stats['succeeded']} ok", "green"),
            " · ",
            (f"{stats['failed']} failed", "red" if stats["failed"] else "dim"),
            " · ",
            (f"{stats['cancelled']} cancelled", "dim"),
            (f"   success {stats['success_rate']:.0f}%", "dim"),
            (f"   median wait {human_duration(stats['median_wait_seconds'])}", "dim"),
            ("      ctrl-c to exit", "dim"),
        )

    # -- render -----------------------------------------------------------
    def render(self) -> Layout:
        jobs = self.service.list_jobs(all_jobs=False, limit=30)
        layout = Layout()
        layout.split_column(
            Layout(self.machine_panel(), size=8),
            Layout(self.queue_panel(jobs), name="queue"),
            Layout(name="lower", size=12),
            Layout(self.footer(), size=1),
        )
        layout["lower"].split_row(
            Layout(self.pressure_panel()), Layout(self.recent_panel(jobs))
        )
        return layout


def run_dashboard(service: GPUQService, *, interval: float = 2.0, once: bool = False) -> None:
    dashboard = Dashboard(service)
    if once:
        from rich.console import Console

        Console().print(dashboard.render())
        return

    with Live(
        dashboard.render(), refresh_per_second=4, screen=True, transient=False
    ) as live:
        try:
            while True:
                time.sleep(interval)
                live.update(dashboard.render())
        except KeyboardInterrupt:
            pass
