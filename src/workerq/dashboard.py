"""`workerq top` - a live view of the queue and the machine it is protecting.

Answers, at a glance, the three questions that matter when a box is falling
over: what is running, what is holding resources (including work worker-q did not
start), and why is the next job not starting yet.
"""

from __future__ import annotations

import os
import time
from typing import Any

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from workerq import __version__, host
from workerq.core import GPUQService
from workerq.models import JobState
from workerq.resources import capacity
from workerq.util import human_duration, truncate

STATE_STYLES = {
    "RUNNING": "bold green",
    "QUEUED": "yellow",
    "PREPARING": "cyan",
    "SUCCEEDED": "green",
    "FAILED": "bold red",
    "CANCELLED": "magenta",
    "LOST": "red",
}


#: Keys the dashboard responds to, in the order the footer lists them.
KEY_HELP = "j/k scroll  PgUp/PgDn page  g gaming  r/R v/V c/C reserve  0 reset  q quit"


class KeyReader:
    """Single keypresses, without blocking the refresh loop.

    Returns None forever when stdin is not a terminal - piping `workerq top`
    into a file or running it over a pipe must keep working, just read-only.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._posix_state: Any = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __enter__(self) -> KeyReader:
        import sys

        try:
            if not sys.stdin.isatty():
                return self
        except (AttributeError, ValueError):
            return self

        if os.name == "nt":
            try:
                import msvcrt  # noqa: F401

                self._enabled = True
            except ImportError:  # pragma: no cover - not Windows
                pass
            return self

        try:  # pragma: no cover - POSIX
            import termios
            import tty

            self._posix_state = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            self._enabled = True
        except Exception:
            self._posix_state = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._posix_state is not None:  # pragma: no cover - POSIX
            import sys
            import termios

            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._posix_state)
            except Exception:
                pass

    def get(self) -> str | None:
        """The next key as a name ('up', 'pgdn') or a literal character."""
        if not self._enabled:
            return None
        if os.name == "nt":
            return self._get_windows()
        return self._get_posix()  # pragma: no cover - POSIX

    def _get_windows(self) -> str | None:
        import msvcrt

        if not msvcrt.kbhit():
            return None
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            # A two-part sequence: the second character names the special key.
            special = msvcrt.getwch() if msvcrt.kbhit() else ""
            return {
                "H": "up",
                "P": "down",
                "I": "pgup",
                "Q": "pgdn",
                "G": "home",
                "O": "end",
            }.get(special)
        if char in ("\x03", "\x1a"):
            return "q"
        return char

    def _get_posix(self) -> str | None:  # pragma: no cover - POSIX
        import select
        import sys

        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        char = sys.stdin.read(1)
        if char != "\x1b":
            return "q" if char == "\x03" else char
        # An escape sequence: read the rest if it is already buffered.
        rest = ""
        while select.select([sys.stdin], [], [], 0)[0] and len(rest) < 4:
            rest += sys.stdin.read(1)
        return {
            "[A": "up",
            "[B": "down",
            "[5~": "pgup",
            "[6~": "pgdn",
            "[H": "home",
            "[F": "end",
        }.get(rest)


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


#: Estimates are guesses of very different quality, so the source is always
#: shown. A confident-looking wrong finish time is worse than an honest blank.
_ETA_STYLES = {
    "progress": "green",
    "declared": "cyan",
    "learned": "yellow",
    "unknown": "dim",
}


def _eta_cell(job: Any, entry: dict[str, Any] | None) -> Text:
    if job.state == JobState.RUNNING.value:
        remaining = (entry or {}).get("remaining_seconds")
        if remaining is None:
            return Text("unknown", style="dim")
        source = ((entry or {}).get("eta_source") or "unknown").split()[0]
        cell = Text(f"~{human_duration(remaining)} left", style=_ETA_STYLES.get(source, ""))
        if job.progress_fraction:
            cell.append(f"  {job.progress_fraction * 100:.0f}%", style="dim")
        return cell

    starts = (entry or {}).get("starts_in_seconds")
    if starts is None:
        return Text("starts: unknown", style="dim")
    total = (entry or {}).get("remaining_seconds")
    source = ((entry or {}).get("eta_source") or "unknown").split()[0]
    text = f"starts ~{human_duration(starts)}"
    if total is not None:
        text += f", runs {human_duration(total)}"
    return Text(text, style=_ETA_STYLES.get(source, ""))


def _what_cell(job: Any) -> Text:
    """The worker's own description, falling back to the command."""
    if job.description:
        cell = Text(truncate(job.description, 52))
        if job.blocks:
            cell.append(f"  ▸ blocks {truncate(job.blocks, 24)}", style="dim")
        return cell
    return Text(truncate(job.display_command, 60), style="dim")


class Dashboard:
    def __init__(self, service: GPUQService) -> None:
        self.service = service
        self.started = time.monotonic()
        #: First active job to draw. Scrolling exists because a busy queue is
        #: routinely longer than the panel, and a silently truncated list is
        #: how you miss the job you were looking for.
        self.offset = 0
        self.visible_rows = 12
        self.active_count = 0
        #: Transient feedback for the last key pressed.
        self.message: Text | None = None
        self.interactive = False

    # -- interaction ------------------------------------------------------
    def _notify(self, text: str, style: str = "green") -> None:
        self.message = Text(text, style=style)

    def _current_reserve(self) -> Any:
        return self.service.backend.get_reserve()

    def _apply_reserve(
        self,
        *,
        ram_gb: float | None = None,
        vram_gb: float | None = None,
        cpus: int | None = None,
        label: str | None = None,
    ) -> None:
        """Set the live reserve, carrying over whatever was not changed.

        `set_reserve` fills anything unspecified from *config*, so nudging one
        dimension has to restate the other two or they would silently snap back.
        """
        from workerq.core import GPUQError

        current = self._current_reserve()
        gib = 1024.0
        try:
            self.service.set_reserve(
                ram_gb=current.ram_mib / gib if ram_gb is None else max(0.0, ram_gb),
                vram_gb=current.vram_mib / gib if vram_gb is None else max(0.0, vram_gb),
                cpus=current.cpus if cpus is None else max(0, cpus),
                label=current.label if label is None else label,
            )
        except GPUQError as exc:
            self._notify(str(exc), "red")
            return
        held = self._current_reserve()
        self._notify(
            f"held back: {held.ram_mib / gib:.0f} GiB RAM  "
            f"{held.vram_mib / gib:.0f} GiB VRAM  {held.cpus} CPU"
        )

    def toggle_gaming(self) -> None:
        """One key between 'the machine is mine' and 'the queue may have it'."""
        current = self._current_reserve()
        if current.label == "gaming":
            self.service.clear_reserve()
            self._notify("gaming mode off - headroom returned to the queue")
            return
        g = self.service.config.gaming
        self._apply_reserve(
            ram_gb=g.ram_gb, vram_gb=g.vram_gb, cpus=g.cpus, label="gaming"
        )
        if self.message is not None and self.message.style != "red":
            self._notify(
                f"gaming mode ON - holding {g.ram_gb:.0f} GiB RAM, "
                f"{g.vram_gb:.0f} GiB VRAM, {g.cpus} CPU",
                "bold green",
            )

    def handle_key(self, key: str) -> bool:
        """Act on a keypress. False means the dashboard should exit."""
        gib = 1024.0
        page = max(1, self.visible_rows - 1)
        current = None

        if key in ("q", "Q"):
            return False
        if key in ("j", "down"):
            self.offset += 1
        elif key in ("k", "up"):
            self.offset -= 1
        elif key == "pgdn":
            self.offset += page
        elif key == "pgup":
            self.offset -= page
        elif key == "home":
            self.offset = 0
        elif key == "end":
            self.offset = max(0, self.active_count - self.visible_rows)
        elif key == "g":
            self.toggle_gaming()
        elif key == "0":
            self.service.clear_reserve()
            self._notify("reserve cleared - back to the configured headroom")
        elif key in ("r", "R", "v", "V", "c", "C"):
            current = self._current_reserve()
            if key == "r":
                self._apply_reserve(ram_gb=current.ram_mib / gib - 2)
            elif key == "R":
                self._apply_reserve(ram_gb=current.ram_mib / gib + 2)
            elif key == "v":
                self._apply_reserve(vram_gb=current.vram_mib / gib - 1)
            elif key == "V":
                self._apply_reserve(vram_gb=current.vram_mib / gib + 1)
            elif key == "c":
                self._apply_reserve(cpus=current.cpus - 1)
            else:
                self._apply_reserve(cpus=current.cpus + 1)
        else:
            return True

        self.offset = max(0, min(self.offset, max(0, self.active_count - 1)))
        return True

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
            for device in gpu.devices:
                label = "VRAM" if len(gpu.devices) == 1 else f"VRAM{device.index}"
                rows.add_row(
                    label,
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
        commit_limit = self.service.config.resources.max_commit_percent
        commit_style = "bold red" if (mem.commit_percent or 0) >= commit_limit else ""
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
        forecast = self.service.forecast(jobs)
        table = Table(box=None, pad_edge=False, expand=True)
        table.add_column("ID", justify="right", style="bold", width=5)
        table.add_column("STATE", width=9)
        table.add_column("PRI", width=8)
        table.add_column("PROJECT", width=14, overflow="ellipsis")
        table.add_column("TIME", justify="right", width=8)
        table.add_column("ETA", width=17)
        table.add_column("REQ", width=11)
        table.add_column("WHAT", overflow="ellipsis")

        active = [j for j in jobs if not j.is_terminal]
        self.active_count = len(active)
        # Clamp here as well as on keypress: the queue shrinks under you as
        # jobs finish, and an offset past the end would show an empty panel.
        self.offset = max(0, min(self.offset, max(0, len(active) - 1)))
        shown = active[self.offset : self.offset + self.visible_rows]
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
                _eta_cell(job, forecast.get(job.id)),
                " ".join(request) or "-",
                _what_cell(job),
            )
            if job.state == JobState.QUEUED.value:
                reason = self.service.queue_wait_reason(job)
                if reason:
                    table.add_row(
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        Text(f"{_glyphs()[3]} {reason}", style="yellow"),
                    )

        if not shown:
            table.add_row("", Text("idle", style="dim"), "", "", "", "", "", "")

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
        # Say what is off-screen. A list that silently stops at the panel edge
        # is how you conclude a job is missing when it is merely below.
        hidden_above = self.offset
        hidden_below = max(0, len(active) - self.offset - len(shown))
        if hidden_above or hidden_below:
            header.append("   ")
            header.append(
                f"showing {self.offset + 1}-{self.offset + len(shown)} of {len(active)}",
                style="bold cyan",
            )
            if hidden_below:
                header.append(f"  ({hidden_below} below)", style="dim")
        reserve = self._current_reserve()
        if reserve.label:
            header.append(f"   [{reserve.label}]", style="bold magenta")
        return Panel(
            Group(header, table), title="queue", border_style="blue", padding=(0, 1)
        )

    def pressure_panel(self) -> Panel:
        """Who is actually holding memory - including work worker-q never started."""
        own = self.service.own_pids()
        table = Table(box=None, pad_edge=False, expand=True)
        table.add_column("PID", justify="right", width=7)
        table.add_column("RAM", justify="right", width=9)
        table.add_column("PROCESS", overflow="ellipsis")
        table.add_column("", width=8)

        for proc in host.top_processes(8):
            if proc.memory_mib < 200:
                continue
            tag = Text("worker-q", style="green") if proc.pid in own else Text("foreign", style="yellow")
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
                from workerq.report import classify_failure

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
            (f"workerq {__version__}", "dim"),
            ("   24h: ", "dim"),
            (f"{stats['succeeded']} ok", "green"),
            " · ",
            (f"{stats['failed']} failed", "red" if stats["failed"] else "dim"),
            " · ",
            (f"{stats['cancelled']} cancelled", "dim"),
            (f"   success {stats['success_rate']:.0f}%", "dim"),
            (f"   median wait {human_duration(stats['median_wait_seconds'])}", "dim"),
        )

    def keybar(self) -> Text:
        """The last action taken, or the keys available. Feedback wins."""
        if self.message is not None:
            return self.message
        if not self.interactive:
            return Text("ctrl-c to exit", style="dim")
        return Text(KEY_HELP, style="dim")

    # -- render -----------------------------------------------------------
    def render(self) -> Layout:
        jobs = self.service.list_jobs(all_jobs=False, limit=30)
        layout = Layout()
        layout.split_column(
            Layout(self.machine_panel(), size=8),
            Layout(self.queue_panel(jobs), name="queue"),
            Layout(name="lower", size=12),
            Layout(self.keybar(), size=1),
            Layout(self.footer(), size=1),
        )
        layout["lower"].split_row(
            Layout(self.pressure_panel()), Layout(self.recent_panel(jobs))
        )
        return layout


def _queue_rows(height: int) -> int:
    """How many job rows the queue panel can hold at this terminal height.

    The panel is the flexible one in the layout, so nothing else knows: the
    machine panel takes 8, the lower row 12, the keybar and footer one each,
    and the panel's own border and header take four more.
    """
    return max(3, height - 8 - 12 - 2 - 4)


#: How often keys are polled. Short enough that scrolling feels immediate,
#: long enough that an idle dashboard costs nothing.
_KEY_POLL_SECONDS = 0.05


def run_dashboard(service: GPUQService, *, interval: float = 2.0, once: bool = False) -> None:
    dashboard = Dashboard(service)
    if once:
        from rich.console import Console

        console = Console()
        dashboard.visible_rows = _queue_rows(console.size.height)
        console.print(dashboard.render())
        return

    with Live(
        dashboard.render(), refresh_per_second=8, screen=True, transient=False
    ) as live:
        with KeyReader() as keys:
            dashboard.interactive = keys.enabled
            # Fit the queue panel to the terminal: the panel is flexible, so
            # this is the only place that knows how many rows it can hold.
            height = getattr(live.console.size, "height", 40)
            dashboard.visible_rows = _queue_rows(height)
            last_refresh = 0.0
            try:
                while True:
                    key = keys.get()
                    if key is not None:
                        if not dashboard.handle_key(key):
                            break
                        live.update(dashboard.render())
                        last_refresh = time.monotonic()
                        continue
                    now = time.monotonic()
                    if now - last_refresh >= interval:
                        height = getattr(live.console.size, "height", 40)
                        dashboard.visible_rows = _queue_rows(height)
                        live.update(dashboard.render())
                        last_refresh = now
                    time.sleep(_KEY_POLL_SECONDS)
            except KeyboardInterrupt:
                pass
