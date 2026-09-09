"""`workerq top` - a live view of the queue and the machine it is protecting.

Answers, at a glance, the three questions that matter when a box is falling
over: what is running, what is holding resources (including work worker-q did not
start), and why is the next job not starting yet.
"""

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from typing import Any

from rich import box
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
from workerq.theme import ACCENT, CHROME, MUTED, STATE_STYLES
from workerq.util import human_duration, truncate



#: Keys the dashboard responds to, in the order the keybar lists them.
KEY_GROUPS = (
    "j/k arrows select",
    "PgUp/PgDn page",
    "Home/End",
    "g gaming",
    "r/R v/V c/C reserve",
    "0 reset",
    "q quit",
)


def key_help() -> str:
    """The keybar, grouped by what each set of keys is for."""
    return f" {_glyphs()['sep']} ".join(KEY_GROUPS)


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
            # Read it unconditionally. Gating on kbhit() loses the race when
            # the console has not yet flushed the second half, and the orphan
            # then arrives on the next poll as a bare "H" or "P" - which is
            # how an arrow key becomes a keypress that does nothing.
            special = msvcrt.getwch()
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


#: Drawing characters, richest first. Console encodings on Windows still
#: range from UTF-8 to cp437, and a meter made of mojibake is worse than a
#: plainer one, so each set is tried whole before it is used.
_GLYPH_SETS: tuple[dict[str, str], ...] = (
    {
        "full": "█",
        "track": "─",
        "rule": "─",
        "arrow": "↳",
        "dot": "●",
        "sep": "·",
        "mark": "▸",
        # Eighth-width blocks, so a meter resolves pressure finer than a cell.
        "partial": " ▏▎▍▌▋▊▉",
    },
    {
        "full": "#",
        "track": "-",
        "rule": "-",
        "arrow": ">",
        "dot": "*",
        "sep": "|",
        "mark": ">",
        "partial": " ",
    },
)


@lru_cache(maxsize=1)
def _glyphs() -> dict[str, str]:
    """The richest drawing set this terminal's encoding can represent.

    Cached: it is consulted once per table row, and the answer cannot change
    while the process runs.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    for candidate in _GLYPH_SETS:
        try:
            "".join(candidate.values()).encode(encoding)
            return candidate
        except (UnicodeEncodeError, LookupError):
            continue
    return _GLYPH_SETS[-1]


def _meter_style(fraction: float) -> str:
    """Colour by pressure. Only the danger band is bold - if everything
    shouts, the one meter that matters does not stand out."""
    if fraction >= 0.90:
        return "bold red"
    if fraction >= 0.75:
        return "yellow"
    return "green"


def _bar(used: float | None, total: float | None, width: int = 24) -> Text:
    """A meter that turns colour as pressure rises."""
    g = _glyphs()
    if not total or used is None:
        return Text(g["rule"] * width + "  n/a", style=CHROME)
    fraction = max(0.0, min(1.0, used / total))
    style = _meter_style(fraction)

    # Sub-cell resolution: a partial block for the remainder, so 61% and 65%
    # are not the same picture on a 24-cell meter.
    exact = fraction * width
    filled = min(width, int(exact))
    partial = g["partial"]
    tail = partial[int((exact - filled) * len(partial))] if filled < width else ""

    bar = Text(g["full"] * filled, style=style)
    if tail.strip():
        bar.append(tail, style=style)
    bar.append(g["track"] * (width - filled - len(tail.strip())), style=CHROME)
    bar.append(f"  {fraction * 100:5.1f}%", style=style)
    return bar


def _panel(body: Any, title: str) -> Panel:
    """Every panel, drawn the same way.

    Rounded and dim rather than square and blue: the borders are scaffolding,
    and at four panels a screen they should be almost invisible.
    """
    return Panel(
        body,
        title=Text(title, style=MUTED),
        title_align="left",
        border_style=CHROME,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _table(**kwargs: Any) -> Table:
    """A borderless table with quiet headings, so rows read as a list."""
    return Table(
        box=None,
        pad_edge=False,
        expand=True,
        header_style=MUTED,
        **kwargs,
    )


def _state_cell(state: str) -> Text:
    """A status dot and a lower-case word.

    Shouting UPPERCASE at every row spends emphasis on the one column that
    never changes; the dot carries the colour and scans faster than the text.
    """
    style = STATE_STYLES.get(state, "")
    cell = Text(f"{_glyphs()['dot']} ", style=style)
    cell.append(state.lower(), style=style)
    return cell


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
            cell.append(f"  {_glyphs()['mark']} blocks {truncate(job.blocks, 24)}", style=MUTED)
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
        #: The highlighted job, as an index into the active list. Keys move
        #: this and the viewport follows, so the row you are reading stays put
        #: instead of sliding out from under you as the list scrolls.
        self.selected = 0
        self.visible_rows = 12
        self.active_count = 0
        #: Transient feedback for the last key pressed.
        self.message: Text | None = None
        self.interactive = False
        #: Lines the machine panel drew last frame. It varies with the number
        #: of GPUs, and the layout must not reserve space that nothing fills:
        #: blank rows above a full queue are rows the queue could have used.
        self.machine_rows = 4

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
            self.selected += 1
        elif key in ("k", "up"):
            self.selected -= 1
        elif key == "pgdn":
            # Move the viewport too, so the cursor keeps its place on screen
            # rather than walking to the bottom edge and sticking there.
            self.selected += page
            self.offset += page
        elif key == "pgup":
            self.selected -= page
            self.offset -= page
        elif key == "home":
            self.selected = 0
        elif key == "end":
            self.selected = self.active_count - 1
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

        self._follow_selection(self.active_count)
        return True

    def _follow_selection(self, count: int) -> None:
        """Keep the highlighted row on screen, and the screen full.

        The viewport may not start later than the last full page: scrolling
        into empty space below the final job is what made paging feel broken,
        because the panel emptied out while the counter still claimed rows.
        """
        self.selected = max(0, min(self.selected, max(0, count - 1)))
        last_top = max(0, count - self.visible_rows)
        if self.selected < self.offset:
            self.offset = self.selected
        elif self.selected >= self.offset + self.visible_rows:
            self.offset = self.selected - self.visible_rows + 1
        self.offset = max(0, min(self.offset, last_top))

    # -- panels -----------------------------------------------------------
    def machine_panel(self) -> Panel:
        gpu = self.service.gpu_info()
        mem = host.memory()
        cap = capacity(self.service.config, gpu=gpu, mem=mem)
        sep = _glyphs()["sep"]
        rows = Table.grid(padding=(0, 1))
        rows.add_column(style=MUTED, width=10, no_wrap=True)
        rows.add_column(width=34, no_wrap=True)
        # Must not wrap: the panel is sized from the row count, so a detail
        # that spilled onto a second line pushed the last row off the bottom.
        rows.add_column(overflow="ellipsis", no_wrap=True)

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
            rows.add_row("VRAM", Text("no NVIDIA GPU", style=CHROME), gpu.error or "")

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
                f"{cap.usable_ram_mib / 1024:.0f} GiB RAM {sep} {cap.usable_cpus} CPU "
                f"{sep} {cap.usable_vram_mib / 1024:.0f} GiB VRAM",
                style=MUTED,
            ),
            Text("(after reserved headroom)", style="dim"),
        )
        self._add_node_rows(rows, sep)
        self.machine_rows = rows.row_count
        return _panel(rows, "machine")

    def _add_node_rows(self, rows: Any, sep: str) -> None:
        """One line per other machine, read from what the dispatcher last saw.

        Deliberately free of I/O. This panel redraws about once a second and an
        SSH round trip costs ~540 ms on this pair, so polling here would stall
        the dashboard for half of every frame. The dispatcher already polls
        every node; it publishes what it saw into the queue meta table and this
        reads it from there.

        The consequence to be honest about: what is shown is as fresh as the
        dispatcher's last poll, and if the dispatcher is not running it is
        stale. So the age is shown whenever it is not current, rather than
        letting an old number pass as a live one.
        """
        from workerq import nodes as nodemod

        config = self.service.config
        if not getattr(config, "nodes", None):
            return
        try:
            reports = nodemod.published_reports(config, self.service.backend.store)
        except Exception:
            reports = {}

        for node in config.nodes:
            report = reports.get(node.name)
            label = node.name[:10]
            if report is None:
                rows.add_row(
                    label,
                    Text("no report yet", style=CHROME),
                    Text("the dispatcher has not polled it", style="dim"),
                )
                continue
            if not report.online:
                rows.add_row(
                    label,
                    Text("offline", style="bold red"),
                    Text(report.error or "unreachable", style="dim"),
                )
                continue

            device = report.gpu.devices[0] if report.gpu and report.gpu.devices else None
            mem = report.host_memory
            # Kept short on purpose. This grid shares fixed column widths with
            # the local rows above, and Rich shrinks every column when the
            # total overflows - so a chatty node line truncated "Commit" to
            # "Com…" on the rows that matter most.
            # The bar already carries VRAM, so this says the things it cannot.
            detail = []
            if mem is not None:
                detail.append(f"ram {_gib(mem.used_mib)}/{_gib(mem.total_mib)}G")
            detail.append(f"{len(report.running)} job")
            if not node.enabled:
                detail.append("DRAINED")
            # Only shown when it is not current: an age on every line reads as
            # noise, an age on a stale line reads as a warning.
            if report.age_seconds > max(10.0, node.poll_interval_seconds * 3):
                detail.append(f"{report.age_seconds:.0f}s ago")
            bar = _bar(device.memory_used_mib, device.memory_total_mib) if device else Text("")
            rows.add_row(
                label,
                bar,
                Text(f" {sep} ".join(detail), style="" if node.enabled else MUTED),
            )

    def _fit(self, active: list[Any]) -> tuple[list[Any], dict[int, str]]:
        """The jobs that fit in the panel, and why any of them are waiting.

        A queued job draws a second line naming what it is short of, so a
        fixed job count overflows the panel: the list spills past the border
        and the "showing x-y of n" counter promises rows you cannot see.
        Budget in lines instead, and report what was actually drawn.
        """
        memo: dict[int, str | None] = {}

        def wait_reason(job: Any) -> str | None:
            if job.state != JobState.QUEUED.value:
                return None
            if job.id not in memo:
                memo[job.id] = self.service.queue_wait_reason(job)
            return memo[job.id]

        while True:
            shown: list[Any] = []
            reasons: dict[int, str] = {}
            lines = 0
            for job in active[self.offset :]:
                reason = wait_reason(job)
                cost = 2 if reason else 1
                if shown and lines + cost > self.visible_rows:
                    break
                shown.append(job)
                if reason:
                    reasons[job.id] = reason
                lines += cost
            # Wait-reason lines can push the highlighted row past the bottom.
            # Scroll on until it is back on screen; the selection is the one
            # thing the panel must never hide.
            if self.selected < self.offset + len(shown) or self.offset >= len(active) - 1:
                return shown, reasons
            self.offset += 1

    def queue_panel(self, jobs: list[Any]) -> Panel:
        summary = self.service.status_summary()
        forecast = self.service.forecast(jobs)
        table = _table()
        table.add_column("ID", justify="right", style=MUTED, width=5)
        table.add_column("STATE", width=11, no_wrap=True)
        table.add_column("PRI", width=8, no_wrap=True)
        table.add_column("PROJECT", width=14, overflow="ellipsis", no_wrap=True)
        table.add_column("TIME", justify="right", width=8, no_wrap=True)
        table.add_column("ETA", width=17, no_wrap=True)
        table.add_column("REQ", width=11, no_wrap=True)
        # The only elastic column: it absorbs the width the others do not
        # need, and truncates rather than wrapping a job onto a second line.
        table.add_column("WHAT", overflow="ellipsis", no_wrap=True, ratio=1, min_width=16)

        active = [j for j in jobs if not j.is_terminal]
        self.active_count = len(active)
        # Clamp here as well as on keypress: the queue shrinks under you as
        # jobs finish, and an offset past the end would show an empty panel.
        self._follow_selection(len(active))
        shown, reasons = self._fit(active)
        for index, job in enumerate(shown, start=self.offset):
            row_style = "reverse" if index == self.selected else ""
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
                _state_cell(job.state),
                job.priority,
                job.project,
                age,
                _eta_cell(job, forecast.get(job.id)),
                " ".join(request) or "-",
                _what_cell(job),
                style=row_style,
            )
            reason = reasons.get(job.id)
            if reason:
                table.add_row(
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    Text(f"{_glyphs()['arrow']} {reason}", style="yellow"),
                    style=row_style,
                )


        daemon = (
            Text("running", style="green")
            if summary["daemon_running"]
            else Text("NOT RUNNING", style="bold red")
        )
        sep = f" {_glyphs()['sep']} "
        header = Text.assemble(
            ("slots ", MUTED),
            (str(summary["backend_slots"]), ""),
            (f"{sep}dispatcher ", MUTED),
            daemon,
            (sep, MUTED),
            (f"{summary['counts'].get('RUNNING', 0)} running", "green"),
            (sep, MUTED),
            (f"{summary['counts'].get('QUEUED', 0)} queued", "yellow"),
        )
        # Say what is off-screen. A list that silently stops at the panel edge
        # is how you conclude a job is missing when it is merely below.
        hidden_above = self.offset
        hidden_below = max(0, len(active) - self.offset - len(shown))
        if hidden_above or hidden_below:
            header.append(sep, style=MUTED)
            header.append(
                f"showing {self.offset + 1}-{self.offset + len(shown)} of {len(active)}",
                style=ACCENT,
            )
            if hidden_below:
                header.append(f" ({hidden_below} below)", style=MUTED)
        reserve = self._current_reserve()
        if reserve.label:
            header.append(f"{sep}[{reserve.label}]", style=f"bold {ACCENT}")
        if not shown:
            # An empty queue is the good case. Say so plainly instead of
            # drawing column headings over nothing.
            return _panel(
                Group(header, Text("\n  nothing running or queued", style=CHROME)),
                "queue",
            )
        return _panel(Group(header, table), "queue")

    def pressure_panel(self) -> Panel:
        """Who is actually holding memory - including work worker-q never started."""
        own = self.service.own_pids()
        table = _table()
        table.add_column("PID", justify="right", style=MUTED, width=7)
        table.add_column("RAM", justify="right", width=9)
        table.add_column("PROCESS", overflow="ellipsis", no_wrap=True, ratio=1, min_width=12)
        table.add_column("", width=8, no_wrap=True)

        for proc in host.top_processes(8):
            if proc.memory_mib < 200:
                continue
            # Everything yellow is nothing yellow: what matters is which of
            # these the queue actually started.
            tag = (
                Text("worker-q", style="green")
                if proc.pid in own
                else Text("foreign", style=MUTED)
            )
            style = "bold red" if proc.memory_gib >= 8 else ""
            table.add_row(
                str(proc.pid),
                Text(f"{proc.memory_gib:.1f} GiB", style=style),
                proc.name,
                tag,
            )
        return _panel(table, "memory holders")

    def recent_panel(self, jobs: list[Any]) -> Panel:
        table = _table()
        table.add_column("ID", justify="right", style=MUTED, width=5)
        table.add_column("STATE", width=11, no_wrap=True)
        table.add_column("PROJECT", width=12, overflow="ellipsis", no_wrap=True)
        table.add_column("RUNTIME", justify="right", width=7, no_wrap=True)
        table.add_column("EXIT", justify="right", width=4, no_wrap=True)
        # This panel is half the screen, so the elastic column needs a floor:
        # without one the fixed widths eat it and the reason a job failed -
        # the whole point of the panel - silently disappears.
        table.add_column("WHY", overflow="ellipsis", no_wrap=True, ratio=1, min_width=10)

        finished = [j for j in jobs if j.is_terminal][:8]
        for job in finished:
            why = ""
            if job.state in (JobState.FAILED.value, JobState.LOST.value):
                from workerq.report import classify_failure

                why = classify_failure(self.service, job).label
            table.add_row(
                str(job.id),
                _state_cell(job.state),
                job.project,
                human_duration(job.runtime_seconds),
                "-" if job.exit_code is None else str(job.exit_code),
                Text(why, style="red" if why else ""),
            )
        if not finished:
            return _panel(
                Text("\n  nothing finished yet", style=CHROME), "recently finished"
            )
        return _panel(table, "recently finished")

    def footer(self) -> Text:
        stats = self.service.throughput(hours=24)
        sep = f" {_glyphs()['sep']} "
        return Text.assemble(
            (f"workerq {__version__}", MUTED),
            (f"{sep}24h ", MUTED),
            (f"{stats['succeeded']} ok", "green"),
            (sep, CHROME),
            (f"{stats['failed']} failed", "red" if stats["failed"] else MUTED),
            (sep, CHROME),
            (f"{stats['cancelled']} cancelled", MUTED),
            (f"{sep}success {stats['success_rate']:.0f}%", MUTED),
            (f"{sep}median wait {human_duration(stats['median_wait_seconds'])}", MUTED),
        )

    def keybar(self) -> Text:
        """The last action taken, or the keys available. Feedback wins."""
        if self.message is not None:
            return self.message
        if not self.interactive:
            return Text("ctrl-c to exit", style=MUTED)
        return Text(key_help(), style=MUTED)

    # -- render -----------------------------------------------------------
    def render(self, height: int | None = None) -> Layout:
        jobs = self.service.list_jobs(all_jobs=False, limit=30)
        layout = Layout()
        # Built first: only now is machine_rows right for this frame's GPUs,
        # and the queue gets whatever height is left over.
        machine = self.machine_panel()
        machine_height = self.machine_rows + 2
        lower_height = _LOWER_MAX
        if height is not None:
            lower_height, self.visible_rows = _layout_sizes(height, machine_height)
        layout.split_column(
            Layout(machine, size=machine_height),
            Layout(self.queue_panel(jobs), name="queue"),
            Layout(name="lower", size=lower_height),
            Layout(self.keybar(), size=1),
            Layout(self.footer(), size=1),
        )
        # Not an even split: "why did it fail" needs prose, a PID table does not.
        layout["lower"].split_row(
            Layout(self.pressure_panel(), ratio=45),
            Layout(self.recent_panel(jobs), ratio=55),
        )
        return layout


#: Lines the lower row wants, and the least it can work with.
_LOWER_MAX, _LOWER_MIN = 12, 6
#: The queue panel's own overhead: two borders, the summary line, the
#: column headings.
_QUEUE_CHROME = 4


def _layout_sizes(height: int, machine: int) -> tuple[int, int]:
    """Split the terminal between the lower row and the queue.

    The queue is why the dashboard exists, so the lower row gives ground
    first. Nothing here may claim rows the panel cannot draw: an overstated
    count is what made the "showing x-y of n" counter promise jobs that were
    clipped off the bottom.
    """
    spare = max(0, height - machine - 2)  # keybar and footer
    lower = max(_LOWER_MIN, min(_LOWER_MAX, spare - _LOWER_MIN))
    return lower, max(1, spare - lower - _QUEUE_CHROME)


#: How often keys are polled. Short enough that scrolling feels immediate,
#: long enough that an idle dashboard costs nothing.
_KEY_POLL_SECONDS = 0.05


def run_dashboard(service: GPUQService, *, interval: float = 2.0, once: bool = False) -> None:
    dashboard = Dashboard(service)
    if once:
        from rich.console import Console

        console = Console()
        console.print(dashboard.render(console.size.height))
        return

    with Live(
        dashboard.render(), refresh_per_second=8, screen=True, transient=False
    ) as live:
        with KeyReader() as keys:
            dashboard.interactive = keys.enabled

            def draw() -> None:
                # Re-read the height every frame: the window can be resized
                # under a dashboard that is meant to be left running for days.
                live.update(dashboard.render(getattr(live.console.size, "height", 40)))

            last_refresh = 0.0
            try:
                while True:
                    key = keys.get()
                    if key is not None:
                        if not dashboard.handle_key(key):
                            break
                        draw()
                        last_refresh = time.monotonic()
                        continue
                    now = time.monotonic()
                    if now - last_refresh >= interval:
                        draw()
                        last_refresh = now
                    time.sleep(_KEY_POLL_SECONDS)
            except KeyboardInterrupt:
                pass
