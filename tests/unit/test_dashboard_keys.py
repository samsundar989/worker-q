"""Interactive controls in `workerq top`.

Scrolling exists because a busy queue is routinely longer than the panel, and
a list that silently stops at the edge is how you conclude a job is missing
when it is merely below. Gaming mode exists so reclaiming the machine is one
key rather than a remembered command line.
"""

from __future__ import annotations

import pytest

from workerq.core import GPUQService
from workerq.dashboard import (
    _LOWER_MIN,
    _QUEUE_CHROME,
    Dashboard,
    KeyReader,
    _layout_sizes,
    key_help,
)

GIB = 1024.0


@pytest.fixture
def dash(service: GPUQService) -> Dashboard:
    service.ensure_ready()
    d = Dashboard(service)
    d.visible_rows = 5
    d.active_count = 20
    return d


# -- scrolling --------------------------------------------------------------


def test_arrows_move_the_highlight_not_the_list(dash: Dashboard):
    """The row you are reading should stay put while the cursor walks down."""
    assert dash.handle_key("j") is True
    assert (dash.selected, dash.offset) == (1, 0)
    dash.handle_key("down")
    assert (dash.selected, dash.offset) == (2, 0)
    dash.handle_key("k")
    assert (dash.selected, dash.offset) == (1, 0)
    dash.handle_key("up")
    assert (dash.selected, dash.offset) == (0, 0)


def test_the_list_scrolls_once_the_highlight_reaches_the_edge(dash: Dashboard):
    for _ in range(dash.visible_rows):
        dash.handle_key("down")
    assert dash.selected == dash.visible_rows
    assert dash.offset == 1
    assert dash.selected < dash.offset + dash.visible_rows


def test_scrolling_stops_at_the_top(dash: Dashboard):
    dash.handle_key("k")
    dash.handle_key("k")
    assert dash.offset == 0
    assert dash.selected == 0


def test_scrolling_stops_at_the_bottom(dash: Dashboard):
    """The last page stays full: scrolling into blank space below the final
    job is what made paging look broken."""
    for _ in range(50):
        dash.handle_key("j")
    assert dash.selected == dash.active_count - 1
    assert dash.offset == dash.active_count - dash.visible_rows


def test_the_highlight_is_always_on_screen(dash: Dashboard):
    for key in ("end", "home", "pgdn", "pgdn", "pgup", "j", "k"):
        dash.handle_key(key)
        assert dash.offset <= dash.selected < dash.offset + dash.visible_rows


def test_a_shrinking_queue_pulls_the_highlight_back(dash: Dashboard):
    """Jobs finish under you; the cursor must not point past the end."""
    dash.handle_key("end")
    dash.active_count = 3
    dash.handle_key("j")
    assert dash.selected == 2
    assert dash.offset == 0


def test_paging_moves_most_of_a_screen(dash: Dashboard):
    dash.handle_key("pgdn")
    assert dash.offset == dash.visible_rows - 1
    dash.handle_key("pgup")
    assert dash.offset == 0


def test_home_and_end(dash: Dashboard):
    dash.handle_key("end")
    assert dash.offset == dash.active_count - dash.visible_rows
    dash.handle_key("home")
    assert dash.offset == 0


def test_an_empty_queue_cannot_be_scrolled_off(dash: Dashboard):
    dash.active_count = 0
    dash.handle_key("j")
    dash.handle_key("pgdn")
    assert dash.offset == 0


# -- fitting the panel ------------------------------------------------------


class _Job:
    """Just enough job for the row-budget arithmetic."""

    def __init__(self, job_id: int, state: str) -> None:
        self.id = job_id
        self.state = state


def _queue(dash: Dashboard, states: list[str], reason: str | None = "needs RAM"):
    dash.service.queue_wait_reason = lambda job: reason  # type: ignore[method-assign]
    return [_Job(i, state) for i, state in enumerate(states)]


def test_a_waiting_job_costs_two_lines(dash: Dashboard):
    """A queued job draws a second line saying what it is short of. Counting
    it as one row is what overflowed the panel and made the counter lie."""
    jobs = _queue(dash, ["QUEUED"] * 10)
    shown, reasons = dash._fit(jobs)
    assert len(shown) == dash.visible_rows // 2
    assert len(reasons) == len(shown)


def test_running_jobs_fill_the_panel_one_line_each(dash: Dashboard):
    jobs = _queue(dash, ["RUNNING"] * 10)
    shown, reasons = dash._fit(jobs)
    assert len(shown) == dash.visible_rows
    assert reasons == {}


def test_a_queued_job_with_nothing_to_report_costs_one_line(dash: Dashboard):
    jobs = _queue(dash, ["QUEUED"] * 10, reason=None)
    shown, _ = dash._fit(jobs)
    assert len(shown) == dash.visible_rows


def test_one_job_always_shows_even_if_it_overflows(dash: Dashboard):
    """A panel too short for a single row should still name the job."""
    dash.visible_rows = 1
    jobs = _queue(dash, ["QUEUED"])
    shown, _ = dash._fit(jobs)
    assert len(shown) == 1


def test_fitting_scrolls_until_the_highlight_is_visible(dash: Dashboard):
    """Wait-reason lines can push the selection off the bottom; the panel has
    to follow it rather than highlight a row nobody can see."""
    jobs = _queue(dash, ["QUEUED"] * 10)
    dash.active_count = len(jobs)
    dash.selected = 6
    shown, _ = dash._fit(jobs)
    assert dash.offset <= dash.selected < dash.offset + len(shown)


# -- fitting the terminal ---------------------------------------------------


@pytest.mark.parametrize("height", range(14, 61))
def test_the_panel_never_claims_rows_it_cannot_draw(height: int):
    """The counter says "showing x-y of n"; if the layout hands the queue
    fewer lines than that, it is promising jobs that are clipped away."""
    machine = 6
    lower, rows = _layout_sizes(height, machine)
    drawn = height - machine - 2 - lower - _QUEUE_CHROME
    assert rows <= max(1, drawn)
    assert rows >= 1
    assert lower >= _LOWER_MIN


def test_a_taller_terminal_shows_more_jobs():
    _, short = _layout_sizes(30, 6)
    _, tall = _layout_sizes(50, 6)
    assert tall > short


def test_more_gpus_cost_the_queue_rows_not_the_layout():
    """A second GPU adds a machine-panel line; it must come out of the queue
    rather than overflow the screen."""
    _, one_gpu = _layout_sizes(40, 6)
    _, two_gpus = _layout_sizes(40, 7)
    assert two_gpus == one_gpu - 1


# -- the keybar -------------------------------------------------------------


def test_the_keybar_lists_every_key_group():
    text = key_help()
    for fragment in ("select", "page", "gaming", "reserve", "quit"):
        assert fragment in text


# -- gaming mode ------------------------------------------------------------


def test_gaming_mode_holds_back_the_configured_headroom(dash: Dashboard):
    dash.service.config.gaming.ram_gb = 24.0
    dash.service.config.gaming.vram_gb = 22.0
    dash.service.config.gaming.cpus = 8

    dash.handle_key("g")
    held = dash.service.backend.get_reserve()
    assert held.label == "gaming"
    assert held.ram_mib == pytest.approx(24 * GIB)
    assert held.vram_mib == pytest.approx(22 * GIB)
    assert held.cpus == 8


def test_gaming_mode_toggles_back_off(dash: Dashboard):
    configured = dash.service.backend.get_reserve()
    dash.handle_key("g")
    assert dash.service.backend.get_reserve().label == "gaming"
    dash.handle_key("g")
    after = dash.service.backend.get_reserve()
    assert after.label is None
    assert after.ram_mib == pytest.approx(configured.ram_mib)


# -- nudging the reserve ----------------------------------------------------


def test_nudging_ram_leaves_the_other_dimensions_alone(dash: Dashboard):
    """set_reserve fills anything unspecified from *config*, so a nudge has to
    restate the rest or they snap back."""
    before = dash.service.backend.get_reserve()
    dash.handle_key("V")  # VRAM up 1 GiB
    dash.handle_key("R")  # RAM up 2 GiB
    after = dash.service.backend.get_reserve()
    assert after.ram_mib == pytest.approx(before.ram_mib + 2 * GIB)
    assert after.vram_mib == pytest.approx(before.vram_mib + 1 * GIB)
    assert after.cpus == before.cpus


def test_nudging_cpus_up_and_down(dash: Dashboard):
    before = dash.service.backend.get_reserve()
    dash.handle_key("C")
    assert dash.service.backend.get_reserve().cpus == before.cpus + 1
    dash.handle_key("c")
    assert dash.service.backend.get_reserve().cpus == before.cpus


def test_a_reserve_that_would_stop_everything_is_refused(dash: Dashboard):
    """Better to say no on screen than to wedge the queue with a keypress."""
    dash.service.config.gaming.ram_gb = 10_000.0
    dash.handle_key("g")
    assert dash.message is not None and dash.message.style == "red"
    assert dash.service.backend.get_reserve().label is None


def test_reset_returns_the_headroom(dash: Dashboard):
    dash.handle_key("g")
    assert dash.service.backend.get_reserve().label == "gaming"
    dash.handle_key("0")
    assert dash.service.backend.get_reserve().label is None


# -- plumbing ---------------------------------------------------------------


def test_q_exits_and_unknown_keys_do_nothing(dash: Dashboard):
    assert dash.handle_key("q") is False
    assert dash.handle_key("Z") is True
    assert dash.offset == 0


def test_keys_are_disabled_without_a_terminal():
    """Piping `workerq top` somewhere must keep working, just read-only."""
    reader = KeyReader()
    with reader:
        assert reader.enabled is False
        assert reader.get() is None
