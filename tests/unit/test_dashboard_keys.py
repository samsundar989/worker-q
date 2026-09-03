"""Interactive controls in `workerq top`.

Scrolling exists because a busy queue is routinely longer than the panel, and
a list that silently stops at the edge is how you conclude a job is missing
when it is merely below. Gaming mode exists so reclaiming the machine is one
key rather than a remembered command line.
"""

from __future__ import annotations

import pytest

from workerq.core import GPUQService
from workerq.dashboard import Dashboard, KeyReader

GIB = 1024.0


@pytest.fixture
def dash(service: GPUQService) -> Dashboard:
    service.ensure_ready()
    d = Dashboard(service)
    d.visible_rows = 5
    d.active_count = 20
    return d


# -- scrolling --------------------------------------------------------------


def test_scrolling_moves_one_row_at_a_time(dash: Dashboard):
    assert dash.handle_key("j") is True
    assert dash.offset == 1
    dash.handle_key("down")
    assert dash.offset == 2
    dash.handle_key("k")
    assert dash.offset == 1
    dash.handle_key("up")
    assert dash.offset == 0


def test_scrolling_stops_at_the_top(dash: Dashboard):
    dash.handle_key("k")
    dash.handle_key("k")
    assert dash.offset == 0


def test_scrolling_stops_at_the_bottom(dash: Dashboard):
    for _ in range(50):
        dash.handle_key("j")
    assert dash.offset == dash.active_count - 1


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
