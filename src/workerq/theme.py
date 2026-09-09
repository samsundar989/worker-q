"""One palette, shared by every surface worker-q prints.

The chrome recedes so the data can carry the colour. Saturated borders and a
different colour for every state is the look of a library demo; a tool you
stare at while a box is falling over should draw the eye only where something
is actually wrong.

`status` and `top` used to keep separate copies of this table and had drifted
apart - the same job was bold green in one and plain green in the other.
"""

from __future__ import annotations

CHROME = "grey35"
"""Borders, rules, and the empty half of a meter."""

MUTED = "grey58"
"""Column headings, labels, and secondary text."""

ACCENT = "cyan"
"""The cursor, counters - anything you can act on."""

STATE_STYLES = {
    "RUNNING": "green",
    "QUEUED": "yellow",
    "PREPARING": ACCENT,
    "SUCCEEDED": "green",
    "FAILED": "bold red",
    "CANCELLED": MUTED,  # an abandoned job is not an alarm
    "LOST": "red",
}
