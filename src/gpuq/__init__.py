"""Backward-compatibility shim: `gpuq` is now `workerq`.

Kept because three things still reference the old name and breaking any of them
would lose work:

* agents on this machine are told to run `gpuq ...` by their CLAUDE.md policy;
* a queued job's runner argv may be `python -m gpuq _run <id>`, recorded before
  the rename and executed after it;
* a dispatcher started before the rename is still running as `-m gpuq _daemon`.

Everything here forwards to `workerq`. New code should import `workerq`.
"""

from __future__ import annotations

import sys

import workerq as _workerq
from workerq import *  # noqa: F401,F403
from workerq import (  # noqa: F401
    BACKEND_NAME,
    BACKEND_VERSION,
    TASK_SPOOLER_PINNED_TAG,
    __version__,
)

# Make `import gpuq.core` (and friends) resolve to the real modules, so any
# stale import path keeps working rather than failing at job start.
for _name in (
    "backends",
    "claude_policy",
    "cleanup",
    "cli",
    "config",
    "core",
    "dashboard",
    "db",
    "doctor",
    "gpu",
    "host",
    "models",
    "report",
    "resources",
    "runner",
    "snapshot",
    "telemetry",
    "util",
    "winproc",
):
    try:
        _module = __import__(f"workerq.{_name}", fromlist=["_"])
    except ImportError:  # pragma: no cover - optional extras
        continue
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

__all__ = ["__version__", "BACKEND_NAME", "BACKEND_VERSION", "TASK_SPOOLER_PINNED_TAG"]
