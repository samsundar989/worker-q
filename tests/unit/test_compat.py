"""The `gpuq` -> `workerq` rename must not break anything already in flight.

Three things still reference the old name: agent policy files telling workers
to run `gpuq ...`, queued jobs whose runner argv was recorded as
`python -m gpuq _run <id>`, and a dispatcher started before the rename. Each
would lose real work if the shim regressed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_legacy_top_level_import_still_works():
    import gpuq
    import workerq

    assert gpuq.__version__ == workerq.__version__
    assert gpuq.BACKEND_NAME == workerq.BACKEND_NAME


@pytest.mark.parametrize(
    "name",
    [
        "core",
        "config",
        "db",
        "cli",
        "models",
        "runner",
        "snapshot",
        "gpu",
        "host",
        "resources",
        "telemetry",
        "report",
        "util",
        "winproc",
        "doctor",
        "cleanup",
        "claude_policy",
        "backends",
    ],
)
def test_legacy_submodule_imports_resolve_to_the_real_module(name: str):
    legacy = __import__(f"gpuq.{name}", fromlist=["_"])
    real = __import__(f"workerq.{name}", fromlist=["_"])
    assert legacy is real


def test_legacy_symbols_are_the_same_objects():
    from gpuq.core import GPUQService as LegacyService
    from workerq.core import GPUQService as RealService

    assert LegacyService is RealService


def test_python_dash_m_gpuq_still_runs():
    """A job queued before the rename executes `python -m gpuq _run <id>`."""
    proc = subprocess.run(
        [sys.executable, "-m", "gpuq", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "worker" in proc.stdout.lower() or proc.stdout.strip()


def test_python_dash_m_workerq_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "workerq", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_new_jobs_are_recorded_against_the_new_module(service):
    """Fresh submissions should use `-m workerq`, not the legacy alias."""
    argv = service.runner_argv(42)
    assert argv[1:] == ["-m", "workerq", "_run", "42"]
