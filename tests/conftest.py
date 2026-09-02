"""Shared fixtures.

Every test runs against a throwaway state directory and config file, so the
suite can never touch the user's real queue, database or snapshots.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from workerq.config import (
    BackendConfig,
    ClaudeConfig,
    Config,
    CoreConfig,
    GpuConfig,
    ResourcesConfig,
)
from workerq.core import GPUQService


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "gpuq test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "gpuq test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )


@pytest.fixture
def isolated_config(tmp_path: Path) -> Config:
    """A Config rooted entirely inside tmp_path."""
    state_dir = tmp_path / "state"
    config = Config(
        core=CoreConfig(
            state_dir=str(state_dir),
            max_concurrent_jobs=1,
            cancel_grace_seconds=2,
        ),
        gpu=GpuConfig(
            default_gpu_count=0,  # tests must not depend on GPU availability
            free_memory_threshold_percent=0,
        ),
        backend=BackendConfig(poll_interval_seconds=0.1),
        # Admission control is exercised deterministically in
        # tests/unit/test_resources.py. Leaving it on here would make every
        # queue test depend on how much memory this machine happens to have
        # free while the suite runs.
        resources=ResourcesConfig(enforce=False),
        claude=ClaudeConfig(install_user_policy=False),
        source_path=tmp_path / "config.toml",
        profile="pytest",
    )
    config.ensure_dirs()
    config.save()
    return config


@pytest.fixture
def service(isolated_config: Config):
    """An initialized service whose dispatcher is stopped on teardown."""
    svc = GPUQService(isolated_config)
    svc.ensure_ready()
    yield svc
    try:
        for job in svc.db.active_jobs():
            try:
                svc.cancel(job.id, force=True)
            except Exception:
                pass
        svc.backend.shutdown(timeout=10.0)
    except Exception:
        pass
    finally:
        svc.close()


@pytest.fixture
def live_service(service: GPUQService):
    """A service with a running dispatcher daemon (integration tests)."""
    service.initialize()
    assert service.backend.ensure_daemon(timeout=30.0), "dispatcher failed to start"
    return service


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A small Git repository with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main", "."], repo)
    _git(["config", "user.email", "test@example.invalid"], repo)
    _git(["config", "user.name", "gpuq test"], repo)
    _git(["config", "core.autocrlf", "false"], repo)
    (repo / "tracked.txt").write_text("original\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored/\n*.log\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-qm", "initial"], repo)
    return repo


@pytest.fixture
def git_helper():
    return _git


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.1) -> bool:
    """Poll until `predicate()` is truthy or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def waiter():
    return wait_for


@pytest.fixture
def python_exe() -> str:
    return sys.executable
