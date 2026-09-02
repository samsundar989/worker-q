"""Process control: PID identity, locking, and console hygiene.

The console tests exist because of a real regression: gpuq's background
dispatcher was leaving a `conhost.exe` behind for every daemon it started.
A virtualenv `python.exe` is a launcher that gives the real interpreter a
console regardless of CreationFlags, so the fix is the interpreter, not the
flags - and that is easy to undo by accident.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from workerq.winproc import (
    ExclusiveLock,
    detached_creationflags,
    is_locked,
    pid_matches,
    process_creation_time,
    terminate_tree,
    windowless_python,
)

IS_WINDOWS = os.name == "nt"
windows_only = pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific behaviour")


# --------------------------------------------------------------------------
# PID identity (spec 29.5: never kill an unverified stale PID)
# --------------------------------------------------------------------------


def test_creation_time_for_live_and_dead_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        stamp = process_creation_time(proc.pid)
        assert stamp is not None
        assert process_creation_time(proc.pid) == stamp  # stable
        assert pid_matches(proc.pid, stamp)
        assert not pid_matches(proc.pid, stamp + 1)  # wrong identity
    finally:
        proc.kill()
        proc.wait(timeout=30)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and process_creation_time(proc.pid) is not None:
        time.sleep(0.1)
    assert process_creation_time(proc.pid) is None


def test_creation_time_of_bogus_pids():
    for pid in (0, -1, None, 999_999_999):
        assert process_creation_time(pid) is None
        assert not pid_matches(pid, 12345)


def test_pid_without_a_recorded_stamp_is_never_matched():
    """No stamp means unverified, which must never authorise a kill."""
    assert not pid_matches(os.getpid(), None)


def test_terminate_tree_refuses_a_mismatched_identity():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        stamp = process_creation_time(proc.pid)
        assert terminate_tree(proc.pid, expected_creation=(stamp or 0) + 999) is False
        assert process_creation_time(proc.pid) is not None  # still alive
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_terminate_tree_kills_descendants():
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time;"
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
            "print(c.pid, flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = int(parent.stdout.readline().strip())
    assert process_creation_time(child_pid) is not None

    assert terminate_tree(parent.pid)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and process_creation_time(child_pid) is not None:
        time.sleep(0.1)
    assert process_creation_time(child_pid) is None, "grandchild survived"
    parent.wait(timeout=30)


# --------------------------------------------------------------------------
# Single-instance lock
# --------------------------------------------------------------------------


def test_exclusive_lock_is_exclusive(tmp_path: Path):
    path = tmp_path / "d.lock"
    first = ExclusiveLock(path)
    assert first.acquire()
    try:
        assert is_locked(path)
        assert ExclusiveLock(path).acquire() is False
    finally:
        first.release()
    assert not is_locked(path)
    second = ExclusiveLock(path)
    assert second.acquire()
    second.release()


def test_lock_creates_missing_directories(tmp_path: Path):
    lock = ExclusiveLock(tmp_path / "nested" / "deeper" / "d.lock")
    assert lock.acquire()
    lock.release()


# --------------------------------------------------------------------------
# Console hygiene
# --------------------------------------------------------------------------


@windows_only
def test_windowless_python_resolves_pythonw():
    resolved = windowless_python(sys.executable)
    assert Path(resolved).name.lower().startswith("pythonw")
    assert Path(resolved).exists()
    assert Path(resolved).parent == Path(sys.executable).parent


@windows_only
def test_windowless_python_is_idempotent():
    once = windowless_python(sys.executable)
    assert windowless_python(once) == once


def test_windowless_python_falls_back_for_unknown_executables(tmp_path: Path):
    fake = str(tmp_path / "some-other-runtime.exe")
    assert windowless_python(fake) == fake


_TREE_CONHOSTS = r"""
$root = {pid}
$all = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name
$seen = @($root); $frontier = @($root)
while ($frontier.Count -gt 0) {{
  $next = @()
  foreach ($p in $frontier) {{
    foreach ($c in ($all | Where-Object ParentProcessId -eq $p)) {{
      if ($seen -notcontains $c.ProcessId) {{ $seen += $c.ProcessId; $next += $c.ProcessId }}
    }}
  }}
  $frontier = $next
}}
(($all | Where-Object {{ $seen -contains $_.ProcessId -and $_.Name -eq 'conhost.exe' }})
  | Measure-Object).Count
"""


def _tree_conhost_count(pid: int) -> int:
    """Console hosts within one process tree.

    Counting them system-wide is unusable as an assertion: other tools on a
    developer machine start and stop consoles constantly. Scoping to the tree
    we launched makes the measurement exact.
    """
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _TREE_CONHOSTS.format(pid=pid)],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()
    return int(out or 0)


@windows_only
@pytest.mark.timeout(180)
def test_background_process_leaves_no_console():
    """gpuq's own background processes must not create a console host.

    Regression guard: with a plain venv `python.exe` this leaks one
    `conhost.exe` per background process, which the user sees as stray
    terminal windows accumulating.
    """
    proc = subprocess.Popen(
        [windowless_python(sys.executable), "-c", "import time; time.sleep(8)"],
        creationflags=detached_creationflags(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    try:
        time.sleep(3.0)
        consoles = _tree_conhost_count(proc.pid)
        assert consoles == 0, (
            f"background process created {consoles} console host(s); "
            "gpuq must launch daemons with a console-less interpreter"
        )
    finally:
        terminate_tree(proc.pid)
        proc.wait(timeout=30)


@windows_only
@pytest.mark.timeout(240)
def test_dispatcher_daemon_leaves_no_console(isolated_config):
    """The same guarantee, exercised through the real backend."""
    from workerq.backends.local_dispatcher import LocalDispatcherBackend

    backend = LocalDispatcherBackend(isolated_config)
    try:
        assert backend.ensure_daemon(timeout=60.0), "dispatcher failed to start"
        pid = backend.daemon_pid()
        assert pid
        consoles = _tree_conhost_count(pid)
        assert consoles == 0, f"the dispatcher created {consoles} console host(s)"
    finally:
        backend.shutdown(timeout=30.0)
        backend.close()


@windows_only
@pytest.mark.timeout(240)
def test_dispatcher_shutdown_leaves_no_process_behind(isolated_config):
    """A stopped dispatcher must actually exit, not linger holding a console."""
    from workerq.backends.local_dispatcher import LocalDispatcherBackend

    backend = LocalDispatcherBackend(isolated_config)
    assert backend.ensure_daemon(timeout=60.0)
    pid = backend.daemon_pid()
    assert pid

    assert backend.shutdown(timeout=30.0), "dispatcher did not release its lock"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and process_creation_time(pid) is not None:
        time.sleep(0.2)
    assert process_creation_time(pid) is None, f"dispatcher pid {pid} is still running"
    assert not is_locked(backend.lock_path)
    backend.close()


_VISIBLE_IN_TREE = r"""
Add-Type @'
using System; using System.Runtime.InteropServices;
public class V {{
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc c, IntPtr l);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
  public delegate bool EnumWindowsProc(IntPtr h, IntPtr l);
  public static System.Collections.Generic.List<uint> Vis() {{
    var o = new System.Collections.Generic.List<uint>();
    EnumWindows((h,l) => {{ if (IsWindowVisible(h)) {{ uint p; GetWindowThreadProcessId(h, out p); o.Add(p); }} return true; }}, IntPtr.Zero);
    return o; }} }}
'@
$root = {pid}
$all = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name
$seen = @($root); $f = @($root)
while ($f.Count -gt 0) {{
  $n = @()
  foreach ($p in $f) {{ foreach ($c in ($all | Where-Object ParentProcessId -eq $p)) {{
    if ($seen -notcontains $c.ProcessId) {{ $seen += $c.ProcessId; $n += $c.ProcessId }} }} }}
  $f = $n
}}
$vis = [V]::Vis()
@($seen | Where-Object {{ $vis -contains $_ }}).Count
"""


def _visible_windows_in_tree(pid: int) -> int:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _VISIBLE_IN_TREE.format(pid=pid)],
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout.strip()
    return int((out.splitlines() or ["0"])[-1] or 0)


_HELPER_PROBE = r"""
import os, subprocess, sys
sys.path.insert(0, {src!r})
from workerq.winproc import no_window_kwargs
open(sys.argv[1], "w").write(str(os.getpid()))
subprocess.run(["ping", "-n", "9", "127.0.0.1"], capture_output=True, **no_window_kwargs())
"""


@windows_only
@pytest.mark.timeout(240)
def test_helper_subprocess_opens_no_visible_window(tmp_path: Path):
    """Helper commands must never flash a console window.

    Regression guard for the real bug this fixes: gpuq's dispatcher has no
    console of its own, and on Windows a console-less parent that launches a
    console program (`nvidia-smi`, `git`, `taskkill`) makes the system open a
    brand-new *visible* console for it. Since the dispatcher polls nvidia-smi,
    that surfaced as a window popping up every few seconds.
    """
    src = str(Path(__file__).resolve().parents[2] / "src")
    probe = tmp_path / "probe.py"
    probe.write_text(_HELPER_PROBE.format(src=src), encoding="utf-8")
    pid_file = tmp_path / "pid.txt"

    proc = subprocess.Popen(
        [windowless_python(sys.executable), str(probe), str(pid_file)],
        creationflags=detached_creationflags(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.2)
        assert pid_file.exists(), "probe never started"
        real_pid = int(pid_file.read_text(encoding="utf-8").strip())

        time.sleep(1.5)  # the helper is mid-flight
        visible = _visible_windows_in_tree(real_pid)
        assert visible == 0, (
            f"{visible} visible console window(s) appeared; every helper "
            "subprocess must pass winproc.no_window_kwargs()"
        )
    finally:
        terminate_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass


@windows_only
def test_no_window_kwargs_sets_the_flag():
    from workerq.winproc import CREATE_NO_WINDOW, no_window_kwargs

    assert no_window_kwargs()["creationflags"] & CREATE_NO_WINDOW


def test_every_helper_subprocess_uses_no_window():
    """Static guard: no new bare helper subprocess slips in unnoticed.

    User job commands are exempt - they are launched with `child_creationflags`
    and their own console handling.
    """
    import re

    src = Path(__file__).resolve().parents[2] / "src" / "gpuq"
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"subprocess\.(run|Popen)\(", text):
            # Flags may be inline in the call, or assembled into a kwargs dict
            # just above it, so inspect the surrounding region in both
            # directions rather than only what follows.
            region = text[max(0, match.start() - 1800) : match.start() + 900]
            if "no_window_kwargs()" in region or "creationflags" in region:
                continue
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}")
    assert not offenders, (
        "these subprocess calls set no creation flags and will pop a console "
        f"window when run from the console-less dispatcher: {offenders}"
    )
