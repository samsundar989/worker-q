"""Windows process-tree control.

The dispatcher needs three guarantees that a bare `Popen` cannot give:

* a spawned job survives the shell that submitted it (detached, no console);
* cancelling a job kills the *whole* descendant tree, not just the direct child;
* a PID is never killed without proving it is still the process we started
  (spec section 29.5), which we do by comparing process creation time.

Job Objects provide the first two. On non-Windows platforms the same API is
implemented with POSIX process groups so the module stays importable and the
unit tests run anywhere.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

IS_WINDOWS = os.name == "nt"

# subprocess creation flags used for detached, console-less children.
if IS_WINDOWS:  # pragma: no branch
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_NO_WINDOW = 0x08000000
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
else:  # pragma: no cover - POSIX fallback
    DETACHED_PROCESS = 0
    CREATE_NEW_PROCESS_GROUP = 0
    CREATE_NO_WINDOW = 0
    CREATE_BREAKAWAY_FROM_JOB = 0
    BELOW_NORMAL_PRIORITY_CLASS = 0


def detached_creationflags() -> int:
    """Flags for a background process with no console window."""
    if not IS_WINDOWS:  # pragma: no cover
        return 0
    return DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW


def child_creationflags(*, background: bool = False) -> int:
    """Flags for a job child: its own process group so it can be signalled.

    `background` additionally runs it below normal priority. With several jobs
    sharing the CPUs, that is what keeps the desktop responsive: Windows hands
    the interactive session the CPU it needs regardless of how much work is
    queued behind it. Children inherit the class, so setting it on the runner
    covers the whole job tree.

    It changes scheduling priority only. It does not cap CPU or memory, and a
    job alone on an idle machine runs at full speed.
    """
    if not IS_WINDOWS:  # pragma: no cover
        return 0
    flags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    if background:
        flags |= BELOW_NORMAL_PRIORITY_CLASS
    return flags


def posix_background_niceness(background: bool) -> int:
    """`nice` increment for a job child on POSIX. 0 means unchanged."""
    return 5 if background else 0


def no_window_kwargs() -> dict[str, Any]:
    """Popen/run keyword arguments for a *helper* subprocess.

    Mandatory for every console program gpuq shells out to - `nvidia-smi`,
    `git`, `taskkill`. Without it these pop a visible console window, and they
    do it worst exactly where it is least acceptable: a parent with no console
    of its own (the dispatcher daemon, the job runner) makes Windows allocate a
    brand-new *visible* console for each child. The dispatcher polls
    `nvidia-smi`, so that becomes a window flashing every few seconds.

    Only for short-lived helpers whose output gpuq captures. User job commands
    are launched separately, with `child_creationflags`.
    """
    if not IS_WINDOWS:  # pragma: no cover - POSIX
        return {}
    return {"creationflags": CREATE_NO_WINDOW}


def windowless_python(executable: str | None = None) -> str:
    """The interpreter to use for gpuq's own background processes.

    On Windows a virtualenv's `python.exe` is a launcher that gives the real
    interpreter a console, so a background process leaves a `conhost.exe`
    behind no matter which CreationFlags are passed - measurably, DETACHED,
    CREATE_NO_WINDOW and both together all produce one. `pythonw.exe` is built
    without a console subsystem and produces none.

    It lives in the same directory, so it is the same environment and `-m gpuq`
    resolves identically. Falls back to the given executable when absent.
    """
    import os.path

    executable = executable or sys.executable
    if not IS_WINDOWS or not executable:
        return executable
    directory, name = os.path.split(executable)
    lowered = name.lower()
    if not lowered.startswith("python") or lowered.startswith("pythonw"):
        return executable
    candidate = os.path.join(directory, name[: len("python")] + "w" + name[len("python") :])
    return candidate if os.path.isfile(candidate) else executable


def posix_child_kwargs() -> dict[str, Any]:  # pragma: no cover - POSIX only
    """Give POSIX children their own process group, mirroring Job Objects."""
    if IS_WINDOWS:
        return {}
    return {"start_new_session": True}


# --------------------------------------------------------------------------
# PID identity
# --------------------------------------------------------------------------


def process_creation_time(pid: int | None) -> int | None:
    """Return an opaque creation stamp for `pid`, or None if it is not running.

    Two processes may reuse a PID, but never with the same creation time, so
    storing this alongside the PID makes a later kill provably safe.
    """
    if not pid or pid <= 0:
        return None
    if IS_WINDOWS:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return None
        try:
            # A process that has exited still answers GetProcessTimes for as
            # long as anyone holds a handle to it, so the exit code must be
            # checked too - otherwise this reports a dead process as live and
            # orphan recovery would adopt a job that is already over.
            STILL_ACTIVE = 259
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            if code.value != STILL_ACTIVE:
                return None

            creation = wintypes.FILETIME()
            exit_t = wintypes.FILETIME()
            kernel_t = wintypes.FILETIME()
            user_t = wintypes.FILETIME()
            ok = k32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_t),
                ctypes.byref(kernel_t),
                ctypes.byref(user_t),
            )
            if not ok:
                return None
            return (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        finally:
            k32.CloseHandle(handle)
    try:  # pragma: no cover - POSIX
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            fields = fh.read().rsplit(")", 1)[-1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def pid_matches(pid: int | None, expected_creation: int | None) -> bool:
    """True when `pid` is alive and is the same process we recorded."""
    if not pid:
        return False
    actual = process_creation_time(pid)
    if actual is None:
        return False
    if expected_creation is None:
        # No stamp recorded (pre-upgrade row): treat as unverified, refuse kill.
        return False
    return actual == expected_creation


# --------------------------------------------------------------------------
# Job Objects
# --------------------------------------------------------------------------


class ProcessGroup:
    """A killable container for a process and everything it spawns.

    Windows: a Job Object. POSIX: the child's own process group.

    `KILL_ON_JOB_CLOSE` is deliberately NOT set: if the dispatcher daemon dies,
    long-running training jobs must keep running. Cancellation after a daemon
    restart falls back to a creation-time-verified `taskkill /T`.
    """

    def __init__(self, name_hint: str = "gpuq") -> None:
        self._handle: Any = None
        self._pids: list[int] = []
        self.name_hint = name_hint
        if IS_WINDOWS:
            self._create_job_object()

    def _create_job_object(self) -> None:  # pragma: no cover - Windows only
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        handle = k32.CreateJobObjectW(None, None)
        self._handle = handle if handle else None

    def assign(self, pid: int) -> bool:
        """Put a running process into the group. Best effort."""
        self._pids.append(int(pid))
        if not IS_WINDOWS or not self._handle:
            return False
        import ctypes  # pragma: no cover - Windows only
        from ctypes import wintypes

        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        proc = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
        if not proc:
            return False
        try:
            return bool(k32.AssignProcessToJobObject(self._handle, proc))
        finally:
            k32.CloseHandle(proc)

    def signal_break(self) -> bool:
        """Politely ask the child group to stop (CTRL_BREAK / SIGTERM).

        On Windows this only reaches children that share a console with the
        sender; a console-less dispatcher cannot deliver it, which is why
        cancellation always has `terminate()` as its backstop.
        """
        if IS_WINDOWS:  # pragma: no cover - Windows only
            import ctypes

            CTRL_BREAK_EVENT = 1
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            ok = False
            for pid in self._pids:
                ok = bool(k32.GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, int(pid))) or ok
            return ok
        import signal  # pragma: no cover - POSIX

        ok = False
        for pid in self._pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                ok = True
            except OSError:
                pass
        return ok

    def terminate(self) -> bool:
        """Hard-kill every process in the group."""
        killed = False
        if IS_WINDOWS and self._handle:  # pragma: no cover - Windows only
            import ctypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            killed = bool(k32.TerminateJobObject(self._handle, 1))
        if not killed:
            for pid in self._pids:
                killed = terminate_tree(pid) or killed
        return killed

    def close(self) -> None:
        if IS_WINDOWS and self._handle:  # pragma: no cover - Windows only
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._handle)
        self._handle = None


def terminate_tree(pid: int, *, expected_creation: int | None = None) -> bool:
    """Kill `pid` and its descendants.

    When `expected_creation` is supplied the PID identity is verified first, so
    a recycled PID belonging to unrelated work is never killed.
    """
    if not pid or pid <= 0:
        return False
    if expected_creation is not None and not pid_matches(pid, expected_creation):
        return False
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                capture_output=True,
                timeout=30,
                check=False,
                **no_window_kwargs(),
            )
        else:  # pragma: no cover - POSIX
            import signal

            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        return False

    # Termination is asynchronous; give it a moment before reporting failure,
    # since callers log and escalate based on this result.
    import time

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process_creation_time(pid) is None:
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------
# Single-instance lock
# --------------------------------------------------------------------------


class ExclusiveLock:
    """Cross-platform advisory file lock used to keep one dispatcher alive."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        self._fh: Any = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            fh = open(self.path, "a+b")
        except OSError:
            return False
        try:
            if IS_WINDOWS:
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - POSIX
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if IS_WINDOWS:
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - POSIX
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


def is_locked(path: str | os.PathLike[str]) -> bool:
    """True when another process currently holds the lock."""
    probe = ExclusiveLock(path)
    if probe.acquire():
        probe.release()
        return False
    return True
