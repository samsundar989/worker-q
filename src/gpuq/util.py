"""Small shared helpers: time, paths, atomic writes, process liveness."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

IS_WINDOWS = os.name == "nt"

# --------------------------------------------------------------------------
# Time (spec 29.11: UTC ISO-8601 internally)
# --------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat(timespec="microseconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_seconds(iso: str | None, *, until: str | None = None) -> float | None:
    start = parse_iso(iso)
    if start is None:
        return None
    end = parse_iso(until) if until else utcnow()
    if end is None:
        end = utcnow()
    return max(0.0, (end - start).total_seconds())


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw]?)\s*$", re.IGNORECASE)
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "": 86400}


def parse_duration(text: str) -> timedelta:
    """Parse '7d', '30m', '90' (bare number = days, matching --older-than)."""
    m = _DURATION_RE.match(text or "")
    if not m:
        raise ValueError(f"invalid duration: {text!r} (use forms like 7d, 12h, 30m)")
    return timedelta(seconds=float(m.group(1)) * _DURATION_UNITS[m.group(2).lower()])


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def expand_path(value: str | os.PathLike[str]) -> Path:
    """Expand ~ and environment variables, return an absolute path."""
    text = os.path.expandvars(str(value))
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return Path(os.path.normpath(str(p)))


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Fully resolve (follows links). Falls back to expand_path when missing."""
    p = expand_path(value)
    try:
        return p.resolve()
    except OSError:  # pragma: no cover - exotic filesystems
        return p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_within(child: Path, parent: Path) -> bool:
    """True when `child` is `parent` or a descendant. Used to guard deletion."""
    try:
        c = Path(os.path.normcase(os.path.normpath(str(child.absolute()))))
        p = Path(os.path.normcase(os.path.normpath(str(parent.absolute()))))
    except OSError:  # pragma: no cover
        return False
    return c == p or p in c.parents


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write via temp file + replace so readers never see a partial file."""
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def restrict_permissions(path: Path) -> None:
    """Best-effort owner-only permissions (spec 30)."""
    try:
        if IS_WINDOWS:
            # POSIX bits are largely advisory on Windows; the state dir lives
            # under the user profile which is already ACL-protected.
            return
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    except OSError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------
# Processes
# --------------------------------------------------------------------------


def is_pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if IS_WINDOWS:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_process_tree(pid: int, *, force: bool = True) -> bool:
    """Terminate a process and its descendants. Best effort, never raises."""
    if not is_pid_alive(pid):
        return True
    try:
        if IS_WINDOWS:
            args = ["taskkill", "/PID", str(pid), "/T"]
            if force:
                args.append("/F")
            from gpuq.winproc import no_window_kwargs

            subprocess.run(
                args, capture_output=True, timeout=30, check=False, **no_window_kwargs()
            )
        else:  # pragma: no cover - POSIX path kept for future backends
            import signal

            os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
    except Exception:
        return False
    return not is_pid_alive(pid)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def display_command(argv: list[str], shell_mode: bool = False) -> str:
    if shell_mode and argv:
        return argv[0] if len(argv) == 1 else " ".join(argv)
    try:
        return shlex.join(argv)
    except Exception:  # pragma: no cover
        return " ".join(argv)


def truncate(text: str, width: int) -> str:
    if width <= 1 or len(text) <= width:
        return text
    return text[: width - 1] + "…"


def human_bytes(num: float | None, *, unit: str = "MiB") -> str:
    """Format a MiB quantity as GiB when large."""
    if num is None:
        return "-"
    if unit == "MiB":
        gib = num / 1024.0
        return f"{gib:.1f} GiB" if gib >= 1 else f"{num:.0f} MiB"
    return f"{num}"


ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_assignment(text: str) -> tuple[str, str]:
    """Parse KEY=VALUE with validation (spec 30: validate env keys)."""
    if "=" not in text:
        raise ValueError(f"--env expects KEY=VALUE, got {text!r}")
    key, _, value = text.partition("=")
    key = key.strip()
    if not ENV_KEY_RE.match(key):
        raise ValueError(f"invalid environment variable name: {key!r}")
    return key, value


def hostname() -> str:
    import socket

    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover
        return "unknown"


def python_executable() -> str:
    return sys.executable or "python"
