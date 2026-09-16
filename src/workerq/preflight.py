"""Cheap checks that catch a doomed job before it waits in the queue.

A job that cannot start still costs the queue its whole wait. In the 24 hours
to 2026-09-16 the median wait was 42 minutes against a median runtime under 6,
and a large share of failures died in their first second: a script that is not
in the snapshot (#735, #768), an absolute path that does not exist (#736), a
syntax error in a file the agent had just edited (#1212: "'[' was never closed").
Every one of those is decidable at submit time, from the frozen snapshot, in
milliseconds.

The rule is **only definite errors refuse a job**. Anything uncertain - a shell
string, a module that may be installed rather than local, a file the command
might create - passes. A false refusal is worse than a late failure, because it
blocks work that would have run.

Syntax is checked with this interpreter. When the job's own Python can be read
(its venv's pyvenv.cfg, or .python-version) and is newer, parsing is skipped, so
syntax this interpreter does not know is never reported as an error.
"""

from __future__ import annotations

import ast
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Local modules followed from the entry point before giving up. Enough for a
#: research repo's own package; a vendored tree is not worth the submit latency.
MAX_FILES = 300
MAX_SECONDS = 3.0

_PYTHON = re.compile(r"^(python|pythonw|py)(\d+(\.\d+)?)?(\.exe)?$", re.IGNORECASE)
#: Interpreter options that consume the next argument.
_PY_OPTS_WITH_VALUE = {"-W", "-X", "--check-hash-based-pycs"}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _basename(token: str) -> str:
    return token.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _resolve(token: str, cwd: Path) -> Path:
    path = Path(token)
    return path if path.is_absolute() else cwd / path


def _is_python(token: str) -> bool:
    return bool(_PYTHON.match(_basename(token)))


def _python_target(args: list[str]) -> tuple[str, str] | None:
    """("script", path) or ("module", name) for a python argv tail, if any."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-m" and i + 1 < len(args):
            return "module", args[i + 1]
        if arg.startswith("-m") and len(arg) > 2:
            return "module", arg[2:]
        if arg == "-c" or arg.startswith("-c"):
            return None
        if arg in _PY_OPTS_WITH_VALUE:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return "script", arg
    return None


def _module_file(name: str, roots: list[Path]) -> Path | None:
    parts = name.split(".")
    for root in roots:
        base = root.joinpath(*parts)
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def _compile_closure(entry: Path, roots: list[Path], report: Report) -> None:
    """Parse the entry file and the local modules it imports.

    An import that does not resolve to a file under `roots` is assumed to be
    installed and is not followed. Only a file that exists and fails to parse
    is an error.
    """
    started = time.monotonic()
    seen: set[Path] = set()
    todo = [entry]
    while todo and len(seen) < MAX_FILES and time.monotonic() - started < MAX_SECONDS:
        path = todo.pop()
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        try:
            source = path.read_bytes()
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            where = f"{path}:{exc.lineno}" if exc.lineno else str(path)
            report.errors.append(f"{where}: {type(exc).__name__}: {exc.msg}")
            continue
        except (ValueError, UnicodeDecodeError):
            continue
        report.checked.append(str(path))
        search = [path.parent] + roots
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    mod = node.module or ""
                    target = _module_file(mod, [base]) if mod else None
                    if target is not None:
                        todo.append(target)
                    for alias in node.names:
                        sub = _module_file(f"{mod}.{alias.name}".strip("."), [base])
                        if sub is not None:
                            todo.append(sub)
                    continue
                if node.module:
                    names = [node.module]
                    names += [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                target = _module_file(name, search)
                if target is not None:
                    todo.append(target)


def _version_from_text(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _job_python_version(program: str, cwd: Path) -> tuple[int, int] | None:
    """The job's Python version when it can be read without running anything."""
    if "/" in program or "\\" in program:
        exe = _resolve(program, cwd)
        for folder in (exe.parent, exe.parent.parent):
            cfg = folder / "pyvenv.cfg"
            if cfg.is_file():
                try:
                    for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
                        key, _, value = line.partition("=")
                        if key.strip() in ("version", "version_info"):
                            return _version_from_text(value)
                except OSError:
                    return None
    pin = cwd / ".python-version"
    if pin.is_file():
        try:
            return _version_from_text(pin.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return None
    return None


def check(command: list[str], cwd: Path, *, shell_mode: bool = False) -> Report:
    """Definite reasons this command cannot start, judged in `cwd`."""
    report = Report()
    if shell_mode or not command:
        return report
    program = str(command[0])
    has_dir = "/" in program or "\\" in program

    # A path-shaped program must exist. A bare name is resolved on PATH at run
    # time, which this machine's PATH says nothing reliable about.
    if has_dir:
        resolved = _resolve(program, cwd)
        if not resolved.exists():
            hint = (
                " It is relative, so it is looked for in the frozen snapshot, which "
                "excludes gitignored paths such as a .venv - declare it as "
                "passthrough in .gpuq.toml or use an absolute path."
                if not Path(program).is_absolute()
                else ""
            )
            report.errors.append(f"program not found: {program}.{hint}")
            return report

    args = [str(a) for a in command[1:]]
    # `uv run [opts] python script.py` and `uv run script.py`.
    if _basename(program).lower() in ("uv", "uv.exe") and args[:1] == ["run"]:
        rest = args[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[1:]
        if rest and _is_python(rest[0]):
            args = rest[1:]
        elif rest and rest[0].endswith(".py"):
            args = rest
        else:
            return report
    elif not _is_python(program):
        return report

    target = _python_target(args)
    if target is None:
        return report
    # Syntax newer than this interpreter would be reported as an error that is
    # not one. Existence is still checked; parsing is skipped.
    version = _job_python_version(program, cwd)
    parse = version is None or version <= sys.version_info[:2]
    kind, value = target
    roots = [cwd]
    if kind == "script":
        script = _resolve(value, cwd)
        if not script.exists():
            report.errors.append(
                f"script not found: {value} (looked in {script.parent}). A relative "
                "script is read from the frozen snapshot, which holds only files git "
                "can see - commit it, un-ignore it, or declare its folder passthrough."
            )
            return report
        if script.is_file() and parse:
            _compile_closure(script, [script.parent, cwd], report)
    else:
        module = _module_file(value, roots)
        if module is not None and parse:
            _compile_closure(module, roots, report)
    return report
