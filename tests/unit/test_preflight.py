"""Refusing at submit what cannot start, and nothing else.

Each doomed case below is a real failure that waited in the queue first. The
other half matters as much: a false refusal blocks work that would have run,
so anything uncertain must pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workerq import preflight
from workerq.core import GPUQError, GPUQService, SubmitRequest


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def test_a_script_missing_from_the_snapshot_is_refused(tmp_path):
    """#735: runs/validate_kagsim_batch_job727.py was gitignored."""
    report = preflight.check(["python", "-u", "runs/validate.py"], tmp_path)
    assert not report.ok and "script not found" in report.errors[0]


def test_an_absolute_script_that_does_not_exist_is_refused(tmp_path):
    """#736: the same script by absolute path, which did not exist either."""
    missing = str(tmp_path / "nope" / "x.py")
    assert not preflight.check(["python", missing], tmp_path).ok


def test_a_relative_interpreter_missing_from_the_snapshot_is_refused(tmp_path):
    report = preflight.check([".venv/Scripts/python.exe", "x.py"], tmp_path)
    assert not report.ok and "passthrough" in report.errors[0]


def test_a_syntax_error_in_the_script_is_refused(tmp_path):
    """#1212: "'[' was never closed"."""
    _tree(tmp_path, {"tools/screen.py": "peers = [1, 2\n"})
    report = preflight.check(["python", "tools/screen.py"], tmp_path)
    assert not report.ok and "tools" in report.errors[0] and ":1" in report.errors[0]


def test_a_syntax_error_in_a_local_import_is_refused(tmp_path):
    _tree(tmp_path, {
        "run.py": "from search import instrument\n",
        "search/__init__.py": "",
        "search/instrument.py": "def f(:\n    pass\n",
    })
    report = preflight.check(["python", "run.py"], tmp_path)
    assert not report.ok and "instrument.py" in report.errors[0]


def test_a_module_run_with_dash_m_is_parsed(tmp_path):
    _tree(tmp_path, {"search/__init__.py": "", "search/instrument.py": "x = (\n"})
    assert not preflight.check(["python", "-m", "search.instrument", "run"], tmp_path).ok


def test_relative_imports_are_followed(tmp_path):
    _tree(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/main.py": "from . import helper\n",
        "pkg/helper.py": "if True\n",
    })
    assert not preflight.check(["python", "-m", "pkg.main"], tmp_path).ok


def test_uv_run_is_understood(tmp_path):
    _tree(tmp_path, {"local_sim/score.py": "x = '\n"})
    assert not preflight.check(["uv", "run", "python", "-u", "local_sim/score.py"], tmp_path).ok


@pytest.mark.parametrize(
    "command",
    [
        ["python", "-m", "pytest", "-q"],           # installed module
        ["python", "-c", "print(1)"],
        ["powershell", "-Command", "python x.py"],  # a shell string: not judged
        ["cmd", "/c", "missing.bat"],               # a bare program: PATH decides
        ["python"],
    ],
)
def test_what_cannot_be_decided_passes(tmp_path, command):
    assert preflight.check(command, tmp_path).ok


def test_valid_code_with_installed_imports_passes(tmp_path):
    _tree(tmp_path, {"train.py": "import numpy as np\nimport os\nprint(np, os)\n"})
    assert preflight.check(["python", "train.py", "--epochs", "3"], tmp_path).ok


def test_shell_mode_is_not_judged(tmp_path):
    assert preflight.check(["python missing.py"], tmp_path, shell_mode=True).ok


def test_syntax_newer_than_this_interpreter_is_not_called_an_error(tmp_path):
    """A job on a newer Python may use syntax this one rejects."""
    _tree(tmp_path, {
        ".venv/pyvenv.cfg": "home = x\nversion = 3.99.0\n",
        ".venv/Scripts/python.exe": "",
        "new.py": "def f(:\n",
    })
    assert preflight.check([".venv/Scripts/python.exe", "new.py"], tmp_path).ok


def test_a_refused_submission_says_why_and_how_to_override(isolated_config, tmp_path):
    isolated_config.core.preflight = True
    svc = GPUQService(isolated_config)
    svc.ensure_ready()
    try:
        with pytest.raises(GPUQError, match="--no-preflight"):
            svc.submit(SubmitRequest(
                command=["python", "missing.py"], project="pf", gpus=0,
                snapshot=False, cwd=str(tmp_path),
            ))
        # and the override works
        svc.submit(SubmitRequest(
            command=["python", "missing.py"], project="pf", gpus=0,
            snapshot=False, cwd=str(tmp_path), preflight=False,
        ))
    finally:
        svc.backend.shutdown(timeout=10.0)
        svc.close()
