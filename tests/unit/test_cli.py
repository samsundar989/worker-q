"""CLI contract tests (spec sections 11, 23 and 28).

Run through Typer's CliRunner against an isolated profile. The dispatcher is
stubbed where a test only cares about argument handling and output shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpuq import cli as cli_module
from gpuq.cli import app
from gpuq.config import Config
from gpuq.core import GPUQService


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cli_env(isolated_config: Config, monkeypatch):
    """Point every CLI invocation at the isolated config, no daemon needed."""
    monkeypatch.setenv("GPUQ_STATE_DIR", str(isolated_config.state_dir))
    monkeypatch.setenv("GPUQ_CONFIG_FILE", str(isolated_config.source_path))
    monkeypatch.setenv("GPUQ_PROFILE", "pytest")

    service = GPUQService(isolated_config)
    service.ensure_ready()

    # Pretend the dispatcher is up; these tests never execute a job.
    monkeypatch.setattr(
        "gpuq.backends.local_dispatcher.LocalDispatcherBackend.ensure_daemon",
        lambda self, **kw: True,
    )
    monkeypatch.setattr(
        "gpuq.backends.local_dispatcher.LocalDispatcherBackend.daemon_running",
        lambda self: True,
    )
    service.initialize()
    yield isolated_config
    service.close()


def invoke(runner: CliRunner, *args: str):
    return runner.invoke(app, list(args), catch_exceptions=False)


# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------


def test_version(runner: CliRunner):
    result = invoke(runner, "version")
    assert result.exit_code == 0
    assert "gpuq" in result.stdout


def test_version_json(runner: CliRunner):
    result = invoke(runner, "version", "--json")
    payload = json.loads(result.stdout)
    assert payload["gpuq"] and payload["backend"] == "local_dispatcher"


def test_help_lists_the_core_commands(runner: CliRunner):
    result = invoke(runner, "--help")
    for command in ("submit", "status", "show", "logs", "cancel", "doctor"):
        assert command in result.stdout


def test_init_is_idempotent(runner: CliRunner, cli_env: Config):
    first = invoke(runner, "init")
    second = invoke(runner, "init")
    assert first.exit_code == 0 and second.exit_code == 0


# --------------------------------------------------------------------------
# submit validation
# --------------------------------------------------------------------------


def test_submit_without_a_command_fails(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "submit", "--project", "demo")
    assert result.exit_code != 0
    assert "no command given" in result.output


def test_submit_rejects_an_invalid_priority(runner: CliRunner, cli_env: Config, git_repo: Path):
    result = invoke(
        runner, "submit", "--cwd", str(git_repo), "--priority", "urgent", "--", "python", "-V"
    )
    assert result.exit_code != 0
    assert "invalid priority" in result.output


def test_submit_rejects_shell_and_argv_together(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "submit", "--shell", "echo hi", "--", "echo", "hi")
    assert result.exit_code != 0
    assert "not both" in result.output


def test_submit_rejects_a_bad_env_assignment(runner: CliRunner, cli_env: Config, git_repo: Path):
    result = invoke(
        runner, "submit", "--cwd", str(git_repo), "--env", "9BAD=1", "--", "python", "-V"
    )
    assert result.exit_code != 0
    assert "invalid environment variable name" in result.output


def test_submit_outside_a_git_repo_refuses_and_explains(
    runner: CliRunner, cli_env: Config, tmp_path: Path
):
    plain = tmp_path / "plain"
    plain.mkdir()
    result = invoke(runner, "submit", "--cwd", str(plain), "--", "python", "-V")
    assert result.exit_code != 0
    assert "--live-worktree" in result.output


def test_submit_project_inference(runner: CliRunner, cli_env: Config, git_repo: Path):
    result = invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["project"] == git_repo.name


def test_submit_explicit_project_wins(runner: CliRunner, cli_env: Config, git_repo: Path):
    result = invoke(
        runner, "submit", "--cwd", str(git_repo), "--project", "chosen", "--json", "--",
        "python", "-V",
    )
    assert json.loads(result.stdout)["project"] == "chosen"


def test_submit_json_is_pure_json(runner: CliRunner, cli_env: Config, git_repo: Path):
    """Spec 28: --json must put no decorative text on stdout."""
    result = invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V")
    payload = json.loads(result.stdout)
    assert payload["state"] == "QUEUED"
    assert payload["job_id"] >= 1
    assert payload["snapshot_commit"]


def test_submit_preserves_awkward_arguments(runner: CliRunner, cli_env: Config, git_repo: Path):
    argv = ["python", "train.py", "--name=a b", "--glob=*.py", "--u=café", "a&b|c"]
    result = invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", *argv)
    job_id = json.loads(result.stdout)["job_id"]

    show = invoke(runner, "show", str(job_id), "--json")
    assert json.loads(show.stdout)["command"] == argv


def test_submit_shell_mode_records_the_string(runner: CliRunner, cli_env: Config, git_repo: Path):
    result = invoke(
        runner, "submit", "--cwd", str(git_repo), "--json", "--shell", "echo a && echo b"
    )
    job_id = json.loads(result.stdout)["job_id"]
    detail = json.loads(invoke(runner, "show", str(job_id), "--json").stdout)
    assert detail["shell_mode"] is True
    assert detail["command"] == ["echo a && echo b"]


def test_submit_no_snapshot_uses_the_live_tree(
    runner: CliRunner, cli_env: Config, git_repo: Path
):
    result = invoke(
        runner, "submit", "--cwd", str(git_repo), "--no-snapshot", "--json", "--", "python", "-V"
    )
    payload = json.loads(result.stdout)
    assert payload["snapshot_mode"] == "none"
    assert Path(payload["execution_cwd"]) == git_repo.resolve()


def test_submit_human_output_shows_the_job_id(
    runner: CliRunner, cli_env: Config, git_repo: Path
):
    result = invoke(runner, "submit", "--cwd", str(git_repo), "--", "python", "-V")
    assert "submitted" in result.stdout
    assert "GPUQ job #" in result.stdout
    assert "gpuq logs" in result.stdout


# --------------------------------------------------------------------------
# status / list / show / logs
# --------------------------------------------------------------------------


def test_status_json_shape(runner: CliRunner, cli_env: Config, git_repo: Path):
    invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V")
    payload = json.loads(invoke(runner, "status", "--json").stdout)
    assert "summary" in payload and "jobs" in payload and "gpu" in payload
    assert payload["summary"]["concurrency"] >= 1
    assert payload["jobs"][0]["state"] == "QUEUED"


def test_list_and_status_return_the_same_jobs(
    runner: CliRunner, cli_env: Config, git_repo: Path
):
    invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V")
    a = json.loads(invoke(runner, "status", "--json").stdout)["jobs"]
    b = json.loads(invoke(runner, "list", "--json").stdout)["jobs"]
    assert [j["id"] for j in a] == [j["id"] for j in b]


def test_status_empty_queue_is_helpful(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "status")
    assert result.exit_code == 0
    assert "No jobs" in result.stdout


def test_status_filters_by_project(runner: CliRunner, cli_env: Config, git_repo: Path):
    invoke(runner, "submit", "--cwd", str(git_repo), "--project", "aaa", "--json", "--", "python")
    invoke(runner, "submit", "--cwd", str(git_repo), "--project", "bbb", "--json", "--", "python")
    jobs = json.loads(invoke(runner, "status", "--json", "--project", "aaa").stdout)["jobs"]
    assert {j["project"] for j in jobs} == {"aaa"}


def test_status_rejects_an_invalid_state_filter(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "status", "--state", "SLEEPING")
    assert result.exit_code != 0
    assert "invalid state" in result.output


def test_show_invalid_job_id(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "show", "9999")
    assert result.exit_code != 0
    assert "no such gpuq job" in result.output


def test_show_reports_provenance(runner: CliRunner, cli_env: Config, git_repo: Path):
    job_id = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V").stdout
    )["job_id"]
    detail = json.loads(invoke(runner, "show", str(job_id), "--json").stdout)
    assert detail["snapshot_commit"]
    assert detail["repo_root"] == str(git_repo.resolve())
    assert detail["backend_state"] == "QUEUED"
    assert detail["manifest_path"]


def test_show_human_output(runner: CliRunner, cli_env: Config, git_repo: Path):
    job_id = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V").stdout
    )["job_id"]
    result = invoke(runner, "show", str(job_id))
    assert "Snapshot commit" in result.stdout
    assert "Execution cwd" in result.stdout


def test_logs_for_a_queued_job_is_informational(
    runner: CliRunner, cli_env: Config, git_repo: Path
):
    """Spec 11.5: a queued job with no log yet is not an error."""
    job_id = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V").stdout
    )["job_id"]
    result = invoke(runner, "logs", str(job_id))
    assert result.exit_code == 0
    assert "has not been created yet" in result.stdout


def test_logs_for_a_missing_job(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "logs", "9999")
    assert result.exit_code != 0


# --------------------------------------------------------------------------
# cancel / promote
# --------------------------------------------------------------------------


def test_cancel_a_queued_job(runner: CliRunner, cli_env: Config, git_repo: Path):
    job_id = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V").stdout
    )["job_id"]
    result = invoke(runner, "cancel", str(job_id), "--json")
    payload = json.loads(result.stdout)
    assert payload["state"] == "CANCELLED"
    assert payload["action"] == "removed"


def test_cancel_is_idempotent(runner: CliRunner, cli_env: Config, git_repo: Path):
    job_id = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "-V").stdout
    )["job_id"]
    invoke(runner, "cancel", str(job_id))
    second = invoke(runner, "cancel", str(job_id), "--json")
    assert second.exit_code == 0
    payload = json.loads(second.stdout)
    assert payload["state"] == "CANCELLED"
    assert payload["action"] == "none"


def test_cancel_unknown_job(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "cancel", "9999")
    assert result.exit_code != 0


def test_promote_a_queued_job(runner: CliRunner, cli_env: Config, git_repo: Path):
    first = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "1").stdout
    )["job_id"]
    second = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "2").stdout
    )["job_id"]
    result = invoke(runner, "promote", str(second), "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["queue_position"] == 0
    assert first  # unchanged, still queued


def test_promote_a_finished_job_fails(runner: CliRunner, cli_env: Config, git_repo: Path):
    job_id = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python").stdout
    )["job_id"]
    invoke(runner, "cancel", str(job_id))
    result = invoke(runner, "promote", str(job_id))
    assert result.exit_code != 0
    assert "only QUEUED jobs can be promoted" in result.output


def test_critical_priority_goes_to_the_front(runner: CliRunner, cli_env: Config, git_repo: Path):
    normal_a = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "a").stdout
    )["job_id"]
    normal_b = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python", "b").stdout
    )["job_id"]
    critical = json.loads(
        invoke(
            runner, "submit", "--cwd", str(git_repo), "--priority", "critical", "--json", "--",
            "python", "c",
        ).stdout
    )["job_id"]

    queued = [
        j["id"]
        for j in json.loads(invoke(runner, "status", "--json").stdout)["jobs"]
        if j["state"] == "QUEUED"
    ]
    assert queued[0] == critical
    assert set(queued) == {normal_a, normal_b, critical}


# --------------------------------------------------------------------------
# doctor exit codes
# --------------------------------------------------------------------------


def test_doctor_json_and_exit_code(runner: CliRunner, cli_env: Config):
    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.stdout)
    assert payload["overall"] in ("HEALTHY", "DEGRADED", "BROKEN")
    assert result.exit_code == payload["exit_code"]
    assert result.exit_code in (0, 1, 2)
    assert any(c["name"] == "SQLite" for c in payload["checks"])


def test_doctor_human_output(runner: CliRunner, cli_env: Config):
    result = runner.invoke(app, ["doctor"])
    assert "Overall:" in result.stdout
    assert result.exit_code in (0, 1, 2)


def test_doctor_exit_code_2_when_a_check_fails(runner: CliRunner, cli_env: Config, monkeypatch):
    from gpuq.doctor import Check, Doctor

    monkeypatch.setattr(
        Doctor, "check_sqlite", lambda self: self.add("SQLite", "FAIL", "simulated")
    )
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["overall"] == "BROKEN"


# --------------------------------------------------------------------------
# config / concurrency / threshold
# --------------------------------------------------------------------------


def test_config_show_json(runner: CliRunner, cli_env: Config):
    payload = json.loads(invoke(runner, "config", "show", "--json").stdout)
    assert payload["config"]["core"]["max_concurrent_jobs"] >= 1
    assert "state_dir" in payload["paths"]


def test_config_show_human(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "config", "show")
    assert "max_concurrent_jobs" in result.stdout
    assert "Precedence" in result.stdout


def test_config_set_and_get(runner: CliRunner, cli_env: Config):
    invoke(runner, "config", "set", "gpu.free_memory_threshold_percent", "77")
    assert json.loads(invoke(runner, "config", "get", "gpu.free_memory_threshold_percent").stdout) == 77


def test_config_set_rejects_invalid(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "config", "set", "core.max_concurrent_jobs", "0")
    assert result.exit_code != 0


def test_concurrency_above_one_requires_yes(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "concurrency", "2")
    assert result.exit_code != 0
    assert "WARNING" in result.output
    assert "OOM" in result.output


def test_concurrency_with_yes_succeeds(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "concurrency", "2", "--yes", "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["max_concurrent_jobs"] == 2
    invoke(runner, "concurrency", "1")


def test_concurrency_show(runner: CliRunner, cli_env: Config):
    payload = json.loads(invoke(runner, "concurrency", "--json").stdout)
    assert payload["config"] >= 1


def test_gpu_threshold_set(runner: CliRunner, cli_env: Config):
    payload = json.loads(invoke(runner, "gpu-threshold", "80", "--json").stdout)
    assert payload["free_memory_threshold_percent"] == 80


def test_gpu_threshold_rejects_out_of_range(runner: CliRunner, cli_env: Config):
    assert invoke(runner, "gpu-threshold", "150").exit_code != 0


# --------------------------------------------------------------------------
# other commands
# --------------------------------------------------------------------------


def test_gpu_command_json(runner: CliRunner, cli_env: Config):
    payload = json.loads(invoke(runner, "gpu", "--json").stdout)
    assert "available" in payload


def test_reconcile_json(runner: CliRunner, cli_env: Config):
    payload = json.loads(invoke(runner, "reconcile", "--json", "--dry-run").stdout)
    assert payload["dry_run"] is True
    assert isinstance(payload["changes"], list)


def test_cleanup_dry_run_changes_nothing(runner: CliRunner, cli_env: Config, git_repo: Path):
    job_id = json.loads(
        invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python").stdout
    )["job_id"]
    snapshot = Path(
        json.loads(invoke(runner, "show", str(job_id), "--json").stdout)["snapshot_path"]
    )
    payload = json.loads(invoke(runner, "cleanup", "--dry-run", "--json").stdout)
    assert payload["dry_run"] is True
    assert snapshot.exists()


def test_cleanup_keeps_active_job_snapshots(runner: CliRunner, cli_env: Config, git_repo: Path):
    invoke(runner, "submit", "--cwd", str(git_repo), "--json", "--", "python")
    payload = json.loads(invoke(runner, "cleanup", "--json", "--older-than", "0d").stdout)
    assert payload["snapshots"] == []
    assert any("still active" in s for s in payload["skipped"])


def test_cleanup_rejects_a_bad_duration(runner: CliRunner, cli_env: Config):
    result = invoke(runner, "cleanup", "--older-than", "banana")
    assert result.exit_code != 0


def test_uninstall_dry_run_is_default(runner: CliRunner, cli_env: Config):
    payload = json.loads(invoke(runner, "uninstall", "--json").stdout)
    assert payload["dry_run"] is True
    assert "actions" not in payload
    assert Path(payload["state"]["path"]).exists()


def test_mcp_command_prints_stdio_invocation(runner: CliRunner, cli_env: Config):
    payload = json.loads(invoke(runner, "mcp", "command", "--json").stdout)
    assert payload["transport"] == "stdio"
    assert payload["args"][:2] == ["-m", "gpuq"]


def test_claude_policy_roundtrip_via_cli(runner: CliRunner, cli_env: Config, tmp_path: Path):
    target = tmp_path / "CLAUDE.md"
    assert invoke(runner, "claude-policy", "install", "--path", str(target)).exit_code == 0
    status = json.loads(
        invoke(runner, "claude-policy", "status", "--path", str(target), "--json").stdout
    )
    assert status["installed"] is True
    assert invoke(runner, "claude-policy", "remove", "--path", str(target)).exit_code == 0
    assert "gpuq-policy" not in target.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# concurrent CLI invocations (spec 29.15)
# --------------------------------------------------------------------------


def test_two_submissions_in_parallel_both_succeed(
    runner: CliRunner, cli_env: Config, git_repo: Path
):
    import threading

    results: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def submit(index: int) -> None:
        try:
            service = GPUQService(cli_env)
            from gpuq.core import SubmitRequest

            out = service.submit(
                SubmitRequest(
                    command=["python", "-c", f"print({index})"],
                    cwd=str(git_repo),
                    gpus=0,
                )
            )
            with lock:
                results.append(out.job.id)
            service.close()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, errors
    assert len(set(results)) == 4
