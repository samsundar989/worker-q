"""Config loading, validation and precedence (spec section 23)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuq.config import (
    Config,
    ConfigError,
    CoreConfig,
    GpuConfig,
    default_config_path,
    default_state_dir,
    get_dotted,
    load_config,
    set_dotted_and_save,
)


def test_defaults_load(tmp_path: Path):
    """Built-in defaults, isolated from any real config file on this machine."""
    config = load_config(tmp_path / "absent.toml", environ={})
    assert config.core.max_concurrent_jobs == 1
    assert config.core.default_priority == "normal"
    assert config.core.snapshot_mode == "git"
    assert config.gpu.free_memory_threshold_percent == 90
    assert config.gpu.exclusive_by_default is True
    assert config.claude.install_user_policy is True


def test_tilde_expands(tmp_path: Path):
    config = load_config(tmp_path / "absent.toml", environ={})
    assert "~" not in str(config.state_dir)
    assert config.state_dir.is_absolute()
    assert "~" not in str(default_state_dir(None))
    assert "~" not in str(default_config_path(None))


def test_derived_paths_are_under_state_dir(tmp_path: Path):
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    for path in (
        config.db_path,
        config.logs_dir,
        config.snapshots_dir,
        config.run_dir,
        config.backend_dir,
        config.jobs_dir,
    ):
        assert str(path).startswith(str(tmp_path))


@pytest.mark.parametrize("percent", [-1, 101, 1000])
def test_invalid_percentages_rejected(percent: int):
    with pytest.raises(ConfigError, match="between 0 and 100"):
        Config(gpu=GpuConfig(free_memory_threshold_percent=percent))


def test_non_integer_percentage_rejected():
    with pytest.raises(ConfigError, match="must be an integer"):
        Config(gpu=GpuConfig(free_memory_threshold_percent=True))


@pytest.mark.parametrize("count", [0, -1, -100])
def test_concurrency_below_one_rejected(count: int):
    with pytest.raises(ConfigError, match=">= 1"):
        Config(core=CoreConfig(max_concurrent_jobs=count))


def test_invalid_priority_rejected():
    with pytest.raises(ConfigError, match="default_priority"):
        Config(core=CoreConfig(default_priority="urgent"))


def test_invalid_snapshot_mode_rejected():
    with pytest.raises(ConfigError, match="snapshot_mode"):
        Config(core=CoreConfig(snapshot_mode="magic"))


def test_file_overrides_defaults(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[core]\nmax_concurrent_jobs = 3\n\n[gpu]\nfree_memory_threshold_percent = 42\n",
        encoding="utf-8",
    )
    config = load_config(path, environ={})
    assert config.core.max_concurrent_jobs == 3
    assert config.gpu.free_memory_threshold_percent == 42


def test_env_overrides_file(tmp_path: Path):
    """Precedence: env beats the config file."""
    path = tmp_path / "config.toml"
    path.write_text("[core]\nmax_concurrent_jobs = 3\n", encoding="utf-8")
    config = load_config(path, environ={"GPUQ_MAX_CONCURRENT_JOBS": "7"})
    assert config.core.max_concurrent_jobs == 7


def test_generic_env_form(tmp_path: Path):
    config = load_config(
        tmp_path / "missing.toml", environ={"GPUQ_GPU_FREE_MEMORY_THRESHOLD_PERCENT": "55"}
    )
    assert config.gpu.free_memory_threshold_percent == 55


def test_cli_overrides_env(tmp_path: Path):
    """Precedence: an explicit override beats env."""
    config = load_config(tmp_path / "missing.toml", environ={"GPUQ_MAX_CONCURRENT_JOBS": "7"})
    overridden = config.with_overrides(**{"core.max_concurrent_jobs": 2})
    assert overridden.core.max_concurrent_jobs == 2


def test_task_spooler_section_is_accepted_as_alias(tmp_path: Path):
    """A spec-shaped [task_spooler] table still configures the backend."""
    path = tmp_path / "config.toml"
    path.write_text("[task_spooler]\nmax_finished = 55\n", encoding="utf-8")
    config = load_config(path, environ={})
    assert config.backend.max_finished == 55


def test_unknown_keys_are_ignored_not_fatal(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[core]\nmax_concurrent_jobs = 2\nfuture_option = 5\n\n[brandnew]\nx = 1\n",
        encoding="utf-8",
    )
    config = load_config(path, environ={})
    assert config.core.max_concurrent_jobs == 2


def test_invalid_toml_raises(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[core\nbroken", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path, environ={})


def test_roundtrip_through_toml(tmp_path: Path):
    path = tmp_path / "config.toml"
    original = Config(
        core=CoreConfig(state_dir=str(tmp_path / "state"), max_concurrent_jobs=2),
        gpu=GpuConfig(free_memory_threshold_percent=75),
        source_path=path,
    )
    original.save()
    reloaded = load_config(path, environ={})
    assert reloaded.core.max_concurrent_jobs == 2
    assert reloaded.gpu.free_memory_threshold_percent == 75


def test_set_dotted_and_save(tmp_path: Path):
    path = tmp_path / "config.toml"
    config = Config(core=CoreConfig(state_dir=str(tmp_path)), source_path=path)
    updated = set_dotted_and_save(config, "gpu.free_memory_threshold_percent", "85")
    assert updated.gpu.free_memory_threshold_percent == 85
    assert load_config(path, environ={}).gpu.free_memory_threshold_percent == 85


def test_set_dotted_validates(tmp_path: Path):
    config = Config(core=CoreConfig(state_dir=str(tmp_path)), source_path=tmp_path / "c.toml")
    with pytest.raises(ConfigError):
        set_dotted_and_save(config, "core.max_concurrent_jobs", "0")
    with pytest.raises(ConfigError, match="unknown configuration key"):
        set_dotted_and_save(config, "core.nope", "1")


def test_get_dotted(tmp_path: Path):
    config = load_config(tmp_path / "absent.toml", environ={})
    assert get_dotted(config, "core.max_concurrent_jobs") == 1
    with pytest.raises(ConfigError):
        get_dotted(config, "core.does_not_exist")


def test_profile_isolates_state_dir(tmp_path: Path):
    plain = load_config(tmp_path / "absent.toml", environ={})
    profiled = load_config(tmp_path / "absent.toml", environ={"GPUQ_PROFILE": "testing"})
    assert plain.state_dir != profiled.state_dir
    assert str(profiled.state_dir).endswith("gpuq-testing")


def test_boolean_coercion(tmp_path: Path):
    path = tmp_path / "config.toml"
    config = load_config(path, environ={"GPUQ_CLAUDE_INSTALL_USER_POLICY": "false"})
    assert config.claude.install_user_policy is False
    config = load_config(path, environ={"GPUQ_CLAUDE_INSTALL_USER_POLICY": "yes"})
    assert config.claude.install_user_policy is True


def test_bad_integer_env_is_a_clear_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="must be an integer"):
        load_config(tmp_path / "c.toml", environ={"GPUQ_MAX_CONCURRENT_JOBS": "lots"})
