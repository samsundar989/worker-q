"""The node registry, and reading a machine's state over a wire.

Phase 1 of docs/multi-node.md. Nothing here dispatches a job; these are the
guarantees placement will rest on: a node round-trips through config, a report
survives serialisation with its `None`s intact, an incompatible node is refused
rather than guessed at, and an unreachable one is explained rather than
described as empty.
"""

from __future__ import annotations

import pytest

from workerq import nodes
from workerq.config import (
    Config,
    ConfigError,
    CoreConfig,
    NodeConfig,
    load_config,
)
from workerq.resources import ResourceRequest, admit

GIB = 1024.0


def write_config(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return load_config(path)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_nodes_round_trip_through_toml(tmp_path):
    config = Config(core=CoreConfig(state_dir=str(tmp_path)), source_path=tmp_path / "c.toml")
    config.nodes = [
        NodeConfig(name="3080ti", address="desktop-unr95nb", user="samsu"),
        NodeConfig(name="spare", address="10.0.0.9", port=2222, enabled=False),
    ]
    config.validate()
    path = config.save()

    reloaded = load_config(path)
    assert [n.name for n in reloaded.nodes] == ["3080ti", "spare"]
    assert reloaded.node("3080ti").target == "samsu@desktop-unr95nb"
    assert reloaded.node("spare").port == 2222
    # A disabled node stays registered but is not offered to placement.
    assert [n.name for n in reloaded.active_nodes()] == ["3080ti"]


def test_a_node_may_not_be_called_local(tmp_path):
    """`local` always means this machine; a registry entry must not shadow it."""
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    config.nodes = [NodeConfig(name="local", address="somewhere")]
    with pytest.raises(ConfigError, match="reserved"):
        config.validate()


def test_duplicate_node_names_are_refused(tmp_path):
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    config.nodes = [
        NodeConfig(name="a", address="x"),
        NodeConfig(name="a", address="y"),
    ]
    with pytest.raises(ConfigError, match="duplicate"):
        config.validate()


def test_unknown_keys_inside_a_node_are_ignored(tmp_path):
    """A config written by a newer worker-q must not brick an older one."""
    config = write_config(
        tmp_path,
        '[[node]]\nname = "n1"\naddress = "h1"\nfuture_option = 7\n',
    )
    assert config.node("n1").address == "h1"


def test_a_node_without_a_name_is_fatal(tmp_path):
    """Unlike an unknown key. Silently dropping it hides a node someone added."""
    with pytest.raises(ConfigError, match="name"):
        write_config(tmp_path, '[[node]]\naddress = "h1"\n')


def test_a_single_node_table_is_rejected_with_advice(tmp_path):
    """`[node]` instead of `[[node]]` is the easy mistake; say so."""
    with pytest.raises(ConfigError, match=r"\[\[node\]\]"):
        write_config(tmp_path, '[node]\nname = "n1"\naddress = "h1"\n')


def test_no_nodes_means_an_ordinary_single_machine_install(tmp_path):
    config = write_config(tmp_path, "[core]\nmax_concurrent_jobs = 2\n")
    assert config.nodes == []
    assert config.active_nodes() == []


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def test_a_local_report_describes_this_machine(tmp_path):
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    report = nodes.local_report(config)

    assert report.is_local and report.online
    assert report.protocol == nodes.NODE_PROTOCOL_VERSION
    assert report.cpus and report.cpus >= 1
    assert report.snapshot(config) is not None


def test_a_report_survives_the_wire_without_turning_none_into_zero(tmp_path):
    """`None` and `0` mean different things everywhere in this codebase.

    A device that reported no free memory must not come back claiming zero,
    which admission would read as "full" rather than "unknown".
    """
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    payload = nodes.local_payload(config)
    payload["gpu"] = {
        "available": True,
        "devices": [
            {
                "index": 0,
                "uuid": "u",
                "name": "RTX 3080 Ti",
                "memory_total_mib": 12 * GIB,
                "memory_used_mib": None,
                "memory_free_mib": None,
                "utilization_percent": None,
            }
        ],
    }
    rebuilt = nodes._parse_payload("3080ti", payload)
    device = rebuilt.gpu.devices[0]
    assert device.memory_total_mib == 12 * GIB
    assert device.memory_free_mib is None
    assert device.memory_used_mib is None


def test_a_remote_report_feeds_admission_for_that_machine(tmp_path):
    """The join between phase 1 and phase 2: a report becomes a NodeSnapshot.

    A 20 GiB job is refused on a reported 12 GiB card, using the same
    `admit()` that guards the local machine.
    """
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    payload = nodes.local_payload(config)
    payload["host"] = {
        "total_mib": 16 * GIB,
        "available_mib": 13 * GIB,
        "commit_used_mib": 3 * GIB,
        "commit_limit_mib": 24 * GIB,
    }
    payload["commit_ceiling_mib"] = 64 * GIB
    payload["cpus"] = 16
    payload["gpu"] = {
        "available": True,
        "devices": [{
            "index": 0, "uuid": "u", "name": "RTX 3080 Ti",
            "memory_total_mib": 12 * GIB, "memory_used_mib": 1 * GIB,
            "memory_free_mib": 11 * GIB, "utilization_percent": 0.0,
        }],
    }
    payload["reserve"] = {"ram_mib": 3 * GIB, "vram_mib": 1 * GIB, "cpus": 1}

    snapshot = nodes._parse_payload("3080ti", payload).snapshot(config)
    assert snapshot is not None and snapshot.name == "3080ti"

    small = ResourceRequest(ram_mib=3 * GIB, vram_mib=0.0, cpus=1)
    large = ResourceRequest(ram_mib=8 * GIB, vram_mib=20 * GIB, cpus=1)
    assert admit(config, small, [], node=snapshot).admit
    assert not admit(config, large, [], node=snapshot).admit


def test_an_unreachable_node_reports_no_capacity_not_zero_capacity(tmp_path):
    """A scheduler must never read "off" as "idle and empty"."""
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    report = nodes.NodeReport(name="gone", error="connection refused")
    assert not report.online
    assert report.snapshot(config) is None


# --------------------------------------------------------------------------
# Compatibility
# --------------------------------------------------------------------------


def test_protocol_mismatch_refuses_and_version_match_does_not_reassure(tmp_path):
    """The check `--version` could not do.

    Both machines really did report 1.2.0 while one lacked a config key, a
    schema column and the current way of measuring RAM, so equal versions are
    not evidence and only the protocol may refuse.
    """
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    local = nodes.local_report(config)

    same = nodes.NodeReport(name="n", protocol=local.protocol, version="0.0.1")
    ok, why = nodes.compatibility(local, same)
    assert ok and why is None

    newer = nodes.NodeReport(name="n", protocol=local.protocol + 1, version=local.version)
    ok, why = nodes.compatibility(local, newer)
    assert not ok and "protocol" in why


def test_a_node_too_old_to_report_a_protocol_is_refused(tmp_path):
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    local = nodes.local_report(config)
    ok, why = nodes.compatibility(local, nodes.NodeReport(name="n", version="1.2.0"))
    assert not ok and "too old" in why


def test_an_offline_node_reports_its_own_reason(tmp_path):
    config = Config(core=CoreConfig(state_dir=str(tmp_path)))
    local = nodes.local_report(config)
    ok, why = nodes.compatibility(local, nodes.NodeReport(name="n", error="machine is off"))
    assert not ok and why == "machine is off"


# --------------------------------------------------------------------------
# Failure explanations
# --------------------------------------------------------------------------


def test_a_rich_framed_remote_error_does_not_surface_as_a_box_border():
    """The naive `splitlines()[-1]` returned a border, which is what prompted this.

    Typer frames "No such command" in box-drawing characters, so the last line
    of stderr is decoration and the reason is in the middle.
    """
    stderr = (
        "┌─ Error ────┐\n"
        "│ No such command '_node-report'. │\n"
        "└──────────┘"
    )
    message = nodes._explain_failure(2, stderr, "")
    assert "predates" in message
    assert not any(ch in message for ch in "┌└│")


@pytest.mark.parametrize(
    "stderr, expected",
    [
        ("Permission denied (publickey).", "key"),
        ("ssh: Could not resolve hostname foo", "resolve"),
        ("connect to host foo port 22: Connection refused", "refused"),
        ("'workerq' is not recognized as an internal or external command", "not found"),
    ],
)
def test_common_ssh_failures_are_named_not_echoed(stderr, expected):
    """"Offline" covers several different problems that need different fixes."""
    assert expected in nodes._explain_failure(255, stderr, "")


def test_an_unrecognised_failure_still_says_something_useful():
    assert "weird" in nodes._explain_failure(1, "something weird happened", "")
    assert "exited 1" in nodes._explain_failure(1, "", "")


def test_an_unknown_top_level_section_does_not_brick_an_older_workerq(tmp_path):
    """The guarantee that was written down but only half-implemented.

    Unknown keys *inside* a section were always ignored so that "a config
    written by a newer gpuq does not brick an older one". A whole new
    top-level table was not covered, and adding `[[node]]` duly bricked every
    install that predated it - including the one serving the live queue, which
    could no longer parse its own config to be told to stop.
    """
    config = write_config(
        tmp_path,
        "[core]\nmax_concurrent_jobs = 3\n\n"
        '[[future_feature]]\nname = "x"\n\n'
        'scalar_at_top_level = 7\n',
    )
    assert config.core.max_concurrent_jobs == 3
    assert config.nodes == []
