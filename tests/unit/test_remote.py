"""Running one job on another machine.

Phase 4 of docs/multi-node.md. The remote is stubbed here; what these pin is
the contract - a spec crosses as a file rather than a command line, the
primary's id survives in a label so a dropped connection can be reconciled,
and an incompatible peer is refused rather than half-understood.
"""

from __future__ import annotations

import json

import pytest

from workerq import remote
from workerq.config import NodeConfig
from workerq.nodes import RemoteResult


def node() -> NodeConfig:
    return NodeConfig(name="w", address="host")


def spec(**kw) -> remote.JobSpec:
    base = dict(
        project="proj",
        argv=["python", "train.py"],
        cwd=r"D:\repos\proj\.gpuq-work\job-000042",
        origin_job_id=42,
        origin_host="PRIMARY",
    )
    base.update(kw)
    return remote.JobSpec(**base)


# --------------------------------------------------------------------------
# The label is the only durable link between two job records
# --------------------------------------------------------------------------


def test_the_primary_id_survives_in_the_label():
    assert remote.parse_origin_label(spec().label) == ("PRIMARY", 42)


def test_a_hostname_containing_a_colon_still_round_trips():
    """Parsing must not assume the host has no colons - an IPv6 literal does."""
    label = remote.origin_label(7, "fe80::1")
    assert remote.parse_origin_label(label) == ("fe80::1", 7)


@pytest.mark.parametrize("label", [None, "", "gpuq:1:proj:normal", "wq-origin:host", "wq-origin:host:abc"])
def test_labels_worker_q_did_not_write_are_not_misread(label):
    assert remote.parse_origin_label(label) is None


# --------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------


def test_a_spec_round_trips_through_json():
    original = spec(ram_gb=4.0, vram_gb=6.0, cpus=2, env={"K": "V"}, passthrough=[".venv"])
    rebuilt = remote.JobSpec.from_dict(json.loads(original.to_json()))
    assert rebuilt.argv == original.argv
    assert rebuilt.env == {"K": "V"}
    assert rebuilt.passthrough == [".venv"]
    assert rebuilt.label == original.label


def test_unknown_fields_in_a_spec_are_dropped_not_fatal():
    """Same rule as the config loader: a newer sender must not break an older
    receiver at the point where it has already accepted the protocol."""
    data = json.loads(spec().to_json())
    data["some_future_field"] = True
    assert remote.JobSpec.from_dict(data).origin_job_id == 42


def test_argv_with_quotes_and_spaces_survives_unchanged():
    """The reason a spec is a file and not a command line.

    These would each need different escaping through ssh, then cmd.exe, then
    typer. Through a JSON file they need none.
    """
    nasty = ["python", "-c", 'print("a b"); x = {1: 2}', "--out=C:\\a b\\c.txt", "*.py", "ünicode"]
    rebuilt = remote.JobSpec.from_dict(json.loads(spec(argv=nasty).to_json()))
    assert rebuilt.argv == nasty


# --------------------------------------------------------------------------
# Submitting
# --------------------------------------------------------------------------


class Scripted:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def __call__(self, node, command, *, timeout=None):
        self.calls.append(command)
        for fragment, (ok, out) in self.replies.items():
            if fragment in command:
                return RemoteResult(ok, 0 if ok else 1, out, "", None if ok else out)
        return RemoteResult(True, 0, "", "")


def _stub_transport(monkeypatch, replies, *, copy_ok=True):
    scripted = Scripted(replies)
    monkeypatch.setattr(remote.nodes, "run_remote", scripted)
    monkeypatch.setattr(
        remote.nodes,
        "copy_to_node",
        lambda n, l, r, **kw: RemoteResult(copy_ok, 0 if copy_ok else 1, "", "", None if copy_ok else "no route"),
    )
    monkeypatch.setattr(remote, "expand_remote", lambda n, text: text.replace("%TEMP%", r"D:\tmp"))
    return scripted


def test_a_submission_returns_the_nodes_own_job_id(monkeypatch):
    scripted = _stub_transport(
        monkeypatch,
        {"_submit-spec": (True, json.dumps({"job_id": 77, "state": "QUEUED"}))},
    )
    result = remote.submit(node(), spec())
    assert result["job_id"] == 77
    assert any("_submit-spec" in c for c in scripted.calls)


def test_a_failed_transfer_never_looks_like_a_queued_job(monkeypatch):
    """The dangerous confusion: believing a job was placed when it was not."""
    _stub_transport(monkeypatch, {}, copy_ok=False)
    with pytest.raises(remote.RemoteJobError, match="spec"):
        remote.submit(node(), spec())


def test_a_reply_without_a_job_id_is_an_error_not_a_silent_zero(monkeypatch):
    _stub_transport(monkeypatch, {"_submit-spec": (True, json.dumps({"error": "nope"}))})
    with pytest.raises(remote.RemoteJobError):
        remote.submit(node(), spec())


def test_banner_lines_before_the_json_are_tolerated(monkeypatch):
    """cmd.exe on the far side may print before the payload."""
    noisy = "some banner\r\n" + json.dumps({"job_id": 5})
    _stub_transport(monkeypatch, {"_submit-spec": (True, noisy)})
    assert remote.submit(node(), spec())["job_id"] == 5


# --------------------------------------------------------------------------
# Reconciling
# --------------------------------------------------------------------------


def test_a_job_can_be_found_by_the_primarys_id_alone(monkeypatch):
    """The path that survives a connection dropped between submit and record.

    If the link fails after the node queued the job but before the primary
    stored the remote id, the id is lost and only the label remains.
    """
    listing = json.dumps({"jobs": [
        {"id": 1, "label": "wq-origin:OTHER:42"},
        {"id": 2, "label": remote.origin_label(42, "PRIMARY")},
        {"id": 3, "label": None},
    ]})
    _stub_transport(monkeypatch, {"list --all": (True, listing)})
    found = remote.find_by_origin(node(), 42, "PRIMARY")
    assert found is not None and found["id"] == 2


def test_no_match_is_none_rather_than_an_exception(monkeypatch):
    _stub_transport(monkeypatch, {"list --all": (True, json.dumps({"jobs": []}))})
    assert remote.find_by_origin(node(), 42, "PRIMARY") is None


def test_an_unreachable_node_raises_rather_than_reporting_no_such_job(monkeypatch):
    """Not found and could-not-ask must never collapse into one answer.

    Treating "unreachable" as "gone" is how a running job gets duplicated.
    """
    _stub_transport(monkeypatch, {"list --all": (False, "connection refused")})
    with pytest.raises(remote.RemoteJobError):
        remote.find_by_origin(node(), 42, "PRIMARY")


def test_show_reports_the_nodes_view(monkeypatch):
    _stub_transport(monkeypatch, {"show 77": (True, json.dumps({"id": 77, "state": "RUNNING"}))})
    assert remote.job(node(), 77)["state"] == "RUNNING"


def test_an_unparseable_reply_is_an_error(monkeypatch):
    _stub_transport(monkeypatch, {"show 77": (True, "not json at all")})
    with pytest.raises(remote.RemoteJobError):
        remote.job(node(), 77)
