"""Running one job on another machine, and following it there.

Phase 4 of docs/multi-node.md. The node runs a complete worker-q, so this is a
*client* of that install, not a second scheduler: it submits, asks, cancels and
retrieves. Every decision about whether the job may start is still made by the
node's own `resources.admit()` against its own live numbers.

**A job spec crosses as a file, never as a command line.** worker-q already
made this decision once, for the runner: `workerq _run <id>` takes only an id
and reads the argv back from the database as JSON, because on Windows every
`Popen(list)` is joined by `list2cmdline` and re-parsed by the child's C
runtime, and that round trip has real edge cases. Dispatching remotely stacks
three more layers on top - ssh, `cmd.exe`, and typer - so a command line
carrying arbitrary user argv would be a quoting bug waiting to happen. The spec
is written as JSON, copied, and read by `workerq _submit-spec` on the far side.

**The primary's job id travels in the label.** The node assigns its own id, so
the only durable link between the two records is a label the node stores and
`find_by_label` can look up. That is what makes reconciliation after a dropped
connection possible, and it is why the label format is a constant here rather
than a string built at three call sites.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from workerq import nodes
from workerq.config import NodeConfig
from workerq.staging import _q, expand_remote
from workerq.util import hostname

#: Bumped with `nodes.NODE_PROTOCOL_VERSION` when the spec's shape changes.
SPEC_PROTOCOL = 1

REMOTE_SPEC_DIR = "%TEMP%\\workerq-specs"

#: How a remote job records which primary job it is. Parsed by
#: `parse_origin_label`, never by hand.
ORIGIN_LABEL_PREFIX = "wq-origin"


class RemoteJobError(RuntimeError):
    """A remote job operation failed. Never means "the job is gone"."""


def origin_label(job_id: int, host: str | None = None) -> str:
    return f"{ORIGIN_LABEL_PREFIX}:{host or hostname()}:{job_id}"


def parse_origin_label(label: str | None) -> tuple[str, int] | None:
    """(origin host, origin job id) for a label this module wrote."""
    if not label or not label.startswith(ORIGIN_LABEL_PREFIX + ":"):
        return None
    parts = label.split(":")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    return ":".join(parts[1:-1]), int(parts[-1])


@dataclass
class JobSpec:
    """Everything the far side needs to reproduce one submission.

    `cwd` is the worktree `staging.ship_snapshot` materialised, and the node is
    told to treat it as live: the tree is *already* a frozen snapshot, so
    snapshotting it again would freeze a snapshot and lose the commit identity
    the primary recorded.
    """

    project: str
    argv: list[str]
    cwd: str
    origin_job_id: int
    origin_host: str = ""
    protocol: int = SPEC_PROTOCOL

    ram_gb: float | None = None
    vram_gb: float | None = None
    cpus: int | None = None
    gpus: int | None = None
    preemptible: bool | None = None
    share_gpu: bool = False
    priority: str | None = None
    shell: str | None = None
    describe: str | None = None
    blocks: str | None = None
    eta_seconds: float | None = None
    env: dict[str, str] = field(default_factory=dict)
    passthrough: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.origin_host:
            self.origin_host = hostname()

    @property
    def label(self) -> str:
        return origin_label(self.origin_job_id, self.origin_host)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# Submitting
# --------------------------------------------------------------------------


def submit(node: NodeConfig, spec: JobSpec) -> dict[str, Any]:
    """Queue `spec` on `node` and return the node's own submission record.

    Two calls: one to copy the spec, one to run it. The node's admission
    control then decides, on its own live numbers, whether it may start - this
    function only places the work there.
    """
    spec_dir = expand_remote(node, REMOTE_SPEC_DIR)
    remote_spec = f"{spec_dir}\\job-{spec.origin_job_id:06d}.json"

    with tempfile.TemporaryDirectory(prefix="workerq-spec-") as tmp:
        local = Path(tmp) / "spec.json"
        local.write_text(spec.to_json(), encoding="utf-8")
        prep = nodes.run_remote(node, f"mkdir {_q(spec_dir)} 2>nul & exit /b 0")
        if not prep.ok:
            raise RemoteJobError(f"could not prepare spec dir on {node.name}: {prep.error}")
        sent = nodes.copy_to_node(node, local, remote_spec)
        if not sent.ok:
            raise RemoteJobError(f"could not send the job spec to {node.name}: {sent.error}")

    result = nodes.run_remote(
        node,
        f"{node.workerq_path} _submit-spec {_q(remote_spec)} --json",
        timeout=max(node.timeout_seconds, 180.0),
    )
    if not result.ok:
        raise RemoteJobError(f"remote submit failed on {node.name}: {result.error}")

    payload = _json_object(result.stdout)
    if payload is None:
        raise RemoteJobError(f"remote submit returned no JSON: {result.out[:200]}")
    if "job_id" not in payload:
        raise RemoteJobError(f"remote submit returned {payload}")
    return payload


# --------------------------------------------------------------------------
# Following
# --------------------------------------------------------------------------


def _json_object(text: str) -> dict[str, Any] | None:
    """Extract the JSON object from a reply that may carry banner lines."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        value = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def job(node: NodeConfig, remote_id: int) -> dict[str, Any]:
    """The node's record for one job."""
    result = nodes.run_remote(node, f"{node.workerq_path} show {remote_id} --json")
    if not result.ok:
        raise RemoteJobError(result.error or "remote show failed")
    payload = _json_object(result.stdout)
    if payload is None:
        raise RemoteJobError(f"unparseable reply: {result.out[:200]}")
    return payload


def find_by_origin(node: NodeConfig, job_id: int, host: str | None = None) -> dict[str, Any] | None:
    """Locate a job on the node by the primary's id.

    This is the reconciliation path. After a dropped connection the primary may
    hold a remote id it cannot trust, or none at all if the link failed between
    submitting and recording - the label is written by the node at submit time,
    so it survives both.
    """
    wanted = origin_label(job_id, host)
    result = nodes.run_remote(node, f"{node.workerq_path} list --all --limit 500 --json")
    if not result.ok:
        raise RemoteJobError(result.error or "remote list failed")
    payload = _json_object(result.stdout)
    if payload is None:
        return None
    for entry in payload.get("jobs", []):
        if entry.get("label") == wanted:
            return entry
    return None


def cancel(node: NodeConfig, remote_id: int, *, force: bool = False) -> bool:
    flag = " --force" if force else ""
    result = nodes.run_remote(
        node,
        f"{node.workerq_path} cancel {remote_id}{flag} --json",
        timeout=max(node.timeout_seconds, 120.0),
    )
    return result.ok


def log_tail(node: NodeConfig, remote_id: int, lines: int = 100) -> str:
    result = nodes.run_remote(
        node, f"{node.workerq_path} logs {remote_id} --tail {lines}"
    )
    if not result.ok:
        raise RemoteJobError(result.error or "remote logs failed")
    return result.stdout


def fetch_log(node: NodeConfig, remote_log_path: str, dest: Path) -> bool:
    """Bring a finished job's log home.

    Small, and it is the difference between a job's output surviving the worker
    being switched off and not. Failure is not fatal: the log still exists on
    the node.
    """
    from workerq import nodes as nodemod

    scp = nodemod.shutil.which("scp") or "scp"
    argv = [scp, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if node.port != 22:
        argv += ["-P", str(node.port)]
    argv += [f"{node.target}:{remote_log_path}", str(dest)]
    try:
        proc = nodemod.subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=300,
            **nodemod.no_window_kwargs(),
        )
    except Exception:
        return False
    return proc.returncode == 0 and dest.exists()
