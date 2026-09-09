"""Reading the state of a machine, whether or not it is this one.

Phase 1 of docs/multi-node.md. Nothing here dispatches anything; it answers
"what does that machine look like right now", which is the input every later
placement decision needs.

Three things shape this module.

**One call per node per interval.** Measured on this pair, a single
`ssh host cmd` costs 543 ms while the network under it is 19 ms, and five
commands batched into one call cost 630 ms - so the cost is the connection,
not the work. Windows OpenSSH cannot multiplex, so the only lever is making
fewer calls. Hence `_node-report`: one command that returns everything a
placement decision could want, rather than a query per question.

**A report is about the past.** It is stamped with the age of the reading and
nothing is allowed to pretend otherwise. The primary uses a report to
*predict* whether a job would be admitted; the node itself re-runs the same
`resources.admit()` against live numbers before starting anything. A stale
prediction therefore fails safe.

**Version strings cannot detect skew.** Two installs both reported 1.2.0 while
one predated a config key, a schema column and a change to how RAM is measured.
So the handshake compares `NODE_PROTOCOL_VERSION` - bumped only when this
report's shape or meaning changes - and reports the build for diagnosis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from workerq import __version__, host
from workerq.config import LOCAL_NODE, Config, NodeConfig
from workerq.gpu import GpuInfo, query_gpus
from workerq.resources import NodeSnapshot, Reserve, ResourceRequest, cpu_count
from workerq.util import hostname, utcnow_iso
from workerq.winproc import no_window_kwargs

#: Bumped only when the shape or meaning of a node report changes. This, not
#: `__version__`, is what decides whether two installs can talk.
NODE_PROTOCOL_VERSION = 1

_GIB_MIB = 1024.0


@dataclass
class NodeReport:
    """One machine's answer to "what do you look like right now".

    `error` being set means everything else is unusable - a node that could not
    be reached reports no capacity rather than zero capacity, because the two
    must never be confused by a scheduler.
    """

    name: str
    #: True for this machine, which is read directly rather than over SSH.
    is_local: bool = False
    protocol: int | None = None
    version: str | None = None
    #: What the machine calls itself, which need not match its registry name.
    hostname: str | None = None

    host_memory: host.HostMemory | None = None
    gpu: GpuInfo | None = None
    cpus: int | None = None
    commit_ceiling_mib: float | None = None
    reserve: Reserve | None = None

    #: Declared footprints of jobs running there, so the caller can compute
    #: reservations itself rather than trusting a remote sum.
    running: list[ResourceRequest] = field(default_factory=list)
    queued: int = 0
    slots: int | None = None
    daemon_running: bool | None = None

    #: The node's own clock, for detecting skew before it corrupts a runtime.
    remote_time: str | None = None
    #: Seconds since this report was taken, filled by the cache.
    age_seconds: float = 0.0
    #: Wall time the report itself took, useful for tuning the poll interval.
    latency_seconds: float | None = None

    error: str | None = None

    @property
    def online(self) -> bool:
        return self.error is None

    def snapshot(self, config: Config) -> NodeSnapshot | None:
        """The value `resources.admit()` needs to judge this machine.

        Returns None when the report is unusable, so a caller cannot
        accidentally admit against a machine it never heard from.
        """
        if not self.online or self.host_memory is None:
            return None
        return NodeSnapshot(
            mem=self.host_memory,
            gpu=self.gpu,
            cpus=self.cpus or 1,
            commit_ceiling_mib=self.commit_ceiling_mib,
            reserve=self.reserve or Reserve.from_config(config),
            name=self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_local": self.is_local,
            "protocol": self.protocol,
            "version": self.version,
            "hostname": self.hostname,
            "online": self.online,
            "error": self.error,
            "cpus": self.cpus,
            "commit_ceiling_mib": self.commit_ceiling_mib,
            "host": self.host_memory.to_dict() if self.host_memory else None,
            "gpu": self.gpu.to_dict() if self.gpu is not None else None,
            "reserve": self.reserve.to_dict() if self.reserve else None,
            "running": [r.to_dict() for r in self.running],
            "queued": self.queued,
            "slots": self.slots,
            "daemon_running": self.daemon_running,
            "remote_time": self.remote_time,
            "age_seconds": self.age_seconds,
            "latency_seconds": self.latency_seconds,
        }


# --------------------------------------------------------------------------
# Building a report for this machine
# --------------------------------------------------------------------------


def local_payload(config: Config, service: Any | None = None) -> dict[str, Any]:
    """The JSON `workerq _node-report` prints.

    Kept separate from `local_report` so the wire format has exactly one
    definition, used both by the node answering and by the caller parsing.
    """
    mem = host.memory()
    gpu = query_gpus(include_processes=False)
    reserve = Reserve.from_config(config)
    running: list[dict[str, Any]] = []
    queued = 0
    slots: int | None = None
    daemon: bool | None = None

    # Every lookup below is best-effort. A node that can describe its memory
    # but not its queue is still worth reporting; refusing to answer at all
    # would make it indistinguishable from one that is switched off.
    if service is not None:
        try:
            reserve = service.backend.get_reserve() or reserve
        except Exception:
            pass
        try:
            summary = service.status_summary()
            slots = summary.get("backend_slots")
            daemon = summary.get("daemon_running")
        except Exception:
            pass
        for state, sink in (("RUNNING", running), ("QUEUED", None)):
            try:
                jobs = service.list_jobs(state=state, limit=500, refresh=False)
            except Exception:
                continue
            if sink is None:
                queued = len(jobs)
                continue
            for job in jobs:
                sink.append(
                    {
                        "ram_mib": float(getattr(job, "requested_ram_mib", None) or 0.0),
                        "vram_mib": float(getattr(job, "requested_vram_mib", None) or 0.0),
                        "cpus": int(getattr(job, "requested_cpus", None) or 0),
                        "gpu_count": int(getattr(job, "requested_gpu_count", None) or 0),
                    }
                )

    return {
        "protocol": NODE_PROTOCOL_VERSION,
        "version": __version__,
        "hostname": hostname(),
        "time": utcnow_iso(),
        "cpus": cpu_count(),
        "commit_ceiling_mib": host.commit_ceiling_mib(mem),
        "host": mem.to_dict(),
        "gpu": gpu.to_dict(),
        "reserve": reserve.to_dict(),
        "running": running,
        "queued": queued,
        "slots": slots,
        "daemon_running": daemon,
    }


def local_report(config: Config, service: Any | None = None) -> NodeReport:
    started = time.monotonic()
    payload = local_payload(config, service)
    report = _parse_payload(LOCAL_NODE, payload)
    report.is_local = True
    report.latency_seconds = time.monotonic() - started
    return report


# --------------------------------------------------------------------------
# Reading a remote machine
# --------------------------------------------------------------------------


def _parse_payload(name: str, payload: dict[str, Any]) -> NodeReport:
    mem = host.HostMemory(**{
        k: v for k, v in (payload.get("host") or {}).items()
        if k in {"total_mib", "available_mib", "commit_used_mib", "commit_limit_mib", "error"}
    })
    gpu_raw = payload.get("gpu")
    gpu = GpuInfo.from_dict(gpu_raw) if gpu_raw else None
    res_raw = payload.get("reserve") or {}
    reserve = Reserve(
        ram_mib=float(res_raw.get("ram_mib", 0.0)),
        vram_mib=float(res_raw.get("vram_mib", 0.0)),
        cpus=int(res_raw.get("cpus", 0)),
        label=res_raw.get("label"),
        expires_at=res_raw.get("expires_at"),
    ) if res_raw else None

    return NodeReport(
        name=name,
        protocol=payload.get("protocol"),
        version=payload.get("version"),
        hostname=payload.get("hostname"),
        host_memory=mem,
        gpu=gpu,
        cpus=payload.get("cpus"),
        commit_ceiling_mib=payload.get("commit_ceiling_mib"),
        reserve=reserve,
        running=[
            ResourceRequest(
                ram_mib=float(r.get("ram_mib", 0.0)),
                vram_mib=float(r.get("vram_mib", 0.0)),
                cpus=int(r.get("cpus", 0)),
                gpu_count=int(r.get("gpu_count", 0)),
            )
            for r in payload.get("running", [])
        ],
        queued=int(payload.get("queued", 0) or 0),
        slots=payload.get("slots"),
        daemon_running=payload.get("daemon_running"),
        remote_time=payload.get("time"),
    )


def ssh_command(node: NodeConfig, remote: str) -> list[str]:
    """The ssh invocation for one remote command.

    `BatchMode=yes` is not a nicety: without it a missing key turns a poll into
    a password prompt on a background thread, and the dispatcher hangs rather
    than reporting a node offline.
    """
    ssh = shutil.which("ssh") or "ssh"
    argv = [ssh, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    argv += ["-o", f"ConnectTimeout={max(1, int(node.timeout_seconds))}"]
    if node.port != 22:
        argv += ["-p", str(node.port)]
    argv += [node.target, remote]
    return argv


#: Box-drawing and quadrant characters. Rich frames its errors in these, so the
#: last line of a failed remote command is usually a border rather than a
#: reason - which is exactly the line a naive `splitlines()[-1]` picks.
_BOX = set("─━│┃┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬╱╲╳▀▄█▌▐░▒▓")


def _explain_failure(code: int, stderr: str | None, stdout: str | None) -> str:
    """Turn a failed remote command into something worth reading.

    A node is "offline" for several quite different reasons, and the whole
    value of the node list is telling them apart: a machine that is switched
    off needs a different response from one running a worker-q too old to
    answer.
    """
    text = ((stderr or "") + "\n" + (stdout or "")).strip()
    lowered = text.lower()

    if "no such command" in lowered or "_node-report" in lowered and "usage" in lowered:
        return (
            "remote worker-q does not have `_node-report` - it predates node "
            "support; upgrade that machine"
        )
    if "permission denied" in lowered or "publickey" in lowered:
        return "ssh rejected the key (check administrators_authorized_keys and its ACL)"
    if "could not resolve" in lowered or "name or service not known" in lowered:
        return "hostname did not resolve"
    if "connection refused" in lowered:
        return "connection refused - is sshd running there?"
    if "connection timed out" in lowered or "timed out" in lowered:
        return "connection timed out - machine off, or asleep"
    if "is not recognized" in lowered or "cannot find the path" in lowered:
        return "workerq was not found at the configured path on that machine"

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or all(ch in _BOX or ch.isspace() for ch in stripped):
            continue
        return stripped[:200]
    return f"ssh exited {code}"


def remote_report(node: NodeConfig) -> NodeReport:
    """Ask one machine for its state. Never raises."""
    started = time.monotonic()
    argv = ssh_command(node, f"{node.workerq_path} _node-report --json")
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=node.timeout_seconds,
            encoding="utf-8",
            errors="replace",
            **no_window_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return NodeReport(
            name=node.name,
            error=f"timed out after {node.timeout_seconds:.0f}s",
            latency_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return NodeReport(
            name=node.name,
            error=f"{type(exc).__name__}: {exc}",
            latency_seconds=time.monotonic() - started,
        )

    latency = time.monotonic() - started
    if proc.returncode != 0:
        return NodeReport(
            name=node.name,
            error=_explain_failure(proc.returncode, proc.stderr, proc.stdout),
            latency_seconds=latency,
        )

    # cmd.exe on the far side may prepend banner lines; take the JSON object.
    text = (proc.stdout or "").strip()
    start = text.find("{")
    if start < 0:
        return NodeReport(name=node.name, error="no JSON in reply", latency_seconds=latency)
    try:
        payload = json.loads(text[start:])
    except json.JSONDecodeError as exc:
        return NodeReport(name=node.name, error=f"unparseable reply: {exc}", latency_seconds=latency)

    report = _parse_payload(node.name, payload)
    report.latency_seconds = latency
    return report


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


class ReportCache:
    """Holds the last report per node, with its age.

    The age matters more than the report. A placement decision records how
    stale its input was, so a misplacement can be explained afterwards instead
    of being argued about.
    """

    def __init__(self) -> None:
        self._reports: dict[str, tuple[float, NodeReport]] = {}

    def get(self, node: NodeConfig, *, max_age: float | None = None) -> NodeReport:
        max_age = node.poll_interval_seconds if max_age is None else max_age
        now = time.monotonic()
        cached = self._reports.get(node.name)
        if cached is not None and now - cached[0] < max_age:
            report = cached[1]
            report.age_seconds = now - cached[0]
            return report
        report = remote_report(node)
        self._reports[node.name] = (now, report)
        report.age_seconds = 0.0
        return report

    def invalidate(self, name: str) -> None:
        self._reports.pop(name, None)


# --------------------------------------------------------------------------
# Compatibility
# --------------------------------------------------------------------------


def compatibility(local: NodeReport, remote: NodeReport) -> tuple[bool, str | None]:
    """Can these two installs talk?

    Only the protocol number may refuse. A differing `__version__` is reported
    but not fatal, because it is the check that already failed to notice a real
    skew - both machines said 1.2.0 while one lacked a config key, a schema
    column and the current way of measuring RAM.
    """
    if not remote.online:
        return False, remote.error
    if remote.protocol is None:
        return False, "node did not report a protocol version (worker-q too old)"
    if remote.protocol != local.protocol:
        return False, (
            f"protocol {remote.protocol} != {local.protocol}; upgrade both "
            "machines to the same worker-q before dispatching"
        )
    return True, None
