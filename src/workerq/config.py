"""Configuration loading and validation.

Precedence (spec section 10):  CLI flag > GPUQ_* env > config.toml > default.
CLI flags are applied by the caller via `Config.with_overrides`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from workerq.util import atomic_write_text, ensure_dir, expand_path

VALID_PRIORITIES = ("critical", "high", "normal", "low")

#: Reserved node name meaning "this machine". Never a registry entry.
LOCAL_NODE = "local"
VALID_SNAPSHOT_MODES = ("git", "none", "copy")


class ConfigError(ValueError):
    """Raised for malformed or invalid configuration."""


# --------------------------------------------------------------------------
# Profile-aware default locations
# --------------------------------------------------------------------------


def active_profile() -> str | None:
    p = os.environ.get("GPUQ_PROFILE", "").strip()
    return p or None


def default_state_dir(profile: str | None = None) -> Path:
    profile = profile if profile is not None else active_profile()
    suffix = f"-{profile}" if profile else ""
    return expand_path(f"~/.local/state/gpuq{suffix}")


def default_config_path(profile: str | None = None) -> Path:
    override = os.environ.get("GPUQ_CONFIG_FILE") or os.environ.get("GPUQ_CONFIG")
    if override:
        return expand_path(override)
    profile = profile if profile is not None else active_profile()
    name = f"config-{profile}.toml" if profile else "config.toml"
    return expand_path(f"~/.config/gpuq/{name}")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass
class CoreConfig:
    state_dir: str = ""  # filled in Config.__post_init__
    #: Ceiling, not the scheduler. Admission control decides what actually
    #: runs: a job starts only when its declared RAM/VRAM/CPU fits alongside
    #: what is already running. This exists so a bug cannot launch twenty
    #: processes, and so the machine has a bound even with enforcement off.
    max_concurrent_jobs: int = 4
    default_priority: str = "normal"
    snapshot_mode: str = "git"
    cleanup_successful_snapshots_after_days: int = 7
    cleanup_failed_snapshots_after_days: int = 14
    cancel_grace_seconds: int = 15


@dataclass
class GpuConfig:
    default_gpu_count: int = 1
    free_memory_threshold_percent: int = 90
    exclusive_by_default: bool = True


@dataclass
class BackendConfig:
    # Canonical section for the execution backend. A `[task_spooler]` table in a
    # config file is accepted as an alias for spec compatibility.
    name: str = "local_dispatcher"
    max_finished: int = 1000
    poll_interval_seconds: float = 0.25
    daemon_heartbeat_stale_seconds: int = 30


@dataclass
class ResourcesConfig:
    """Admission control for *any* heavy workload, not just GPU work.

    A job is only started when its declared RAM/VRAM/CPU request fits in the
    headroom that is actually free right now, after subtracting what already
    running gpuq jobs have reserved. That is what lets small jobs run in
    parallel while big ones serialise, and what stops the box being pushed
    into swap or commit exhaustion by work gpuq did not start.
    """

    #: Master switch. Off means priority + slot count are the only limits.
    enforce: bool = True

    #: Assumed request for jobs that declare nothing, so undeclared work is
    #: never treated as free.
    default_ram_gb: float = 4.0
    default_vram_gb: float = 0.0
    default_cpus: int = 1

    #: Never handed out - headroom for the OS, editors and the agents.
    reserve_ram_gb: float = 8.0
    reserve_vram_gb: float = 1.0
    reserve_cpus: int = 2

    #: Hard stops.
    #:
    #: Commit charge is a real failure mode - Windows refuses allocations once
    #: the system commit limit is reached - but the limit is not fixed: a
    #: system-managed pagefile grows on demand. A high percentage on its own
    #: therefore says little. Measured on a live workstation, half of all
    #: samples taken while a job ran exceeded 88% commit while physical RAM
    #: stayed around 42% free, and the limit itself drifted from 81 to 94 GiB.
    #: Blocking there stops work for no reason.
    #:
    #: Measured further: the limit demonstrably *grows* as commit rises (81.9
    #: -> 87.5 GiB average across the bands on one workstation), and even in
    #: the 98-101% band physical RAM averaged 41.5% free and never ran out. So
    #: a high commit charge on a machine with a system-managed pagefile is not
    #: a danger signal - it is the pagefile doing its job.
    #:
    #: The outright stop therefore sits where allocations genuinely start
    #: failing rather than where growth is still catching up. Below it, commit
    #: only blocks when physical memory is short too - the combination that
    #: actually precedes thrashing, and the one worth refusing work over.
    max_commit_percent: int = 99
    #: Safety margin, as a percentage of physical RAM, kept free of the commit
    #: limit so the OS and unqueued work can still allocate. The gate that
    #: matters is absolute: a job's RAM + VRAM must fit the remaining commit
    #: headroom, because on Windows the GPU driver backs video memory with
    #: system commit and the limit stops rising once the pagefile hits its
    #: configured maximum.
    commit_headroom_percent: int = 5
    #: Retained so an existing config file keeps loading; superseded by the
    #: headroom check above, which fires on the failure these never caught.
    commit_soft_percent: int = 88
    commit_soft_free_percent: int = 25
    #: Physical memory floor. This is the primary gate: it is the thing whose
    #: exhaustion actually freezes a desktop.
    min_host_free_percent: int = 10

    #: How long a job may sit blocked before gpuq says so loudly.
    blocked_warning_seconds: int = 900


@dataclass
class SchedulingConfig:
    """How the dispatcher walks the queue.

    Strict head-of-line order is safest but wastes the machine: one oversized
    job at the head parks a queue full of work that would fit alongside what is
    already running. Backfill looks past it - bounded, so a large job cannot be
    deferred indefinitely by a stream of small ones.
    """

    #: Let jobs further down the queue start when the head cannot. Off restores
    #: strict head-of-line order.
    backfill: bool = True
    #: How far past a blocked job to look in one tick. Bounds the work done per
    #: tick and keeps the queue roughly in priority order.
    backfill_max_skip: int = 8
    #: Once the blocked job at the head has waited this long, backfill stops and
    #: the queue drains until it can run. This is the starvation guard: it caps
    #: how long backfill may delay a job that cannot be packed.
    backfill_head_wait_seconds: int = 900
    #: Let worker-q choose which machine a job runs on, once a node is
    #: registered. Off pins everything here unless `--node` says otherwise,
    #: which is the escape hatch if placement ever misbehaves.
    auto_placement: bool = True

    #: How long the queue may stay held for one job before backfilling resumes.
    #: Draining the machine only helps when queue pressure is what keeps the head
    #: out. When it is instead waiting on a job with hours left to run, an
    #: unbounded hold idles the whole machine and buys the head nothing, so the
    #: guard gives up after this long and lets work that fits through again.
    backfill_max_hold_seconds: int = 1800

    #: Run job processes below normal priority so the desktop stays responsive
    #: when several of them share the CPUs. Scheduling priority only - it does
    #: not cap CPU or memory, and a job alone on an idle machine is unaffected.
    background_priority: bool = True

    #: Runtime pressure guard. Admission is a prediction; this is what happens
    #: when the prediction is wrong - an under-declared job, a foreign
    #: workload, a game launched without claiming a reserve. Sustained physical
    #: memory pressure stops new work and then displaces the newest
    #: preemptible job. Two thresholds, so it does not oscillate.
    pressure_free_percent: int = 12
    pressure_recover_percent: int = 20
    #: Consecutive 10s samples below the floor before acting.
    pressure_samples: int = 3


@dataclass
class PreemptionConfig:
    """When a higher-priority job may displace a running one.

    Preemption requeues the displaced job, which means its command runs again
    from the start. That is destructive for anything not resumable, so it is
    opt-in per job by default and hedged with guards against thrashing and
    starvation.
    """

    enabled: bool = True
    #: Only displace jobs that declared `--preemptible`. Turning this off lets
    #: any lower-priority job be displaced, which will lose work.
    require_opt_in: bool = True
    #: A job must have run this long before it can be displaced, so a burst of
    #: high-priority submissions cannot leave nothing making progress.
    min_runtime_seconds: int = 60
    #: After this many displacements a job stops being a candidate, so it
    #: cannot be starved forever.
    max_preemptions: int = 3
    #: Time allowed for the job to stop cleanly before its tree is killed.
    grace_seconds: int = 30


@dataclass
class GamingConfig:
    """Headroom to claim when you want the machine back for something else.

    A flat section on purpose: the config writer regenerates the file from
    these dataclasses, so a nested table of named presets would be deleted by
    the next `config set`. One preset, reachable from `workerq top` with a
    single key, covers the case people actually have.
    """

    ram_gb: float = 24.0
    vram_gb: float = 24.0
    cpus: int = 8


@dataclass
class ClaudeConfig:
    install_user_policy: bool = True
    hide_cuda_in_safe_launcher: bool = False


@dataclass
class NodeConfig:
    """A second machine worker-q can see, and later dispatch to.

    Registered as repeated `[[node]]` tables rather than a fixed section,
    because there is no sensible default node and the count is not known in
    advance. `local` is reserved: it always means this host, so a registry
    entry can never shadow it.

    `address` is deliberately not called `host` - `workerq.host` is the local
    memory-inspection module, and a job row already has a `host` column meaning
    the machine that submitted it. Three different senses of the word in one
    codebase is one too many.
    """

    name: str
    address: str
    user: str | None = None
    port: int = 22
    #: Remote polling is a whole SSH connection - measured at ~543 ms on this
    #: pair, against 19 ms of network - so this is seconds, not the 0.25 s
    #: local dispatch tick. See docs/multi-node.md 5.3.
    poll_interval_seconds: float = 3.0
    #: Seconds to wait for a node report before treating the node as offline.
    timeout_seconds: float = 20.0
    #: Directory on that machine holding one clone per project. A repo's own
    #: path need not match the primary's - snapshots are materialised under
    #: worker-q's state directory and `execution_cwd` is relative to the repo
    #: root - so only the parent has to be agreed.
    repo_root: str = r"%USERPROFILE%\Documents"
    #: Path to `workerq.exe` on that machine. The default resolves through the
    #: remote PATH, which a non-interactive SSH session does not always have.
    workerq_path: str = "%USERPROFILE%\\.local\\bin\\workerq.exe"
    #: Set false to keep a node registered but ignored, without deleting it.
    enabled: bool = True

    @property
    def target(self) -> str:
        """The `user@address` (or bare address) an ssh command takes."""
        return f"{self.user}@{self.address}" if self.user else self.address


@dataclass
class Config:
    core: CoreConfig = field(default_factory=CoreConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    gaming: GamingConfig = field(default_factory=GamingConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)

    #: Registered remote machines, in `[[node]]` order. Empty means worker-q
    #: behaves exactly as a single-machine install.
    nodes: list[NodeConfig] = field(default_factory=list)

    #: Path the config was loaded from (may not exist yet).
    source_path: Path | None = None
    profile: str | None = None

    def __post_init__(self) -> None:
        if not self.core.state_dir:
            self.core.state_dir = str(default_state_dir(self.profile))
        self.validate()

    # -- derived paths ----------------------------------------------------
    @property
    def state_dir(self) -> Path:
        return expand_path(self.core.state_dir)

    @property
    def db_path(self) -> Path:
        return self.state_dir / "gpuq.sqlite3"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def snapshots_dir(self) -> Path:
        return self.state_dir / "snapshots"

    @property
    def run_dir(self) -> Path:
        return self.state_dir / "run"

    @property
    def tmp_dir(self) -> Path:
        return self.state_dir / "tmp"

    @property
    def backend_dir(self) -> Path:
        return self.state_dir / "backend"

    @property
    def jobs_dir(self) -> Path:
        return self.state_dir / "jobs"

    def all_dirs(self) -> list[Path]:
        return [
            self.state_dir,
            self.logs_dir,
            self.snapshots_dir,
            self.run_dir,
            self.tmp_dir,
            self.backend_dir,
            self.jobs_dir,
        ]

    def ensure_dirs(self) -> None:
        for d in self.all_dirs():
            ensure_dir(d)

    def job_dir(self, job_id: int) -> Path:
        return self.jobs_dir / str(job_id)

    def log_name(self, job_id: int) -> str:
        return f"job-{job_id:06d}.log"

    def log_path(self, job_id: int) -> Path:
        return self.logs_dir / self.log_name(job_id)

    # -- validation -------------------------------------------------------
    def validate(self) -> None:
        c, g, b = self.core, self.gpu, self.backend
        if isinstance(c.max_concurrent_jobs, bool) or not isinstance(c.max_concurrent_jobs, int):
            raise ConfigError("core.max_concurrent_jobs must be an integer")
        if c.max_concurrent_jobs < 1:
            raise ConfigError("core.max_concurrent_jobs must be >= 1")
        if c.default_priority not in VALID_PRIORITIES:
            raise ConfigError(
                "core.default_priority must be one of " + ", ".join(VALID_PRIORITIES)
            )
        if c.snapshot_mode not in VALID_SNAPSHOT_MODES:
            raise ConfigError(
                "core.snapshot_mode must be one of " + ", ".join(VALID_SNAPSHOT_MODES)
            )
        for name in (
            "cleanup_successful_snapshots_after_days",
            "cleanup_failed_snapshots_after_days",
        ):
            if getattr(c, name) < 0:
                raise ConfigError(f"core.{name} must be >= 0")
        if c.cancel_grace_seconds < 0:
            raise ConfigError("core.cancel_grace_seconds must be >= 0")
        if isinstance(g.free_memory_threshold_percent, bool) or not isinstance(
            g.free_memory_threshold_percent, int
        ):
            raise ConfigError("gpu.free_memory_threshold_percent must be an integer")
        if not 0 <= g.free_memory_threshold_percent <= 100:
            raise ConfigError("gpu.free_memory_threshold_percent must be between 0 and 100")
        if g.default_gpu_count < 0:
            raise ConfigError("gpu.default_gpu_count must be >= 0")
        if b.max_finished < 1:
            raise ConfigError("backend.max_finished must be >= 1")
        if b.poll_interval_seconds <= 0:
            raise ConfigError("backend.poll_interval_seconds must be > 0")
        pre = self.preemption
        for name in ("min_runtime_seconds", "max_preemptions", "grace_seconds"):
            if getattr(pre, name) < 0:
                raise ConfigError(f"preemption.{name} must be >= 0")
        sch = self.scheduling
        if sch.backfill_max_skip < 0:
            raise ConfigError("scheduling.backfill_max_skip must be >= 0")
        if sch.backfill_head_wait_seconds < 0:
            raise ConfigError("scheduling.backfill_head_wait_seconds must be >= 0")
        if sch.backfill_max_hold_seconds < 0:
            raise ConfigError("scheduling.backfill_max_hold_seconds must be >= 0")
        for name in ("pressure_free_percent", "pressure_recover_percent"):
            if not 0 <= getattr(sch, name) <= 100:
                raise ConfigError(f"scheduling.{name} must be between 0 and 100")
        if sch.pressure_recover_percent < sch.pressure_free_percent:
            raise ConfigError(
                "scheduling.pressure_recover_percent must be >= pressure_free_percent"
            )
        if sch.pressure_samples < 1:
            raise ConfigError("scheduling.pressure_samples must be >= 1")
        r = self.resources
        for name in ("default_ram_gb", "default_vram_gb", "reserve_ram_gb", "reserve_vram_gb"):
            if getattr(r, name) < 0:
                raise ConfigError(f"resources.{name} must be >= 0")
        for name in ("default_cpus", "reserve_cpus"):
            if getattr(r, name) < 0:
                raise ConfigError(f"resources.{name} must be >= 0")
        for name in ("max_commit_percent", "min_host_free_percent"):
            value = getattr(r, name)
            if not 0 <= value <= 100:
                raise ConfigError(f"resources.{name} must be between 0 and 100")

        seen: set[str] = set()
        for node in self.nodes:
            if not node.name:
                raise ConfigError("every [[node]] needs a name")
            if node.name == LOCAL_NODE:
                raise ConfigError(
                    f"{LOCAL_NODE!r} is reserved for this machine and cannot be "
                    "used as a node name"
                )
            if node.name in seen:
                raise ConfigError(f"duplicate node name: {node.name}")
            seen.add(node.name)
            if not node.address:
                raise ConfigError(f"node {node.name!r} needs an address")
            if not 0 < node.port < 65536:
                raise ConfigError(f"node {node.name!r} has an invalid port")
            if node.poll_interval_seconds <= 0:
                raise ConfigError(f"node {node.name!r} poll_interval_seconds must be > 0")
            if node.timeout_seconds <= 0:
                raise ConfigError(f"node {node.name!r} timeout_seconds must be > 0")

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "core": asdict(self.core),
            "gpu": asdict(self.gpu),
            "backend": asdict(self.backend),
            "resources": asdict(self.resources),
            "scheduling": asdict(self.scheduling),
            "preemption": asdict(self.preemption),
            "gaming": asdict(self.gaming),
            "claude": asdict(self.claude),
        }

    def node(self, name: str) -> NodeConfig | None:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def active_nodes(self) -> list[NodeConfig]:
        return [n for n in self.nodes if n.enabled]

    def to_toml(self) -> str:
        lines: list[str] = [
            "# worker-q configuration",
            "# Precedence: CLI flag > GPUQ_* env var > this file > built-in default",
            "",
        ]
        for section, values in self.to_dict().items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                lines.append(f"{key} = {_toml_value(value)}")
            lines.append("")
        for node in self.nodes:
            lines.append("[[node]]")
            for key, value in asdict(node).items():
                if value is None:
                    continue
                lines.append(f"{key} = {_toml_value(value)}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: Path | None = None) -> Path:
        target = path or self.source_path or default_config_path(self.profile)
        ensure_dir(target.parent)
        atomic_write_text(target, self.to_toml())
        self.source_path = target
        return target

    def with_overrides(self, **overrides: Any) -> Config:
        """Return a copy with dotted-key overrides applied (the CLI layer)."""
        data = self.to_dict()
        for dotted, value in overrides.items():
            if value is None:
                continue
            _set_dotted(data, dotted.replace("__", "."), value, coerce=True)
        return _from_dict(
            data, source_path=self.source_path, profile=self.profile, nodes=self.nodes
        )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return '"' + text + '"'


# --------------------------------------------------------------------------
# Dotted-key access
# --------------------------------------------------------------------------

_SECTION_TYPES: dict[str, type] = {
    "core": CoreConfig,
    "gpu": GpuConfig,
    "backend": BackendConfig,
    "resources": ResourcesConfig,
    "scheduling": SchedulingConfig,
    "preemption": PreemptionConfig,
    "gaming": GamingConfig,
    "claude": ClaudeConfig,
}

# Aliases keep spec-shaped `[task_spooler]` config files working.
_KEY_ALIASES = {
    "task_spooler.max_finished": "backend.max_finished",
    "task_spooler.binary": "backend.name",
}


def normalize_key(dotted: str) -> str:
    return _KEY_ALIASES.get(dotted.strip(), dotted.strip())


_TYPE_BY_NAME: dict[str, type] = {"int": int, "float": float, "bool": bool, "str": str}


def _field_type(section: str, key: str) -> type | None:
    cls = _SECTION_TYPES.get(section)
    if cls is None:
        return None
    for f in fields(cls):
        if f.name == key:
            name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "str")
            return _TYPE_BY_NAME.get(name, str)
    return None


def coerce_value(section: str, key: str, value: Any) -> Any:
    target = _field_type(section, key)
    if target is None:
        raise ConfigError(f"unknown configuration key: {section}.{key}")
    if isinstance(value, str):
        text = value.strip()
        if target is bool:
            low = text.lower()
            if low in ("1", "true", "yes", "on"):
                return True
            if low in ("0", "false", "no", "off"):
                return False
            raise ConfigError(f"{section}.{key} must be a boolean, got {value!r}")
        if target is int:
            try:
                return int(text, 10)
            except ValueError:
                raise ConfigError(
                    f"{section}.{key} must be an integer, got {value!r}"
                ) from None
        if target is float:
            try:
                return float(text)
            except ValueError:
                raise ConfigError(f"{section}.{key} must be a number, got {value!r}") from None
        return text
    if target is bool:
        return bool(value)
    if target is int and not isinstance(value, bool):
        return int(value)
    if target is float:
        return float(value)
    return value


def _set_dotted(data: dict[str, Any], dotted: str, value: Any, *, coerce: bool = False) -> None:
    dotted = normalize_key(dotted)
    if "." not in dotted:
        raise ConfigError(f"configuration key must be section.key, got {dotted!r}")
    section, key = dotted.split(".", 1)
    if section not in data:
        raise ConfigError(f"unknown configuration section: {section}")
    if key not in data[section]:
        raise ConfigError(f"unknown configuration key: {section}.{key}")
    data[section][key] = coerce_value(section, key, value) if coerce else value


def get_dotted(config: Config, dotted: str) -> Any:
    dotted = normalize_key(dotted)
    section, _, key = dotted.partition(".")
    data = config.to_dict()
    if section not in data or key not in data.get(section, {}):
        raise ConfigError(f"unknown configuration key: {dotted}")
    return data[section][key]


def set_dotted_and_save(config: Config, dotted: str, value: Any) -> Config:
    """Validate, apply and persist a single configuration change."""
    data = config.to_dict()
    _set_dotted(data, dotted, value, coerce=True)
    updated = _from_dict(
        data, source_path=config.source_path, profile=config.profile, nodes=config.nodes
    )
    updated.save()
    return updated


def _from_dict(
    data: dict[str, Any],
    *,
    source_path: Path | None = None,
    profile: str | None = None,
    nodes: list[NodeConfig] | None = None,
) -> Config:
    """Rebuild a Config from the nested section dict.

    `nodes` is passed separately because the node registry is deliberately not
    part of `to_dict()`: that dict feeds the dotted-key coercion machinery,
    which is built for scalar keys inside fixed sections. Anything rebuilding a
    Config from it must carry the registry across explicitly. Forgetting to do so
    deleted every registered node the first time `workerq config set` ran, and
    the dispatcher then had nowhere to place anything while `node list` still
    showed the node online.
    """
    return Config(
        core=CoreConfig(**data.get("core", {})),
        gpu=GpuConfig(**data.get("gpu", {})),
        backend=BackendConfig(**data.get("backend", {})),
        resources=ResourcesConfig(**data.get("resources", {})),
        scheduling=SchedulingConfig(**data.get("scheduling", {})),
        preemption=PreemptionConfig(**data.get("preemption", {})),
        gaming=GamingConfig(**data.get("gaming", {})),
        claude=ClaudeConfig(**data.get("claude", {})),
        nodes=(
            list(nodes)
            if nodes is not None
            else [NodeConfig(**entry) for entry in data.get("node", [])]
        ),
        source_path=source_path,
        profile=profile,
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _apply_file(data: dict[str, Any], raw: dict[str, Any], path: Path) -> None:
    for section, values in raw.items():
        if section == "node":
            data["node"] = _read_nodes(values, path)
            continue
        if not isinstance(values, dict):
            # Ignored, not fatal - the same rule the per-key branch below
            # follows, and for the same reason: a config written by a newer
            # worker-q must not brick an older one.
            #
            # This was learned the hard way. `[[node]]` was added as a
            # top-level array of tables, and every older install - including
            # the one actually serving the queue - died parsing its own config
            # the moment a node was registered. The guarantee was already
            # written down; it just did not extend to the shape of a section,
            # only to the keys inside one.
            continue
        for key, value in values.items():
            dotted = normalize_key(f"{section}.{key}")
            sect, _, k = dotted.partition(".")
            if sect not in data or k not in data[sect]:
                # Unknown keys are ignored rather than fatal, so a config written
                # by a newer gpuq does not brick an older one.
                continue
            data[sect][k] = coerce_value(sect, k, value)


def _read_nodes(values: Any, path: Path) -> list[dict[str, Any]]:
    """Parse `[[node]]` tables.

    Unknown keys *inside* a node are dropped, for the same reason unknown
    section keys are: a config written by a newer worker-q must not brick an
    older one. A malformed node is fatal, though - silently skipping it would
    leave someone with a node they registered and worker-q cannot see.
    """
    if not isinstance(values, list):
        raise ConfigError(f"{path}: [node] must be written as repeated [[node]] tables")
    known = {f.name for f in fields(NodeConfig)}
    out: list[dict[str, Any]] = []
    for entry in values:
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: each [[node]] must be a table")
        if not entry.get("name"):
            raise ConfigError(f"{path}: every [[node]] needs a name")
        out.append({k: v for k, v in entry.items() if k in known})
    return out


_ENV_ALIASES = {
    "GPUQ_STATE_DIR": "core.state_dir",
    "GPUQ_MAX_CONCURRENT_JOBS": "core.max_concurrent_jobs",
    "GPUQ_DEFAULT_PRIORITY": "core.default_priority",
    "GPUQ_SNAPSHOT_MODE": "core.snapshot_mode",
    "GPUQ_GPU_FREE_MEMORY_THRESHOLD_PERCENT": "gpu.free_memory_threshold_percent",
    "GPUQ_BACKEND": "backend.name",
}


def _apply_env(data: dict[str, Any], environ: dict[str, str]) -> None:
    """GPUQ_<SECTION>_<KEY>, plus a few convenience aliases."""
    for env_key, dotted in _ENV_ALIASES.items():
        if env_key in environ:
            _set_dotted(data, dotted, environ[env_key], coerce=True)
    for section in _SECTION_TYPES:
        for key in data[section]:
            env_key = f"GPUQ_{section.upper()}_{key.upper()}"
            if env_key in environ:
                _set_dotted(data, f"{section}.{key}", environ[env_key], coerce=True)


_UNSET: Any = object()


def load_config(
    path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    profile: str | None = _UNSET,
) -> Config:
    """Load configuration honouring the documented precedence chain."""
    environ = dict(os.environ if environ is None else environ)
    if profile is _UNSET:
        profile = environ.get("GPUQ_PROFILE", "").strip() or None

    if path is not None:
        config_path = expand_path(path)
    elif "GPUQ_CONFIG_FILE" in environ:
        config_path = expand_path(environ["GPUQ_CONFIG_FILE"])
    elif "GPUQ_CONFIG" in environ:
        config_path = expand_path(environ["GPUQ_CONFIG"])
    else:
        suffix = f"-{profile}" if profile else ""
        config_path = expand_path(f"~/.config/gpuq/config{suffix}.toml")

    data = Config(profile=profile).to_dict()
    data["core"]["state_dir"] = str(default_state_dir(profile))

    if config_path.exists():
        try:
            raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{config_path}: invalid TOML: {exc}") from exc
        _apply_file(data, raw, config_path)

    _apply_env(data, environ)
    return _from_dict(data, source_path=config_path, profile=profile)
