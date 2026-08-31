# Future: multiple GPUs, remote workers, Slurm

Nothing here is implemented. This records the seams that were left in place so
these can be added without a redesign, and the point at which each stops being
worth building yourself.

## The seam

Everything execution-related sits behind one protocol,
`src/gpuq/backends/base.py`:

```python
class SchedulerBackend(Protocol):
    def health(self) -> dict: ...
    def initialize(self) -> None: ...
    def submit(self, argv, *, label, gpu_count, slots=1, log_name=None,
               priority_rank=100, cwd=None, env=None) -> int: ...
    def list_jobs(self) -> list[BackendJob]: ...
    def get_job(self, backend_id: int) -> BackendJob: ...
    def get_state(self, backend_id: int) -> str: ...
    def output_path(self, backend_id: int) -> Path | None: ...
    def remove_queued(self, backend_id: int) -> None: ...
    def terminate_running(self, backend_id: int, *, force=False) -> None: ...
    def promote(self, backend_id: int) -> None: ...
    def set_slots(self, count: int) -> None: ...
    def find_by_label(self, label: str) -> BackendJob | None: ...
```

`GPUQService`, the CLI and the MCP adapter only ever see `BackendJob`. Backend
states map to GPUQ states in exactly one function, `core.map_backend_state`.
A new backend is a new module plus a branch in `build_backend`.

The database already reserves columns for this work:

```sql
node TEXT,
minimum_vram_gb REAL,
estimated_duration_seconds REAL
```

Adding a backend does not require a schema migration.

## Stage 1 — More local GPUs

Mostly done, deliberately not enabled.

The dispatcher already allocates *distinct* devices per job and sets
`CUDA_VISIBLE_DEVICES` accordingly, so on a multi-GPU host:

```bash
gpuq concurrency 2 --yes
```

gives one job per GPU rather than two jobs fighting over one. It stays behind
`--yes` and a warning because a single misjudged raise reintroduces exactly
the OOM this tool exists to prevent, and V1's tested configuration is one
slot.

What is still missing before this should be a default:

- per-job VRAM requirements (`minimum_vram_gb`) instead of whole-device
  exclusivity;
- topology awareness (NVLink pairs, PCIe groups) for multi-GPU jobs;
- a device-affinity policy so a 4-GPU job is not starved by a stream of
  1-GPU jobs.

## Stage 2 — GPU Task Spooler on Linux

On a Linux or WSL2 host the intended backend is the upstream GPU-aware Task
Spooler fork, pinned in `gpuq/__init__.py` as `TASK_SPOOLER_PINNED_TAG`:

`https://github.com/justanhduc/task-spooler`

A `TaskSpoolerBackend` would map the protocol onto:

| Protocol method | Task Spooler |
| --- | --- |
| `submit` | `ts -L <label> -G <n> -- gpuq _run <id>` |
| `list_jobs` / `get_job` | `ts -M json` |
| `output_path` | `ts -o <id>` |
| `remove_queued` | `ts -r <id>` |
| `terminate_running` | `ts -k <id>` (process group) |
| `promote` | `ts -u <id>` |
| `set_slots` | `ts -S <n>` |
| GPU gating | `--set_gpu_free_perc` |

Requirements before trusting it: verify the installed binary advertises
`-G/--gpus`, `--set_gpu_free_perc` and `-M/--serialize` — capability must be
detected from `ts -h`, never inferred from a version string — and give it a
dedicated socket (`TS_SOCKET`) under the GPUQ state directory so it cannot
collide with a distro Task Spooler the user already runs.

Parse only `-M json`. Never scrape the human table.

## Stage 3 — A second machine

The tempting move is `RemoteTaskSpoolerBackend` over SSH: same protocol, with
`ssh host ts ...` instead of a local call. It works, and it is a trap beyond
about two machines, because it quietly requires:

- shared or replicated snapshots (the remote host must see the frozen source);
- log retrieval and streaming across the link;
- credential and connection management;
- failure semantics when the link drops mid-job (is it lost, or still going?);
- placement policy across heterogeneous GPUs.

That list is a distributed scheduler. Two hosts is the honest ceiling for
hand-rolling it.

### What routing needs

`node`, `preferred_gpu_model`, `minimum_vram_gb`, `cpu_cores`, `ram_gb`,
`estimated_duration`, `deadline` — the first three already exist in the
schema.

## Stage 4 — Slurm

Once placement across several machines actually matters, use Slurm rather than
growing a bespoke scheduler. It already solves multi-node placement, fair
share, preemption, accounting, reservations and gres/GPU allocation, and it is
what the surrounding ecosystem expects.

A `SlurmBackend` is a thin mapping:

| Protocol method | Slurm |
| --- | --- |
| `submit` | `sbatch --gres=gpu:<n> --job-name=gpuq:<id> --wrap 'gpuq _run <id>'` |
| `list_jobs` | `squeue --json` / `sacct --json` |
| `get_state` | `sacct -j <id> --format=State,ExitCode --parsable2` |
| `remove_queued` / `terminate_running` | `scancel <id>` |
| `promote` | `scontrol update jobid=<id> priority=...` |
| `set_slots` | partition/QoS configuration, not a runtime call |

GPUQ keeps its own value on top: the ergonomic agent CLI, source snapshots,
provenance manifests, the Claude policy, and one command surface whether the
job lands locally or on a cluster.

Note that `set_slots` has no per-user equivalent in Slurm — concurrency
becomes a cluster policy. `GPUQService.set_concurrency` would need to report
that honestly rather than silently doing nothing.

## What stays out regardless

Web dashboard, Kubernetes/Kueue/Volcano, Ray conversion, fractional VRAM
scheduling, MIG management, automatic checkpoint/preemption, an LLM deciding
whether an allocation is safe, cloud workers, ETA prediction, experiment
tracking, artifact stores, automatic git pushes.

Most are separate products. The rest reintroduce the failure mode GPUQ exists
to remove: something clever deciding it is probably fine to start a second
heavy job.
