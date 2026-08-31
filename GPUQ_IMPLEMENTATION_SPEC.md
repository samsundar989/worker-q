# GPUQ — Agent GPU Workload Broker
## Implementation Specification and Claude Code Handoff

**Status:** Implementation-ready  
**Primary goal:** Eliminate GPU/system OOM collisions caused by multiple AI coding agents launching heavy workloads concurrently, while allowing those agents to enqueue work and continue working asynchronously.  
**Target environment:** Linux / WSL2 workstation with NVIDIA CUDA GPU(s), Claude Code agents with shell access.  
**V1 scope:** One workstation, exclusive heavy-job execution by default, persistent queue, source snapshots, priorities, logs, cancellation, health checks, and an optional MCP adapter built on the same core API.  
**Future-compatible:** Multiple local GPUs, remote workers, second workstation, Slurm backend.

---

# 0. CLAUDE CODE EXECUTION DIRECTIVE

**Claude Code: read this entire document before changing files. Then implement the system end-to-end.**

Do not stop after scaffolding, writing a plan, or implementing only the happy path. The task is complete only when:

1. `gpuq` is installed and callable from a new shell.
2. The GPU-aware Task Spooler backend is installed or the system clearly reports a temporary degraded backend.
3. `gpuq doctor` passes all mandatory checks.
4. Multiple GPU jobs can be submitted from separate terminals/repositories and only the configured number run concurrently.
5. Jobs survive the submitting terminal/session closing.
6. Job status, logs, cancellation, promotion, and cleanup work.
7. A queued Git-backed job executes from the source snapshot captured at submission time, not from later edits.
8. The shared Claude Code GPU policy is installed or an idempotent installer command is provided and tested.
9. Automated unit/integration tests pass.
10. A real smoke test is performed with harmless GPU commands when NVIDIA hardware is available.
11. `README.md` contains a concise “start using this now” section.
12. Any remaining limitation is explicitly documented; do not silently leave TODOs on the critical path.

Prefer reliable, boring behavior over clever scheduling.

If an operation requires `sudo` and the current environment cannot grant it, do everything else automatically and print the exact minimal command the human must run. Do not redesign the system around missing sudo if a local user installation is possible.

---

# 1. Problem Statement

Multiple Claude/Codex/other coding agents are working in parallel across AI projects. Those agents currently launch training, inference, simulation, benchmark, and GPU-heavy test commands directly.

This creates several failure modes:

- Two or more jobs allocate large portions of VRAM simultaneously.
- GPU OOMs terminate runs.
- Host RAM can also be exhausted.
- GPU jobs started from different terminals have no shared coordination.
- Agents block waiting for training instead of queuing it and continuing useful work.
- A queued command may later run against code that has changed since submission.
- It is difficult to answer: “what is running?”, “what is next?”, “which project owns the GPU?”, “where are the logs?”, and “what source version produced this result?”

The system must create one common workload gate through which heavy jobs are submitted.

---

# 2. Core Design Decision

Do **not** build a GPU scheduler from scratch.

Use **GPU Task Spooler** as the local process execution / GPU allocation backend and build a thin `gpuq` control layer around it.

GPU Task Spooler provides:

- persistent terminal-independent queues,
- CPU/GPU task execution,
- automatic GPU allocation,
- `CUDA_VISIBLE_DEVICES` assignment,
- job logs,
- job labels,
- dependencies,
- concurrent-slot limits,
- job reordering,
- running-job termination by process group,
- JSON serialization,
- a configurable GPU free-memory threshold.

Official project:
`https://github.com/justanhduc/task-spooler`

Pin the initial tested backend to a known version/tag (currently `v2.0.0` as of this specification) rather than tracking `master` blindly. Isolate the version string in one configuration/constants location so upgrading later is trivial.

The custom system owns:

- ergonomic agent CLI,
- project/job metadata,
- Git snapshots,
- durable metadata DB,
- job priorities,
- safety policy,
- health diagnostics,
- backend abstraction,
- eventual MCP interface,
- future multi-node routing.

---

# 3. V1 Architecture

```text
Claude A ─────┐
Claude B ─────┼──── gpuq CLI ─────────────┐
Claude C ─────┘                            │
                                          ▼
                                  ┌─────────────────┐
                                  │ GPUQ Core API   │
                                  │                 │
                                  │ config          │
                                  │ SQLite metadata │
                                  │ snapshots       │
                                  │ validation      │
                                  └────────┬────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │ TaskSpooler     │
                                  │ Backend         │
                                  └────────┬────────┘
                                           │
                                shared TS socket/queue
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │ gpuq runner     │
                                  │ wrapper         │
                                  └────────┬────────┘
                                           │
                                           ▼
                                      NVIDIA GPU
```

Optional, after the CLI is working:

```text
Claude MCP client ──> GPUQ MCP server ──> same GPUQ Core API
```

The MCP server MUST NOT duplicate scheduling/business logic.

---

# 4. Non-Negotiable V1 Safety Model

## 4.1 Exclusive heavy workloads by default

Default:

```yaml
max_concurrent_jobs: 1
gpu_mode: exclusive
```

V1 must favor avoiding OOMs over maximizing utilization.

Do not implement fractional GPU memory bin-packing in V1.

A 3 GB job may temporarily monopolize a 32 GB GPU. That is acceptable. Reliability is the priority.

## 4.2 Shared queue across all agents

Every `gpuq` invocation for the current Unix user must resolve to the same Task Spooler socket and same GPUQ state directory unless an explicit profile is selected.

Default:

```text
~/.local/state/gpuq/
    gpuq.sqlite3
    logs/
    snapshots/
    run/
    tmp/
    backend/
    config.toml
```

Task Spooler socket:

```text
~/.local/state/gpuq/run/task-spooler.sock
```

Do not use an ambiguous default `/tmp` socket if a deterministic per-user state location works.

If Task Spooler requires its socket to be in `/tmp` on a specific platform/version, derive a deterministic path such as:

```text
/tmp/gpuq-<uid>.sock
```

and store that resolved path in config.

## 4.3 GPU availability threshold

Default:

```yaml
gpu_free_memory_threshold_percent: 90
```

Configure Task Spooler with its `--set_gpu_free_perc` option.

Also perform a `gpuq` preflight visibility check with `nvidia-smi` so `doctor` can warn about foreign GPU processes.

## 4.4 Fail safe

If the scheduler/backend cannot be contacted:

- `gpuq submit` must fail clearly.
- It must never “helpfully” run the command directly.
- Error message must explain how to run `gpuq doctor`.
- Return non-zero.

This is critical.

---

# 5. Technology Choices

Use:

- Python 3.11+ for GPUQ.
- Standard library where practical.
- `typer` + `rich` are acceptable for CLI ergonomics.
- `pydantic` is acceptable for typed config/models.
- SQLite via stdlib `sqlite3`.
- TOML config via `tomllib`.
- `pytest` for tests.
- `uv` for project/dev dependency management if available.
- Fallback instructions for `pipx` / venv if `uv` is unavailable.
- Official MCP Python SDK v2 only for the optional MCP adapter.

Do not require:

- Docker,
- Kubernetes,
- Redis,
- Postgres,
- a web server,
- root daemon,
- Slurm,
- cloud services.

---

# 6. Repository Layout

Create a dedicated repository similar to:

```text
agent-gpu-broker/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── src/
│   └── gpuq/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── db.py
│       ├── models.py
│       ├── core.py
│       ├── runner.py
│       ├── snapshot.py
│       ├── gpu.py
│       ├── doctor.py
│       ├── claude_policy.py
│       ├── cleanup.py
│       ├── util.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── task_spooler.py
│       └── mcp/
│           ├── __init__.py
│           └── server.py
├── scripts/
│   ├── bootstrap.sh
│   ├── uninstall.sh
│   └── smoke_test.sh
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_db.py
│   │   ├── test_snapshot.py
│   │   ├── test_backend.py
│   │   ├── test_cli.py
│   │   └── test_policy.py
│   └── integration/
│       ├── test_queue.py
│       ├── test_cancel.py
│       ├── test_snapshot_execution.py
│       └── test_recovery.py
└── docs/
    ├── architecture.md
    ├── troubleshooting.md
    └── future-slurm.md
```

The exact filenames may vary slightly, but preserve the module boundaries.

---

# 7. Backend Interface

Define a backend abstraction immediately.

Example:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

@dataclass(frozen=True)
class BackendJob:
    backend_id: int
    state: str
    label: str | None
    output_path: Path | None
    pid: int | None

class SchedulerBackend(Protocol):
    def health(self) -> dict: ...
    def initialize(self) -> None: ...

    def submit(
        self,
        argv: Sequence[str],
        *,
        label: str,
        gpu_count: int,
        slots: int = 1,
        log_name: str | None = None,
    ) -> int: ...

    def list_jobs(self) -> list[BackendJob]: ...
    def get_job(self, backend_id: int) -> BackendJob: ...
    def get_state(self, backend_id: int) -> str: ...
    def output_path(self, backend_id: int) -> Path | None: ...
    def remove_queued(self, backend_id: int) -> None: ...
    def terminate_running(self, backend_id: int) -> None: ...
    def promote(self, backend_id: int) -> None: ...
    def set_slots(self, count: int) -> None: ...
```

V1 implementation:

```text
TaskSpoolerBackend
```

Future implementations:

```text
RemoteTaskSpoolerBackend
SlurmBackend
```

Do not leak raw `ts` command construction throughout the application.

---

# 8. Task Spooler Backend Requirements

## 8.1 Detection

`gpuq doctor` and `bootstrap.sh` must verify:

```bash
ts -V
ts -h
```

GPU-aware backend is considered present only if help output contains the required GPU features, including at least:

```text
-G / --gpus
--set_gpu_free_perc
-M / --serialize
```

Do not infer GPU support from version string alone.

## 8.2 Installation

Preferred local/non-root path:

1. Clone the pinned Task Spooler tag into:
   `~/.local/share/gpuq/vendor/task-spooler`
2. Set `CUDA_HOME`.
3. Build using the project's documented CMake or Make installation path.
4. Install locally where possible.
5. Ensure the resulting `ts` is found before an incompatible distro package.

The backend installation logic should attempt to discover CUDA in this order:

```text
$CUDA_HOME
/usr/local/cuda
dirname(dirname(which nvcc))
common WSL CUDA locations
```

Validate:

```bash
test -x "$CUDA_HOME/bin/nvcc"
```

If CUDA toolkit headers/compiler are unavailable but the NVIDIA runtime is working:

- report this accurately;
- allow a **degraded CPU-only Task Spooler execution backend** with `max_concurrent_jobs=1`;
- still use the queue to prevent broker-managed workloads from overlapping;
- mark `gpuq doctor` GPU-backend status as WARN, not PASS;
- never claim that foreign GPU workloads are automatically detected in degraded mode.

The final “fully healthy” target remains the GPU-aware fork.

## 8.3 Environment

All backend subprocess calls must pass the same environment builder.

Set:

```text
TS_SOCKET
TS_SAVELIST
TS_MAXFINISHED
```

and any other deterministic Task Spooler variables needed by the installed version.

Create parent directories before use.

Initialize:

```text
max simultaneous TS jobs = config.max_concurrent_jobs
GPU free percentage = config.gpu_free_memory_threshold_percent
log directory = GPUQ logs directory
```

Ensure this initialization is idempotent.

## 8.4 Submission

Do not build a single shell string from user arguments.

Prefer passing an argument vector into Task Spooler:

```text
ts [options] gpuq _run <gpuq_job_id> -- <argv...>
```

If the Task Spooler CLI requires shell semantics internally, use the narrowest possible quoting layer and `shlex.join()` only at the final boundary.

User-supplied command arguments must round-trip correctly when they contain:

- spaces,
- quotes,
- glob characters,
- `=`,
- Unicode,
- shell metacharacters that are intended as literal argv.

If the user intentionally wants shell syntax, expose it explicitly:

```bash
gpuq submit --shell 'python train.py > result.txt && python grade.py'
```

Normal `gpuq submit -- ...` must be argv-safe and non-shell.

---

# 9. GPUQ Database

Use one SQLite DB:

```text
~/.local/state/gpuq/gpuq.sqlite3
```

On connection:

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA foreign_keys=ON;
```

Use migrations with a tiny schema-version table.

## 9.1 Jobs table

Minimum schema:

```sql
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend TEXT NOT NULL,
    backend_job_id INTEGER,

    project TEXT NOT NULL,
    label TEXT,
    priority TEXT NOT NULL,

    repo_root TEXT,
    submitted_cwd TEXT NOT NULL,
    execution_cwd TEXT,

    command_json TEXT NOT NULL,
    shell_mode INTEGER NOT NULL DEFAULT 0,

    requested_gpu_count INTEGER NOT NULL DEFAULT 1,
    gpu_mode TEXT NOT NULL DEFAULT 'exclusive',

    snapshot_mode TEXT NOT NULL,
    snapshot_commit TEXT,
    snapshot_path TEXT,

    host TEXT NOT NULL,
    submitter_pid INTEGER,
    submitter_agent TEXT,

    state TEXT NOT NULL,
    exit_code INTEGER,
    runner_pid INTEGER,

    queued_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,

    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Indexes:

```sql
CREATE INDEX idx_jobs_state ON jobs(state);
CREATE INDEX idx_jobs_backend_id ON jobs(backend, backend_job_id);
CREATE INDEX idx_jobs_project ON jobs(project);
CREATE INDEX idx_jobs_queued_at ON jobs(queued_at);
```

## 9.2 State enum

Normalize internal states:

```text
PREPARING
QUEUED
RUNNING
SUCCEEDED
FAILED
CANCELLED
LOST
```

The backend state may have more/fewer states. Map them in one place.

## 9.3 Atomic submission

Submission must be crash-safe enough that the DB and backend do not silently disagree.

Recommended flow:

1. Validate request.
2. Insert DB row as `PREPARING`.
3. Create snapshot.
4. Update snapshot metadata.
5. Submit `gpuq _run <job_id> ...` to backend.
6. Store `backend_job_id`.
7. Mark `QUEUED`.
8. Print success.

If backend submission fails:

- mark row `FAILED`;
- store error;
- cleanup snapshot if safe;
- return non-zero.

If process crashes between backend submission and DB update, implement `gpuq reconcile` / `doctor` logic using the GPUQ label to recover the backend ID if possible.

Backend labels must include a unique marker such as:

```text
gpuq:<gpuq_job_id>:<project>:<priority>
```

---

# 10. Configuration

Default user config:

```text
~/.config/gpuq/config.toml
```

Example:

```toml
[core]
state_dir = "~/.local/state/gpuq"
max_concurrent_jobs = 1
default_priority = "normal"
snapshot_mode = "git"
cleanup_successful_snapshots_after_days = 7
cleanup_failed_snapshots_after_days = 14

[gpu]
default_gpu_count = 1
free_memory_threshold_percent = 90
exclusive_by_default = true

[task_spooler]
binary = "ts"
socket = "~/.local/state/gpuq/run/task-spooler.sock"
max_finished = 1000

[claude]
install_user_policy = true
hide_cuda_in_safe_launcher = false
```

CLI/env precedence:

```text
CLI flag
GPUQ_* environment variable
config.toml
built-in default
```

Implement `gpuq config show`.

Implement:

```bash
gpuq config set core.max_concurrent_jobs 1
gpuq config set gpu.free_memory_threshold_percent 90
```

If generic nested setting mutation becomes unnecessary complexity, V1 may instead expose focused commands:

```bash
gpuq concurrency 1
gpuq gpu-threshold 90
```

But `config show` is mandatory.

---

# 11. CLI Contract

Use `gpuq` as the only user/agent-facing executable.

## 11.1 Initialize

```bash
gpuq init
```

Responsibilities:

- create state/config directories,
- initialize DB,
- initialize backend,
- set Task Spooler slots,
- set log directory,
- set free-memory threshold,
- run core health checks,
- print next commands.

Idempotent.

## 11.2 Submit

```bash
gpuq submit \
  --project arc-agi \
  --priority normal \
  -- python train.py --config configs/exp17.yaml
```

Required behavior:

- infer Git repository root from cwd;
- infer project from repo basename if omitted;
- validate command is non-empty;
- snapshot source by default;
- create DB row;
- enqueue one GPU by default;
- print GPUQ job ID prominently;
- print state / queue info;
- return immediately.

Supported V1 flags:

```text
--project NAME
--priority critical|high|normal|low
--gpus N
--label TEXT
--cwd PATH
--snapshot / --no-snapshot
--live-worktree
--shell STRING
--env KEY=VALUE       repeatable
--passthrough PATH    repeatable
```

`--no-snapshot` and `--live-worktree` must be explicit opt-outs.

Example output:

```text
GPUQ job #42 submitted
Project: arc-agi
Priority: normal
State: QUEUED
Backend job: 17
Snapshot: 9f613e2
Logs: gpuq logs 42 --follow
```

## 11.3 List/status

Both should work:

```bash
gpuq status
gpuq list
```

Display active and recent jobs.

Columns:

```text
ID
STATE
PRI
PROJECT
AGE/RUNTIME
GPU
BACKEND_ID
COMMAND
```

Default order:

1. RUNNING
2. QUEUED by effective queue order
3. recent finished jobs newest first

Flags:

```text
--all
--project NAME
--state STATE
--json
```

Machine-readable JSON is mandatory for agents/scripts.

## 11.4 Show

```bash
gpuq show 42
```

Show:

- full command,
- project,
- state,
- backend state,
- submit/start/end times,
- exit code,
- repo root,
- snapshot commit/path,
- execution cwd,
- requested GPUs,
- assigned `CUDA_VISIBLE_DEVICES` if captured,
- log path,
- error.

Support `--json`.

## 11.5 Logs

```bash
gpuq logs 42
gpuq logs 42 --follow
gpuq logs 42 --tail 100
```

Do not force users to know Task Spooler IDs.

For queued jobs with no log yet:

```text
Job #42 is queued; output file has not been created yet.
```

Return success for this informational state unless piping semantics strongly favor otherwise.

## 11.6 Cancel

```bash
gpuq cancel 42
```

Behavior:

- queued backend job -> remove with Task Spooler remove action;
- running backend job -> SIGTERM process group via Task Spooler;
- update DB to `CANCELLED` when termination/removal is confirmed;
- if already finished, report final state and do nothing;
- if SIGTERM does not finish within a configured grace period, V1 may provide:
  `gpuq cancel 42 --force`
  using SIGKILL only if it can safely target the process group.

Never kill based solely on a stale PID without validating job ownership.

## 11.7 Promote

```bash
gpuq promote 42
```

V1 Task Spooler supports “put this queued job first.”

Policy:

- `critical`: submit and promote.
- `high`: record priority and, when practical, order ahead of `normal/low` without destabilizing already-running work.
- `normal`: FIFO.
- `low`: FIFO behind normal when achievable safely.

Important: do not build an elaborate custom dispatcher solely to get four perfect queues in V1.

If Task Spooler cannot exactly maintain four levels using safe reorder operations, document the operational mapping:

```text
critical -> front of queue
high     -> best-effort ahead of normal/low
normal   -> FIFO
low      -> best-effort tail
```

The DB preserves the semantic priority so a future dispatcher can implement it exactly.

Never preempt a currently running V1 job merely because a new critical job arrives.

## 11.8 Doctor

```bash
gpuq doctor
```

This command is mandatory and should be excellent.

Checks:

- Python version.
- GPUQ config parse.
- state directories writable.
- SQLite read/write transaction.
- `ts` found.
- `ts` version/help compatible.
- expected TS socket reachable or can start.
- TS slot count equals config.
- TS log dir equals config.
- TS GPU free threshold equals config when supported.
- `nvidia-smi` exists.
- NVIDIA driver responds.
- GPU inventory parsed.
- CUDA toolkit detected where required.
- GPU-aware Task Spooler capability present.
- no incompatible stale Task Spooler server/version.
- snapshot dependencies (`git`) available.
- user Claude policy present, if enabled.
- optional MCP server import/test.

Output:

```text
PASS  SQLite
PASS  Task Spooler v2.x
PASS  GPU allocation support
PASS  NVIDIA driver
PASS  RTX ...
PASS  queue concurrency = 1
PASS  GPU free threshold = 90%
WARN  GPU currently in use by PID ...
PASS  Claude policy installed

Overall: HEALTHY
```

Use exit codes:

```text
0 = healthy
1 = warnings/degraded but usable
2 = broken / unsafe to submit
```

## 11.9 Reconcile

```bash
gpuq reconcile
```

Repair metadata after crashes/restarts by comparing DB rows with Task Spooler serialized state.

Rules:

- DB QUEUED + backend running -> RUNNING if runner metadata missing.
- DB QUEUED/RUNNING + backend finished -> recover final state/exit information where available.
- DB active + backend missing, runner absent -> LOST.
- finished DB jobs remain immutable except missing derived fields.
- never resurrect cancelled jobs automatically.

`doctor` may invoke a read-only reconciliation check; actual mutation can be an explicit function called by normal status operations if well-tested.

## 11.10 Cleanup

```bash
gpuq cleanup
gpuq cleanup --dry-run
gpuq cleanup --older-than 7d
```

Remove:

- expired snapshot worktrees/directories,
- orphan temp files,
- optional old successful logs according to retention config.

Never delete:

- active job snapshot,
- active logs,
- source repo,
- DB,
- failed-job evidence within retention window.

---

# 12. Git Snapshot Design

This is mandatory because agents continue modifying code after enqueue.

## 12.1 Goal

If job #42 is submitted at source state S42 and Claude then edits the repo to S43 before #42 starts, job #42 MUST execute S42.

## 12.2 Snapshot behavior

Default for a Git repository:

- include tracked files at current working-tree content;
- include staged changes;
- include unstaged changes;
- include untracked **non-ignored** files;
- exclude `.git`;
- exclude ignored caches/datasets/virtualenvs unless explicitly passed through.

Do not force the user to commit WIP.

## 12.3 Recommended implementation

Use a temporary Git index to create an ephemeral commit without modifying the real index or branch.

Conceptual steps:

```bash
GIT_INDEX_FILE=<temp-index> git read-tree HEAD
GIT_INDEX_FILE=<temp-index> git add -A
TREE=$(GIT_INDEX_FILE=<temp-index> git write-tree)
SNAPSHOT=$(printf '%s\n' "gpuq snapshot job 42" | git commit-tree "$TREE" -p HEAD)
```

Important:

- use subprocess argv, not fragile shell interpolation;
- handle repository with no HEAD/unborn branch;
- preserve the real index;
- clean temp index on failure;
- record the ephemeral commit hash.

Then create the execution snapshot using one of:

### Preferred A: detached worktree

```bash
git worktree add --detach <snapshot_path> <snapshot_commit>
```

Advantages:

- efficient;
- Git-aware;
- easy provenance.

Requirements:

- cleanup with `git worktree remove --force`;
- run `git worktree prune` carefully;
- handle stale worktree metadata.

### Acceptable B: git archive extraction

Use only if it proves more robust for this implementation.

If using archive, document submodule behavior.

## 12.4 Ignored runtime data

A training repo may depend on ignored paths such as:

```text
data/
datasets/
checkpoints/
.cache/
models/
```

Provide project config:

```toml
# .gpuq.toml
[snapshot]
passthrough = ["data", "datasets", "checkpoints"]
```

For each passthrough path:

- if it exists in the live repo and is absent in the snapshot,
- symlink snapshot path -> live path;
- resolve real paths;
- prevent traversal outside the repository unless explicitly allowed by an absolute CLI `--passthrough`;
- record the symlink mapping in job metadata.

Do not copy giant datasets by default.

## 12.5 Non-Git directories

Default behavior:

- refuse snapshot mode with a clear message;
- tell the user to use `--live-worktree` or initialize Git.

Optional enhancement:

- lightweight copy snapshot for small non-Git trees.

Do not silently pretend a live directory is immutable.

---

# 13. Job Runner

All backend-executed commands must call an internal runner:

```text
gpuq _run <gpuq_job_id> -- <user argv>
```

This command is hidden/internal but directly testable.

Runner responsibilities:

1. Load job from DB.
2. Verify job/backend ownership.
3. Determine execution cwd.
4. Record:
   - state = RUNNING,
   - started_at,
   - runner PID,
   - hostname,
   - `CUDA_VISIBLE_DEVICES`,
   - GPU inventory snapshot,
   - relevant Python/CUDA environment metadata.
5. `chdir(execution_cwd)`.
6. Apply allowed job-specific environment variables.
7. Execute the user command preserving signals.
8. Capture exit code.
9. Mark:
   - `SUCCEEDED` for 0,
   - `FAILED` otherwise,
   unless state has already been explicitly marked CANCELLED.
10. Record finished_at.
11. Exit with the user's exit code.

Prefer `os.execvpe()` only if doing so does not prevent the runner from recording final status.

A straightforward wrapper using `subprocess.Popen` + signal forwarding is acceptable and often better because final state must be persisted.

Signal behavior:

- forward SIGTERM/SIGINT to child process group;
- ensure child process tree terminates;
- avoid zombie children;
- runner termination caused by `gpuq cancel` must produce a sensible cancelled/failure reconciliation state.

Task Spooler itself creates a process group for its command; still test the nested process-group behavior carefully.

---

# 14. GPU Inspection

Implement `gpuq/gpu.py`.

Use `nvidia-smi` rather than adding NVML Python dependencies unless NVML materially improves reliability.

Commands should query machine-readable CSV/noheader fields such as:

```text
index
uuid
name
memory.total
memory.used
memory.free
utilization.gpu
```

Also query compute applications where supported:

```text
pid
process_name
used_gpu_memory
gpu_uuid
```

Parser requirements:

- handle `N/A`;
- handle no processes;
- handle WSL differences;
- never crash status/doctor because one optional metric is unavailable.

Expose:

```bash
gpuq gpu
gpuq gpu --json
```

Example:

```text
GPU 0  NVIDIA GeForce RTX ...
Memory: 4.2 / 31.8 GiB
Free:   86.8%
Util:   12%
Processes:
  12345 python 3.8 GiB
```

This is informational. Task Spooler remains the resource allocator.

---

# 15. Foreign GPU Workload Safety

Broker-managed jobs are safe from each other when `max_concurrent_jobs=1`.

Foreign/direct commands can still consume the GPU.

Mitigations:

## 15.1 GPU-aware Task Spooler

Set:

```text
free-memory threshold = 90%
```

so a queued GPU job does not claim a GPU that is already heavily occupied.

## 15.2 Claude instructions

Install a user-level `~/.claude/CLAUDE.md` policy block.

Official Claude Code user instructions live there and apply across projects.

The installer MUST be idempotent and use markers:

```markdown
<!-- gpuq-policy:start -->
...
<!-- gpuq-policy:end -->
```

If a file already exists:

- preserve all existing content;
- replace only the marked block;
- make a timestamped backup before first modification;
- do not duplicate the policy.

Policy text:

```markdown
<!-- gpuq-policy:start -->
## GPU / Heavy Workload Policy

This machine uses `gpuq` to coordinate expensive workloads across concurrent agents.

NEVER directly launch a command expected to substantially use NVIDIA CUDA/VRAM
or large enough host resources that overlapping runs could cause OOM.

This includes, unless clearly tiny:
- model training or fine-tuning
- substantial inference/evaluation
- CUDA-heavy test suites
- torchrun / accelerate launches
- GPU benchmarks
- GPU simulators
- hyperparameter/ablation sweeps
- commands that load large models
- other long-running high-memory experiments

Submit them instead:

    gpuq submit --project <project> --priority normal -- <command> <args...>

Use `--priority critical` only for work that is genuinely blocking urgent progress.

After submitting, continue independent coding/research/testing instead of waiting
unless the result is required for the next action.

Inspect work with:

    gpuq status
    gpuq show <job_id>
    gpuq logs <job_id>
    gpuq logs <job_id> --follow

Cancel with:

    gpuq cancel <job_id>

Do not bypass `gpuq` just because `nvidia-smi` currently looks idle.
The shared queue is the source of truth for broker-managed heavy work.

Small CPU-only commands and genuinely lightweight tests may run directly.
<!-- gpuq-policy:end -->
```

Implement:

```bash
gpuq claude-policy install
gpuq claude-policy status
gpuq claude-policy remove
```

Do not remove unrelated user instructions.

## 15.3 Optional stronger launcher

Implement, but do not enable globally without an explicit command:

```bash
gpuq claude-safe-launcher install
```

This creates something like:

```text
~/.local/bin/claude-gpu-safe
```

which launches Claude Code with:

```bash
CUDA_VISIBLE_DEVICES=""
```

The `gpuq` backend must explicitly remove inherited `CUDA_VISIBLE_DEVICES=""` when contacting/starting Task Spooler as needed.

This is defense-in-depth only.

Document the tradeoff: hiding CUDA may break legitimate lightweight GPU detection commands run directly by Claude.

---

# 16. Priority Policy

Expose:

```text
critical
high
normal
low
```

Store all four exactly in DB.

Effective V1 semantics:

```text
CRITICAL
- move ahead of queued non-critical work
- never interrupt currently running job

HIGH
- prefer ahead of NORMAL/LOW if safe queue reorder is available

NORMAL
- standard FIFO

LOW
- best-effort behind normal/high
```

Aging:

Do not implement complex aging in V1 unless trivial.

Starvation protection can be a V2 feature.

Priority must never change resource safety rules.

---

# 17. Queue Concurrency

Default:

```text
1
```

Implement:

```bash
gpuq concurrency
gpuq concurrency set 1
```

Changing to `2+` must print a warning:

```text
WARNING: concurrent GPU jobs can cause VRAM OOM.
GPUQ V1 assumes exclusive heavy-job execution.
```

Require `--yes` for non-interactive change above 1, or an equivalent explicit confirmation flag.

Agents should not change concurrency automatically.

If multiple physical GPUs exist, V1 may eventually safely use more than one slot if each job requests one distinct GPU and Task Spooler guarantees allocation. However, **do not block initial completion on this optimization**.

The initial safe default remains one total heavy job.

---

# 18. Logs

Use Task Spooler as the primary stdout/stderr collector.

GPUQ should also record a small structured execution manifest.

For each job:

```text
~/.local/state/gpuq/jobs/<id>/
    manifest.json
    environment.json
    result.json
```

Task Spooler output can remain under:

```text
~/.local/state/gpuq/logs/
```

`gpuq show` should resolve the backend output file dynamically and/or persist it after creation.

Never lose the mapping from GPUQ ID -> Task Spooler ID -> log.

---

# 19. Manifest

Write a JSON manifest at submission and update derived metadata after run.

Example:

```json
{
  "gpuq_job_id": 42,
  "backend": "task_spooler",
  "backend_job_id": 17,
  "project": "arc-agi",
  "priority": "normal",
  "command": ["python", "train.py", "--config", "configs/exp17.yaml"],
  "submitted_cwd": "/home/user/projects/arc-agi",
  "execution_cwd": "/home/user/.local/state/gpuq/snapshots/42/repo",
  "repo_root": "/home/user/projects/arc-agi",
  "snapshot_commit": "9f613e2...",
  "snapshot_passthrough": ["data"],
  "gpu_count": 1,
  "gpu_mode": "exclusive",
  "submitted_at": "...",
  "started_at": null,
  "finished_at": null,
  "host": "..."
}
```

This makes experiment provenance inspectable without opening SQLite.

---

# 20. MCP Adapter

Build this only after the CLI/core tests pass. It should be thin.

Use the official MCP Python SDK current stable v2.

Package dependency can be an optional extra:

```toml
[project.optional-dependencies]
mcp = ["mcp[cli]>=2,<3"]
```

Expose tools:

```text
gpu_submit
gpu_status
gpu_job
gpu_logs
gpu_cancel
gpu_promote
gpu_info
```

Suggested structured input for `gpu_submit`:

```python
project: str | None
command: list[str]
priority: Literal["critical", "high", "normal", "low"] = "normal"
gpus: int = 1
cwd: str | None = None
snapshot: bool = True
label: str | None = None
env: dict[str, str] | None = None
```

Output:

```json
{
  "job_id": 42,
  "state": "QUEUED",
  "project": "arc-agi",
  "priority": "normal",
  "message": "Submitted. Continue independent work; inspect with gpu_job/gpu_logs when needed."
}
```

MCP must call the same `GPUQService` / core functions as CLI.

Do not shell out to `gpuq` from MCP unless used temporarily during development.

## 20.1 Claude Code registration

Document an installation command appropriate for Claude Code's current MCP configuration, but do not hard-code a stale format without testing the installed Claude version.

At minimum provide:

```bash
gpuq mcp command
```

which prints the stdio server command.

And:

```bash
gpuq mcp test
```

which constructs an in-memory MCP client/server test if the SDK is installed.

The CLI remains the supported fallback even if MCP is not configured.

---

# 21. Bootstrap Script

Create:

```bash
scripts/bootstrap.sh
```

It must be safe to rerun.

High-level flow:

```text
1. Detect Linux/WSL.
2. Verify Python >= 3.11.
3. Verify git.
4. Verify nvidia-smi.
5. Detect uv/pipx/venv strategy.
6. Detect compatible GPU Task Spooler.
7. If absent, install pinned GPU Task Spooler locally.
8. Install gpuq package.
9. Run gpuq init.
10. Run gpuq doctor.
11. Install Claude policy block.
12. Run non-destructive queue smoke test.
13. If GPU available, optionally run GPU smoke test.
14. Print five commands needed for daily usage.
```

Avoid destructive commands.

If an old/incompatible Task Spooler daemon is detected, do not blindly kill unrelated user work. Print the detected socket/process and either:

- use GPUQ's isolated socket, or
- stop only the GPUQ-owned server.

Because GPUQ uses a dedicated socket, coexistence should normally be possible.

---

# 22. Uninstall

Create:

```bash
scripts/uninstall.sh
```

And preferably:

```bash
gpuq uninstall --dry-run
```

Uninstall must distinguish:

- GPUQ package,
- GPUQ state,
- Task Spooler vendor install,
- Claude policy block.

Default uninstall should preserve logs/database unless `--purge`.

Never delete user source repos.

---

# 23. Unit Tests

At minimum:

## Config
- defaults load;
- `~` expands;
- invalid percentages rejected;
- concurrency < 1 rejected;
- config precedence works.

## DB
- schema initializes;
- migration is idempotent;
- concurrent inserts work with WAL;
- state transitions validated;
- terminal state cannot accidentally revert to QUEUED.

## Backend
Mock `ts`.

Test:

- env contains expected socket;
- submit builds argv safely;
- GPU capability detection;
- state mapping;
- queued cancellation calls remove;
- running cancellation calls terminate;
- promote calls urgent action;
- JSON serialization parsing;
- malformed backend output fails clearly.

## Snapshot
Create temporary Git repo.

Test:

1. clean repo snapshot.
2. staged changes included.
3. unstaged changes included.
4. untracked non-ignored file included.
5. ignored file excluded.
6. real Git index unchanged.
7. real branch HEAD unchanged.
8. later source edit does not change snapshot.
9. passthrough symlink created.
10. cleanup removes worktree cleanly.
11. filename with spaces.
12. Unicode filename.
13. unborn repository if supported; otherwise friendly error.

## CLI
Using Typer test runner or subprocess.

Test:

- submit validation.
- project inference.
- JSON output.
- invalid job ID.
- cancellation idempotence.
- queued logs message.
- doctor exit codes.

## Claude policy
- append to new file;
- preserve existing file;
- update existing marked block;
- no duplicate blocks;
- remove only GPUQ block;
- backup behavior.

---

# 24. Integration Tests

Use a dedicated isolated Task Spooler socket.

Do not touch the user's production queue during tests.

## 24.1 Serialization test

Submit:

```bash
python -c "import time; print('hello'); time.sleep(1); print('done')"
```

Verify:

- QUEUED/RUNNING/SUCCEEDED transition.
- log output.
- exit code.

## 24.2 Queue exclusivity

Submit three jobs that each:

- record start time,
- sleep 2 seconds,
- record end time.

With concurrency `1`, prove intervals do not overlap.

This is the most important non-GPU integration test.

## 24.3 Terminal independence

Submit a job from a subprocess that exits immediately.

Verify job still runs and completes.

## 24.4 Cancellation

Queue two jobs.

Cancel second while queued.

Verify it never executes.

Run a long first job.

Cancel while running.

Verify child process tree is terminated.

## 24.5 Snapshot execution

1. Repo script prints `VALUE = "A"`.
2. Submit while queue is blocked.
3. Edit live repo to print `"B"`.
4. Release queue.
5. Job log must print `"A"`.

Mandatory.

## 24.6 Recovery

Simulate:

- GPUQ process exits after submission.
- New GPUQ process runs `status/reconcile`.
- Job remains discoverable.
- backend ID/log mapping recovered.

---

# 25. NVIDIA Smoke Test

Only run if `nvidia-smi` and a CUDA-capable Python environment are available.

Avoid allocating huge memory.

Suggested PyTorch test if torch is installed:

```bash
gpuq submit --project gpuq-smoke -- \
  python -c 'import torch,time; print(torch.cuda.get_device_name()); x=torch.zeros((1024,1024),device="cuda"); print(x.sum().item()); time.sleep(2)'
```

If PyTorch is not installed, use a safe existing CUDA sample/tool or simply verify assigned environment and `nvidia-smi`.

Do not install a huge PyTorch package merely for the smoke test unless already part of the environment.

Then submit two harmless GPU sleeps and verify only one starts at a time.

---

# 26. Acceptance Test Script

Create:

```text
scripts/smoke_test.sh
```

It should:

1. create an isolated temp Git repo;
2. submit three non-GPU sleeps through production GPUQ or an isolated test profile;
3. verify no overlap;
4. verify logs;
5. verify cancellation;
6. verify snapshot immutability;
7. print PASS/FAIL summary.

If using the production queue, ensure no existing job is disturbed.

Prefer a temporary test profile/socket.

---

# 27. Daily Agent Workflow

The final README must put this near the top.

## Submit a training job

```bash
gpuq submit --project arc-agi -- \
  python train.py --config configs/exp17.yaml
```

## Urgent blocking evaluation

```bash
gpuq submit --project arc-agi --priority critical -- \
  python evaluate.py --checkpoint latest
```

## See queue

```bash
gpuq status
```

## Follow logs

```bash
gpuq logs 42 --follow
```

## Cancel

```bash
gpuq cancel 42
```

## Inspect source provenance

```bash
gpuq show 42
```

## Check system health

```bash
gpuq doctor
```

Claude agents should enqueue and continue working whenever possible.

---

# 28. Recommended Output Formatting

Human output should be compact and easy to scan.

Example:

```text
$ gpuq status

GPU: NVIDIA GeForce RTX ...
Concurrency: 1
Free VRAM: 29.8 / 31.8 GiB

 ID   STATE      PRI       PROJECT         RUNTIME   COMMAND
 58   RUNNING    normal    pokemon-ai      12m       python train.py ...
 59   QUEUED     critical  arc-agi         4m wait   python evaluate.py ...
 60   QUEUED     normal    biohub          1m wait   python sweep.py ...

Use: gpuq logs 58 --follow
```

Machine mode:

```bash
gpuq status --json
```

must contain no decorative text on stdout.

Errors go to stderr.

---

# 29. Reliability Rules

1. **Never bypass the queue on failure.**
2. **Never execute user command before DB/backend submission succeeds.**
3. **Never mutate the user's Git branch/index for snapshots.**
4. **Never delete active snapshot/log data.**
5. **Never kill a process using an unverified stale PID.**
6. **Never assume Task Spooler default socket; always set GPUQ socket.**
7. **Never parse human Task Spooler output if JSON serialization is available.**
8. **Never store commands using lossy string concatenation; store argv as JSON.**
9. **Never run arbitrary shell syntax unless user selected `--shell`.**
10. **Never let a failed doctor check silently degrade safety.**
11. **Use UTC ISO-8601 timestamps internally.**
12. **Use atomic file writes for manifests/config updates.**
13. **All filesystem paths stored in DB should be absolute/resolved where appropriate.**
14. **State-changing DB operations should be transactional.**
15. **CLI operations must tolerate two agents invoking GPUQ simultaneously.**

---

# 30. Security / Command Safety

This is a local trusted-user tool, not a multi-tenant sandbox.

Still:

- do not use `shell=True` for normal submission;
- validate env keys;
- do not interpolate raw user strings into generated shell code;
- protect snapshot cleanup against path traversal;
- resolve cleanup targets and verify they are descendants of GPUQ state dir before deletion;
- set reasonable file permissions on socket/state DB;
- do not expose an unauthenticated network HTTP server in V1;
- MCP should use stdio by default.

---

# 31. Failure Cases and Expected Behavior

## Task Spooler server restarts

Run reconciliation.

Queued jobs should be recoverable if Task Spooler persisted them. If not recoverable, mark `LOST` rather than pretending they completed.

## Machine reboots

Document that Task Spooler queue persistence and automatic restart behavior must be validated.

For V1, after reboot:

```bash
gpuq init
gpuq reconcile
gpuq status
```

should restore a coherent state.

If automatic queued-job continuation across reboot is not guaranteed by backend behavior, state that explicitly. Do not claim stronger persistence than tested.

Optional later: user systemd service.

## Snapshot creation fails

Submission fails before backend enqueue.

## Snapshot cleanup fails

Job result remains valid. Emit warning. Cleanup may retry later.

## `nvidia-smi` unavailable

If configured job requires GPU:

- doctor broken/degraded.
- submit should refuse GPU job unless an explicit CPU/degraded mode exists.

## Existing foreign CUDA job

GPU-aware Task Spooler should withhold GPU until free-memory threshold is met.

`gpuq gpu` and `doctor` show the foreign process.

## Agent submits a command that spawns detached children

Test and document cancellation behavior. Prefer process-group ownership.

---

# 32. Optional User Systemd Service

Do not require a custom dispatcher service in V1.

Task Spooler is the queue daemon.

A user service may be useful later for:

- reconciliation after login,
- automatic queue health,
- notifications.

If added, keep it optional:

```text
~/.config/systemd/user/gpuq-reconcile.service
```

Do not add always-running Python infrastructure unless it provides a concrete V1 benefit.

---

# 33. Notifications — Deferred

Not required for completion.

Future:

```text
gpuq notify
Slack
desktop notification
Claude hook
webhook
```

Do not delay V1 for this.

---

# 34. Multi-Machine / Second GPU Host — Deferred but Designed For

Do not implement remote routing in the critical path.

The backend abstraction and schema should allow future fields:

```text
node
preferred_gpu_model
minimum_vram_gb
cpu_cores
ram_gb
estimated_duration
deadline
```

Future architecture:

```text
gpuq broker
   ├── local TaskSpoolerBackend
   ├── remote TaskSpoolerBackend over SSH
   └── SlurmBackend
```

At the point multi-node scheduling becomes important, Slurm is likely preferable to growing a bespoke distributed scheduler.

---

# 35. Explicitly Out of Scope for V1

Do not implement these unless the complete V1 above is already passing:

- web dashboard,
- Kubernetes,
- Kueue,
- Volcano,
- Ray runtime conversion,
- fractional VRAM scheduling,
- MIG management,
- automatic checkpoint/preemption,
- LLM deciding whether resource allocation is safe,
- cloud workers,
- Windows-native scheduler,
- generalized cluster orchestration,
- ETA prediction,
- experiment tracking platform,
- artifact store,
- hyperparameter service,
- automatic Git pushes,
- automatic result interpretation.

---

# 36. Definition of Done

The implementation is not complete until this exact scenario works.

Open terminal A in project A:

```bash
gpuq submit --project project-a -- \
  python long_gpu_job_a.py
```

Open terminal B in project B:

```bash
gpuq submit --project project-b -- \
  python long_gpu_job_b.py
```

Open terminal C in project C:

```bash
gpuq submit --project project-c --priority critical -- \
  python urgent_gpu_job_c.py
```

Expected:

```text
project-a -> RUNNING
project-c -> QUEUED ahead of project-b
project-b -> QUEUED
```

Only one heavy job is running.

Close all three submission terminals.

Expected:

- project A continues.
- project C starts next.
- project B starts after C.
- logs remain accessible in a new shell.

Before project B begins, edit project B source.

Expected:

- queued project B runs the source snapshot from submission, not the edited live tree.

Then:

```bash
gpuq status
gpuq show <id>
gpuq logs <id>
gpuq doctor
```

must all provide coherent results.

Cancellation:

```bash
gpuq cancel <queued-id>
gpuq cancel <running-id>
```

must work as documented.

Claude Code user policy must be installed and visible from Claude's memory/config inspection.

Automated tests must pass.

This is the release gate.

---

# 37. Implementation Order

Claude Code should implement in this order and keep the repository runnable after each stage.

## Stage 1 — Skeleton and config

- pyproject
- package
- config
- state directories
- DB
- core models

Tests pass.

## Stage 2 — Task Spooler backend

- isolated socket
- capability detection
- initialization
- submit/list/show/log/remove/kill/promote
- JSON parsing

Tests pass with mocked and real TS where available.

## Stage 3 — Basic CLI

- init
- submit
- status/list
- show
- logs
- cancel
- promote
- doctor

At this point the queue should already be usable.

## Stage 4 — Snapshotting

- temp Git index
- ephemeral commit
- worktree/archive
- passthrough
- cleanup

Snapshot integration test passes.

## Stage 5 — Runner and provenance

- `_run`
- state transitions
- manifests
- environment capture
- exit code
- signal forwarding

## Stage 6 — Reliability

- reconcile
- crash cases
- cleanup
- concurrent CLI calls
- isolated test profiles
- doctor robustness

## Stage 7 — Claude integration

- policy installer
- status/remove
- optional safe launcher
- README agent workflow

## Stage 8 — MCP

- optional dependency
- stdio server
- same core service
- in-memory tests
- registration instructions

## Stage 9 — End-to-end validation

- automated tests
- smoke test
- real harmless GPU test
- fresh-shell installation test
- README cleanup
- final doctor

Do not reverse this order by building MCP/UI before the queue is proven.

---

# 38. Final Handoff Report Required From Claude Code

At completion, Claude Code should return a concise report containing:

```text
1. Installation path / repo path
2. `gpuq --version`
3. Task Spooler version and GPU capability status
4. `gpuq doctor` final output
5. automated test summary
6. smoke-test summary
7. whether a real NVIDIA GPU smoke test ran
8. Claude policy installation status
9. MCP status
10. exact commands to begin using GPUQ now
11. any remaining non-critical limitations
```

Do not finish with “implementation complete” without evidence.

---

# 39. Quick Start Target

The finished tool should reduce daily use to:

```bash
gpuq doctor

gpuq submit --project my-project -- \
  python train.py

gpuq status

gpuq logs 1 --follow
```

That is the product.

---

# 40. Source / API References

Use these current upstream references while implementing. Verify actual installed command behavior rather than relying on memory.

## GPU Task Spooler

Repository:
`https://github.com/justanhduc/task-spooler`

Installation:
`https://github.com/justanhduc/task-spooler/blob/master/INSTALL.md`

Important upstream capabilities to use:

- `TS_VISIBLE_DEVICES`
- `TS_SOCKET`
- `TS_SAVELIST`
- `TS_MAXFINISHED`
- `--set_gpu_free_perc`
- `--get_gpu_free_perc`
- `--set_logdir`
- `--get_logdir`
- `-M json`
- `-G`
- `-S`
- `-r`
- `-k`
- `-u`
- `-U`
- `-o`
- `-p`
- `-s`
- `-i`

## Claude Code persistent instructions

Current docs:
`https://code.claude.com/docs/en/memory`

User-level instructions:
`~/.claude/CLAUDE.md`

Project-level instructions:
`./CLAUDE.md` or `./.claude/CLAUDE.md`

Claude docs explicitly distinguish CLAUDE.md guidance from enforceable hooks/permissions; GPUQ should therefore treat the policy as behavioral guidance, not a hard security boundary.

## MCP Python SDK

Current stable documentation:
`https://py.sdk.modelcontextprotocol.io/`

Use v2 current stable line for new code.

---

# 41. Final Design Principle

The success condition is not “GPUQ has many features.”

The success condition is:

> Every agent can submit expensive work in seconds, continue coding, and trust that the machine will not launch another broker-managed heavy workload until it is safe.

Keep the implementation centered on that invariant.
