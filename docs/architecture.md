# GPUQ architecture

## The one invariant

> Every agent can submit expensive work in seconds, continue coding, and trust
> that the machine will not launch another broker-managed heavy workload until
> it is safe.

Everything below exists to hold that invariant. Where a design choice traded
utilisation for reliability, reliability won.

## Shape

```text
Claude A ─┐
Claude B ─┼── gpuq CLI ──┐
Claude C ─┘              │
                         ▼
                 ┌──────────────────┐
                 │  GPUQ Core API   │   config, SQLite metadata,
                 │  (GPUQService)   │   snapshots, validation
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ SchedulerBackend │   backends/base.py (Protocol)
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ LocalDispatcher  │   shared queue DB + a single
                 │ Backend + daemon │   detached dispatcher process
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  gpuq _run       │   runner wrapper: provenance,
                 │  (per job)       │   signals, exit code
                 └────────┬─────────┘
                          ▼
                      NVIDIA GPU
```

The optional MCP server sits beside the CLI and calls the *same*
`GPUQService`. It contains no scheduling logic.

## Why a dispatcher daemon

The spec's Linux design uses GPU Task Spooler as the execution backend, for a
good reason: do not write a scheduler when a proven one exists.

This machine is Windows, and Task Spooler is Linux-only — it is built on Unix
domain sockets, `fork`, and POSIX process groups. Running it inside WSL would
gate WSL processes, while the workloads that actually need gating here are
Windows-native (`.venv\Scripts\python.exe` with `torch+cu129`). A queue that
cannot see the jobs it is meant to serialise is not a safety mechanism.

So GPUQ ships a dispatcher of equivalent *scope* — not an ambitious scheduler
— behind the `SchedulerBackend` protocol that the spec defines for exactly
this purpose. It provides only what the backend contract requires:

| Capability | How |
| --- | --- |
| persistent, terminal-independent queue | SQLite queue DB + detached daemon |
| concurrent-slot limit | `slots` in queue meta, enforced in the dispatch loop |
| GPU free-memory gating | `nvidia-smi` poll before claiming a device |
| `CUDA_VISIBLE_DEVICES` assignment | device allocator in the dispatcher |
| per-job logs | stdout/stderr redirected to `logs/job-NNNNNN.log` |
| labels | `gpuq:<id>:<project>:<priority>` |
| job reordering | `priority_rank` + `position` columns |
| process-group termination | Windows Job Objects, `taskkill /T` fallback |
| machine-readable state | the queue DB itself; never scraped text |

Nothing above the backend module knows which backend is in use. Adding
`TaskSpoolerBackend` on a Linux host means writing one module and changing
`backend.name` in the config.

### No IPC

The dispatcher has no socket, port or protocol. The queue database *is* the
channel: the CLI writes intent (enqueue, cancel flag, promote), the daemon
polls and acts. SQLite in WAL mode with `BEGIN IMMEDIATE` gives the atomicity;
`claim_for_start` is a conditional `UPDATE ... WHERE state='QUEUED'`, so two
dispatchers could not double-start a job even if two somehow existed.

This removes an entire class of failure (port conflicts, orphaned listeners,
auth) and satisfies the spec's "no unauthenticated network server" rule by
having no server at all. The cost is up to `poll_interval_seconds` (0.25s)
of dispatch latency, which is irrelevant for jobs measured in minutes.

### Single instance

The daemon holds an exclusive lock (`msvcrt.locking` / `flock`) on
`run/dispatcher.lock` for its entire life. Start-up is race-free by
construction: any process may *spawn* a daemon; the spawned process tries to
take the lock and exits silently if it loses. The starter then waits for a
fresh heartbeat rather than assuming success.

`doctor` distinguishes "no daemon" from "lock held but heartbeat stale" — the
second is a wedged dispatcher and is reported as a failure, not a warning.

## Job lifecycle

```text
PREPARING ──► QUEUED ──► RUNNING ──► SUCCEEDED
     │           │          │      ╲► FAILED
     │           │          │       ╲► CANCELLED
     ╰───────────┴──────────┴────────► LOST
```

Terminal states are immutable. `Database.update_job` validates every
transition against `ALLOWED_TRANSITIONS`, so a cancelled job can never be
resurrected and a finished job can never revert to `QUEUED` — including by
reconciliation, which uses `try_update_state` and simply declines when the
transition is illegal.

### Submission (crash-safe ordering)

1. validate the request;
2. insert the row as `PREPARING`;
3. build the snapshot;
4. record snapshot metadata and write `manifest.json`;
5. enqueue with the backend;
6. store `backend_job_id`;
7. mark `QUEUED`.

The user's command never runs before steps 1–7 succeed. If the backend is
unreachable, submission fails non-zero and tells the user to run `gpuq doctor`
— it never "helpfully" runs the command directly, which would be precisely the
OOM the tool exists to prevent.

A crash between 5 and 6 leaves a `PREPARING` row with no backend id. The
backend label carries the gpuq job id, so `gpuq reconcile` re-attaches it. A
`PREPARING` row older than five minutes with no matching backend job becomes
`LOST` — never `SUCCEEDED`.

### Why the runner takes only a job id

The backend executes `gpuq _run <job_id>` — the user's argv is *not* on that
command line. The runner reads it back from the database as JSON.

The spec requires that arguments containing spaces, quotes, globs, `=`,
Unicode and shell metacharacters round-trip exactly. On Windows every
`Popen(list)` is joined by `list2cmdline` and re-parsed by the child's C
runtime; that round-trip has real edge cases. Passing the argv through JSON in
SQLite has none. `gpuq _run <id> -- <argv...>` still works for direct testing.

## Snapshots

`git worktree add --detach` on an ephemeral commit built through a
**temporary index** (`GIT_INDEX_FILE`), so the user's real index, working tree,
branch and HEAD are untouched. The commit is anchored under
`refs/gpuq/snapshots/<id>` so `git gc` cannot prune a queued job's source.

Passthrough paths (datasets, checkpoints) are linked, not copied: directory
junctions on Windows, symlinks elsewhere.

That last point creates the sharpest hazard in the codebase. A junction looks
like an ordinary directory to `os.walk` — `os.path.islink` returns **False**
for it — so a naive recursive delete of a snapshot would follow the junction
and destroy the live dataset. `snapshot.unlink_reparse_points` detaches every
junction and symlink *before* any recursive removal runs, including before
`git worktree remove`. `tests/unit/test_snapshot.py::
test_cleanup_does_not_delete_live_passthrough_data` exists because an earlier
implementation did exactly this and deleted the fixture's data.

## Cancellation

Windows cannot deliver a POSIX `SIGTERM` to a console-less child, so
cancellation is a two-stage escalation with a guaranteed backstop:

1. the CLI sets a cancel flag in the queue DB;
2. the runner (and the dispatcher) see it, attempt a polite `CTRL_BREAK`;
3. after `cancel_grace_seconds`, the process tree is terminated via the Job
   Object; `--force` skips straight to this.

Job Objects — not PID lists — are what make the *tree* die, so a job that
spawned detached grandchildren (a `torchrun`, a dataloader pool) is fully
cleaned up. `test_cancellation_kills_the_whole_process_tree` verifies it.

`KILL_ON_JOB_CLOSE` is deliberately **not** set: if the dispatcher dies, a
multi-hour training run must keep running. The consequence is that a restarted
dispatcher has no handle for jobs it inherited, so it *adopts* them (tracking
pid + creation time) and reaps them by watching the process disappear. A kill
after a restart falls back to `taskkill /T`, but only after
`winproc.pid_matches` confirms the PID's creation time still matches — a
recycled PID belonging to unrelated work is never killed.

### Console hygiene

gpuq's background processes are launched with **`pythonw.exe`**, not
`python.exe`, and this is load-bearing rather than cosmetic.

A virtualenv's `python.exe` on Windows is a launcher that re-execs the real
interpreter, and the real interpreter gets a console. Measured: `DETACHED_PROCESS`,
`CREATE_NO_WINDOW`, and both combined *all* leave a `conhost.exe` behind — the
CreationFlags are not the lever, the interpreter is. `pythonw.exe` is built
against the Windows subsystem and creates none.

Without this, every dispatcher and every job wrapper leaked a console host that
the user sees as terminal windows accumulating. `tests/unit/test_winproc.py`
guards it by counting console hosts **within the launched process tree** —
counting them system-wide is unusable as an assertion, because other tools on a
developer machine open and close consoles constantly.

Consequence to keep in mind: under `pythonw.exe`, `sys.stdout` is `None` unless
redirected. The dispatcher always redirects the runner's output to the job log,
and the runner falls back to `DEVNULL` if it ever is not.

### PID identity

A PID alone is not an identity: Windows recycles them. Every stored PID is
paired with the process creation time from `GetProcessTimes`, and
`terminate_tree(pid, expected_creation=...)` refuses to act when they disagree.

One consequence worth knowing: a virtualenv `python.exe` may be a *trampoline*
that re-execs the real interpreter as a child. The PID the dispatcher launched
and the PID the runner reports for itself are then both correct and different.
`own_gpu_pids()` collects both, or `doctor` would report gpuq's own job as a
foreign GPU process.

## Foreign workloads

Broker-managed jobs are safe from each other at `max_concurrent_jobs = 1`.
Work started *outside* gpuq is not something gpuq can control, and the tool
does not pretend otherwise. Two partial mitigations:

- **Free-memory threshold.** A queued GPU job waits until a device is at least
  `gpu.free_memory_threshold_percent` free, so it will not pile onto a GPU
  something else is already using. `gpuq status` shows what a job is waiting
  for rather than appearing mysteriously stuck.
- **Agent policy.** `gpuq claude-policy install` writes behavioural guidance
  into `~/.claude/CLAUDE.md`. Claude's documentation distinguishes CLAUDE.md
  from hooks and permissions, so this is guidance, not enforcement.

On a desktop GPU the compositor and browsers hold a few GiB permanently, so a
90% threshold can stall the queue outright. `gpuq init` and `gpuq doctor`
detect this and *recommend* a value; neither changes it silently.

## Files on disk

```text
~/.config/gpuq/config.toml            configuration
~/.local/state/gpuq/
    gpuq.sqlite3                      job metadata (WAL)
    backend/queue.sqlite3             dispatcher queue state (WAL)
    logs/job-000042.log               combined stdout/stderr
    snapshots/42/repo/                detached worktree
    jobs/42/manifest.json             provenance
    jobs/42/environment.json          interpreter, CUDA, GPU inventory at start
    jobs/42/result.json               state and exit code
    run/dispatcher.lock               single-instance lock
    run/dispatcher.log                daemon log
```

Two databases, deliberately. The core DB is gpuq's; the queue DB belongs to
the backend. Keeping them apart is what makes the backend swappable — and the
spec's reconciliation model already assumes the two can disagree after a
crash, which is what `gpuq reconcile` repairs.

## Concurrency and multiple GPUs

`max_concurrent_jobs = 1` is the shipped default and the tested configuration.
The dispatcher does allocate distinct devices per job, so raising the slot
count on a multi-GPU host is coherent — but it is opt-in behind `--yes` and a
loud warning, because two jobs on one GPU is exactly the OOM this tool exists
to prevent.

Head-of-line blocking when the GPU is busy is intentional: a queued `critical`
job must not be overtaken by a `low` one merely because it needs a device that
is momentarily unavailable.

## Deferred by design

Remote workers, Slurm, fractional VRAM scheduling, MIG, preemption and
checkpointing are out of scope for V1. The schema reserves `node`,
`minimum_vram_gb` and `estimated_duration_seconds` so adding them later is a
migration, not a redesign. See [future-slurm.md](future-slurm.md).
