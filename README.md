# worker-q — a resource broker for machines shared by AI coding agents

One shared gate for heavy work - GPU, RAM and CPU - so several AI coding agents
can run in parallel across projects without their jobs colliding and taking the
machine down.

> The CLI is `workerq`. `worker-q` is kept as a working alias, since that is what
> earlier agent policies and queued jobs refer to.

Submit expensive work in seconds, keep coding, and trust that the machine will
not start another broker-managed heavy job until it is safe.

---

## Start using this now

```bash
workerq doctor                                   # check the machine is healthy

workerq submit --project my-project -- \
  python train.py --config configs/exp17.yaml # queue a job, return immediately

workerq status                                   # what is running / next
workerq logs 1 --follow                          # stream a job's output
```

That is the whole daily loop. Everything below is detail.

The four things worth knowing:

1. **Only one heavy job runs at a time** (`max_concurrent_jobs = 1`). A 3 GiB
   job may monopolise a 32 GiB GPU. That is deliberate: reliability beats
   utilisation.
2. **Jobs outlive your terminal.** Close the shell, close the editor, the job
   keeps running and the logs stay readable from any new shell.
3. **A queued job runs the source as it was at submission time.** Keep editing
   the repo the moment you have submitted; the job is unaffected.
4. **worker-q never runs your command directly.** If the queue is unreachable,
   submission fails loudly and tells you to run `workerq doctor`.

---

## Daily agent workflow

### Submit a training job

```bash
workerq submit --project arc-agi -- \
  python train.py --config configs/exp17.yaml
```

### Urgent blocking evaluation

```bash
workerq submit --project arc-agi --priority critical -- \
  python evaluate.py --checkpoint latest
```

`critical` jumps ahead of everything queued. It never interrupts a job that is
already running.

### See the queue

```bash
workerq status
workerq status --json          # for scripts and agents
```

```text
GPU 0: NVIDIA GeForce RTX 5090  2.9 / 31.8 GiB used  (91% free)
Concurrency: 1   GPU free threshold: 85%   Dispatcher: running

 ID  STATE     PRI       PROJECT          AGE/RUNTIME  GPU  BE  COMMAND
 58  RUNNING   normal    pokemon-ai        12m           1  14  python train.py ...
 59  QUEUED    critical  arc-agi           4m wait       1  15  python evaluate.py ...
 60  QUEUED    normal    biohub            1m wait       1  16  python sweep.py ...
```

### Follow logs

```bash
workerq logs 42 --follow
workerq logs 42 --tail 100
```

### Cancel

```bash
workerq cancel 42            # queued -> removed; running -> stopped
workerq cancel 42 --force    # skip the grace period, kill the process tree now
```

### Inspect source provenance

```bash
workerq show 42
```

Tells you the snapshot commit, the exact directory the job ran in, the assigned
`CUDA_VISIBLE_DEVICES`, exit code, and where the logs are.

### Check system health

```bash
workerq doctor
```

Exit code `0` healthy, `1` degraded but usable, `2` broken — do not submit.

---


## Declaring what a job needs

worker-q brokers **any** heavy workload, not just GPU work. Host RAM exhaustion is
the most common way a dev box falls over, and a slot count cannot see it.

```bash
workerq submit --project biohub --ram 24 --vram 12 --cpus 4 --   C:/Users/you/Documents/biohub/.venv/Scripts/python.exe -m celltrack train
```

`--ram` and `--vram` are peak GiB. A job starts only when its declared
footprint fits, judged two ways at once:

* against **measured free memory** right now, which is the only thing that
  accounts for workloads worker-q did not start;
* against the **sum of what running jobs reserved**, because a job that started
  moments ago has not grown to full size yet.

That is what lets several small jobs run together while two large ones
serialise, without you picking a slot count. A blocked job says why:

```text
 60  QUEUED   normal  biohub   4m wait  24G 4c
     ↳ needs 24.0 GiB RAM but only 9.1 GiB is free after the 10% floor
```

A job that asks for more than the machine has is rejected at submit time rather
than queued forever.

Undeclared jobs are charged a modest default (4 GiB RAM, 1 CPU) so they are
never treated as free — but estimate properly, since a job that under-declares
can be admitted when it should have waited.

To run more than one job at a time, raise the slot cap; admission control keeps
it honest:

```bash
workerq concurrency 4 --yes
```

Tune the guard rails in `[resources]` (see `workerq resources` for current values):

```toml
[resources]
enforce = true
reserve_ram_gb = 8.0        # never handed out: OS, editors, agents
reserve_cpus = 2
min_host_free_percent = 10  # floor a job may not eat into
max_commit_percent = 88     # hard stop; Windows fails allocations near this
default_ram_gb = 4.0        # charged to jobs that declare nothing
```

---


## Watching the queue and diagnosing crashes

```bash
workerq top        # live dashboard: queue, machine pressure, who holds memory
workerq report     # why recent jobs failed, and whose workload it was
workerq resources  # capacity, headroom, and the limits being enforced
```

`workerq top` refreshes in place and shows four things at once: VRAM / RAM /
commit-charge meters that turn amber then red under pressure, the live queue
with a wait reason under anything that is blocked, the largest memory consumers
tagged **worker-q** or **foreign**, and recently finished jobs with a one-line cause
for each failure. Ctrl-C exits. `--once` prints a single frame, which is what
you want in a script.

That `foreign` tag is usually the answer when the box falls over: it is a heavy
workload running outside the queue, which worker-q can see but cannot schedule
around.

`workerq report` classifies every recent failure - CUDA OOM, host OOM, killed,
missing file, import error, application exception - and groups them by project
and by the agent that submitted them. For failures caused by the machine rather
than the code, it prints what memory looked like at that moment:

```text
Last 24h  33 finished - 11 ok - 18 failed - 4 cancelled - success 38%

CAUSE                                  N
host out of memory                     1
job raised an exception                4

  #51 biohub host out of memory (exit 1, claude-code)
      MemoryError: could not read frame 78 ... the host is out of memory
      at the time: host 6% free, commit 94%
      -> declare --ram so worker-q holds the job until that much is actually free
```

Add `--pressure` to also list what peaked in that window, and `--json` for
anything you want to parse.

---

## Installation

Requires Python 3.11+, git, and (for GPU gating) a working NVIDIA driver.

```bash
uv tool install --from . worker-q        # from a clone
workerq init
workerq claude-policy install
workerq doctor
```

`workerq init` is idempotent: it creates the state directories, the database and
the dispatcher, and re-applies your configured concurrency and GPU threshold.

`scripts/bootstrap.sh` does all of the above in one step and finishes with a
non-destructive queue smoke test.

### Where things live

```text
~/.config/gpuq/config.toml          configuration
~/.local/state/gpuq/
    gpuq.sqlite3                    job metadata
    logs/job-000042.log             job output
    snapshots/42/repo               frozen source for job 42
    jobs/42/manifest.json           provenance, environment, result
    backend/queue.sqlite3           dispatcher queue state
    run/                            dispatcher lock, heartbeat, daemon log
```

---

## Teaching your agents to use it

```bash
workerq claude-policy install
```

This writes a marker-delimited block into `~/.claude/CLAUDE.md`, the user-level
memory file Claude Code reads across all projects. It is idempotent, backs the
file up before the first change, preserves every other instruction, and can be
removed cleanly:

```bash
workerq claude-policy status
workerq claude-policy remove
```

Claude's own documentation distinguishes `CLAUDE.md` guidance from enforceable
hooks and permissions, so treat this as behavioural guidance — one layer of
defence, not a hard boundary.

An optional stronger measure exists and is **not** enabled by default:

```bash
workerq claude-safe-launcher install
```

It creates `claude-gpu-safe`, which starts Claude Code with
`CUDA_VISIBLE_DEVICES=""` so a command the agent runs directly cannot reach the
GPU. Trade-off: legitimate lightweight GPU probes run directly will also see no
device. Queued worker-q jobs are unaffected — the dispatcher restores the real
device list.

---

## Source snapshots

The problem this solves: you submit job #42, then keep editing. Without
snapshots, #42 would run whatever the code looks like when it finally starts.

For a Git repository, `workerq submit` freezes the working tree at submission time
— tracked content, staged changes, unstaged changes, and untracked non-ignored
files — into an ephemeral commit built through a **temporary Git index**. Your
real index, working tree, branch and HEAD are never touched. The job then runs
in a detached worktree of that commit.

```bash
workerq show 42          # Snapshot commit: 9f613e2...
```

### Ignored data your job still needs

Datasets and checkpoints are usually gitignored, so they are not in the
snapshot. Link them in rather than copying:

```toml
# .gpuq.toml in your repo root
[snapshot]
passthrough = ["data", "datasets", "checkpoints"]
```

or per submission:

```bash
workerq submit --passthrough data --passthrough checkpoints -- python train.py
```

Directories become junctions/symlinks back to the live path — no bulk copying.
Relative passthrough paths may not escape the repository; only an explicit
absolute path may.

### Opting out

```bash
workerq submit --live-worktree -- python train.py   # run against the live tree
workerq submit --no-snapshot -- python train.py     # same, no snapshot at all
```

A non-Git directory refuses snapshot mode with a clear message rather than
pretending a live directory is immutable.

---

## Priorities

```text
critical   front of the queue
high       ahead of normal/low
normal     FIFO
low        behind everything else
```

All four are stored exactly as given. The dispatcher orders strictly by
priority, then by arrival. **No priority ever preempts a running job**, and
priority never relaxes a resource-safety rule.

```bash
workerq promote 42        # move a queued job to the front by hand
```

### Making a whole project more important

Rather than remembering `--priority` on every submission, set it once for the
project. Every worker on it inherits the setting, and queued jobs are re-ranked
immediately:

```bash
workerq priority arc-agi high --note "comp deadline Friday"
workerq priority                       # show all project policies
workerq priority arc-agi --clear       # back to the default
```

Precedence, most specific first:

1. `--priority` on the submission (a worker asking for something specific wins)
2. the project policy set by `workerq priority`
3. `[project] priority` in that repo's `.gpuq.toml`
4. `core.default_priority`

Use `workerq priority` for "this project matters *this week*" — it needs no repo
edits and no worker changes. Use `.gpuq.toml` for a project that is permanently
more or less important than the rest.

> Ordering is strict: a steady stream of `critical` work will keep `low` work
> waiting indefinitely. There is no aging in V1, so prefer `high` over
> `critical` for sustained importance, and reserve `critical` for genuinely
> blocking work.

---

## Concurrency and the GPU threshold

```bash
workerq concurrency               # show
workerq concurrency 2 --yes       # raise (warns; --yes required)
workerq gpu-threshold 85          # require 85% free VRAM before starting a GPU job
```

The GPU threshold is what protects you from *foreign* work — a job someone
started outside worker-q, or a browser holding VRAM. A queued GPU job waits until a
device is at least this free, and `workerq status` shows what it is waiting for.

On a desktop machine the compositor and browsers hold a few GiB permanently, so
a 90% threshold can stall the queue. `workerq init` and `workerq doctor` detect this
and suggest a value; they never change it silently.

---

## Configuration

```bash
workerq config show
workerq config set core.max_concurrent_jobs 1
workerq config set gpu.free_memory_threshold_percent 85
```

Precedence: **CLI flag > `GPUQ_*` environment variable > `config.toml` >
built-in default.**

```toml
[core]
state_dir = "~/.local/state/gpuq"
max_concurrent_jobs = 1
default_priority = "normal"
snapshot_mode = "git"
cleanup_successful_snapshots_after_days = 7
cleanup_failed_snapshots_after_days = 14
cancel_grace_seconds = 15

[gpu]
default_gpu_count = 1
free_memory_threshold_percent = 90
exclusive_by_default = true

[backend]
name = "local_dispatcher"
max_finished = 1000
poll_interval_seconds = 0.25

[claude]
install_user_policy = true
hide_cuda_in_safe_launcher = false
```

Set `GPUQ_PROFILE=test` to get a completely separate queue, database and state
directory — that is how the test suite avoids touching your real queue.

---

## Maintenance

```bash
workerq reconcile              # repair metadata after a crash or reboot
workerq cleanup --dry-run      # see what retention would remove
workerq cleanup                # remove expired snapshots and orphan temp files
workerq uninstall --dry-run    # see exactly what removal would touch
```

Cleanup never deletes an active job's snapshot or logs, never touches anything
outside the worker-q state directory, and keeps failed-job evidence for its own
longer retention window.

After a reboot:

```bash
workerq init && workerq reconcile && workerq status
```

Jobs that were **running** when the machine went down are reported `LOST`, not
silently marked complete. Jobs that were still **queued** are recovered and run.

---

## MCP (optional)

The CLI is the supported interface. An MCP adapter over the same core API is
available if you prefer tool calls:

```bash
uv tool install --from . --with 'mcp[cli]' worker-q
workerq mcp test          # build the server in-process, list its tools
workerq mcp command       # print the stdio command to register
```

Tools: `gpu_submit`, `gpu_status`, `gpu_job`, `gpu_logs`, `gpu_cancel`,
`gpu_promote`, `gpu_info`. It is a thin shim — all logic lives in
`GPUQService`, exactly as the CLI uses it. stdio only; no network listener.

---

## How it works

```text
Claude A ─┐
Claude B ─┼── workerq CLI ──► GPUQ Core ──► dispatcher daemon ──► your GPU job
Claude C ─┘                (SQLite,       (one detached
                            snapshots)     process, N slots)
```

`workerq submit` validates, snapshots, records the job and enqueues it — then
returns. A single detached dispatcher daemon owns execution: it is the only
process that launches user work, which is what makes the one-job-at-a-time
invariant hold across unrelated terminals.

Backends sit behind a `SchedulerBackend` protocol
(`src/workerq/backends/base.py`). V1 ships `LocalDispatcherBackend`. On Linux the
same protocol is the seam where GPU Task Spooler would slot in, and it is where
a remote or Slurm backend goes later.

See [docs/architecture.md](docs/architecture.md),
[docs/troubleshooting.md](docs/troubleshooting.md) and
[docs/future-slurm.md](docs/future-slurm.md).

---

## Command reference

| Command | Purpose |
| --- | --- |
| `workerq init` | Create state, database and dispatcher. Idempotent. |
| `workerq submit [--ram N --vram N --cpus N] -- CMD` | Queue a job and return immediately. |
| `workerq status` / `workerq list` | Show the queue. `--json` for agents. |
| `workerq top` | Live dashboard: queue, pressure, memory owners. |
| `workerq report` | Why recent jobs failed, grouped by cause and project. |
| `workerq resources` | Capacity, headroom and enforced limits. |
| `workerq show ID` | Full detail and source provenance. |
| `workerq logs ID [--follow] [--tail N]` | Job output. |
| `workerq cancel ID [--force]` | Cancel queued or running work. |
| `workerq promote ID` | Move a queued job to the front. |
| `workerq priority [PROJECT LEVEL]` | Show/set a project's default priority. |
| `workerq doctor` | Health checks. Exit 0/1/2. |
| `workerq gpu` | GPU inventory and who holds VRAM. |
| `workerq reconcile` | Repair metadata after a crash. |
| `workerq cleanup` | Retention for snapshots and temp files. |
| `workerq concurrency [N]` | Show/set concurrent job limit. |
| `workerq gpu-threshold [N]` | Show/set required free VRAM percent. |
| `workerq config show/get/set` | Configuration. |
| `workerq claude-policy install/status/remove` | Agent policy block. |
| `workerq mcp command/test/serve` | Optional MCP adapter. |
| `workerq uninstall --dry-run` | Preview removal. |

Every command that produces data supports `--json`, which writes only JSON to
stdout. Errors always go to stderr.


## Windows notes

worker-q runs natively on Windows and gates Windows-native CUDA jobs (your project
venv's `python.exe` with `torch+cuXXX`), which is where the OOM risk actually
lives on this machine.

**Name the interpreter with an absolute path.** A queued job runs a frozen
snapshot of your repository, and `.venv` is normally gitignored — so a relative
`.venv/Scripts/python.exe` will not exist inside the snapshot:

```bash
# good
workerq submit --project my-project -- \
  C:/Users/you/Documents/my-project/.venv/Scripts/python.exe train.py

# also fine
workerq submit --project my-project --passthrough .venv -- \
  .venv/Scripts/python.exe train.py
```

worker-q tells you this explicitly if a job fails that way.

**No terminal windows.** worker-q's dispatcher and job wrappers run under
`pythonw.exe`, and every helper command it shells out to (`nvidia-smi`, `git`,
`taskkill`) is launched with `CREATE_NO_WINDOW`. Both are necessary: a
console-less parent that launches a console program makes Windows open a new
*visible* console for it, and the dispatcher polls `nvidia-smi` regularly. The
test suite asserts that no visible window ever appears.

**Cancellation** uses Windows Job Objects, so `workerq cancel` kills the whole
process tree — including detached grandchildren such as dataloader workers or
`torchrun` ranks. Windows cannot deliver a POSIX `SIGTERM` to a console-less
child, so the polite stop is best-effort and the tree kill is the guarantee;
`--force` skips the grace period.

**`--shell`** runs through `cmd.exe /c` on Windows (`/bin/sh -c` elsewhere).

**Job encoding.** Logs are UTF-8, so jobs get `PYTHONIOENCODING=utf-8` unless
you override it with `--env`.


## License

MIT. See [LICENSE](LICENSE).

---
