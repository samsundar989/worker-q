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

1. **Jobs run together only when they fit.** What decides is the footprint each
   job declares, not a slot count: a job starts when its `--ram/--vram/--cpus`
   fit alongside everything already running. Small jobs overlap, large ones
   serialise. Declare honestly - an undeclared job is charged a small default
   and may be admitted when it should have waited.
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

`critical` jumps ahead of everything queued. It interrupts a running job only if
that job was submitted `--preemptible`.

### See the queue

```bash
workerq status
workerq status --json          # for scripts and agents
```

```text
GPU 0: NVIDIA GeForce RTX 5090  2.9 / 31.8 GiB used  (91% free)
Concurrency: 4   GPU free threshold: 85%   Dispatcher: running

 ID  STATE     PRI       PROJECT          AGE/RUNTIME  GPU  BE  COMMAND
 58  RUNNING   normal    pokemon-ai        12m           1  14  python train.py ...
 61  RUNNING   normal    kaggriculture      3m           0  17  python features.py ...
 59  QUEUED    critical  arc-agi           4m wait       1  15  python evaluate.py ...
                         ^ needs 22.0 GiB VRAM; running job(s) reserve 20.0 GiB of 30.8 usable
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
 60  QUEUED  normal  biohub  4m w  starts ~12m, runs 1h  24G 4c  celltrack train, holdout_44b6
     ↳ needs 24.0 GiB RAM but only 9.1 GiB is free after the 10% floor
```

A job that asks for more than the machine has is rejected at submit time rather
than queued forever.

Undeclared jobs are charged a modest default (4 GiB RAM, 1 CPU) so they are
never treated as free — but estimate properly, since a job that under-declares
can be admitted when it should have waited.

`max_concurrent_jobs` (default 4) is a ceiling, not the scheduler - it exists so
a bug cannot launch twenty processes. Admission control decides what actually
runs. Raise or lower it with `workerq concurrency N`.

Tune the guard rails in `[resources]` (see `workerq resources` for current values):

```toml
[resources]
enforce = true
reserve_ram_gb = 8.0        # never handed out: OS, editors, agents
reserve_vram_gb = 4.0       # the desktop itself uses 3-4 GiB
reserve_cpus = 2
min_host_free_percent = 10  # floor a job may not eat into
max_commit_percent = 99     # hard stop; below this, commit only blocks
                            # when physical RAM is short too
default_ram_gb = 4.0        # charged to jobs that declare nothing
```

### Headroom: keeping the desktop usable

worker-q never hands out everything. Two layers, adjusted differently:

| | Held back | Adjust |
|---|---|---|
| RAM | `reserve_ram_gb`, plus a live `min_host_free_percent` floor no job may eat into | config |
| VRAM | `reserve_vram_gb` | config |
| CPU | `reserve_cpus`, **plus every job runs below normal priority** | config |

```bash
workerq resources                 # what is reserved and what is left
workerq config set resources.reserve_cpus 3
workerq restart                   # apply it; running jobs keep going
```

`config set` changes the standing baseline. The dispatcher reads those at
startup, so it tells you a restart is needed rather than silently doing
nothing. Only `max_concurrent_jobs` and the GPU threshold apply immediately.

Two more guards you rarely touch: the **pressure guard** stops new work and
displaces the newest `--preemptible` job if physical RAM stays below
`scheduling.pressure_free_percent`, and `gpu.free_memory_threshold_percent`
requires a device to be that free before a GPU job lands on it.

### The live dashboard

```bash
workerq top
```

| Key | |
|---|---|
| `j` `k` / `↑` `↓` | scroll the queue |
| `PgUp` `PgDn` `Home` `End` | page and jump |
| `g` | **gaming mode** on/off - hold back the `[gaming]` headroom in one key |
| `r` `R` | RAM held back, down / up 2 GiB |
| `v` `V` | VRAM held back, down / up 1 GiB |
| `c` `C` | CPU cores held back, down / up 1 |
| `0` | give it all back |
| `q` | quit |

Every change applies to the queue immediately - running jobs are never
displaced, the new limit governs what starts next. What gaming mode claims is
configurable:

```toml
[gaming]
ram_gb = 24.0
vram_gb = 24.0
cpus = 8
```

Piping `workerq top` somewhere still works; it is just read-only.

### Taking the machine back

To play a game, join a call, or just get the desktop back, claim resources from
the queue. It applies at once - no restart - and running jobs are left to
finish:

```bash
workerq reserve --ram 24 --vram 22 --cpus 8 --label gaming --for 2h
workerq reserve              # what is held, and what is left for jobs
workerq reserve --clear      # give it back
```

Anything blocked by a reserve says so by name. `--for` releases it
automatically, so a temporary claim cannot become a permanent mystery.

Job processes also run below normal priority, so the desktop stays responsive
when several share the CPUs. Turn that off with
`scheduling.background_priority = false`.

### How to pick the numbers

This is the thing people get wrong, and a wrong number does not error — the job
just waits, sometimes for hours. worker-q measures what jobs actually use, so it
can tell you:

```bash
workerq resources --verify          # declared vs actual, with a suggestion
workerq requests 103 --suggest      # what this command's own history says
```

```text
Job #103 declares 28.0 GiB RAM
Worst of 7 past run(s): 10.2 GiB
Suggested declaration: 16 GiB  (peak + 50% headroom)

Declaring 28 GiB reserves 12 GiB more than this command has ever needed,
which is what keeps it waiting.
Apply it: workerq requests 103 --ram 16
```

Got it wrong? Fix it in place. The job keeps its source snapshot and its
position in the queue:

```bash
workerq requests 103 --ram 16
```

**Declare against free memory, not installed memory.** A 64 GiB workstation may
only ever have ~30 GiB genuinely free, because editors, browsers and other
agents hold the rest. Admission compares your declaration against what is free,
so a 28 GiB request on such a machine is not "half the box" — it is more than is
ever available, and the job will never start. `workerq submit` warns when a
declaration has rarely been satisfiable here.

Over-declaring is safe but packs badly and keeps your own work waiting;
under-declaring is what takes the machine down.

---


## Saying what a job is, and how long it takes

A command line does not say what a job is for, what is waiting on it, or when it
will be done. worker-q cannot invent any of that, so it takes it from the worker
and shows it in `workerq top`:

```bash
workerq submit --project biohub --ram 24   --describe "120-epoch celltrack train, holdout_44b6"   --blocks "slice 067 promotion gate"   --eta 90m -- python -m celltrack train
```

`--eta` accepts `90m`, `1h30m`, `5400`. Both the description and the estimate can
be corrected once the job knows better than the person who queued it:

```bash
workerq eta 61 45m
workerq describe 61 "revised: 40 epochs, early-stopped" --blocks "slice 067"
```

### Where the estimate comes from

The queue would rather say nothing than say something confident and wrong, so
every ETA is tagged with its source, and `unknown` is a normal answer:

| Source | Meaning | Shown |
| --- | --- | --- |
| `progress` | the job reported its own completion fraction | green |
| `declared` | a worker passed `--eta`, or corrected it at runtime | cyan |
| `learned` | median of past successful runs of this same command | yellow |
| `unknown` | fewer than two comparable runs, and nobody said | dim |

Learning needs no setup. Every job records a **command signature** — the shape of
the command with flag *values*, paths and numbers stripped out — so
`train --fold A` and `train --fold B` count as runs of the same thing. After two
successful runs in a project, worker-q offers a median; runs older than 30 days
are ignored, because a command's cost changes as its code does.

Queued jobs also get a projected start, laid out over the free slots in dispatch
order. A job whose duration is unknown makes the jobs behind it unknown too,
rather than quietly optimistic.

### Reporting progress (the accurate option)

Every job is handed a file path in `$WORKERQ_PROGRESS`. Write a fraction to it and
the ETA stops being a guess — it becomes measured from the job's own pace, which
is the only source that knows epoch 3 is slower than epoch 90:

```python
import os
with open(os.environ["WORKERQ_PROGRESS"], "w") as fh:
    fh.write(f'{{"frac": {epoch / total}, "note": "epoch {epoch}/{total}"}}')
```

A bare `0.42`, a `42%`, or that JSON all work. worker-q polls the file every few
seconds; the note appears next to the percentage in the dashboard. Nothing about
the job changes if it never writes the file.

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

All four are stored exactly as given. The dispatcher orders by priority, then by
arrival, and starts a job when it fits. A job it cannot fit does not park the
queue: work behind it that *does* fit may start first, bounded so the blocked
job cannot be deferred indefinitely. **Priority never relaxes a resource-safety
rule**, and it displaces a running job only if that job opted in with
`--preemptible`.

```bash
workerq promote 42        # move a queued job to the front by hand
```

### Raising a job after you submitted it

```bash
workerq bump 42 critical      # this job now outranks everything below it
```

A raised job jumps ahead of everything queued that it now outranks. If the
machine is busy it can also **stop a running job** — but only one that was
submitted `--preemptible`.

### Preemption

```bash
workerq submit --project arc-agi --ram 20 --preemptible -- python train.py
```

`--preemptible` means *"safe to stop and re-run"*. A displaced job is **not**
cancelled and has **not** failed: it returns to the queue, keeps the same job
id, and runs its command again from the start.

That last part is the whole risk. **Only mark a job preemptible if re-running it
is safe** — it resumes from a checkpoint, or it is cheap to repeat. A six-hour
training run with no checkpointing should never be preemptible, because being
displaced throws the six hours away.

Nothing is displaced unless every one of these holds:

- preemption is enabled, and the waiting job **strictly outranks** the running one;
- the running job declared `--preemptible`;
- it has already run for `min_runtime_seconds` (default 60), so a burst of urgent
  work cannot leave nothing making progress;
- displacing it **actually lets the waiter start** — the admission check is re-run
  against the freed resources first, so no work is destroyed for nothing.

A displaced job writes a banner into its own log, and `workerq show` reports what
stopped it:

```text
Times preempted    1
Preempted by       58
```

Follow it to completion with:

```bash
workerq wait 42     # blocks, then exits with the job's own exit code
```

`wait` is the notification primitive: the job id never changes, so waiting on it
survives any number of preemptions.

Tune the guard rails in `[preemption]`:

```toml
[preemption]
enabled = true
require_opt_in = true      # false lets any lower-priority job be displaced
min_runtime_seconds = 60
max_preemptions = 3
grace_seconds = 30         # time to stop cleanly before the tree is killed
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
workerq config set core.max_concurrent_jobs 4
workerq config set gpu.free_memory_threshold_percent 85
```

Precedence: **CLI flag > `GPUQ_*` environment variable > `config.toml` >
built-in default.**

```toml
[core]
state_dir = "~/.local/state/gpuq"
max_concurrent_jobs = 4
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
| `workerq submit [--ram N --vram N --cpus N] [--describe T --blocks W --eta D] -- CMD` | Queue a job and return immediately. |
| `workerq status` / `workerq list` | Show the queue. `--json` for agents. |
| `workerq top` | Live dashboard: queue, pressure, memory owners. |
| `workerq report` | Why recent jobs failed, grouped by cause and project. |
| `workerq resources` | Capacity, headroom and enforced limits. |
| `workerq show ID` | Full detail and source provenance. |
| `workerq logs ID [--follow] [--tail N]` | Job output. |
| `workerq cancel ID [--force]` | Cancel queued or running work. |
| `workerq promote ID` | Move a queued job to the front. |
| `workerq bump ID LEVEL` | Raise one job's priority; may displace running work. |
| `workerq wait ID` | Block until a job finishes; exits with its exit code. |
| `workerq eta ID DURATION` | Set or correct a job's expected wall time. |
| `workerq describe ID [TEXT] [--blocks W]` | Say what a job is and what waits on it. |
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
