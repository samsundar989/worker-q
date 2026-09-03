# Troubleshooting

Start here:

```bash
workerq doctor
```

Exit code `0` healthy, `1` degraded but usable, `2` broken — do not submit.
Every check prints a `->` hint when it is not `PASS`.

---

## A job is stuck in QUEUED and nothing is running

`workerq status` shows the reason under the job. The usual causes:

### "waiting for GPU memory: GPU0 85% free < 90% required"

The free-VRAM gate is doing its job — but on a desktop the compositor and
browsers permanently hold a few GiB, so the default 90% can never be met.

```bash
workerq gpu                    # see who actually holds VRAM
workerq gpu-threshold 80       # pick a threshold above your idle baseline
```

Choose a value a little below your true idle free percentage. Too low and a
worker-q job will start on top of a foreign CUDA process; too high and the queue
stalls. `workerq doctor` warns whenever the current threshold would block.

### Dispatcher not running

```bash
workerq status     # "Dispatcher: NOT RUNNING"
workerq init       # starts it; idempotent
```

If it will not start, read the daemon's own output:

```bash
cat ~/.local/state/gpuq/run/dispatcher.out    # startup errors
cat ~/.local/state/gpuq/run/dispatcher.log    # per-tick activity
```

### The machine is genuinely full

`workerq status` prints the reason under each queued job — which resource is
short, and how much of it running jobs have reserved. `workerq promote <id>`
moves a job to the front of the queue, but it still has to fit.

If the job it is waiting behind was submitted `--preemptible`, `workerq bump
<id> critical` can displace it. Otherwise the running work finishes first.

### It says there is plenty of RAM free, and my job still will not start

Admission compares your declared `--ram` against what is *free*, minus the
`min_host_free_percent` floor — not against installed RAM. On a machine with a
large steady baseline (editors, browsers, several agents) those are very
different numbers, and a job can be accepted at submit and then wait for
headroom that never arrives.

`workerq resources --verify` shows what past jobs actually used against what
they declared, with a suggested number. For one job:

```bash
workerq requests 103 --suggest      # what its own history says it needs
workerq requests 103 --ram 16       # fix it in place, keeping its queue position
```

Over-declaring is safe but packs badly, and a declaration well above real usage
is the usual cause of a job that queues forever. Submitting now warns when a
declaration has rarely been satisfiable on this machine, or when it is far above
what the same command has historically used.

---

## `gpuq: command not found`

The executable lives in `~/.local/bin`. Confirm it is on PATH:

```bash
ls ~/.local/bin/gpuq*
echo $PATH
```

On Windows, `%USERPROFILE%\.local\bin` must be in the **user** PATH. After
changing it, open a new terminal — existing shells keep the old PATH.

---

## "the worker-q dispatcher is not running and could not be started"

Submission refuses rather than running your command directly, on purpose.

```bash
workerq doctor
workerq init
```

If `doctor` reports **"No stale dispatcher: FAIL — lock held but heartbeat is
Nm old"**, a wedged daemon is holding the lock. It names the PID:

```bash
taskkill /PID <pid> /F      # Windows
kill <pid>                  # POSIX
workerq init
```

---

## A job failed immediately

```bash
workerq logs <id>
```

The runner prints a banner with the execution directory, snapshot commit and
resolved command before your program's output, which usually identifies the
problem at a glance.

### "No such file or directory" for a script that exists

The job runs in the **snapshot**, not your live directory. Files that are
gitignored are not in the snapshot. Link them in:

```bash
workerq submit --passthrough data --passthrough checkpoints -- python train.py
```

or permanently, in your repo's `.gpuq.toml`:

```toml
[snapshot]
passthrough = ["data", "datasets", "checkpoints"]
```

Check what a job actually saw:

```bash
workerq show <id>          # Execution cwd, Snapshot commit, Passthrough
```

### The wrong version of my code ran

That is snapshot semantics working correctly: a queued job runs the source as
it was **at submission time**. `workerq show <id>` prints the snapshot commit, and
the tree is still on disk at the printed snapshot path.

If you want the live tree, opt out explicitly:

```bash
workerq submit --live-worktree -- python train.py
```

### `ModuleNotFoundError` for a package that is installed

The job inherits the dispatcher's environment, not your current shell's. If
your project uses a virtualenv, name its interpreter explicitly:

```bash
workerq submit -- .venv/Scripts/python.exe train.py     # Windows
workerq submit -- .venv/bin/python train.py             # POSIX
```

This is the recommended form regardless — it makes the manifest unambiguous
about which interpreter produced a result.

---

## `UnicodeEncodeError` in a job

worker-q sets `PYTHONIOENCODING=utf-8` for jobs because logs are written and read
as UTF-8. If your program needs a different encoding, override it:

```bash
workerq submit --env PYTHONIOENCODING=cp1252 -- python legacy.py
```

---

## Stray terminal windows

worker-q launches its dispatcher and job wrappers with `pythonw.exe`, which creates
no console at all, so worker-q itself should contribute **zero** terminal windows.

Check what is actually worker-q's:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*worker-q*' } |
  Select-Object ProcessId, Name, CommandLine
```

You should see at most two entries per active profile (a launcher plus the real
interpreter), both `pythonw.exe`. Anything else on the machine opening consoles
is not gpuq.

To count console hosts belonging to worker-q specifically, look for `conhost.exe`
whose parent is one of those PIDs — there should be none.

Stop a dispatcher cleanly (never `taskkill` it; running jobs are tracked
through it):

```bash
workerq _stop-daemon                      # current profile
GPUQ_PROFILE=smoke workerq _stop-daemon   # a named profile
```

A dispatcher left over from an old profile is harmless but pointless; stopping
it does not affect other profiles. Note that a job's *own* command may create a
console of its own while it runs — that belongs to your program, and it goes
away when the job ends.

---

## My box is crashing / jobs die with MemoryError

Start here:

```bash
workerq report --pressure     # what failed, why, and what held memory
workerq top                   # live view while it is happening
```

`workerq report` separates failures caused by **the machine** (CUDA OOM, host OOM,
killed) from failures caused by **the job's own code**, and groups them by
project and submitting agent. If the verdict blames resource exhaustion, the
usual cause is one of these two.

### Something heavy is running outside the queue

This is the common one. A slot count cannot see work worker-q did not start, so a
single unqueued job can exhaust the box while worker-q believes it is idle.

`workerq top` tags every large process **worker-q** or **foreign**, and `workerq doctor`
raises "Unqueued heavy workloads". Fix it by submitting that work:

```bash
workerq submit --project the-other-project --ram 20 -- <its command>
```

### Jobs are not declaring what they need

An undeclared job is charged only a small default, so worker-q may admit it when it
should have waited. Declare real numbers:

```bash
workerq submit --project biohub --ram 24 --cpus 4 -- python -m celltrack train
```

`workerq show <id>` prints what a job actually requested.

---

## A job is QUEUED and worker-q says it is waiting for RAM

That is admission control, and it is preventing a crash rather than causing a
problem. `workerq status` prints the reason:

```text
 60  QUEUED  normal  biohub  4m wait
     ↳ needs 24.0 GiB RAM but only 9.1 GiB is free after the 10% floor
```

It starts on its own once the memory frees up. Check what is holding it with
`workerq top`. If the request was simply too pessimistic, cancel and resubmit with
a smaller `--ram`.

Current limits and headroom:

```bash
workerq resources
```

If the machine is genuinely bigger than worker-q thinks it can use, adjust the
guard rails rather than disabling them:

```bash
workerq config set resources.reserve_ram_gb 6
workerq config set resources.max_commit_percent 90
```

Turning enforcement off entirely (`resources.enforce = false`) restores the old
behaviour where only the slot count limits jobs - and with it the crashes.

---

## "Commit charge 91% of the limit" in doctor

Windows fails allocations as the system commit charge approaches its limit,
even while physical RAM still looks free, so worker-q stops starting jobs before
that point. This is reported as **degraded**, not broken: submitting still
works, jobs simply wait, and they resume by themselves.

If it stays high, something large is resident. `workerq top` names it. Growing the
page file raises the limit if that is genuinely what you want.

---

## My job says PREEMPTED — did it fail?

No. It was not cancelled and it did not crash: a higher-priority job displaced
it. It keeps the same job id, went back to the queue, and will run its command
again from the start.

**Do not resubmit it** — that would duplicate the work. Track the original:

```bash
workerq show 42     # "Times preempted", "Preempted by"
workerq wait 42     # blocks until it finishes, exits with its exit code
```

Its log carries a banner naming what displaced it.

### Why was it eligible at all?

Only because it was submitted `--preemptible`. Without that flag nothing
displaces a running job. If a job should never be interrupted, submit it without
the flag — or remove `preemptible = true` from the project's `.gpuq.toml`.

### It keeps getting preempted

`max_preemptions` (default 3) caps this: after that many displacements a job
stops being a candidate, so it cannot be starved indefinitely. If it is still
thrashing, the usual cause is a stream of `critical` submissions — check with:

```bash
workerq report
workerq priority              # is a whole project pinned high?
```

### Turning preemption off

```bash
workerq config set preemption.enabled false
```

Running jobs then always finish, and urgent work waits for a free slot.

---

## The ETA says "unknown", or is obviously wrong

`unknown` is deliberate, not a failure. worker-q shows an estimate only when it
has a defensible source, and it never guesses from a command line alone. Check
which source a job has:

```bash
workerq show <id> --json | python -c "import json,sys; print(json.load(sys.stdin)['estimate'])"
```

* **`unknown`** — nobody passed `--eta`, the job reports no progress, and this
  command has fewer than two successful runs in this project in the last 30
  days. Pass `--eta`, or have the job write `$WORKERQ_PROGRESS`.
* **`learned` but wrong** — the median comes from past runs of the same command
  *shape*. If `--epochs 5` and `--epochs 500` are the same shape, they are
  pooled, because flag values are deliberately stripped from the signature.
  Declare `--eta` on the outliers; a declared estimate always beats a learned one.
* **`declared` but wrong** — someone's guess has gone stale. Correct it in place
  with `workerq eta <id> <duration>`; there is no penalty for revising.
* **`progress` but jumpy** — the job's reported fraction is not linear in time
  (a slow first epoch, a long final checkpoint). The estimate is honest about
  what the job said; smooth the reported fraction if it matters.

A queued job showing `starts: unknown` is usually behind a job whose own
duration is unknown. That is intentional: an unknown duration poisons everything
behind it in the projection rather than producing a confident wrong start time.

Progress written but not showing? The job must write to the path in
`$WORKERQ_PROGRESS`, not a path of its own choosing, and worker-q polls it every
few seconds — a job shorter than one poll may finish before it is read.

---

## Cancel did not stop everything

```bash
workerq cancel <id> --force
```

Without `--force`, worker-q attempts a polite stop first and waits
`core.cancel_grace_seconds` (default 15) before killing the tree. On Windows a
console-less child cannot receive a polite signal at all, so the grace period
usually just elapses — `--force` skips it.

If a process survived, worker-q refused to kill it because it could not prove the
PID was still that job's process. `workerq show <id>` reports the PID it recorded.

---

## Jobs after a reboot

Task queues do not survive a reboot on their own:

```bash
workerq init && workerq reconcile && workerq status
```

- Jobs that were **queued** are recovered and will run.
- Jobs that were **running** when the machine went down are marked `LOST`.
  They are never silently reported as complete.

There is no automatic restart-on-login service in V1. Add one yourself if you
want it (a user systemd unit, or a Task Scheduler entry running `workerq init`).

---

## `workerq status` disagrees with reality

```bash
workerq reconcile --dry-run     # what it would change
workerq reconcile               # repair
```

Reconciliation compares the metadata database with the dispatcher's queue and
repairs drift after crashes. It never resurrects a cancelled job and never
alters a job that already finished.

---

## Two databases?

Yes, and that is deliberate:

- `gpuq.sqlite3` — worker-q's job metadata.
- `backend/queue.sqlite3` — the dispatcher's own queue state.

Keeping them separate is what makes the execution backend swappable.
`workerq reconcile` is the sanctioned way to make them agree.

---

## Disk filling up

```bash
workerq cleanup --dry-run
workerq cleanup
workerq cleanup --older-than 3d
```

Snapshots are Git worktrees and cost roughly one working tree each. Retention
defaults to 7 days for successful jobs and 14 for failed ones, so failure
evidence outlives success.

Cleanup never touches an active job's snapshot, anything outside the worker-q
state directory, or the live data behind a passthrough link.

---

## Starting over

```bash
workerq uninstall --dry-run     # exactly what would be removed
workerq uninstall --execute     # stops the dispatcher, removes the policy block
workerq uninstall --execute --purge   # also deletes the state directory
```

Source repositories are never touched.

---

## Isolating an experiment from your real queue

```bash
GPUQ_PROFILE=scratch workerq init
GPUQ_PROFILE=scratch workerq submit --project test -- python x.py
```

A profile gets its own state directory, database, config and dispatcher. This
is how the test suite and `scripts/smoke_test.sh` avoid touching production.
