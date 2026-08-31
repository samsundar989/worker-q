# Troubleshooting

Start here:

```bash
gpuq doctor
```

Exit code `0` healthy, `1` degraded but usable, `2` broken — do not submit.
Every check prints a `->` hint when it is not `PASS`.

---

## A job is stuck in QUEUED and nothing is running

`gpuq status` shows the reason under the job. The usual causes:

### "waiting for GPU memory: GPU0 85% free < 90% required"

The free-VRAM gate is doing its job — but on a desktop the compositor and
browsers permanently hold a few GiB, so the default 90% can never be met.

```bash
gpuq gpu                    # see who actually holds VRAM
gpuq gpu-threshold 80       # pick a threshold above your idle baseline
```

Choose a value a little below your true idle free percentage. Too low and a
gpuq job will start on top of a foreign CUDA process; too high and the queue
stalls. `gpuq doctor` warns whenever the current threshold would block.

### Dispatcher not running

```bash
gpuq status     # "Dispatcher: NOT RUNNING"
gpuq init       # starts it; idempotent
```

If it will not start, read the daemon's own output:

```bash
cat ~/.local/state/gpuq/run/dispatcher.out    # startup errors
cat ~/.local/state/gpuq/run/dispatcher.log    # per-tick activity
```

### Another job is genuinely running

That is the design. `gpuq status` shows the running job. Use
`gpuq promote <id>` to move a queued job to the front — it will still wait for
the current job to finish, because gpuq never preempts.

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

## "the gpuq dispatcher is not running and could not be started"

Submission refuses rather than running your command directly, on purpose.

```bash
gpuq doctor
gpuq init
```

If `doctor` reports **"No stale dispatcher: FAIL — lock held but heartbeat is
Nm old"**, a wedged daemon is holding the lock. It names the PID:

```bash
taskkill /PID <pid> /F      # Windows
kill <pid>                  # POSIX
gpuq init
```

---

## A job failed immediately

```bash
gpuq logs <id>
```

The runner prints a banner with the execution directory, snapshot commit and
resolved command before your program's output, which usually identifies the
problem at a glance.

### "No such file or directory" for a script that exists

The job runs in the **snapshot**, not your live directory. Files that are
gitignored are not in the snapshot. Link them in:

```bash
gpuq submit --passthrough data --passthrough checkpoints -- python train.py
```

or permanently, in your repo's `.gpuq.toml`:

```toml
[snapshot]
passthrough = ["data", "datasets", "checkpoints"]
```

Check what a job actually saw:

```bash
gpuq show <id>          # Execution cwd, Snapshot commit, Passthrough
```

### The wrong version of my code ran

That is snapshot semantics working correctly: a queued job runs the source as
it was **at submission time**. `gpuq show <id>` prints the snapshot commit, and
the tree is still on disk at the printed snapshot path.

If you want the live tree, opt out explicitly:

```bash
gpuq submit --live-worktree -- python train.py
```

### `ModuleNotFoundError` for a package that is installed

The job inherits the dispatcher's environment, not your current shell's. If
your project uses a virtualenv, name its interpreter explicitly:

```bash
gpuq submit -- .venv/Scripts/python.exe train.py     # Windows
gpuq submit -- .venv/bin/python train.py             # POSIX
```

This is the recommended form regardless — it makes the manifest unambiguous
about which interpreter produced a result.

---

## `UnicodeEncodeError` in a job

gpuq sets `PYTHONIOENCODING=utf-8` for jobs because logs are written and read
as UTF-8. If your program needs a different encoding, override it:

```bash
gpuq submit --env PYTHONIOENCODING=cp1252 -- python legacy.py
```

---

## Stray terminal windows

gpuq launches its dispatcher and job wrappers with `pythonw.exe`, which creates
no console at all, so gpuq itself should contribute **zero** terminal windows.

Check what is actually gpuq's:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*gpuq*' } |
  Select-Object ProcessId, Name, CommandLine
```

You should see at most two entries per active profile (a launcher plus the real
interpreter), both `pythonw.exe`. Anything else on the machine opening consoles
is not gpuq.

To count console hosts belonging to gpuq specifically, look for `conhost.exe`
whose parent is one of those PIDs — there should be none.

Stop a dispatcher cleanly (never `taskkill` it; running jobs are tracked
through it):

```bash
gpuq _stop-daemon                      # current profile
GPUQ_PROFILE=smoke gpuq _stop-daemon   # a named profile
```

A dispatcher left over from an old profile is harmless but pointless; stopping
it does not affect other profiles. Note that a job's *own* command may create a
console of its own while it runs — that belongs to your program, and it goes
away when the job ends.

---

## Cancel did not stop everything

```bash
gpuq cancel <id> --force
```

Without `--force`, gpuq attempts a polite stop first and waits
`core.cancel_grace_seconds` (default 15) before killing the tree. On Windows a
console-less child cannot receive a polite signal at all, so the grace period
usually just elapses — `--force` skips it.

If a process survived, gpuq refused to kill it because it could not prove the
PID was still that job's process. `gpuq show <id>` reports the PID it recorded.

---

## Jobs after a reboot

Task queues do not survive a reboot on their own:

```bash
gpuq init && gpuq reconcile && gpuq status
```

- Jobs that were **queued** are recovered and will run.
- Jobs that were **running** when the machine went down are marked `LOST`.
  They are never silently reported as complete.

There is no automatic restart-on-login service in V1. Add one yourself if you
want it (a user systemd unit, or a Task Scheduler entry running `gpuq init`).

---

## `gpuq status` disagrees with reality

```bash
gpuq reconcile --dry-run     # what it would change
gpuq reconcile               # repair
```

Reconciliation compares the metadata database with the dispatcher's queue and
repairs drift after crashes. It never resurrects a cancelled job and never
alters a job that already finished.

---

## Two databases?

Yes, and that is deliberate:

- `gpuq.sqlite3` — gpuq's job metadata.
- `backend/queue.sqlite3` — the dispatcher's own queue state.

Keeping them separate is what makes the execution backend swappable.
`gpuq reconcile` is the sanctioned way to make them agree.

---

## Disk filling up

```bash
gpuq cleanup --dry-run
gpuq cleanup
gpuq cleanup --older-than 3d
```

Snapshots are Git worktrees and cost roughly one working tree each. Retention
defaults to 7 days for successful jobs and 14 for failed ones, so failure
evidence outlives success.

Cleanup never touches an active job's snapshot, anything outside the gpuq
state directory, or the live data behind a passthrough link.

---

## Starting over

```bash
gpuq uninstall --dry-run     # exactly what would be removed
gpuq uninstall --execute     # stops the dispatcher, removes the policy block
gpuq uninstall --execute --purge   # also deletes the state directory
```

Source repositories are never touched.

---

## Isolating an experiment from your real queue

```bash
GPUQ_PROFILE=scratch gpuq init
GPUQ_PROFILE=scratch gpuq submit --project test -- python x.py
```

A profile gets its own state directory, database, config and dispatcher. This
is how the test suite and `scripts/smoke_test.sh` avoid touching production.
