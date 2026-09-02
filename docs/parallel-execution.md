# Parallel execution and live resource limits

Status: implemented. This is kept as the record of why it is built this way,
and of the evidence behind each threshold.

This document covers two changes that belong together:

1. **Parallel execution** — let several jobs run at once when their declared
   footprints genuinely fit, instead of serialising everything behind one slot.
2. **Live resource limits** — let the machine's owner reclaim RAM, VRAM or CPU
   at any moment (to play a game, join a call, or just get the desktop back)
   without stopping the queue or restarting the daemon.

The second is not a nice-to-have bolted onto the first. It is the emergency
brake that makes the first safe to turn on.

---

## 1. The problem, as observed

`max_concurrent_jobs = 1`, so exactly one job runs no matter how small it is or
how much headroom the machine has. A representative snapshot of the live queue:

```
110  RUNNING  normal  kaggriculture  0 GPUs, 14 GiB RAM, 10 CPU   <- holds the only slot
111  QUEUED   normal  arc-agi-3      1 GPU,  14 GiB RAM,  4 CPU   <- GPU is 86% free
112  QUEUED   normal  arc-whest      1 GPU,  24 GiB RAM,  4 CPU
103  QUEUED   low     biohub         waiting 2h20m
```

A CPU-only job holds the gate while the GPU sits 86% idle and two GPU jobs wait
behind it. Against 53.6 GiB RAM / 30.8 GiB VRAM / 14 CPU usable, job 111 fits
alongside 110 exactly (RAM 28/53.6, VRAM 22/30.8, CPU 14/14). It could have been
running.

The same shape produced the original complaint: a long `low`-priority biohub
train arrived first, took the only slot, and held it for hours while
`normal`-priority work from other projects queued behind it. Priority cannot
help — ordering only decides who is next, not how many run.

### Cost, measured

From 108 jobs and 6,055 telemetry samples in the live install:

| | |
|---|---|
| Total queue wait | **1,314 min** |
| Total job runtime | 1,076 min |
| Worst single wait | 217 min |
| Jobs needing 0 GPUs | 18 of 108 |

The queue spends more time waiting than computing. Per project, the jobs that
suffer most are the small ones: kaggriculture waits 3.8x its own compute time,
arc-whest 1.8x. The 18 GPU-free jobs could have run alongside a GPU job almost
any time.

---

## 2. Why this is mostly already built

`resources.admit()` is already a correct multi-job resource broker. It takes the
list of currently running jobs, sums their declared reservations, and checks the
candidate against usable capacity:

```python
reserved = sum_reservations(running)
...
# 2. Would it fit once every running job reaches its declared size?
if reserved.ram_mib + request.ram_mib > cap.usable_ram_mib:
    return Decision(False, ...)
```

There is a passing unit test called `test_several_small_jobs_run_in_parallel`
(`tests/unit/test_resources.py:115`) and another, `test_large_jobs_serialise`,
whose docstring states the design intent precisely:

> RAM is reported as almost entirely free, because the running job has not grown
> to its declared size yet. Measured free memory alone would happily admit the
> second job; the reservation check is what prevents it.

That is the ramp-up race — the classic way a naive parallel scheduler kills a
machine — and it is already handled, because admission checks *declared* size
rather than *current* size. Commit `070453c` said the goal out loud: "small jobs
may run in parallel, large ones serialise, without picking a slot count."

**None of it is reachable in production.** The slot gate runs first, in
`dispatcher.py:511`:

```python
if in_flight >= slots:          # slots == 1
    self._consider_preemption(row, "waiting for a free slot")
    break
```

`max_concurrent_jobs = 1` is the last surviving piece of the original
exclusivity model. `docs/future-slurm.md:45` already calls Stage 1 "mostly done,
deliberately not enabled."

---

## 3. Three findings that change the shape of the work

### 3.1 RAM is over-declared 2-5x; VRAM is honest

Declared versus observed peak, from telemetry:

| job | project | declared RAM | observed | declared VRAM | observed |
|---|---|---|---|---|---|
| 97 | biohub | 24.0 | 4.3 | 4.0 | 5.1 |
| 96 | biohub | 16.0 | 9.9 | 20.0 | 19.3 |
| 94 | biohub | 16.0 | 12.1 | 20.0 | 17.2 |
| 90 | arc-whest | 12.0 | 3.3 | 10.0 | 3.7 |
| 86 | arc-whest | 20.0 | 20.1 | 0.0 | 2.6 |
| 84 | kaggriculture | 14.0 | 4.2 | 0.0 | 2.7 |

Over-declaring is safe — the ledger stays conservative — but it is why packing
is poor. VRAM being declared accurately matters a great deal, because VRAM has
no swap: a VRAM ledger can be trusted in a way a RAM ledger cannot.

The corresponding risk runs the other way. An **undeclared** job is charged
`default_ram_gb = 4.0` and 1 CPU. One such job under a single slot is harmless.
Several in parallel, each actually using 20 GiB, is how the machine falls over.
Undeclared work is the main new hazard this change introduces.

### 3.2 The commit-charge hard stop is miscalibrated

`max_commit_percent = 88`, and **2,429 of 4,843 running-job samples exceed it**.
It is a global stop that blocks every job, including one asking for nothing
(`resources.py:182-189`).

But commit charge is not a measure of the thing that freezes a machine:

| commit % | samples | avg physical RAM free | min free |
|---|---|---|---|
| < 70 | 1442 | 54.6% | 37.7% |
| 70-88 | 2186 | 49.0% | 24.1% |
| 88-95 | 557 | 48.2% | 34.7% |
| >= 95 | 1874 | **41.9%** | 22.4% |

At 95%+ commit the machine still averages 41.9% physical RAM free. The worst
pressure ever recorded was 13.8 GiB still available — the box has never actually
run out. Windows commit charge measures committed address space against a
pagefile limit that grows on demand; the limit drifted from 81.3 to 93.6 GiB
during the sampling window. When idle, commit never once exceeded 88%.

Left as-is, this gate would veto backfill roughly half the time a job is
running. It has to be recalibrated before parallelism does anything useful.

### 3.3 Nothing measures what a job actually uses, and nothing watches pressure

There is no per-job RSS or VRAM sampling. `jobs` has only `requested_*` columns,
never `actual_*`. The runner launches the child and waits; it never looks at it.
Telemetry is machine-wide and records a single scalar `running_job_id`, taken as
`running_ids[0]` (`dispatcher.py:408`) — with N jobs running it attributes all
machine pressure to an arbitrary one.

There is also no runtime guard. Once admitted, a job can exceed its declaration
without limit and nothing notices, throttles, or stops it. Under one slot the
blast radius is one job. Under N it is the desktop.

---

## 4. Design

The slot count stops being the limiter and becomes a **safety ceiling** — a
backstop against a bug launching twenty jobs, not the mechanism that decides
what runs. The reservation ledger becomes the real gate.

Three things must be added that do not exist today: **backfill**, a **runtime
pressure guard**, and **per-job usage measurement**. And one thing must be made
adjustable at runtime: **the reserve**.

### 4.1 The reserve, and why it becomes live

Today `capacity()` computes what worker-q may hand out as total minus a fixed
reserve read from config:

```python
usable_ram = max(0.0, total_ram - r.reserve_ram_gb * _GIB_MIB)
usable_cpus = max(1, total_cpus - r.reserve_cpus)
usable_vram = max(0.0, total_vram - r.reserve_vram_gb * _GIB_MIB)
```

Defaults are 8 GiB RAM, 1 GiB VRAM, 2 CPU — enough for an editor and the agents,
nowhere near enough for a game.

Config alone cannot express "give me the machine back for two hours", for two
reasons. The daemon captures `self.config` once at startup, so a config edit does
nothing until it is restarted (this is why the integration tests bounce the
daemon after touching resource config). And a config edit is permanent until
edited back, which is exactly the wrong shape for a temporary claim.

**The reserve moves into the queue-store meta table and is re-read every tick.**
This is not a new mechanism: `META_SLOTS` and `META_GPU_FREE_PERC` already work
this way, which is why `workerq concurrency N` takes effect immediately while
every other setting needs a restart.

New meta keys: `reserve_ram_mib`, `reserve_vram_mib`, `reserve_cpus`,
`reserve_label`, `reserve_expires_at`. Unset falls back to config, so behaviour
is unchanged until someone sets one.

```python
@dataclass(frozen=True)
class Reserve:
    """Headroom worker-q promises never to hand out."""
    ram_mib: float
    vram_mib: float
    cpus: int
    label: str | None = None
    expires_at: str | None = None

    @classmethod
    def from_config(cls, config: Config) -> Reserve: ...
```

`capacity()` and `admit()` gain an optional `reserve: Reserve | None`, defaulting
to `Reserve.from_config(config)`. Both stay pure functions of their arguments —
they are heavily unit-tested and must remain so.

### 4.2 The reserve CLI

```bash
workerq reserve                                # show reserve + effective capacity
workerq reserve --ram 24 --vram 22 --cpus 8    # claim it now
workerq reserve --ram 24 --for 2h              # claim it, auto-release later
workerq reserve --clear                        # back to config defaults
```

Named presets are **deferred**, not shipped. `Config.to_toml()` regenerates the
whole file from flat dataclass sections, so a nested `[reserve_presets.gaming]`
table would be silently deleted the next time anything called `config set`.
Adding them means teaching the config writer about nested tables first; shipping
them before that would quietly eat the user's config. `--label` covers the
naming need in the meantime.

**Running jobs are not touched by default.** Tightening the reserve applies to
admission from the next tick; work already in flight finishes. This is the
honest behaviour and it must be reported honestly, because tightening can leave
the ledger temporarily over-committed:

```
Reserve set to 24 GiB RAM / 22 GiB VRAM / 8 CPU  (gaming, releases in 2h)
2 running jobs hold 38 GiB RAM and 20 GiB VRAM; they will finish first.
  #110 kaggriculture  14 GiB  not preemptible
  #114 biohub         24 GiB  preemptible  -- re-run with --evict to stop it now
```

`--evict` is **not shipped yet**. When added it will additionally preempt
running jobs, newest first, until the ledger fits the new budget, obeying the
existing preemption guards: only jobs submitted `--preemptible` are ever
stopped, the same contract as priority-driven preemption. Today the command
reports which running jobs are holding the resources and which of them are
preemptible, so the choice can be made by hand with `workerq cancel`.

Three safety rules:

- A reserve larger than the machine is rejected, not clamped — otherwise the
  queue silently stops forever.
- Setting a reserve that makes a queued job impossible says so at once, reusing
  the `_reject_impossible` logic that already runs at submit time: *"job 111
  needs 22 GiB VRAM; with this reserve only 9.8 GiB is available — it will never
  start."*
- `--for` writes `reserve_expires_at` and the dispatcher clears it on expiry, so
  a temporary claim cannot become a permanent mystery.

**Visibility is mandatory.** A non-default reserve must appear in the `status`
header, in `top`, and in the `wait_reason` of anything it blocks — otherwise the
first symptom is "why is nothing starting?" with no way to find out. The wait
reason should name it: `held back by reserve 'gaming' (22 GiB VRAM)`.

One interaction to document: `gpu.free_memory_threshold_percent` (currently 70)
gates whole-device allocation independently of the reserve. A game using VRAM
pushes free% below the threshold and blocks GPU jobs anyway. The two controls
need to be understood together, and `workerq reserve` should report both.

### 4.3 Backfill

Today the dispatch loop `break`s on the first job it cannot place — three times,
at `dispatcher.py:516`, `:528` and `:539`. Head-of-line blocking is deliberate
("a queued critical job must not be overtaken just because the GPU is busy") but
under parallelism it defeats the entire point: one oversized job at the head
parks a queue full of jobs that would fit.

Replace the `break`s with a bounded `continue`. Bounded, because unrestricted
backfill starves large jobs — a steady stream of small work would defer a big
job forever. The head job gets a reservation and a pass-over budget; once it has
been skipped N times or waited longer than a deadline, backfill stops and the
queue drains until it can run.

This matters more than it sounds. There is **no anti-starvation logic anywhere**
today: the sort key has no aging term, and `preemption.max_preemptions` is
defined, validated and documented as the starvation guard but is never read by
any code path.

### 4.4 The runtime pressure guard

Admission is a prediction. The guard is what happens when the prediction is
wrong — an under-declared job, a foreign workload, a game launched without
claiming a reserve.

The dispatcher already samples the machine every 10 seconds. Add: sustained
physical-RAM pressure across consecutive samples stops new admissions
immediately; if it worsens, displace the newest preemptible job. Hysteresis on
both edges so it does not oscillate.

Note this is the one thing preemption currently *cannot* do. `_consider_preemption`
re-runs `admit()` with victims removed, but freeing a *reservation* does not
change *measured* free memory until the victim actually exits — so preemption
for live memory pressure structurally almost never fires today.

### 4.5 Keeping the desktop responsive

Distinct from memory, and directly relevant to "do not slow my machine": with 14
usable CPUs shared between parallel jobs, CPU saturation is what makes a desktop
feel frozen even when RAM is fine.

Launch job children at `BELOW_NORMAL_PRIORITY_CLASS`. Windows will then hand the
interactive desktop the CPU it needs regardless of how many jobs are running.
`winproc.py` already owns every creation flag, so this is a small, well-isolated
change with a large perceived benefit.

### 4.6 Shared GPU mode

On a single-GPU machine, `_devices_in_use()` gives whole-device exclusivity: a
device held by one job is excluded from every other. **Two GPU jobs can
therefore never co-run today, regardless of the slot count.** Only CPU-only jobs
can backfill.

Getting jobs 111 and 112 to overlap needs per-device VRAM ledger admission
instead of whole-device exclusion. The hook already exists: `gpu_mode`
(`exclusive` / `shared`) is in the schema, is set at submit from
`gpu.exclusive_by_default`, is displayed by `workerq show` — and is never read
by anything.

This is the highest-risk phase and stays opt-in. VRAM OOM is unforgiving and
there is no swap to absorb a mistake. The mitigating fact from 3.1 is that VRAM
is the one resource this machine's jobs declare accurately.

Also fix while here: the VRAM ledger checks against the **summed** VRAM of all
devices while placement is per-device (`resources.py:121` vs `:243`). Harmless on
one GPU, wrong on two.

---

## 5. Phases

Ordered so that each phase is independently useful and the risky ones come last.

**Phase 0 — Measurement (no behaviour change). DONE.**
Per-job process-tree RSS and VRAM sampling in the runner; `jobs` schema v6 with
`peak_ram_mib` / `peak_vram_mib`; a new `job_samples` table for per-job
attribution. Ship `workerq resources --verify` to show declared versus actual.
Nothing else in this plan can be validated without it.

*Note:* `telemetry.sqlite3` has no migration framework — it is bare
`CREATE TABLE IF NOT EXISTS`. So per-job attribution goes in a **new table**, not
a new column on `samples`. The jobs DB does have proper versioned migrations
(`schema_version` + incremental `_MIGRATIONS`, currently v5), so v6 there is
routine.

**Phase 1 — Live reserve. DONE.**
`Reserve`, meta keys, `capacity()`/`admit()` overrides, `workerq reserve` with
presets, expiry, `--evict`, and status/top visibility. Deliberately before
parallelism: it is the brake, and it should exist before the accelerator.

**Phase 2 — Multi-slot execution with ledger admission. DONE.**
Raise the ceiling; bounded backfill with a head-of-queue reservation and
starvation guard; store a wait reason for *every* queued job. Today only the
head job gets one, and slot exhaustion stores none at all — with N slots that is
the first confusing thing anyone will hit.

**Phase 3 — Recalibrate the hard stops. DONE.**
Physical availability as the primary gate; commit charge as a secondary,
growth-aware guard. This is what makes Phase 2 useful in practice rather than
vetoed half the time.

**Phase 4 — Pressure guard and process priority. DONE.**
The runtime backstop (4.4) and `BELOW_NORMAL_PRIORITY_CLASS` (4.5).

**Phase 5 — Shared GPU mode. DONE (opt-in via `--share-gpu`).**
Opt-in per-device VRAM ledger (4.6). Only after Phase 0 has produced real
declared-versus-actual VRAM data.

**Phase 6 — Fallout and parity. DONE.**
Detailed in §6.

---

### What Phases 0 and 1 actually shipped

| Area | Change |
|---|---|
| `db.py` | schema v6: `peak_ram_mib`, `peak_vram_mib`, `usage_samples`. NULL means never measured |
| `host.py` | `all_processes()`, `tree_memory_mib(roots)` - process-tree RAM attribution |
| `gpu.py` | `tree_vram_mib(info, pids)`, returning None when unmeasurable (see 3.1) |
| `runner.py` | `_watch_usage` thread sampling the job's tree every 15s, recording peaks |
| `report.py` | `declared_vs_observed()` |
| `cli.py` | `workerq resources --verify` |
| `resources.py` | `Reserve` type; `capacity()`/`admit()`/`describe_capacity()` take one |
| `dispatcher.py` | reserve meta keys, `read_reserve()`/`clear_reserve()`, `_reserve()` re-read per tick with expiry |
| `local_dispatcher.py` | `set_reserve()` / `get_reserve()` |
| `core.py` | `get_reserve()`, `set_reserve()`, `clear_reserve()` with validation and impact reporting |
| `cli.py` | `workerq reserve` |

Nothing in Phase 1 changes how many jobs run. It changes how much of the machine
they may collectively use, and lets that be changed while they run.

One bug worth recording, because it would have made the feature silently
useless: `is_expired` was first written on `age_seconds()`, which clamps at
zero. A deadline an hour in the future therefore read as already reached, so
every `--for` reserve would have been released on the next tick. It compares
instants now.

---

## 6. Known fallout

Things that will break or mislead the moment concurrency exceeds 1.

**Correctness prerequisite.** MCP's `gpu_submit` exposes none of `--ram`,
`--vram`, `--cpus`, `--preemptible` (`mcp/server.py:231-283`). Every
MCP-submitted job is therefore charged the 4 GiB / 1 CPU default. Per 3.1 that
is the main OOM vector under parallelism, so this is a prerequisite, not
cleanup. It also hardcodes `priority="normal"`, bypassing project standing
priority entirely.

**Health and scripts.** `doctor` returns WARN whenever `slots != 1`
(`doctor.py:196-203`), which would make `bootstrap.sh` exit non-zero.
`scripts/smoke_test.sh` and `scripts/acceptance_dod.sh` both assert exclusivity
directly, and several of their steps use a long "blocker" job to hold the only
slot while something else is tested — those become flaky rather than failing
cleanly.

**Tests.** Two integration tests assert exclusivity by name and would need to
invert: `test_queue_exclusivity_no_overlap` and `test_only_one_job_is_ever_running`
(`tests/integration/test_queue.py:195`, `:223`). Note `conftest.py:57` pins
`max_concurrent_jobs=1` and `conftest.py:69` sets `resources.enforce=False`, so
new concurrency tests must opt into both. `backend.set_slots(n)` takes effect on
the next tick with no restart; `config.resources` changes do not.

**UI that assumes one running job.**

| Location | Issue |
|---|---|
| `cli.py:442-444` | status footer advertises `running[0]` as *the* job to follow |
| `dashboard.py:129-131` | machine panel renders only `gpu.devices[0]` |
| `dashboard.py:210-212` | wait-reason row passes 7 values into an 8-column table |
| `dashboard.py:183`, `:272` | hardcoded `[:12]` / `[:8]` caps with no "+N more" |
| `dashboard.py:239-255` | pressure panel cannot map a PID back to a job |
| `dashboard.py:150` | commit red-line hardcoded to 88, ignores config |
| `eta.py:256` | forecast uses *config* slots while status shows *backend* slots |
| `eta.py:285-318` | ~~forecast ignores admission~~ — fixed: a job with a recorded wait reason reports an unknown start rather than "starts ~0s" |
| `cli.py:411-419` | no RAM/VRAM/CPU or assigned-device column anywhere in `status` |

**Docs.** README lines 10-11, 31-33, 60-61, 71, 391-393, 494-495, 513, 523,
593-603; `architecture.md:5-7`, `:224`, `:261-271`; `troubleshooting.md:46-50`
(already factually wrong since preemption shipped), `:315`; `mcp/server.py:224-228`
(agent-facing, claims one job at a time); `claude_policy.py` (already describes
the parallel behaviour this plan delivers).

---

## 7. What this does not fix

Parallelism does not resolve priority inversion. When the machine is genuinely
full, a `low`-priority job still holds resources against `normal` work.
Preemption exists but requires `--preemptible`, and the biohub trains do not opt
in — though they checkpoint (`resume from epoch 51`), so they are good
candidates. A project-level `preemptible` default in `.gpuq.toml` would address
it; `resolve_preemptible` (`core.py:431-443`) already reads that file.

Two related gaps stay open unless Phase 2 closes them: the dispatch sort key has
no aging term, and `preemption.max_preemptions` is still dead code.

---

## 8. Operational note

`META_SLOTS` in the queue DB shadows `core.max_concurrent_jobs`, and the daemon
re-seeds from the DB on restart (`dispatcher.py:664-667`). Editing config.toml
and restarting therefore does nothing. Use `workerq concurrency N --yes`. The
same will be true of the reserve, by design — `workerq reserve` is the interface,
and config only supplies the fallback and the presets.

---

## 9. What shipped, in the end

`max_concurrent_jobs` defaults to **4** and is a ceiling, not the scheduler.
Admission control decides what runs.

| Phase | Change |
|---|---|
| 0 | per-job usage sampling (schema v6), `job_samples` telemetry, `workerq resources --verify` |
| 1 | `Reserve` in dispatcher meta, re-read per tick; `workerq reserve` with `--for` expiry |
| 2 | bounded backfill with a starvation guard; a wait reason for every queued job |
| 3 | commit charge blocks near the limit, or lower only when physical RAM is short too |
| 4 | pressure guard displaces the newest preemptible job; jobs run below normal priority |
| 5 | `--share-gpu`: per-device VRAM packing between jobs that both opted in |
| 6 | MCP resource parity, doctor, dashboard, forecast, scripts, docs |

### Deliberately not built

**`workerq reserve --evict`.** Tightening the reserve leaves running jobs alone;
the command reports what is holding resources and which of those are
preemptible, and the choice is made by hand with `workerq cancel`. Automatic
eviction is a small addition to `set_reserve` when it is wanted.

**Named reserve presets.** `Config.to_toml()` regenerates the file from flat
dataclass sections, so a nested `[reserve_presets.gaming]` table would be
deleted by the next `config set`. That needs the config writer taught about
nested tables first; `--label` covers the naming need meanwhile.

**Learned resource declarations.** Phase 0 records what jobs use, and `eta.py`
already learns durations per command signature. Learning a RAM default the same
way is the obvious next step, but wants more than one machine's data before it
starts overriding what people declare.

### Two bugs this work surfaced

`Reserve.is_expired` was written on `age_seconds()`, which clamps at zero, so a
deadline an hour away read as already reached and every timed reserve would have
been released on the next tick.

The runner told preemption from cancellation by whether `preempt_by` was set. A
pressure-guard stop has nobody to name, so it would have been read as a
cancellation and the job marked CANCELLED instead of requeued, losing the work.
The intent is carried explicitly now.
