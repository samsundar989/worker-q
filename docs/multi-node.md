# Multi-node: dispatching to a second machine

Status: **implemented and running.** Jobs submitted on the primary are
placed on either machine automatically, run there, and their results and logs
come home. What remains is listed in [§10](#10-phases).

This supersedes Stage 3 of
[future-slurm.md](future-slurm.md), which sketched the idea and then argued
against it. The argument still holds at scale; what changed is that the target
is exactly two machines, which that document itself names as the honest
ceiling for hand-rolling.

Target topology:

| | `SAM_MEGA_PC` (primary) | `DESKTOP-UNR95NB` (worker) |
| --- | --- | --- |
| GPU | RTX 5090, 32 GiB (31.8 usable) | **RTX 3080 Ti, 12 GiB** |
| CPU | Ryzen 7 9800X3D, 8c/16t (Zen 5 + X3D) | Ryzen 7 5800X, 8c/16t (Zen 3) |
| RAM | 64 GiB | **16 GiB** |
| Driver | 596.49 | 591.86 |
| Arch | Blackwell, `sm_120` | Ampere, `sm_86` |
| Role | interactive: agents, editors, browsers | dedicated worker, nothing else |
| Link | Wi-Fi, 192.168.1.181 | Wi-Fi |

The worker being *dedicated* is load-bearing throughout. It removes the need
for keyboard-idle suspension, desktop responsiveness heuristics and a standing
gaming reserve on that side, and it makes auto-logon acceptable — which, as
[§2.1](#21-the-session-0-constraint-and-why-it-is-smaller-than-it-looks) explains, is the difference between CUDA
working and not.

---

## 0. The asymmetry is the design

The worker has **a quarter of the primary's RAM, a little over a third of its
VRAM, and a CPU roughly 1.3–1.5x slower per core.** It is not a second
workstation. It is a helper, and every decision below is shaped by that.

Two consequences, both of which invert what a symmetric design would do.

**The point is not to add capacity. It is to free the 5090.** A 12 GiB card
cannot take the jobs that make the primary a bottleneck — those are exactly the
jobs wanting 20–30 GiB of VRAM. What it *can* take is the stream of small and
medium work (evaluations, feature extraction, sweeps over small models,
CPU-heavy preprocessing) that currently sits in front of them. Moving that
stream off the primary is worth more than the worker's raw throughput suggests,
because it removes queue wait from the big jobs rather than trying to add
capacity for them.

**Faster silicon means "local" is not automatically the cheap choice, and
"remote" is not automatically the fast one.** Sending a job to the worker
trades a slower run for an earlier start. That trade is only worth making when
the primary is contended. [§5.4](#54-scoring-spill-under-contention-stay-home-when-idle)
makes this an explicit rule rather than a preference.

### 0.1 Is this worth building? Measured, not guessed

Yes, and by a wider margin than the hardware gap suggests. Against the 395
finished jobs in the metadata database at the time of writing:

| | share of all finished jobs |
| --- | --- |
| Would fit the worker (≤11 GiB VRAM, ≤13 GiB RAM as declared) | **65%** |
| …of which need **no GPU at all** (0 GiB VRAM declared) | **49%** |
| Cannot fit as declared — VRAM ≥12 GiB (55 jobs) or RAM >13 GiB (84 jobs) | 35% |

The headline is the second row. **Half the queue is CPU- and RAM-bound work
that never touches the GPU** — `kaggriculture` alone is 132 jobs, almost all of
them declaring 0 VRAM. For that half the 5090 contributes nothing, and a
5800X with 16 GiB is a perfectly good machine to run it on. Those jobs are not
merely *eligible* for the worker; they are the reason to build this.

The VRAM distribution is strongly bimodal, which is what makes placement easy:

```text
    0 GiB  ████████████████████████████████████████████  246
  1–10 GiB ████████████████                               94
 12–28 GiB ███████████                                    55   ← 5090 only
```

There is very little in the 11–12 GiB band where the decision would be close.
A job either obviously fits the 3080 Ti or obviously does not.

RAM is not a constraint at all in practice: median measured peak **2.0 GiB**,
p90 **5.9 GiB**, and only 7 of 261 measured jobs ever exceeded the worker's
13 GiB of usable RAM.

### 0.2 Over-declaration will cost you the worker

Eligibility is judged on what a job *declares*, because that is all that is
known before it runs. But this queue over-declares RAM by 3–5x — the same
finding that motivated [parallel-execution.md](parallel-execution.md) §3.1, and
still true:

```text
kaggriculture     declared 14 GiB   n=20    mean measured peak  3.9 GiB
                  declared 10 GiB   n=16    mean measured peak  2.0 GiB
                  declared  8 GiB   n= 9    mean measured peak  1.6 GiB
```

The consequence for this design is direct and quantified:

| Eligibility judged on | jobs that fit the worker |
| --- | --- |
| RAM as **declared** | 256 / 395 (65%) |
| RAM as **measured** | 306 / 395 (77%) |

**50 jobs are excluded from the worker purely because their RAM declaration is
wrong** — 26 in `biohub`, 20 in `kaggriculture`, 7 in `arc-whest`. Nothing
about them needs the 5090; they are exiled by a number nobody corrected.

> **Caveat, added after commit `16b2846`.** The measured peaks above predate
> the switch from working-set to commit measurement. `tasklist` reported
> resident memory, and a CUDA job holding 18.8 GiB of commit appeared there as
> 1.0 GiB — so historical `peak_ram_mib` understates GPU jobs badly and
> CPU-only jobs mildly. Only 2 of 271 measurements postdate the fix.
>
> What survives the caveat: the 49% of jobs declaring **0 VRAM** are CPU-only,
> where working set and commit are close, so the over-declaration finding holds
> for them — and they are the worker's main workload anyway. What must be
> re-checked once new data accumulates: the 26 `biohub` GPU jobs in the "50
> excluded by a wrong number" figure, whose true commit footprint was never
> measured. Re-run the analysis after a week of jobs under the new
> measurement before relying on the 77% figure.

So the SUGGEST machinery in `eta.py` becomes *more* valuable in a two-node
world, not less: today a bad declaration costs some utilisation on one machine,
and after this work it costs access to an entire second machine. Two
implications for the phasing:

- Correcting declarations on the busiest projects is worth doing **before**
  Phase 5, and needs no new code — `workerq requests <id> --suggest` already
  computes the right number from history.
- Placement should surface the gap where it bites. When a job is ruled out of a
  node on a declaration that its own history says is 3x too high, the wait
  reason should say so:

  > `will not fit on desktop-unr95nb (declares 14.0 GiB RAM); past runs of this
  > command peaked at 3.9 GiB — see workerq requests 391 --suggest`

  That turns an invisible loss of capacity into a one-line fix.

---

## 1. Restating the invariant

Today:

> Every agent can submit expensive work in seconds, continue coding, and trust
> that the machine will not launch another broker-managed heavy workload until
> it is safe.

The naive extension — one scheduler that reaches across the network and starts
processes on both machines — breaks this, because the thing that makes the
guarantee true is that admission control runs *on the machine it is protecting*,
against numbers it measured a fraction of a second ago. `resources.admit()`
reads `GlobalMemoryStatusEx`, the live commit charge and `nvidia-smi` at the
moment of decision. A number that crossed a Wi-Fi link is a number about the
past.

So the invariant becomes two sentences, and the split between them is the
central design decision:

> **Placement is central. Admission is local.**
>
> One queue decides *which machine* a job should go to. That machine decides
> *whether it is safe to start it now*, using the same code, against its own
> live measurements.

Everything below follows from that. In particular it is why the worker runs a
complete worker-q rather than a thin execution agent: admission control,
the pressure guard, preemption, Job Object process-tree kills, PID-identity
checks, progress watching and log capture all already exist and are tested.
Reimplementing any of them over SSH would be strictly worse.

---

## 2. Prerequisites and transport

### 2.1 The Session 0 constraint, and why it is smaller than it looks

Under WDDM — the only mode a GeForce card supports; TCC is datacenter-only — a
process in Windows **Session 0** has no access to display devices and therefore
no access to CUDA. Anything installed as a Windows service runs in Session 0.

Consequence: **the worker's dispatcher must not be a Windows service.** It runs
in an interactive session, started by a Task Scheduler task at logon. Locking
the screen is fine; the session survives. This is not a worker-q limitation — it
is why HTCondor's own answer to GPU jobs on Windows was "run personal Condor
with a user logged in".

The important structural point is that **this design already avoids the
problem, rather than depending on it going away.** SSH carries *control-plane
traffic only* — `submit`, `status`, `cancel`, `logs`, the node report. It never
launches a job. Jobs are launched by the worker's own dispatcher, in the
console session, exactly as they are on the primary today. So the case that
must work is "CUDA in a console session", which is the ordinary case that works
on every gaming PC.

What the Phase 0 experiment actually decides is therefore not *whether this is
possible* but *how much setup it needs*:

| CUDA works in… | implication |
| --- | --- |
| console session (certain) | the design works; dispatcher starts at logon |
| SSH session **with a console user logged in** | jobs could be launched from SSH too — but see below, they should not be |
| SSH session **with nobody logged in** | auto-logon is unnecessary; the machine can sit at the login screen |

Only the first row is load-bearing. The other two remove manual steps.

**Corrected after trying it.** An earlier draft of this section said the second
row meant the dispatcher could be "started and restarted remotely". It cannot,
not with `workerq restart`: a dispatcher started inside an SSH session **dies
when that session ends**, even though it is spawned detached, because Windows
OpenSSH tears down its session's processes on disconnect. The symptom is
particularly unhelpful — `_dispatcher-status` reports a pid and a heartbeat
that only *looks* fresh, for the thirty seconds before it goes stale.

The remote restart that does work goes through the logon task, whose principal
is `Interactive`, so the dispatcher lands in the console session where CUDA
also works:

```powershell
ssh <node> schtasks /Run /TN "worker-q dispatcher"
```

`workerq restart` now warns when `SSH_CONNECTION` is set rather than appearing
to succeed, and `workerq node check` flags a node whose dispatcher is down —
reachable and compatible is not the same as able to run anything.

Related, and not optional: **never dispatch over an RDP session.** RDP swaps in
a dummy display driver and the NVIDIA driver is not activated. SSH sessions
keep the real driver, which is a second reason SSH is the transport.

See [§10, Phase 0](#phase-0--prove-the-ground-no-code) and the runbook in
[multi-node-setup.md](multi-node-setup.md).

### 2.2 Transport: Tailscale for the network, OpenSSH for the shell

Both machines already run Tailscale, so it is the network layer. It is *not*
the shell layer: **Tailscale SSH's server component is Linux/macOS only** and
cannot target Windows, so OpenSSH Server runs on the worker and Tailscale
simply carries it.

What Tailscale buys, concretely:

- A **stable name**. The worker's LAN address is DHCP-assigned and will change;
  a node registry pinned to `192.168.1.x` breaks silently on a router reboot.
  MagicDNS gives a name that does not move.
- **It keeps working off-LAN**, so a laptop or a relocated machine does not
  need the registry re-pointed.
- Encryption and identity without worker-q owning any of it.

The one thing to verify is that peers connect **directly** rather than via a
DERP relay — `tailscale ping <peer>` says which. On the same LAN it should be
direct; a relayed path would put every snapshot bundle through Tailscale's
servers.

OpenSSH Server is a built-in optional Windows feature; installing it creates
the port-22 firewall rule. Key authentication only. Two gotchas that each cost
an afternoon if unknown:

- A user in the **Administrators** group does not read `~/.ssh/authorized_keys`.
  It reads `C:\ProgramData\ssh\administrators_authorized_keys`, which must be
  ACL'd to SYSTEM + Administrators only and must be **UTF-8 without BOM**.
  PowerShell's `>` redirection and `Out-File -Encoding utf8` both write a BOM;
  that is the classic silent failure. Use `-Encoding ascii`.
- Leave `DefaultShell` alone. The default `cmd.exe` spawns faster than
  PowerShell and avoids a second layer of quoting; worker-q should invoke
  `powershell -NoProfile -Command` explicitly on the rare occasions it needs
  to.

**Windows OpenSSH has no connection multiplexing** (`ControlMaster` is
unsupported). Every `ssh` invocation pays a fresh TCP connect and handshake.
This directly shapes the polling design in [§5.3](#53-one-call-per-tick), and
the per-call cost should be measured in Phase 0 rather than assumed.

### 2.3 No network listener

worker-q has no socket, port or protocol today, and this design keeps it that
way. The queue database is the local channel; SSH is the remote one. There is
still nothing listening that worker-q owns. Retaining this is worth some
inefficiency: it removes port conflicts, orphaned listeners and an entire
authentication surface from a tool that runs on a personal machine.

### 2.4 Commit is the binding constraint on a 16 GiB box

Under WDDM the driver backs video allocations with **system commit**, so a
job's commit cost is roughly its RAM *plus* its VRAM. `resources.admit()`
already knows this — it is the check that exists because a job died here at
100% commit with 40% of RAM free.

On the primary that check almost never binds: 64 GiB of RAM plus a pagefile
leaves a large ceiling. On the worker it binds constantly. A job using 10 GiB
of VRAM and 4 GiB of RAM wants ~14 GiB of commit on a machine with 16 GiB of
physical RAM.

`host.commit_ceiling_mib()` is RAM plus the pagefile's **configured maximum**,
and it deliberately falls back to the *current* limit when the pagefile is
system-managed, because assuming unbounded growth is the unsafe direction. That
conservatism is invisible at 64 GiB and crippling at 16 GiB: a system-managed
pagefile on the worker will refuse work the machine could comfortably run.

**Action, and the single highest-leverage setting on that machine:** give the
worker a **fixed pagefile with an explicit maximum** — 48–64 GiB, on the
fastest drive with room. That makes the commit ceiling known, stable and
generous, and it is what lets a 12 GiB card be used to its edge on a 16 GiB
host. Verify afterwards with `workerq resources --json` on the worker: the
`commit.ceiling_mib` figure should reflect RAM + pagefile maximum, not RAM +
whatever Windows has grown to so far.

Disk budget on the worker is therefore: pagefile, plus snapshots, plus
whatever datasets are staged locally. This needs checking in Phase 0.

### 2.5 The worker needs its own configuration

worker-q's defaults are tuned for a 64 GiB interactive workstation and are
actively wrong on a dedicated 16 GiB box. `reserve_ram_gb = 8.0` gives away
half the machine; `reserve_vram_gb = 4.0` gives away a third of a 12 GiB card.

A reasonable starting point for `~/.config/gpuq/config.toml` on the worker:

```toml
[core]
max_concurrent_jobs = 3          # ceiling only; admission still decides

[gpu]
free_memory_threshold_percent = 80
exclusive_by_default = true

[resources]
enforce = true
default_ram_gb = 3.0             # undeclared work is still not free
default_vram_gb = 0.0
default_cpus = 1
reserve_ram_gb = 3.0             # dedicated box: OS + worker-q only
reserve_vram_gb = 1.0            # a 12 GiB card cannot spare 4
reserve_cpus = 1
min_host_free_percent = 10
```

There is no `[gaming]` reserve on the worker, and `[claude]` policy
installation is off — no agents run there.

`workerq node check` should print both machines' configurations side by side,
because a misconfigured worker is indistinguishable from a scheduling bug.

---


---

## 3. Shape

```text
SAM_MEGA_PC (primary)                      DESKTOP-UNR95NB (worker)
──────────────────────                     ───────────────────────
Claude A ─┐
Claude B ─┼─ workerq CLI
Claude C ─┘        │
                   ▼
           ┌───────────────┐
           │  GPUQService  │  one job table, one id space
           └───────┬───────┘
                   ▼
           ┌───────────────┐
           │ ClusterBackend│  placement: which node, and when
           └───┬───────┬───┘
               │       │
       ┌───────┘       └────────┐
       ▼                        ▼
┌──────────────┐        ┌─────────────────┐        ssh
│LocalDispatch │        │ RemoteWorkerq   │ ─────────────────────┐
│   Backend    │        │    Backend      │                      │
└──────┬───────┘        └─────────────────┘                      ▼
       ▼                                                ┌────────────────┐
  dispatcher daemon                                     │ workerq (full) │
       ▼                                                │  dispatcher    │
  workerq _run <id>                                     └───────┬────────┘
       ▼                                                        ▼
    RTX 5090                                              workerq _run <id>
                                                                ▼
                                                          RTX 3080 Ti
```

`ClusterBackend` itself implements `SchedulerBackend`, so `GPUQService`, the
CLI and the MCP adapter are unchanged — they still hold exactly one backend.
This honours the promise in `backends/base.py` that a new backend is "a new
module plus a branch in `build_backend`", with one addition: `ClusterBackend`
is the first backend that *composes* others.

### 3.1 Identity

`jobs.id` on the primary stays the only id a user or agent ever types. It is
already globally meaningful; nothing changes.

`jobs.node` — the column reserved in migration 1 for exactly this — records
where a job was placed. `ClusterBackend` keeps a routing table mapping its own
`backend_id` to `(node, remote_backend_id)`. The natural key is the job label,
which already carries the primary's job id (`gpuq:<id>:<project>:<priority>`)
and which the protocol already exposes through `find_by_label` — that is what
makes reconciliation after a link drop possible.

Remote ids are provenance, surfaced only under `workerq show <id> --json`.

---

## 4. The queue lives on the primary

**Nodes never hold a backlog.** A job is pushed to a node only in the tick in
which that node is expected to start it. Placement is therefore re-evaluated
every tick, not decided at submit time.

This is worth stating plainly because the alternative is tempting and wrong.
If jobs were routed at submission, a job assigned to the worker at 09:00 would
still be sitting there at 11:00 while the primary went idle, and `bump`,
`cancel` and re-placement would each need a distributed answer. Holding the
queue centrally means:

- `workerq status` is one list, in one order, with one set of wait reasons;
- `bump` and `cancel` of a queued job are local operations, as today;
- a job is never stranded behind a node that got busy after it was assigned;
- the wait reason can say *why no node will take it*, naming both machines.

The cost is that a node can sit idle for up to one primary tick after
finishing. At a 0.25 s local tick and a 2–5 s remote poll ([§5.3](#53-one-call-per-tick)),
that is irrelevant for jobs measured in minutes.

---

## 5. Placement

### 5.1 Eligibility, then fit

For each queued job, in the existing `(priority_rank, position, id)` order,
compute the set of eligible nodes:

1. **Pinned.** `--node <name>` restricts to one node. `--node local` is an
   explicit way to keep a job home.
2. **Reachable.** The node's last report is fresh and its dispatcher is
   healthy. A stale node is not eligible for *new* work (see
   [§7](#7-failure-semantics) for what happens to work already on it).
3. **Data present.** Every `--passthrough` path the job declares must exist on
   the candidate node. This is the answer to data locality: the declarations
   already exist, so no new bookkeeping is introduced, and a job whose dataset
   is only on the primary simply is not eligible for the worker — with a wait
   reason that says which path was missing.
4. **Repo present.** The node has a clone of the job's repo, or can make one
   ([§6](#6-getting-the-source-there)).
5. **Physically possible.** This is now the *dominant* filter, not an edge
   case. The worker has 12 GiB of VRAM (about 11 usable after its reserve) and
   16 GiB of RAM (about 13 usable), so a large share of real jobs will fail it.
   It must be reported as a permanent property of the node rather than as a
   transient wait:

   > `will not fit on desktop-unr95nb (needs 18.0 GiB VRAM, card has 12.0);
   > waiting for the 5090`

   is actionable. "waiting for VRAM" is not. The same applies to RAM and —
   because of [§2.4](#24-commit-is-the-binding-constraint-on-a-16-gib-box) — to
   the two summed against the commit ceiling.

Then, for each eligible node, ask: *would this job be admitted there right
now?* — using `resources.admit()` against that node's reported state.

### 5.2 Making `admit()` node-agnostic

`admit()` is already nearly pure: it takes `config, request, running, gpu, mem,
reserve`. Two things reach for the local machine:

- `resources.capacity()` calls `cpu_count()` → `os.cpu_count()`;
- `admit()` calls `host.commit_ceiling_mib(mem)`, which reads this machine's
  registry for the pagefile maximum.

The refactor is to introduce a `NodeSnapshot` carrying `mem`, `gpu`,
`cpu_count`, `commit_ceiling_mib`, `reserve` and `running`, and to thread it
through `capacity()` and `admit()` instead of reaching for globals. Local
behaviour is unchanged — the local snapshot is built from the same calls in the
same order.

This is a small mechanical change with disproportionate leverage. It is the
whole of what the scheduler needs to reason about a machine it is not running
on, and as a side effect it makes admission unit-testable against hypothetical
machines, which today it is not.

Note what this refactor deliberately does *not* do: it does not move the
authoritative decision off the worker. The primary's evaluation is a
prediction, made from a report up to a few seconds old. The worker's own
dispatcher re-runs the identical function against live numbers before it starts
anything. Divergence is rare, self-correcting, and always fails in the safe
direction — the worker refuses, the primary re-places on the next tick.

### 5.3 One call per tick

Because Windows OpenSSH cannot multiplex, the primary must not make one SSH
call per node per job per tick. Instead: **one call per node per poll interval**,
returning everything.

A new hidden command `workerq _node-report --json` on the worker returns, in a
single payload:

- `resources.describe_capacity()` — capacity, host memory, commit ceiling, the
  live reserve;
- the running jobs with their declared `ram/vram/cpus` (so the primary can
  compute `sum_reservations` itself);
- GPU device inventory and per-device free memory;
- worker-q version, dispatcher health, slot count;
- the node's own queue depth (which should normally be zero — see
  [§4](#4-the-queue-lives-on-the-primary)).

**Measured on this pair, 2026-09-09.** The batching assumption was worth
checking rather than assuming, and it holds convincingly:

| | |
| --- | --- |
| Tailscale RTT, direct over LAN | **13–28 ms** |
| One `ssh host cmd` round trip | **543 ms** (509–568 over 5 runs) |
| Five commands in **one** ssh call | **630 ms** |
| `scp` throughput | **6.1 MB/s** (49 Mbit/s) |

Two conclusions follow directly.

**Nearly all of the cost is fixed.** 543 ms for one command, 630 ms for five —
so the connection costs ~530 ms and each additional command about 22 ms. Five
separate calls would be 2.7 s; one batched call is 0.63 s, a **4.3x saving**.
The single-`_node-report`-per-tick design is not a micro-optimisation, it is
the difference between a workable poll loop and a broken one.

**The wire is not the problem; the handshake is.** 19 ms of network under
520 ms of SSH and process spawn. There is no tuning that fixes this on Windows
— `ControlMaster` is unsupported — so the answer is only ever "make fewer
calls".

Poll interval for remote nodes is therefore **3 s** (not the 0.25 s local
tick), giving a duty cycle of roughly one-sixth and a report at most ~3.5 s
stale. Reports are cached with an explicit staleness stamp, and every placement
decision records how old the report it used was, so `workerq show --json` can
explain a misplacement after the fact.

### 5.4 Scoring: spill under contention, stay home when idle

With asymmetric hardware, "best fit" is the wrong objective. The worker runs
the same job more slowly, so sending work there is a win only when it buys an
earlier start. The rule is therefore about **contention, not fit**:

| Primary | Worker | Decision |
| --- | --- | --- |
| — | cannot take it | **local**, always |
| can start now | can take it, **nothing else queued** | **local** — faster silicon, no transfer, and moving it buys nothing |
| can start now | can take it, **other jobs queued behind** | **worker** — this job runs slower, but the one behind it starts now |
| cannot start now | can start now | **worker** |
| cannot start now | cannot start now | queue on the primary; the wait reason names the constraint on *both* machines |

The third row is the whole feature. It is also the row that is subtly wrong if
implemented naively: the queue behind the job must be counted as *jobs that
would become startable on the primary if this one moved*, not as raw queue
depth. A queue full of 30 GiB jobs is not a reason to exile a small one.

Two tie-breakers sit below the contention rule:

- **Data gravity.** A job with large `--passthrough` inputs present on both
  machines prefers the primary, where they were staged first.
- **Headroom.** Between two nodes that would both start it now, prefer the one
  left with more usable headroom afterwards.

Once [Phase 7](#phase-7--node-aware-learning) supplies per-node durations, the
depth heuristic should be replaced by the comparison it stands in for:

```text
finish_local  = forecast_queue_wait_local + duration_on_primary
finish_worker = bundle_transfer           + duration_on_worker
```

Pick the smaller. Until there is history on both machines that comparison
cannot be made honestly, and a confidently wrong ETA is worse than none — so
the heuristic ships first and is replaced, not guessed at.

The existing backfill and starvation guards apply unchanged throughout.
---

## 6. Getting the source there

Both machines are on Wi-Fi, so per-job wire cost is a first-class constraint,
not an afterthought.

The worker clones each repo **once, from GitHub, over its own internet
connection** — not across the Wi-Fi link, and not from the primary. It has the
git credentials already.

Per job, the primary ships only a **thin bundle**: the snapshot commit computed
against what the worker already has.

```text
primary                                          worker
───────                                          ──────
git commit-tree  →  refs/gpuq/snapshots/<id>
                    (already built today)

                     ask: what do you have?  ──►  git rev-parse HEAD of clone
                                             ◄──  <base>

git bundle create job-<id>.bundle
    <snapshot> --not <base>          ──scp──►    job-<id>.bundle
                                                 git bundle unbundle
                                                 git worktree add --detach
                                                     <dest> <snapshot>
                                                 (link passthrough paths)
```

The bundle contains only the delta between the worker's clone and the snapshot
commit — for the usual case of a dirty worktree on a branch the worker already
has, that is kilobytes. `--passthrough` data is never transferred: it must
pre-exist on the node, which is precisely what the eligibility check in
[§5.1](#51-eligibility-then-fit) enforces.

On the worker, the staged worktree is submitted with `--live-worktree`. That
mode already means "run against this tree, do not snapshot it", which is
exactly right for a tree that is *already* a frozen snapshot. No new flag is
needed on the worker side, and the worker's manifest records the same commit
hash as the primary's — provenance stays intact across the link.

Steady-state per-job wire cost: a delta bundle out, a log and `result.json`
back. Both small.

**Measured throughput is 6.1 MB/s** (Wi-Fi to Wi-Fi, both ends). That makes the
design's choices load-bearing rather than merely tidy:

| Transfer | At 6.1 MB/s |
| --- | --- |
| Typical thin bundle (dirty worktree, tens of KB) | well under a second |
| A 50 MB bundle | 8 seconds |
| biohub's `.venv` (5.2 GB) | **15 minutes** |
| biohub's `data/train` (80 GB) | **3.7 hours** |

The last two rows are why environments are built on the worker and datasets are
fetched from their original source there, never pushed across this link.

It also argues for a **guard on bundle size**. A snapshot is built with
`git add -A`, so a stray large untracked file lands in the bundle and turns a
sub-second transfer into minutes with no explanation. Refuse — or at minimum
warn loudly and name the offending path — above a threshold; 64 MB is a
reasonable line, being ten seconds on this link and far larger than any honest
source delta.

### 6.1 The snapshot ships source, not environment

This is where the driver and architecture skew in the table at the top of this
document actually lands, and it is easy to miss because it is not a scheduling
problem at all.

A snapshot contains tracked and untracked-but-not-ignored files. `.venv` is
gitignored, so it is **never** in the snapshot; today a job reaches its
interpreter either by absolute path or by declaring `--passthrough .venv`.
Either way, **the environment is a property of the machine, not of the job.**

So the worker needs its own virtualenv per project, built on the worker, with
wheels for its own driver (591.86) and its own architecture (`sm_86`). It
cannot be copied from the primary: a Windows venv hardcodes paths, and wheels
selected for Blackwell are not what an Ampere card wants.

Two consequences:

- `workerq node stage` must cover **environment provisioning**, not just
  `git clone`. Cloning a repo whose jobs then fail on a missing interpreter is
  a worse outcome than not staging it at all.
- Passthrough verification ([§5.1](#51-eligibility-then-fit)) will catch a
  missing `.venv` and refuse to place the job, which is the correct behaviour —
  but the message must say *"`.venv` is not present on desktop-unr95nb; run
  `workerq node stage`"*, not just report a missing path.

The driver gap itself is not otherwise a problem: a standard PyTorch cu12x
wheel carries both `sm_86` and `sm_120`. A wheel built for one architecture
only, or a project pinned to a driver newer than 591.86, is a per-project
staging failure and should be discovered at stage time rather than at run time.

### 6.2 Junctions and cleanup

`snapshot.apply_passthrough` creates directory junctions on Windows, and
`snapshot.unlink_reparse_points` exists because a naive recursive delete of a
snapshot would follow a junction and destroy the live dataset. That hazard is
now duplicated on a second machine, where a mistake destroys data the primary
cannot see and did not put there.

The worker's own `workerq cleanup` handles its own snapshots with its own
guards — this is another dividend of running a full worker-q there rather than
an exec agent. The primary must not reach across and delete anything under the
worker's state directory.

---

## 7. Failure semantics

`future-slurm.md` named this as the trap: *"failure semantics when the link
drops mid-job (is it lost, or still going?)"*. Running a complete worker-q on
the worker is what makes the answer easy.

| Event | Behaviour |
| --- | --- |
| Link drops while a job runs | The job **keeps running**. The worker's dispatcher owns it, in its own Job Object, with its own logs and progress file. The primary marks it `RUNNING (node unreachable)` and stops placing new work there. |
| Link returns | Reconcile by label. The job's real state comes back from the worker; the primary adopts whatever actually happened, including that it finished twenty minutes ago. |
| Worker rebooted mid-job | The job died with the machine. The worker's dispatcher restarts and finds no matching process; the job reconciles to `FAILED`/`LOST` exactly as a local crash does today. |
| Worker offline at submit | Not eligible. The job queues on the primary and runs locally when it fits. |
| Primary restarts | The worker is unaffected and keeps running its jobs. On restart the primary re-reads its routing table and reconciles by label. |

**Never auto-resubmit.** A job that is unreachable is not a job that is gone,
and duplicating a multi-hour training run is worse than waiting. `workerq
reconcile` stays the repair path, as it is today for local crashes. The
existing rule that terminal states are immutable does the rest: a job that the
worker recorded as `SUCCEEDED` cannot be dragged back to `QUEUED` by a
reconnecting primary.

The one genuinely new state is *unreachable*, and it is deliberately a
**display qualifier, not a job state**. Adding an eighth state to
`ALLOWED_TRANSITIONS` to describe a property of the network rather than of the
job would be a mistake; `RUNNING` is still true.

---

## 8. What breaks that is not obvious

### 8.1 ETA and SUGGEST are silently node-blind

`eta.py` learns durations by `(project, command_signature)`, with no node
dimension:

```sql
WHERE project = ? AND command_signature = ? AND state = 'SUCCEEDED'
```

A 3080 Ti is not merely slower than a 5090 — on some workloads it is two to
three times slower. Pooling runs from both machines
produces an ETA that is right for neither, and the failure is silent — the
number just quietly becomes wrong, which is worse than the honest "unknown"
worker-q shows before it has history.

The distinction to draw is that **not all learned quantities are node-dependent**:

- **Duration** is. Key it by `(project, signature, node)`, or store a per-node
  speed factor and scale. Keying is simpler and honest; scaling recovers
  history faster. Prefer keying, and fall back to the pooled estimate marked
  as approximate until a node has its own history.
- **Peak RAM** is not, to a good approximation. A batch is a batch. Keep
  pooling it — this is what feeds the SUGGEST column, and halving its sample
  count for no reason would hurt.
- **Peak VRAM** is mostly not, with one asymmetric caveat that matters more
  here than it would between equal cards. A job that sizes itself to available
  VRAM measures larger on the 5090 than on the 3080 Ti. Pool the samples, but
  a suggestion derived from a 5090 run must never be used to declare a job
  ineligible for the worker on VRAM it would not actually have used there —
  otherwise the first big-card run permanently exiles the job from the small
  card.

### 8.2 Version skew — and why `--version` cannot detect it

Two installs that can drift. The wire format is `--json` output from one
worker-q parsed by another, so a mismatch is a parsing failure at the worst
moment.

The obvious check is to compare versions. **It does not work, and this is not
hypothetical — the two machines are already skewed:**

```text
PRIMARY  workerq 1.2.0   config get scheduling.backfill_max_hold_seconds -> 1800
WORKER   workerq 1.2.0   config get scheduling.backfill_max_hold_seconds -> error: unknown key
```

Identical version strings, different code. The worker is at `f887068`; the
primary carries `16b2846` and `9b42cff` on top of it, which added a config key,
a database column (`vram_source`) and changed how RAM is measured. `__version__`
was not bumped, because it is a release number and these were not releases —
and it is stored in two places with no single source of truth, so it will be
missed again.

So the node handshake must compare something that actually tracks behaviour:

- **A `NODE_PROTOCOL_VERSION` constant**, bumped only when the node report
  format or the semantics of a field change. This is the thing that decides
  whether two installs can talk, and it is the only check that should *refuse*
  dispatch.
- **The build commit**, reported alongside for diagnostics — read from the
  source checkout at install time. It answers "why does that node behave
  differently" without requiring anyone to have bumped anything.
- **A capability list** for optional features, so a node that cannot yet
  measure VRAM by whole-card delta says so rather than returning nulls the
  primary misreads as zero.

`workerq node check` prints all three side by side and refuses to dispatch on a
protocol mismatch. Comparing `--version` alone would have reported these two
machines as identical.

### 8.3 Two configs

The worker has its own `config.toml` with its own reserve, thresholds and
concurrency. That is correct — a 12 GiB card wants a different
`free_memory_threshold_percent` than a 32 GiB one, and the worker needs no
gaming reserve at all. But it is a second file to keep coherent, and a
misconfigured worker looks like a scheduling bug. `workerq node check` should
print both configurations side by side.

### 8.4 Clock skew

All timestamps are ISO-8601 UTC, which is the right foundation, but ages and
runtimes computed on the primary from timestamps written by the worker will be
wrong if the clocks differ. The node report should carry the worker's current
time so the primary can measure and warn about skew rather than silently
mis-render a runtime.

### 8.5 Dead config that will now matter more

`preemption.max_preemptions` is documented as the anti-starvation guard, is
validated in `config.validate()`, and is **never read** — `Job.preemption_count`
is written by the runner and never consulted by the scheduler. This is a
pre-existing bug, but a two-node queue displaces jobs more often, so it should
be fixed before or alongside this work rather than after.

### 8.6 Not all passthrough is alike, and one kind diverges silently

The eligibility check in [§5.1](#51-eligibility-then-fit) verifies that each
declared `--passthrough` path **exists** on the candidate node. Running that
check by hand against the prepared worker, 2026-09-09, gave a result worth
recording before any code depends on the assumption it overturns:

```text
kaggriculture     1 of  9 passthrough paths present
arc-whest         1 of 29 present
biohub            4 of 10 present
```

**None of the three projects was dispatchable**, despite all three having been
staged, built and smoke-tested on the worker. The smoke tests passed because the
worker's clones are behind the primary and do not contain `.gpuq.toml`, so no
passthrough was declared and the jobs happened not to need any. The moment the
primary dispatches, it applies *its* `.gpuq.toml`, and the truth appears.

That is the check earning its place. But sizing what was missing showed the
deeper point: **treating every passthrough entry the same is what makes the
number look frightening.** They fall into four kinds, and only one of them
should ever be copied.

| Kind | Example | What to do | Cost if you get it wrong |
| --- | --- | --- | --- |
| **Environment** | `.venv`, `.venv-gpu` (2.9 GB) | **Build on the worker.** Windows venvs hardcode paths; wheels differ by driver and arch. | Copying it appears to work and then fails obscurely |
| **Regenerable cache** | `.cache` (12.3 GB), `weights_P01.f32` (3.0 GB), `runs/cache` | **Let the worker rebuild it.** | A cache derived on a 5090 is not obviously valid for a 3080 Ti |
| **Real input** | truth `.npz`, `models/clean`, benchmark tapes | **Copy.** Usually small. | The job cannot run |
| **Write target** | `runs`, `runs/records`, `outputs`, `benchmarks/ablations` | **Reconcile — see below.** | Silent divergence |

Applying that classification collapses the staging cost from about 22 GB across
the two projects to **roughly 1 GB of genuine inputs**. `arc-whest`'s 29 entries
are 3 venvs to build, one 3 GB regenerable cache, and ~0.7 GB of truth arrays —
about two minutes of copying, not nineteen.

#### The write-target problem is universal here, not a biohub quirk

All three projects write through a passthrough, and each `.gpuq.toml` says so in
its own words — kaggriculture calls `runs` "league/trace/search OUTPUT; the
junction makes a job's results land in the real repository", and arc-whest warns
that "a job that WRITES through one writes into the real repository".

On one machine that is exactly right, and it is why the junction exists. Across
two machines it is a silent correctness bug:

- **`runs/records` and `runs`** — two machines appending their own results to
  their own local copies produce two divergent histories that nothing
  reconciles.
- **Absolute output paths.** biohub's `.gpuq.toml` instructs jobs to write to
  `C:/Users/samsu/Documents/biohub/outputs/submission.csv`. Repos live under
  `C:\Users\samsu\Documents\<project>` on **both** machines, so that path
  resolves on the worker too — to the worker's disk. The job runs correctly,
  reports `SUCCEEDED`, and leaves its output on a machine nobody is looking at.

That second failure is worse than a missing dataset, because a missing dataset
fails loudly and this succeeds wrongly.

So `.gpuq.toml` must distinguish inputs from outputs:

```toml
[snapshot]
passthrough = [".venv", "experiments/packed/D11_truth_256x32.npz"]  # read
outputs     = ["runs", "runs/records", "outputs"]                   # written back
```

and worker-q's rule becomes: **a job that declares outputs either runs on the
primary, or has those outputs pulled back on completion.** Because every project
here has write targets, this is not a refinement to add later — it is a
prerequisite for dispatching anything, which is why it is now
[Phase 4b](#phase-4b--output-reconciliation--done) rather than a footnote.
---

## 9. CLI surface

New:

```bash
workerq node add desktop-unr95nb --host 192.168.1.x --user samsu   # register
workerq node list [--json]                                   # inventory + live state
workerq node check [NAME]                                    # ssh, version, CUDA, config diff
workerq node stage NAME --repo <path>                        # first clone, once per repo
workerq node drain NAME                                      # finish current work, accept no more
workerq node rm NAME
```

Changed:

```bash
workerq submit --node desktop-unr95nb -- ...    # pin (also: --node local)
workerq status                           # both machines in one table, NODE column
workerq top                              # per-node resource panels
workerq logs <id> --follow               # transparently streams from the owning node
workerq cancel / bump / promote / wait   # unchanged surface, node-aware underneath
workerq reserve --node desktop-unr95nb          # reclaim a specific machine
workerq resources --node desktop-unr95nb
workerq doctor                           # checks every registered node
```

Hidden: `workerq _node-report --json`.

Nodes live in `config.toml` as an array of tables:

```toml
[[node]]
name = "desktop-unr95nb"
host = "192.168.1.42"
user = "samsu"
poll_interval_seconds = 3
```

`config.py` currently loads fixed dataclass sections; an array of tables is a
new shape there and needs handling that preserves the existing "unknown keys
are ignored rather than fatal" rule.

Everything is behind `[cluster] enabled = false` until the whole path is
proven. With no nodes registered, worker-q must behave exactly as it does
today, and the existing test suite must pass unmodified — that is the
regression bar for Phases 1–4.

---

## 10. Phases

Each phase is independently useful and independently revertible.

### Phase 0 — prove the ground (no code)

The point is to fail fast if the platform says no. Hardware is now known; what
is left is behaviour. **The step-by-step runbook is
[multi-node-setup.md](multi-node-setup.md)**; the summary is:

1. Enable OpenSSH Server on the worker; key auth; confirm the
   `administrators_authorized_keys` path works.
2. From the primary, over SSH, run `nvidia-smi` and a real
   `torch.cuda.is_available()` **with the worker's screen locked and no RDP
   session**. This is the Session 0 question in
   [§2.1](#21-the-session-0-constraint-and-why-it-is-smaller-than-it-looks) and it decides whether any of this is
   possible.
3. Confirm auto-logon plus a Task-Scheduler-at-logon dispatcher survives a
   reboot.
4. **Set a fixed pagefile with an explicit maximum** ([§2.4](#24-commit-is-the-binding-constraint-on-a-16-gib-box))
   and confirm `workerq resources --json` on the worker reports the expected
   `commit.ceiling_mib`.
5. Check free disk on the worker against pagefile + snapshots + the datasets
   you intend to stage there.
6. Measure round-trip latency of a trivial `ssh <node> cmd` and throughput of a
   50 MB `scp`. These set the poll interval and validate the thin-bundle
   choice.
7. Build one project's venv on the worker and run its smallest real job by
   hand, start to finish. This is the environment question in
   [§6.1](#61-the-snapshot-ships-source-not-environment), and it is the second
   most likely thing to go wrong after Session 0.

### Phase 1 — see both machines (no dispatch) ✅ registry and reports done

Node registry, `workerq node add/list/check`, `_node-report`. No job ever
leaves the primary.

Shipped: `[[node]]` tables in `config.toml` (`local` reserved, duplicates and
malformed entries refused, unknown keys inside a node dropped so a newer
config cannot brick an older worker-q); `workerq.nodes` with `NodeReport`,
`local_payload`/`remote_report`, a `ReportCache` that stamps every reading with
its age, and `compatibility()` gating on `NODE_PROTOCOL_VERSION` rather than
`__version__`; `workerq node add/rm/list/check`; and the hidden
`_node-report --json`.

`NodeReport.snapshot()` returns the `NodeSnapshot` from
[Phase 2](#phase-2--node-agnostic-admission--done), which is the join between
the two phases: a report from the worker feeds the same `admit()` that guards
the primary. A test asserts a 20 GiB job is refused against a *reported*
12 GiB card.

Two things learned by pointing it at the real machine. An unreachable node
must report **no** capacity rather than zero capacity, or a scheduler reads
"switched off" as "idle and empty". And "offline" covers several problems that
need different fixes — wrong key, no sshd, machine asleep, worker-q too old —
so the reason is classified rather than echoed; the first version surfaced a
Rich box-drawing border as the error text, because the reason was in the
middle of the output and the border was the last line.

Still to do in this phase: surfacing nodes in `status` and `top`.

This ships real value on its own — one dashboard for both machines — at
essentially zero risk, and it exercises the transport, the report format, the
staleness handling and the version check before anything depends on them.

### Phase 2 — node-agnostic admission ✅ done

The `NodeSnapshot` refactor of [§5.2](#52-making-admit-node-agnostic). Pure
refactor: no behaviour change, no new commands.

`resources.NodeSnapshot` carries the four readings `admit()` used to take for
itself — host memory, GPU inventory, CPU count and the commit ceiling — plus
the reserve and a node name. `capacity()` and `admit()` take an optional
`node=`; every existing caller still passes `gpu=`/`mem=`/`reserve=` and gets
a locally-built snapshot, which is what kept the diff to one module.

The deliverable is six tests in `tests/unit/test_resources.py` that admit and
refuse against *described* machines — the real 5090 and 3080 Ti — rather than
whichever host runs the suite. They pin the behaviours placement depends on:
a 0-VRAM job is admitted on either machine; a 20 GiB job is refused on the
smaller card with VRAM named in the reason; identical running load fills the
worker while the primary still has room; and a machine with no pagefile
refuses on commit what the same machine with 48 GiB accepts. None of those
questions could be asked before, and the last one could not even be simulated,
because the ceiling came from the local registry.

### Phase 3 — source and environment staging ✅ shipping done

`workerq node stage`, delta-bundle creation, remote unbundle and worktree
materialisation, passthrough verification. Environment provisioning (per
[§6.1](#61-the-snapshot-ships-source-not-environment)) is still manual.

**Proven against the real pair.** A snapshot of this repository was shipped to
the worker and materialised: **7,536 bytes, 4.1 seconds**, and the worktree
came out at exactly the primary's commit. The economic claim in
[§6](#6-getting-the-source-there) is therefore measured rather than assumed —
7.5 KB instead of the 67 MB the repository would otherwise cost.

Three things the real machine corrected:

- **Directory names differ between machines.** This repository is `gpu-queue`
  on the primary and `worker-q` on the worker, so deriving the remote path from
  the local basename was wrong on the very first repo it met. Identity is the
  **origin URL**, normalised so `git@github.com:me/x.git` and
  `https://github.com/me/x` are one repository; the directory name is only a
  fast path and the fallback for a repo with no origin.
- **scp does not expand `%TEMP%`.** A command sent over SSH runs through
  `cmd.exe` and expands it, but scp talks to the sftp subsystem, which took the
  literal `%TEMP%\...` as a directory name. Anything handed to scp is resolved
  to a real path first.
- **A node that already has the commit is a result, not a failure.** Re-placing
  a job onto a node that ran it before produces an empty bundle, which git
  reports as an error; staging treats it as "already present" and skips
  straight to the worktree.

Cleanup uses `git worktree remove`, never a recursive delete — passthrough
entries are junctions to live data, and following one would destroy a dataset
the primary cannot see and did not put there.

### Phase 4 — dispatch, pinned only 🚧 job lifecycle proven end to end

**A job submitted here has run on the worker.** `RAN ON DESKTOP-UNR95NB`, in a
worktree materialised from a shipped bundle, with the log retrieved afterwards.
The full round trip - ship, submit, run, reconcile by label, fetch log, clean
up - works against the real pair.

`workerq/remote.py` is the client: `submit`, `job`, `find_by_origin`, `cancel`,
`log_tail`, `fetch_log`. `_submit-spec` is the receiving half.

**A job spec crosses as a file, never as a command line.** This repeats the
decision already made for the runner, which takes only a job id and reads argv
back from the database, because on Windows a `Popen` list is joined by
`list2cmdline` and re-parsed by the child's C runtime. Remote dispatch stacks
ssh, `cmd.exe` and typer on top of that, so user argv on a command line is a
quoting bug waiting for the first path with a space in it.

**The primary's job id travels in the job's label.** The node assigns its own
id, so the label is the only durable link between the two records - and the
case it exists for is the connection dropping *between* the node queueing a job
and the primary recording its id. The id is lost then; the label is not.

#### A correction to the architecture in [§3](#3-shape)

The diagram shows a `ClusterBackend` implementing `SchedulerBackend` and
composing the local and remote backends, so `GPUQService` stays unchanged. That
does not work, and the reason is worth recording rather than quietly redrawing.

`SchedulerBackend.submit()` is called at **submission** time. Placement happens
at **dispatch** time, one tick before a node will actually start the job
([§4](#4-the-queue-lives-on-the-primary)) - deliberately, so a job is never
committed to a machine that gets busy before it runs. A backend method that
must return a placement at submit time cannot express that.

So the shape is: **the dispatcher is the placement engine, and `remote.py` is a
client it calls.** There is no `ClusterBackend`. The local dispatcher gains
node awareness in its dispatch loop, and remote jobs are excluded from local
slot and reservation accounting, because their footprint is on the other
machine.

#### Path bugs found here

Two more path bugs of the same family were found and fixed here, both of which
only appear against a real machine:

- **A worktree path handed to the far side must be expanded.** `%USERPROFILE%`
  is resolved by `cmd.exe` for a command sent over SSH, but not by scp, and not
  by the remote *Python* reading a job spec. Remote repo paths are now resolved
  once, at the point they are built.
- **scp wants forward slashes.** Uploads tolerate backslashes; downloads do
  not, which made the rule look optional until a log that plainly existed came
  back "No such file or directory".

### Phase 4b — output reconciliation ✅ done

Promoted out of a footnote by the finding in [§8.6](#86-not-all-passthrough-is-alike-and-one-kind-diverges-silently):
every project staged so far writes through a passthrough, so without this there
was nothing that could safely be dispatched at all.

**The dangerous case is refused before a job is placed.** `workerq/travel.py`
scans the command for absolute paths *inside the repository* and refuses to let
the job travel:

> the command writes to an absolute path inside the repository:
> `C:/Users/samsu/Documents/biohub/outputs/submission.csv`. That path exists on
> the other machine too, so the job would succeed there and leave its output on
> a machine you are not looking at.

A pin to a named node fails at **submit** time rather than waiting in the queue
against a message the submitter never sees. The check is deliberately narrow:
an absolute path *outside* the repo is left alone, because a missing dataset
fails loudly and that is the safe direction — it is only the paths that
*resolve* on both machines that diverge silently.

**What the job writes comes home.** `[snapshot] outputs` in `.gpuq.toml` is
read alongside `passthrough`, travels in the job spec, and is collected when
the job ends — before it is recorded as finished, because a job marked
`SUCCEEDED` whose results are still on the other machine is the exact failure
this phase exists to prevent.

Two decisions in the collector are worth knowing:

- **Only files modified after the job started come back.** A declared output
  path is a junction to the node's *live* repository, so both machines have
  their own copy of that directory and a wholesale copy would clobber one with
  the other. Verified both ways: a file written during the job is collected, and
  the same file with a marker set after it was written is not.
- **The comparison happens on the node, against the node's own clock.**
  Comparing a remote file's timestamp against this machine's clock would
  silently include or drop files whenever the two disagree, and
  [§8.4](#84-clock-skew) says they will.

Results return as one archive rather than file by file: at 6.1 MB/s and ~540 ms
per connection, a hundred small result files copied individually would spend a
minute in handshakes to move a megabyte. Extraction refuses any archive member
that would land outside the repository.

Collection failing does not fail the job — the results still exist on the node
— but it is logged as *"results are still on `<node>`… they are not lost"*
rather than passing quietly.

### Phase 5 — automatic placement ✅ done

worker-q now chooses. Proven with two jobs that could not both run here:

```text
job 8: it blocks a later job here, so looking for another machine
job 8: placed on 3080ti as its job 12 (28004 bytes shipped)
job 9: keeping local: moving it would not free anything
```

`A on DESKTOP-UNR95NB` and `B on Sam_Mega_PC`, in parallel rather than one
after the other.

The contention test is a **counterfactual, not a queue-depth count**: is there
a job behind this one that cannot start now but could if this one went
elsewhere? Depth would be the wrong measure, because a queue full of 30 GiB
jobs is not a reason to exile a small one — moving it frees nothing they can
use.

Every placement decision is logged once, when it changes. A job that ran on the
slower machine, or did not, is otherwise impossible to argue about afterwards;
and an unthrottled line on a loop that ticks four times a second would
reproduce the 303,164 identical lines that commit `16b2846` had to fix.

`[scheduling] auto_placement = false` is the escape hatch, and `status` grows a
`NODE` column — but only on a machine that has somewhere else to send work.

One bug this work introduced and fixed: **`workerq config set` silently deleted
the node registry.** `to_dict()` deliberately excludes nodes because it feeds
the dotted-key coercion machinery, but both mutation paths rebuilt a `Config`
from that dict — so the first config change after registering a node wiped it,
and the dispatcher then had nowhere to place anything while `node list` still
showed the node online.

### Phase 6 — the unhappy paths ✅ mostly done

Deliberately its own phase: this is what `future-slurm.md` warned about, and
folding it into Phase 5 would mean shipping the happy path and discovering the
rest in production.

Done:

- **Link drop.** A node that cannot be polled leaves its jobs `RUNNING`.
  Unreachable is not finished, and collapsing those two is how a live training
  run gets recorded as failed and started again somewhere else.
- **No duplicate work.** `_submit-spec` is idempotent by label: re-sending a
  spec adopts the job already there rather than creating a second one. This
  guards the window where the node accepts a job and the sender dies before
  recording its id — the job is still `QUEUED` on the sender, so it would
  otherwise be sent again. The check lives on the *receiving* side, because
  asking from the sender would add a whole SSH round trip to every placement to
  guard against something that almost never happens.
- **Protocol refusal.** A version gap produces a parse error *after* the work
  has been queued on the far side, so dispatch stops on a protocol mismatch.
  Only the protocol may refuse; a differing `__version__` is reported and
  tolerated, because that is the check which already failed to notice a real
  skew between these two machines.
- **Clock skew.** Output collection compares file times against the node's own
  clock — deliberately, since comparing against this machine's would be wrong
  whenever the two disagree. But a node minutes out would silently collect the
  wrong files and report success, so more than two minutes of skew stops
  dispatch. Unknown is not treated as wrong.
- **`node drain` / `node enable`.** Stops placement without touching what is
  running. It deliberately does not cancel: a drain that killed jobs would just
  be `cancel`, and the reason to drain is usually that you want the work to
  finish.

Still open: surfacing *"running but its node is unreachable"* in `status`. The
job is safe and the state is correct; it simply looks like any other running
job until you check `workerq node list`.

### Phase 1 — monitoring, completed alongside

`status` and `top` both show the other machine:

```text
VRAM    █▌──────────────────────   6.3%  2.0 / 31.8 GiB  util 7%
RAM     ███████████▉────────────  49.9%  30.7 / 61.6 GiB
Commit  ████████████▎───────────  51.2%  47.9 / 93.6 GiB
Usable  54 GiB RAM · 14 CPU · 31 GiB VRAM
3080ti  █▎──────────────────────   5.6%  ram 7.4/15.9G · 0 job
```

Nothing on a render path does I/O. The dispatcher polls each node on a
background thread and publishes what it saw into the queue meta table; `top`,
`status` and `node list` read it from there. That is the mechanism the reserve
and slot count already use — the queue database is the channel — and it is
forced here by arithmetic: the dashboard redraws about once a second and an SSH
round trip costs ~540 ms, so polling on the render path would stall it for half
of every frame.

Polling cannot live in the dispatch tick either. At 540 ms against a 0.25 s
loop it would stall cancellation and reaping a fifth of the time. But
publishing *must* stay on the main loop, because a SQLite connection belongs to
the thread that created it — writing from the poller raised, the exception was
swallowed, and reports silently never appeared.

### Phase 7 — node-aware learning ⛔ not started

`eta.py` still pools durations across machines, so a 3080 Ti run and a 5090 run
of the same command feed one estimate that is right for neither — see
[§8.1](#81-eta-and-suggest-are-silently-node-blind). `jobs.node` now records
where each job ran, which is the input this needs; the work is to key duration
history by it while leaving peak RAM pooled.

### Phase 7 — node-aware learning

The ETA/SUGGEST corrections of [§8.1](#81-eta-and-suggest-are-silently-node-blind).
Separate because it needs history from both machines to validate, which does
not exist until Phases 4–6 have been running for a while.

---

## 11. What stays out

- **A third machine.** Two is the ceiling this design is honest about. A third
  is the point at which `future-slurm.md`'s argument reasserts itself and the
  answer becomes an existing scheduler.
- **Cross-node preemption.** A critical job displaces a preemptible job on the
  node it lands on. "Should I displace a job on the worker or wait for the
  primary" is a real scheduling question and not one worth answering here.
- **Migrating a running job.** A preempted job restarts from the beginning, as
  it does today.
- **A shared filesystem, or worker-q replicating datasets.** Data is staged by
  the user, and verified by worker-q. Making the primary a file server over
  Wi-Fi is the thing this design is shaped to avoid.
- **A network listener, an agent protocol, or a web dashboard.** SSH is the
  transport; `--json` is the wire format; `workerq top` is the dashboard.
- **Automatic resubmission of anything.**

---

## 12. Open questions

1. **Which of the 35% that cannot fit are genuinely large, and which are
   mis-declared?** [§0.2](#02-over-declaration-will-cost-you-the-worker) shows
   50 jobs are excluded by a wrong RAM number alone. Correcting the busiest
   projects' declarations is free capacity and needs no new code.
2. **Undeclared data is the failure mode this design cannot see.** A job that
   hardcodes `D:\datasets\...` in a config file rather than declaring it with
   `--passthrough` will pass eligibility and then fail on the worker with a
   confusing error. Options: do nothing and let it fail loudly; require an
   explicit `--anywhere` opt-in for a job to be allowed to travel; or let the
   first failure on a node teach worker-q to pin that project. This is the
   sharpest remaining unknown, and it is a UX question rather than a technical
   one.
3. **Retention on the worker.** The primary keeps 7 days of successful
   snapshots. The worker should keep fewer — it has less disk, and the primary
   holds the authoritative provenance — but "fewer" needs a number, and it
   interacts with the pagefile sizing in [§2.4](#24-commit-is-the-binding-constraint-on-a-16-gib-box).
4. **Which projects to stage first.** Staging has a real per-project cost (a
   clone plus a venv plus datasets). Rather than staging everything, pick the
   two or three projects that generate the most small jobs, and let
   eligibility keep the rest local until it is worth the effort.
5. **Whether `max_concurrent_jobs = 3` on the worker is optimistic.** With
   13 GiB of usable RAM and the default 3 GiB charge for undeclared work,
   three concurrent jobs is close to the ceiling. It may want to be 2.
