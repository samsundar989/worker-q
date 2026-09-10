# worker-q web

A locally-served, dependency-free web UI that sits beside `workerq top` and
adds the two things a TUI frame cannot give: **history you can interrogate**,
and **a verdict on whether declarations match reality**.

    workerq web

Serves on `127.0.0.1:7676` (`[core] web_port`), opens a browser, and reads the
same databases the CLI does. This document is the design record; everything it
describes is implemented.

## Why not just improve the TUI

`workerq top` renders one frame of the present. Everything it needs is on
screen for two seconds and then replaced. The questions this feature set is
about are all *cross-sectional*:

- what did this command do the last twenty times it ran?
- which machine has been doing the work, and did placement help?
- is a 12 GiB declaration honest, or is it parking 10 GiB that nothing uses?
- are we waiting in the queue longer than we are computing?

None of those fit in a frame. They need sorting, filtering, a time axis and a
pointer you can hover. That is a browser's job.

The TUI stays. It is the right tool over SSH and on a wedged box, and it is
already the fastest way to see the machine falling over.

## What the data said before any of this was built

Measured against the live store, 486 jobs, 328 of them successful:

| | |
| --- | --- |
| median measured peak / declared RAM | **0.25** |
| jobs over-declaring RAM by 4x or more | **136 of 276** |
| RAM reserved but never used | **631 GiB-hours** |
| median runtime / declared ETA | **0.20** |
| measured jobs with 2 or fewer samples | **63 of 262** |
| jobs with a recorded node | **4 of 486** |
| `job_started` events still retained | **239 of 486** |

The feature is worth building. Half the queue's waiting is for memory that is
never touched, and the ETAs the queue view prints are pessimistic by 5x. But
four of those rows are also a warning: **the record is not yet good enough to
draw the picture.** Phase 0 exists to fix that first.

---

## Phase 0 - fix the record

A view cannot show what was never written down. Each item below was a
prerequisite for a specific view. All five landed before anything rendered a
number, and each improves the CLI too - `workerq resources --verify` is more
correct now whether or not the web UI is running.

### 0.1 Separate the commit ledger from the RAM ledger

**This is the correctness spine of the whole feature.** Get it wrong and the
tool will confidently advise shrinking declarations that are already correct,
which is how the box goes down.

`peak_ram_mib` is not RAM. `host._windows_processes` records `PrivateUsage`,
which is **commit charge** for the whole process tree, and the module comment
says why: working set reports a training job as tiny while it is the largest
thing on the machine. Meanwhile `--ram` is admitted against *free physical
memory*. Under WDDM the driver backs video allocations with system commit, so
a GPU job's commit is roughly its RAM plus its VRAM, and `resources.admit`
already knows this and gates on it.

So a GPU job whose measured peak exceeds its declared RAM is usually not
under-declared at all. Of the 21 jobs whose peak exceeds declared RAM, **18
declared VRAM, and 11 of those fit inside declared RAM + VRAM**. Only five are
genuinely over their whole commit budget.

The fix is presentational, not a migration:

- compare `peak_ram_mib` against `requested_ram_mib + requested_vram_mib` for
  any job that declared VRAM, and against `requested_ram_mib` otherwise;
- label the measured column **peak commit**, not peak RAM, everywhere;
- derive a verdict per job: `ok`, `over-declared`, `under-declared`,
  `unmeasured`, `low-confidence`.

Do not rename the column. `eta._learned_peak` and `report.declared_vs_observed`
both read it, and the name is load-bearing in a schema already at version 8.

### 0.2 Bring remote measurements home

Jobs that ran on the second machine have `peak_ram_mib` NULL and
`usage_samples` 0. The node measured them perfectly well; the numbers just
never crossed back. `_reap_remote` already fetches the log and collects
declared outputs, and `remote.list_jobs` already returns the node's full
`Job.to_dict()`, which contains every field needed.

Copy `peak_ram_mib`, `peak_vram_mib`, `usage_samples`, `peak_source`,
`vram_source` and `progress_fraction` onto the primary's row at reap.

Without this, the machines view and the accuracy view silently exclude every
remote job, which is precisely the comparison the feature is for.

### 0.3 Record which machine, for local jobs too

`jobs.node` is NULL for 482 of 486 rows because NULL means local. That is fine
in the store and **must not be migrated**: `eta._durations_for` matches
`node IS NULL` to find same-machine history, and rewriting the column would
break duration learning.

Coalesce NULL to `local` in the read layer only.

### 0.4 Stop routine noise from evicting job history

`telemetry.prune` keeps the newest 50,000 events across all kinds. 49,454 of
them are `job_blocked`. The result is that lifecycle events for most jobs are
already gone, and the events table currently reaches back six days while the
samples table reaches back eight.

Make retention per-kind: a generous floor for `job_started`, `job_finished`,
`job_preempted`, `job_cancel` and `reserve`, and a separate, lower cap for
`job_blocked` and `resource_pressure`.

Also populate `events.job_id`. It is NULL in all 50,000 rows; only
`backend_job_id` is set, so a per-job timeline needs a join through
`jobs.backend_job_id` that nothing currently does.

### 0.5 Make short-job peaks trustworthy

The runner samples usage every 15 seconds. A 41-second job therefore records
two samples, and a two-sample peak is close to meaningless. It is whatever the
tree happened to hold at two arbitrary instants.

Sample every 2 seconds for the first 30 seconds, then fall back to 15. The
cost is negligible: `host.memory` is a `ctypes` call into
`GlobalMemoryStatusEx`, and the tree walk is already cached for 3 seconds.

Independently of that fix, the UI must treat `usage_samples` as a confidence
signal and never render a peak from 2 samples with the same authority as one
from 600.

---

## Phase 1 - the server

### Shape

`workerq web [--port N] [--host 127.0.0.1] [--open]`, a new command in
`cli.py` beside `top`.

**No new dependencies.** The project ships with Typer and Rich and nothing
else, and that restraint is worth keeping for a tool whose whole job is to
stop this machine falling over. `http.server.ThreadingHTTPServer` from the
standard library is sufficient for a single-user local dashboard.

**Loopback only, always.** The spec's rule against an unauthenticated network
server is why the dispatcher has no socket. A local UI does not breach that
rule as long as it cannot be reached from off the machine. Refuse a
non-loopback `--host` unless a config key explicitly allows it.

Port 8787 is already taken on this machine by an unrelated project, so the
default is 7676. It is `[core] web_port` rather than a section of its own,
because `config.to_toml` regenerates the file on every `config set` and one
more key is a far smaller change than one more section.

### How it talks to the queue

The webapp is a **client, like the CLI**. It introduces no new control
channel, and the database stays the only thing the dispatcher listens to.

- **Reads** go straight to the three SQLite files in read-only mode. WAL means
  readers never block the dispatcher and it never blocks them.
- **Writes** go through `GPUQService`, never raw SQL, because state
  transitions are validated in `Database.update_job` and cancel, promote and
  reserve are meta-flag protocols the daemon polls for. A write that bypasses
  the service can resurrect a cancelled job.
- **Node state** comes from `nodes.published_reports`, which reads the
  dispatcher's published reports out of the queue meta table. Never SSH on a
  request path; a round trip is about 540 ms.

Two hazards, both already documented in the codebase because both have bitten:

1. **A SQLite connection belongs to the thread that created it.** This broke
   the dispatcher's node reporting once and the runner's progress watcher
   once. Use thread-local connections; never share a `Database` or
   `QueueStore` across request threads.
2. **A long-lived read transaction pins the WAL and lets it grow.** The
   telemetry WAL is already 4 MB against a 53 MB database. Keep every read
   short, and never hold a cursor open across a response body.

Serialize writes behind a single lock. Concurrent writers get five seconds of
`busy_timeout` and then `SQLITE_BUSY`, and the CLI does not always survive
that.

### Endpoints

| Route | Returns |
| --- | --- |
| `GET /api/overview` | the frame the TUI renders: jobs, summary, forecast, throughput, reserve, GPU, host, top processes, node reports |
| `GET /api/jobs` | filtered, sorted, paginated history |
| `GET /api/jobs/{id}` | `job_detail` plus usage verdict, events, siblings |
| `GET /api/jobs/{id}/series` | telemetry samples for the run window, joined via `job_samples` |
| `GET /api/jobs/{id}/log?offset=` | incremental tail |
| `GET /api/accuracy` | declared against observed, grouped by project or signature |
| `GET /api/signatures/{sig}` | every run of one command, with its calibration |
| `GET /api/nodes` | published node reports |
| `GET /api/events` | lifecycle and pressure events |
| `POST /api/jobs/{id}/{cancel,bump,eta,describe,requests}` | through `GPUQService` |

`/api/overview` is close to free: `Dashboard.render` already composes exactly
this from five service calls, so the payload has a proven shape.

Refresh by polling at 2 seconds, matching the TUI's own interval. Server-sent
events would be tidier but hold a thread each in `ThreadingHTTPServer`; add
that later only if polling proves visibly stale.

### Frontend

No build step, no CDN, everything vendored under `src/workerq/web/static/`
and declared as package data. The tool has to work on a machine with no
network, with a browser opened from a terminal.

Plain ES modules, inline SVG charts drawn by hand, CSS custom properties for
theming. Hand-rolled charts sound like more work than a library, but the
charts here are few and specific, and it avoids shipping a megabyte of
JavaScript to draw six of them.

Take the state palette from `theme.STATE_STYLES` rather than inventing a third
one. The CLI and the TUI already share it deliberately.

---

## Phase 2 - the views

### Now

What `top` shows, with room to breathe. Both machines side by side, running
jobs with real progress bars, the queue with each blocked job's wait reason,
and the memory consumers tagged worker-q or foreign.

This view earns its place by making the pressure legible, not by being new.

### History

The main ask, and the easy win: **the jobs table is never pruned**, so the
record is complete back to the first job.

A sortable, filterable table over every job: id, project, description, node,
state, priority, queued, started, finished, wait, runtime, exit code, declared
against peak. Facets for project, node, state, priority, command signature and
date.

Above it, a timeline. Jobs as bars on a time axis, one lane per machine, so
overlap, serialization and placement are visible at a glance. Concurrency is
the story the flat table cannot tell.

### Job detail

Everything about one run, in one place: argv, snapshot commit, passthrough,
environment, the resource series for its window, its lifecycle events, its
log, its declaration against what it actually used with the verdict from 0.1,
and every sibling that shares its command signature.

The sibling list is what turns a single run into a trend.

### Accuracy

The estimation-quality view, and the reason for phase 0.

Per project and per command signature: the distribution of measured peak
against declared, ETA accuracy, and a confidence badge driven by
`usage_samples`. A ranked list of the worst offenders in both directions, each
with the exact command to fix it, `workerq requests <id> --ram N`, because a
finding you cannot act on in one copy-paste does not get acted on.

Show over-declaration and under-declaration as **separate rankings**, not two
ends of one axis. They have opposite consequences: over-declaring makes other
people wait, under-declaring takes the machine down. The second list should be
short and loud.

Surface the aggregate cost in GiB-hours reserved but unused. It is currently
631, and a single number is what makes the case for going and fixing the
declarations.

### Machines

Per node: jobs run, success rate, median runtime, median queue wait, GPU and
RAM utilization over time, and how often the second machine sat idle while
this one had a queue.

That last one is the honest test of `_choose_node`. Placement only pays when
moving a job lets a different job start sooner, and this view is where you
find out whether that is happening.

Meaningless until 0.2 lands.

### Efficiency

The "make jobs faster" half of the request, which is mostly not about the jobs.

- queue wait against runtime, per project. If we wait longer than we compute,
  the declarations are the bottleneck, not the code;
- aggregated block reasons, so the actual binding resource is named rather
  than guessed;
- head-of-line holds and backfill skips;
- preemption events, with what displaced what;
- idle machine while the queue was non-empty.

---

## Phase 3 - debuggability

Aesthetics are phase 2's problem. Debuggability is a separate feature and it
is the one that makes the difference between a dashboard and a tool.

Every panel offers the raw JSON behind it, names the service call that
produced it, and gives the equivalent CLI command. Every action shows the
`workerq` command it is about to run before it runs it.

That way the web UI never becomes a place where numbers appear without
provenance, and anything found in the browser can be reproduced in a terminal.

---

## Testing

`tests/unit/test_web.py`, using `http.client` against a server on an ephemeral
port. No new test dependencies either.

Cover:

- the route table and the shape of each JSON payload;
- **the accuracy classifier**, especially the commit-against-RAM verdict from
  0.1, with a GPU job that looks over-peak and is not;
- thread-local connection handling under concurrent requests;
- refusal to bind a non-loopback address;
- the remote-measurement merge from 0.2, against a fake node payload.

The existing fixtures in `tests/conftest.py` already build a temporary state
directory, so most of this is assembly rather than new scaffolding.

## What the first run found

The views were pointed at this machine's own 519 jobs as soon as they
rendered, which is the only real test of whether they say anything:

- the median job uses **22%** of the footprint it declares, and **1,500
  GiB-hours** of declared budget has been held and never touched;
- ten jobs went over the footprint they declared, and after the phase 0.1
  correction that list is *ten*, not the thirty-one a naive comparison
  produced - the other twenty-one were well-sized GPU jobs;
- the median job finishes in **16%** of its declared ETA;
- the second machine was idle for essentially every telemetry sample in which
  the queue had work waiting, against 11% for this one.

The last of those is the one worth acting on, and it is not something any
single job's record could have shown.

## One measurement that had to be thrown away

The first version of "idle while the queue had work" joined through
`job_samples`, and reported the second machine as idle 100% of the time. That
was not a finding - `job_samples` is written by the local dispatcher for jobs
it is running, and a remote job never enters `self.running`, so the number was
guaranteed by construction.

It now measures against the jobs' own start and finish times. The lesson
generalises: a metric that can only produce one answer is worse than no
metric, because it looks like a result.

## Noticed in passing

Two things found while mapping the code, unrelated to this feature but worth
fixing while nearby:

- `node_drain` and `node_enable` are each defined twice in `cli.py`, with
  identical bodies. Only the later registration takes effect, so an edit to
  the first copy would silently do nothing.
- `preemption.max_preemptions` is validated in `config.validate` and never
  read by the scheduler. `_preemption_candidates` has no `preemption_count`
  guard and no aging term, so a preemptible job can in principle be displaced
  indefinitely.
