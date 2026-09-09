# Implementing the worker node

Everything that has to happen on **`DESKTOP-UNR95NB`** (RTX 3080 Ti 12 GiB,
Ryzen 7 5800X, 16 GiB) to turn it from a spare desktop into a worker-q node the
primary can dispatch to. Companion to the design in [multi-node.md](multi-node.md).

> **Almost all of this can be done today.** worker-q already runs standalone on
> any Windows box, so Stages 0–6 need no new code — at the end of Stage 6 the
> worker is a fully working, independently usable worker-q installation that you
> can submit to *while sitting at it*. Only the primary-side dispatch
> (`RemoteWorkerqBackend`, `ClusterBackend`, placement) waits on
> [Phases 4–6](multi-node.md#10-phases). Doing this now de-risks that work
> substantially, because when it lands the worker is already known-good.

**Legend.** **[P]** = run on the primary (`SAM_MEGA_PC`). **[W]** = on the
worker. **[W-admin]** = elevated PowerShell on the worker (right-click → Run as
administrator).

| Stage | What | When |
| --- | --- | --- |
| [0](#stage-0--the-gate) | SSH reachability and the CUDA session test | **First. Stop and report after this.** |
| [1](#stage-1--make-it-a-server-not-a-desktop) | Power, pagefile, Defender, updates, auto-logon | Once |
| [2](#stage-2--toolchain) | Git, Python, uv, driver | Once |
| [3](#stage-3--install-and-configure-worker-q) | Install, config, `doctor` | Once |
| [4](#stage-4--dispatcher-lifecycle) | Start at logon, survive reboot | Once |
| [5](#stage-5--per-project-staging) | Clone, venv, datasets | **Repeats per project** |
| [6](#stage-6--verify-the-node-standalone) | Prove it works on its own | Once |
| [7](#stage-7--ongoing-operations) | Upgrades, retention, disk | Ongoing |

---

## Stage 0 — the gate

Nothing below matters if this fails. Report the results before continuing.

### 0.1 [P] Log back into Tailscale

The primary is currently **logged out**, which blocks everything else.

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" login
& "C:\Program Files\Tailscale\tailscale.exe" status
```

You should see both machines. Note the worker's name (probably
`desktop-unr95nb`). If MagicDNS names do not resolve, enable MagicDNS at
<https://login.tailscale.com/admin/dns>.

Tailscale is the *network*, not the shell: its SSH server component is
Linux/macOS only and cannot target Windows. The worker runs OpenSSH Server;
Tailscale carries it and gives us a name that does not change when the router
hands out a new DHCP lease.

### 0.2 [W-admin] Install OpenSSH Server

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd

Get-Service sshd | Format-Table Name,Status,StartType -AutoSize
Get-NetFirewallRule -Name *OpenSSH-Server* | Format-Table Name,Enabled,Profile -AutoSize
```

Expect `sshd  Running  Automatic` and a firewall rule with `Enabled  True`.

**Do not change `DefaultShell`.** The default `cmd.exe` starts faster than
PowerShell and avoids a second layer of quoting; worker-q calls
`powershell -NoProfile -Command` explicitly when it needs to.

### 0.3 [W-admin] Authorize the primary's key

The primary already has a key; no new one is needed:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpO1eux9gBnd8dnhC6CLER8sYpXsVaMsklrakc4gM8n samsundar989@gmail.com
```

Your worker account is an administrator, so sshd **ignores
`~/.ssh/authorized_keys`** and reads this file instead:

```powershell
$key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpO1eux9gBnd8dnhC6CLER8sYpXsVaMsklrakc4gM8n samsundar989@gmail.com'
$f   = "$env:ProgramData\ssh\administrators_authorized_keys"

$key | Out-File -FilePath $f -Encoding ascii
icacls $f /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
Get-Content $f
```

Two things silently break this:

- **`-Encoding ascii` is load-bearing.** `>`, `Out-File` defaults and
  `-Encoding utf8` all write a byte-order mark; sshd then ignores the file with
  no useful error.
- **The ACL must be exactly SYSTEM + Administrators.** sshd refuses a file any
  other account can write.

### 0.4 [P] Test the connection

```powershell
ssh desktop-unr95nb "hostname"
& "C:\Program Files\Tailscale\tailscale.exe" ping desktop-unr95nb
```

Expect `DESKTOP-UNR95NB`, and `via DIRECT`. `via DERP` means traffic is relayed
through Tailscale's servers — it works, but every snapshot bundle takes the
scenic route.

### 0.5 The CUDA session test

Install the toolchain first if it is not present — this is Stage 2 work, pulled
forward because the test needs it:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
```

Open a **new** shell so `PATH` refreshes, then:

```powershell
mkdir C:\cudatest; cd C:\cudatest
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Save as `C:\cudatest\check.py`:

```python
import torch, os
print("session   :", os.environ.get("SESSIONNAME", "(none)"))
print("torch     :", torch.__version__)
print("available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device    :", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok :", float((x @ x).sum()))
```

Run `C:\cudatest\.venv\Scripts\python.exe C:\cudatest\check.py` in **three**
situations and record which work:

| # | Situation | How |
| --- | --- | --- |
| **A** | Console session | At the worker's keyboard, normal PowerShell. **Not over RDP.** |
| **B** | SSH, console user logged in, screen **locked** | Sign in at the worker, `Win`+`L`, then `ssh desktop-unr95nb` from the primary |
| **C** | SSH, **nobody** logged in | Sign out fully on the worker, then `ssh desktop-unr95nb` from the primary |

**Only A is load-bearing.** worker-q's dispatcher runs in the console session
and launches every job from there; SSH carries only `submit`/`status`/`logs`/
`cancel`, which never touch CUDA.

- **A works** (expected): proceed.
- **B also works**: jobs *can* be launched over SSH. This does **not** mean the
  dispatcher can be restarted that way — one started in an SSH session dies
  with the session; see [4.3](#43-restarting-the-dispatcher-remotely).
- **C also works**: no auto-logon needed; skip [1.5](#15-w-auto-logon-only-if-05c-failed).
- **A fails**: stop. Something is wrong with the driver or card, unrelated to
  this project.

### 0.6 [W] Inventory

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object @{n='TotalRAM_GB';e={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}},
                @{n='FreeRAM_GB'; e={[math]::Round($_.FreePhysicalMemory/1MB,1)}}
Get-CimInstance Win32_ComputerSystem | Select-Object AutomaticManagedPagefile
Get-CimInstance Win32_PageFileSetting | Select-Object Name,InitialSize,MaximumSize
Get-PSDrive -PSProvider FileSystem |
  Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}},
                     @{n='TotalGB';e={[math]::Round(($_.Used+$_.Free)/1GB,1)}}
```

### 0.7 [P] Measure the link

```powershell
1..5 | ForEach-Object { (Measure-Command { ssh desktop-unr95nb "hostname" }).TotalMilliseconds }

$f = "$env:TEMP\50mb.bin"; fsutil file createnew $f 52428800
Measure-Command { scp $f desktop-unr95nb:C:/Windows/Temp/50mb.bin }
Remove-Item $f; ssh desktop-unr95nb "del C:\Windows\Temp\50mb.bin"
```

**Measured 2026-09-09.** Recorded here so the design's constants have a
provenance:

| | Result |
| --- | --- |
| `ssh desktop-unr95nb "hostname"` | `DESKTOP-UNR95NB` |
| `tailscale ping` | `via 192.168.1.202:41641` — **DIRECT**, not DERP |
| Tailscale RTT | 13–28 ms |
| SSH round trip, 5 runs | 509, 541, 543, 549, 568 ms — **median 543 ms** |
| Five commands in **one** ssh call | 630 ms |
| `scp` of 50 MB | 8.2 s → **6.1 MB/s (49 Mbit/s)** |

Two things worth internalising:

- **The wire is fine; the handshake is not.** 19 ms of network sits under
  ~520 ms of SSH and process spawn. Windows OpenSSH cannot multiplex, so this
  is not tunable — the only lever is making fewer calls.
- **Batching works almost perfectly.** One command 543 ms, five commands
  630 ms: ~530 ms fixed plus ~22 ms each. Five separate calls would be 2.7 s.
  This is why the design polls each node with a single `_node-report` call, and
  why the remote poll interval is 3 s rather than the 0.25 s local tick.

---

## Stage 1 — make it a server, not a desktop

### 1.1 [W-admin] Power

A worker that sleeps is a worker that drops jobs.

```powershell
powercfg /change standby-timeout-ac 0     # never sleep
powercfg /change disk-timeout-ac 0        # never spin down disks
powercfg /change monitor-timeout-ac 15    # screen off is fine
powercfg /hibernate off                   # frees ~16 GiB and disables Fast Startup
```

`powercfg /hibernate off` matters twice: it returns RAM-sized disk space you
need for the pagefile, and it disables Fast Startup, which otherwise makes
"restart" not a real restart and hides boot-time problems.

In the BIOS, set **restore power state after AC loss** to *On*, so the machine
comes back by itself.

### 1.2 [W-admin] Fixed pagefile

The single highest-leverage setting on this machine. Under WDDM the driver
backs video allocations with system commit, so a job costs roughly RAM + VRAM
of commit. `host.commit_ceiling_mib()` is RAM plus the pagefile's **configured
maximum**, and falls back to the current limit when the pagefile is
system-managed — invisible conservatism at 64 GiB, crippling at 16 GiB. See
[multi-node.md §2.4](multi-node.md#24-commit-is-the-binding-constraint-on-a-16-gib-box).

Only if 0.6 showed `AutomaticManagedPagefile : True` **and** you have the disk.
Size at 2–4× RAM; 48 GiB is a good target. Do not take the drive below ~20%
free.

GUI (safer, and what I would use):

> `sysdm.cpl` → Advanced → Performance **Settings** → Advanced → Virtual memory
> **Change** → untick *Automatically manage* → **Custom size** → Initial
> `16384` MB, Maximum `49152` MB → **Set** → OK → reboot.

Equivalent in PowerShell:

```powershell
$cs = Get-CimInstance Win32_ComputerSystem
Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile=$false}
$pf = Get-CimInstance Win32_PageFileSetting
Set-CimInstance -InputObject $pf -Property @{InitialSize=16384; MaximumSize=49152}
Restart-Computer
```

After rebooting, `Get-CimInstance Win32_PageFileSetting` must report the sizes,
and later `workerq resources --json` must show a `commit.ceiling_mib` of
roughly RAM + 48 GiB.

### 1.3 [W-admin] Defender exclusions

Snapshotting creates git worktrees with thousands of small files. Real-time
scanning each one is a large, silent tax on every submission.

```powershell
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.local\state\gpuq"
Add-MpPreference -ExclusionPath "C:\work"
Add-MpPreference -ExclusionProcess "python.exe"
Add-MpPreference -ExclusionProcess "pythonw.exe"
Add-MpPreference -ExclusionProcess "git.exe"
```

Adjust `C:\work` to wherever you put repos ([Appendix B](#appendix-b--directory-layout)).

### 1.4 [W] Windows Update

Do not disable updates; do stop them rebooting mid-job. Settings → Windows
Update → Advanced options → set **Active hours** as wide as allowed, and turn
**off** "Restart this device as soon as possible".

### 1.5 [W] Auto-logon (only if 0.5C failed)

Skip this if situation **C** worked — the machine can sit at the login screen.

If it did not, the dispatcher needs a logged-in console session. Use
Sysinternals **Autologon**, which stores the password as an encrypted LSA
secret rather than plaintext in the registry:

```powershell
winget install --id Microsoft.Sysinternals.Autologon -e
```

Run it, enter the account and password, enable. Then reboot and confirm the
machine reaches the desktop unattended. Locking the screen afterwards is fine —
the session survives; **an RDP session does not**, so never leave one connected.

---

## Stage 2 — toolchain

### 2.1 [W] Git and credentials

```powershell
winget install --id Git.Git -e
```

You said the worker already has your git credentials. Verify non-interactively —
this must return refs with no prompt, for a **private** repo:

```powershell
git ls-remote git@github.com:samsundar989/biohub.git | Select-Object -First 3
```

Use a genuinely **private** repo here. Checking against a public one proves
nothing — it succeeds without credentials, so a broken credential helper passes
the test and fails later during Stage 5.

If it prompts, fix it now. Stage 5 clones several repos and the dispatcher can
never answer a credential prompt.

### 2.2 [W] Python

worker-q requires **Python ≥ 3.11** (it uses `tomllib`).

```powershell
winget install --id Python.Python.3.11 -e
py -3.11 --version
```

### 2.3 [W] uv

```powershell
winget install --id astral-sh.uv -e
uv --version
```

`uv tool install` places shims in `%USERPROFILE%\.local\bin`. Confirm that
directory is on `PATH`.

### 2.4 [W] NVIDIA driver

The worker is on **591.86**, the primary on **596.49**. This does not need to
match — torch wheels require a driver at or above a minimum, not an exact
version — but aligning them removes a variable from every future "works here,
not there" question. Updating is optional; if you do, re-run 0.5A afterwards.

The architecture difference (`sm_86` vs `sm_120`) is *not* fixable and does not
need to be: standard cu12x wheels carry both.

---

## Stage 3 — install and configure worker-q

### 3.1 [W] Install

```powershell
git clone git@github.com:samsundar989/worker-q.git C:\work\worker-q
cd C:\work\worker-q
uv tool install --from . worker-q
workerq --version
```

Both `workerq` and the legacy `gpuq` shim are installed.

### 3.2 [W] Initialize

```powershell
workerq init
```

This creates `~/.config/gpuq/config.toml` and `~/.local/state/gpuq/`, and
starts a dispatcher.

### 3.3 [W] Write the worker's configuration

worker-q's defaults are tuned for a 64 GiB interactive workstation and are
actively wrong here. `reserve_ram_gb = 8.0` gives away half the machine;
`reserve_vram_gb = 4.0` gives away a third of a 12 GiB card.

Replace `~/.config/gpuq/config.toml` with:

```toml
# worker-q — DESKTOP-UNR95NB (RTX 3080 Ti 12 GiB, 16 GiB RAM, dedicated worker)

[core]
max_concurrent_jobs = 3
default_priority = "normal"
snapshot_mode = "git"
cleanup_successful_snapshots_after_days = 3
cleanup_failed_snapshots_after_days = 7
cancel_grace_seconds = 15

[gpu]
default_gpu_count = 1
free_memory_threshold_percent = 80
exclusive_by_default = true

[backend]
name = "local_dispatcher"
max_finished = 500
poll_interval_seconds = 0.25
daemon_heartbeat_stale_seconds = 30

[resources]
enforce = true
default_ram_gb = 3.0
default_vram_gb = 0.0
default_cpus = 1
reserve_ram_gb = 3.0
reserve_vram_gb = 1.0
reserve_cpus = 1
min_host_free_percent = 10
commit_headroom_percent = 5
max_commit_percent = 99

[scheduling]
backfill = true
backfill_max_skip = 8
backfill_head_wait_seconds = 900
background_priority = false
pressure_free_percent = 12
pressure_recover_percent = 20
pressure_samples = 3

[preemption]
enabled = true
require_opt_in = true
min_runtime_seconds = 60
grace_seconds = 30

[claude]
install_user_policy = false
```

Four choices worth understanding:

- **`reserve_ram_gb = 3.0`, `reserve_vram_gb = 1.0`.** A dedicated box needs
  headroom for the OS and worker-q itself, nothing more. This leaves ~13 GiB of
  usable RAM and ~11 GiB of usable VRAM, which is what the eligibility figures
  in [multi-node.md §0.1](multi-node.md#01-is-this-worth-building-measured-not-guessed)
  assume.
- **`background_priority = false`.** On the primary this is `true` to keep the
  desktop responsive. There is no desktop to protect here, so run at normal
  priority.
- **`exclusive_by_default = true` with a single GPU.** This means at most one
  GPU job at a time, with the other two slots free for CPU-only work — which is
  exactly the shape of the traffic the worker is meant to absorb (49% of jobs
  declare 0 VRAM). It also means the new whole-card VRAM delta measurement
  (commit `9b42cff`) will actually record peaks here, since it only measures
  when a job is exclusive, on one device, with nothing else on the card. **The
  worker will produce cleaner VRAM data than the primary does.**
- **`install_user_policy = false`.** No agents run on this machine; it must not
  write to `~/.claude/CLAUDE.md`.

> **Not included: `scheduling.backfill_max_hold_seconds`.** It was added by
> commit `16b2846`, which the worker's 1.2.0 build predates, so `config get`
> rejects it there. Unknown keys in a config *file* are ignored rather than
> fatal, so including it would have been harmless but inert. Add it when the
> worker is upgraded past that commit — see [7.1](#71-version-lockstep).

Apply and check:

```powershell
workerq restart
workerq doctor
workerq resources
```

`doctor` should exit 0. `resources` should show ~13 GiB usable RAM, ~11 GiB
usable VRAM, and a commit ceiling reflecting the fixed pagefile.

---

## Stage 4 — dispatcher lifecycle

The dispatcher **must run in the console session**, not as a service —
a Session 0 process has no CUDA access
([multi-node.md §2.1](multi-node.md#21-the-session-0-constraint-and-why-it-is-smaller-than-it-looks)).

### 4.1 [W] Task at logon

```powershell
$exe = "$env:USERPROFILE\.local\bin\workerq.exe"
Test-Path $exe        # must be True before continuing

schtasks /Create /TN "worker-q dispatcher" /TR "`"$exe`" restart" `
         /SC ONLOGON /DELAY 0000:30 /F
```

The 30-second delay lets Tailscale and networking settle first.

**Do not** tick "Run whether user is logged on or not", and do not pass
`/RU SYSTEM`. Either puts the task in Session 0 and silently removes CUDA from
every job it starts. `schtasks /SC ONLOGON` without `/RU` is correct.

### 4.2 [W] Verify it survives a reboot

```powershell
Restart-Computer
# after it comes back, from the primary:
```

```powershell
ssh desktop-unr95nb "%USERPROFILE%\.local\bin\workerq.exe doctor"
```

The dispatcher should already be running, without anyone touching the machine.

### 4.3 Restarting the dispatcher remotely

`ssh <node> workerq restart` **does not work**, and fails in the worst way: it
reports success, and the dispatcher dies as soon as the SSH session closes.
Windows OpenSSH tears down its session's processes on disconnect, detached or
not. For about thirty seconds afterwards `_dispatcher-status` still shows a pid
and a `heartbeat_age_seconds` that has not yet gone stale, so the machine looks
healthy while accepting nothing.

Go through the logon task instead. Its principal is `Interactive`, so the
dispatcher starts in the console session — which is also the only session where
CUDA works:

```powershell
ssh desktop-unr95nb schtasks /Run /TN "worker-q dispatcher"
```

Verify:

```powershell
ssh desktop-unr95nb "%USERPROFILE%\.local\bin\workerq.exe _dispatcher-status"
```

`daemon_running` must be `true`. A fresh heartbeat on its own proves nothing —
that is exactly what a just-killed dispatcher looks like.

`workerq restart` now warns when it detects `SSH_CONNECTION` rather than
appearing to succeed, `workerq node check` flags a node whose dispatcher is
down, and `workerq node list` carries a `DISP` column — reachable and
compatible is not the same as able to run anything.

---

---

## Stage 5 — per-project staging

**This is the part that repeats.** Do it for the projects that generate the
most small jobs first — from the primary's history, `kaggriculture` (132 jobs,
almost all 0 VRAM) is the highest-value first target, then `biohub`'s small
jobs, then `arc-whest`.

### 5.1 Clone

```powershell
git clone git@github.com:samsundar989/<project>.git C:\work\<project>
```

**The path does not need to match the primary.** Snapshots are materialised
under worker-q's own state directory, and `execution_cwd` is derived relative to
the repo root. Put repos wherever you like, consistently.

### 5.2 Build the environment

The snapshot ships **source, not environment** — `.venv` is gitignored, so it is
never in a snapshot. The worker needs its own, built here, with wheels for its
own driver and `sm_86`:

```powershell
cd C:\work\<project>
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the project uses torch, install the CUDA build explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

### 5.3 Stage the data

Check what the project declares:

```powershell
Get-Content C:\work\<project>\.gpuq.toml
```

`[snapshot] passthrough` lists the paths linked into every snapshot. The rule
that decides your work here:

| Declaration | What the worker needs |
| --- | --- |
| **Relative** (`data`, `.venv`, `checkpoints`) | just that subdirectory populated **inside the worker's clone**. Nothing else. |
| **Absolute** (`D:\datasets\imagenet`) | that **exact path** must exist on the worker, or the job is permanently ineligible |

Prefer relative. If a project uses absolute paths, either mirror the path
exactly or accept that its jobs stay on the primary.

Copy the datasets across however suits their size — for anything large, do not
pull it over the Wi-Fi link from the primary; fetch it from its original source
on the worker's own connection.

Verify every declared path resolves:

```powershell
cd C:\work\<project>
@("data",".venv","checkpoints") | ForEach-Object {
  "{0,-16} {1}" -f $_, (Test-Path $_)
}
```

### 5.4 Smoke-test the project

Run the smallest real job **by hand**, in the console session, before trusting
it to the queue:

```powershell
cd C:\work\<project>
.\.venv\Scripts\python.exe <smallest real entrypoint>
```

Then through the queue, locally on the worker:

```powershell
workerq submit --project <project> --ram 4 --vram 6 --passthrough .venv `
  --describe "staging smoke test" -- `
  .\.venv\Scripts\python.exe <smallest real entrypoint>
workerq status
workerq logs <id> --follow
```

> **`--passthrough .venv` is not optional.** Snapshots contain tracked and
> untracked-but-not-ignored files; `.venv` is gitignored, so it is *not* in the
> snapshot and `.\.venv\Scripts\python.exe` does not exist inside it. Without
> the flag the job dies before it runs. Of the staged projects only `biohub`
> declares this in a `.gpuq.toml`; `kaggriculture` and `arc-whest` need the flag
> on every submission until they get one. **The durable fix is a `.gpuq.toml`
> in each project** — it is committed, so it travels with the snapshot and both
> machines then agree without anyone remembering a flag.

### 5.5 biohub — what to stage, and what must not be staged

Measured on the primary, 2026-09-09, with transfer time at the 6.1 MB/s the
link actually delivers:

| Path | Size | Files | Copy across? |
| --- | ---: | ---: | --- |
| `.venv` | 5.21 GB | 33 227 | **Never.** Built on the worker; Windows venvs hardcode paths and the wheels differ. Already done. |
| `data/train` | 79.82 GB | 24 477 | No — 3.7 hours over this link. From its original source, on the worker's own connection. |
| `data/test` | 1.78 GB | 408 | Same. |
| `data/dense_labels` | 0.03 GB | 199 | Yes if not in the download — trivial. |
| `data/sample_submission.csv` | ~0 | 1 | Yes — trivial. |
| `models/public` | 0.74 GB | 199 | **Yes.** ~2 min. Public checkpoints, but copying beats re-deriving provenance. |
| `models/clean` | 0.10 GB | 11 | **Yes, and this one matters.** Fold-legal weights produced by your own training — they cannot be re-downloaded from anywhere. |
| `runs/cache` | 0.64 GB | 2 923 | **No.** It is a cache; it regenerates. A cache built on a 5090 is also a correctness risk on a 3080 Ti. Create the directory empty. |
| `runs/records` | ~0 | 121 | **No — see below.** |
| `artifacts/manifests/kaggle_files.json` | 2.4 MB | 1 | Yes — trivial. |

So: copy `models/public`, `models/clean`, `artifacts/manifests/kaggle_files.json`
and the two small `data/` items — about **0.9 GB, roughly 2.5 minutes**. Create
`runs/cache` and `runs/records` as empty directories. Everything else is already
handled or must not cross the link.

**But do not dispatch biohub to the worker yet.** Two of its declared paths are
things jobs *write*, not read, and that breaks an assumption the placement
design has not yet addressed:

- `runs/records` is "experiment records written by runs". Two machines
  appending to their own local copies produce two divergent histories that
  nothing reconciles.
- The `.gpuq.toml` header instructs jobs to write submissions to an **absolute**
  path, `C:/Users/samsu/Documents/biohub/outputs/submission.csv`. Repos live
  under `C:\Users\samsu\Documents\<project>` on *both* machines, so that path
  resolves on the worker too — to the worker's disk. A dispatched job would run
  correctly, report `SUCCEEDED`, and put the submission on a machine you are not
  looking at.

That second one is worse than a missing dataset, because a missing dataset
fails loudly and this succeeds wrongly. It is written up as
[multi-node.md §8.6](multi-node.md#86-write-targets-diverge-silently), and the
fix is for `.gpuq.toml` to distinguish read inputs from write targets. Until
then biohub is a **primary-only** project — which costs little, since its large
training runs need 17+ GiB of VRAM and could never fit a 12 GiB card anyway.

Staging it now is still worth doing: it makes the worker ready for biohub's
*small* jobs the moment output handling exists.

---

## Stage 6 — verify the node standalone

Everything here works today, with no new code. This is the acceptance test for
the worker as a machine.

```powershell
workerq doctor                    # exit 0
workerq resources --verify        # capacity sane, ~13 GiB RAM / ~11 GiB VRAM usable
workerq status                    # dispatcher running
```

Then prove the four behaviours the design depends on:

1. **A job runs and logs.** Submit the staging smoke test; `workerq logs -f`
   streams it; it reaches `SUCCEEDED` with the right exit code.
2. **Admission actually refuses.** Submit something declaring more than the box
   has (`--ram 40`) and confirm it is rejected immediately rather than queued
   forever, and that the message names the constraint.
3. **Concurrency behaves.** Submit three small CPU-only jobs and one GPU job;
   confirm the GPU job runs exclusively while CPU jobs overlap.
4. **It survives a reboot mid-job.** Start a long job, reboot, and confirm the
   dispatcher restarts and reconciles the job to a terminal state rather than
   leaving it `RUNNING` forever.

When all four pass, the worker is done. What remains is entirely on the primary.

---

## Stage 7 — ongoing operations

### 7.1 Version lockstep

The wire format is one worker-q's `--json` output parsed by another, so a
version mismatch is a parse failure at the worst moment. **Upgrade both
machines together.**

> **The two machines are already skewed, and `--version` does not show it.**
> Both report `workerq 1.2.0`. The worker is at `f887068`; the primary carries
> `16b2846` and `9b42cff` on top, which added a config key, a database column
> and a different way of measuring RAM. `__version__` was not bumped because
> these were not releases.
>
> ```text
> PRIMARY  config get scheduling.backfill_max_hold_seconds -> 1800
> WORKER   config get scheduling.backfill_max_hold_seconds -> error: unknown key
> ```
>
> **Resolved 2026-09-09.** The branch was merged to `main` and the version
> bumped to **1.3.0**, so `--version` is meaningful again for this gap. Bring
> the worker up with the upgrade steps below; after it, both machines must
> report `workerq 1.3.0`. That fixes *this* skew but not the general problem —
> the next unbumped commit reopens it, which is why the durable answer is a
> protocol version. See
> [multi-node.md §8.2](multi-node.md#82-version-skew--and-why---version-cannot-detect-it)
> for why the fix is a protocol version and a build commit rather than a
> version string.

Upgrading over a live install has a known failure mode, and on the worker it is
worse than on the primary. There, `uv tool install` rolls back silently when
something holds the shim. **Here it leaves a broken install** - the running
dispatcher holds `Scripts/`, `uv` cannot remove it, and you get:

```text
error: failed to remove directory ...\uv\tools\worker-q\Scripts: Access is denied
error: uv trampoline failed to canonicalize script path
```

`Scripts/` is then half-written and `workerq.exe` no longer runs, so the node
goes unreachable and you cannot even use it to stop its own daemon. Recover by
killing the daemon directly, reinstalling, and restarting via
[4.3](#43-restarting-the-dispatcher-remotely):

```powershell
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'"   # find -m workerq _daemon
taskkill /PID <pid> /F
```

That is safe on this machine precisely because it is dedicated. On the primary
a live `workerq.exe` is usually somebody's dashboard, and killing it is the
wrong move.

So: stop the dispatcher first, and close any `workerq top`:

```powershell
workerq _stop-daemon
cd C:\work\worker-q; git pull
uv tool install --from . --force worker-q
workerq restart
workerq doctor
```

### 7.2 Retention and disk

The worker's config keeps snapshots for 3/7 days rather than the primary's 7/14
— it has less disk, and the primary holds the authoritative provenance.

```powershell
workerq cleanup --dry-run
workerq cleanup
```

Watch free disk: pagefile (48 GiB) + snapshots + staged datasets on a machine
that also has to hold several venvs.

### 7.3 When it is unreachable

A dropped link does **not** mean a dropped job. The worker's dispatcher owns
its jobs in its own Job Objects; they keep running. Never resubmit — see
[multi-node.md §7](multi-node.md#7-failure-semantics).

---

## Appendix A — what is deliberately *not* installed

| Not installed | Why |
| --- | --- |
| `workerq claude-policy install` | No agents run here. It must not write `~/.claude/CLAUDE.md`. |
| `workerq claude-safe-launcher` | Same. |
| The MCP server (`mcp` extra) | Nothing on this machine speaks MCP. |
| A `[gaming]` reserve | Dedicated worker; nothing to reserve against. |
| worker-q as a Windows **service** | Session 0 has no CUDA. This is the one hard rule. |
| An RDP session left connected | RDP swaps in a dummy display driver and CUDA disappears. |

## Appendix B — directory layout

```text
C:\work\worker-q\                     the tool's own source (for upgrades)
C:\work\<project>\                    one clone per staged project
    .venv\                            built here, never copied from the primary
    data\                             relative passthrough targets, populated here
%USERPROFILE%\.local\bin\workerq.exe  uv tool shim
%USERPROFILE%\.config\gpuq\config.toml
%USERPROFILE%\.local\state\gpuq\
    gpuq.sqlite3                      the worker's own job metadata
    backend\queue.sqlite3
    logs\job-NNNNNN.log
    snapshots\<id>\job-<id>\          materialised from the primary's bundle
    run\dispatcher.lock
```

## Appendix C — troubleshooting

| Symptom | Cause |
| --- | --- |
| `ssh` still asks for a password | BOM in `administrators_authorized_keys`, or its ACL grants more than SYSTEM + Administrators. Rewrite with `-Encoding ascii`, re-run `icacls`. |
| `torch.cuda.is_available()` is `False` over SSH but `True` at the keyboard | Expected in situation C without auto-logon. Not a problem for jobs — the dispatcher runs in the console session. |
| CUDA vanished after connecting with RDP | Disconnect RDP and use SSH. The dummy display driver deactivates the NVIDIA one. |
| Dispatcher not running after reboot | The logon task was created with `/RU SYSTEM` or "run whether logged on or not" — it is in Session 0. Recreate per [4.1](#41-w-task-at-logon). |
| Jobs queue forever, `status` blames RAM or commit | Usually over-declaration, not a real shortage. `workerq requests <id> --suggest`. On this box also check the pagefile is fixed, not system-managed. |
| `tailscale ping` says `via DERP` | Peers are relayed. Check both are on the same LAN and no firewall blocks direct UDP. |
| `uv tool install` appears to succeed but the version does not change | Something holds the shim — a running dispatcher or `workerq top`. Stop them, reinstall. |
