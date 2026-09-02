#!/usr/bin/env bash
#
# GPUQ bootstrap - safe to re-run.
#
# Detects the platform and toolchain, installs worker-q, initialises the queue,
# installs the Claude policy, runs the health checks and finishes with a
# non-destructive queue smoke test.
#
# Nothing here is destructive: no existing queue, repository or user
# instruction file is ever overwritten without a backup.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILURES=0
WARNINGS=0

say()   { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()    { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn()  { printf '  \033[33mwarn\033[0m  %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
fail()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }
info()  { printf '        %s\n' "$*"; }

# ---------------------------------------------------------------------------
say "1. Platform"
# ---------------------------------------------------------------------------
UNAME="$(uname -s 2>/dev/null || echo unknown)"
case "$UNAME" in
  Linux*)
    if grep -qi microsoft /proc/version 2>/dev/null; then
      ok "WSL2 ($UNAME)"
    else
      ok "Linux ($UNAME)"
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*) ok "Windows ($UNAME)" ;;
  Darwin*) warn "macOS - no NVIDIA GPU gating available" ;;
  *) warn "unrecognised platform: $UNAME" ;;
esac

# ---------------------------------------------------------------------------
say "2. Python >= 3.11"
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in python3 python py; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done
if [ -z "$PYTHON" ]; then
  fail "no Python >= 3.11 found on PATH"
  info "install Python 3.11+ and re-run this script"
  exit 2
fi
ok "$("$PYTHON" --version 2>&1) ($(command -v "$PYTHON"))"

# ---------------------------------------------------------------------------
say "3. git"
# ---------------------------------------------------------------------------
if command -v git >/dev/null 2>&1; then
  ok "$(git --version)"
else
  fail "git not found - source snapshots need it"
fi

# ---------------------------------------------------------------------------
say "4. NVIDIA runtime"
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    ok "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  else
    warn "nvidia-smi is present but the driver did not respond"
  fi
else
  warn "nvidia-smi not found - jobs will still be serialised, GPU gating is off"
fi

# ---------------------------------------------------------------------------
say "5. Install strategy"
# ---------------------------------------------------------------------------
INSTALLER=""
if command -v uv >/dev/null 2>&1; then
  INSTALLER="uv"; ok "uv $(uv --version 2>&1 | awk '{print $2}')"
elif command -v pipx >/dev/null 2>&1; then
  INSTALLER="pipx"; ok "pipx"
else
  INSTALLER="venv"; warn "neither uv nor pipx found - falling back to a private venv"
fi

# ---------------------------------------------------------------------------
say "6. Execution backend"
# ---------------------------------------------------------------------------
# On Linux the GPU-aware Task Spooler fork is the intended backend. Everywhere
# else (and on Linux until it is built) worker-q ships its own dispatcher, which
# implements the same SchedulerBackend contract.
if command -v ts >/dev/null 2>&1 && ts -h 2>&1 | grep -q -- '--set_gpu_free_perc'; then
  ok "GPU-aware Task Spooler detected: $(ts -V 2>&1 | head -1)"
  info "worker-q on this machine uses its built-in dispatcher; the backend"
  info "abstraction in src/workerq/backends/base.py is where TS would slot in."
else
  ok "using the built-in worker-q dispatcher (no external queue daemon required)"
fi

# ---------------------------------------------------------------------------
say "7. Install worker-q"
# ---------------------------------------------------------------------------
GPUQ_BIN=""
case "$INSTALLER" in
  uv)
    if uv tool install --force --from "$REPO_ROOT" worker-q >/dev/null 2>&1; then
      ok "installed with uv tool install"
    else
      fail "uv tool install failed"
      uv tool install --force --from "$REPO_ROOT" worker-q 2>&1 | tail -5
    fi
    ;;
  pipx)
    pipx install --force "$REPO_ROOT" >/dev/null 2>&1 && ok "installed with pipx" || fail "pipx install failed"
    ;;
  venv)
    VENV="$HOME/.local/share/worker-q/venv"
    "$PYTHON" -m venv "$VENV" >/dev/null 2>&1
    "$VENV/bin/pip" install -q --upgrade pip >/dev/null 2>&1 || true
    if "$VENV/bin/pip" install -q "$REPO_ROOT" >/dev/null 2>&1; then
      ok "installed into $VENV"
      GPUQ_BIN="$VENV/bin/workerq"
    else
      fail "venv install failed"
    fi
    ;;
esac

if [ -z "$GPUQ_BIN" ]; then
  if command -v workerq >/dev/null 2>&1; then
    GPUQ_BIN="$(command -v workerq)"
  else
    for candidate in "$HOME/.local/bin/workerq" "$HOME/.local/bin/workerq.exe"; do
      [ -x "$candidate" ] && GPUQ_BIN="$candidate" && break
    done
  fi
fi

if [ -z "$GPUQ_BIN" ]; then
  fail "workerq is installed but not on PATH"
  info "add \$HOME/.local/bin to PATH, then re-run"
  exit 2
fi
ok "workerq: $GPUQ_BIN ($("$GPUQ_BIN" --version 2>&1))"

# ---------------------------------------------------------------------------
say "8. Initialise the queue"
# ---------------------------------------------------------------------------
"$GPUQ_BIN" init || fail "workerq init failed"

# ---------------------------------------------------------------------------
say "9. Install the Claude policy"
# ---------------------------------------------------------------------------
"$GPUQ_BIN" claude-policy install || warn "policy install failed"

# ---------------------------------------------------------------------------
say "10. Health checks"
# ---------------------------------------------------------------------------
"$GPUQ_BIN" doctor
DOCTOR=$?
case "$DOCTOR" in
  0) ok "doctor: HEALTHY" ;;
  1) warn "doctor: DEGRADED but usable" ;;
  *) fail "doctor: BROKEN - do not submit jobs yet" ;;
esac

# ---------------------------------------------------------------------------
say "11. Queue smoke test (non-destructive, isolated profile)"
# ---------------------------------------------------------------------------
if [ -x "$REPO_ROOT/scripts/smoke_test.sh" ]; then
  if bash "$REPO_ROOT/scripts/smoke_test.sh" >/tmp/gpuq-bootstrap-smoke.log 2>&1; then
    ok "smoke test passed"
  else
    warn "smoke test reported problems - see /tmp/gpuq-bootstrap-smoke.log"
    tail -20 /tmp/gpuq-bootstrap-smoke.log
  fi
else
  warn "scripts/smoke_test.sh not found"
fi

# ---------------------------------------------------------------------------
say "Done"
# ---------------------------------------------------------------------------
printf '\n%s\n\n' "The five commands you need:"
cat <<'USAGE'
  workerq doctor                                        check the machine is healthy
  workerq submit --project NAME -- python train.py      queue a job, return at once
  workerq status                                        what is running / next
  workerq logs <id> --follow                            stream a job's output
  workerq cancel <id>                                   stop a queued or running job
USAGE

if [ "$FAILURES" -gt 0 ]; then
  printf '\n\033[31m%s failure(s), %s warning(s).\033[0m\n' "$FAILURES" "$WARNINGS"
  exit 2
elif [ "$WARNINGS" -gt 0 ]; then
  printf '\n\033[33m%s warning(s) - usable, see above.\033[0m\n' "$WARNINGS"
  exit 1
fi
printf '\n\033[32mAll checks passed.\033[0m\n'
exit 0
