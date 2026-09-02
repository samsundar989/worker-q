#!/usr/bin/env bash
#
# GPUQ acceptance smoke test (spec section 26).
#
# Runs entirely inside an isolated GPUQ profile (GPUQ_PROFILE=smoke) with its
# own state directory, database and dispatcher, so the production queue is
# never touched or disturbed.
#
# Verifies:
#   1. three queued jobs execute without their intervals overlapping
#   2. logs are captured and readable
#   3. a queued job can be cancelled and never runs
#   4. a running job can be cancelled
#   5. a queued job runs its submission-time source snapshot, not later edits
#   6. jobs survive the submitting shell exiting
#   7. status/show/doctor stay coherent

set -uo pipefail

PASS=0
FAIL=0

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL + 1)); }
step() { printf '\n\033[1m--- %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Isolated profile
# ---------------------------------------------------------------------------
export GPUQ_PROFILE="smoke"
WORK="$(mktemp -d 2>/dev/null || mktemp -d -t gpuq-smoke)"
export GPUQ_STATE_DIR="$WORK/state"
export GPUQ_CONFIG_FILE="$WORK/config.toml"

GPUQ="${GPUQ_BIN:-}"
if [ -z "$GPUQ" ]; then
  if command -v gpuq >/dev/null 2>&1; then
    GPUQ="$(command -v gpuq)"
  else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    for candidate in "$REPO_ROOT/.venv/Scripts/gpuq.exe" "$REPO_ROOT/.venv/bin/gpuq" \
                     "$HOME/.local/bin/gpuq" "$HOME/.local/bin/gpuq.exe"; do
      [ -x "$candidate" ] && GPUQ="$candidate" && break
    done
  fi
fi
if [ -z "$GPUQ" ]; then
  echo "gpuq not found; set GPUQ_BIN or install it first" >&2
  exit 2
fi

cleanup() {
  "$GPUQ" _stop-daemon >/dev/null 2>&1 || true
  sleep 1
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

printf '\033[1mGPUQ smoke test\033[0m\n'
printf '  binary : %s\n' "$GPUQ"
printf '  profile: %s (isolated - the production queue is untouched)\n' "$GPUQ_PROFILE"
printf '  state  : %s\n' "$GPUQ_STATE_DIR"

PY="$("$GPUQ" version --json | python -c 'import json,sys; print(json.load(sys.stdin)["executable"])' 2>/dev/null)"
[ -z "$PY" ] && PY="python"

jq_get() { python -c "import json,sys; d=json.load(sys.stdin); print(d$1)"; }

wait_state() { # job_id, expected, timeout
  local job_id="$1" want="$2" limit="${3:-120}" waited=0
  while [ "$waited" -lt "$limit" ]; do
    local state
    state="$("$GPUQ" show "$job_id" --json 2>/dev/null | jq_get "['state']" 2>/dev/null)"
    [ "$state" = "$want" ] && return 0
    case "$state" in
      SUCCEEDED|FAILED|CANCELLED|LOST) [ "$state" != "$want" ] && return 1 ;;
    esac
    sleep 1; waited=$((waited + 1))
  done
  return 1
}

# ---------------------------------------------------------------------------
step "0. Initialise the isolated queue"
# ---------------------------------------------------------------------------
if "$GPUQ" init >/dev/null 2>&1; then pass "gpuq init"; else bad "gpuq init"; exit 2; fi
"$GPUQ" gpu-threshold 0 >/dev/null 2>&1   # never gate on GPU memory in the smoke test
# Several checks below prove *serialisation* - a blocker job holds the queue
# while the source tree is edited underneath it. That is a property of one
# slot, not of the shipped default, which is now a ceiling. Pin it.
"$GPUQ" concurrency 1 >/dev/null 2>&1

# ---------------------------------------------------------------------------
step "1. Build a temporary git repository"
# ---------------------------------------------------------------------------
REPO="$WORK/repo"
mkdir -p "$REPO"
(
  cd "$REPO"
  git init -q -b main .
  git config user.email "smoke@example.invalid"
  git config user.name "gpuq smoke"
  git config core.autocrlf false
  cat > interval.py <<'PYEOF'
import json, os, sys, time
out, name, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
start = time.time(); time.sleep(secs); end = time.time()
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, name + ".json"), "w") as fh:
    json.dump({"name": name, "start": start, "end": end}, fh)
print("interval", name, "ok", flush=True)
PYEOF
  echo 'VALUE = "A"' > value.py
  cat > show.py <<'PYEOF'
import value
print("RESULT:", value.VALUE, flush=True)
PYEOF
  git add -A >/dev/null 2>&1
  git commit -qm "smoke fixtures" >/dev/null 2>&1
) && pass "temporary repo created" || bad "could not create the temporary repo"

RESULTS="$WORK/intervals"
mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
step "2. Three queued jobs must not overlap"
# ---------------------------------------------------------------------------
IDS=""
for name in first second third; do
  id="$(cd "$REPO" && "$GPUQ" submit --project smoke --gpus 0 --json -- \
        "$PY" interval.py "$RESULTS" "$name" 2 | jq_get "['job_id']")"
  IDS="$IDS $id"
done
pass "submitted three jobs:$IDS"

for id in $IDS; do
  wait_state "$id" SUCCEEDED 240 || bad "job #$id did not succeed"
done

OVERLAP="$(python - "$RESULTS" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
runs = []
for name in ("first", "second", "third"):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p):
        print("MISSING " + name); raise SystemExit
    runs.append(json.load(open(p)))
runs.sort(key=lambda r: r["start"])
for a, b in zip(runs, runs[1:]):
    if a["end"] > b["start"] + 1e-6:
        print("OVERLAP %s/%s" % (a["name"], b["name"])); raise SystemExit
print("OK")
PYEOF
)"
[ "$OVERLAP" = "OK" ] && pass "execution intervals never overlapped" \
                      || bad "exclusivity broken: $OVERLAP"

# ---------------------------------------------------------------------------
step "3. Logs are captured"
# ---------------------------------------------------------------------------
FIRST_ID="$(echo $IDS | awk '{print $1}')"
if "$GPUQ" logs "$FIRST_ID" 2>/dev/null | grep -q "interval first ok"; then
  pass "job output is readable via gpuq logs"
else
  bad "job output missing from gpuq logs"
fi
"$GPUQ" logs "$FIRST_ID" --tail 5 >/dev/null 2>&1 && pass "gpuq logs --tail works" \
                                                  || bad "gpuq logs --tail failed"

# ---------------------------------------------------------------------------
step "4. Cancelling a queued job prevents execution"
# ---------------------------------------------------------------------------
BLOCKER="$(cd "$REPO" && "$GPUQ" submit --project smoke --gpus 0 --json -- \
           "$PY" -c "import time; time.sleep(6)" | jq_get "['job_id']")"
wait_state "$BLOCKER" RUNNING 60 || bad "blocker never started"

MARKER="$WORK/should_not_exist.txt"
VICTIM="$(cd "$REPO" && "$GPUQ" submit --project smoke --gpus 0 --json -- \
          "$PY" -c "open(r'$MARKER','w').write('ran')" | jq_get "['job_id']")"
CANCEL_STATE="$("$GPUQ" cancel "$VICTIM" --json | jq_get "['state']")"
[ "$CANCEL_STATE" = "CANCELLED" ] && pass "queued job cancelled" \
                                  || bad "cancel returned $CANCEL_STATE"

# ---------------------------------------------------------------------------
step "5. Cancelling a running job stops it"
# ---------------------------------------------------------------------------
CANCEL_RUNNING="$("$GPUQ" cancel "$BLOCKER" --force --json | jq_get "['state']")"
[ "$CANCEL_RUNNING" = "CANCELLED" ] && pass "running job terminated" \
                                    || bad "running cancel returned $CANCEL_RUNNING"

sleep 3
[ -f "$MARKER" ] && bad "a cancelled job executed anyway" \
                 || pass "cancelled queued job never executed"

# ---------------------------------------------------------------------------
step "6. Snapshot immutability"
# ---------------------------------------------------------------------------
BLOCKER2="$(cd "$REPO" && "$GPUQ" submit --project smoke --gpus 0 --json -- \
            "$PY" -c "import time; time.sleep(5)" | jq_get "['job_id']")"
wait_state "$BLOCKER2" RUNNING 60 || bad "second blocker never started"

SNAP_JOB="$(cd "$REPO" && "$GPUQ" submit --project smoke --gpus 0 --json -- \
            "$PY" show.py | jq_get "['job_id']")"
echo 'VALUE = "B"' > "$REPO/value.py"    # edit the live tree while queued

if wait_state "$SNAP_JOB" SUCCEEDED 240; then
  if "$GPUQ" logs "$SNAP_JOB" | grep -q "RESULT: A"; then
    pass "queued job ran its submission-time snapshot, not the later edit"
  else
    bad "job ran edited source: $("$GPUQ" logs "$SNAP_JOB" | grep RESULT)"
  fi
else
  bad "snapshot job did not succeed"
fi
echo 'VALUE = "A"' > "$REPO/value.py"

# ---------------------------------------------------------------------------
step "7. Jobs survive the submitting shell exiting"
# ---------------------------------------------------------------------------
SURVIVE_MARKER="$WORK/survived.txt"
# The marker path is passed as its own argv element rather than embedded in the
# Python source: only then does the shell translate it into a native path the
# job's interpreter can actually open.
cat > "$REPO/survive.py" <<'PYEOF'
import sys, time
time.sleep(2)
open(sys.argv[1], "w").write("yes")
PYEOF
DETACHED_ID="$(
  bash -c "cd '$REPO' && '$GPUQ' submit --project smoke --gpus 0 --json -- \
           '$PY' survive.py '$SURVIVE_MARKER'" \
  | jq_get "['job_id']"
)"
# the submitting shell has already exited at this point
if wait_state "$DETACHED_ID" SUCCEEDED 240 && [ -f "$SURVIVE_MARKER" ]; then
  pass "job outlived the shell that submitted it"
else
  bad "job did not survive its submitting shell"
fi

# ---------------------------------------------------------------------------
step "8. status / show / doctor stay coherent"
# ---------------------------------------------------------------------------
"$GPUQ" status --json >/dev/null 2>&1 && pass "gpuq status --json" || bad "gpuq status --json"
"$GPUQ" show "$FIRST_ID" --json >/dev/null 2>&1 && pass "gpuq show --json" || bad "gpuq show --json"
"$GPUQ" reconcile --json >/dev/null 2>&1 && pass "gpuq reconcile" || bad "gpuq reconcile"

"$GPUQ" doctor >/dev/null 2>&1
case $? in
  0) pass "gpuq doctor: HEALTHY" ;;
  1) pass "gpuq doctor: DEGRADED (usable)" ;;
  *) bad  "gpuq doctor: BROKEN" ;;
esac

# ---------------------------------------------------------------------------
printf '\n\033[1m========================================\033[0m\n'
printf '  passed: %s\n  failed: %s\n' "$PASS" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  printf '\033[32m  SMOKE TEST PASSED\033[0m\n\n'
  exit 0
fi
printf '\033[31m  SMOKE TEST FAILED\033[0m\n\n'
exit 1
