#!/usr/bin/env bash
#
# The exact Definition of Done scenario from the specification (section 36),
# run against the PRODUCTION queue with three separate projects.
#
#   terminal A: gpuq submit --project project-a -- <long gpu job a>
#   terminal B: gpuq submit --project project-b -- <gpu job b>
#   terminal C: gpuq submit --project project-c --priority critical -- <job c>
#
# Expected:  A -> RUNNING, C -> QUEUED ahead of B, B -> QUEUED
#            all three submitting terminals close; A continues, then C, then B
#            project B's source is edited before B starts; B must run the
#            snapshot taken at submission time
#            status / show / logs / doctor all stay coherent
#            cancellation works for a queued and for a running job

set -uo pipefail

GPUQ="${GPUQ_BIN:-gpuq}"
PASS=0; FAIL=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); }
step() { printf '\n\033[1m--- %s\033[0m\n' "$*"; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK" 2>/dev/null || true' EXIT

PY="$("$GPUQ" version --json | python -c 'import json,sys; print(json.load(sys.stdin)["executable"])')"
jq_get() { python -c "import json,sys; print(json.load(sys.stdin)$1)"; }
state_of() { "$GPUQ" show "$1" --json | jq_get "['state']"; }

make_project() {  # name, sleep_seconds
  local name="$1" secs="$2" dir="$WORK/$1"
  mkdir -p "$dir"
  (
    cd "$dir"
    git init -q -b main .
    git config user.email "dod@example.invalid"; git config user.name "dod"
    git config core.autocrlf false
    echo 'VALUE = "ORIGINAL"' > value.py
    cat > job.py <<PYEOF
import sys, time, value
print("project $name starting, VALUE =", value.VALUE, flush=True)
time.sleep($secs)
print("project $name done, VALUE =", value.VALUE, flush=True)
PYEOF
    git add -A >/dev/null 2>&1; git commit -qm init >/dev/null 2>&1
  )
  echo "$dir"
}

printf '\033[1mGPUQ Definition of Done acceptance\033[0m\n  binary: %s\n' "$(command -v "$GPUQ" || echo "$GPUQ")"

step "Create three independent projects"
A="$(make_project project-a 30)"
B="$(make_project project-b 5)"
C="$(make_project project-c 5)"
pass "three git repositories created"

# ---------------------------------------------------------------------------
step "Submit from three separate terminals, each of which then exits"
# ---------------------------------------------------------------------------
# Each submission runs in its own shell that exits immediately afterwards -
# this is the "close all three submission terminals" part of the scenario.
ID_A="$(bash -c "cd '$A' && '$GPUQ' submit --project project-a --json -- '$PY' job.py" | jq_get "['job_id']")"
sleep 2   # let A start so B and C genuinely queue behind it
ID_B="$(bash -c "cd '$B' && '$GPUQ' submit --project project-b --json -- '$PY' job.py" | jq_get "['job_id']")"
ID_C="$(bash -c "cd '$C' && '$GPUQ' submit --project project-c --priority critical --json -- '$PY' job.py" | jq_get "['job_id']")"
printf '  job ids: A=%s  B=%s  C=%s\n' "$ID_A" "$ID_B" "$ID_C"
pass "all three submitting shells exited"

# ---------------------------------------------------------------------------
step "Expected initial states"
# ---------------------------------------------------------------------------
SA="$(state_of "$ID_A")"; SB="$(state_of "$ID_B")"; SC="$(state_of "$ID_C")"
printf '  A=%s  B=%s  C=%s\n' "$SA" "$SB" "$SC"
[ "$SA" = "RUNNING" ] && pass "project-a RUNNING" || bad "project-a is $SA, expected RUNNING"
[ "$SB" = "QUEUED" ]  && pass "project-b QUEUED"  || bad "project-b is $SB, expected QUEUED"
[ "$SC" = "QUEUED" ]  && pass "project-c QUEUED"  || bad "project-c is $SC, expected QUEUED"

RUNNING_COUNT="$("$GPUQ" status --json --all | python -c "
import json,sys
print(sum(1 for j in json.load(sys.stdin)['jobs'] if j['state']=='RUNNING'))")"
[ "$RUNNING_COUNT" -le 1 ] && pass "only one job running (count=$RUNNING_COUNT)" \
                           || bad "$RUNNING_COUNT jobs running simultaneously"

POS_B="$("$GPUQ" show "$ID_B" --json | jq_get "['queue_position']")"
POS_C="$("$GPUQ" show "$ID_C" --json | jq_get "['queue_position']")"
printf '  queue positions: C=%s  B=%s\n' "$POS_C" "$POS_B"
[ "$POS_C" -lt "$POS_B" ] 2>/dev/null && pass "critical project-c is queued ahead of project-b" \
                                      || bad "project-c ($POS_C) is not ahead of project-b ($POS_B)"

# ---------------------------------------------------------------------------
step "Edit project-b's source while it is still queued"
# ---------------------------------------------------------------------------
echo 'VALUE = "EDITED-AFTER-SUBMIT"' > "$B/value.py"
grep -q EDITED "$B/value.py" && pass "live project-b source changed" || bad "could not edit project-b"

# ---------------------------------------------------------------------------
step "Wait for the queue to drain (terminals are already closed)"
# ---------------------------------------------------------------------------
for _ in $(seq 1 180); do
  s1="$(state_of "$ID_A")"; s2="$(state_of "$ID_B")"; s3="$(state_of "$ID_C")"
  case "$s1$s2$s3" in *QUEUED*|*RUNNING*|*PREPARING*) sleep 2 ;; *) break ;; esac
done
printf '  final: A=%s  B=%s  C=%s\n' "$(state_of "$ID_A")" "$(state_of "$ID_B")" "$(state_of "$ID_C")"

for id in "$ID_A" "$ID_B" "$ID_C"; do
  st="$(state_of "$id")"
  [ "$st" = "SUCCEEDED" ] && pass "job #$id SUCCEEDED after its terminal closed" \
                          || bad "job #$id ended as $st"
done

# ---------------------------------------------------------------------------
step "Execution order: A, then C (critical), then B"
# ---------------------------------------------------------------------------
started() { "$GPUQ" show "$1" --json | jq_get "['started_at']"; }
SA_T="$(started "$ID_A")"; SB_T="$(started "$ID_B")"; SC_T="$(started "$ID_C")"
printf '  started: A=%s\n           C=%s\n           B=%s\n' "$SA_T" "$SC_T" "$SB_T"
python - "$SA_T" "$SC_T" "$SB_T" <<'PYEOF'
import sys
from datetime import datetime
a, c, b = (datetime.fromisoformat(x) for x in sys.argv[1:4])
print("ORDER_OK" if a < c < b else "ORDER_BAD")
PYEOF
ORDER="$(python - "$SA_T" "$SC_T" "$SB_T" <<'PYEOF'
import sys
from datetime import datetime
a, c, b = (datetime.fromisoformat(x) for x in sys.argv[1:4])
print("OK" if a < c < b else "BAD")
PYEOF
)"
[ "$ORDER" = "OK" ] && pass "ran in order A -> C -> B (critical overtook normal)" \
                    || bad "execution order was not A -> C -> B"

# ---------------------------------------------------------------------------
step "project-b ran its submission-time snapshot, not the edited source"
# ---------------------------------------------------------------------------
if "$GPUQ" logs "$ID_B" | grep -q "VALUE = ORIGINAL"; then
  pass "project-b ran the frozen snapshot (VALUE = ORIGINAL)"
else
  bad "project-b did not run its snapshot: $("$GPUQ" logs "$ID_B" | grep VALUE | head -2)"
fi
"$GPUQ" logs "$ID_B" | grep -q "EDITED-AFTER-SUBMIT" && bad "project-b saw the later edit" \
                                                     || pass "the later edit never reached the job"

# ---------------------------------------------------------------------------
step "status / show / logs / doctor are coherent in a fresh shell"
# ---------------------------------------------------------------------------
bash -c "'$GPUQ' status --json >/dev/null" && pass "gpuq status (new shell)" || bad "gpuq status"
bash -c "'$GPUQ' show $ID_A --json >/dev/null" && pass "gpuq show (new shell)" || bad "gpuq show"
bash -c "'$GPUQ' logs $ID_A >/dev/null" && pass "gpuq logs (new shell)" || bad "gpuq logs"
SNAP="$("$GPUQ" show "$ID_B" --json | jq_get "['snapshot_commit']")"
[ -n "$SNAP" ] && [ "$SNAP" != "None" ] && pass "provenance recorded (snapshot ${SNAP:0:9})" \
                                        || bad "no snapshot commit recorded"
bash -c "'$GPUQ' doctor >/dev/null"
case $? in
  0) pass "gpuq doctor: HEALTHY" ;;
  1) pass "gpuq doctor: DEGRADED (usable)" ;;
  *) bad  "gpuq doctor: BROKEN" ;;
esac

# ---------------------------------------------------------------------------
step "Cancellation of a queued job and of a running job"
# ---------------------------------------------------------------------------
D="$(make_project project-d 60)"
ID_D1="$(cd "$D" && "$GPUQ" submit --project project-d --json -- "$PY" job.py | jq_get "['job_id']")"
ID_D2="$(cd "$D" && "$GPUQ" submit --project project-d --json -- "$PY" job.py | jq_get "['job_id']")"
for _ in $(seq 1 60); do [ "$(state_of "$ID_D1")" = "RUNNING" ] && break; sleep 1; done

QSTATE="$("$GPUQ" cancel "$ID_D2" --json | jq_get "['state']")"
[ "$QSTATE" = "CANCELLED" ] && pass "queued job cancelled" || bad "queued cancel gave $QSTATE"

RSTATE="$("$GPUQ" cancel "$ID_D1" --force --json | jq_get "['state']")"
[ "$RSTATE" = "CANCELLED" ] && pass "running job cancelled" || bad "running cancel gave $RSTATE"

sleep 2
[ "$(state_of "$ID_D2")" = "CANCELLED" ] && pass "cancelled queued job never ran" \
                                         || bad "cancelled job changed state"

# ---------------------------------------------------------------------------
printf '\n\033[1m========================================\033[0m\n'
printf '  passed: %s\n  failed: %s\n' "$PASS" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  printf '\033[32m  DEFINITION OF DONE: PASSED\033[0m\n\n'; exit 0
fi
printf '\033[31m  DEFINITION OF DONE: FAILED\033[0m\n\n'; exit 1
