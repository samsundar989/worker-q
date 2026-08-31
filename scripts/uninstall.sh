#!/usr/bin/env bash
#
# GPUQ uninstall.
#
# Distinguishes four separable things:
#   1. the gpuq package
#   2. gpuq state (database, logs, snapshots)   - preserved unless --purge
#   3. the vendored backend, if one was built
#   4. the Claude policy block
#
# Source repositories are never touched.

set -uo pipefail

PURGE=0
ASSUME_YES=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: uninstall.sh [--purge] [--yes] [--dry-run]

  --purge    also delete the gpuq state directory (database, logs, snapshots)
  --yes      do not prompt for confirmation
  --dry-run  show what would happen and change nothing
EOF
}

for arg in "$@"; do
  case "$arg" in
    --purge) PURGE=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
act()  { printf '  \033[32mdone\033[0m  %s\n' "$*"; }
skip() { printf '  \033[33mskip\033[0m  %s\n' "$*"; }
plan() { printf '  would  %s\n' "$*"; }

GPUQ=""
command -v gpuq >/dev/null 2>&1 && GPUQ="$(command -v gpuq)"
for candidate in "$HOME/.local/bin/gpuq" "$HOME/.local/bin/gpuq.exe"; do
  [ -z "$GPUQ" ] && [ -x "$candidate" ] && GPUQ="$candidate"
done

STATE_DIR="${GPUQ_STATE_DIR:-$HOME/.local/state/gpuq}"
CONFIG_FILE="${GPUQ_CONFIG_FILE:-$HOME/.config/gpuq/config.toml}"
VENDOR_DIR="$HOME/.local/share/gpuq"

say "What will be removed"
printf '  package      : gpuq\n'
printf '  state        : %s %s\n' "$STATE_DIR" \
  "$([ "$PURGE" -eq 1 ] && echo '(WILL BE DELETED)' || echo '(preserved - use --purge)')"
printf '  config       : %s\n' "$CONFIG_FILE"
printf '  vendor       : %s\n' "$VENDOR_DIR"
printf '  claude policy: the gpuq block in ~/.claude/CLAUDE.md only\n'
printf '\n  never touched: your source repositories\n'

if [ "$DRY_RUN" -eq 1 ]; then
  say "Dry run - nothing will change"
  plan "stop the gpuq dispatcher"
  plan "remove the Claude policy block"
  plan "uninstall the gpuq package"
  [ "$PURGE" -eq 1 ] && plan "delete $STATE_DIR" || plan "keep $STATE_DIR"
  exit 0
fi

if [ "$ASSUME_YES" -ne 1 ]; then
  printf '\nProceed? [y/N] '
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "aborted"; exit 1 ;;
  esac
fi

say "1. Stop the dispatcher"
if [ -n "$GPUQ" ]; then
  "$GPUQ" _stop-daemon >/dev/null 2>&1 && act "dispatcher stopped" || skip "dispatcher was not running"
else
  skip "gpuq not on PATH"
fi

say "2. Remove the Claude policy block"
if [ -n "$GPUQ" ]; then
  "$GPUQ" claude-policy remove >/dev/null 2>&1 && act "policy block removed (other instructions kept)" \
                                               || skip "no policy block found"
else
  skip "gpuq not on PATH - remove the gpuq-policy block from ~/.claude/CLAUDE.md by hand"
fi

say "3. Uninstall the package"
REMOVED=0
if command -v uv >/dev/null 2>&1 && uv tool uninstall gpuq >/dev/null 2>&1; then
  act "uv tool uninstall gpuq"; REMOVED=1
fi
if [ "$REMOVED" -eq 0 ] && command -v pipx >/dev/null 2>&1 && pipx uninstall gpuq >/dev/null 2>&1; then
  act "pipx uninstall gpuq"; REMOVED=1
fi
if [ "$REMOVED" -eq 0 ] && [ -d "$VENDOR_DIR/venv" ]; then
  rm -rf "$VENDOR_DIR/venv" && act "removed $VENDOR_DIR/venv"; REMOVED=1
fi
[ "$REMOVED" -eq 0 ] && skip "could not determine how gpuq was installed - remove it manually"

say "4. State"
if [ "$PURGE" -eq 1 ]; then
  case "$STATE_DIR" in
    "$HOME"/*)
      rm -rf "$STATE_DIR" && act "deleted $STATE_DIR" ;;
    *)
      skip "refusing to delete an unexpected state path: $STATE_DIR" ;;
  esac
  [ -f "$CONFIG_FILE" ] && rm -f "$CONFIG_FILE" && act "deleted $CONFIG_FILE"
else
  skip "kept $STATE_DIR (job history, logs and snapshots)"
  printf '        delete it later with: rm -rf %s\n' "$STATE_DIR"
fi

say "Done"
exit 0
