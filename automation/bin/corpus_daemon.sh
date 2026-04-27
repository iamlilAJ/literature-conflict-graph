#!/usr/bin/env bash
set -u -o pipefail

REPO_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
LOG_DIR="${AIGRAPH_CORPUS_LOG_DIR:-$REPO_DIR/automation/logs}"
CORPUS_ROOT="${AIGRAPH_CORPUS_ROOT:-$REPO_DIR/data/corpus/arxiv_reasoning}"
FROM_YEAR="${AIGRAPH_CORPUS_FROM_YEAR:-2022}"
TO_YEAR="${AIGRAPH_CORPUS_TO_YEAR:-2026}"
SEED_LIMIT="${AIGRAPH_CORPUS_SEED_LIMIT:-50}"
BATCH_SIZE="${AIGRAPH_CORPUS_BATCH_SIZE:-5}"
SLEEP_SECONDS="${AIGRAPH_CORPUS_SLEEP_SECONDS:-300}"
ACTIVE_SLEEP_SECONDS="${AIGRAPH_CORPUS_ACTIVE_SLEEP_SECONDS:-$SLEEP_SECONDS}"
IDLE_SLEEP_SECONDS="${AIGRAPH_CORPUS_IDLE_SLEEP_SECONDS:-$SLEEP_SECONDS}"
SEED_EVERY="${AIGRAPH_CORPUS_SEED_EVERY:-24}"
VALIDATE_EVERY="${AIGRAPH_CORPUS_VALIDATE_EVERY:-12}"
RUN_ONCE="${AIGRAPH_CORPUS_ONCE:-0}"
RESEED_ON_EMPTY="${AIGRAPH_CORPUS_RESEED_ON_EMPTY:-1}"
LOCK_DIR="$LOG_DIR/corpus_daemon.lock"
PID_FILE="$LOCK_DIR/pid"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -x "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No usable Python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" >"$PID_FILE"
    return 0
  fi

  if [ -f "$PID_FILE" ]; then
    local existing_pid
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$existing_pid" ] && ! kill -0 "$existing_pid" 2>/dev/null; then
      rm -rf "$LOCK_DIR"
      mkdir "$LOCK_DIR"
      printf '%s\n' "$$" >"$PID_FILE"
      return 0
    fi
  fi

  log "Another corpus daemon appears to be running; exiting."
  return 1
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

run_step() {
  local label="$1"
  shift
  log "Starting: $label"
  if "$@"; then
    log "Finished: $label"
    return 0
  fi
  log "Failed: $label"
  return 1
}

unfinished_manifest_count() {
  if [ ! -f "$CORPUS_ROOT/papers.jsonl" ]; then
    printf '0\n'
    return 0
  fi
  "$PYTHON_BIN" - "$CORPUS_ROOT/papers.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = 0
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        payload = json.loads(line)
    except Exception:
        continue
    if payload.get("sync_status") != "complete":
        count += 1
print(count)
PY
}

if ! acquire_lock; then
  exit 0
fi
trap cleanup EXIT

iteration=0
while true; do
  iteration=$((iteration + 1))
  remaining_after_sync=""
  log "Corpus loop iteration $iteration"

  if [ "$iteration" -eq 1 ] || [ $((iteration % SEED_EVERY)) -eq 0 ]; then
    run_step \
      "seed reasoning corpus" \
      "$PYTHON_BIN" -m aigraph.cli corpus-seed-reasoning \
        --root "$CORPUS_ROOT" \
        --from-year "$FROM_YEAR" \
        --to-year "$TO_YEAR" \
        --per-query-limit "$SEED_LIMIT" || true
  fi

  run_step \
    "sync corpus batch" \
    "$PYTHON_BIN" -m aigraph.cli corpus-sync-arxiv \
      --root "$CORPUS_ROOT" \
      --batch-size "$BATCH_SIZE" || true

  remaining_after_sync="$(unfinished_manifest_count 2>/dev/null || printf '1\n')"
  if [ "$RESEED_ON_EMPTY" = "1" ] && [ "$RUN_ONCE" != "1" ]; then
    if [ "${remaining_after_sync:-1}" = "0" ]; then
      log "Manifest fully synced; triggering immediate reseed."
      run_step \
        "seed reasoning corpus (queue refill)" \
        "$PYTHON_BIN" -m aigraph.cli corpus-seed-reasoning \
          --root "$CORPUS_ROOT" \
          --from-year "$FROM_YEAR" \
          --to-year "$TO_YEAR" \
          --per-query-limit "$SEED_LIMIT" || true

      run_step \
        "sync corpus batch (post-refill)" \
        "$PYTHON_BIN" -m aigraph.cli corpus-sync-arxiv \
          --root "$CORPUS_ROOT" \
          --batch-size "$BATCH_SIZE" || true

      remaining_after_sync="$(unfinished_manifest_count 2>/dev/null || printf '1\n')"
    fi
  fi

  if [ "$iteration" -eq 1 ] || [ $((iteration % VALIDATE_EVERY)) -eq 0 ]; then
    run_step \
      "validate corpus" \
      "$PYTHON_BIN" -m aigraph.cli corpus-validate \
        --root "$CORPUS_ROOT" || true
  fi

  if [ "$RUN_ONCE" = "1" ]; then
    log "AIGRAPH_CORPUS_ONCE=1, exiting after one loop."
    break
  fi

  if [ "${remaining_after_sync:-1}" = "0" ]; then
    sleep_seconds="$IDLE_SLEEP_SECONDS"
  else
    sleep_seconds="$ACTIVE_SLEEP_SECONDS"
  fi
  log "Sleeping for ${sleep_seconds}s (remaining=${remaining_after_sync:-unknown})"
  sleep "$sleep_seconds"
done
