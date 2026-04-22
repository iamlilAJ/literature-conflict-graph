#!/usr/bin/env bash
set -u -o pipefail

REPO_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG_DIR="${AIGRAPH_CORPUS_LOG_DIR:-$REPO_DIR/automation/logs}"
LOG_FILE="${LOG_DIR}/corpus_supervisor.out"
RESTART_DELAY="${AIGRAPH_CORPUS_RESTART_DELAY:-10}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_FILE"
}

while true; do
  log "Starting corpus daemon"
  /bin/bash "$REPO_DIR/automation/bin/corpus_daemon.sh" "$REPO_DIR" >>"$LOG_FILE" 2>&1
  exit_code=$?
  log "Corpus daemon exited with code ${exit_code}; restarting in ${RESTART_DELAY}s"
  sleep "$RESTART_DELAY"
done
