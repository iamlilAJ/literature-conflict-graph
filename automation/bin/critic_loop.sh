#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
AUTOMATION_DIR="${2:-$REPO_DIR/automation}"
RUNS_DIR="${3:-$REPO_DIR/outputs/runs}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

cd "$REPO_DIR"

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
  set +a
fi

"$PYTHON_BIN" -m aigraph.cli automation-critic \
  --automation-dir "$AUTOMATION_DIR" \
  --runs-dir "$RUNS_DIR" \
  --limit 8

"$PYTHON_BIN" -m aigraph.cli automation-fix-bundle \
  --automation-dir "$AUTOMATION_DIR" \
  --runs-dir "$RUNS_DIR" \
  --max-issues 3
