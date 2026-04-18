#!/usr/bin/env zsh
set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
AUTOMATION_DIR="${2:-$REPO_DIR/automation}"
RUNS_DIR="${3:-$REPO_DIR/outputs/runs}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

cd "$REPO_DIR"

"$PYTHON_BIN" -m aigraph.cli automation-fix-run \
  --automation-dir "$AUTOMATION_DIR" \
  --runs-dir "$RUNS_DIR" \
  --repo-dir "$REPO_DIR" \
  --max-issues 3 \
  --push \
  --open-pr

