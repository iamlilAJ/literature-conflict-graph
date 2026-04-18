#!/usr/bin/env zsh
set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
GENERATED_CRONTAB="$REPO_DIR/automation/cron/generated.crontab"

cd "$REPO_DIR"

"$PYTHON_BIN" -m aigraph.cli automation-crontab \
  --repo-dir "$REPO_DIR" \
  --python-bin "$PYTHON_BIN" \
  --output "$GENERATED_CRONTAB"

crontab "$GENERATED_CRONTAB"

echo "Installed crontab from $GENERATED_CRONTAB"
crontab -l
