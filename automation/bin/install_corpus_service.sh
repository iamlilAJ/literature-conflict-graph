#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVICE_NAME="${SERVICE_NAME:-aigraph-corpus}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_PATH="${REPO_DIR}/automation/logs/corpus_daemon.out"

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

start_nohup_service() {
  mkdir -p "${REPO_DIR}/automation/logs"
  if pgrep -f "corpus_daemon.sh ${REPO_DIR}" >/dev/null 2>&1; then
    pkill -f "corpus_daemon.sh ${REPO_DIR}" || true
    sleep 1
  fi
  nohup /bin/bash "${REPO_DIR}/automation/bin/corpus_daemon.sh" "${REPO_DIR}" \
    >"${LOG_PATH}" 2>&1 < /dev/null &
  local pid="$!"
  sleep 2
  if ps -p "${pid}" >/dev/null 2>&1; then
    echo "Started ${SERVICE_NAME} with nohup (pid ${pid}); logs: ${LOG_PATH}"
    return 0
  fi
  echo "Failed to start ${SERVICE_NAME} with nohup" >&2
  return 1
}

cat >/tmp/${SERVICE_NAME}.service <<EOF
[Unit]
Description=aigraph 24/7 corpus ingest daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
ExecStart=/bin/bash ${REPO_DIR}/automation/bin/corpus_daemon.sh ${REPO_DIR}
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  ${SUDO} install -m 644 /tmp/${SERVICE_NAME}.service "${SERVICE_PATH}"
  ${SUDO} systemctl daemon-reload
  ${SUDO} systemctl enable --now "${SERVICE_NAME}"
  ${SUDO} systemctl status --no-pager "${SERVICE_NAME}" || true
  echo "Installed and started ${SERVICE_NAME} via ${SERVICE_PATH}"
else
  start_nohup_service
fi
