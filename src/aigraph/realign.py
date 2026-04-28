"""Lazy on-access realignment of persisted run artifacts.

Each run on disk carries a `schema_version` stamp on its `status.json`. When the
running code's `SCHEMA_VERSION` is higher than what was persisted, the run's
deterministic outputs (graph.json, anomalies.jsonl, index.html) are rebuilt
in-process the next time the server touches the run.

Bumping `SCHEMA_VERSION` is the contract that says "a deploy of this commit
should re-derive these artifacts on every persisted run before serving them".
Bump it whenever the output shape of `build_graph`, `detect_anomalies`, or the
visualize HTML changes in a way the legacy on-disk version would not handle.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .anomalies import detect_anomalies
from .graph import build_graph, save_graph
from .models import Claim, Paper
from .visualize import render_visualization


# Bump when build_graph / detect_anomalies / visualize output changes such that
# legacy persisted artifacts will look wrong rendered with new code.
#  v1 — legacy 17-node-type schema
#  v2 — graph simplification (6 node types) + bridge_opportunity strict filter +
#       replication_conflict + canonicalization + co_cites + weighted contradicts
SCHEMA_VERSION = 2


_run_locks: dict[str, threading.Lock] = {}
_master_lock = threading.Lock()


def _lock_for(run_dir: Path) -> threading.Lock:
    key = str(run_dir.resolve())
    with _master_lock:
        lock = _run_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _run_locks[key] = lock
        return lock


def _read_jsonl(path: Path, model: Any) -> list[Any]:
    if not path.exists():
        return []
    items: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(model(**json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return items


def is_run_realignable(run_dir: Path) -> tuple[bool, str]:
    """Return (needs_rebuild, reason). The reason field is for logs/tests."""
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return False, "no status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "malformed status.json"
    if status.get("status") != "done":
        return False, f"status={status.get('status')} (not done)"
    if not (run_dir / "claims.jsonl").exists() or (run_dir / "claims.jsonl").stat().st_size == 0:
        return False, "no claims.jsonl"
    if not (run_dir / "papers.jsonl").exists():
        return False, "no papers.jsonl"
    persisted = int(status.get("schema_version") or 0)
    if persisted >= SCHEMA_VERSION:
        return False, f"already at schema_version={persisted}"
    return True, f"persisted schema_version={persisted} < current {SCHEMA_VERSION}"


def realign_run(run_dir: Path) -> bool:
    """Rebuild graph.json + anomalies.jsonl + index.html from claims + papers.

    Returns True if a rebuild ran, False if the run was already current or
    missing the inputs to rebuild. Idempotent + thread-safe per run_dir.
    Failures inside the rebuild propagate; callers wrap in try/except so a
    single bad run does not break unrelated requests.
    """
    needs, _ = is_run_realignable(run_dir)
    if not needs:
        return False
    lock = _lock_for(run_dir)
    with lock:
        # Re-check inside the lock — another thread may have just rebuilt.
        needs, _ = is_run_realignable(run_dir)
        if not needs:
            return False
        claims = _read_jsonl(run_dir / "claims.jsonl", Claim)
        papers = _read_jsonl(run_dir / "papers.jsonl", Paper)
        g = build_graph(claims, papers=papers)
        save_graph(g, run_dir / "graph.json")
        anomalies = detect_anomalies(g, claims)
        with (run_dir / "anomalies.jsonl").open("w", encoding="utf-8") as f:
            for anomaly in anomalies:
                f.write(anomaly.model_dump_json() + "\n")
        render_visualization(run_dir, run_dir / "index.html")
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["schema_version"] = SCHEMA_VERSION
        status["realigned_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return True


def realign_run_safe(run_dir: Path) -> bool:
    """Same as realign_run but swallows + logs exceptions. Use from request paths
    where a single broken run should not 500 the request."""
    try:
        return realign_run(run_dir)
    except Exception as exc:
        logging.getLogger("aigraph.realign").warning(
            "realign skipped for %s: %s: %s", run_dir, type(exc).__name__, exc
        )
        return False
