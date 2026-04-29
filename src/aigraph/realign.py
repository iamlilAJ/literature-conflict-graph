"""Lazy on-access realignment of persisted run artifacts.

Each run on disk carries a ``schema_version`` stamp on its ``status.json``.
When the running code's ``SCHEMA_VERSION`` is higher than what was persisted,
the run's deterministic outputs are rebuilt in-process the next time the
server touches the run.

**Scope — what gets rebuilt and what does NOT.**

Lazy realign only re-derives the three artifacts that are pure functions of
``claims.jsonl`` + ``papers.jsonl``:

* ``graph.json``      (from ``build_graph``)
* ``anomalies.jsonl`` (from ``detect_anomalies``)
* ``index.html``      (from ``render_visualization``)

It does **not** touch the LLM-derived or scoring-derived artifacts:

* ``hypotheses.jsonl``  (template + LLM)
* ``creator_hypotheses.jsonl``  (LLM)
* ``open_questions.jsonl``  (LLM)
* ``insights.jsonl``  (community + LLM)
* ``selected_hypotheses.md``  (scoring weights + MMR)
* ``overview.json``  (a summary of the above)

If you bump ``SCHEMA_VERSION`` because of a change to ``scoring.py`` weights,
``hypotheses.py`` templates, the MMR selector, or the report renderer, lazy
realign will leave the listed LLM/scoring artifacts stale. Those need an
explicit batch re-run (typically via ``aigraph generate-hypotheses`` /
``aigraph select`` or a fresh ``run_pipeline``). Only bump
``SCHEMA_VERSION`` for changes whose effect is captured by build_graph /
detect_anomalies / visualize alone.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .anomalies import detect_anomalies
from .graph import build_graph, load_graph, save_graph
from .models import Claim, Paper
from .visualize import render_visualization


# Bump when build_graph / detect_anomalies / visualize output changes such that
# legacy persisted artifacts will look wrong rendered with new code.
#  v1 — legacy 17-node-type schema
#  v2 — graph simplification (6 node types) + bridge_opportunity strict filter +
#       replication_conflict + canonicalization + co_cites + weighted contradicts
SCHEMA_VERSION = 2


# Cap the per-run lock cache so a long-running server does not accumulate one
# Lock object per ever-touched run_dir. Real concurrent rebuilds are rare (a
# run rebuilds at most once then is permanently fresh), so eviction is safe:
# the worst case is two threads both grab a freshly-allocated lock and the
# second one's double-check inside realign_run sees the rebuilt
# schema_version and exits without redoing work.
_RUN_LOCK_CACHE_MAX = 128
_run_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()
_master_lock = threading.Lock()


def _lock_for(run_dir: Path) -> threading.Lock:
    key = str(run_dir.resolve())
    with _master_lock:
        lock = _run_locks.get(key)
        if lock is not None:
            _run_locks.move_to_end(key)
            return lock
        lock = threading.Lock()
        _run_locks[key] = lock
        if len(_run_locks) > _RUN_LOCK_CACHE_MAX:
            _run_locks.popitem(last=False)
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
        _preserve_cites_stance(run_dir / "graph.json", g)
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


def _preserve_cites_stance(old_graph_path: Path, new_graph: "object") -> None:
    """Copy stance / stance_confidence / stance_rationale from the prior on-disk
    graph onto the freshly-rebuilt one. Without this, a SCHEMA_VERSION bump
    would silently erase any stance work done by `classify_cites_edges` —
    `build_graph` is called here with `classify_stance=False`, so stance is
    not regenerated.

    Wrapped so a corrupt prior graph never blocks the rebuild from
    completing — stance is best-effort persistence, not a correctness gate.
    """
    if not old_graph_path.exists():
        return
    try:
        old_graph = load_graph(old_graph_path)
    except Exception as exc:  # pragma: no cover - defensive on corrupt JSON
        logging.getLogger("aigraph.realign").warning(
            "stance preservation skipped (could not load prior graph at %s): %s",
            old_graph_path,
            exc,
        )
        return
    for u, v, _key, data in old_graph.edges(keys=True, data=True):
        if data.get("edge_type") != "cites":
            continue
        stance = data.get("stance")
        if stance is None:
            continue
        new_edge_data = new_graph.get_edge_data(u, v) or {}
        for new_key, nd in new_edge_data.items():
            if nd.get("edge_type") == "cites":
                nd["stance"] = stance
                if data.get("stance_confidence") is not None:
                    nd["stance_confidence"] = data["stance_confidence"]
                if data.get("stance_rationale") is not None:
                    nd["stance_rationale"] = data["stance_rationale"]
                break


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
