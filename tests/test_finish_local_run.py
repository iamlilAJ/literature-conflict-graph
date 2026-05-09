"""Unit tests for the continuation runner's anomaly cap.

The full e2e flow is exercised by the bundled
``artifacts/runs/arxiv-reasoning-v0.7-100p/`` integration run; this
test pins the standalone cap helper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from finish_local_run import cap_anomalies_top_n  # noqa: E402


def _write_anomalies(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "anomalies.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_cap_takes_top_n_by_topology_score(tmp_path):
    """Cap returns top-N rows sorted by topology_score descending,
    and counter reflects the FULL set, not the cap."""
    anomalies = [
        {"anomaly_id": "a1", "type": "impact_conflict", "topology_score": 0.10},
        {"anomaly_id": "a2", "type": "impact_conflict", "topology_score": 0.90},
        {"anomaly_id": "a3", "type": "evidence_gap", "topology_score": 0.50},
        {"anomaly_id": "a4", "type": "evidence_gap", "topology_score": 0.30},
        {"anomaly_id": "a5", "type": "metric_mismatch", "topology_score": 0.70},
    ]
    src = _write_anomalies(tmp_path, anomalies)
    out, full_counts = cap_anomalies_top_n(src, n=3)

    written = [json.loads(line) for line in out.open()]
    assert [r["anomaly_id"] for r in written] == ["a2", "a5", "a3"]
    # counter reflects all 5, not the top 3
    assert full_counts == {"impact_conflict": 2, "evidence_gap": 2, "metric_mismatch": 1}


def test_cap_handles_n_larger_than_input(tmp_path):
    """Asking for more than exists returns the full set."""
    anomalies = [{"anomaly_id": f"a{i}", "type": "x", "topology_score": float(i)} for i in range(3)]
    src = _write_anomalies(tmp_path, anomalies)
    out, full_counts = cap_anomalies_top_n(src, n=100)

    written = [json.loads(line) for line in out.open()]
    assert len(written) == 3
    assert full_counts == {"x": 3}


def test_cap_handles_missing_topology_score(tmp_path):
    """Rows without topology_score are sorted as 0 (still kept)."""
    anomalies = [
        {"anomaly_id": "a1", "type": "x", "topology_score": 0.5},
        {"anomaly_id": "a2", "type": "x"},  # no topology_score
        {"anomaly_id": "a3", "type": "x", "topology_score": 0.9},
    ]
    src = _write_anomalies(tmp_path, anomalies)
    out, _ = cap_anomalies_top_n(src, n=2)

    written = [json.loads(line) for line in out.open()]
    # a3 (0.9) and a1 (0.5) win; a2 (default 0.0) drops
    assert sorted(r["anomaly_id"] for r in written) == ["a1", "a3"]


def test_cap_writes_to_anomalies_top_jsonl_in_same_dir(tmp_path):
    """Output path is anomalies_top.jsonl alongside the input."""
    src = _write_anomalies(tmp_path, [{"anomaly_id": "x", "type": "y", "topology_score": 0.0}])
    out, _ = cap_anomalies_top_n(src, n=10)
    assert out.parent == src.parent
    assert out.name == "anomalies_top.jsonl"


def test_cap_handles_unknown_type(tmp_path):
    """Anomaly missing 'type' field gets bucketed under 'unknown'."""
    anomalies = [
        {"anomaly_id": "a1", "topology_score": 0.5},  # no type
        {"anomaly_id": "a2", "type": "evidence_gap", "topology_score": 0.3},
    ]
    src = _write_anomalies(tmp_path, anomalies)
    _, counts = cap_anomalies_top_n(src, n=10)
    assert counts == {"unknown": 1, "evidence_gap": 1}
