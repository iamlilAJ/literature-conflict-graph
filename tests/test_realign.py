"""Tests for src/aigraph/realign.py — lazy on-access run realignment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aigraph.models import Claim, Paper
from aigraph.realign import (
    SCHEMA_VERSION,
    is_run_realignable,
    realign_run,
    realign_run_safe,
)


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def _stub_run(tmp_path: Path, run_id: str, *, schema_version: int | None = None, status: str = "done") -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    claims = [
        Claim(claim_id="c001", paper_id="p001", claim_text="x", method="RAG", task="QA", direction="positive"),
        Claim(claim_id="c002", paper_id="p002", claim_text="y", method="RAG", task="QA", direction="negative"),
    ]
    papers = [
        Paper(paper_id="p001", title="A", year=2024, venue="ACL"),
        Paper(paper_id="p002", title="B", year=2024, venue="ACL"),
    ]
    _write_jsonl(run_dir / "claims.jsonl", claims)
    _write_jsonl(run_dir / "papers.jsonl", papers)
    status_payload = {"run_id": run_id, "status": status, "stage": "complete"}
    if schema_version is not None:
        status_payload["schema_version"] = schema_version
    (run_dir / "status.json").write_text(json.dumps(status_payload), encoding="utf-8")
    return run_dir


def test_run_with_no_schema_version_is_realignable(tmp_path):
    run_dir = _stub_run(tmp_path, "run-old")
    needs, reason = is_run_realignable(run_dir)
    assert needs
    assert "< current" in reason


def test_run_already_at_current_version_is_skipped(tmp_path):
    run_dir = _stub_run(tmp_path, "run-current", schema_version=SCHEMA_VERSION)
    needs, reason = is_run_realignable(run_dir)
    assert not needs
    assert "already at" in reason


def test_run_at_future_version_is_skipped(tmp_path):
    # Defensive: never downgrade.
    run_dir = _stub_run(tmp_path, "run-future", schema_version=SCHEMA_VERSION + 5)
    needs, _ = is_run_realignable(run_dir)
    assert not needs


def test_in_progress_run_is_not_realigned(tmp_path):
    run_dir = _stub_run(tmp_path, "run-running", status="running")
    needs, reason = is_run_realignable(run_dir)
    assert not needs
    assert "running" in reason


def test_run_without_claims_is_not_realigned(tmp_path):
    run_dir = _stub_run(tmp_path, "run-empty")
    (run_dir / "claims.jsonl").write_text("", encoding="utf-8")
    needs, reason = is_run_realignable(run_dir)
    assert not needs
    assert "no claims" in reason


def test_realign_run_writes_artifacts_and_stamps_version(tmp_path):
    run_dir = _stub_run(tmp_path, "run-stale")
    assert not (run_dir / "graph.json").exists()
    assert not (run_dir / "anomalies.jsonl").exists()
    assert not (run_dir / "index.html").exists()
    assert realign_run(run_dir) is True
    assert (run_dir / "graph.json").exists()
    assert (run_dir / "anomalies.jsonl").exists()
    assert (run_dir / "index.html").exists()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["schema_version"] == SCHEMA_VERSION
    assert "realigned_at" in status


def test_realign_run_is_idempotent_after_first_pass(tmp_path):
    run_dir = _stub_run(tmp_path, "run-twice")
    assert realign_run(run_dir) is True
    # Second call: schema_version now matches, no rebuild.
    assert realign_run(run_dir) is False


def test_realign_run_safe_swallows_corrupt_inputs(tmp_path):
    run_dir = _stub_run(tmp_path, "run-corrupt")
    (run_dir / "claims.jsonl").write_text("{not json}\n", encoding="utf-8")
    # _read_jsonl skips bad lines so build_graph just gets an empty list and
    # the rebuild succeeds with zero claims; safe wrapper never raises.
    result = realign_run_safe(run_dir)
    # Either rebuilt (True) or skipped due to empty claims input — never raises.
    assert isinstance(result, bool)
