"""Query-layer hypothesis-quality cleanup (non-frozen seam).

Covers _is_degenerate_anomaly + _select's self-conflict drop and per-anomaly
cap. Uses a tmp run dir so no real corpus is needed. 0 LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import aigraph_query as q  # noqa: E402
from aigraph.models import Anomaly  # noqa: E402


# ---- _is_degenerate_anomaly ---------------------------------------------

def _anom(type_, **shared):
    return Anomaly(anomaly_id="a1", type=type_, central_question="q",
                   shared_entities={k: v for k, v in shared.items() if v is not None})


def test_degenerate_method_equals_task():
    assert q._is_degenerate_anomaly(_anom("impact_conflict", method="planning", task="planning"))


def test_clean_method_task_not_degenerate():
    assert not q._is_degenerate_anomaly(_anom("impact_conflict", method="rag", task="qa"))


def test_bridge_from_to_collapse_is_degenerate():
    assert q._is_degenerate_anomaly(
        _anom("bridge_opportunity", method_from="rag", method_to="rag",
              task_from="qa", task_to="qa"))


def test_bridge_distinct_tasks_not_degenerate():
    assert not q._is_degenerate_anomaly(
        _anom("bridge_opportunity", method_from="rag", method_to="rag",
              task_from="qa", task_to="reasoning"))


def test_none_anomaly_not_degenerate():
    assert not q._is_degenerate_anomaly(None)


def test_normalization_handles_punctuation_case():
    # "Self-Planning" vs "self planning" should still match as the same entity
    assert q._is_degenerate_anomaly(_anom("evidence_gap", method="Self-Planning", task="self planning"))


# ---- _select end-to-end on a tmp run ------------------------------------

@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()

    def w(name, rows):
        (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    w("papers.jsonl", [
        {"paper_id": "p1", "title": "RAG paper", "year": 2024, "venue": "X"},
        {"paper_id": "p2", "title": "Planning paper", "year": 2024, "venue": "Y"},
    ])
    w("claims.jsonl", [
        {"claim_id": "c1", "paper_id": "p1", "claim_text": "rag helps qa",
         "method": "rag", "task": "qa", "direction": "positive",
         "canonical_method": "rag", "canonical_task": "qa"},
        {"claim_id": "c2", "paper_id": "p1", "claim_text": "rag hurts qa",
         "method": "rag", "task": "qa", "direction": "negative",
         "canonical_method": "rag", "canonical_task": "qa"},
        {"claim_id": "c3", "paper_id": "p2", "claim_text": "planning planning result",
         "method": "planning", "task": "planning", "direction": "positive",
         "canonical_method": "planning", "canonical_task": "planning"},
    ])
    w("anomalies.jsonl", [
        {"anomaly_id": "aClean", "type": "impact_conflict",
         "central_question": "Why do papers disagree about rag on qa?",
         "claim_ids": ["c1", "c2"], "shared_entities": {"method": "rag", "task": "qa"}},
        {"anomaly_id": "aSelf", "type": "impact_conflict",
         "central_question": "When does planning help on planning?",
         "claim_ids": ["c3"], "shared_entities": {"method": "planning", "task": "planning"}},
    ])
    # clean anomaly: 3 near-duplicate frames (EXACTLY-3); self anomaly: 1
    w("hypotheses_scored.jsonl", [
        {"hypothesis_id": "h1", "anomaly_id": "aClean", "hypothesis": "rag qa moderator one",
         "mechanism": "m", "explains_claims": ["c1", "c2"], "predictions": ["a", "b"],
         "minimal_test": "t", "scope_conditions": {"method": "rag"}},
        {"hypothesis_id": "h2", "anomaly_id": "aClean", "hypothesis": "rag qa moderator two",
         "mechanism": "m", "explains_claims": ["c1", "c2"], "predictions": ["a", "b"],
         "minimal_test": "t", "scope_conditions": {"method": "rag"}},
        {"hypothesis_id": "h3", "anomaly_id": "aClean", "hypothesis": "rag qa moderator three",
         "mechanism": "m", "explains_claims": ["c1", "c2"], "predictions": ["a", "b"],
         "minimal_test": "t", "scope_conditions": {"method": "rag"}},
        {"hypothesis_id": "h4", "anomaly_id": "aSelf", "hypothesis": "planning planning idea",
         "mechanism": "m", "explains_claims": ["c3"], "predictions": ["a", "b"],
         "minimal_test": "t", "scope_conditions": {"method": "planning"}},
    ])
    return d


def _sel_ids(run_dir, **kw):
    selected, *_ = q._select(run_dir, "rag qa planning", k=10, max_hypotheses=30,
                             mmr_lambda=0.7, min_anomalies=1, **kw)
    return [h.hypothesis_id for h in (selected or [])]


def test_select_drops_self_conflict(run_dir):
    ids = _sel_ids(run_dir, drop_self_conflict=True, max_per_anomaly=99)
    assert "h4" not in ids          # the planning/planning self-conflict is dropped
    assert any(i in ids for i in ("h1", "h2", "h3"))  # clean ones survive


def test_select_keeps_self_conflict_when_disabled(run_dir):
    ids = _sel_ids(run_dir, drop_self_conflict=False, max_per_anomaly=99)
    assert "h4" in ids


def test_per_anomaly_cap(run_dir):
    # cap=2 → at most 2 of h1/h2/h3 (same anomaly) survive into candidates
    ids = _sel_ids(run_dir, drop_self_conflict=False, max_per_anomaly=2)
    clean = [i for i in ids if i in ("h1", "h2", "h3")]
    assert len(clean) <= 2


def test_no_cap_keeps_all_clean(run_dir):
    ids = _sel_ids(run_dir, drop_self_conflict=False, max_per_anomaly=0)
    clean = [i for i in ids if i in ("h1", "h2", "h3")]
    assert len(clean) == 3
